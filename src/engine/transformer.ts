import { add, gelu, layerNorm, matVec, relu, rmsNorm, silu, softmax, zeros } from './math'
import { gaussian, hashString, mulberry32, randomMatrix, randomVec } from './rng'
import { applySampling } from './sampling'
import { tokenize } from './tokenizer'
import { demoVocab, toyLogits } from './toyLm'
import type { HeadTrace, LayerTrace, ModelConfig, Token, Trace } from './types'

const EMB_CACHE_MAX = 256
const embCache = new Map<string, number[]>()

function embedToken(token: Token, d: number): number[] {
  const key = `${d}:${token.id}:${token.text}`
  const hit = embCache.get(key)
  if (hit) return hit
  const rand = mulberry32(hashString(`emb:${token.text}:${token.id}`))
  const vec = new Array<number>(d)
  for (let i = 0; i < d; i++) vec[i] = gaussian(rand) * 0.55
  if (embCache.size >= EMB_CACHE_MAX) {
    const oldest = embCache.keys().next().value
    if (oldest !== undefined) embCache.delete(oldest)
  }
  embCache.set(key, vec)
  return vec
}

const sinCache = new Map<string, number[]>()

function sinusoidal(pos: number, d: number): number[] {
  const key = `${pos}:${d}`
  const hit = sinCache.get(key)
  if (hit) return hit
  const out = new Array<number>(d)
  for (let i = 0; i < d; i++) {
    const div = pos / 10000 ** ((2 * Math.floor(i / 2)) / d)
    out[i] = i % 2 === 0 ? Math.sin(div) : Math.cos(div)
  }
  sinCache.set(key, out)
  return out
}

function learnedPos(pos: number, d: number, seed: number): number[] {
  const rand = mulberry32(hashString(`pos:${seed}:${pos}`))
  return randomVec(d, rand, 0.25)
}

const ropeFreqCache = new Map<number, number[]>()

function ropeInvFreqs(dim: number, theta = 10000): number[] {
  const hit = ropeFreqCache.get(dim)
  if (hit) return hit
  const freqs: number[] = []
  for (let i = 0; i + 1 < dim; i += 2) freqs.push(1 / theta ** (i / dim))
  ropeFreqCache.set(dim, freqs)
  return freqs
}

function applyRope(vec: number[], pos: number): number[] {
  const dim = vec.length
  const freqs = ropeInvFreqs(dim)
  const out = new Array<number>(dim)
  let p = 0
  for (let i = 0; i + 1 < dim; i += 2, p++) {
    const a = pos * freqs[p]!
    const c = Math.cos(a)
    const s = Math.sin(a)
    const x = vec[i]!
    const y = vec[i + 1]!
    out[i] = x * c - y * s
    out[i + 1] = x * s + y * c
  }
  if (dim & 1) out[dim - 1] = vec[dim - 1]!
  return out
}

function alibiBias(q: number, k: number, head: number, nHeads: number): number {
  const slope = 2 ** (-(head + 1) * (8 / nHeads))
  return slope * (k - q)
}

function kvHeadIndex(head: number, nHeads: number, nKv: number): number {
  const group = Math.max(1, Math.floor(nHeads / nKv))
  return Math.min(nKv - 1, Math.floor(head / group))
}

function projectRows(rows: number[][], weight: number[][]): number[][] {
  const n = rows.length
  const out = new Array<number[]>(n)
  for (let i = 0; i < n; i++) out[i] = matVec(weight, rows[i]!)
  return out
}

function maskScore(
  raw: number,
  qi: number,
  kj: number,
  architecture: ModelConfig['architecture'],
  kind: 'self' | 'cross',
  srcLen: number,
): number {
  if (kind === 'cross') return raw
  if (architecture === 'encoder') return raw
  if (architecture === 'encdec' && qi >= srcLen) {
    const dqi = qi - srcLen
    const dkj = kj - srcLen
    if (dkj > dqi) return Number.NEGATIVE_INFINITY
    return raw
  }
  if (kj > qi) return Number.NEGATIVE_INFINITY
  return raw
}

function runHeads(
  xs: number[][],
  nHeads: number,
  nKvHeads: number,
  dModel: number,
  Wq: number[][],
  Wk: number[][],
  Wv: number[][],
  positional: ModelConfig['positional'],
  architecture: ModelConfig['architecture'],
  kind: 'self' | 'cross',
  kvSource: number[][],
  srcLen: number,
): HeadTrace[] {
  const headDim = dModel / nHeads
  const Q = projectRows(xs, Wq)
  const K = projectRows(kvSource, Wk)
  const V = projectRows(kvSource, Wv)
  const scale = 1 / Math.sqrt(headDim)
  const seqQ = Q.length
  const seqK = K.length
  const useRope = positional === 'rope'
  const useAlibi = positional === 'alibi' && kind === 'self'

  const heads = new Array<HeadTrace>(nHeads)
  for (let h = 0; h < nHeads; h++) {
    const kvh = kvHeadIndex(h, nHeads, nKvHeads)
    const qOff = h * headDim
    const kOff = kvh * headDim
    const qHead = new Array<number[]>(seqQ)
    for (let pos = 0; pos < seqQ; pos++) {
      const slice = Q[pos]!.slice(qOff, qOff + headDim)
      qHead[pos] = useRope ? applyRope(slice, pos) : slice
    }
    const kHead = new Array<number[]>(seqK)
    for (let pos = 0; pos < seqK; pos++) {
      const slice = K[pos]!.slice(kOff, kOff + headDim)
      kHead[pos] = useRope && kind === 'self' ? applyRope(slice, pos) : slice
    }
    const vHead = new Array<number[]>(seqK)
    for (let pos = 0; pos < seqK; pos++) vHead[pos] = V[pos]!.slice(kOff, kOff + headDim)

    const scoresRaw = new Array<number[]>(seqQ)
    const scoresMasked = new Array<number[]>(seqQ)
    const attn = new Array<number[]>(seqQ)
    const out = new Array<number[]>(seqQ)

    for (let qi = 0; qi < seqQ; qi++) {
      const q = qHead[qi]!
      const rawRow = new Array<number>(seqK)
      const maskRow = new Array<number>(seqK)
      const finite = new Array<number>(seqK)
      for (let kj = 0; kj < seqK; kj++) {
        const k = kHead[kj]!
        let s = 0
        for (let d = 0; d < headDim; d++) s += q[d]! * k[d]!
        s *= scale
        if (useAlibi) s += alibiBias(qi, kj, h, nHeads)
        rawRow[kj] = s
        const masked = maskScore(s, qi, kj, architecture, kind, srcLen)
        maskRow[kj] = masked
        finite[kj] = Number.isFinite(masked) ? masked : -1e9
      }
      scoresRaw[qi] = rawRow
      scoresMasked[qi] = maskRow
      const w = softmax(finite, 1)
      attn[qi] = w
      const acc = zeros(headDim)
      for (let j = 0; j < seqK; j++) {
        const alpha = w[j]!
        const vj = vHead[j]!
        for (let d = 0; d < headDim; d++) acc[d]! += alpha * vj[d]!
      }
      out[qi] = acc
    }
    heads[h] = { q: qHead, k: kHead, v: vHead, scoresRaw, scoresMasked, attn, out }
  }
  return heads
}

function concatHeads(heads: HeadTrace[]): number[][] {
  const seq = heads[0]!.out.length
  const nHeads = heads.length
  const headDim = heads[0]!.out[0]!.length
  const width = nHeads * headDim
  const out = new Array<number[]>(seq)
  for (let i = 0; i < seq; i++) {
    const row = new Array<number>(width)
    let o = 0
    for (let h = 0; h < nHeads; h++) {
      const v = heads[h]!.out[i]!
      for (let d = 0; d < headDim; d++) row[o++] = v[d]!
    }
    out[i] = row
  }
  return out
}

function ffnForward(
  xs: number[][],
  kind: ModelConfig['ffn'],
  W1: number[][],
  W2: number[][],
  Wg?: number[][],
): { gate?: number[][]; up?: number[][]; hidden: number[][]; out: number[][] } {
  if (kind === 'swiglu' && Wg) {
    const gate = projectRows(xs, Wg)
    for (let i = 0; i < gate.length; i++) {
      const row = gate[i]!
      for (let j = 0; j < row.length; j++) row[j] = silu(row[j]!)
    }
    const up = projectRows(xs, W1)
    const hidden = new Array<number[]>(gate.length)
    for (let i = 0; i < gate.length; i++) {
      const g = gate[i]!
      const u = up[i]!
      const row = new Array<number>(g.length)
      for (let j = 0; j < g.length; j++) row[j] = g[j]! * u[j]!
      hidden[i] = row
    }
    return { gate, up, hidden, out: projectRows(hidden, W2) }
  }
  const act = kind === 'gelu' ? gelu : relu
  const hidden = projectRows(xs, W1)
  for (let i = 0; i < hidden.length; i++) {
    const row = hidden[i]!
    for (let j = 0; j < row.length; j++) row[j] = act(row[j]!)
  }
  return { hidden, out: projectRows(hidden, W2) }
}

function normRows(xs: number[][], kind: ModelConfig['norm'], gamma: number[], beta: number[]) {
  const n = xs.length
  const out = new Array<number[]>(n)
  if (kind === 'rmsnorm') {
    for (let i = 0; i < n; i++) out[i] = rmsNorm(xs[i]!, gamma)
  } else {
    for (let i = 0; i < n; i++) out[i] = layerNorm(xs[i]!, gamma, beta)
  }
  return out
}

type LayerWeights = {
  gamma1: number[]
  beta1: number[]
  gamma2: number[]
  beta2: number[]
  Wq: number[][]
  Wk: number[][]
  Wv: number[][]
  Wo: number[][]
  W1: number[][]
  W2: number[][]
  Wg: number[][]
  cross?: { WcQ: number[][]; WcK: number[][]; WcV: number[][]; WcO: number[][] }
}

function allocLayerWeights(rand: () => number, dModel: number, withCross: boolean): LayerWeights {
  const hidden = dModel * 2
  const gamma1 = new Array<number>(dModel)
  for (let i = 0; i < dModel; i++) gamma1[i] = 1 + gaussian(rand) * 0.05
  const beta1 = randomVec(dModel, rand, 0.02)
  const gamma2 = new Array<number>(dModel)
  for (let i = 0; i < dModel; i++) gamma2[i] = 1 + gaussian(rand) * 0.05
  const beta2 = randomVec(dModel, rand, 0.02)
  const w: LayerWeights = {
    gamma1,
    beta1,
    gamma2,
    beta2,
    Wq: randomMatrix(dModel, dModel, rand),
    Wk: randomMatrix(dModel, dModel, rand),
    Wv: randomMatrix(dModel, dModel, rand),
    Wo: randomMatrix(dModel, dModel, rand),
    W1: randomMatrix(hidden, dModel, rand),
    W2: randomMatrix(dModel, hidden, rand),
    Wg: randomMatrix(hidden, dModel, rand),
  }
  if (withCross) {
    w.cross = {
      WcQ: randomMatrix(dModel, dModel, rand),
      WcK: randomMatrix(dModel, dModel, rand),
      WcV: randomMatrix(dModel, dModel, rand),
      WcO: randomMatrix(dModel, dModel, rand),
    }
  }
  return w
}

type WeightPack = { encoder?: LayerWeights[]; decoder: LayerWeights[] }

const WEIGHT_CACHE_MAX = 12
const weightCache = new Map<string, WeightPack>()

function weightKey(cfg: ModelConfig): string {
  return `${cfg.seed}|${cfg.nLayers}|${cfg.dModel}|${cfg.architecture === 'encdec' ? 'encdec' : 'mono'}`
}

function getWeightPack(cfg: ModelConfig): WeightPack {
  const key = weightKey(cfg)
  const hit = weightCache.get(key)
  if (hit) {
    weightCache.delete(key)
    weightCache.set(key, hit)
    return hit
  }
  const rand = mulberry32(cfg.seed)
  const pack: WeightPack =
    cfg.architecture === 'encdec'
      ? {
          encoder: Array.from({ length: cfg.nLayers }, () => allocLayerWeights(rand, cfg.dModel, false)),
          decoder: Array.from({ length: cfg.nLayers }, () => allocLayerWeights(rand, cfg.dModel, true)),
        }
      : {
          decoder: Array.from({ length: cfg.nLayers }, () => allocLayerWeights(rand, cfg.dModel, false)),
        }
  weightCache.set(key, pack)
  if (weightCache.size > WEIGHT_CACHE_MAX) {
    const oldest = weightCache.keys().next().value
    if (oldest !== undefined) weightCache.delete(oldest)
  }
  return pack
}

function copyRows(xs: number[][]): number[][] {
  const n = xs.length
  const out = new Array<number[]>(n)
  for (let i = 0; i < n; i++) out[i] = xs[i]!.slice()
  return out
}

function runStack(
  xs: number[][],
  cfg: ModelConfig,
  weights: LayerWeights[],
  encoderOut?: number[][],
  srcLen = 0,
): LayerTrace[] {
  const { dModel, nHeads } = cfg
  const nKv =
    cfg.attention === 'mha' ? nHeads : cfg.attention === 'mqa' ? 1 : cfg.nKvHeads
  let h = copyRows(xs)
  const layers: LayerTrace[] = []

  for (let li = 0; li < weights.length; li++) {
    const w = weights[li]!
    const input = copyRows(h)
    const n1 = normRows(input, cfg.norm, w.gamma1, w.beta1)
    const heads = runHeads(
      n1,
      nHeads,
      nKv,
      dModel,
      w.Wq,
      w.Wk,
      w.Wv,
      cfg.positional,
      cfg.architecture,
      'self',
      n1,
      srcLen,
    )
    const concat = concatHeads(heads)
    const attnProj = projectRows(concat, w.Wo)
    const residual1 = new Array<number[]>(input.length)
    for (let i = 0; i < input.length; i++) residual1[i] = add(input[i]!, attnProj[i]!)

    let cross: LayerTrace['cross']
    let afterSelf = residual1
    if (cfg.architecture === 'encdec' && encoderOut && w.cross) {
      const gammaC = new Array<number>(dModel)
      for (let i = 0; i < dModel; i++) gammaC[i] = 1
      const nC = normRows(residual1, cfg.norm, gammaC, w.beta1)
      const cHeads = runHeads(
        nC,
        nHeads,
        nHeads,
        dModel,
        w.cross.WcQ,
        w.cross.WcK,
        w.cross.WcV,
        'none',
        cfg.architecture,
        'cross',
        encoderOut,
        srcLen,
      )
      const cConcat = concatHeads(cHeads)
      const cProj = projectRows(cConcat, w.cross.WcO)
      afterSelf = new Array<number[]>(residual1.length)
      for (let i = 0; i < residual1.length; i++) afterSelf[i] = add(residual1[i]!, cProj[i]!)
      cross = { heads: cHeads, out: cProj }
    }

    const n2 = normRows(afterSelf, cfg.norm, w.gamma2, w.beta2)
    const ff = ffnForward(n2, cfg.ffn, w.W1, w.W2, w.Wg)
    const residual2 = new Array<number[]>(afterSelf.length)
    for (let i = 0; i < afterSelf.length; i++) residual2[i] = add(afterSelf[i]!, ff.out[i]!)

    const moe =
      li === weights.length - 1
        ? {
            router: n2.map((row) => {
              const slice = row.slice(0, 8)
              slice[0] = (slice[0] ?? 0) + 0.4
              return softmax(slice)
            }),
            chosen: n2.map((_, token) => ({ token, experts: [0, 1 + (token % 3)] })),
          }
        : undefined

    layers.push({
      input,
      norm1: n1,
      heads,
      attnConcat: concat,
      attnProj,
      residual1: afterSelf,
      norm2: n2,
      ffnGate: ff.gate,
      ffnUp: ff.up,
      ffnHidden: ff.hidden,
      ffnOut: ff.out,
      residual2,
      cross,
      moe,
    })
    h = residual2
  }
  return layers
}

function positionMix(
  embeddings: number[][],
  cfg: ModelConfig,
): { posSignal: number[][]; positioned: number[][] } {
  const n = embeddings.length
  const posSignal = new Array<number[]>(n)
  const positioned = new Array<number[]>(n)
  for (let i = 0; i < n; i++) {
    const row = embeddings[i]!
    if (cfg.positional === 'sinusoidal') posSignal[i] = sinusoidal(i, cfg.dModel)
    else if (cfg.positional === 'learned') posSignal[i] = learnedPos(i, cfg.dModel, cfg.seed)
    else posSignal[i] = zeros(cfg.dModel)
    positioned[i] =
      cfg.positional === 'sinusoidal' || cfg.positional === 'learned'
        ? add(row, posSignal[i]!)
        : row.slice()
  }
  return { posSignal, positioned }
}

function kvFromLayer(layer: LayerTrace) {
  const last = layer.heads[0]!.k.length - 1
  return {
    prefillK: layer.heads.map((h) => h.k),
    prefillV: layer.heads.map((h) => h.v),
    decodeQ: layer.heads.map((h) => h.q[last]!),
    decodeScores: layer.heads.map((h) => h.attn[last]!),
  }
}

function dummySampled(vocab: string[]): Trace['sampled'] {
  return {
    candidates: vocab.map((token) => ({ token, logit: 0, filtered: true, prob: 0 })),
    picked: vocab[0] ?? '',
    method: 'greedy',
  }
}

/** Forward pass only. Sampling knobs are applied by `attachSampling`. */
export function runForward(sourceText: string, targetText: string, cfg: ModelConfig): Trace {
  const pack = getWeightPack(cfg)
  const srcTok = tokenize(sourceText, cfg.showBos)
  const tgtTok = cfg.architecture === 'encdec' ? tokenize(targetText || 'yes', true) : undefined

  const srcEmb = srcTok.tokens.map((t) => embedToken(t, cfg.dModel))
  const { posSignal, positioned } = positionMix(srcEmb, cfg)

  let encoderLayers: LayerTrace[] | undefined
  let layers: LayerTrace[]
  if (cfg.architecture === 'encdec' && tgtTok && pack.encoder) {
    const encCfg: ModelConfig = { ...cfg, architecture: 'encoder' }
    encoderLayers = runStack(positioned, encCfg, pack.encoder)
    const encOut = encoderLayers[encoderLayers.length - 1]!.residual2
    const tgtEmb = tgtTok.tokens.map((t) => embedToken(t, cfg.dModel))
    const tgtPos = positionMix(tgtEmb, cfg).positioned
    layers = runStack(tgtPos, cfg, pack.decoder, encOut, srcTok.tokens.length)
  } else {
    layers = runStack(positioned, cfg, pack.decoder)
  }

  const last = layers[layers.length - 1]!
  const gamma = new Array<number>(cfg.dModel)
  for (let i = 0; i < cfg.dModel; i++) gamma[i] = 1
  const beta = zeros(cfg.dModel)
  const finalNorm = new Array<number[]>(last.residual2.length)
  for (let i = 0; i < last.residual2.length; i++) {
    const row = last.residual2[i]!
    finalNorm[i] = cfg.norm === 'rmsnorm' ? rmsNorm(row, gamma) : layerNorm(row, gamma, beta)
  }
  const pooled = zeros(cfg.dModel)
  for (let i = 0; i < finalNorm.length; i++) {
    const row = finalNorm[i]!
    for (let j = 0; j < cfg.dModel; j++) pooled[j]! += row[j]!
  }
  const inv = 1 / Math.max(finalNorm.length, 1)
  for (let j = 0; j < cfg.dModel; j++) pooled[j]! *= inv

  const vocab = demoVocab(tgtTok?.tokens ?? srcTok.tokens)
  const rawLogits = toyLogits(tgtTok?.tokens ?? srcTok.tokens, vocab)

  return {
    config: cfg,
    sourceText,
    targetText,
    tokenize: srcTok,
    targetTokenize: tgtTok,
    embeddings: srcEmb,
    posSignal,
    positioned,
    layers,
    encoderLayers,
    finalNorm,
    pooled,
    vocab,
    rawLogits,
    sampled: dummySampled(vocab),
    kv: kvFromLayer(last),
  }
}

export function attachSampling(trace: Trace, cfg: ModelConfig, sourceText: string, targetText: string): Trace {
  const sampleRand = mulberry32(cfg.seed + hashString(sourceText + '|' + targetText))
  const sampled = applySampling(
    trace.rawLogits,
    trace.vocab,
    cfg.sampling,
    cfg.temperature,
    cfg.topK,
    cfg.topP,
    cfg.minP,
    sampleRand,
  )
  return {
    ...trace,
    config: {
      ...trace.config,
      sampling: cfg.sampling,
      temperature: cfg.temperature,
      topK: cfg.topK,
      topP: cfg.topP,
      minP: cfg.minP,
    },
    sampled: { ...sampled, method: cfg.sampling },
  }
}

export function runModel(sourceText: string, targetText: string, cfg: ModelConfig): Trace {
  return attachSampling(runForward(sourceText, targetText, cfg), cfg, sourceText, targetText)
}

export function tokensForView(trace: Trace): Token[] {
  if (trace.config.architecture === 'encdec' && trace.targetTokenize) return trace.targetTokenize.tokens
  return trace.tokenize.tokens
}
