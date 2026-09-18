import { add, gelu, layerNorm, matVec, relu, rmsNorm, silu, softmax, zeros } from './math'
import { gaussian, hashString, mulberry32, randomMatrix, randomVec } from './rng'
import { applySampling } from './sampling'
import { tokenize } from './tokenizer'
import { demoVocab, toyLogits } from './toyLm'
import type { HeadTrace, LayerTrace, ModelConfig, Token, Trace } from './types'

function embedToken(token: Token, d: number): number[] {
  const rand = mulberry32(hashString(`emb:${token.text}:${token.id}`))
  return Array.from({ length: d }, () => gaussian(rand) * 0.55)
}

function sinusoidal(pos: number, d: number): number[] {
  return Array.from({ length: d }, (_, i) => {
    const div = pos / 10000 ** ((2 * Math.floor(i / 2)) / d)
    return i % 2 === 0 ? Math.sin(div) : Math.cos(div)
  })
}

function learnedPos(pos: number, d: number, seed: number): number[] {
  const rand = mulberry32(hashString(`pos:${seed}:${pos}`))
  return randomVec(d, rand, 0.25)
}

function applyRope(vec: number[], pos: number, theta = 10000): number[] {
  const out = vec.slice()
  for (let i = 0; i + 1 < vec.length; i += 2) {
    const freq = 1 / theta ** (i / vec.length)
    const a = pos * freq
    const c = Math.cos(a)
    const s = Math.sin(a)
    const x = vec[i]!
    const y = vec[i + 1]!
    out[i] = x * c - y * s
    out[i + 1] = x * s + y * c
  }
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
  return rows.map((r) => matVec(weight, r))
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

  return Array.from({ length: nHeads }, (_, h) => {
    const kvh = kvHeadIndex(h, nHeads, nKvHeads)
    const qOff = h * headDim
    const kOff = kvh * headDim
    const qHead = Q.map((row, pos) => {
      const slice = row.slice(qOff, qOff + headDim)
      return positional === 'rope' ? applyRope(slice, pos) : slice
    })
    const kHead = K.map((row, pos) => {
      const slice = row.slice(kOff, kOff + headDim)
      return positional === 'rope' && kind === 'self' ? applyRope(slice, pos) : slice
    })
    const vHead = V.map((row) => row.slice(kOff, kOff + headDim))

    const scoresRaw = qHead.map((q, qi) =>
      kHead.map((k, kj) => {
        let s = 0
        for (let i = 0; i < headDim; i++) s += q[i]! * k[i]!
        s *= scale
        if (positional === 'alibi' && kind === 'self') s += alibiBias(qi, kj, h, nHeads)
        return s
      }),
    )
    const scoresMasked = scoresRaw.map((row, qi) =>
      row.map((s, kj) => maskScore(s, qi, kj, architecture, kind, srcLen)),
    )
    const attn = scoresMasked.map((row) => {
      const finite = row.map((v) => (Number.isFinite(v) ? v : -1e9))
      return softmax(finite, 1)
    })
    const out = attn.map((w) => {
      const acc = zeros(headDim)
      for (let j = 0; j < vHead.length; j++) {
        for (let d = 0; d < headDim; d++) acc[d]! += w[j]! * vHead[j]![d]!
      }
      return acc
    })
    return { q: qHead, k: kHead, v: vHead, scoresRaw, scoresMasked, attn, out }
  })
}

function concatHeads(heads: HeadTrace[]): number[][] {
  const seq = heads[0]!.out.length
  return Array.from({ length: seq }, (_, i) => heads.flatMap((h) => h.out[i]!))
}

function ffnForward(
  xs: number[][],
  kind: ModelConfig['ffn'],
  W1: number[][],
  W2: number[][],
  Wg?: number[][],
): { gate?: number[][]; up?: number[][]; hidden: number[][]; out: number[][] } {
  if (kind === 'swiglu' && Wg) {
    const gate = projectRows(xs, Wg).map((row) => row.map(silu))
    const up = projectRows(xs, W1)
    const hidden = gate.map((g, i) => g.map((v, j) => v * up[i]![j]!))
    const out = projectRows(hidden, W2)
    return { gate, up, hidden, out }
  }
  const act = kind === 'gelu' ? gelu : relu
  const hidden = projectRows(xs, W1).map((row) => row.map(act))
  const out = projectRows(hidden, W2)
  return { hidden, out }
}

function normRows(xs: number[][], kind: ModelConfig['norm'], gamma: number[], beta: number[]) {
  return xs.map((x) => (kind === 'rmsnorm' ? rmsNorm(x, gamma) : layerNorm(x, gamma, beta)))
}

function runStack(
  xs: number[][],
  cfg: ModelConfig,
  rand: () => number,
  encoderOut?: number[][],
  srcLen = 0,
): LayerTrace[] {
  const { dModel, nHeads, nKvHeads, nLayers } = cfg
  const dHeadTotal = dModel
  const hidden = dModel * 2
  let h = xs.map((r) => r.slice())
  const layers: LayerTrace[] = []

  for (let li = 0; li < nLayers; li++) {
    const gamma1 = Array.from({ length: dModel }, () => 1 + gaussian(rand) * 0.05)
    const beta1 = randomVec(dModel, rand, 0.02)
    const gamma2 = Array.from({ length: dModel }, () => 1 + gaussian(rand) * 0.05)
    const beta2 = randomVec(dModel, rand, 0.02)
    const Wq = randomMatrix(dHeadTotal, dModel, rand)
    const Wk = randomMatrix(dHeadTotal, dModel, rand)
    const Wv = randomMatrix(dHeadTotal, dModel, rand)
    const Wo = randomMatrix(dModel, dModel, rand)
    const W1 = randomMatrix(hidden, dModel, rand)
    const W2 = randomMatrix(dModel, hidden, rand)
    const Wg = randomMatrix(hidden, dModel, rand)

    const input = h.map((r) => r.slice())
    const n1 = normRows(input, cfg.norm, gamma1, beta1)
    const heads = runHeads(
      n1,
      nHeads,
      cfg.attention === 'mha' ? nHeads : cfg.attention === 'mqa' ? 1 : nKvHeads,
      dModel,
      Wq,
      Wk,
      Wv,
      cfg.positional,
      cfg.architecture,
      'self',
      n1,
      srcLen,
    )
    const concat = concatHeads(heads)
    const attnProj = projectRows(concat, Wo)
    const residual1 = input.map((row, i) => add(row, attnProj[i]!))

    let cross: LayerTrace['cross']
    let afterSelf = residual1
    if (cfg.architecture === 'encdec' && encoderOut) {
      const gammaC = Array.from({ length: dModel }, () => 1)
      const nC = normRows(residual1, cfg.norm, gammaC, beta1)
      const WcQ = randomMatrix(dHeadTotal, dModel, rand)
      const WcK = randomMatrix(dHeadTotal, dModel, rand)
      const WcV = randomMatrix(dHeadTotal, dModel, rand)
      const WcO = randomMatrix(dModel, dModel, rand)
      const cHeads = runHeads(
        nC,
        nHeads,
        nHeads,
        dModel,
        WcQ,
        WcK,
        WcV,
        'none',
        cfg.architecture,
        'cross',
        encoderOut,
        srcLen,
      )
      const cConcat = concatHeads(cHeads)
      const cProj = projectRows(cConcat, WcO)
      afterSelf = residual1.map((row, i) => add(row, cProj[i]!))
      cross = { heads: cHeads, out: cProj }
    }

    const n2 = normRows(afterSelf, cfg.norm, gamma2, beta2)
    const ff = ffnForward(n2, cfg.ffn, W1, W2, Wg)
    const residual2 = afterSelf.map((row, i) => add(row, ff.out[i]!))

    const moe =
      li === nLayers - 1
        ? {
            router: n2.map((row) => softmax(row.slice(0, 8).map((v, i) => v + (i === 0 ? 0.4 : 0)))),
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
  const posSignal = embeddings.map((row, i) => {
    if (cfg.positional === 'sinusoidal') return sinusoidal(i, cfg.dModel)
    if (cfg.positional === 'learned') return learnedPos(i, cfg.dModel, cfg.seed)
    if (cfg.positional === 'none' || cfg.positional === 'rope' || cfg.positional === 'alibi') {
      return zeros(cfg.dModel)
    }
    return zeros(row.length)
  })
  const positioned = embeddings.map((row, i) => {
    if (cfg.positional === 'sinusoidal' || cfg.positional === 'learned') return add(row, posSignal[i]!)
    return row.slice()
  })
  return { posSignal, positioned }
}

function kvFromLayer(layer: LayerTrace) {
  const nHeads = layer.heads.length
  const seq = layer.heads[0]!.k.length
  const prefillK = Array.from({ length: nHeads }, (_, h) =>
    Array.from({ length: seq }, (_, t) => layer.heads[h]!.k[t]!),
  )
  const prefillV = Array.from({ length: nHeads }, (_, h) =>
    Array.from({ length: seq }, (_, t) => layer.heads[h]!.v[t]!),
  )
  const last = seq - 1
  const decodeQ = layer.heads.map((h) => h.q[last]!)
  const decodeScores = layer.heads.map((h) => h.attn[last]!)
  return { prefillK, prefillV, decodeQ, decodeScores }
}

export function runModel(sourceText: string, targetText: string, cfg: ModelConfig): Trace {
  const rand = mulberry32(cfg.seed)
  const srcTok = tokenize(sourceText, cfg.showBos)
  const tgtTok = cfg.architecture === 'encdec' ? tokenize(targetText || 'yes', true) : undefined

  const srcEmb = srcTok.tokens.map((t) => embedToken(t, cfg.dModel))
  const { posSignal, positioned } = positionMix(srcEmb, cfg)

  let encoderLayers: LayerTrace[] | undefined
  let layers: LayerTrace[]
  if (cfg.architecture === 'encdec' && tgtTok) {
    const encCfg: ModelConfig = { ...cfg, architecture: 'encoder' }
    encoderLayers = runStack(positioned, encCfg, rand)
    const encOut = encoderLayers.at(-1)!.residual2
    const tgtEmb = tgtTok.tokens.map((t) => embedToken(t, cfg.dModel))
    const tgtPos = positionMix(tgtEmb, cfg).positioned
    layers = runStack(tgtPos, cfg, rand, encOut, srcTok.tokens.length)
  } else {
    layers = runStack(positioned, cfg, rand)
  }

  const last = layers.at(-1)!
  const gamma = Array.from({ length: cfg.dModel }, () => 1)
  const beta = zeros(cfg.dModel)
  const finalNorm = last.residual2.map((row) =>
    cfg.norm === 'rmsnorm' ? rmsNorm(row, gamma) : layerNorm(row, gamma, beta),
  )
  const pooled = finalNorm
    .reduce((acc, row) => acc.map((v, i) => v + row[i]!), zeros(cfg.dModel))
    .map((v) => v / finalNorm.length)

  const vocab = demoVocab(tgtTok?.tokens ?? srcTok.tokens)
  const rawLogits = toyLogits(tgtTok?.tokens ?? srcTok.tokens, vocab)
  const sampleRand = mulberry32(cfg.seed + hashString(sourceText + '|' + targetText))
  const sampled = applySampling(
    rawLogits,
    vocab,
    cfg.sampling,
    cfg.temperature,
    cfg.topK,
    cfg.topP,
    cfg.minP,
    sampleRand,
  )

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
    sampled: { ...sampled, method: cfg.sampling },
    kv: kvFromLayer(last),
  }
}

export function tokensForView(trace: Trace): Token[] {
  if (trace.config.architecture === 'encdec' && trace.targetTokenize) return trace.targetTokenize.tokens
  return trace.tokenize.tokens
}
