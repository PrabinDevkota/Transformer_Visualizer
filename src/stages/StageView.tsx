import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { GLOSSARY, STAGES, type StageId } from '../content/stages'
import { entropy, formatNum } from '../engine/math'
import { tokensForView } from '../engine/transformer'
import type { ModelConfig, Trace } from '../engine/types'
import { Heatmap, HoverReadout } from '../viz/Heatmap'
import { AttentionArcs } from '../viz/Arcs'
import { Formula, RoPEPlanes } from '../viz/RoPE'
import { TokenRow, TokenSource } from '../viz/Tokens'
import { VectorBars, VectorStrip } from '../viz/Vectors'
import { Chip } from '../components/Fields'
import { sequential } from '../viz/color'

type Props = {
  stage: StageId
  trace: Trace
  token: number
  head: number
  layer: number
  setToken: (i: number) => void
  setHead: (i: number) => void
  setLayer: (i: number) => void
}

export function StageView(props: Props) {
  const meta = STAGES.find((s) => s.id === props.stage)!
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="mb-5 flex items-end justify-between gap-6">
        <div>
          <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.2em] text-faint">
            {meta.group} · {meta.n}
          </div>
          <h2 className="font-display text-[28px] leading-none tracking-tight text-ink">{meta.title}</h2>
        </div>
        {meta.formula && <Formula>{meta.formula}</Formula>}
      </header>
      <p className="mb-6 max-w-3xl text-[14px] leading-6 text-mute">{meta.blurb}</p>
      <AnimatePresence mode="wait">
        <motion.div
          key={props.stage}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.22 }}
          className="min-h-0 flex-1"
        >
          <StageBody {...props} />
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

function StageBody(props: Props) {
  switch (props.stage) {
    case 'tokenize':
      return <TokenizeStage {...props} />
    case 'embed':
      return <EmbedStage {...props} />
    case 'position':
      return <PositionStage {...props} />
    case 'norm':
      return <NormStage {...props} />
    case 'qkv':
      return <QkvStage {...props} />
    case 'scores':
      return <ScoresStage {...props} />
    case 'mask':
      return <MaskStage {...props} />
    case 'softmax':
      return <SoftmaxStage {...props} />
    case 'attend':
      return <AttendStage {...props} />
    case 'heads':
      return <HeadsStage {...props} />
    case 'residual':
      return <ResidualStage {...props} />
    case 'ffn':
      return <FfnStage {...props} />
    case 'stack':
      return <StackStage {...props} />
    case 'unembed':
      return <UnembedStage {...props} />
    case 'sampling':
      return <SamplingStage {...props} />
    case 'kvcache':
      return <KvStage {...props} />
    case 'arch':
      return <ArchStage cfg={props.trace.config} />
    case 'gqa':
      return <GqaStage cfg={props.trace.config} />
    case 'moe':
      return <MoeStage {...props} />
    case 'speculative':
      return <SpecStage />
  }
}

function Controls({
  trace,
  token,
  head,
  layer,
  setToken,
  setHead,
  setLayer,
  heads,
}: {
  trace: Trace
  token: number
  head: number
  layer: number
  setToken: (i: number) => void
  setHead: (i: number) => void
  setLayer: (i: number) => void
  heads?: boolean
}) {
  const tokens = tokensForView(trace)
  const L = trace.layers.length
  const H = trace.layers[0]?.heads.length ?? 0
  return (
    <div className="mb-4 flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-1">
        {Array.from({ length: L }, (_, i) => (
          <Chip key={i} on={layer === i} onClick={() => setLayer(i)}>
            L{i}
          </Chip>
        ))}
      </div>
      {heads && (
        <div className="flex items-center gap-1">
          {Array.from({ length: H }, (_, i) => (
            <Chip key={i} on={head === i} onClick={() => setHead(i)}>
              H{i}
            </Chip>
          ))}
        </div>
      )}
      <div className="flex items-center gap-1">
        <Chip onClick={() => setToken(Math.max(0, token - 1))}>prev tok</Chip>
        <Chip onClick={() => setToken(Math.min(tokens.length - 1, token + 1))}>next tok</Chip>
      </div>
      <div className="text-[11px] text-mute">
        query token <span className="font-mono text-gold">{tokens[token]?.text ?? '—'}</span>
      </div>
    </div>
  )
}

function TokenizeStage({ trace, token, setToken }: Props) {
  const [step, setStep] = useState(0)
  const steps = trace.tokenize.steps
  const cur = steps[Math.min(step, steps.length - 1)]
  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_280px]">
      <div>
        <TokenSource text={trace.tokenize.raw} tokens={trace.tokenize.tokens} active={token} />
        <div className="mt-5">
          <TokenRow tokens={trace.tokenize.tokens} active={token} onPick={setToken} />
        </div>
        <div className="mt-8">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[12px] text-mute">BPE merges on this sentence</span>
            <div className="flex gap-1">
              <Chip on={false} onClick={() => setStep((s) => Math.max(0, s - 1))}>
                prev
              </Chip>
              <Chip on={false} onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))}>
                merge
              </Chip>
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            {cur?.tokens.map((t, i) => (
              <span
                key={`${t}-${i}`}
                className="rounded border border-line bg-elev px-1.5 py-0.5 font-mono text-[12px] text-ink"
              >
                {t === ' ' ? '▁' : t}
              </span>
            ))}
          </div>
          {cur?.merge && (
            <p className="mt-3 font-mono text-[12px] text-gold">
              {cur.merge.left} + {cur.merge.right} → {cur.merge.result}
            </p>
          )}
        </div>
      </div>
      <aside className="space-y-3 text-[12px] leading-5 text-mute">
        <p>Byte-pair encoding starts from characters (or bytes) and repeatedly merges the most common adjacent pair. GPT-2/4-class models use BPE; T5/Gemma often use SentencePiece Unigram.</p>
        <p>Special tokens like <span className="font-mono text-ink">&lt;bos&gt;</span> and <span className="font-mono text-ink">&lt;eos&gt;</span> mark sequence edges. They are real vocabulary entries, not decoration.</p>
        <p className="font-mono text-[11px] text-faint">{trace.tokenize.tokens.length} tokens · ids are stable hashes in this toy</p>
        {trace.targetTokenize && (
          <div className="rounded-md border border-line bg-elev p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.16em] text-faint">Decoder prefix</div>
            <TokenRow tokens={trace.targetTokenize.tokens} />
          </div>
        )}
      </aside>
    </div>
  )
}

function EmbedStage({ trace, token, setToken }: Props) {
  const toks = trace.tokenize.tokens
  return (
    <div>
      <TokenRow tokens={toks} active={token} onPick={setToken} />
      <p className="mt-4 mb-3 text-[12px] text-mute">
        Embedding table lookup — each row is d_model = {trace.config.dModel}. Color is signed magnitude.
      </p>
      <VectorStrip
        rows={trace.embeddings}
        labels={toks.map((t) => t.text)}
        dimLabels
      />
      <p className="mt-6 max-w-2xl text-[12px] leading-5 text-mute">
        Nearby tokens in a trained model have related directions. This demo uses seeded vectors so the same token always maps to the same row — useful for watching the pipeline, not for semantics.
      </p>
    </div>
  )
}

function PositionStage({ trace, token, setToken }: Props) {
  const cfg = trace.config
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div>
        <TokenRow tokens={trace.tokenize.tokens} active={token} onPick={setToken} />
        <p className="mt-4 mb-3 text-[12px] text-mute">
          {cfg.positional === 'rope' && 'RoPE does not add a vector. It rotates Q and K at each token index. Change the query token to see the angle.'}
          {cfg.positional === 'sinusoidal' && 'Classic Transformer: a deterministic sin/cos vector is added to the embedding.'}
          {cfg.positional === 'learned' && 'GPT-2 style: a learned vector per index, added to the token embedding. Length is frozen at train context.'}
          {cfg.positional === 'alibi' && 'ALiBi adds a distance bias inside attention scores instead of a position vector.'}
          {cfg.positional === 'none' && 'NoPE — order must leak from the causal mask and depth. Some long-context hybrids mix RoPE and NoPE layers.'}
        </p>
        {(cfg.positional === 'sinusoidal' || cfg.positional === 'learned') && (
          <VectorStrip rows={trace.posSignal} labels={trace.tokenize.tokens.map((t) => t.text)} />
        )}
        {cfg.positional === 'rope' && <RoPEPlanes pos={token} dim={8} />}
      </div>
      <div>
        <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">After position mix</h3>
        <VectorStrip rows={trace.positioned} labels={trace.tokenize.tokens.map((t) => t.text)} />
      </div>
    </div>
  )
}

function NormStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="grid gap-8 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Residual in</h3>
          <VectorStrip rows={L.input} labels={toks.map((t) => t.text)} />
        </div>
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">
            {trace.config.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm'} out
          </h3>
          <VectorStrip rows={L.norm1} labels={toks.map((t) => t.text)} />
        </div>
      </div>
    </div>
  )
}

function QkvStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="grid gap-6 lg:grid-cols-3">
        {(['q', 'k', 'v'] as const).map((name) => (
          <div key={name}>
            <h3 className="mb-2 font-mono text-[12px] uppercase tracking-[0.16em] text-gold">{name}</h3>
            <VectorStrip rows={h[name]} labels={toks.map((t) => t.text)} />
          </div>
        ))}
      </div>
      <p className="mt-5 text-[12px] text-mute">
        Head {head} · dim {h.q[0]?.length}. In GQA, several Q heads share the same K/V slice — inspect GQA · MLA for the cache implication.
      </p>
    </div>
  )
}

function ScoresStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  const [hov, setHov] = useState<{ r: number; c: number; v: number } | null>(null)
  const labels = toks.map((t) => t.text)
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <Heatmap
        matrix={h.scoresRaw}
        rowLabels={labels}
        colLabels={labels}
        highlight={hov ? { r: hov.r, c: hov.c } : { r: token, c: token }}
        onHover={(r, c, v) => setHov({ r, c, v })}
      />
      <div className="mt-3">
        <HoverReadout
          label="Q · K"
          value={hov ? `${labels[hov.r]} → ${labels[hov.c]}  ${formatNum(hov.v, 3)}` : 'hover a cell'}
        />
      </div>
    </div>
  )
}

function MaskStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  const labels = toks.map((t) => t.text)
  const vis = h.scoresMasked.map((row) => row.map((v) => (Number.isFinite(v) ? v : -3)))
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <p className="mb-3 text-[12px] text-mute">
        Architecture <span className="text-gold">{trace.config.architecture}</span>
        {trace.config.architecture === 'decoder' && ' — causal: upper triangle is −∞ so softmax becomes 0.'}
        {trace.config.architecture === 'encoder' && ' — bidirectional: every token may look at every other token.'}
        {trace.config.architecture === 'encdec' && ' — decoder self-attn is causal; cross-attn (later) is full over the source.'}
      </p>
      <Heatmap matrix={vis} rowLabels={labels} colLabels={labels} highlight={{ r: token, c: Math.min(token, labels.length - 1) }} />
    </div>
  )
}

function SoftmaxStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  const row = h.attn[token] ?? h.attn[0]!
  const H = entropy(row)
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <TokenRow tokens={toks} active={token} onPick={setToken} />
      <div className="mt-6">
        <VectorBars values={row} labels={toks.map((t) => t.text)} />
      </div>
      <div className="mt-6 grid max-w-xl gap-2 font-mono text-[12px] text-mute">
        <div>Σ p = {row.reduce((s, v) => s + v, 0).toFixed(3)} (must be 1)</div>
        <div>entropy = {H.toFixed(3)} bits · {H < 1 ? 'peaky' : H > 2 ? 'diffuse' : 'mixed'}</div>
      </div>
      <div className="mt-6">
        <Heatmap
          matrix={h.attn}
          rowLabels={toks.map((t) => t.text)}
          colLabels={toks.map((t) => t.text)}
          mode="prob"
          highlight={{ r: token, c: row.indexOf(Math.max(...row)) }}
          cell={20}
        />
      </div>
    </div>
  )
}

function AttendStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <AttentionArcs tokens={toks} weights={h.attn[token] ?? []} query={token} />
      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Values</h3>
          <VectorStrip rows={h.v} labels={toks.map((t) => t.text)} />
        </div>
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Head output (weighted V)</h3>
          <VectorStrip rows={h.out} labels={toks.map((t) => t.text)} />
        </div>
      </div>
    </div>
  )
}

function HeadsStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="grid gap-4 sm:grid-cols-2">
        {L.heads.map((h, i) => (
          <button key={i} type="button" onClick={() => setHead(i)} className="rounded-lg border border-line bg-elev p-3 text-left">
            <div className="mb-2 font-mono text-[11px] text-mute">head {i}{i === head ? ' · selected' : ''}</div>
            <Heatmap matrix={h.attn} mode="prob" cell={12} />
          </button>
        ))}
      </div>
      <div className="mt-6">
        <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Concat + W_O</h3>
        <VectorStrip rows={L.attnProj} labels={toks.map((t) => t.text)} />
      </div>
    </div>
  )
}

function ResidualStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="grid gap-6 lg:grid-cols-3">
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">x</h3>
          <VectorStrip rows={L.input} labels={toks.map((t) => t.text)} />
        </div>
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Attn(x)</h3>
          <VectorStrip rows={L.attnProj} labels={toks.map((t) => t.text)} />
        </div>
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">x + Attn(x)</h3>
          <VectorStrip rows={L.residual1} labels={toks.map((t) => t.text)} />
        </div>
      </div>
    </div>
  )
}

function FfnStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <p className="mb-4 text-[12px] text-mute">
        Activation: <span className="text-gold">{trace.config.ffn}</span>
        {trace.config.ffn === 'swiglu' && ' — gate ⊙ up, then down-project. SiLU(x) = x·σ(x).'}
      </p>
      <div className="grid gap-6 lg:grid-cols-2">
        {L.ffnGate && (
          <div>
            <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">SiLU gate</h3>
            <VectorStrip rows={L.ffnGate.map((r) => r.slice(0, 16))} labels={toks.map((t) => t.text)} />
          </div>
        )}
        <div>
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">FFN write-back</h3>
          <VectorStrip rows={L.ffnOut} labels={toks.map((t) => t.text)} />
        </div>
      </div>
      <div className="mt-6">
        <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Block output (after 2nd residual)</h3>
        <VectorStrip rows={L.residual2} labels={toks.map((t) => t.text)} />
      </div>
    </div>
  )
}

function StackStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const toks = tokensForView(trace)
  return (
    <div>
      <Controls {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="relative ml-4 border-l border-line pl-6">
        {trace.layers.map((L, i) => (
          <button
            type="button"
            key={i}
            onClick={() => setLayer(i)}
            className="relative mb-4 w-full rounded-lg border border-line bg-elev p-3 text-left"
          >
            <span className="absolute -left-[31px] top-4 h-3 w-3 rounded-full border border-gold bg-bg" />
            <div className="mb-2 font-mono text-[11px] text-mute">layer {i}</div>
            <VectorStrip rows={[L.residual2[token] ?? []]} labels={[toks[token]?.text ?? '']} />
          </button>
        ))}
        <div className="relative rounded-lg border border-gold/30 bg-gold/5 p-3">
          <span className="absolute -left-[31px] top-4 h-3 w-3 rounded-full bg-gold" />
          <div className="mb-2 font-mono text-[11px] text-gold">final RMSNorm</div>
          <VectorStrip rows={[trace.finalNorm[token] ?? []]} labels={[toks[token]?.text ?? '']} />
        </div>
      </div>
    </div>
  )
}

function UnembedStage({ trace, token, setToken }: Props) {
  const toks = tokensForView(trace)
  if (trace.config.architecture === 'encoder') {
    return (
      <div>
        <p className="mb-4 text-[13px] text-mute">
          Encoder-only models (BERT, embedding towers) do not predict the next token. They pool a sequence vector — often the first token ([CLS]) or a mean.
        </p>
        <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Mean-pooled representation</h3>
        {trace.pooled && <VectorBars values={trace.pooled} />}
      </div>
    )
  }
  return (
    <div>
      <TokenRow tokens={toks} active={token} onPick={setToken} />
      <p className="mt-4 mb-3 text-[12px] text-mute">
        Last-token hidden state is projected to a tiny demo vocabulary. Production models use 32k–200k+ tokens; the linear map is |V| × d.
      </p>
      <VectorStrip rows={[trace.finalNorm[token] ?? []]} labels={['h_last']} dimLabels />
      <div className="mt-8">
        <h3 className="mb-3 text-[12px] uppercase tracking-[0.16em] text-faint">Raw logits</h3>
        <VectorBars values={trace.rawLogits} labels={trace.vocab} />
      </div>
    </div>
  )
}

function SamplingStage({ trace }: Props) {
  const cands = [...trace.sampled.candidates].sort((a, b) => b.logit - a.logit)
  const live = cands.filter((c) => !c.filtered)
  const H = entropy(trace.sampled.candidates.map((c) => c.prob))
  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_240px]">
      <div>
        <p className="mb-4 text-[12px] text-mute">
          Method <span className="text-gold">{trace.sampled.method}</span>
          {' · '}T={trace.config.temperature}
          {' · '}top-k={trace.config.topK}
          {' · '}top-p={trace.config.topP}
          {' · '}min-p={trace.config.minP}
        </p>
        <div className="space-y-2">
          {cands.map((c) => (
            <div key={c.token} className="grid grid-cols-[88px_1fr_52px] items-center gap-2">
              <span className={`font-mono text-[12px] ${c.filtered ? 'text-faint line-through' : 'text-ink'}`}>
                {c.token}
                {c.token === trace.sampled.picked ? ' ←' : ''}
              </span>
              <div className="h-2 overflow-hidden rounded-sm bg-soft">
                <motion.div
                  className="h-full rounded-sm"
                  initial={{ width: 0 }}
                  animate={{ width: `${c.prob * 100}%` }}
                  style={{ background: c.filtered ? '#3a3b42' : sequential(c.prob, 1) }}
                />
              </div>
              <span className="text-right font-mono text-[11px] text-mute">{(c.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <p className="mt-5 font-mono text-[12px] text-gold">
          drew “{trace.sampled.picked}” · {live.length} tokens kept · entropy {H.toFixed(2)} bits
        </p>
        <p className="mt-3 max-w-xl text-[12px] leading-5 text-mute">
          Next-token scores here come from a tiny n-gram demo so temperature and nucleus sampling are readable. Production logits come from the unembedding of a trained model; the filters are identical.
        </p>
      </div>
      <aside className="space-y-4 text-[12px] leading-5 text-mute">
        <p><b className="text-ink font-medium">Greedy</b> — always argmax. Deterministic, often repetitive.</p>
        <p><b className="text-ink font-medium">Temperature</b> — ℓ/T. Low T ≈ greedy. High T ≈ chaos.</p>
        <p><b className="text-ink font-medium">Top-k</b> — hard cap on candidates.</p>
        <p><b className="text-ink font-medium">Top-p</b> — keep a probability nucleus. Size adapts per step.</p>
        <p><b className="text-ink font-medium">Min-p</b> — drop anything below min_p × p_max.</p>
        <p><b className="text-ink font-medium">Beam</b> — keep several partial strings. Common in translation, rare in chat.</p>
      </aside>
    </div>
  )
}

function KvStage({ trace, head, setHead, token, setToken, layer, setLayer }: Props) {
  const { prefillK, decodeQ, decodeScores } = trace.kv
  const toks = tokensForView(trace)
  const seq = prefillK[0]?.length ?? 0
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="mb-4 flex gap-6 text-[12px] text-mute">
        <span>
          <i className="mr-2 inline-block h-2 w-2 rounded-full bg-gold not-italic" />
          prefill: write K,V for all {seq} tokens
        </span>
        <span>
          <i className="mr-2 inline-block h-2 w-2 rounded-full bg-teal not-italic" />
          decode: one new Q, attend over cache
        </span>
      </div>
      <div className="overflow-x-auto">
        <div className="flex gap-1">
          {Array.from({ length: seq }, (_, t) => (
            <motion.div
              key={t}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: t * 0.06 }}
              className="w-14 rounded-md border border-line bg-elev p-2"
            >
              <div className="mb-1 truncate font-mono text-[10px] text-mute">{toks[t]?.text}</div>
              <div className="h-10 rounded-sm" style={{ background: sequential(decodeScores[head]?.[t] ?? 0, 1) }} />
            </motion.div>
          ))}
          <motion.div
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-14 rounded-md border border-gold/40 bg-gold/10 p-2"
          >
            <div className="mb-1 font-mono text-[10px] text-gold">new</div>
            <div className="h-10 rounded-sm bg-gold/40" />
          </motion.div>
        </div>
      </div>
      <p className="mt-5 max-w-2xl text-[12px] leading-5 text-mute">
        Cache size ≈ layers × kv_heads × seq × d_head × 2 (K and V) × bytes. That is why GQA, MLA, sliding windows, and quantization exist. Decode is usually memory-bound: the GPU spends its time reading this cache, not multiplying.
      </p>
      {decodeQ[head] && (
        <div className="mt-6">
          <h3 className="mb-2 text-[12px] uppercase tracking-[0.16em] text-faint">Decode query (last token)</h3>
          <VectorBars values={decodeQ[head]!} />
        </div>
      )}
    </div>
  )
}

function ArchStage({ cfg }: { cfg: ModelConfig }) {
  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <ArchCard
        title="Encoder-only"
        on={cfg.architecture === 'encoder'}
        lines={['BERT, embedding models', 'Bidirectional mask', 'Pooled vector, no next-token', 'MLM / NLU']}
      />
      <ArchCard
        title="Decoder-only"
        on={cfg.architecture === 'decoder'}
        lines={['GPT, Llama, Qwen, Gemma', 'Causal mask', 'Autoregressive generation', 'The 2024–2026 default']}
      />
      <ArchCard
        title="Encoder–decoder"
        on={cfg.architecture === 'encdec'}
        lines={['Original Transformer, T5', 'Encoder full + decoder causal', 'Cross-attention to source', 'Translation, FIM, ASR']}
      />
      <div className="lg:col-span-3 rounded-lg border border-line bg-elev p-5">
        <svg viewBox="0 0 760 200" className="w-full">
          <ArchBlock x={20} y={50} label="tokens" />
          <ArchBlock x={150} y={50} label="embed + pos" />
          <ArchBlock x={300} y={20} label="self-attn" sub={cfg.architecture === 'encoder' ? 'full' : 'causal'} />
          {cfg.architecture === 'encdec' && <ArchBlock x={300} y={110} label="cross-attn" sub="to encoder" />}
          <ArchBlock x={450} y={50} label={cfg.ffn} />
          <ArchBlock x={600} y={50} label={cfg.architecture === 'encoder' ? 'pool' : 'logits'} />
          <line x1="108" y1="80" x2="150" y2="80" stroke="#d4b483" />
          <line x1="238" y1="80" x2="300" y2="50" stroke="#d4b483" />
          <line x1="388" y1="50" x2="450" y2="80" stroke="#d4b483" />
          <line x1="538" y1="80" x2="600" y2="80" stroke="#d4b483" />
        </svg>
        <p className="mt-3 text-[12px] text-mute">
          Switch architecture in the top bar. Masking, cross-attention, and the output head update immediately.
        </p>
      </div>
    </div>
  )
}

function ArchCard({ title, on, lines }: { title: string; on: boolean; lines: string[] }) {
  return (
    <div className={`rounded-lg border p-4 ${on ? 'border-gold/50 bg-gold/5' : 'border-line bg-elev'}`}>
      <h3 className="mb-2 text-[15px] text-ink">{title}</h3>
      <ul className="space-y-1 text-[12px] text-mute">
        {lines.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </div>
  )
}

function ArchBlock({ x, y, label, sub }: { x: number; y: number; label: string; sub?: string }) {
  return (
    <g>
      <rect x={x} y={y} width={88} height={60} rx={6} fill="#101114" stroke="#26272e" />
      <text x={x + 44} y={y + 34} textAnchor="middle" fill="#eceae4" fontSize={11}>
        {label}
      </text>
      {sub && (
        <text x={x + 44} y={y + 48} textAnchor="middle" fill="#8b8a84" fontSize={9}>
          {sub}
        </text>
      )}
    </g>
  )
}

function GqaStage({ cfg }: { cfg: ModelConfig }) {
  const q = cfg.nHeads
  const kv = cfg.attention === 'mha' ? q : cfg.attention === 'mqa' ? 1 : cfg.nKvHeads
  return (
    <div>
      <p className="mb-5 text-[13px] text-mute">
        Current attention: <span className="text-gold">{cfg.attention.toUpperCase()}</span> · {q} query heads · {kv} KV heads
      </p>
      <svg viewBox="0 0 640 220" className="w-full max-w-3xl">
        {Array.from({ length: q }, (_, i) => (
          <g key={`q${i}`}>
            <rect x={20} y={16 + i * 48} width={72} height={36} rx={4} fill="#101114" stroke="#d4b483" />
            <text x={56} y={38 + i * 48} textAnchor="middle" fill="#d4b483" fontSize={11}>
              Q{i}
            </text>
          </g>
        ))}
        {Array.from({ length: kv }, (_, i) => (
          <g key={`k${i}`}>
            <rect x={280} y={16 + i * (180 / kv)} width={72} height={36} rx={4} fill="#101114" stroke="#7aa8a0" />
            <text x={316} y={38 + i * (180 / kv)} textAnchor="middle" fill="#7aa8a0" fontSize={11}>
              KV{i}
            </text>
          </g>
        ))}
        {Array.from({ length: q }, (_, i) => {
          const kvh = cfg.attention === 'mha' ? i : cfg.attention === 'mqa' ? 0 : Math.min(kv - 1, Math.floor(i / Math.max(1, q / kv)))
          const y1 = 34 + i * 48
          const y2 = 34 + kvh * (180 / kv)
          return <line key={`l${i}`} x1="92" y1={y1} x2="280" y2={y2} stroke="#8b8a84" />
        })}
        <text x={500} y={40} fill="#8b8a84" fontSize={12}>
          cache ∝ KV heads
        </text>
        <text x={500} y={60} fill="#eceae4" fontSize={13}>
          {kv} × seq × d
        </text>
      </svg>
      <div className="mt-6 grid gap-4 md:grid-cols-3 text-[12px] leading-5 text-mute">
        <p><b className="text-ink font-medium">MHA</b> — one KV per query head. Best quality, fattest cache. GPT-2, original Transformer.</p>
        <p><b className="text-ink font-medium">GQA</b> — groups of queries share KV. Llama 2/3, Mistral, Qwen. The current default.</p>
        <p><b className="text-ink font-medium">MLA</b> — cache a low-rank latent, not full KV. DeepSeek-V2/V3. Compress then absorb into RoPE.</p>
      </div>
    </div>
  )
}

function MoeStage({ trace, token }: Props) {
  const moe = trace.layers.at(-1)?.moe
  const row = moe?.router[token] ?? moe?.router[0] ?? []
  return (
    <div>
      <p className="mb-4 text-[13px] text-mute">
        Router softmax over 8 toy experts for token “{tokensForView(trace)[token]?.text}”. Real DeepSeekMoE uses many fine-grained experts plus shared ones; Llama 4 uses fewer large experts and sometimes alternates dense/MoE layers.
      </p>
      <div className="grid grid-cols-4 gap-2 max-w-xl">
        {row.map((p, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="rounded-md border border-line bg-elev p-3"
          >
            <div className="font-mono text-[11px] text-mute">E{i}</div>
            <div className="mt-2 h-16 overflow-hidden rounded-sm bg-soft">
              <motion.div
                className="w-full bg-gold"
                initial={{ height: 0 }}
                animate={{ height: `${p * 100}%` }}
                style={{ marginTop: `${(1 - p) * 100}%` }}
              />
            </div>
            <div className="mt-1 font-mono text-[11px] text-ink">{(p * 100).toFixed(0)}%</div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

function SpecStage() {
  const draft = ['the', 'cat', 'sat', 'on']
  const verdict = ['ok', 'ok', 'ok', 'no']
  return (
    <div>
      <div className="mb-6 flex flex-wrap gap-3">
        {draft.map((t, i) => (
          <motion.div
            key={t}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.18 }}
            className={`rounded-md border px-3 py-2 font-mono text-[13px] ${
              verdict[i] === 'ok' ? 'border-teal/40 bg-teal/10 text-ink' : 'border-coral/40 bg-coral/10 text-coral'
            }`}
          >
            {t}
            <div className="mt-1 text-[10px] uppercase tracking-wider text-faint">
              {verdict[i] === 'ok' ? 'accept' : 'reject · resample'}
            </div>
          </motion.div>
        ))}
      </div>
      <ol className="max-w-2xl space-y-2 text-[13px] leading-6 text-mute">
        <li>1. A small draft model proposes k tokens serially (cheap).</li>
        <li>2. The large model scores those k+1 positions in one forward pass (parallel, uses the unused compute of decode).</li>
        <li>3. Accept the longest matching prefix. On the first mismatch, resample from the large model’s distribution.</li>
        <li>4. If done correctly, the output law equals the large model — faster, not approximate.</li>
      </ol>
    </div>
  )
}

export function Glossary() {
  return (
    <div className="space-y-4">
      {GLOSSARY.map((g) => (
        <div key={g.term}>
          <div className="font-mono text-[11px] text-gold">{g.term}</div>
          <p className="mt-1 text-[12px] leading-5 text-mute">{g.body}</p>
        </div>
      ))}
    </div>
  )
}

