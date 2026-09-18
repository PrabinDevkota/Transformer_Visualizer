import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { GLOSSARY, STAGES, type StageId } from '../content/stages'
import { entropy, softmax } from '../engine/math'
import { tokensForView } from '../engine/transformer'
import { generateSteps, type GenStep } from '../engine/generate'
import { classifyAttention, patternHint, patternLabel } from '../engine/patterns'
import type { ModelConfig, Trace } from '../engine/types'
import { Heatmap } from '../viz/Heatmap'
import { AttentionArcs } from '../viz/Arcs'
import { Bipartite } from '../viz/Bipartite'
import { NeuronContrib } from '../viz/Neuron'
import { RoPEPlanes } from '../viz/RoPE'
import { Formula, Tex } from '../viz/Math'
import { TokenRow, TokenSource } from '../viz/Tokens'
import { VectorBars, VectorStrip } from '../viz/Vectors'
import { Chip, Slider } from '../components/Fields'
import { sequential } from '../viz/color'
import { ColorSlide } from '../viz/ColorLegend'
import { InspectPanel, fmt } from '../viz/Inspect'

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
  const [openWhy, setOpenWhy] = useState(false)
  return (
    <div className="mx-auto flex min-h-0 w-full max-w-6xl flex-1 flex-col">
      <header className="mb-5 flex flex-col items-center gap-3 text-center">
        <div>
          <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.2em] text-faint">
            {meta.group} · {meta.n}
          </div>
          <h2 className="font-display text-[32px] leading-none tracking-tight text-ink">{meta.title}</h2>
        </div>
        {meta.formula && <Formula latex={meta.formula} />}
      </header>
      <p className="mx-auto mb-4 max-w-2xl text-center text-[15px] leading-6 text-mute">{meta.blurb}</p>
      {meta.id !== 'color' && (
      <div className="mx-auto mb-8 w-full max-w-2xl">
        <button
          type="button"
          onClick={() => setOpenWhy((v) => !v)}
          className="mb-2 font-mono text-[10px] uppercase tracking-[0.18em] text-gold"
        >
          {openWhy ? 'hide why' : 'why this exists'}
        </button>
        {openWhy && (
          <div className="rounded-lg border border-line bg-elev px-5 py-4 text-left">
            {meta.why.map((p, i) => (
              <p key={i} className="mt-2 text-[13px] leading-6 text-mute first:mt-0">
                {p}
              </p>
            ))}
          </div>
        )}
      </div>
      )}
      <AnimatePresence mode="wait">
        <motion.div
          key={props.stage}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.22 }}
          className="flex min-h-0 w-full flex-1 flex-col items-center"
        >
          <div className="w-full">
            <StageBody {...props} />
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

function StageBody(props: Props) {
  switch (props.stage) {
    case 'color':
      return <ColorSlide embedded />
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
    <div className="mb-6 flex flex-wrap items-center justify-center gap-3">
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
  const t = trace.tokenize.tokens[token]
  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_280px]">
      <div>
        <TokenSource text={trace.tokenize.raw} tokens={trace.tokenize.tokens} active={token} onPick={setToken} />
        <div className="mt-5">
          <TokenRow tokens={trace.tokenize.tokens} active={token} onPick={setToken} />
        </div>
        <InspectPanel
          title={t ? `token ${token}` : undefined}
          value={t?.text}
          facts={
            t
              ? [
                  { k: 'id', v: String(t.id) },
                  { k: 'index', v: String(token) },
                  { k: 'span', v: t.special ? 'special' : `[${t.start}, ${t.end})` },
                  { k: 'chars', v: t.special ? '—' : trace.tokenize.raw.slice(t.start, t.end) || t.text },
                  { k: 'kind', v: t.special ? 'special' : 'subword' },
                  { k: 'length', v: String(t.text.length) },
                ]
              : undefined
          }
          note="Tap a token in the sentence or the chip row. BPE merges common character pairs into longer pieces so the vocabulary stays small."
        />
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
            {cur?.tokens.map((piece, i) => (
              <button
                type="button"
                key={`${piece}-${i}`}
                onClick={() => {
                  const idx = trace.tokenize.tokens.findIndex((tok) => tok.text === piece.trim())
                  if (idx >= 0) setToken(idx)
                }}
                className="rounded border border-line bg-elev px-1.5 py-0.5 font-mono text-[12px] text-ink hover:border-gold/50"
              >
                {piece === ' ' ? '▁' : piece}
              </button>
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
        name="embedding"
        note="Each cell is one dimension of the token embedding. Color is signed magnitude — tap a cell for the exact number and the full row."
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
      <div className="mb-6 grid gap-3 sm:grid-cols-3">
        <Analogy title="Query" body="The search you type. “What am I looking for at this token?”" />
        <Analogy title="Key" body="The page title in the results. “What does this token advertise?”" />
        <Analogy title="Value" body="The page body you actually copy if the titles match." />
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        {(['q', 'k', 'v'] as const).map((name) => (
          <div key={name}>
            <h3 className="mb-2 font-mono text-[12px] uppercase tracking-[0.16em] text-gold">{name}</h3>
            <VectorStrip
              rows={h[name]}
              labels={toks.map((t) => t.text)}
              name={name.toUpperCase()}
              note="Tap a cell. Q is “what I look for”, K is “what I contain”, V is “what I pass along”."
            />
          </div>
        ))}
      </div>
      <p className="mt-5 text-center text-[12px] text-mute">
        Head {head} · dim {h.q[0]?.length}. In GQA, several Q heads share the same K/V slice — inspect GQA · MLA for the cache implication.
      </p>
    </div>
  )
}

function Analogy({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-line bg-elev p-3 text-left">
      <div className="font-mono text-[11px] text-gold">{title}</div>
      <p className="mt-1 text-[12px] leading-5 text-mute">{body}</p>
    </div>
  )
}

function ScoresStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  const labels = toks.map((t) => t.text)
  const dh = h.q[0]?.length ?? 1
  const [cell, setCell] = useState({ r: token, c: Math.min(token, labels.length - 1) })
  const q = h.q[cell.r] ?? h.q[token] ?? []
  const k = h.k[cell.c] ?? h.k[token] ?? []
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <Heatmap
        matrix={h.scoresRaw}
        rowLabels={labels}
        colLabels={labels}
        highlight={{ r: cell.r, c: cell.c }}
        onPick={(r, c) => {
          setToken(r)
          setCell({ r, c })
        }}
        explain={(r, c, v) => {
          const qq = h.q[r] ?? []
          const kk = h.k[c] ?? []
          const rawDot = qq.reduce((s, qi, d) => s + qi * (kk[d] ?? 0), 0)
          return {
            title: `${labels[r]} · ${labels[c]}`,
            facts: [
              { k: 'layer / head', v: `L${layer}  H${head}` },
              { k: 'Q·K', v: fmt(rawDot, 5) },
              { k: '√dₕ', v: fmt(Math.sqrt(dh), 4) },
              { k: 'Q·K / √dₕ', v: fmt(v, 5) },
              { k: 'query vec', v: qq.map((x) => fmt(x, 2)).join('  ') },
              { k: 'key vec', v: kk.map((x) => fmt(x, 2)).join('  ') },
            ],
            note: 'This is compatibility before softmax. In a decoder, cells where j > i (the future) get −∞ next. Scroll to the neuron view to see which dimensions built this number.',
          }
        }}
      />
      <div className="mt-8">
        <NeuronContrib
          q={q}
          k={k}
          scale={Math.sqrt(dh)}
          queryLabel={labels[cell.r] ?? ''}
          keyLabel={labels[cell.c] ?? ''}
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
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <p className="mb-3 text-center text-[12px] text-mute">
        Architecture <span className="text-gold">{trace.config.architecture}</span>
        {trace.config.architecture === 'decoder' && ' — causal: upper triangle is −∞ so softmax becomes 0.'}
        {trace.config.architecture === 'encoder' && ' — bidirectional: every token may look at every other token.'}
        {trace.config.architecture === 'encdec' && ' — decoder self-attn is causal; cross-attn (later) is full over the source.'}
      </p>
      <Heatmap
        matrix={h.scoresMasked}
        rowLabels={labels}
        colLabels={labels}
        highlight={{ r: token, c: Math.min(token, labels.length - 1) }}
        onPick={(r) => setToken(r)}
        explain={(r, c, v) => {
          const legal = Number.isFinite(v)
          return {
            title: legal ? `${labels[r]} can see ${labels[c]}` : `${labels[r]} cannot see ${labels[c]}`,
            facts: [
              { k: 'raw score', v: fmt(h.scoresRaw[r]?.[c] ?? 0, 5) },
              { k: 'after mask', v: legal ? fmt(v, 5) : '−∞' },
              { k: 'allowed', v: legal ? 'yes' : 'no' },
              { k: 'rule', v: r >= c ? 'past / self' : 'future' },
            ],
            note: legal
              ? 'This cell stays a real number and will get probability mass in softmax.'
              : 'Softmax(−∞) = 0, so no information flows from this key into the query.',
          }
        }}
      />
    </div>
  )
}

function SoftmaxStage({ trace, layer, setLayer, token, setToken, head, setHead }: Props) {
  const L = trace.layers[layer] ?? trace.layers[0]!
  const h = L.heads[head] ?? L.heads[0]!
  const toks = tokensForView(trace)
  const [attnT, setAttnT] = useState(1)
  const masked = (h.scoresMasked[token] ?? []).map((v) => (Number.isFinite(v) ? v : -1e9))
  const row = softmax(masked, attnT)
  const H = entropy(row)
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <TokenRow tokens={toks} active={token} onPick={setToken} />
      <div className="mx-auto mt-4 max-w-sm">
        <div className="mb-1 text-center text-[10px] uppercase tracking-[0.16em] text-faint">attention temperature</div>
        <Slider value={attnT} min={0.2} max={2} step={0.1} onChange={setAttnT} />
        <p className="mt-1 text-center text-[11px] text-mute">
          Same trick as sampling: divide scores by T before softmax. Low T → one token. High T → blend.
        </p>
      </div>
      <div className="mt-6">
        <VectorBars
          values={row}
          labels={toks.map((t) => t.text)}
          asProb
          name="α"
          note="Softmax of the masked scores for this query. Bars sum to 1. Drag temperature to watch the mass move — Transformer Explainer’s live sampling idea, applied to attention."
        />
      </div>
      <div className="mt-6 mx-auto grid max-w-xl gap-2 font-mono text-[12px] text-mute">
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
          onPick={(r) => setToken(r)}
          explain={(r, c, v) => ({
            title: `P(${toks[c]?.text} | query ${toks[r]?.text})`,
            facts: [
              { k: 'αᵢⱼ', v: fmt(v, 5) },
              { k: 'percent', v: `${(v * 100).toFixed(2)}%` },
              { k: 'score before', v: fmt(h.scoresMasked[r]?.[c] ?? 0, 4) },
            ],
            note: 'Rows are queries, columns are keys. Each row is a probability distribution. The slider above only reshapes the bar chart for the selected query.',
          })}
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
      <p className="mb-3 text-center text-[12px] text-mute">
        BertViz head view: queries on the left, keys on the right. Thickness is attention mass.
      </p>
      <Bipartite tokens={toks} weights={h.attn[token] ?? []} query={token} />
      <div className="mt-8">
        <AttentionArcs tokens={toks} weights={h.attn[token] ?? []} query={token} />
      </div>
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
      <h3 className="mb-2 text-center text-[12px] uppercase tracking-[0.16em] text-faint">
        Model view · layers × heads
      </h3>
      <p className="mb-4 text-center text-[12px] text-mute">
        BertViz’s bird’s-eye: every head of every layer. Click a thumbnail. Labels are rough pattern guesses.
      </p>
      <div className="mb-8 overflow-x-auto">
        <div className="inline-flex min-w-full flex-col items-center gap-3">
          {trace.layers.map((layerTrace, li) => (
            <div key={li} className="flex items-start gap-3">
              <div className="w-8 pt-6 font-mono text-[11px] text-faint">L{li}</div>
              <div className="flex flex-wrap gap-3">
                {layerTrace.heads.map((hd, hi) => {
                  const pat = classifyAttention(hd.attn)
                  const on = layer === li && head === hi
                  return (
                    <button
                      type="button"
                      key={hi}
                      title={patternHint(pat)}
                      onClick={() => {
                        setLayer(li)
                        setHead(hi)
                      }}
                      className={`rounded-md border p-2 ${on ? 'border-gold bg-gold/10' : 'border-line bg-elev'}`}
                    >
                      <div className="mb-1 font-mono text-[10px] text-mute">H{hi}</div>
                      <Heatmap matrix={hd.attn} mode="prob" cell={11} inspect={false} />
                      <div className="mt-1 font-mono text-[10px] text-gold">{patternLabel(pat)}</div>
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
      <p className="mb-4 text-center text-[12px] text-mute">{patternHint(classifyAttention(L.heads[head]!.attn))}</p>
      <div className="mt-6">
        <h3 className="mb-2 text-center text-[12px] uppercase tracking-[0.16em] text-faint">head {head} in detail</h3>
        <Heatmap
          matrix={L.heads[head]!.attn}
          mode="prob"
          rowLabels={toks.map((t) => t.text)}
          colLabels={toks.map((t) => t.text)}
          highlight={{ r: token, c: 0 }}
          onPick={(r) => setToken(r)}
        />
      </div>
      <div className="mt-6">
        <h3 className="mb-2 text-center text-[12px] uppercase tracking-[0.16em] text-faint">
          Concat + <Tex expr="W_O" />
        </h3>
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
      <VectorStrip rows={[trace.finalNorm[token] ?? []]} labels={['h last']} dimLabels name="h last" />
      <div className="mt-8">
        <h3 className="mb-3 text-[12px] uppercase tracking-[0.16em] text-faint">Raw logits</h3>
        <VectorBars values={trace.rawLogits} labels={trace.vocab} />
      </div>
    </div>
  )
}

function SamplingStage({ trace }: Props) {
  const [sel, setSel] = useState(0)
  const [story, setStory] = useState<GenStep[] | null>(null)
  const [shown, setShown] = useState(0)
  const cands = [...trace.sampled.candidates].sort((a, b) => b.logit - a.logit)
  const live = cands.filter((c) => !c.filtered)
  const H = entropy(trace.sampled.candidates.map((c) => c.prob))
  const c = cands[sel] ?? cands[0]
  const temps = [0.3, trace.config.temperature, 1.5]
  const uniqueT = [...new Set(temps.map((t) => Number(t.toFixed(2))))]

  const runGen = () => {
    const steps = generateSteps(trace.sourceText, trace.targetText, trace.config, 8)
    setStory(steps)
    setShown(1)
  }

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
        <div className="mb-6 flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={runGen}
            className="rounded-md border border-gold/40 bg-gold/10 px-3 py-1.5 font-mono text-[11px] text-gold"
          >
            generate continuation
          </button>
          {story && shown < story.length && (
            <button
              type="button"
              onClick={() => setShown((s) => Math.min(story.length, s + 1))}
              className="rounded-md border border-line px-3 py-1.5 font-mono text-[11px] text-mute"
            >
              next token
            </button>
          )}
        </div>
        {story && (
          <div className="mb-6 rounded-lg border border-line bg-elev p-4">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">autoregressive loop</div>
            <p className="font-mono text-[14px] leading-7 text-ink">
              {story[0]?.prompt}
              {story.slice(0, shown).map((st, i) => (
                <span key={i} className="text-gold">
                  {' '}
                  {st.picked}
                </span>
              ))}
            </p>
            <p className="mt-2 text-[12px] text-mute">
              Each gold word was sampled, then fed back as input — Bycroft / Explainer’s “predict, append, repeat.”
            </p>
          </div>
        )}
        <h3 className="mb-3 text-[12px] uppercase tracking-[0.16em] text-faint">Temperature compare</h3>
        <div className="mb-6 grid gap-3 sm:grid-cols-3">
          {uniqueT.map((T) => {
            const p = softmax(trace.rawLogits, T)
            const ranked = p
              .map((prob, i) => ({ prob, tok: trace.vocab[i]! }))
              .sort((a, b) => b.prob - a.prob)
              .slice(0, 5)
            return (
              <div key={T} className="rounded-md border border-line bg-elev p-3">
                <div className="mb-2 font-mono text-[11px] text-gold">T = {T}</div>
                {ranked.map((r) => (
                  <div key={r.tok} className="flex items-center gap-2 py-0.5">
                    <span className="w-14 truncate font-mono text-[11px] text-ink">{r.tok}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-sm bg-soft">
                      <div className="h-full bg-gold" style={{ width: `${r.prob * 100}%` }} />
                    </div>
                    <span className="w-10 text-right font-mono text-[10px] text-mute">{(r.prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )
          })}
        </div>
        <div className="space-y-2">
          {cands.map((cand, i) => (
            <button
              type="button"
              key={cand.token}
              onClick={() => setSel(i)}
              className={`grid w-full grid-cols-[88px_1fr_52px] items-center gap-2 rounded-md px-1 py-1 text-left ${
                sel === i ? 'bg-soft ring-1 ring-gold/40' : ''
              }`}
            >
              <span className={`font-mono text-[12px] ${cand.filtered ? 'text-faint line-through' : 'text-ink'}`}>
                {cand.token}
                {cand.token === trace.sampled.picked ? ' ←' : ''}
              </span>
              <div className="h-2 overflow-hidden rounded-sm bg-soft">
                <motion.div
                  className="h-full rounded-sm"
                  initial={{ width: 0 }}
                  animate={{ width: `${cand.prob * 100}%` }}
                  style={{ background: cand.filtered ? '#3a3b42' : sequential(cand.prob, 1) }}
                />
              </div>
              <span className="text-right font-mono text-[11px] text-mute">{(cand.prob * 100).toFixed(1)}%</span>
            </button>
          ))}
        </div>
        {c && (
          <InspectPanel
            title={`candidate “${c.token}”`}
            value={`${(c.prob * 100).toFixed(2)}%`}
            facts={[
              { k: 'logit', v: fmt(c.logit, 4) },
              { k: 'logit / T', v: fmt(c.logit / Math.max(trace.config.temperature, 1e-8), 4) },
              { k: 'kept', v: c.filtered ? 'no — filtered out' : 'yes' },
              { k: 'drawn', v: c.token === trace.sampled.picked ? 'yes' : 'no' },
              { k: 'rank', v: String(sel + 1) },
              { k: 'entropy', v: `${H.toFixed(3)} bits` },
            ]}
            note={
              c.filtered
                ? 'This token was zeroed by top-k / top-p / min-p / greedy, then the rest was renormalized.'
                : 'After filters, softmax(logit / T) over the kept set. The model samples from these bars.'
            }
          />
        )}
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
  const [sel, setSel] = useState(0)
  const w = decodeScores[head]?.[sel] ?? 0
  const kvec = prefillK[head]?.[sel]
  return (
    <div>
      <Controls heads {...{ trace, token, head, layer, setToken, setHead, setLayer }} />
      <div className="mb-4 flex justify-center gap-6 text-[12px] text-mute">
        <span>
          <i className="mr-2 inline-block h-2 w-2 rounded-full bg-gold not-italic" />
          prefill: write K,V for all {seq} tokens
        </span>
        <span>
          <i className="mr-2 inline-block h-2 w-2 rounded-full bg-teal not-italic" />
          decode: one new Q, attend over cache
        </span>
      </div>
      <div className="flex justify-center overflow-x-auto">
        <div className="flex gap-1">
          {Array.from({ length: seq }, (_, t) => (
            <motion.button
              type="button"
              key={t}
              onClick={() => {
                setSel(t)
                setToken(t)
              }}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: t * 0.06 }}
              className={`w-16 rounded-md border p-2 ${sel === t ? 'border-gold bg-gold/10' : 'border-line bg-elev'}`}
            >
              <div className="mb-1 truncate font-mono text-[10px] text-mute">{toks[t]?.text}</div>
              <div className="h-10 rounded-sm" style={{ background: sequential(decodeScores[head]?.[t] ?? 0, 1) }} />
              <div className="mt-1 font-mono text-[10px] text-faint">{((decodeScores[head]?.[t] ?? 0) * 100).toFixed(0)}%</div>
            </motion.button>
          ))}
        </div>
      </div>
      <InspectPanel
        title={`cached slot ${sel} · ${toks[sel]?.text ?? ''}`}
        value={`${(w * 100).toFixed(2)}%`}
        facts={[
          { k: 'token', v: toks[sel]?.text ?? String(sel) },
          { k: 'position', v: String(sel) },
          { k: 'decode α', v: fmt(w, 5) },
          { k: 'K vector', v: (kvec ?? []).map((x) => fmt(x, 2)).join('  ') },
          { k: 'head', v: String(head) },
        ]}
        note="During decode, this K (and V) is reused. Only the new token computes a fresh Q, K, V; history is read from cache."
      />
      <p className="mt-5 max-w-2xl text-[12px] leading-5 text-mute">
        Cache size ≈ layers × KV heads × sequence × dₕ × 2 (K and V) × bytes. That is why GQA, MLA, sliding windows, and quantization exist. Decode is usually memory-bound: the GPU spends its time reading this cache, not multiplying.
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
  const [sel, setSel] = useState(0)
  const p = row[sel] ?? 0
  const chosen = moe?.chosen[token]?.experts ?? []
  return (
    <div>
      <p className="mb-4 text-center text-[13px] text-mute">
        Router softmax over 8 toy experts for token “{tokensForView(trace)[token]?.text}”. Real DeepSeekMoE uses many fine-grained experts plus shared ones; Llama 4 uses fewer large experts and sometimes alternates dense/MoE layers.
      </p>
      <div className="mx-auto grid max-w-xl grid-cols-4 gap-2">
        {row.map((prob, i) => (
          <motion.button
            type="button"
            key={i}
            onClick={() => setSel(i)}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className={`rounded-md border p-3 ${sel === i ? 'border-gold bg-gold/10' : 'border-line bg-elev'}`}
          >
            <div className="font-mono text-[11px] text-mute">E{i}</div>
            <div className="mt-2 h-16 overflow-hidden rounded-sm bg-soft">
              <motion.div
                className="w-full bg-gold"
                initial={{ height: 0 }}
                animate={{ height: `${prob * 100}%` }}
                style={{ marginTop: `${(1 - prob) * 100}%` }}
              />
            </div>
            <div className="mt-1 font-mono text-[11px] text-ink">{(prob * 100).toFixed(0)}%</div>
          </motion.button>
        ))}
      </div>
      <InspectPanel
        title={`expert ${sel}`}
        value={`${(p * 100).toFixed(2)}%`}
        facts={[
          { k: 'gate', v: fmt(p, 5) },
          { k: 'active', v: chosen.includes(sel) ? 'top-k selected' : 'skipped this token' },
          { k: 'token', v: tokensForView(trace)[token]?.text ?? String(token) },
          { k: 'chosen set', v: chosen.map((e) => `E${e}`).join(', ') },
        ]}
        note="Only the selected experts run their FFN. The others cost no compute for this token — that is the sparsity."
      />
    </div>
  )
}

function SpecStage() {
  const draft = ['the', 'cat', 'sat', 'on']
  const verdict = ['ok', 'ok', 'ok', 'no']
  const why = [
    'Draft and target both wanted “the”. Accept.',
    'Draft and target both wanted “cat”. Accept.',
    'Draft and target both wanted “sat”. Accept.',
    'Draft proposed “on”; target disagrees. Reject and resample from the large model.',
  ]
  const [sel, setSel] = useState(0)
  return (
    <div>
      <div className="mb-6 flex flex-wrap justify-center gap-3">
        {draft.map((t, i) => (
          <motion.button
            type="button"
            key={t}
            onClick={() => setSel(i)}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.18 }}
            className={`rounded-md border px-3 py-2 font-mono text-[13px] ${
              sel === i ? 'ring-1 ring-gold/50' : ''
            } ${
              verdict[i] === 'ok' ? 'border-teal/40 bg-teal/10 text-ink' : 'border-coral/40 bg-coral/10 text-coral'
            }`}
          >
            {t}
            <div className="mt-1 text-[10px] uppercase tracking-wider text-faint">
              {verdict[i] === 'ok' ? 'accept' : 'reject · resample'}
            </div>
          </motion.button>
        ))}
      </div>
      <InspectPanel
        title={`draft token ${sel + 1} · “${draft[sel]}”`}
        value={verdict[sel] === 'ok' ? 'accept' : 'reject'}
        facts={[
          { k: 'role', v: 'draft proposal' },
          { k: 'index', v: String(sel) },
          { k: 'verdict', v: verdict[sel] === 'ok' ? 'matches target' : 'mismatch' },
          { k: 'next step', v: verdict[sel] === 'ok' ? 'keep and check the rest' : 'resample from target logits' },
        ]}
        note={why[sel]}
      />
      <ol className="mx-auto mt-6 max-w-2xl space-y-2 text-[13px] leading-6 text-mute">
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

