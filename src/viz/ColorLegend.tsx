import { useState, type ReactNode } from 'react'
import {
  colorBarCss,
  diverging,
  DIVERGING_STOPS,
  ink,
  sequential,
  SEQUENTIAL_STOPS,
  TOKEN_COLORS,
} from './color'

function Bar({
  stops,
  left,
  mid,
  right,
}: {
  stops: [number, string][]
  left: string
  mid?: string
  right: string
}) {
  return (
    <div>
      <div className="h-3 w-full rounded-sm" style={{ background: colorBarCss(stops) }} />
      <div className="mt-1 flex justify-between font-mono text-[10px] text-faint">
        <span>{left}</span>
        {mid && <span>{mid}</span>}
        <span>{right}</span>
      </div>
    </div>
  )
}

export function CompactScale({ mode = 'signed' }: { mode?: 'signed' | 'prob' }) {
  if (mode === 'prob') {
    return <Bar stops={SEQUENTIAL_STOPS} left="0 · ignore" mid="some" right="1 · focus" />
  }
  return <Bar stops={DIVERGING_STOPS} left="−  negative" mid="0" right="+  positive" />
}

export function PipelineLegend() {
  return (
    <div className="mx-auto mt-5 flex max-w-xl flex-wrap items-end justify-center gap-x-8 gap-y-3">
      <div className="w-48">
        <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.16em] text-faint">signed · QKV / embed</div>
        <CompactScale mode="signed" />
      </div>
      <div className="w-48">
        <div className="mb-1 font-mono text-[9px] uppercase tracking-[0.16em] text-faint">mass · attention / p</div>
        <CompactScale mode="prob" />
      </div>
    </div>
  )
}

const SIGNED_DEMO = [-1, -0.6, -0.25, 0, 0.25, 0.6, 1]
const PROB_DEMO = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.3]

function DemoCells({
  values,
  mode,
}: {
  values: number[]
  mode: 'signed' | 'prob'
}) {
  const [sel, setSel] = useState(mode === 'prob' ? values.length - 1 : 0)
  const v = values[sel] ?? 0
  return (
    <div>
      <div className="flex gap-1">
        {values.map((x, i) => {
          const fill = mode === 'prob' ? sequential(x, 1) : diverging(x, 1)
          return (
            <button
              type="button"
              key={i}
              onClick={() => setSel(i)}
              className="h-9 flex-1 rounded-sm font-mono text-[10px]"
              style={{
                background: fill,
                color: ink(fill),
                boxShadow: sel === i ? '0 0 0 1.5px #eceae4' : undefined,
              }}
            >
              {mode === 'prob' ? `${Math.round(x * 100)}` : x.toFixed(1)}
            </button>
          )
        })}
      </div>
      <p className="mt-2 font-mono text-[11px] text-mute">
        {mode === 'prob'
          ? `p = ${(v * 100).toFixed(0)}% · ${v >= 0.2 ? 'this is where the budget went' : v >= 0.08 ? 'some mass' : 'almost ignored'}`
          : `${v > 0 ? 'positive — adds' : v < 0 ? 'negative — subtracts' : 'near zero — quiet'} · ${v.toFixed(2)}`}
      </p>
    </div>
  )
}

function Beat({ kicker, title, children }: { kicker: string; title: string; children: ReactNode }) {
  return (
    <section className="mt-10 first:mt-0">
      <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-gold">{kicker}</div>
      <h4 className="mt-1 font-display text-[20px] tracking-tight text-ink">{title}</h4>
      <div className="story mt-3 space-y-3">{children}</div>
    </section>
  )
}

function ColorRead() {
  return (
    <article className="border-t border-line pt-8">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-gold">Read</div>
      <h3 className="mt-2 font-display text-[26px] leading-tight tracking-tight text-ink">
        How to see a transformer
      </h3>
      <p className="story-lede mt-5 text-[16px] leading-8 text-ink">
        You are not looking at a picture of a transformer. You are looking at its numbers, dyed so your eye can do
        arithmetic. The rest of this page is the legend for that dye — from one square up to the whole residual stream.
      </p>

      <Beat kicker="I" title="Start with one square">
        <p>
          That square is one entry. It might be one dimension of one token’s embedding. It might be one query attending
          to one key. The fill <em>is</em> the value. There is no extra meaning hiding in the hue. If two squares look
          the same, their numbers are close — in magnitude, and in sign.
        </p>
        <p>
          Tap it. The inspect panel is the same number written as digits: the row (query), the column (key), the rank
          in that row, sometimes the share of mass. Nothing important lives only on hover. If you cannot tap it, it is
          not a number you are meant to read.
        </p>
      </Beat>

      <Beat kicker="II" title="Hue is the sign">
        <p>
          Cool <b>blue</b> means the component is negative. It subtracts. It inhibits. It points the other way. Warm{' '}
          <b>amber</b> means positive. It adds. It activates. The dead-center charcoal is zero: that axis is quiet.
        </p>
        <p>
          This is the language of embeddings, of Q, K, and V, of residuals, of raw attention scores before softmax — any
          tensor that is allowed to be negative. A row of mixed blue and gold is a vector with both directions in it.
          That is normal. Features in a residual stream are signed; a “cat” direction and its opposite live on the same
          line.
        </p>
        <aside>
          <b>Where you will see it.</b> Embed, Position, Norm, QKV, Scores (pre-softmax), Residual, FFN, the neuron
          view (each bar is q<sub>d</sub> × k<sub>d</sub> / √dₕ). Neuron bars that go amber agreed; bars that go blue
          fought. They still add up to the score.
        </aside>
      </Beat>

      <Beat kicker="III" title="Brightness is the size">
        <p>
          A pale gold cell is a large positive. A dim steel cell is a small negative. Brightness tracks how far the
          number sits from zero — |value| — not how “important” the token is.
        </p>
        <p>
          Do not confuse a loud coordinate with a loud token. A word can be doing all the work and still have many quiet
          dimensions. Importance lives in what the rest of the net does with the vector: which keys it matches, which
          values it copies. The inspect panel’s row L2 and row mean are the size of the whole vector, if you want that
          instead of one cell.
        </p>
      </Beat>

      <Beat kicker="IV" title="Probability speaks a different dialect">
        <p>
          After softmax, every number is between 0 and 1, and a full attention row sums to 1. There is no negative
          attention. So the palette changes on purpose: <b>near-black</b> is “this key is ignored,” <b>teal</b> is
          “some mass,” <b>bright gold</b> is “this is where the budget went.”
        </p>
        <p>
          You are watching a finite budget being spent, not a signed field. That is why an attention heatmap must never
          look like an embedding strip. Sampling bars use the same gold scale: the bright bar is the token the model is
          about to say. Temperature, top-k, top-p, min-p only change how that gold is distributed — they do not change
          what gold means.
        </p>
        <aside>
          <b>Where you will see it.</b> Softmax, Attend, Heads, Arcs, Bipartite, Sampling. Line thickness on the arcs
          and the bipartite view is the same mass: thick gold is a large α<sub>ij</sub>. A row of gold always adds to
          one. If it does not, you are not looking at probabilities.
        </aside>
      </Beat>

      <Beat kicker="V" title="Names that survive the matrices">
        <p>
          Tokens need a color that is not a number. The left stripe on a chip is identity. “cat” keeps that stripe from
          tokenize through Q, K, the heatmap labels, the arcs, the bipartite nodes. Stripe is not magnitude. Follow the
          stripe with your eye; read the fill for the value.
        </p>
        <p>
          The sentence view underlines each piece with the same stripe, so you can find the span in the original text
          without trusting the subword spelling. Special tokens (BOS, EOS) stay unmarked — they are punctuation for the
          model, not words for you.
        </p>
        <div className="flex flex-wrap gap-1.5 pt-1">
          {TOKEN_COLORS.map((c, i) => (
            <span
              key={c}
              className="rounded border border-line px-2 py-0.5 font-mono text-[11px] text-ink"
              style={{ boxShadow: `inset 3px 0 0 ${c}` }}
            >
              t{i}
            </span>
          ))}
        </div>
      </Beat>

      <Beat kicker="VI" title="You are the ring">
        <p>
          A white or gold outline is you. That is the cell, token, head, or bar you tapped. The query node in the
          bipartite and arc views wears a white ring; its fill is still the token’s stripe. Selection never recodes the
          number. It only says <em>this one</em>.
        </p>
        <p>
          Play steps the story forward through the pipeline. Arrow keys do the same. Space pauses. The footer’s{' '}
          <b>colors</b> opens this slide without leaving the matrix you were on — a legend you can hold beside the
          arithmetic.
        </p>
      </Beat>

      <Beat kicker="VII" title="Doors that are locked">
        <p>
          Some cells refuse to be numbers. In a decoder, the future is masked: the score is −∞, the square stays the
          blackest black, and softmax gives it 0. That is not a quiet dimension. That is a door that is locked. The
          inspect panel will say −∞, and “future? yes.” Encoder self-attention has no such door — every token may look
          at every other. Cross-attention is a third shape: decoder queries, encoder keys, no causal mask.
        </p>
        <p>
          GQA and MQA change how many unique key-value colors exist, not what those colors mean. Several query heads
          share one KV head; you will see fewer distinct KV rows, and the same sequential gold when they attend.
        </p>
      </Beat>

      <Beat kicker="VIII" title="Walking the model with your eyes">
        <p>
          Find “cat” by its stripe. Open <b>Embed</b> — a row of blue and amber, one cell per dimension, the residual
          stream’s first state. Open <b>Q K V</b> — three more rows, same stripe, different numbers: they were
          multiplied by W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>. Open <b>Scores</b> — still signed; a large amber
          cell is a key this query likes, a blue cell is a key it rejects. Open <b>Softmax</b> — the language switches.
          Gold is now a budget. The gold cells in that row are the tokens whose value vectors will be mixed into “cat.”
        </p>
        <p>
          Open <b>FFN</b> and the residual again — signed, because the stream is being edited, not replaced. Open{' '}
          <b>Sample</b> — gold over the vocabulary. Temperature sharpens or flattens that gold; top-k and top-p throw
          some of it away and renormalize. The next token is whichever gold bar you (or the sampler) pick.
        </p>
        <blockquote>
          Same token, same stripe, every page. Signed fill for tensors that may go negative. Gold fill for mass that
          must sum to one. Outline for you. Blackest black for a lock, not a zero.
        </blockquote>
      </Beat>

      <Beat kicker="IX" title="Why not a rainbow">
        <p>
          Jet and its cousins invent ridges where the math is smooth, and they collapse for red–green color vision.
          Cool–warm for signed tensors and dark-to-gold for mass are two encodings that cannot be mistaken for each
          other. An embedding strip will never look like an attention row. That is the whole design.
        </p>
      </Beat>

      <section className="mt-12 rounded-xl border border-line bg-soft/50 px-5 py-5">
        <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-gold">Keep</div>
        <dl className="mt-4 grid gap-3 text-[13px] leading-6 sm:grid-cols-[7.5rem_1fr]">
          <dt className="font-mono text-[11px] text-ink">Signed</dt>
          <dd className="text-mute">Blue − · charcoal 0 · amber +. Embeddings, QKV, residuals, raw scores, neuron bars.</dd>
          <dt className="font-mono text-[11px] text-ink">Mass</dt>
          <dd className="text-mute">Dark ignore · teal some · gold focus. Softmax rows, arcs, bipartite, next-token bars. A row sums to 1.</dd>
          <dt className="font-mono text-[11px] text-ink">Stripe</dt>
          <dd className="text-mute">Who the token is. Not how large. Same stripe from the sentence to the last matrix.</dd>
          <dt className="font-mono text-[11px] text-ink">Outline</dt>
          <dd className="text-mute">The thing you tapped. Never recodes the value.</dd>
          <dt className="font-mono text-[11px] text-ink">−∞</dt>
          <dd className="text-mute">Causal mask. Blackest cell. Softmax → 0. Not a quiet dimension — a locked future.</dd>
          <dt className="font-mono text-[11px] text-ink">Brightness</dt>
          <dd className="text-mute">|value|, not importance. A crucial token may still be a quiet row of near-zeros.</dd>
          <dt className="font-mono text-[11px] text-ink">Tap</dt>
          <dd className="text-mute">Every cell, bar, arc, and chip. Rank, share of mass, L2, and the exact float live in the panel underneath.</dd>
        </dl>
      </section>
    </article>
  )
}

export function ColorSlide({ embedded }: { embedded?: boolean }) {
  return (
    <div
      className={
        embedded
          ? 'mx-auto w-full max-w-3xl'
          : 'mx-auto max-w-3xl rounded-2xl border border-line bg-elev px-8 py-8 shadow-[0_0_0_1px_#26272e] sm:px-10 sm:py-10'
      }
    >
      {!embedded && (
        <>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-faint">Slide · encoding</div>
          <h2 className="mt-2 font-display text-[28px] tracking-tight text-ink">Color is the number</h2>
          <p className="mt-3 text-[15px] leading-7 text-mute">
            Nothing here is decorative. Hue and brightness are the value. If two cells look the same, they are close in
            magnitude — and in sign.
          </p>
        </>
      )}

      <div className={embedded ? 'space-y-8' : 'mt-8 space-y-8'}>
        <section>
          <h3 className="mb-2 font-mono text-[11px] text-gold">Signed values · cool–warm</h3>
          <CompactScale mode="signed" />
          <div className="mt-3">
            <DemoCells values={SIGNED_DEMO} mode="signed" />
          </div>
        </section>

        <section>
          <h3 className="mb-2 font-mono text-[11px] text-gold">Attention &amp; probabilities · dark → gold</h3>
          <CompactScale mode="prob" />
          <div className="mt-3">
            <DemoCells values={PROB_DEMO} mode="prob" />
          </div>
        </section>
      </div>

      <div className="mt-12">
        <ColorRead />
      </div>
    </div>
  )
}
