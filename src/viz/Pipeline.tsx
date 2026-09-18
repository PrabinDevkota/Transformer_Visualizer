import clsx from 'clsx'
import type { StageId } from '../content/stages'

const STEPS: { id: StageId; label: string }[] = [
  { id: 'tokenize', label: 'tokens' },
  { id: 'embed', label: 'embed' },
  { id: 'position', label: 'position' },
  { id: 'qkv', label: 'Q K V' },
  { id: 'attend', label: 'attention' },
  { id: 'ffn', label: 'FFN' },
  { id: 'unembed', label: 'logits' },
  { id: 'sampling', label: 'sample' },
]

const ALIAS: Partial<Record<StageId, StageId>> = {
  norm: 'qkv',
  scores: 'attend',
  mask: 'attend',
  softmax: 'attend',
  heads: 'attend',
  residual: 'ffn',
  stack: 'ffn',
  kvcache: 'sampling',
  arch: 'sampling',
  gqa: 'attend',
  moe: 'ffn',
  speculative: 'sampling',
}

export function Pipeline({
  current,
  onPick,
}: {
  current: StageId
  onPick: (id: StageId) => void
}) {
  const mapped = ALIAS[current] ?? current
  const idx = STEPS.findIndex((s) => s.id === mapped)
  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5">
      {STEPS.map((s, i) => (
        <div key={s.id} className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => onPick(s.id)}
            className={clsx(
              'rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider',
              current === s.id || mapped === s.id
                ? 'border-gold bg-gold/15 text-gold'
                : idx >= 0 && i < idx
                  ? 'border-line bg-elev text-mute'
                  : 'border-line text-faint hover:text-ink',
            )}
          >
            {s.label}
          </button>
          {i < STEPS.length - 1 && <span className="text-[10px] text-faint">→</span>}
        </div>
      ))}
    </div>
  )
}
