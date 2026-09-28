import clsx from 'clsx'
import { STAGES, type StageId } from '../content/stages'

export function StageNav({
  current,
  onPick,
  horizontal,
}: {
  current: StageId
  onPick: (id: StageId) => void
  horizontal?: boolean
}) {
  if (horizontal) {
    return (
      <nav className="flex gap-1 overflow-x-auto px-3 py-2">
        {STAGES.map((s) => (
          <button
            type="button"
            key={s.id}
            onClick={() => onPick(s.id)}
            className={clsx(
              'shrink-0 rounded-md px-2 py-1 font-mono text-[11px]',
              current === s.id ? 'bg-soft text-ink' : 'text-mute hover:text-ink',
            )}
          >
            {s.n} {s.title}
          </button>
        ))}
      </nav>
    )
  }

  let lastGroup = ''
  return (
    <nav className="flex h-full flex-col gap-0.5 overflow-y-auto pr-1">
      {STAGES.map((s, idx) => {
        const showGroup = s.group !== lastGroup
        lastGroup = s.group
        return (
          <div key={s.id}>
            {showGroup && (
              <div
                className={clsx(
                  'mb-1 px-2 text-[10px] uppercase tracking-[0.18em] text-faint',
                  idx === 0 ? 'mt-0' : 'mt-4',
                )}
              >
                {s.group}
              </div>
            )}
            <button
              type="button"
              onClick={() => onPick(s.id)}
              className={clsx(
                'flex w-full items-baseline gap-2 rounded-md px-2 py-1.5 text-left transition-colors',
                current === s.id ? 'bg-soft text-ink' : 'text-mute hover:bg-elev hover:text-ink',
              )}
            >
              <span className="w-5 font-mono text-[10px] text-faint">{s.n}</span>
              <span className="text-[13px]">{s.title}</span>
            </button>
          </div>
        )
      })}
    </nav>
  )
}
