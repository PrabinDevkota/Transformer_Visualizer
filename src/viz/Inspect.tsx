export type Fact = { k: string; v: string }

export function InspectPanel({
  title,
  value,
  facts,
  note,
  empty = 'Tap any cell, bar, or token to inspect it',
}: {
  title?: string
  value?: string
  facts?: Fact[]
  note?: string
  empty?: string
}) {
  if (!title && value === undefined) {
    return (
      <div className="mt-4 rounded-lg border border-dashed border-line px-4 py-3 text-center text-[12px] text-faint">
        {empty}
      </div>
    )
  }
  return (
    <div className="mt-4 w-full rounded-lg border border-gold/30 bg-elev px-4 py-3 text-left">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div className="font-mono text-[12px] text-gold">{title}</div>
        {value !== undefined && (
          <div className="font-mono text-[22px] leading-none tabular-nums text-ink">{value}</div>
        )}
      </div>
      {facts && facts.length > 0 && (
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
          {facts.map((f) => (
            <div key={f.k}>
              <dt className="text-[10px] uppercase tracking-[0.14em] text-faint">{f.k}</dt>
              <dd className="break-all font-mono text-[12px] text-ink">{f.v}</dd>
            </div>
          ))}
        </dl>
      )}
      {note && <p className="mt-3 text-[12px] leading-5 text-mute">{note}</p>}
    </div>
  )
}

export function fmt(n: number, d = 4): string {
  if (!Number.isFinite(n)) return '−∞  (masked)'
  const a = Math.abs(n)
  if (a >= 100) return n.toFixed(1)
  if (a >= 10) return n.toFixed(2)
  return n.toFixed(d)
}
