import clsx from 'clsx'
import type { ReactNode } from 'react'

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex min-w-0 flex-col gap-1">
      <span className="text-[10px] uppercase tracking-[0.16em] text-faint">{label}</span>
      {children}
    </label>
  )
}

export function Select({
  value,
  onChange,
  options,
}: {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-8 w-full truncate rounded-md border border-line bg-elev px-2 font-mono text-[11px] text-ink outline-none hover:border-gold/40 focus:border-gold"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-8 w-full accent-[#d4b483]"
      />
      <span className="w-10 text-right font-mono text-[11px] text-gold">{value}</span>
    </div>
  )
}

export function Chip({
  on,
  children,
  onClick,
}: {
  on?: boolean
  children: ReactNode
  onClick?: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'rounded-md border px-2 py-1 font-mono text-[11px]',
        on ? 'border-gold bg-gold/10 text-gold' : 'border-line text-mute hover:text-ink',
      )}
    >
      {children}
    </button>
  )
}
