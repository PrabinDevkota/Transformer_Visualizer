import { motion } from 'motion/react'
import { diverging, maxAbs } from './color'
import { formatNum } from '../engine/math'

export function VectorStrip({
  rows,
  labels,
  dimLabels,
}: {
  rows: number[][]
  labels?: string[]
  dimLabels?: boolean
}) {
  if (!rows.length) return null
  const abs = maxAbs(rows)
  const d = rows[0]!.length
  const cell = 16

  return (
    <div className="overflow-x-auto">
      <div className="inline-flex flex-col gap-1">
        {dimLabels && (
          <div className="flex gap-px pl-[72px] font-mono text-[9px] text-faint">
            {Array.from({ length: d }, (_, i) => (
              <div key={i} className="w-4 text-center">
                {i}
              </div>
            ))}
          </div>
        )}
        {rows.map((row, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-16 truncate text-right font-mono text-[11px] text-mute">{labels?.[i] ?? i}</div>
            <div className="flex gap-px">
              {row.map((v, j) => (
                <motion.div
                  key={j}
                  title={`${formatNum(v, 3)}`}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: (i * d + j) * 0.008 }}
                  className="h-4 rounded-[2px]"
                  style={{ width: cell, background: diverging(v, abs) }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function VectorBars({
  values,
  labels,
  highlight,
}: {
  values: number[]
  labels?: string[]
  highlight?: number[]
}) {
  const max = Math.max(...values.map(Math.abs), 1e-6)
  return (
    <div className="flex items-end gap-1.5 h-28">
      {values.map((v, i) => {
        const h = (Math.abs(v) / max) * 100
        const on = highlight?.includes(i)
        return (
          <div key={i} className="flex flex-col items-center gap-1 min-w-[22px]">
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: `${h}%` }}
              transition={{ type: 'spring', stiffness: 120, damping: 18, delay: i * 0.03 }}
              className="w-3.5 origin-bottom rounded-sm"
              style={{
                background: v < 0 ? '#c97a63' : on ? '#e8d7b0' : '#d4b483',
                opacity: on === false ? 0.25 : 0.95,
              }}
            />
            <span className="max-w-[36px] truncate font-mono text-[9px] text-mute">{labels?.[i] ?? i}</span>
          </div>
        )
      })}
    </div>
  )
}
