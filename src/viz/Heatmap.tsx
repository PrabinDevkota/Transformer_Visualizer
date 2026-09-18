import { motion } from 'motion/react'
import { formatNum } from '../engine/math'
import { diverging, maxAbs, maxVal, sequential } from './color'

type Props = {
  matrix: number[][]
  rowLabels?: string[]
  colLabels?: string[]
  mode?: 'signed' | 'prob'
  highlight?: { r: number; c: number } | null
  onHover?: (r: number, c: number, v: number) => void
  cell?: number
}

export function Heatmap({
  matrix,
  rowLabels,
  colLabels,
  mode = 'signed',
  highlight,
  onHover,
  cell = 22,
}: Props) {
  if (!matrix.length || !matrix[0]?.length) {
    return <div className="text-[12px] text-mute">No matrix yet.</div>
  }
  const rows = matrix.length
  const cols = matrix[0].length
  const abs = maxAbs(matrix)
  const mx = maxVal(matrix)
  const labelW = rowLabels ? 56 : 0
  const labelH = colLabels ? 42 : 0
  const w = labelW + cols * cell + 8
  const h = labelH + rows * cell + 8

  return (
    <svg width={w} height={h} className="overflow-visible font-mono text-[9px]">
      {colLabels?.map((lab, j) => (
        <text
          key={`c${j}`}
          x={labelW + j * cell + cell / 2}
          y={labelH - 8}
          textAnchor="middle"
          fill="#8b8a84"
          transform={`rotate(-35 ${labelW + j * cell + cell / 2} ${labelH - 8})`}
        >
          {lab}
        </text>
      ))}
      {rowLabels?.map((lab, i) => (
        <text key={`r${i}`} x={labelW - 6} y={labelH + i * cell + cell / 2 + 3} textAnchor="end" fill="#8b8a84">
          {lab}
        </text>
      ))}
      {matrix.map((row, i) =>
        row.map((v, j) => {
          const fill = mode === 'prob' ? sequential(v, mx) : diverging(v, abs)
          const on = highlight?.r === i && highlight?.c === j
          return (
            <motion.rect
              key={`${i}-${j}`}
              x={labelW + j * cell + 1}
              y={labelH + i * cell + 1}
              width={cell - 2}
              height={cell - 2}
              rx={3}
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: (i * cols + j) * 0.012, duration: 0.25 }}
              fill={fill}
              stroke={on ? '#eceae4' : 'transparent'}
              strokeWidth={on ? 1.2 : 0}
              onMouseEnter={() => onHover?.(i, j, v)}
              style={{ cursor: onHover ? 'crosshair' : 'default' }}
            />
          )
        }),
      )}
    </svg>
  )
}

export function HoverReadout({
  label,
  value,
}: {
  label: string
  value: string | number | null
}) {
  return (
    <div className="flex items-baseline gap-2 font-mono text-[11px] text-mute">
      <span>{label}</span>
      <span className="text-ink">{typeof value === 'number' ? formatNum(value, 3) : (value ?? '—')}</span>
    </div>
  )
}
