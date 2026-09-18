import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { diverging, ink, MASK_COLOR, maxAbs, maxVal, sequential } from './color'
import { CompactScale } from './ColorLegend'
import { InspectPanel, fmt, type Fact } from './Inspect'

export type CellPick = { r: number; c: number; v: number }

type Props = {
  matrix: number[][]
  rowLabels?: string[]
  colLabels?: string[]
  mode?: 'signed' | 'prob'
  highlight?: { r: number; c: number } | null
  onPick?: (r: number, c: number, v: number) => void
  cell?: number
  explain?: (r: number, c: number, v: number) => { title?: string; facts?: Fact[]; note?: string }
  empty?: string
  inspect?: boolean
}

export function Heatmap({
  matrix,
  rowLabels,
  colLabels,
  mode = 'signed',
  highlight,
  onPick,
  cell,
  explain,
  empty,
  inspect = true,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState(720)
  const [sel, setSel] = useState<CellPick | null>(
    highlight ? { r: highlight.r, c: highlight.c, v: matrix[highlight.r]?.[highlight.c] ?? 0 } : null,
  )

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setBox(el.clientWidth)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  if (!matrix.length || !matrix[0]?.length) {
    return <div className="text-[12px] text-mute">No matrix yet.</div>
  }

  const rows = matrix.length
  const cols = matrix[0].length
  const abs = maxAbs(matrix.map((row) => row.map((v) => (Number.isFinite(v) ? v : 0))))
  const mx = maxVal(matrix.map((row) => row.map((v) => (Number.isFinite(v) ? v : 0))))
  const labelW = rowLabels ? 80 : 8
  const labelH = colLabels ? 56 : 8
  const autoCell = Math.floor((box - labelW - 16) / Math.max(cols, 1))
  const cellSize = cell ?? Math.max(40, Math.min(80, autoCell || 52))
  const font = cellSize >= 40 ? 11 : 9
  const showNum = cellSize >= 34
  const w = labelW + cols * cellSize + 8
  const h = labelH + rows * cellSize + 8
  const active = sel ?? (highlight ? { r: highlight.r, c: highlight.c, v: matrix[highlight.r]?.[highlight.c] ?? 0 } : null)

  const pick = (r: number, c: number, v: number) => {
    setSel({ r, c, v })
    onPick?.(r, c, v)
  }

  const extra = active ? explain?.(active.r, active.c, active.v) : undefined
  const row = active ? (rowLabels?.[active.r] ?? `row ${active.r}`) : ''
  const col = active ? (colLabels?.[active.c] ?? `col ${active.c}`) : ''
  const rank =
    active &&
    1 +
      matrix[active.r]!.filter((x, j) => j !== active.c && (Number.isFinite(x) ? x : -1e9) > (Number.isFinite(active.v) ? active.v : -1e9))
        .length

  return (
    <div className="w-full">
      <div ref={ref} className="flex w-full justify-center overflow-x-auto">
        <svg width={w} height={h} className="overflow-visible font-mono select-none" style={{ fontSize: font }}>
          {colLabels?.map((lab, j) => (
            <text
              key={`c${j}`}
              x={labelW + j * cellSize + cellSize / 2}
              y={labelH - 10}
              textAnchor="middle"
              fill="#8b8a84"
              className="cursor-pointer"
              onClick={() => pick(active?.r ?? 0, j, matrix[active?.r ?? 0]![j]!)}
              transform={`rotate(-32 ${labelW + j * cellSize + cellSize / 2} ${labelH - 10})`}
            >
              {lab}
            </text>
          ))}
          {rowLabels?.map((lab, i) => (
            <text
              key={`r${i}`}
              x={labelW - 8}
              y={labelH + i * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              fill="#8b8a84"
              className="cursor-pointer"
              onClick={() => pick(i, active?.c ?? i, matrix[i]![active?.c ?? i]!)}
            >
              {lab}
            </text>
          ))}
          {matrix.map((rowVals, i) =>
            rowVals.map((v, j) => {
              const fill = !Number.isFinite(v)
                ? MASK_COLOR
                : mode === 'prob'
                  ? sequential(v, mx)
                  : diverging(v, abs)
              const on = active?.r === i && active?.c === j
              const label =
                !Number.isFinite(v) || v <= -2.5
                  ? '−∞'
                  : mode === 'prob'
                    ? v >= 0.01
                      ? `${Math.round(v * 100)}%`
                      : v.toFixed(2)
                    : v.toFixed(1)
              return (
                <g
                  key={`${i}-${j}`}
                  className="cursor-pointer"
                  onPointerDown={(e) => {
                    e.preventDefault()
                    pick(i, j, v)
                  }}
                >
                  <motion.rect
                    x={labelW + j * cellSize + 1.5}
                    y={labelH + i * cellSize + 1.5}
                    width={cellSize - 3}
                    height={cellSize - 3}
                    rx={4}
                    initial={{ opacity: 0, scale: 0.7 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (i * cols + j) * 0.008, duration: 0.22 }}
                    fill={fill}
                    stroke={on ? '#eceae4' : 'transparent'}
                    strokeWidth={on ? 2 : 0}
                  />
                  {showNum && (
                    <text
                      x={labelW + j * cellSize + cellSize / 2}
                      y={labelH + i * cellSize + cellSize / 2 + 4}
                      textAnchor="middle"
                      fill={ink(fill)}
                      fontSize={cellSize >= 56 ? 11 : 9}
                      style={{ pointerEvents: 'none' }}
                    >
                      {label}
                    </text>
                  )}
                </g>
              )
            }),
          )}
        </svg>
      </div>
      {inspect && (
        <div className="mx-auto mt-3 w-64">
          <CompactScale mode={mode === 'prob' ? 'prob' : 'signed'} />
        </div>
      )}
      {inspect && (
      <InspectPanel
        empty={empty}
        title={active ? extra?.title ?? `${row} → ${col}` : undefined}
        value={active ? (mode === 'prob' && Number.isFinite(active.v) ? `${(active.v * 100).toFixed(2)}%` : fmt(active.v)) : undefined}
        facts={
          active
            ? [
                { k: 'query (row)', v: `${row}  ·  i=${active.r}` },
                { k: 'key (col)', v: `${col}  ·  j=${active.c}` },
                { k: mode === 'prob' ? 'probability' : 'score', v: fmt(active.v, 5) },
                { k: 'rank in row', v: rank ? `${rank} / ${cols}` : '—' },
                { k: 'future?', v: active.c > active.r ? 'yes — masked in decoder' : 'no' },
                ...(extra?.facts ?? []),
              ]
            : undefined
        }
        note={extra?.note}
      />
      )}
    </div>
  )
}
