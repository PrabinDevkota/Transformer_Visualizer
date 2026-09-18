import { useState } from 'react'
import { motion } from 'motion/react'
import { diverging, maxAbs, sequential } from './color'
import { CompactScale } from './ColorLegend'
import { formatNum } from '../engine/math'
import { InspectPanel, fmt } from './Inspect'

export function VectorStrip({
  rows,
  labels,
  dimLabels,
  name = 'vector',
  note,
}: {
  rows: number[][]
  labels?: string[]
  dimLabels?: boolean
  name?: string
  note?: string
}) {
  const [sel, setSel] = useState<{ i: number; j: number } | null>(null)
  if (!rows.length) return null
  const abs = maxAbs(rows)
  const d = rows[0]!.length
  const cell = 22
  const picked = sel ? rows[sel.i]?.[sel.j] : undefined
  const row = sel ? rows[sel.i] : undefined

  return (
    <div className="w-full">
      <div className="flex w-full justify-center overflow-x-auto">
        <div className="inline-flex flex-col gap-1.5">
          {dimLabels && (
            <div className="flex gap-px pl-[72px] font-mono text-[10px] text-faint">
              {Array.from({ length: d }, (_, i) => (
                <div key={i} className="w-[22px] text-center">
                  {i}
                </div>
              ))}
            </div>
          )}
          {rows.map((vals, i) => (
            <div key={i} className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setSel({ i, j: sel?.j ?? 0 })}
                className="w-16 truncate text-right font-mono text-[12px] text-mute hover:text-gold"
              >
                {labels?.[i] ?? i}
              </button>
              <div className="flex gap-px">
                {vals.map((v, j) => {
                  const on = sel?.i === i && sel?.j === j
                  return (
                    <motion.button
                      type="button"
                      key={j}
                      title={`${labels?.[i] ?? i} [${j}] = ${formatNum(v, 4)}`}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: (i * d + j) * 0.006 }}
                      onClick={(e) => {
                        e.stopPropagation()
                        setSel({ i, j })
                      }}
                      className="h-6 rounded-[3px] outline-none ring-offset-0"
                      style={{
                        width: cell,
                        background: diverging(v, abs),
                        boxShadow: on ? '0 0 0 1.5px #eceae4' : undefined,
                      }}
                    />
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mx-auto mt-3 w-56">
        <CompactScale mode="signed" />
      </div>
      <InspectPanel
        empty={`Tap a ${name} cell`}
        title={sel ? `${name} · ${labels?.[sel.i] ?? `row ${sel.i}`}` : undefined}
        value={picked !== undefined ? fmt(picked) : undefined}
        facts={
          sel && row
            ? [
                { k: 'token', v: String(labels?.[sel.i] ?? sel.i) },
                { k: 'dimension', v: `d[${sel.j}]` },
                { k: 'value', v: fmt(picked ?? 0, 5) },
                { k: 'row L2', v: fmt(Math.sqrt(row.reduce((s, v) => s + v * v, 0))) },
                { k: 'row mean', v: fmt(row.reduce((s, v) => s + v, 0) / row.length) },
                { k: 'vector', v: row.map((v) => fmt(v, 2)).join('  ') },
              ]
            : undefined
        }
        note={note}
      />
    </div>
  )
}

export function VectorBars({
  values,
  labels,
  highlight,
  name = 'value',
  note,
  asProb,
}: {
  values: number[]
  labels?: string[]
  highlight?: number[]
  name?: string
  note?: string
  asProb?: boolean
}) {
  const [sel, setSel] = useState<number | null>(highlight?.[0] ?? null)
  const max = Math.max(...values.map(Math.abs), 1e-6)
  const i = sel ?? 0
  const v = values[i]
  const sum = values.reduce((s, x) => s + Math.max(x, 0), 0)

  return (
    <div className="w-full">
      <div className="flex h-40 w-full items-end justify-center gap-2">
        {values.map((val, idx) => {
          const h = (Math.abs(val) / max) * 100
          const on = sel === idx || highlight?.includes(idx)
          return (
            <button
              type="button"
              key={idx}
              onClick={() => setSel(idx)}
              className="flex min-w-[28px] flex-col items-center gap-1"
            >
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: `${h}%` }}
                transition={{ type: 'spring', stiffness: 120, damping: 18, delay: idx * 0.03 }}
                className="w-5 origin-bottom rounded-sm"
                style={{
                  background: asProb ? sequential(Math.max(val, 0), 1) : diverging(val, max),
                  outline: sel === idx ? '1.5px solid #eceae4' : undefined,
                  filter: on ? 'brightness(1.12)' : undefined,
                  opacity: highlight && !highlight.includes(idx) && sel !== idx ? 0.3 : 0.95,
                }}
              />
              <span className="max-w-[48px] truncate font-mono text-[10px] text-mute">{labels?.[idx] ?? idx}</span>
            </button>
          )
        })}
      </div>
      <div className="mx-auto mt-3 w-56">
        <CompactScale mode={asProb ? 'prob' : 'signed'} />
      </div>
      <InspectPanel
        empty={`Tap a ${name} bar`}
        title={sel !== null ? `${name} · ${labels?.[i] ?? i}` : undefined}
        value={v !== undefined ? (asProb ? `${(v * 100).toFixed(2)}%` : fmt(v)) : undefined}
        facts={
          sel !== null && v !== undefined
            ? [
                { k: 'item', v: String(labels?.[i] ?? i) },
                { k: 'index', v: String(i) },
                { k: name, v: fmt(v, 5) },
                { k: 'share of mass', v: sum > 0 ? `${((Math.max(v, 0) / sum) * 100).toFixed(1)}%` : '—' },
                { k: 'rank', v: `${1 + values.filter((x) => x > v).length} / ${values.length}` },
              ]
            : undefined
        }
        note={note}
      />
    </div>
  )
}
