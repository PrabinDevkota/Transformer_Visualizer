import { useState } from 'react'
import { motion } from 'motion/react'
import { sequential, tokenColor } from './color'
import { CompactScale } from './ColorLegend'
import type { Token } from '../engine/types'
import { InspectPanel, fmt } from './Inspect'

export function AttentionArcs({
  tokens,
  weights,
  query,
  onPick,
}: {
  tokens: Token[]
  weights: number[]
  query: number
  onPick?: (i: number) => void
}) {
  const [sel, setSel] = useState<number | null>(null)
  const n = tokens.length
  const w = Math.max(640, n * 88)
  const h = 180
  const y = 132
  const xs = tokens.map((_, i) => 40 + (i * (w - 80)) / Math.max(n - 1, 1))
  const active = sel ?? weights.indexOf(Math.max(...weights, 0))
  const pick = (i: number) => {
    setSel(i)
    onPick?.(i)
  }

  return (
    <div className="w-full">
      <div className="flex w-full justify-center overflow-x-auto">
        <svg width={w} height={h} className="overflow-visible">
          {tokens.map((_, j) => {
            const a = weights[j] ?? 0
            const x1 = xs[query] ?? 0
            const x2 = xs[j] ?? 0
            const cpx = (x1 + x2) / 2
            const cpy = 28 + (1 - a) * 70
            const d = `M ${x1} ${y} Q ${cpx} ${cpy} ${x2} ${y}`
            const on = active === j
            return (
              <motion.path
                key={j}
                d={d}
                fill="none"
                stroke={sequential(a, 1)}
                strokeWidth={(on ? 2.4 : 1.2) + a * 4}
                className="cursor-pointer"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: on ? 1 : 0.2 + a * 0.7 }}
                transition={{ duration: 0.6, delay: j * 0.05 }}
                onPointerDown={() => pick(j)}
              />
            )
          })}
          {tokens.map((t, i) => (
            <g key={`n${i}`} className="cursor-pointer" onPointerDown={() => pick(i)}>
              <circle
                cx={xs[i]}
                cy={y}
                r={i === query || active === i ? 8 : 5}
                fill={tokenColor(i)}
                stroke={i === query ? '#eceae4' : active === i ? '#fff4c2' : tokenColor(i)}
                strokeWidth={i === query || active === i ? 2 : 0}
              />
              <text x={xs[i]} y={h - 8} textAnchor="middle" fill="#8b8a84" fontSize={12} fontFamily="IBM Plex Mono">
                {t.text}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="mx-auto mt-3 w-56">
        <CompactScale mode="prob" />
      </div>
      <InspectPanel
        empty="Tap an arc or token"
        title={
          active >= 0
            ? `${tokens[query]?.text ?? 'q'} attends to ${tokens[active]?.text ?? active}`
            : undefined
        }
        value={active >= 0 ? `${((weights[active] ?? 0) * 100).toFixed(2)}%` : undefined}
        facts={
          active >= 0
            ? [
                { k: 'query', v: `${tokens[query]?.text}  ·  i=${query}` },
                { k: 'value source', v: `${tokens[active]?.text}  ·  j=${active}` },
                { k: 'αᵢⱼ', v: fmt(weights[active] ?? 0, 5) },
                { k: 'rank', v: `${1 + weights.filter((x) => x > (weights[active] ?? 0)).length} / ${n}` },
              ]
            : undefined
        }
        note="Each query mixes value vectors: zᵢ = Σⱼ αᵢⱼ vⱼ. This weight is how much of that token’s value is copied in."
      />
    </div>
  )
}
