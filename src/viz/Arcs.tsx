import { motion } from 'motion/react'
import { sequential } from './color'
import type { Token } from '../engine/types'

export function AttentionArcs({
  tokens,
  weights,
  query,
}: {
  tokens: Token[]
  weights: number[]
  query: number
}) {
  const n = tokens.length
  const w = Math.max(360, n * 52)
  const h = 120
  const y = 88
  const xs = tokens.map((_, i) => 28 + (i * (w - 56)) / Math.max(n - 1, 1))

  return (
    <svg width={w} height={h} className="overflow-visible">
      {tokens.map((_, j) => {
        const a = weights[j] ?? 0
        const x1 = xs[query] ?? 0
        const x2 = xs[j] ?? 0
        const cpx = (x1 + x2) / 2
        const cpy = 18 + (1 - a) * 40
        const d = `M ${x1} ${y} Q ${cpx} ${cpy} ${x2} ${y}`
        return (
          <motion.path
            key={j}
            d={d}
            fill="none"
            stroke={sequential(a, 1)}
            strokeWidth={1.2 + a * 4}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.25 + a * 0.75 }}
            transition={{ duration: 0.6, delay: j * 0.05 }}
          />
        )
      })}
      {tokens.map((t, i) => (
        <g key={`n${i}`}>
          <circle
            cx={xs[i]}
            cy={y}
            r={i === query ? 6 : 4}
            fill={i === query ? '#d4b483' : '#2a2b32'}
            stroke="#d4b483"
            strokeWidth={i === query ? 0 : 1}
          />
          <text x={xs[i]} y={h - 6} textAnchor="middle" fill="#8b8a84" fontSize={10} fontFamily="IBM Plex Mono">
            {t.text}
          </text>
        </g>
      ))}
    </svg>
  )
}
