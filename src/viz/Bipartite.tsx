import { sequential, tokenColor } from './color'
import { CompactScale } from './ColorLegend'
import type { Token } from '../engine/types'
import { InspectPanel, fmt } from './Inspect'
import { useState } from 'react'

export function Bipartite({
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
  const h = Math.max(220, n * 28)
  const y = (i: number) => 18 + (i * (h - 36)) / Math.max(n - 1, 1)
  const active = sel ?? weights.indexOf(Math.max(...weights, 0))
  const pick = (i: number) => {
    setSel(i)
    onPick?.(i)
  }

  return (
    <div className="w-full">
      <div className="flex w-full justify-center overflow-x-auto">
        <svg width={420} height={h} className="overflow-visible">
          <text x={48} y={12} fill="#5c5b57" fontSize={10} fontFamily="IBM Plex Mono" textAnchor="middle">
            query
          </text>
          <text x={372} y={12} fill="#5c5b57" fontSize={10} fontFamily="IBM Plex Mono" textAnchor="middle">
            key / value
          </text>
          {tokens.map((_, j) => {
            const a = weights[j] ?? 0
            return (
              <line
                key={j}
                x1={72}
                y1={y(query)}
                x2={348}
                y2={y(j)}
                stroke={sequential(a, 1)}
                strokeWidth={0.6 + a * 7}
                opacity={0.2 + a * 0.8}
                className="cursor-pointer"
                onPointerDown={() => pick(j)}
              />
            )
          })}
          {tokens.map((t, i) => (
            <g key={`q${i}`} className="cursor-pointer" onPointerDown={() => pick(i)}>
              <circle cx={48} cy={y(i)} r={i === query ? 6 : 4} fill={tokenColor(i)} stroke={i === query ? '#eceae4' : tokenColor(i)} strokeWidth={i === query ? 2 : 0} />
              <text x={36} y={y(i) + 4} textAnchor="end" fill="#8b8a84" fontSize={11} fontFamily="IBM Plex Mono">
                {t.text}
              </text>
            </g>
          ))}
          {tokens.map((t, i) => (
            <g key={`k${i}`} className="cursor-pointer" onPointerDown={() => pick(i)}>
              <circle cx={372} cy={y(i)} r={active === i ? 6 : 4} fill={tokenColor(i)} stroke={active === i ? '#fff4c2' : tokenColor(i)} strokeWidth={active === i ? 2 : 0} />
              <text x={384} y={y(i) + 4} fill="#8b8a84" fontSize={11} fontFamily="IBM Plex Mono">
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
        empty="Tap a link — BertViz-style head view"
        title={active >= 0 ? `${tokens[query]?.text} → ${tokens[active]?.text}` : undefined}
        value={active >= 0 ? `${((weights[active] ?? 0) * 100).toFixed(2)}%` : undefined}
        facts={
          active >= 0
            ? [
                { k: 'query', v: `${tokens[query]?.text}  ·  i=${query}` },
                { k: 'key', v: `${tokens[active]?.text}  ·  j=${active}` },
                { k: 'weight', v: fmt(weights[active] ?? 0, 5) },
              ]
            : undefined
        }
        note="Left tokens issue queries. Right tokens offer keys/values. Line thickness is how much of that value is mixed into the query — the classic Tensor2Tensor / BertViz head view."
      />
    </div>
  )
}
