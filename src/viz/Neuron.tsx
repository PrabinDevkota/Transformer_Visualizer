import { VectorBars } from './Vectors'
import { fmt } from './Inspect'

export function NeuronContrib({
  q,
  k,
  scale,
  queryLabel,
  keyLabel,
}: {
  q: number[]
  k: number[]
  scale: number
  queryLabel: string
  keyLabel: string
}) {
  const parts = q.map((qi, i) => (qi * (k[i] ?? 0)) / scale)
  const sum = parts.reduce((s, v) => s + v, 0)
  return (
    <div className="w-full">
      <h3 className="mb-2 text-center text-[12px] uppercase tracking-[0.16em] text-faint">
        Neuron view · {queryLabel} · {keyLabel}
      </h3>
      <p className="mb-3 text-center text-[12px] text-mute">
        Each bar is one head dimension: q<sub>d</sub> × k<sub>d</sub> / √dₕ. They add up to the score {fmt(sum, 3)}.
      </p>
      <VectorBars
        values={parts}
        labels={parts.map((_, i) => `d${i}`)}
        name="contrib"
        note="BertViz’s neuron view: a score is not magic — it is the sum of these per-dimension products. A large bar means that axis of Q and K agreed."
      />
    </div>
  )
}
