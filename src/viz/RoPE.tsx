import { useState } from 'react'
import { motion } from 'motion/react'
import { InspectPanel, fmt } from './Inspect'

export function RoPEPlanes({ pos, dim = 8 }: { pos: number; dim?: number }) {
  const planes = Math.floor(dim / 2)
  const [sel, setSel] = useState(0)
  const theta = pos / 10000 ** ((2 * sel) / dim)
  const deg = (theta * 180) / Math.PI

  return (
    <div className="w-full">
      <div className="flex flex-wrap justify-center gap-6">
        {Array.from({ length: planes }, (_, i) => {
          const th = pos / 10000 ** ((2 * i) / dim)
          const d = (th * 180) / Math.PI
          const on = sel === i
          return (
            <button type="button" key={i} onClick={() => setSel(i)} className="flex flex-col items-center gap-2">
              <svg width={88} height={88} viewBox="-44 -44 88 88">
                <circle r={36} fill="none" stroke={on ? '#d4b483' : '#26272e'} />
                <line x1="-36" y1="0" x2="36" y2="0" stroke="#26272e" />
                <line x1="0" y1="-36" x2="0" y2="36" stroke="#26272e" />
                <motion.line
                  x1="0"
                  y1="0"
                  x2="28"
                  y2="0"
                  stroke="#d4b483"
                  strokeWidth={2}
                  animate={{ rotate: d }}
                  style={{ originX: '0px', originY: '0px' }}
                  transition={{ type: 'spring', stiffness: 80, damping: 16 }}
                />
                <circle r={2.5} fill="#eceae4" />
              </svg>
              <div className={`font-mono text-[10px] ${on ? 'text-gold' : 'text-mute'}`}>
                plane {i} · θ={th.toFixed(2)}
              </div>
            </button>
          )
        })}
      </div>
      <InspectPanel
        title={`RoPE plane ${sel} at position ${pos}`}
        value={`${deg.toFixed(1)}°`}
        facts={[
          { k: 'position', v: String(pos) },
          { k: 'plane', v: `dims (${sel * 2}, ${sel * 2 + 1})` },
          { k: 'θ', v: fmt(theta, 5) },
          { k: 'cos θ', v: fmt(Math.cos(theta), 5) },
          { k: 'sin θ', v: fmt(Math.sin(theta), 5) },
          { k: 'base', v: '10000' },
        ]}
        note="Even/odd pairs of Q and K are rotated by this angle. The dot product then depends on relative position pᵢ − pⱼ."
      />
    </div>
  )
}

