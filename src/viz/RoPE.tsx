import { motion } from 'motion/react'

export function RoPEPlanes({ pos, dim = 8 }: { pos: number; dim?: number }) {
  const planes = Math.floor(dim / 2)
  return (
    <div className="flex flex-wrap gap-4">
      {Array.from({ length: planes }, (_, i) => {
        const theta = pos / 10000 ** ((2 * i) / dim)
        const deg = (theta * 180) / Math.PI
        return (
          <div key={i} className="flex flex-col items-center gap-2">
            <svg width={88} height={88} viewBox="-44 -44 88 88">
              <circle r={36} fill="none" stroke="#26272e" />
              <line x1="-36" y1="0" x2="36" y2="0" stroke="#26272e" />
              <line x1="0" y1="-36" x2="0" y2="36" stroke="#26272e" />
              <motion.line
                x1="0"
                y1="0"
                x2="28"
                y2="0"
                stroke="#d4b483"
                strokeWidth={2}
                animate={{ rotate: deg }}
                style={{ originX: '0px', originY: '0px' }}
                transition={{ type: 'spring', stiffness: 80, damping: 16 }}
              />
              <circle r={2.5} fill="#eceae4" />
            </svg>
            <div className="font-mono text-[10px] text-mute">
              plane {i} · θ={theta.toFixed(2)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function Formula({ children }: { children: string }) {
  return (
    <div className="rounded-md border border-line bg-soft px-3 py-2 font-mono text-[12px] text-gold/90 tracking-wide">
      {children}
    </div>
  )
}
