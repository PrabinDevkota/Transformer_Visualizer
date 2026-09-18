import { motion } from 'motion/react'
import { TOKEN_COLORS } from './color'
import type { Token } from '../engine/types'
import clsx from 'clsx'

export function TokenRow({
  tokens,
  active,
  onPick,
}: {
  tokens: Token[]
  active?: number
  onPick?: (i: number) => void
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {tokens.map((t, i) => (
        <motion.button
          key={`${t.id}-${i}`}
          type="button"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.04 }}
          onClick={() => onPick?.(i)}
          className={clsx(
            'rounded-md border px-2 py-1 font-mono text-[12px] transition-colors',
            t.special && 'text-faint border-line',
            !t.special && 'text-ink',
            active === i ? 'border-gold bg-gold/10' : 'border-line bg-elev hover:border-gold/40',
          )}
          style={!t.special ? { boxShadow: `inset 3px 0 0 ${TOKEN_COLORS[i % TOKEN_COLORS.length]}` } : undefined}
        >
          {t.text}
        </motion.button>
      ))}
    </div>
  )
}

export function TokenSource({ text, tokens, active }: { text: string; tokens: Token[]; active?: number }) {
  return (
    <p className="font-mono text-[13px] leading-7 text-ink">
      {tokens
        .filter((t) => !t.special)
        .map((t, i) => {
          const real = tokens.indexOf(t)
          const on = active === real
          return (
            <span
              key={`${t.start}-${i}`}
              className={clsx('rounded-sm px-0.5', on && 'bg-gold/20 text-gold')}
            >
              {text.slice(t.start, t.end)}
            </span>
          )
        })}
    </p>
  )
}
