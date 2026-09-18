import { entropy } from './math'

export type AttnPattern = 'diagonal' | 'previous' | 'bos' | 'uniform' | 'mixed'

export function classifyAttention(attn: number[][]): AttnPattern {
  if (!attn.length) return 'mixed'
  const n = attn.length
  let diag = 0
  let prev = 0
  let bos = 0
  const ents: number[] = []
  for (let i = 0; i < n; i++) {
    const row = attn[i]!
    const m = row.indexOf(Math.max(...row))
    if (m === i) diag++
    if (m === Math.max(0, i - 1) && i > 0) prev++
    if (m === 0) bos++
    ents.push(entropy(row))
  }
  const meanH = ents.reduce((s, v) => s + v, 0) / n
  if (meanH > Math.log2(Math.max(n, 2)) * 0.72) return 'uniform'
  if (diag / n >= 0.55) return 'diagonal'
  if (prev / Math.max(n - 1, 1) >= 0.55) return 'previous'
  if (bos / n >= 0.55) return 'bos'
  return 'mixed'
}

export function patternLabel(p: AttnPattern): string {
  if (p === 'diagonal') return 'self'
  if (p === 'previous') return 'prev-token'
  if (p === 'bos') return 'attend BOS'
  if (p === 'uniform') return 'diffuse'
  return 'mixed'
}

export function patternHint(p: AttnPattern): string {
  if (p === 'diagonal') return 'Most mass sits on the token itself — a copy / residual-like head.'
  if (p === 'previous') return 'Looks at the token just before — useful for bigrams and local syntax.'
  if (p === 'bos') return 'Piles onto the first token. Common as an attention sink in long context.'
  if (p === 'uniform') return 'Averages the context. Semantic pooling rather than a sharp pointer.'
  return 'No single textbook pattern. Real models mix several behaviors in one head.'
}
