import { entropy } from './math'

export type AttnPattern = 'diagonal' | 'previous' | 'bos' | 'uniform' | 'mixed'

function rowArgmax(row: number[]): number {
  let i = 0
  const n = row.length
  for (let k = 1; k < n; k++) if (row[k]! > row[i]!) i = k
  return i
}

export function classifyAttention(attn: number[][]): AttnPattern {
  const n = attn.length
  if (n === 0) return 'mixed'
  let diag = 0
  let prev = 0
  let bos = 0
  let entSum = 0
  for (let i = 0; i < n; i++) {
    const row = attn[i]!
    const m = rowArgmax(row)
    if (m === i) diag++
    if (i > 0 && m === i - 1) prev++
    if (m === 0) bos++
    entSum += entropy(row)
  }
  const meanH = entSum / n
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
