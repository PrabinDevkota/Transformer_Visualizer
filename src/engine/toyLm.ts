import type { Token } from './types'

const UNIGRAM: Record<string, number> = {
  the: 2.4, a: 2.0, of: 1.6, to: 1.5, and: 1.4, is: 1.3, in: 1.2,
  attention: 1.8, transformer: 1.7, model: 1.4, token: 1.3, layer: 1.2,
  cat: 1.1, sat: 0.9, mat: 0.8, hello: 1.0, world: 0.9,
}

const BIGRAM: Record<string, Record<string, number>> = {
  the: { cat: 2.8, dog: 2.2, transformer: 2.4, model: 2.0, attention: 1.8, next: 1.4, mat: 1.6 },
  cat: { sat: 3.2, is: 1.6, jumped: 1.8, sleeps: 1.4 },
  sat: { on: 3.4, down: 1.5, quietly: 1.1 },
  on: { the: 3.1, a: 1.8, top: 1.2 },
  mat: { and: 1.6, '.': 1.4, today: 1.0 },
  attention: { is: 3.0, heads: 2.2, scores: 2.0, weights: 1.7 },
  is: { all: 2.8, a: 1.8, the: 1.6, used: 1.4 },
  all: { you: 3.1, the: 1.4, about: 1.3 },
  you: { need: 3.2, want: 1.5, can: 1.3 },
  need: { is: 1.2, a: 1.4, '.': 1.6, to: 1.5 },
  hello: { world: 3.0, there: 2.2, ',': 1.6 },
  world: { '!': 2.4, model: 1.3, is: 1.2 },
  transformer: { layer: 2.4, model: 2.2, block: 1.8, decoder: 1.7 },
  layer: { norm: 2.2, uses: 1.6, stacks: 1.4, residual: 1.8 },
  decoder: { only: 2.6, block: 2.0, attends: 1.7 },
  encoder: { decoder: 2.4, only: 2.2, hidden: 1.6 },
  softmax: { turns: 2.2, normalizes: 2.4, over: 1.6 },
  residual: { stream: 2.8, connection: 2.2, add: 1.8 },
}

const FALLBACK = [
  'the', 'a', 'to', 'of', 'and', 'is', 'in', 'that',
  'attention', 'token', 'layer', 'head', 'next',
  '.', ',', '!',
]

export function toyLogits(tokens: Token[], vocab: string[]): number[] {
  const last = [...tokens].reverse().find((t) => !t.special)?.text.toLowerCase() ?? ''
  const table = BIGRAM[last] ?? {}
  return vocab.map((w) => {
    const key = w.toLowerCase()
    const bi = table[key] ?? 0
    const uni = UNIGRAM[key] ?? 0.15
    const fallback = FALLBACK.includes(key) ? 0.25 : 0
    return bi + uni + fallback
  })
}

export function demoVocab(tokens: Token[]): string[] {
  const last = [...tokens].reverse().find((t) => !t.special)?.text.toLowerCase() ?? ''
  const fromBigram = Object.keys(BIGRAM[last] ?? {})
  const base = [
    ...fromBigram,
    'the', 'a', 'cat', 'sat', 'on', 'mat',
    'attention', 'is', 'all', 'you', 'need',
    'transformer', 'layer', 'model', 'token', 'head',
    '.', '!',
  ]
  const seen = new Set<string>()
  const out: string[] = []
  for (const w of base) {
    if (!seen.has(w)) {
      seen.add(w)
      out.push(w)
    }
  }
  return out.slice(0, 16)
}
