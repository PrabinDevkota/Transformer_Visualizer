import type { MergeStep, Token, TokenizeTrace } from './types'

const MERGES = [
  'th', 'he', 'in', 'er', 'an', 'on', 'at', 'en', 'ed', 'to',
  'ing', 'ion', 'the', 'and', 'att', 'ent', 'for', 'orm',
  'trans', 'form', 'former', 'atten', 'attention',
  'is', 'all', 'you', 'need', 'cat', 'sat', 'mat',
  'hel', 'lo', 'hello', 'wor', 'ld', 'world',
  'mod', 'el', 'model', 'lang', 'uage', 'language',
  'dec', 'oder', 'decoder', 'enc', 'oder', 'encoder',
  'soft', 'max', 'softmax', 'query', 'key', 'value',
  'map', 'token', 'vector', 'layer',
]

function applyMerges(pieces: string[]): { tokens: string[]; steps: { tokens: string[]; merge: MergeStep | null }[] } {
  const steps: { tokens: string[]; merge: MergeStep | null }[] = [
    { tokens: [...pieces], merge: null },
  ]
  let tokens = [...pieces]
  for (const merge of MERGES) {
    let changed = true
    while (changed) {
      changed = false
      const next: string[] = []
      let i = 0
      while (i < tokens.length) {
        const a = tokens[i]!
        const b = tokens[i + 1]
        if (b !== undefined && a !== ' ' && b !== ' ' && (a + b).toLowerCase() === merge) {
          next.push(a + b)
          steps.push({
            tokens: [...next, ...tokens.slice(i + 2)],
            merge: { left: a, right: b, result: a + b, index: next.length - 1 },
          })
          i += 2
          changed = true
        } else {
          next.push(a)
          i += 1
        }
      }
      tokens = next
    }
  }
  return { tokens, steps }
}

const vocabIndex = new Map<string, number>()
function idFor(token: string): number {
  const existing = vocabIndex.get(token)
  if (existing !== undefined) return existing
  const id = vocabIndex.size + 16
  vocabIndex.set(token, id)
  return id
}

export function tokenize(text: string, showBos: boolean): TokenizeTrace {
  const raw = text.trim() || 'hello'
  const stream: string[] = []
  const indexOfChar: number[] = []
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]!
    if (ch === ' ') {
      if (stream.at(-1) !== ' ') {
        stream.push(' ')
        indexOfChar.push(i)
      }
    } else {
      stream.push(ch)
      indexOfChar.push(i)
    }
  }

  const { tokens: pieces, steps } = applyMerges(stream)

  const mapped: Token[] = []
  if (showBos) {
    mapped.push({ id: 1, text: '<bos>', start: 0, end: 0, special: true })
  }

  let cursor = 0
  for (const piece of pieces) {
    if (piece === ' ') {
      cursor += 1
      continue
    }
    const start = indexOfChar[cursor] ?? 0
    mapped.push({
      id: idFor(piece.toLowerCase()),
      text: piece,
      start,
      end: start + piece.length,
      special: false,
    })
    cursor += piece.length
  }

  mapped.push({ id: 2, text: '<eos>', start: raw.length, end: raw.length, special: true })

  return {
    raw,
    chars: stream.filter((c) => c !== ' '),
    steps,
    tokens: mapped,
  }
}
