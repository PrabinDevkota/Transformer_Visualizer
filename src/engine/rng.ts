export function mulberry32(seed: number): () => number {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let x = t
    x = Math.imul(x ^ (x >>> 15), x | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

export function hashString(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

export function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12)
  const v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

export function randomMatrix(rows: number, cols: number, rand: () => number, scale = 0.35): number[][] {
  const s = scale / Math.sqrt(cols)
  const m = new Array<number[]>(rows)
  for (let i = 0; i < rows; i++) {
    const row = new Array<number>(cols)
    for (let j = 0; j < cols; j++) row[j] = gaussian(rand) * s
    m[i] = row
  }
  return m
}

export function randomVec(n: number, rand: () => number, scale = 0.2): number[] {
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = gaussian(rand) * scale
  return out
}
