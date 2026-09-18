export function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export function rgbToHex(r: number, g: number, b: number) {
  return `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`
}

function mix(a: string, b: string, t: number) {
  const A = hexToRgb(a)
  const B = hexToRgb(b)
  return rgbToHex(lerp(A[0], B[0], t), lerp(A[1], B[1], t), lerp(A[2], B[2], t))
}

export function diverging(v: number, maxAbs: number) {
  const t = maxAbs === 0 ? 0 : Math.max(-1, Math.min(1, v / maxAbs))
  if (t < 0) return mix('#c97a63', '#1a1b20', -t)
  return mix('#1a1b20', '#d4b483', t)
}

export function sequential(v: number, max: number) {
  const t = max === 0 ? 0 : Math.max(0, Math.min(1, v / max))
  if (t < 0.5) return mix('#16171c', '#7aa8a0', t * 2)
  return mix('#7aa8a0', '#e8d7b0', (t - 0.5) * 2)
}

export function maxAbs(m: number[][]) {
  let x = 0
  for (const row of m) for (const v of row) x = Math.max(x, Math.abs(v))
  return x || 1
}

export function maxVal(m: number[][]) {
  let x = 0
  for (const row of m) for (const v of row) x = Math.max(x, v)
  return x || 1
}

export const TOKEN_COLORS = [
  '#d4b483',
  '#7aa8a0',
  '#c97a63',
  '#7a93b8',
  '#b8a07a',
  '#8a9a7a',
  '#c4a0a8',
  '#9aa0b8',
]
