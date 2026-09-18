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

function ramp(stops: [number, string][], t: number) {
  const u = Math.max(0, Math.min(1, t))
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i - 1]!
    const [t2, c2] = stops[i]!
    if (u <= t2) return mix(c1, c2, (u - t1) / Math.max(t2 - t1, 1e-9))
  }
  return stops[stops.length - 1]![1]
}

/**
 * Cool–warm diverging (colorblind-safer than red–green).
 * Blue = negative, near-black = ~0, amber = positive.
 * Brightness tracks |value|, hue tracks sign.
 */
export const DIVERGING_STOPS: [number, string][] = [
  [0, '#2f6bff'],
  [0.16, '#4d7ad6'],
  [0.34, '#3e5478'],
  [0.5, '#1a1b22'],
  [0.66, '#8a6424'],
  [0.84, '#e0a62e'],
  [1, '#ffe08a'],
]

/**
 * Sequential mass: dark = ignore, gold = focus.
 * Distinct from signed (no blue), so attention heatmaps cannot be
 * mistaken for embeddings.
 */
export const SEQUENTIAL_STOPS: [number, string][] = [
  [0, '#101118'],
  [0.16, '#2c1f78'],
  [0.36, '#1a6b8c'],
  [0.58, '#1aa07a'],
  [0.8, '#e0b03a'],
  [1, '#fff4c2'],
]

export const MASK_COLOR = '#0c0d12'

export function diverging(v: number, maxAbs: number) {
  const t = 0.5 + 0.5 * (maxAbs === 0 ? 0 : Math.max(-1, Math.min(1, v / maxAbs)))
  return ramp(DIVERGING_STOPS, t)
}

export function sequential(v: number, max: number) {
  const t = max === 0 ? 0 : Math.max(0, Math.min(1, v / max))
  return ramp(SEQUENTIAL_STOPS, t)
}

export function maxAbs(m: number[][]) {
  let x = 0
  for (const row of m) for (const v of row) if (Number.isFinite(v)) x = Math.max(x, Math.abs(v))
  return x || 1
}

export function maxVal(m: number[][]) {
  let x = 0
  for (const row of m) for (const v of row) if (Number.isFinite(v)) x = Math.max(x, v)
  return x || 1
}

/** Okabe–Ito categorical palette, punched up for a dark background. */
export const TOKEN_COLORS = [
  '#f0a202',
  '#56b4e9',
  '#3dcc9a',
  '#f0e442',
  '#5b9bd5',
  '#e07a3d',
  '#cc79a7',
  '#9cd1c7',
]

export function tokenColor(i: number) {
  return TOKEN_COLORS[i % TOKEN_COLORS.length]!
}

export function colorBarCss(stops: [number, string][]) {
  return `linear-gradient(90deg, ${stops.map(([t, c]) => `${c} ${t * 100}%`).join(', ')})`
}

export function ink(fill: string) {
  const [r, g, b] = hexToRgb(fill)
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#080809' : '#f3f1ea'
}
