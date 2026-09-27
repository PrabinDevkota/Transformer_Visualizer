export type Vec = number[]
export type Mat = number[][]

export function zeros(n: number): Vec {
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = 0
  return out
}

export function zeros2(r: number, c: number): Mat {
  const out = new Array<Vec>(r)
  for (let i = 0; i < r; i++) out[i] = zeros(c)
  return out
}

export function add(a: Vec, b: Vec): Vec {
  const n = a.length
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = a[i]! + b[i]!
  return out
}

export function sub(a: Vec, b: Vec): Vec {
  const n = a.length
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = a[i]! - b[i]!
  return out
}

export function scale(a: Vec, s: number): Vec {
  const n = a.length
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = a[i]! * s
  return out
}

export function addMat(a: Mat, b: Mat): Mat {
  const n = a.length
  const out = new Array<Vec>(n)
  for (let i = 0; i < n; i++) out[i] = add(a[i]!, b[i]!)
  return out
}

export function dot(a: Vec, b: Vec): number {
  let s = 0
  const n = a.length
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!
  return s
}

export function norm(a: Vec): number {
  return Math.sqrt(dot(a, a))
}

export function mean(a: Vec): number {
  const n = a.length
  if (n === 0) return 0
  let s = 0
  for (let i = 0; i < n; i++) s += a[i]!
  return s / n
}

export function variance(a: Vec): number {
  const n = a.length
  if (n === 0) return 0
  const m = mean(a)
  let q = 0
  for (let i = 0; i < n; i++) {
    const d = a[i]! - m
    q += d * d
  }
  return q / n
}

export function matVec(m: Mat, v: Vec): Vec {
  const rows = m.length
  const out = new Array<number>(rows)
  for (let i = 0; i < rows; i++) out[i] = dot(m[i]!, v)
  return out
}

export function matMul(a: Mat, b: Mat): Mat {
  const bt = transpose(b)
  const rows = a.length
  const cols = bt.length
  const out = new Array<Vec>(rows)
  for (let i = 0; i < rows; i++) {
    const row = a[i]!
    const dest = new Array<number>(cols)
    for (let j = 0; j < cols; j++) dest[j] = dot(row, bt[j]!)
    out[i] = dest
  }
  return out
}

export function transpose(m: Mat): Mat {
  const rows = m.length
  if (rows === 0) return []
  const cols = m[0]!.length
  const out = new Array<Vec>(cols)
  for (let j = 0; j < cols; j++) {
    const col = new Array<number>(rows)
    for (let i = 0; i < rows; i++) col[i] = m[i]![j]!
    out[j] = col
  }
  return out
}

export function softmax(logits: Vec, temperature = 1): Vec {
  const n = logits.length
  const t = temperature > 1e-8 ? temperature : 1e-8
  const out = new Array<number>(n)
  let m = -Infinity
  for (let i = 0; i < n; i++) {
    const v = logits[i]! / t
    out[i] = v
    if (v > m) m = v
  }
  let z = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp(out[i]! - m)
    out[i] = e
    z += e
  }
  const inv = z > 0 ? 1 / z : 0
  for (let i = 0; i < n; i++) out[i] = out[i]! * inv
  return out
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x))
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))
}

export function relu(x: number): number {
  return x > 0 ? x : 0
}

export function rmsNorm(x: Vec, gamma: Vec, eps = 1e-6): Vec {
  const n = x.length
  let ms = 0
  for (let i = 0; i < n; i++) ms += x[i]! * x[i]!
  const inv = 1 / Math.sqrt(ms / Math.max(n, 1) + eps)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = x[i]! * inv * gamma[i]!
  return out
}

export function layerNorm(x: Vec, gamma: Vec, beta: Vec, eps = 1e-6): Vec {
  const n = x.length
  const m = mean(x)
  let q = 0
  for (let i = 0; i < n; i++) {
    const d = x[i]! - m
    q += d * d
  }
  const inv = 1 / Math.sqrt(q / Math.max(n, 1) + eps)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = (x[i]! - m) * inv * gamma[i]! + beta[i]!
  return out
}

export function clamp(n: number, a: number, b: number): number {
  return n < a ? a : n > b ? b : n
}

export function argmax(a: Vec): number {
  let i = 0
  const n = a.length
  for (let k = 1; k < n; k++) if (a[k]! > a[i]!) i = k
  return i
}

export function formatNum(n: number, digits = 2): string {
  const v = Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(digits)
  return n >= 0 ? ` ${v}` : v
}

export function entropy(p: Vec): number {
  let s = 0
  const n = p.length
  for (let i = 0; i < n; i++) {
    const v = p[i]!
    if (v > 0) s += v * Math.log2(v)
  }
  return -s
}

export function topKIndices(values: Vec, k: number): number[] {
  const n = values.length
  const kk = Math.max(1, Math.min(k, n))
  const idx = new Array<number>(n)
  for (let i = 0; i < n; i++) idx[i] = i
  idx.sort((a, b) => values[b]! - values[a]!)
  if (kk === n) return idx
  idx.length = kk
  return idx
}
