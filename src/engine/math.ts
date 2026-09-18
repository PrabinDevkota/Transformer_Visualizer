export type Vec = number[]
export type Mat = number[][]

export function zeros(n: number): Vec {
  return Array.from({ length: n }, () => 0)
}

export function zeros2(r: number, c: number): Mat {
  return Array.from({ length: r }, () => zeros(c))
}

export function add(a: Vec, b: Vec): Vec {
  return a.map((v, i) => v + b[i]!)
}

export function sub(a: Vec, b: Vec): Vec {
  return a.map((v, i) => v - b[i]!)
}

export function scale(a: Vec, s: number): Vec {
  return a.map((v) => v * s)
}

export function addMat(a: Mat, b: Mat): Mat {
  return a.map((row, i) => add(row, b[i]!))
}

export function dot(a: Vec, b: Vec): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!
  return s
}

export function norm(a: Vec): number {
  return Math.sqrt(dot(a, a))
}

export function mean(a: Vec): number {
  return a.reduce((s, v) => s + v, 0) / Math.max(a.length, 1)
}

export function variance(a: Vec): number {
  const m = mean(a)
  return a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(a.length, 1)
}

export function matVec(m: Mat, v: Vec): Vec {
  return m.map((row) => dot(row, v))
}

export function matMul(a: Mat, b: Mat): Mat {
  const bt = transpose(b)
  return a.map((row) => bt.map((col) => dot(row, col)))
}

export function transpose(m: Mat): Mat {
  if (m.length === 0) return []
  return m[0]!.map((_, j) => m.map((row) => row[j]!))
}

export function softmax(logits: Vec, temperature = 1): Vec {
  const t = Math.max(temperature, 1e-8)
  const scaled = logits.map((v) => v / t)
  const m = Math.max(...scaled)
  const exps = scaled.map((v) => Math.exp(v - m))
  const z = exps.reduce((s, v) => s + v, 0)
  return exps.map((v) => v / z)
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x))
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
}

export function relu(x: number): number {
  return Math.max(0, x)
}

export function rmsNorm(x: Vec, gamma: Vec, eps = 1e-6): Vec {
  const ms = Math.sqrt(mean(x.map((v) => v * v)) + eps)
  return x.map((v, i) => (v / ms) * gamma[i]!)
}

export function layerNorm(x: Vec, gamma: Vec, beta: Vec, eps = 1e-6): Vec {
  const m = mean(x)
  const v = variance(x)
  const inv = 1 / Math.sqrt(v + eps)
  return x.map((val, i) => ((val - m) * inv) * gamma[i]! + beta[i]!)
}

export function clamp(n: number, a: number, b: number): number {
  return Math.min(b, Math.max(a, n))
}

export function argmax(a: Vec): number {
  let i = 0
  for (let k = 1; k < a.length; k++) if (a[k]! > a[i]!) i = k
  return i
}

export function formatNum(n: number, digits = 2): string {
  const v = Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(digits)
  return n >= 0 ? ` ${v}` : v
}

export function entropy(p: Vec): number {
  return -p.reduce((s, v) => (v > 0 ? s + v * Math.log2(v) : s), 0)
}

export function topKIndices(values: Vec, k: number): number[] {
  return values
    .map((v, i) => [v, i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, Math.max(1, Math.min(k, values.length)))
    .map(([, i]) => i)
}
