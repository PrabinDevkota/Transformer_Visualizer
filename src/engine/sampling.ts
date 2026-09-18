import { argmax, softmax, topKIndices } from './math'
import type { SampleCandidate, SampleKind } from './types'

export function applySampling(
  logits: number[],
  vocab: string[],
  kind: SampleKind,
  temperature: number,
  topK: number,
  topP: number,
  minP: number,
  rand: () => number,
): { candidates: SampleCandidate[]; picked: string } {
  const n = logits.length
  const keep = Array.from({ length: n }, () => true)

  if (kind === 'greedy') {
    const i = argmax(logits)
    const candidates = vocab.map((token, idx) => ({
      token,
      logit: logits[idx]!,
      filtered: idx !== i,
      prob: idx === i ? 1 : 0,
    }))
    return { candidates, picked: vocab[i]! }
  }

  if (kind === 'topk' || kind === 'temperature') {
    if (kind === 'topk') {
      const keepIdx = new Set(topKIndices(logits, topK))
      for (let i = 0; i < n; i++) keep[i] = keepIdx.has(i)
    }
  }

  if (kind === 'topp') {
    const order = logits.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0])
    const prelim = softmax(order.map(([v]) => v), temperature)
    let cdf = 0
    const keepIdx = new Set<number>()
    for (let i = 0; i < order.length; i++) {
      keepIdx.add(order[i]![1])
      cdf += prelim[i]!
      if (cdf >= topP && keepIdx.size >= 1) break
    }
    for (let i = 0; i < n; i++) keep[i] = keepIdx.has(i)
  }

  if (kind === 'minp') {
    const p = softmax(logits, temperature)
    const pMax = Math.max(...p)
    const thresh = minP * pMax
    for (let i = 0; i < n; i++) keep[i] = p[i]! >= thresh
  }

  if (kind === 'beam') {
    const keepIdx = new Set(topKIndices(logits, Math.min(4, n)))
    for (let i = 0; i < n; i++) keep[i] = keepIdx.has(i)
  }

  const masked = logits.map((v, i) => (keep[i] ? v : -1e9))
  const probs = softmax(masked, temperature)
  const r = rand()
  let cdf = 0
  let pick = 0
  for (let i = 0; i < n; i++) {
    cdf += probs[i]!
    if (r <= cdf) {
      pick = i
      break
    }
  }

  const candidates = vocab.map((token, idx) => ({
    token,
    logit: logits[idx]!,
    filtered: !keep[idx],
    prob: probs[idx]!,
  }))

  return { candidates, picked: vocab[pick]! }
}

export function nucleusCutoff(probsSortedDesc: number[], topP: number): number {
  let cdf = 0
  for (let i = 0; i < probsSortedDesc.length; i++) {
    cdf += probsSortedDesc[i]!
    if (cdf >= topP) return i + 1
  }
  return probsSortedDesc.length
}
