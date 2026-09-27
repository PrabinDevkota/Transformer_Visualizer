import { argmax, softmax, topKIndices } from './math'
import type { SampleCandidate, SampleKind } from './types'

function markKept(n: number, keepIdx: Iterable<number>): boolean[] {
  const keep = new Array<boolean>(n)
  for (let i = 0; i < n; i++) keep[i] = false
  for (const i of keepIdx) keep[i] = true
  return keep
}

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
  let keep: boolean[]

  if (kind === 'greedy') {
    const i = argmax(logits)
    const candidates = new Array<SampleCandidate>(n)
    for (let idx = 0; idx < n; idx++) {
      candidates[idx] = {
        token: vocab[idx]!,
        logit: logits[idx]!,
        filtered: idx !== i,
        prob: idx === i ? 1 : 0,
      }
    }
    return { candidates, picked: vocab[i]! }
  }

  if (kind === 'topk') {
    keep = markKept(n, topKIndices(logits, topK))
  } else if (kind === 'beam') {
    keep = markKept(n, topKIndices(logits, Math.min(4, n)))
  } else if (kind === 'topp') {
    const order = topKIndices(logits, n)
    const prelim = softmax(
      order.map((i) => logits[i]!),
      temperature,
    )
    let cdf = 0
    const keepIdx: number[] = []
    for (let i = 0; i < order.length; i++) {
      keepIdx.push(order[i]!)
      cdf += prelim[i]!
      if (cdf >= topP && keepIdx.length >= 1) break
    }
    keep = markKept(n, keepIdx)
  } else if (kind === 'minp') {
    const p = softmax(logits, temperature)
    let pMax = 0
    for (let i = 0; i < n; i++) if (p[i]! > pMax) pMax = p[i]!
    const thresh = minP * pMax
    keep = new Array<boolean>(n)
    for (let i = 0; i < n; i++) keep[i] = p[i]! >= thresh
  } else {
    keep = new Array<boolean>(n)
    for (let i = 0; i < n; i++) keep[i] = true
  }

  const masked = new Array<number>(n)
  for (let i = 0; i < n; i++) masked[i] = keep[i] ? logits[i]! : -1e9
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

  const candidates = new Array<SampleCandidate>(n)
  for (let idx = 0; idx < n; idx++) {
    candidates[idx] = {
      token: vocab[idx]!,
      logit: logits[idx]!,
      filtered: !keep[idx],
      prob: probs[idx]!,
    }
  }

  return { candidates, picked: vocab[pick]! }
}

export function nucleusCutoff(probsSortedDesc: number[], topP: number): number {
  let cdf = 0
  const n = probsSortedDesc.length
  for (let i = 0; i < n; i++) {
    cdf += probsSortedDesc[i]!
    if (cdf >= topP) return i + 1
  }
  return n
}
