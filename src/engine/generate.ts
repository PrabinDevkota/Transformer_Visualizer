import { runModel } from './transformer'
import type { ModelConfig, SampleCandidate } from './types'

export function appendToken(text: string, tok: string): string {
  if (!tok || tok.startsWith('<')) return text
  if (/^[.,!?;:'"]/.test(tok)) return text + tok
  return text.trimEnd() + ' ' + tok
}

export interface GenStep {
  prompt: string
  picked: string
  candidates: SampleCandidate[]
}

export function generateSteps(seed: string, target: string, cfg: ModelConfig, n = 8): GenStep[] {
  const out: GenStep[] = []
  let prompt = seed.trim() || 'The'
  for (let i = 0; i < n; i++) {
    const trace = runModel(prompt, target, { ...cfg, seed: cfg.seed + i * 17 })
    out.push({
      prompt,
      picked: trace.sampled.picked,
      candidates: trace.sampled.candidates,
    })
    if (trace.sampled.picked === '.' || trace.sampled.picked === '!') break
    prompt = appendToken(prompt, trace.sampled.picked)
  }
  return out
}
