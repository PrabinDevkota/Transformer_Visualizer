import { hashString, mulberry32 } from './rng'
import { applySampling } from './sampling'
import { tokenize } from './tokenizer'
import { demoVocab, toyLogits } from './toyLm'
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

/** Toy next-token loop. Does not rerun the transformer — sampling reads the n-gram table. */
export function generateSteps(seed: string, target: string, cfg: ModelConfig, n = 8): GenStep[] {
  const out: GenStep[] = []
  let prompt = seed.trim() || 'The'
  for (let i = 0; i < n; i++) {
    const tok = tokenize(prompt, cfg.showBos)
    const vocab = demoVocab(tok.tokens)
    const logits = toyLogits(tok.tokens, vocab)
    const sampleRand = mulberry32(cfg.seed + i * 17 + hashString(prompt + '|' + target))
    const sampled = applySampling(
      logits,
      vocab,
      cfg.sampling,
      cfg.temperature,
      cfg.topK,
      cfg.topP,
      cfg.minP,
      sampleRand,
    )
    out.push({
      prompt,
      picked: sampled.picked,
      candidates: sampled.candidates,
    })
    if (sampled.picked === '.' || sampled.picked === '!') break
    prompt = appendToken(prompt, sampled.picked)
  }
  return out
}
