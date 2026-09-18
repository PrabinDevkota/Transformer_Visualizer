export type Architecture = 'decoder' | 'encoder' | 'encdec'
export type PositionalEncoding = 'rope' | 'sinusoidal' | 'learned' | 'alibi' | 'none'
export type AttentionKind = 'mha' | 'gqa' | 'mqa' | 'mla'
export type NormKind = 'rmsnorm' | 'layernorm'
export type FfnKind = 'swiglu' | 'gelu' | 'relu'
export type SampleKind = 'greedy' | 'temperature' | 'topk' | 'topp' | 'minp' | 'beam'

export interface ModelConfig {
  architecture: Architecture
  positional: PositionalEncoding
  attention: AttentionKind
  norm: NormKind
  ffn: FfnKind
  sampling: SampleKind
  temperature: number
  topK: number
  topP: number
  minP: number
  nLayers: number
  nHeads: number
  nKvHeads: number
  dModel: number
  seed: number
  showBos: boolean
}

export interface Token {
  id: number
  text: string
  start: number
  end: number
  special: boolean
}

export interface MergeStep {
  left: string
  right: string
  result: string
  index: number
}

export interface TokenizeTrace {
  raw: string
  chars: string[]
  steps: { tokens: string[]; merge: MergeStep | null }[]
  tokens: Token[]
}

export interface HeadTrace {
  q: number[][]
  k: number[][]
  v: number[][]
  scoresRaw: number[][]
  scoresMasked: number[][]
  attn: number[][]
  out: number[][]
}

export interface CrossTrace {
  heads: HeadTrace[]
  out: number[][]
}

export interface LayerTrace {
  input: number[][]
  norm1: number[][]
  heads: HeadTrace[]
  attnConcat: number[][]
  attnProj: number[][]
  residual1: number[][]
  norm2: number[][]
  ffnGate?: number[][]
  ffnUp?: number[][]
  ffnHidden: number[][]
  ffnOut: number[][]
  residual2: number[][]
  cross?: CrossTrace
  moe?: {
    router: number[][]
    chosen: { token: number; experts: number[] }[]
  }
}

export interface SampleCandidate {
  token: string
  logit: number
  filtered: boolean
  prob: number
}

export interface Trace {
  config: ModelConfig
  sourceText: string
  targetText: string
  tokenize: TokenizeTrace
  targetTokenize?: TokenizeTrace
  embeddings: number[][]
  posSignal: number[][]
  positioned: number[][]
  layers: LayerTrace[]
  encoderLayers?: LayerTrace[]
  finalNorm: number[][]
  pooled?: number[]
  vocab: string[]
  rawLogits: number[]
  sampled: {
    candidates: SampleCandidate[]
    picked: string
    method: SampleKind
  }
  kv: {
    prefillK: number[][][]
    prefillV: number[][][]
    decodeQ: number[][]
    decodeScores: number[][]
  }
}

export const DEFAULT_CONFIG: ModelConfig = {
  architecture: 'decoder',
  positional: 'rope',
  attention: 'gqa',
  norm: 'rmsnorm',
  ffn: 'swiglu',
  sampling: 'topp',
  temperature: 0.8,
  topK: 8,
  topP: 0.9,
  minP: 0.05,
  nLayers: 2,
  nHeads: 4,
  nKvHeads: 2,
  dModel: 16,
  seed: 7,
  showBos: true,
}
