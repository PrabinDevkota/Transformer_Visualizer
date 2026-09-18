export type StageId =
  | 'tokenize'
  | 'embed'
  | 'position'
  | 'norm'
  | 'qkv'
  | 'scores'
  | 'mask'
  | 'softmax'
  | 'attend'
  | 'heads'
  | 'residual'
  | 'ffn'
  | 'stack'
  | 'unembed'
  | 'sampling'
  | 'kvcache'
  | 'arch'
  | 'gqa'
  | 'moe'
  | 'speculative'

export interface StageMeta {
  id: StageId
  n: string
  title: string
  group: string
  blurb: string
  formula?: string
}

export const STAGES: StageMeta[] = [
  {
    id: 'tokenize',
    n: '01',
    title: 'Tokenize',
    group: 'Input',
    blurb: 'Text is split into tokens — subword pieces a model can look up. Modern LLMs use BPE or SentencePiece, not whole words.',
    formula: 'text → [t₀, t₁, …, tₙ] ⊂ V',
  },
  {
    id: 'embed',
    n: '02',
    title: 'Embed',
    group: 'Input',
    blurb: 'Each token id indexes a learned table. The result is a vector in residual-stream space — the model’s working memory.',
    formula: 'xᵢ = E[tᵢ]  ∈ ℝᵈ',
  },
  {
    id: 'position',
    n: '03',
    title: 'Position',
    group: 'Input',
    blurb: 'Self-attention has no inherent order. RoPE rotates Q/K by position so relative distance is baked into the dot product. Alternatives: sinusoidal, learned, ALiBi, NoPE.',
    formula: 'RoPE: (q′, k′) = R_θ,p q,  R_θ,p k',
  },
  {
    id: 'norm',
    n: '04',
    title: 'Norm',
    group: 'Block',
    blurb: 'Modern stacks are pre-norm: normalize, then attend. RMSNorm skips mean-centering — scale only — which is cheaper and stable at LLM scale.',
    formula: 'RMS(x) = x / √(mean(x²)+ε) ⊙ γ',
  },
  {
    id: 'qkv',
    n: '05',
    title: 'Q K V',
    group: 'Block',
    blurb: 'Three linear maps turn the same residual stream into queries (“what am I looking for”), keys (“what I contain”), and values (“what I will pass on”).',
    formula: 'Q = XW_Q    K = XW_K    V = XW_V',
  },
  {
    id: 'scores',
    n: '06',
    title: 'Scores',
    group: 'Block',
    blurb: 'Compatibility is a scaled dot product. Dividing by √d_h keeps the softmax in a healthy range as head dimension grows.',
    formula: 'S = QKᵀ / √dₕ',
  },
  {
    id: 'mask',
    n: '07',
    title: 'Mask',
    group: 'Block',
    blurb: 'Decoder-only models hide the future (causal). Encoders see every token. Seq2seq adds cross-attention: decoder queries, encoder keys/values.',
    formula: 'Sᵢⱼ ← −∞   if j is illegal',
  },
  {
    id: 'softmax',
    n: '08',
    title: 'Softmax',
    group: 'Block',
    blurb: 'Softmax turns scores into a probability distribution over tokens to attend to. Temperature later reuses this same idea on the vocabulary.',
    formula: 'αᵢⱼ = exp(Sᵢⱼ) / Σₖ exp(Sᵢₖ)',
  },
  {
    id: 'attend',
    n: '09',
    title: 'Attend',
    group: 'Block',
    blurb: 'Each query mixes value vectors using those probabilities. Information moves between positions — this is the residual stream’s communication channel.',
    formula: 'zᵢ = Σⱼ αᵢⱼ vⱼ',
  },
  {
    id: 'heads',
    n: '10',
    title: 'Heads',
    group: 'Block',
    blurb: 'Several attentions run in parallel on subspaces, then concatenate. GQA/MQA/MLA exist to shrink the KV cache without dropping query diversity.',
    formula: 'MultiHead = Concat(h₁…hₕ) W_O',
  },
  {
    id: 'residual',
    n: '11',
    title: 'Residual',
    group: 'Block',
    blurb: 'Add the block’s write-back to the stream. Deep transformers work because every layer can leave the previous representation intact.',
    formula: 'x ← x + Attn(Norm(x))',
  },
  {
    id: 'ffn',
    n: '12',
    title: 'FFN',
    group: 'Block',
    blurb: 'A position-wise MLP. SwiGLU gates one projection with SiLU of another — the 2023–2025 default over GELU. This is where most parameters live.',
    formula: 'SwiGLU(x) = (SiLU(xW_g) ⊙ xW_u) W_d',
  },
  {
    id: 'stack',
    n: '13',
    title: 'Stack',
    group: 'Block',
    blurb: 'The same block repeats L times. Early layers copy syntax; later layers do more abstract routing. The residual stream is the shared bus.',
    formula: 'X^{(ℓ+1)} = Block^{(ℓ)}(X^{(ℓ)})',
  },
  {
    id: 'unembed',
    n: '14',
    title: 'Unembed',
    group: 'Output',
    blurb: 'Final RMSNorm, then a linear map to |V| logits. Many models tie this matrix to the embedding table (weight tying). Encoders stop at a pooled vector instead.',
    formula: 'ℓ = W_U · RMS(h_last)',
  },
  {
    id: 'sampling',
    n: '15',
    title: 'Sampling',
    group: 'Output',
    blurb: 'Logits are not yet a choice. Temperature, top-k, top-p, min-p, greedy, and beam search reshape the distribution before a token is drawn.',
    formula: 'p = softmax(ℓ / T)  then filter',
  },
  {
    id: 'kvcache',
    n: '16',
    title: 'KV cache',
    group: 'Output',
    blurb: 'Prefill writes K,V for the prompt. Decode computes one new Q and reuses cached K,V. This is why long context is memory-bound, not compute-bound.',
    formula: 'K ← [K; k_t]   V ← [V; v_t]',
  },
  {
    id: 'arch',
    n: '17',
    title: 'Architectures',
    group: 'Modern',
    blurb: 'Encoder-only (BERT) is bidirectional. Decoder-only (GPT, Llama, Qwen) is causal and generates. Encoder–decoder (T5, original Transformer) translates with cross-attention.',
  },
  {
    id: 'gqa',
    n: '18',
    title: 'GQA · MLA',
    group: 'Modern',
    blurb: 'Grouped-query attention shares KV heads across query groups. MLA (DeepSeek) compresses KV into a latent cache instead. Both exist for inference memory.',
  },
  {
    id: 'moe',
    n: '19',
    title: 'MoE',
    group: 'Modern',
    blurb: 'Mixture-of-Experts replaces the dense FFN with many experts; a router picks a few per token. Capacity goes up while active FLOPs stay bounded.',
    formula: 'y = Σ_{e ∈ Top-k} g_e FFN_e(x)',
  },
  {
    id: 'speculative',
    n: '20',
    title: 'Speculative',
    group: 'Modern',
    blurb: 'A cheap draft model proposes several tokens; the large model verifies them in one parallel pass. Accept or resample — distribution stays that of the target.',
  },
]

export const GLOSSARY = [
  {
    term: 'Temperature',
    body: 'Divide logits by T before softmax. T < 1 sharpens (more greedy). T > 1 flattens (more random). T → 0 is argmax.',
  },
  {
    term: 'Top-k',
    body: 'Keep only the k highest-probability tokens, zero the rest, renormalize. Caps how weird the tail can get.',
  },
  {
    term: 'Top-p / nucleus',
    body: 'Keep the smallest set of tokens whose cumulative probability ≥ p. Adaptive vocabulary size per step.',
  },
  {
    term: 'Min-p',
    body: 'Drop tokens below min_p × p_max. Scales with the head of the distribution — more stable than a fixed top-p in some chat settings.',
  },
  {
    term: 'Repetition penalty',
    body: 'Down-weights tokens already emitted so the model is less likely to loop. A logits hack, not part of the network.',
  },
  {
    term: 'RoPE',
    body: 'Rotary position embedding. Rotate query/key pairs in 2D planes by an angle proportional to position. Encodes relative offset in the dot product.',
  },
  {
    term: 'ALiBi',
    body: 'Attention with linear biases: add a head-specific slope × distance to scores. No extra params; extrapolates to longer context.',
  },
  {
    term: 'NoPE',
    body: 'No explicit positions. Causal mask plus depth can leak order. Some hybrid stacks mix RoPE and NoPE layers.',
  },
  {
    term: 'RMSNorm vs LayerNorm',
    body: 'LayerNorm centers and scales. RMSNorm only scales by RMS. Llama-style models dropped the mean and the bias.',
  },
  {
    term: 'Pre-norm vs Post-norm',
    body: 'Original Transformer: attn then norm. Modern: norm then attn (pre-norm). Pre-norm trains deeper nets without warmup tricks.',
  },
  {
    term: 'SwiGLU / GeGLU',
    body: 'Gated linear units. One branch is a gate (SiLU or GELU), multiplied by an up-projection, then down-projected. Standard in Llama, Mistral, Qwen.',
  },
  {
    term: 'GQA / MQA',
    body: 'Multi-query: one KV head for all Q heads. Grouped-query: several Q heads share each KV head. Shrinks KV cache almost linearly.',
  },
  {
    term: 'MLA',
    body: 'Multi-head latent attention. Down-project KV to a low-rank latent, cache that, up-project when attending. DeepSeek-V2/V3.',
  },
  {
    term: 'QK-Norm',
    body: 'RMSNorm on Q and K inside attention (often before RoPE). Stabilizes training of large or QK-unscaled models (OLMo 2, some Qwen).',
  },
  {
    term: 'Sliding window',
    body: 'Each token attends only to the last W positions (Mistral). Linear memory in sequence length; pair with a few full-attn layers or sinks.',
  },
  {
    term: 'Flash Attention',
    body: 'IO-aware exact attention. Tiles Q,K,V in SRAM so you never materialize the N×N matrix. Same math, much less HBM traffic.',
  },
  {
    term: 'Prefill vs decode',
    body: 'Prefill is compute-heavy (whole prompt, highly parallel). Decode is memory-heavy (one token, huge KV read). Different bottlenecks.',
  },
  {
    term: 'Weight tying',
    body: 'Embedding matrix E and unembedding W_U share weights (W_U = Eᵀ). Fewer params, slightly better sample efficiency.',
  },
  {
    term: 'Residual stream',
    body: 'Anthropic circuits view: the residual is a communication channel. Attention reads/writes, MLPs insert features. Layers don’t “replace” the state.',
  },
  {
    term: 'Induction heads',
    body: 'A two-layer circuit that completes [A][B] … [A] → [B]. Core of in-context learning in toy and real models.',
  },
  {
    term: 'Beam search',
    body: 'Keep the top-B partial sequences instead of one sample. Used in translation; chat models usually sample instead so answers don’t go bland.',
  },
  {
    term: 'Speculative decoding',
    body: 'Draft with a small model, verify in parallel with the large one. Lossless if you reject/resample correctly — same distribution, fewer serial steps.',
  },
  {
    term: 'Mixture of Experts',
    body: 'Sparse FFN: route each token to k of N experts. Mixtral, DeepSeekMoE, Llama 4. Adds parameters without matching active compute.',
  },
  {
    term: 'Dropout',
    body: 'Training-only noise on attention/FFN. Most modern LLM pretraining uses 0 dropout; regularization comes from scale and data.',
  },
]
