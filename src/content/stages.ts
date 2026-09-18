export type StageId =
  | 'color'
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
  why: string[]
  formula?: string
}

export const STAGES: StageMeta[] = [
  {
    id: 'color',
    n: '00',
    title: 'Color',
    group: 'How to read',
    blurb: 'Hue is sign or mass. Brightness is magnitude. Same color means a similar number. The Read below is the whole legend, told in order.',
    why: [
      'A heatmap without a legend is decoration. Blue vs amber is the sign of a residual-stream coordinate; gold vs charcoal is how much softmax budget a key received. If two cells look alike, their numbers are close.',
      'Rainbow palettes (jet) invent false edges and fail for red–green color vision. This visualizer uses two encodings that do not collide: cool–warm for signed tensors, and dark→teal→gold for probabilities. An embedding strip cannot be mistaken for an attention row.',
      'Token stripes are categorical identity, not magnitude. Follow the left edge of a chip to track “cat” through Q, K, and the matrix. The cell fill is the value; the stripe is the name.',
    ],
  },
  {
    id: 'tokenize',
    n: '01',
    title: 'Tokenize',
    group: 'Input',
    blurb: 'Text is split into tokens — subword pieces a model can look up.',
    why: [
      'A neural net cannot read letters. It needs a finite vocabulary of ids. Whole words explode the table (“running”, “runs”, “rerun” would all be separate), and characters are too long a sequence. Subword tokenization is the compromise: common chunks stay one token; rare words break into pieces the model has already seen.',
      'Byte-pair encoding (BPE) starts from characters and repeatedly merges the most frequent adjacent pair. That is why “the” and “ing” become single tokens while a made-up word splits. Special symbols such as beginning/end of sequence are real vocabulary entries — they tell the model where a string starts and when to stop generating.',
    ],
    formula: String.raw`\text{text} \rightarrow [t_0, t_1, \ldots, t_n] \subset V`,
  },
  {
    id: 'embed',
    n: '02',
    title: 'Embed',
    group: 'Input',
    blurb: 'Each token id looks up a learned vector — the residual stream’s starting state.',
    why: [
      'An id is just an integer. The embedding table E turns it into a d-dimensional vector the rest of the net can add, rotate, and attend over. Similar tokens in a trained model land in similar directions; that geometry is learned from data, not hand-coded.',
      'This demo uses seeded vectors so the same token always maps to the same row. The colors are signed magnitudes of each dimension — not “meaning paint.” In a real LLM those directions encode features (syntax, entity, sentiment) that later layers read.',
      'Why a table and not a hash of the letters? Because the model must move nearby meanings with gradient descent. A learned lookup is the cheapest way to give every token its own handle in residual-stream space.',
    ],
    formula: String.raw`x_i = E[t_i] \in \mathbb{R}^{d}`,
  },
  {
    id: 'position',
    n: '03',
    title: 'Position',
    group: 'Input',
    blurb: 'Attention has no order of its own. Position encodings inject “where” into “what.”',
    why: [
      'If you permute the tokens, plain dot-product attention cannot tell. Without a position signal, “the cat sat” and “sat cat the” look identical. That is the bug RoPE, sinusoids, ALiBi, and learned embeddings exist to fix.',
      'RoPE does not add a vector. It rotates each query/key pair in 2-D planes by an angle that depends on index. The dot product then depends on relative offset pᵢ − pⱼ, which is what language actually needs (“the word two seats left”). Sinusoidal and learned encodings add a position vector instead; ALiBi adds a distance bias to the scores; NoPE hopes the causal mask leaks order.',
      'Why rotate rather than add? Addition mixes “what the token is” with “where it is” in the same coordinates. Rotation keeps content magnitude and encodes location in the angle — cleaner at long context, which is why Llama, Qwen, and Mistral standardized on RoPE.',
    ],
    formula: String.raw`(q', k') = R_{\theta,p}\,q,\; R_{\theta,p}\,k`,
  },
  {
    id: 'norm',
    n: '04',
    title: 'Norm',
    group: 'Block',
    blurb: 'Pre-norm RMSNorm rescales the stream before attention so deep stacks stay stable.',
    why: [
      'Each layer adds a write-back onto the residual. After dozens of layers the activations can blow up or die. Normalization keeps the typical size of a vector near 1 so softmax and matrix multiplies stay well-behaved.',
      'LayerNorm subtracts the mean then divides by standard deviation (center + scale). RMSNorm skips the mean: it only divides by root-mean-square. Fewer ops, no bias, and it works as well at LLM scale — that is why Llama dropped LayerNorm.',
      'Why before attention (pre-norm), not after (post-norm, 2017 paper)? Pre-norm lets the residual stay a clean highway: even a badly behaved block cannot overwrite the stream, it can only add a normalized increment. Deep nets then train without heroic warmup.',
    ],
    formula: String.raw`\mathrm{RMS}(x) = \dfrac{x}{\sqrt{\mathrm{mean}(x^{2})+\varepsilon}} \odot \gamma`,
  },
  {
    id: 'qkv',
    n: '05',
    title: 'Q K V',
    group: 'Block',
    blurb: 'Three linear maps turn one stream into “what I seek”, “what I have”, and “what I give.”',
    why: [
      'Attention is content-addressable memory. A query is the lookup key you issue; keys are labels on stored items; values are the payloads you actually copy. Using the same vector for all three would force “how I match” and “what I write” to be identical — too cramped.',
      'So the model learns three matrices. They are just linear maps of the same residual x. Different heads (next stages) take different slices so several lookups can run at once.',
      'Why linear, not an MLP? Speed and residual-stream algebra: a linear read is enough to pick a direction, and it keeps attention a well-studied bilinear form. The MLP later (the FFN) is where nonlinear features get built.',
    ],
    formula: String.raw`Q = XW_Q \quad K = XW_K \quad V = XW_V`,
  },
  {
    id: 'scores',
    n: '06',
    title: 'Scores',
    group: 'Block',
    blurb: 'A scaled dot product measures how well a query matches each key.',
    why: [
      'Dot product is large when two vectors point the same way. That is a cheap, GPU-friendly test for “does this key look like what I asked for?” The matrix S is every query against every key.',
      'Why divide by the square root of head size? A d-dimensional random vector’s dot product has standard deviation about √d. Without scaling, softmax sees huge logits, saturates to one-hot, and gradients vanish. The 2017 paper picked √dₕ so scores stay O(1) as heads get wider.',
      'The number in each cell is not yet “attention.” It is a raw compatibility. Masking and softmax still have to turn it into a probability of how much value to copy.',
    ],
    formula: String.raw`S = \dfrac{QK^{\top}}{\sqrt{d_h}}`,
  },
  {
    id: 'mask',
    n: '07',
    title: 'Mask',
    group: 'Block',
    blurb: 'Illegal positions are set to −∞ so they cannot receive probability.',
    why: [
      'A decoder that writes the next token must not peek at it. Causal masking zeros the upper triangle: query i may only see keys j ≤ i. Without that, training would cheat and generation would not match training.',
      'Encoders (BERT) want the opposite: every token may look at every other, because the job is a bidirectional representation, not left-to-right generation. Encoder–decoder models mix both: the encoder is full, the decoder is causal, and cross-attention lets decoder queries read all encoder keys.',
      'Why −∞ instead of multiplying by zero? Softmax(−∞) = 0 automatically, and the remaining legal scores still sum to 1. A hard multiply-by-zero after softmax would break the distribution.',
    ],
    formula: String.raw`S_{ij} \leftarrow -\infty \quad \text{if } j \text{ is illegal}`,
  },
  {
    id: 'softmax',
    n: '08',
    title: 'Softmax',
    group: 'Block',
    blurb: 'Scores become a probability distribution over tokens to read.',
    why: [
      'We need weights that are positive and sum to 1 so “how much of each value to mix” is a convex combination. Softmax is the standard map from real scores to a simplex: bigger score → exponentially more mass.',
      'Why exponential, not a ReLU-and-normalize? The exponential makes winners win harder (a useful sparse readout) while still giving a gradient to the losers. Temperature T later is the same formula with S/T: low T is peaky, high T is flat.',
      'Entropy of a row tells you the head’s mood: near 0 bits it stares at one token; many bits it averages the context. Both are legitimate — syntax heads are peaky, “semantic pooling” heads are diffuse.',
    ],
    formula: String.raw`\alpha_{ij} = \dfrac{\exp(S_{ij})}{\sum_{k}\exp(S_{ik})}`,
  },
  {
    id: 'attend',
    n: '09',
    title: 'Attend',
    group: 'Block',
    blurb: 'Each query copies a weighted blend of value vectors.',
    why: [
      'This is the actual move of information. Scores and softmax only decided the recipe. Here the recipe is applied: each query becomes a mixture of other positions’ values. That is how “it” can point at “cat” three tokens back.',
      'Why mix values instead of copying the best key? Soft weights let the model interpolate (two antecedents, a little of both) and they keep the function differentiable. Argmax attention exists in some papers but is rarer in LLMs.',
      'The residual stream is the bus; this weighted sum is the write that attention puts on the bus. Later layers read whatever was deposited.',
    ],
    formula: String.raw`z_i = \sum_j \alpha_{ij}\, v_j`,
  },
  {
    id: 'heads',
    n: '10',
    title: 'Heads',
    group: 'Block',
    blurb: 'Several attentions run in parallel on subspaces, then concatenate.',
    why: [
      'One head is one lookup. Language needs many simultaneous lookups: anaphora, syntax, induction, positional copying. Splitting d into h heads is cheaper than running h full d-dimensional attentions, and each head can specialize.',
      'After each head produces its output, they are concatenated back to d and mixed with an output projection. That mix is where heads can cancel or collaborate in the residual stream.',
      'GQA/MQA/MLA do not exist for quality. They exist because inference must cache K and V for every past token. Sharing or compressing KV heads shrinks that cache with only a small quality hit — see the GQA page.',
    ],
    formula: String.raw`\mathrm{MultiHead} = \mathrm{Concat}(h_1,\ldots,h_h)\,W_O`,
  },
  {
    id: 'residual',
    n: '11',
    title: 'Residual',
    group: 'Block',
    blurb: 'Add the block’s write-back to the stream instead of replacing it.',
    why: [
      'If a layer had to output the whole new state, a bad layer would destroy everything the previous ones computed. Adding a delta (x ← x + F(x)) means a layer can do nothing and the stream still survives. That is why 100-layer transformers train at all.',
      'Circuits work thinks of the residual as a communication channel: attention reads from it and writes into it; MLPs insert features. Tokens do not “become” the next layer — they accumulate edits.',
      'The same pattern repeats after the FFN. Two residuals per block is the modern default: one for attention’s write, one for the MLP’s write.',
    ],
    formula: String.raw`x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x))`,
  },
  {
    id: 'ffn',
    n: '12',
    title: 'FFN',
    group: 'Block',
    blurb: 'A position-wise MLP. This is where most parameters live, and where features get built.',
    why: [
      'Attention moves information between tokens. It does not, by itself, compute a new feature at a position. The feed-forward net is applied to each token independently: expand, nonlinear, project back. Empirically this is where facts and “if you see X, write Y” rules live.',
      'Why SwiGLU over ReLU/GELU? A gated linear unit has two branches; one gates the other with SiLU. That extra multiplicative interaction is worth the parameters — Llama, Mistral, Qwen all switched. GeGLU is the GELU cousin (Gemma).',
      'The hidden width is usually 4× (or 8/3× for SwiGLU so parameter count matches). This layer dominates the weight file; attention dominates the inference memory (KV cache). Different bottlenecks, both essential.',
    ],
    formula: String.raw`\mathrm{SwiGLU}(x) = \bigl(\mathrm{SiLU}(xW_g) \odot xW_u\bigr) W_d`,
  },
  {
    id: 'stack',
    n: '13',
    title: 'Stack',
    group: 'Block',
    blurb: 'The same block repeats. Depth is how the model builds long algorithms.',
    why: [
      'One block can copy a token or apply a shallow MLP. Useful behavior — induction, multi-step reasoning — is a composition of many such edits. Stacking L copies with independent weights is the whole model.',
      'Early layers tend to handle local syntax and positional bookkeeping. Middle layers do in-context algorithms. Late layers move the residual toward the unembedding directions of likely next tokens. That is a tendency, not a law; tap each layer’s vector to see it actually change.',
      'Why not one giant layer? Depth lets each step be a small, stable residual edit. Width-without-depth saturates; depth-without-residual is untrainable. The stack-of-residuals is the shape that survived 2017–2026.',
    ],
    formula: String.raw`X^{(\ell+1)} = \mathrm{Block}^{(\ell)}\bigl(X^{(\ell)}\bigr)`,
  },
  {
    id: 'unembed',
    n: '14',
    title: 'Unembed',
    group: 'Output',
    blurb: 'The last hidden vector is scored against every vocabulary direction.',
    why: [
      'Generation is “which token is most aligned with the residual right now?” A linear unembedding sends h into |V| logits. Each logit is essentially a dot product with that token’s output direction.',
      'Why often tie unembedding to the embedding table (transposed)? The same geometry that made “cat” a useful input direction should be the direction you score when predicting “cat.” Tying halves that parameter block and usually helps a little.',
      'Encoder-only models stop earlier: they need a representation, not a next token. They pool (mean or a [CLS] token) and send that vector to a classifier or an embedding index. Switch architecture to encoder to see that path.',
    ],
    formula: String.raw`\ell = W_U \cdot \mathrm{RMS}(h_{\mathrm{last}})`,
  },
  {
    id: 'sampling',
    n: '15',
    title: 'Sampling',
    group: 'Output',
    blurb: 'Logits are not a choice yet. Filters and temperature reshape the distribution, then we draw.',
    why: [
      'Argmax (greedy) repeats itself: the mode of a language model is often a loop. Sampling from softmax(ℓ / T) injects controlled noise so dialogue does not collapse. T < 1 sharpens toward greedy; T > 1 flattens toward chaos; T → 0 is argmax.',
      'Top-k throws away the long tail (hard cap). Top-p keeps the smallest set whose mass is at least p (adaptive cap). Min-p drops anything below a fraction of the best token — it scales with how confident the head is. Beam search keeps several whole strings; great for translation, bland for chat.',
      'These knobs are not inside the network. They are decoding policy. The same trained weights can be a thesaurus or a calculator depending on T, k, and p. Tap a candidate to see its logit, filtered-or-not, and probability after renormalization.',
    ],
    formula: String.raw`p = \mathrm{softmax}(\ell / T)\;\text{ then filter}`,
  },
  {
    id: 'kvcache',
    n: '16',
    title: 'KV cache',
    group: 'Output',
    blurb: 'Prefill writes keys and values once. Decode reuses them so history is not recomputed.',
    why: [
      'Causal attention at step t only needs K and V of tokens 1…t. Those projections do not change when you add token t+1. Saving them (the KV cache) turns an O(t²) recompute into an O(t) read plus one new QKV.',
      'Prefill (the prompt) is compute-heavy and parallel. Decode (one new token at a time) is memory-heavy: the GPU spends its time streaming this cache from HBM, not multiplying. That is why long chats feel slow, and why GQA, MLA, sliding windows, and quantization exist.',
      'Cache size ≈ layers × KV-heads × sequence × dₕ × 2 (K and V) × bytes. Cut KV heads by 4 (GQA) and you cut cache by ~4. That is an inference tax, not a training tax — training still sees full sequences in parallel.',
    ],
    formula: String.raw`K \leftarrow [K;\,k_t] \qquad V \leftarrow [V;\,v_t]`,
  },
  {
    id: 'arch',
    n: '17',
    title: 'Architectures',
    group: 'Modern',
    blurb: 'Same block, three wiring diagrams: encoder, decoder, encoder–decoder.',
    why: [
      'The original 2017 Transformer was encoder–decoder for translation: a bidirectional encoder reads the source, a causal decoder writes the target, cross-attention ties them. T5 still lives here.',
      'Encoder-only (BERT) threw away generation. Full attention over the sentence is the right prior for classification and embeddings — every token should see both sides. There is no next-token head; there is a pooled vector.',
      'Decoder-only (GPT, Llama, Qwen) threw away the encoder. If you train a causal model on enough data, it learns to condition on the prompt as if the prompt were the “source.” One stack, one mask, one training objective. That simplicity won 2024–2026. Switch the dropdown: the mask, the extra cross-attention, and the output head all change for this reason.',
    ],
  },
  {
    id: 'gqa',
    n: '18',
    title: 'GQA · MLA',
    group: 'Modern',
    blurb: 'Share or compress KV so the cache is not one head per query.',
    why: [
      'Multi-head attention gives every query head its own K and V. Quality is fine; the cache is fat. During decode that fat cache is the bottleneck, not the matmuls.',
      'MQA: one KV head for all queries. Tiny cache, sometimes too little key diversity. GQA: groups of queries share a KV head — the current default (Llama 2/3, Mistral, Qwen). MLA (DeepSeek): store a low-rank latent instead of full KV, then up-project. Different engineering, same motive: shrink bytes moved per generated token.',
      'Why not just fewer query heads? Queries are cheap at decode (one new token). Keys/values are expensive (the whole past). Cut the expensive side first.',
    ],
  },
  {
    id: 'moe',
    n: '19',
    title: 'MoE',
    group: 'Modern',
    blurb: 'Many expert FFNs; a router runs only a few per token.',
    why: [
      'The FFN is where most parameters sit. Making it denser (wider) costs FLOPs on every token. Mixture-of-Experts keeps a huge total parameter count but routes each token to k of N experts, so active compute stays near a small dense model.',
      'The router is a softmax over experts. Top-k wins; the rest are skipped. DeepSeek uses many fine experts plus shared ones that always run; Llama 4 uses fewer large experts and sometimes alternates dense/MoE layers. Load balancing matters: if every token picks expert 0, you have a dense net plus wasted weights.',
      'Why not always MoE? Routing is fiddly (training instability, expert imbalance, implementation). Dense FFNs are simpler to train and to ship. MoE wins when you want a bigger brain without a matching energy bill.',
    ],
    formula: String.raw`y = \sum_{e \in \mathrm{Top}\text{-}k} g_e\,\mathrm{FFN}_e(x)`,
  },
  {
    id: 'speculative',
    n: '20',
    title: 'Speculative',
    group: 'Modern',
    blurb: 'A cheap draft proposes tokens; the large model verifies several at once.',
    why: [
      'Decode is serial: one token, then another. GPUs hate serial. Speculative decoding uses a small model (or extra heads) to guess k tokens, then the large model scores them in a single parallel forward pass. Accept the matching prefix; on the first mismatch, resample from the large model.',
      'If you reject correctly, the output distribution equals the large model’s — faster, not approximate. You spend extra FLOPs on the draft and on discarded guesses, but you convert idle decode bandwidth into useful parallel work.',
      'Why it works: the draft is usually right on “the”, “of”, closing quotes — the easy mass of language. The expensive model only has to intervene on the surprising tokens. Tap a chip to see accept vs reject.',
    ],
  },
]

export const GLOSSARY = [
  {
    term: 'Color encoding',
    body: 'Blue = negative, amber = positive, near-black ≈ 0 (signed tensors). Dark → teal → gold = probability mass. Token stripes are identity, not value. Brightness is magnitude, not “importance.”',
  },
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
    body: 'Rotary position embedding. Rotate query/key pairs in 2-D planes by an angle proportional to position. Encodes relative offset in the dot product.',
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
    body: 'Original Transformer: attention then norm. Modern: norm then attention (pre-norm). Pre-norm trains deeper nets without warmup tricks.',
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
    body: 'IO-aware exact attention. Tiles Q, K, V in SRAM so you never materialize the N×N matrix. Same math, much less HBM traffic.',
  },
  {
    term: 'Prefill vs decode',
    body: 'Prefill is compute-heavy (whole prompt, highly parallel). Decode is memory-heavy (one token, huge KV read). Different bottlenecks.',
  },
  {
    term: 'Weight tying',
    body: 'Embedding matrix E and unembedding share weights (unembedding = Eᵀ). Fewer params, slightly better sample efficiency.',
  },
  {
    term: 'Residual stream',
    body: 'The residual is a communication channel. Attention reads/writes, MLPs insert features. Layers do not replace the state; they edit it.',
  },
  {
    term: 'Induction heads',
    body: 'A two-layer circuit that completes [A][B] … [A] → [B]. Core of in-context learning in toy and real models.',
  },
  {
    term: 'Beam search',
    body: 'Keep the top-B partial sequences instead of one sample. Used in translation; chat models usually sample instead so answers do not go bland.',
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
