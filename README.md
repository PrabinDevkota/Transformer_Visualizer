# Layer Trace

Interactive visualizer for a modern transformer block: tokenize → embed → position (RoPE / sinusoidal / ALiBi / NoPE) → RMSNorm → QKV → scores → mask → softmax → multi-head (MHA / GQA / MQA) → residual → SwiGLU → logits → sampling (temperature, top-k, top-p, min-p) → KV cache, plus encoder / decoder / seq2seq, MoE, and speculative decoding.

```bash
npm install
npm run dev
```

Type a sentence, pick an architecture in the top bar, then step through the pipeline. Arrow keys move between stages; space plays.
