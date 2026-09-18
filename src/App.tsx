import { useEffect, useMemo, useRef, useState } from 'react'
import { STAGES, type StageId } from './content/stages'
import { DEFAULT_CONFIG, type ModelConfig } from './engine/types'
import { runModel, tokensForView } from './engine/transformer'
import { Field, Select, Slider } from './components/Fields'
import { StageNav } from './components/StageNav'
import { Pipeline } from './viz/Pipeline'
import { ColorSlide, PipelineLegend } from './viz/ColorLegend'
import { Glossary, StageView } from './stages/StageView'

const EXAMPLES = [
  'The cat sat on the mat',
  'Attention is all you need',
  'Hello world',
  'Transformers map tokens to vectors',
]

export default function App() {
  const [text, setText] = useState(EXAMPLES[0]!)
  const [target, setTarget] = useState('le chat')
  const [cfg, setCfg] = useState<ModelConfig>(DEFAULT_CONFIG)
  const [stage, setStage] = useState<StageId>('tokenize')
  const [token, setToken] = useState(1)
  const [head, setHead] = useState(0)
  const [layer, setLayer] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [glossary, setGlossary] = useState(false)
  const [colors, setColors] = useState(false)

  const patch = (p: Partial<ModelConfig>) => setCfg((c) => ({ ...c, ...p }))
  const booted = useRef(false)

  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const q = p.get('q')
    if (q) setText(q)
    const arch = p.get('arch')
    if (arch === 'decoder' || arch === 'encoder' || arch === 'encdec') {
      setCfg((c) => ({ ...c, architecture: arch }))
    }
    const t = p.get('t')
    if (t) setCfg((c) => ({ ...c, temperature: Number(t) || c.temperature }))
    booted.current = true
  }, [])

  useEffect(() => {
    if (!booted.current) return
    const p = new URLSearchParams()
    p.set('q', text)
    p.set('arch', cfg.architecture)
    p.set('t', String(cfg.temperature))
    window.history.replaceState(null, '', `${window.location.pathname}?${p.toString()}`)
  }, [text, cfg.architecture, cfg.temperature])

  const trace = useMemo(() => runModel(text, target, cfg), [text, target, cfg])
  const toks = tokensForView(trace)
  const stageIndex = STAGES.findIndex((s) => s.id === stage)

  useEffect(() => {
    setToken((t) => Math.min(t, Math.max(0, toks.length - 1)))
    setHead((h) => Math.min(h, Math.max(0, (trace.layers[0]?.heads.length ?? 1) - 1)))
    setLayer((l) => Math.min(l, Math.max(0, trace.layers.length - 1)))
  }, [toks.length, trace.layers])

  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      setStage((cur) => {
        const i = STAGES.findIndex((s) => s.id === cur)
        return STAGES[(i + 1) % STAGES.length]!.id
      })
    }, 2400)
    return () => window.clearInterval(id)
  }, [playing])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return
      if (e.key === 'ArrowRight' || e.key === 'l') {
        setStage(STAGES[Math.min(STAGES.length - 1, stageIndex + 1)]!.id)
      }
      if (e.key === 'ArrowLeft' || e.key === 'h') {
        setStage(STAGES[Math.max(0, stageIndex - 1)]!.id)
      }
      if (e.key === ' ') {
        e.preventDefault()
        setPlaying((p) => !p)
      }
      if (e.key === 'Escape') {
        setColors(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [stageIndex])

  return (
    <div className="flex h-dvh flex-col bg-bg text-ink">
      <header className="shrink-0 border-b border-line px-4 py-3 lg:px-6">
        <div className="mb-3 flex items-baseline justify-between gap-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-faint">Layer trace</div>
            <h1 className="font-display text-[18px] tracking-tight">Modern transformer visualizer</h1>
          </div>
          <div className="hidden text-[11px] text-faint md:block">← → step · space play</div>
        </div>
        <div className="grid gap-3 lg:grid-cols-[1fr_minmax(0,1.2fr)]">
          <div className="flex flex-col gap-2">
            <Field label={cfg.architecture === 'encdec' ? 'Source' : 'Prompt'}>
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="h-9 w-full rounded-md border border-line bg-elev px-3 font-mono text-[13px] outline-none focus:border-gold"
              />
            </Field>
            {cfg.architecture === 'encdec' && (
              <Field label="Decoder prefix">
                <input
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  className="h-9 w-full rounded-md border border-line bg-elev px-3 font-mono text-[13px] outline-none focus:border-gold"
                />
              </Field>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Field label="Example">
              <Select
                value={EXAMPLES.includes(text) ? text : 'custom'}
                onChange={(v) => {
                  if (v !== 'custom') setText(v)
                }}
                options={[
                  { value: 'custom', label: 'custom' },
                  ...EXAMPLES.map((e) => ({ value: e, label: e })),
                ]}
              />
            </Field>
            <Field label="Architecture">
              <Select
                value={cfg.architecture}
                onChange={(v) => patch({ architecture: v as ModelConfig['architecture'] })}
                options={[
                  { value: 'decoder', label: 'Decoder-only' },
                  { value: 'encoder', label: 'Encoder-only' },
                  { value: 'encdec', label: 'Encoder–decoder' },
                ]}
              />
            </Field>
            <Field label="Position">
              <Select
                value={cfg.positional}
                onChange={(v) => patch({ positional: v as ModelConfig['positional'] })}
                options={[
                  { value: 'rope', label: 'RoPE' },
                  { value: 'sinusoidal', label: 'Sinusoidal' },
                  { value: 'learned', label: 'Learned' },
                  { value: 'alibi', label: 'ALiBi' },
                  { value: 'none', label: 'NoPE' },
                ]}
              />
            </Field>
            <Field label="Attention">
              <Select
                value={cfg.attention}
                onChange={(v) => patch({ attention: v as ModelConfig['attention'] })}
                options={[
                  { value: 'mha', label: 'MHA' },
                  { value: 'gqa', label: 'GQA' },
                  { value: 'mqa', label: 'MQA' },
                  { value: 'mla', label: 'MLA' },
                ]}
              />
            </Field>
            <Field label="Norm">
              <Select
                value={cfg.norm}
                onChange={(v) => patch({ norm: v as ModelConfig['norm'] })}
                options={[
                  { value: 'rmsnorm', label: 'RMSNorm' },
                  { value: 'layernorm', label: 'LayerNorm' },
                ]}
              />
            </Field>
            <Field label="FFN">
              <Select
                value={cfg.ffn}
                onChange={(v) => patch({ ffn: v as ModelConfig['ffn'] })}
                options={[
                  { value: 'swiglu', label: 'SwiGLU' },
                  { value: 'gelu', label: 'GELU' },
                  { value: 'relu', label: 'ReLU' },
                ]}
              />
            </Field>
            <Field label="Decode">
              <Select
                value={cfg.sampling}
                onChange={(v) => patch({ sampling: v as ModelConfig['sampling'] })}
                options={[
                  { value: 'greedy', label: 'Greedy' },
                  { value: 'temperature', label: 'Temperature' },
                  { value: 'topk', label: 'Top-k' },
                  { value: 'topp', label: 'Top-p' },
                  { value: 'minp', label: 'Min-p' },
                  { value: 'beam', label: 'Beam' },
                ]}
              />
            </Field>
            <Field label="Layers">
              <Select
                value={String(cfg.nLayers)}
                onChange={(v) => patch({ nLayers: Number(v) })}
                options={[1, 2, 3, 4].map((n) => ({ value: String(n), label: `${n}` }))}
              />
            </Field>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Temperature">
            <Slider value={cfg.temperature} min={0.1} max={2} step={0.1} onChange={(v) => patch({ temperature: v })} />
          </Field>
          <Field label="Top-k">
            <Slider value={cfg.topK} min={1} max={16} step={1} onChange={(v) => patch({ topK: v })} />
          </Field>
          <Field label="Top-p">
            <Slider value={cfg.topP} min={0.1} max={1} step={0.05} onChange={(v) => patch({ topP: v })} />
          </Field>
          <Field label="Min-p">
            <Slider value={cfg.minP} min={0} max={0.5} step={0.01} onChange={(v) => patch({ minP: v })} />
          </Field>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="hidden w-52 shrink-0 border-r border-line p-3 md:block">
          <StageNav current={stage} onPick={setStage} />
        </aside>

        <main className="min-w-0 flex-1 overflow-y-auto px-5 py-8 lg:px-10 lg:py-10">
          <div className="mb-8">
            <Pipeline current={stage} onPick={setStage} />
            <PipelineLegend />
          </div>
          <StageView
            stage={stage}
            trace={trace}
            token={token}
            head={head}
            layer={layer}
            setToken={setToken}
            setHead={setHead}
            setLayer={setLayer}
          />
        </main>

        {glossary && (
          <aside className="hidden w-80 shrink-0 overflow-y-auto border-l border-line p-4 lg:block">
            <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-faint">Glossary</div>
            <Glossary />
          </aside>
        )}
      </div>

      {colors && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/75 p-4 py-10 sm:p-8"
          onClick={() => setColors(false)}
        >
          <div onClick={(e) => e.stopPropagation()} className="w-full max-w-3xl">
            <ColorSlide />
            <button
              type="button"
              onClick={() => setColors(false)}
              className="mt-4 rounded-md border border-line px-3 py-1.5 font-mono text-[11px] text-mute hover:text-ink"
            >
              close · esc
            </button>
          </div>
        </div>
      )}

      <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-line px-4 py-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setStage(STAGES[Math.max(0, stageIndex - 1)]!.id)}
            className="rounded-md border border-line px-2 py-1 font-mono text-[11px] text-mute hover:text-ink"
          >
            prev
          </button>
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded-md border border-gold/40 bg-gold/10 px-3 py-1 font-mono text-[11px] text-gold"
          >
            {playing ? 'pause' : 'play'}
          </button>
          <button
            type="button"
            onClick={() => setStage(STAGES[Math.min(STAGES.length - 1, stageIndex + 1)]!.id)}
            className="rounded-md border border-line px-2 py-1 font-mono text-[11px] text-mute hover:text-ink"
          >
            next
          </button>
          <span className="ml-2 hidden font-mono text-[11px] text-faint sm:inline">
            {String(stageIndex + 1).padStart(2, '0')} / {STAGES.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setColors((c) => !c)}
            className={
              colors
                ? 'rounded-md border border-gold/40 bg-gold/10 px-2 py-1 font-mono text-[11px] text-gold'
                : 'rounded-md border border-line px-2 py-1 font-mono text-[11px] text-mute hover:text-ink'
            }
          >
            colors
          </button>
          <button
            type="button"
            onClick={() => setGlossary((g) => !g)}
            className="rounded-md border border-line px-2 py-1 font-mono text-[11px] text-mute hover:text-ink"
          >
            {glossary ? 'hide notes' : 'glossary'}
          </button>
          <span className="hidden font-mono text-[11px] text-faint md:inline">
            {toks.length} tok · d={cfg.dModel} · {cfg.nHeads}H
          </span>
        </div>
      </footer>
    </div>
  )
}
