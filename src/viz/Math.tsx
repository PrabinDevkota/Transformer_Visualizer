import katex from 'katex'
import 'katex/dist/katex.min.css'

export function Tex({
  expr,
  display = false,
  className = '',
}: {
  expr: string
  display?: boolean
  className?: string
}) {
  const html = katex.renderToString(expr, {
    throwOnError: false,
    displayMode: display,
    output: 'html',
  })
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}

export function Formula({ latex }: { latex: string }) {
  return (
    <div className="formula-box max-w-full overflow-x-auto rounded-md border border-line bg-soft px-4 py-3">
      <Tex expr={latex} display />
    </div>
  )
}
