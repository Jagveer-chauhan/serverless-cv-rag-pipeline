import { Clock, CheckCircle, ShieldAlert, Sparkles } from 'lucide-react'
import { StageTraceItem } from '../types'

interface SLAWaterfallProps {
  traces: StageTraceItem[]
  totalDurationMs?: number
  slaTargetMs?: number
}

const STAGE_LABELS: Record<string, { label: string; color: string }> = {
  text_extraction: { label: '1. Text Extraction (PyMuPDF / OCR)', color: 'bg-cyan-500' },
  chunking: { label: '2. Section Regex Chunking', color: 'bg-teal-500' },
  llm_extraction: { label: '3. LLM Parallel Extraction (Gemma)', color: 'bg-indigo-500' },
  validation: { label: '4. Dynamic Schema Validation', color: 'bg-purple-500' },
  merge: { label: '5. Merge & Deduplication', color: 'bg-pink-500' },
  embedding: { label: '6. Embeddings (all-MiniLM-L6-v2)', color: 'bg-amber-500' },
  vector_upsert: { label: '7. Supabase pgvector Upsert', color: 'bg-emerald-500' },
  rag_verification: { label: '8. RAG Top-1 Verification Gate', color: 'bg-emerald-400' },
  total: { label: 'Total Pipeline', color: 'bg-emerald-500' },
}

export function SLAWaterfall({ traces, totalDurationMs = 0, slaTargetMs = 5000 }: SLAWaterfallProps) {
  const isWithinSLA = totalDurationMs <= slaTargetMs
  // Filter out the 'total' trace for the stage breakdown list
  const stageTraces = traces.filter((t) => t.stage !== 'total')
  const maxStageDuration = Math.max(...stageTraces.map((t) => t.duration_ms), 10)

  return (
    <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
      {/* SLA Header & Compliance Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-emerald-400" />
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold">
            PipelineTracer SLA Waterfall
          </h3>
        </div>

        <div className="flex items-center space-x-2 font-mono text-xs">
          {isWithinSLA ? (
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1 font-semibold">
              <CheckCircle className="w-3.5 h-3.5" />
              SLA MET ({totalDurationMs.toFixed(1)}ms / {slaTargetMs}ms)
            </span>
          ) : (
            <span className="px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1 font-semibold">
              <ShieldAlert className="w-3.5 h-3.5" />
              SLA EXCEEDED ({totalDurationMs.toFixed(1)}ms)
            </span>
          )}
        </div>
      </div>

      {/* Progress / Waterfall Bars */}
      <div className="space-y-2.5">
        {stageTraces.length === 0 ? (
          <div className="p-4 text-center text-xs text-slate-500 font-mono">
            No stage timing traces recorded yet.
          </div>
        ) : (
          stageTraces.map((trace) => {
            const config = STAGE_LABELS[trace.stage] || {
              label: trace.stage,
              color: 'bg-slate-500',
            }
            const widthPct = Math.min(100, Math.max(4, (trace.duration_ms / maxStageDuration) * 100))

            return (
              <div key={trace.stage} className="space-y-1 text-xs">
                <div className="flex items-center justify-between font-mono">
                  <span className="text-slate-300 truncate max-w-[260px]">{config.label}</span>
                  <div className="flex items-center space-x-2 font-semibold">
                    <span className="text-emerald-400">{trace.duration_ms.toFixed(2)} ms</span>
                    <span
                      className={`text-[10px] px-1.5 py-0.2 rounded ${
                        trace.status === 'success'
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : 'bg-rose-500/10 text-rose-400'
                      }`}
                    >
                      {trace.status}
                    </span>
                  </div>
                </div>

                {/* Visual Bar */}
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    style={{ width: `${widthPct}%` }}
                    className={`h-full rounded-full ${config.color} transition-all duration-500`}
                  />
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* SLA Target Benchmark Marker */}
      <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
        <span className="flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-emerald-400" />
          Warm-path target p95 SLA: <strong>&le; 5,000 ms</strong>
        </span>
        <span className="text-slate-300 font-semibold">
          Recorded: <strong className="text-white font-bold">{totalDurationMs.toFixed(2)} ms</strong>
        </span>
      </div>
    </div>
  )
}
