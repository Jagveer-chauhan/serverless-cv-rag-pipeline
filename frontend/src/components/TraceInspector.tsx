import { Activity } from 'lucide-react'
import { StageTraceItem } from '../types'
import { SLAWaterfall } from './SLAWaterfall'

interface TraceInspectorProps {
  traces: StageTraceItem[]
  totalDurationMs?: number
}

export function TraceInspector({ traces, totalDurationMs = 0 }: TraceInspectorProps) {
  return (
    <div className="space-y-4">
      {/* SLA Waterfall Summary */}
      <SLAWaterfall traces={traces} totalDurationMs={totalDurationMs} />

      {/* Raw Trace Table */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
          <Activity className="w-4 h-4 text-emerald-400" />
          <span className="font-semibold uppercase tracking-wider">Detailed Stage Trace Logs</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                <th className="pb-2">Stage</th>
                <th className="pb-2">Duration</th>
                <th className="pb-2">Status</th>
                <th className="pb-2">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {traces.map((t, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30">
                  <td className="py-2.5 font-semibold text-slate-200">{t.stage}</td>
                  <td className="py-2.5 text-emerald-400 font-bold">{t.duration_ms.toFixed(2)} ms</td>
                  <td className="py-2.5">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] ${
                        t.status === 'success'
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : 'bg-rose-500/10 text-rose-400'
                      }`}
                    >
                      {t.status}
                    </span>
                  </td>
                  <td className="py-2.5 text-[11px] text-slate-500">
                    {t.start_time ? new Date(t.start_time).toLocaleTimeString() : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
