import { Zap, Activity, RefreshCw } from 'lucide-react'
import { KeepaliveStatus } from '../types'

interface HeaderProps {
  keepalive: KeepaliveStatus | null
  loading: boolean
  onPing: () => void
}

export function Header({ keepalive, loading, onPing }: HeaderProps) {
  const isWarm = keepalive?.status === 'warm'

  return (
    <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
      {/* Brand & SLA indicator */}
      <div className="flex items-center space-x-3.5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 via-teal-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
          <Zap className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="font-bold text-base text-white tracking-tight">Serverless CV Parsing &amp; RAG</h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono font-medium flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              SLA &le; 5.0s Warm-Path
            </span>
          </div>
          <p className="text-xs text-slate-400">FastAPI &bull; Pydantic v2 &bull; Supabase pgvector &bull; Gemma-3-4B-It</p>
        </div>
      </div>

      {/* Telemetry & Keepalive Status */}
      <div className="flex items-center space-x-3 text-xs font-mono">
        <div className="hidden md:flex items-center space-x-3 px-3 py-1.5 rounded-lg bg-slate-850 border border-slate-800 text-slate-300">
          <div className="flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            <span>RAM: <strong className="text-white">{keepalive ? `${keepalive.memory_usage_mb}MB` : '--'}</strong></span>
          </div>
          <div className="w-px h-3 bg-slate-700" />
          <div>
            <span>UPTIME: <strong className="text-white">{keepalive ? `${Math.round(keepalive.uptime_seconds)}s` : '--'}</strong></span>
          </div>
        </div>

        {/* Warm Worker Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-850 border border-slate-750">
          <span className={`w-2 h-2 rounded-full ${isWarm ? 'bg-emerald-400 shadow-sm shadow-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
          <span className="text-slate-200 font-semibold tracking-wide">
            {loading ? 'CHECKING...' : isWarm ? 'WORKER WARM' : 'STANDBY'}
          </span>
        </div>

        <button
          onClick={onPing}
          title="Trigger keepalive webhook to prevent 15-min Render idle timeout"
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/60 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-emerald-400' : ''}`} />
        </button>
      </div>
    </header>
  )
}
