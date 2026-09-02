import { Loader2, User, Sparkles } from 'lucide-react'

export function InspectorLoadingSkeleton() {
  return (
    <div className="space-y-6 text-xs animate-in fade-in duration-200">
      {/* Loading header indicator banner */}
      <div className="flex items-center justify-between p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-mono text-[11px]">
        <div className="flex items-center space-x-2">
          <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
          <span className="font-semibold">Loading candidate profile &amp; SLA traces...</span>
        </div>
        <Sparkles className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
      </div>

      {/* Candidate Profile Header Card Skeleton */}
      <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-850 border border-slate-800 space-y-4 relative overflow-hidden">
        <div className="flex items-start space-x-3.5">
          {/* Avatar Skeleton */}
          <div className="w-12 h-12 rounded-2xl bg-slate-800 animate-pulse flex items-center justify-center text-slate-700">
            <User className="w-6 h-6 text-slate-700" />
          </div>
          <div className="space-y-2 flex-1">
            {/* Name */}
            <div className="h-5 w-44 bg-slate-800 rounded-lg animate-pulse" />
            {/* Title */}
            <div className="h-3.5 w-28 bg-slate-800/80 rounded-md animate-pulse" />
            {/* Contact Pills */}
            <div className="flex flex-wrap gap-2 pt-1">
              <div className="h-4 w-32 bg-slate-800/60 rounded-full animate-pulse" />
              <div className="h-4 w-24 bg-slate-800/60 rounded-full animate-pulse" />
              <div className="h-4 w-28 bg-slate-800/60 rounded-full animate-pulse" />
            </div>
          </div>
        </div>

        {/* Links bar skeleton */}
        <div className="flex gap-2 pt-3 border-t border-slate-800/80">
          <div className="h-6 w-24 bg-slate-800/70 rounded-lg animate-pulse" />
          <div className="h-6 w-28 bg-slate-800/70 rounded-lg animate-pulse" />
        </div>
      </div>

      {/* Professional Summary Skeleton */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="h-3 w-36 bg-slate-800 rounded-md animate-pulse" />
        <div className="space-y-2">
          <div className="h-3 w-full bg-slate-800/70 rounded animate-pulse" />
          <div className="h-3 w-5/6 bg-slate-800/70 rounded animate-pulse" />
          <div className="h-3 w-4/6 bg-slate-800/70 rounded animate-pulse" />
        </div>
      </div>

      {/* Work Experience Timeline Skeleton */}
      <div className="space-y-3">
        <div className="h-3 w-40 bg-slate-800 rounded-md animate-pulse" />
        <div className="space-y-3">
          {[1, 2].map((i) => (
            <div
              key={i}
              className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3"
            >
              <div className="flex items-start justify-between">
                <div className="space-y-1.5 flex-1">
                  <div className="h-3.5 w-36 bg-slate-800 rounded animate-pulse" />
                  <div className="h-3 w-24 bg-slate-800/80 rounded animate-pulse" />
                </div>
                <div className="h-4 w-24 bg-slate-800/60 rounded-full animate-pulse" />
              </div>
              <div className="h-3 w-full bg-slate-800/60 rounded animate-pulse" />
              <div className="flex gap-1.5 pt-1">
                <div className="h-4 w-12 bg-slate-800 rounded animate-pulse" />
                <div className="h-4 w-16 bg-slate-800 rounded animate-pulse" />
                <div className="h-4 w-14 bg-slate-800 rounded animate-pulse" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Skills Grid Skeleton */}
      <div className="space-y-3">
        <div className="h-3 w-36 bg-slate-800 rounded-md animate-pulse" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[1, 2].map((i) => (
            <div key={i} className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="h-3 w-20 bg-slate-800 rounded animate-pulse" />
              <div className="flex flex-wrap gap-1.5">
                <div className="h-4 w-14 bg-slate-800/70 rounded-lg animate-pulse" />
                <div className="h-4 w-16 bg-slate-800/70 rounded-lg animate-pulse" />
                <div className="h-4 w-12 bg-slate-800/70 rounded-lg animate-pulse" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
