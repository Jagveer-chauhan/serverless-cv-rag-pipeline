import { useState } from 'react'
import { FileText, Search, Trash2, CheckCircle2, Loader2, AlertTriangle, FileCode } from 'lucide-react'
import { CVListItem } from '../types'

interface CVListSidebarProps {
  cvs: CVListItem[]
  selectedId: string | null
  loadingDocId?: string | null
  deletingId?: string | null
  onSelect: (id: string) => void
  onDeleteRequest: (cv: CVListItem, e: React.MouseEvent) => void
  loading: boolean
}

export function CVListSidebar({
  cvs,
  selectedId,
  loadingDocId,
  deletingId,
  onSelect,
  onDeleteRequest,
  loading,
}: CVListSidebarProps) {
  const [searchTerm, setSearchTerm] = useState('')

  const filteredCvs = cvs.filter((cv) =>
    cv.filename.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'rag_ready':
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1 font-mono font-medium">
            <CheckCircle2 className="w-3 h-3" />
            rag_ready
          </span>
        )
      case 'indexing':
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center gap-1 font-mono font-medium">
            <Loader2 className="w-3 h-3 animate-spin" />
            indexing
          </span>
        )
      case 'extracting':
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center gap-1 font-mono font-medium">
            <Loader2 className="w-3 h-3 animate-spin" />
            extracting
          </span>
        )
      case 'failed':
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1 font-mono font-medium">
            <AlertTriangle className="w-3 h-3" />
            failed
          </span>
        )
      default:
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
            {status}
          </span>
        )
    }
  }

  return (
    <aside className="w-full md:w-80 flex-shrink-0 flex flex-col glass-panel rounded-2xl border border-slate-800 overflow-hidden h-[calc(100vh-6.5rem)]">
      {/* Sidebar Header & Search */}
      <div className="p-3.5 border-b border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <FileCode className="w-4 h-4 text-emerald-400" />
            Ingested CVs ({cvs.length})
          </h2>
        </div>

        {/* Search Input */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search candidate / file..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/60 transition-colors"
          />
        </div>
      </div>

      {/* CV List Cards */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {loading && cvs.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-500 flex flex-col items-center justify-center space-y-2">
            <Loader2 className="w-5 h-5 animate-spin text-emerald-400" />
            <span>Loading CVs...</span>
          </div>
        ) : filteredCvs.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-500 font-mono">
            {searchTerm ? 'No matching CVs found.' : 'No CVs uploaded yet.'}
          </div>
        ) : (
          filteredCvs.map((cv) => {
            const isSelected = cv.id === selectedId
            const isDocLoading = loadingDocId === cv.id
            const isDocDeleting = deletingId === cv.id

            return (
              <div
                key={cv.id}
                onClick={() => !isDocDeleting && onSelect(cv.id)}
                className={`group p-3 rounded-xl cursor-pointer transition-all duration-150 border text-xs relative ${
                  isDocDeleting
                    ? 'opacity-60 bg-rose-950/20 border-rose-500/30 cursor-not-allowed'
                    : isSelected
                    ? 'bg-emerald-500/10 border-emerald-500/40 shadow-sm'
                    : 'bg-slate-900/50 hover:bg-slate-900 border-slate-800/80 hover:border-slate-700'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start space-x-2 truncate flex-1">
                    {isDocLoading ? (
                      <Loader2 className="w-4 h-4 mt-0.5 flex-shrink-0 animate-spin text-emerald-400" />
                    ) : (
                      <FileText
                        className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
                          isSelected ? 'text-emerald-400' : 'text-slate-400 group-hover:text-slate-200'
                        }`}
                      />
                    )}
                    <div className="truncate flex-1">
                      <div className="flex items-center gap-1.5">
                        <p className="font-semibold text-slate-100 truncate">{cv.filename}</p>
                        {isDocLoading && (
                          <span className="text-[10px] text-emerald-400 font-mono animate-pulse">
                            loading...
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                        {(cv.file_size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>

                  {/* Delete Button / Deleting State */}
                  {isDocDeleting ? (
                    <div className="flex items-center space-x-1 text-rose-400 text-[10px] font-mono font-medium bg-rose-500/10 px-2 py-0.5 rounded-md border border-rose-500/20">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span>Deleting...</span>
                    </div>
                  ) : (
                    <button
                      onClick={(e) => onDeleteRequest(cv, e)}
                      title="Delete document"
                      className="opacity-0 group-hover:opacity-100 p-1 rounded-md hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 transition-all"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                {/* Status & Latency Footer */}
                <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-slate-800/60 font-mono">
                  {getStatusBadge(cv.status)}

                  {cv.total_duration_ms ? (
                    <span className="text-[11px] text-emerald-400 font-semibold">
                      {cv.total_duration_ms.toFixed(0)} ms
                    </span>
                  ) : null}
                </div>
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}

