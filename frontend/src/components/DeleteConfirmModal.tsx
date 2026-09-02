import { useEffect } from 'react'
import { Trash2, AlertTriangle, FileText, Loader2, X, AlertCircle } from 'lucide-react'
import { CVListItem } from '../types'

interface DeleteConfirmModalProps {
  isOpen: boolean
  cv: CVListItem | null
  isDeleting: boolean
  error: string | null
  onConfirm: () => void
  onClose: () => void
}

export function DeleteConfirmModal({
  isOpen,
  cv,
  isDeleting,
  error,
  onConfirm,
  onClose,
}: DeleteConfirmModalProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isDeleting) {
        onClose()
      }
    }
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown)
    }
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, isDeleting, onClose])

  if (!isOpen || !cv) return null

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200">
      <div 
        className="glass-panel w-full max-w-md rounded-3xl border border-slate-800 p-6 space-y-5 relative shadow-2xl bg-slate-900/95"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={() => !isDeleting && onClose()}
          disabled={isDeleting}
          className="absolute right-5 top-5 p-1.5 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Close dialog"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header Icon + Title */}
        <div className="flex items-start space-x-3.5">
          <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 flex-shrink-0 shadow-lg shadow-rose-500/10">
            <Trash2 className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Delete CV Document</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              This action permanently purges candidate embeddings and metadata.
            </p>
          </div>
        </div>

        {/* CV Details Preview Card */}
        <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/80 space-y-2">
          <div className="flex items-center space-x-2.5">
            <FileText className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span className="text-xs font-semibold text-slate-200 truncate" title={cv.filename}>
              {cv.filename}
            </span>
          </div>
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800/60">
            <span>Size: {(cv.file_size / 1024).toFixed(1)} KB</span>
            <span className="text-slate-500">ID: {cv.id.slice(0, 8)}...</span>
          </div>
        </div>

        {/* Caution Notice */}
        <div className="flex items-start space-x-2 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-[11px] leading-relaxed">
            Deleting this document will remove all indexed vector embeddings from pgvector and clear its extracted JSON profile. This cannot be undone.
          </p>
        </div>

        {/* Error notification if any */}
        {error && (
          <div className="flex items-center space-x-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs font-mono">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-end space-x-2.5 pt-1">
          <button
            type="button"
            onClick={onClose}
            disabled={isDeleting}
            className="px-4 py-2 rounded-xl border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isDeleting}
            className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold flex items-center space-x-2 shadow-lg shadow-rose-600/20 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isDeleting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Deleting Resume...</span>
              </>
            ) : (
              <>
                <Trash2 className="w-4 h-4" />
                <span>Delete Document</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
