import React, { useState, useRef } from 'react'
import { UploadCloud, FileText, CheckCircle2, AlertCircle, X, Loader2 } from 'lucide-react'

interface DropzoneProps {
  onUpload: (files: File[]) => Promise<void>
  isUploading: boolean
}

export function Dropzone({ onUpload, isUploading }: DropzoneProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const isSupportedFile = (file: File) => {
    const name = file.name.toLowerCase()
    return (
      name.endsWith('.pdf') ||
      name.endsWith('.docx') ||
      name.endsWith('.doc') ||
      file.type === 'application/pdf' ||
      file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      file.type === 'application/msword'
    )
  }

  const validateAndAddFiles = (files: FileList | null) => {
    if (!files) return
    setError(null)
    const validDocs: File[] = []

    Array.from(files).forEach((file) => {
      if (isSupportedFile(file)) {
        if (file.size > 15 * 1024 * 1024) {
          setError(`File ${file.name} exceeds 15MB limit.`)
        } else {
          validDocs.push(file)
        }
      } else {
        setError(`Skipped ${file.name}: Only PDF and Word documents (.docx, .doc) are supported.`)
      }
    })

    if (validDocs.length > 0) {
      setSelectedFiles((prev) => [...prev, ...validDocs])
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndAddFiles(e.dataTransfer.files)
    }
  }

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleStartUpload = async () => {
    if (selectedFiles.length === 0 || isUploading) return
    try {
      await onUpload(selectedFiles)
      setSelectedFiles([])
    } catch (err: any) {
      setError(err.message || 'Upload failed')
    }
  }

  return (
    <div className="space-y-4">
      {/* Drag & Drop Box */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 ${
          dragActive
            ? 'border-emerald-400 bg-emerald-500/10 scale-[1.01]'
            : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword"
          onChange={(e) => validateAndAddFiles(e.target.files)}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <UploadCloud className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">
              Drop CV documents (PDF, Word) here, or <span className="text-emerald-400 underline decoration-emerald-500/50">browse</span>
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Supports multi-page PDFs &amp; Word files (.docx, .doc) &bull; Max 15MB
            </p>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center space-x-2 text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Selected File Queue */}
      {selectedFiles.length > 0 && (
        <div className="p-4 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>READY TO INGEST ({selectedFiles.length})</span>
            <button
              onClick={() => setSelectedFiles([])}
              className="text-slate-500 hover:text-slate-300"
            >
              Clear all
            </button>
          </div>

          <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
            {selectedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2.5 rounded-xl bg-slate-850 border border-slate-800 text-xs"
              >
                <div className="flex items-center space-x-2.5 truncate">
                  <FileText className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  <span className="text-slate-200 truncate font-medium">{file.name}</span>
                  <span className="text-slate-500 font-mono flex-shrink-0">
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                {!isUploading && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRemoveFile(idx)
                    }}
                    className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Trigger Ingestion Button */}
          <button
            onClick={handleStartUpload}
            disabled={isUploading}
            className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-xs flex items-center justify-center space-x-2 shadow-lg shadow-emerald-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Running Ingestion Pipeline (&le; 5.0s SLA)...</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4" />
                <span>Ingest &amp; Verify RAG Readiness</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
