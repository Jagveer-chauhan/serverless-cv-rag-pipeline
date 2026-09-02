import React, { useState, useEffect, useCallback } from 'react'
import { Header } from './components/Header'
import { Dropzone } from './components/Dropzone'
import { CVListSidebar } from './components/CVListSidebar'
import { ChatInterface } from './components/ChatInterface'
import { HRProfileView } from './components/HRProfileView'
import { JSONInspector } from './components/JSONInspector'
import { TraceInspector } from './components/TraceInspector'
import { CVListItem, CVDetail, ChatMessage, KeepaliveStatus, Citation } from './types'
import { UserCheck, Code2, Activity, UploadCloud, X, Zap } from 'lucide-react'

export function App() {
  const [cvs, setCvs] = useState<CVListItem[]>([])
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [selectedDoc, setSelectedDoc] = useState<CVDetail | null>(null)
  const [rightTab, setRightTab] = useState<'hr' | 'json' | 'traces'>('hr')
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [keepalive, setKeepalive] = useState<KeepaliveStatus | null>(null)
  const [loadingKeepalive, setLoadingKeepalive] = useState(false)
  const [loadingCvs, setLoadingCvs] = useState(false)

  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

  // Fetch Keepalive
  const fetchKeepalive = useCallback(async () => {
    try {
      setLoadingKeepalive(true)
      const res = await fetch(`${apiBase}/api/v1/keepalive`)
      if (!res.ok) throw new Error(`HTTP error ${res.status}`)
      const data = await res.json()
      setKeepalive(data)
    } catch {
      setKeepalive(null)
    } finally {
      setLoadingKeepalive(false)
    }
  }, [apiBase])

  // Fetch CV List
  const fetchCvs = useCallback(async () => {
    try {
      setLoadingCvs(true)
      const res = await fetch(`${apiBase}/api/v1/cvs`)
      if (!res.ok) throw new Error(`HTTP error ${res.status}`)
      const data = await res.json()
      setCvs(data)
      // Auto-select first doc if none selected
      if (!selectedDocId && data.length > 0) {
        setSelectedDocId(data[0].id)
      }
    } catch {
      // Ignored if offline
    } finally {
      setLoadingCvs(false)
    }
  }, [apiBase, selectedDocId])

  // Fetch Single CV Detail
  const fetchCvDetail = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${apiBase}/api/v1/cvs/${id}`)
      if (!res.ok) throw new Error(`HTTP error ${res.status}`)
      const data: CVDetail = await res.json()
      setSelectedDoc(data)
    } catch {
      setSelectedDoc(null)
    }
  }, [apiBase])

  useEffect(() => {
    fetchKeepalive()
    fetchCvs()
    const interval = setInterval(fetchKeepalive, 30000)
    return () => clearInterval(interval)
  }, [fetchKeepalive, fetchCvs])

  useEffect(() => {
    if (selectedDocId) {
      fetchCvDetail(selectedDocId)
    } else {
      setSelectedDoc(null)
    }
  }, [selectedDocId, fetchCvDetail])

  // Handle Multi-file Upload
  const handleUpload = async (files: File[]) => {
    setIsUploading(true)
    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)

        const res = await fetch(`${apiBase}/api/v1/cvs/upload`, {
          method: 'POST',
          body: formData,
        })

        if (!res.ok) {
          const errJson = await res.json()
          throw new Error(errJson.detail || `Upload failed for ${file.name}`)
        }

        const data = await res.json()
        setSelectedDocId(data.document_id)
      }

      await fetchCvs()
      setShowUploadModal(false)
    } finally {
      setIsUploading(false)
    }
  }

  // Handle Delete CV
  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!window.confirm('Delete this CV and all its indexed embeddings?')) return
    try {
      await fetch(`${apiBase}/api/v1/cvs/${id}`, { method: 'DELETE' })
      if (selectedDocId === id) {
        setSelectedDocId(null)
        setSelectedDoc(null)
      }
      await fetchCvs()
    } catch (err: any) {
      alert(`Delete error: ${err.message}`)
    }
  }

  // Handle SSE Chat
  const handleSendMessage = async (query: string) => {
    if (!query.trim() || isStreaming) return

    const userMessageId = `user-${Date.now()}`
    const assistantMessageId = `assistant-${Date.now()}`

    const userMsg: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    }

    const assistantMsg: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      citations: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setIsStreaming(true)

    try {
      const response = await fetch(`${apiBase}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          document_id: selectedDocId || undefined,
          top_k: 4,
          chat_history: messages.map((m) => ({ role: m.role, content: m.content })),
        }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`Chat request failed with status ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullContent = ''
      let streamCitations: Citation[] = []

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const block of lines) {
          if (!block.trim()) continue
          const blockLines = block.split('\n')
          let eventType = 'message'
          let eventData = ''

          for (const line of blockLines) {
            if (line.startsWith('event: ')) {
              eventType = line.replace('event: ', '').trim()
            } else if (line.startsWith('data: ')) {
              eventData = line.replace('data: ', '').trim()
            }
          }

          if (eventType === 'citations' && eventData) {
            try {
              const parsed = JSON.parse(eventData)
              streamCitations = parsed.citations || []
            } catch {}
          } else if (eventType === 'token' && eventData) {
            try {
              const parsed = JSON.parse(eventData)
              fullContent += parsed.token || ''
            } catch {}
          }

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? {
                    ...msg,
                    content: fullContent,
                    citations: streamCitations,
                    isStreaming: true,
                  }
                : msg
            )
          )
        }
      }

      // Finalize
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: fullContent || 'Unable to generate response from context.',
                citations: streamCitations,
                isStreaming: false,
              }
            : msg
        )
      )
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: `Error: ${err.message || 'Failed to stream response.'}`,
                isStreaming: false,
              }
            : msg
        )
      )
    } finally {
      setIsStreaming(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-emerald-500 selection:text-white">
      {/* Top Header */}
      <Header keepalive={keepalive} loading={loadingKeepalive} onPing={fetchKeepalive} />

      {/* Main Workspace 3-Pane Layout */}
      <main className="flex-1 p-4 grid grid-cols-1 md:grid-cols-12 gap-4 max-w-[1680px] w-full mx-auto">
        {/* Pane 1: Left CV List & Ingestion (Cols 1-3) */}
        <div className="md:col-span-3 flex flex-col space-y-3">
          {/* Ingestion Trigger Action */}
          <button
            onClick={() => setShowUploadModal(true)}
            className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-xs flex items-center justify-center space-x-2 shadow-lg shadow-emerald-600/20 transition-all font-mono"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Ingest New CV (&le; 5.0s SLA)</span>
          </button>

          <CVListSidebar
            cvs={cvs}
            selectedId={selectedDocId}
            onSelect={setSelectedDocId}
            onDelete={handleDelete}
            loading={loadingCvs}
          />
        </div>

        {/* Pane 2: Center Interactive SSE Chat (Cols 4-8) */}
        <div className="md:col-span-5 flex flex-col">
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isStreaming={isStreaming}
            selectedDocName={selectedDoc?.filename}
          />
        </div>

        {/* Pane 3: Right Inspector Panels: HR View, JSON Inspector, SLA Traces (Cols 9-12) */}
        <div className="md:col-span-4 flex flex-col glass-panel rounded-2xl border border-slate-800 overflow-hidden h-[calc(100vh-6.5rem)]">
          {/* Tab Navigation */}
          <div className="p-2 border-b border-slate-800 bg-slate-900/60 flex items-center space-x-1 text-xs font-mono">
            <button
              onClick={() => setRightTab('hr')}
              className={`flex-1 py-1.5 px-2.5 rounded-lg flex items-center justify-center space-x-1.5 transition-colors ${
                rightTab === 'hr'
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <UserCheck className="w-3.5 h-3.5" />
              <span>HR Profile</span>
            </button>

            <button
              onClick={() => setRightTab('json')}
              className={`flex-1 py-1.5 px-2.5 rounded-lg flex items-center justify-center space-x-1.5 transition-colors ${
                rightTab === 'json'
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Raw JSON</span>
            </button>

            <button
              onClick={() => setRightTab('traces')}
              className={`flex-1 py-1.5 px-2.5 rounded-lg flex items-center justify-center space-x-1.5 transition-colors ${
                rightTab === 'traces'
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>SLA Traces</span>
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {!selectedDoc ? (
              <div className="h-full flex flex-col items-center justify-center text-center text-xs text-slate-500 p-6 space-y-2">
                <Zap className="w-8 h-8 text-slate-700" />
                <p>Select an ingested CV from the left sidebar to inspect details.</p>
              </div>
            ) : rightTab === 'hr' ? (
              <HRProfileView parsedData={selectedDoc.parsed_json} />
            ) : rightTab === 'json' ? (
              <JSONInspector data={selectedDoc.parsed_json} />
            ) : (
              <TraceInspector
                traces={selectedDoc.traces || []}
                totalDurationMs={selectedDoc.total_duration_ms || 0}
              />
            )}
          </div>
        </div>
      </main>

      {/* Upload Dropzone Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel w-full max-w-lg rounded-3xl border border-slate-800 p-6 space-y-4 relative shadow-2xl">
            <button
              onClick={() => !isUploading && setShowUploadModal(false)}
              className="absolute right-5 top-5 p-1.5 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="space-y-1">
              <h3 className="text-base font-bold text-white tracking-tight">Ingest CV Documents</h3>
              <p className="text-xs text-slate-400">
                Executes the warm-path 8-stage pipeline with strict &le; 5.0s SLA benchmarking.
              </p>
            </div>

            <Dropzone onUpload={handleUpload} isUploading={isUploading} />
          </div>
        </div>
      )}
    </div>
  )
}

export default App
