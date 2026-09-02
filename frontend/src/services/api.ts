/**
 * Frontend API Service Layer.
 */
import { CVListItem, CVDetail, KeepaliveStatus, Citation, ChatMessage } from '../types'

// Strictly sourced from environment variable VITE_API_BASE_URL
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

export const apiService = {
  async getKeepalive(): Promise<KeepaliveStatus> {
    const res = await fetch(`${API_BASE}/api/v1/keepalive`)
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to get keepalive status`)
    return res.json()
  },

  async getCVList(): Promise<CVListItem[]> {
    const res = await fetch(`${API_BASE}/api/v1/cvs`)
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch CVs`)
    return res.json()
  },

  async getCVDetail(id: string): Promise<CVDetail> {
    const res = await fetch(`${API_BASE}/api/v1/cvs/${id}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch CV detail`)
    return res.json()
  },

  async uploadCV(file: File): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)
    const res = await fetch(`${API_BASE}/api/v1/cvs/upload`, {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `Upload failed for ${file.name}`)
    }
    return res.json()
  },

  async deleteCV(id: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/v1/cvs/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to delete CV`)
  },

  async streamChat(
    query: string,
    documentId: string | null,
    history: ChatMessage[],
    callbacks: {
      onCitations: (citations: Citation[]) => void
      onToken: (token: string) => void
      onDone: () => void
      onError: (err: Error) => void
    }
  ): Promise<void> {
    try {
      const response = await fetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          document_id: documentId || undefined,
          top_k: 4,
          chat_history: history.map((m) => ({ role: m.role, content: m.content })),
        }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`Chat stream failed with status ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''

        for (const block of blocks) {
          if (!block.trim()) continue
          const lines = block.split('\n')
          let eventType = 'message'
          let eventData = ''

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.replace('event: ', '').trim()
            } else if (line.startsWith('data: ')) {
              eventData = line.replace('data: ', '').trim()
            }
          }

          if (eventType === 'citations' && eventData) {
            try {
              const parsed = JSON.parse(eventData)
              callbacks.onCitations(parsed.citations || [])
            } catch {}
          } else if (eventType === 'token' && eventData) {
            try {
              const parsed = JSON.parse(eventData)
              callbacks.onToken(parsed.token || '')
            } catch {}
          }
        }
      }

      callbacks.onDone()
    } catch (err: any) {
      callbacks.onError(err)
    }
  },
}
