export type DocumentStatus = 'queued' | 'extracting' | 'indexing' | 'rag_ready' | 'failed'

export interface StageTraceItem {
  stage: string
  duration_ms: number
  status: string
  start_time: string
  end_time?: string
  metadata?: Record<string, any>
  error_message?: string
}

export interface CVListItem {
  id: string
  filename: string
  file_size: number
  content_type: string
  status: DocumentStatus
  total_duration_ms?: number
  created_at?: string
  updated_at?: string
}

export interface CVDetail {
  id: string
  filename: string
  file_size: number
  content_type: string
  status: DocumentStatus
  error_message?: string
  total_duration_ms?: number
  raw_text?: string
  parsed_json?: any
  chunks: Array<{
    id: string
    chunk_index: number
    section_name: string
    content: string
    token_count: number
    metadata?: any
  }>
  traces: StageTraceItem[]
  created_at?: string
  updated_at?: string
}

export interface Citation {
  chunk_id: string
  section_name: string
  similarity: number
  snippet: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  timestamp: string
  isStreaming?: boolean
}

export interface KeepaliveStatus {
  status: string
  timestamp: string
  uptime_seconds: number
  memory_usage_mb: number
  models_prewarmed: boolean
  sla_target_ms: number
  environment: string
  message: string
}
