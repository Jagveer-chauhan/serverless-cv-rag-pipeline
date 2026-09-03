import React, { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Sparkles, BookOpen, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'
import { ChatMessage, Citation } from '../types'

interface ChatInterfaceProps {
  messages: ChatMessage[]
  onSendMessage: (query: string) => Promise<void>
  isStreaming: boolean
  selectedDocName?: string
}

const QUICK_PROMPTS = [
  'Summarize candidate technical experience',
  'What are the candidate top programming skills & tools?',
  'List all degrees, qualifications, and certifications',
  'What leadership or architect roles did they hold?',
]

function FormattedContent({ content }: { content: string }) {
  if (!content) return null

  // Split lines to format markdown headers, bullet points, bolding
  const lines = content.split('\n')

  return (
    <div className="space-y-1 text-xs leading-relaxed">
      {lines.map((line, idx) => {
        const trimmed = line.trim()
        if (!trimmed) {
          return <div key={idx} className="h-1.5" />
        }

        // Heading 3
        if (trimmed.startsWith('### ')) {
          return (
            <h4 key={idx} className="font-semibold text-emerald-400 text-xs mt-2 pt-1 border-b border-slate-800/40 pb-0.5">
              {trimmed.replace(/^###\s*/, '')}
            </h4>
          )
        }

        // Bullet point
        if (trimmed.startsWith('- ') || trimmed.startsWith('• ') || trimmed.startsWith('* ')) {
          const itemText = trimmed.replace(/^[-•*]\s*/, '')
          return (
            <div key={idx} className="flex items-start space-x-2 pl-1.5 text-slate-300">
              <span className="text-emerald-500 font-bold text-xs mt-0.5">•</span>
              <span className="flex-1">{renderBoldText(itemText)}</span>
            </div>
          )
        }

        // Standard line
        return (
          <p key={idx} className="text-slate-200">
            {renderBoldText(trimmed)}
          </p>
        )
      })}
    </div>
  )
}

function renderBoldText(text: string): React.ReactNode {
  const parts = text.split(/(\*\*.*?\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={i} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      )
    }
    return part
  })
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const [showCitations, setShowCitations] = useState(false)

  const isUser = msg.role === 'user'

  return (
    <div className={`flex items-start space-x-3 text-xs ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 flex-shrink-0 mt-0.5">
          <Bot className="w-4 h-4" />
        </div>
      )}

      <div
        className={`max-w-[85%] rounded-2xl p-3.5 space-y-2.5 ${
          isUser
            ? 'bg-emerald-600 text-white rounded-tr-none'
            : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none shadow-sm'
        }`}
      >
        {/* Message Content */}
        {isUser ? (
          <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
        ) : (
          <div>
            <FormattedContent content={msg.content} />
            {msg.isStreaming && (
              <span className="inline-block w-1.5 h-3 ml-1 bg-emerald-400 animate-pulse align-middle" />
            )}
          </div>
        )}

        {/* Citations section if available (Compact collapsible UI) */}
        {!isUser && msg.citations && msg.citations.length > 0 && (
          <div className="pt-2 border-t border-slate-800/60 space-y-1.5">
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-850 hover:bg-slate-800 border border-slate-750 text-[11px] text-slate-400 hover:text-emerald-400 transition-colors font-mono"
            >
              <BookOpen className="w-3 h-3 text-emerald-400" />
              <span>{msg.citations.length} Source{msg.citations.length > 1 ? 's' : ''} Cited</span>
              {showCitations ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showCitations && (
              <div className="space-y-1.5 pt-1">
                {msg.citations.map((c: Citation, cIdx: number) => (
                  <div
                    key={cIdx}
                    className="p-2 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] space-y-1"
                  >
                    <div className="flex items-center justify-between text-slate-300 font-mono">
                      <span className="font-semibold text-emerald-400">[{c.section_name}]</span>
                      <span className="text-slate-500">{(c.similarity * 100).toFixed(0)}% relevance</span>
                    </div>
                    <p className="text-slate-400 leading-relaxed font-sans text-[11px]">
                      {c.snippet}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 flex-shrink-0 mt-0.5">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  )
}

export function ChatInterface({
  messages,
  onSendMessage,
  isStreaming,
  selectedDocName,
}: ChatInterfaceProps) {
  const [inputQuery, setInputQuery] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isStreaming])

  const handleSend = async () => {
    if (!inputQuery.trim() || isStreaming) return
    const query = inputQuery.trim()
    setInputQuery('')
    await onSendMessage(query)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex-1 flex flex-col glass-panel rounded-2xl border border-slate-800 overflow-hidden h-[calc(100vh-6.5rem)]">
      {/* Chat Header */}
      <div className="px-4 py-3 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-semibold text-white font-mono flex items-center gap-1.5">
              RAG Candidate Intelligence Chat
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 font-normal">
                SSE Live
              </span>
            </h3>
            <p className="text-[11px] text-slate-400 truncate max-w-sm">
              {selectedDocName ? `Scoped to: ${selectedDocName}` : 'Querying all ingested CVs with pgvector'}
            </p>
          </div>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400">
              <Sparkles className="w-6 h-6" />
            </div>
            <div className="max-w-md space-y-1">
              <h4 className="text-sm font-semibold text-white">Ask anything about the candidate</h4>
              <p className="text-xs text-slate-400">
                Vector similarity retrieves relevant resume chunks and streams answers with citations.
              </p>
            </div>

            {/* Quick Prompt Chips */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full max-w-lg mt-2">
              {QUICK_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(prompt)}
                  className="p-2.5 rounded-xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 text-left text-xs text-slate-300 hover:text-white transition-colors"
                >
                  &ldquo;{prompt}&rdquo;
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-3 border-t border-slate-800 bg-slate-900/50">
        <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 focus-within:border-emerald-500/60 transition-colors">
          <input
            type="text"
            placeholder={
              isStreaming
                ? 'Streaming response...'
                : selectedDocName
                ? `Ask about ${selectedDocName}...`
                : 'Ask a question about candidates...'
            }
            value={inputQuery}
            disabled={isStreaming}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent text-xs text-slate-200 placeholder-slate-500 focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={isStreaming || !inputQuery.trim()}
            className="p-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isStreaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  )
}
