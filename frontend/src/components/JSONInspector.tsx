import { useState } from 'react'
import { Copy, Check, Code2 } from 'lucide-react'
import JsonView from '@uiw/react-json-view'
import { darkTheme } from '@uiw/react-json-view/dark'

interface JSONInspectorProps {
  data: any
}

export function JSONInspector({ data }: JSONInspectorProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!data) return
    navigator.clipboard.writeText(JSON.stringify(data, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!data) {
    return (
      <div className="p-8 text-center text-xs text-slate-500 font-mono">
        No JSON extraction data available.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Header bar */}
      <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono">
        <div className="flex items-center space-x-2 text-slate-400">
          <Code2 className="w-4 h-4 text-emerald-400" />
          <span>CVExtractionSchema (Pydantic v2 JSON)</span>
        </div>

        <button
          onClick={handleCopy}
          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white flex items-center space-x-1.5 transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy JSON</span>
            </>
          )}
        </button>
      </div>

      {/* JSON Viewer Container */}
      <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 overflow-x-auto font-mono text-xs max-h-[550px] overflow-y-auto">
        <JsonView
          value={data}
          style={darkTheme}
          displayDataTypes={false}
          displayObjectSize={true}
          collapsed={2}
          enableClipboard={false}
        />
      </div>
    </div>
  )
}
