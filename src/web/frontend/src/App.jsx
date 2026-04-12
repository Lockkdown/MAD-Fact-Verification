import { useState } from 'react'
import InputSection from './components/InputSection'
import ConfigSelector from './components/ConfigSelector'
import AnalyzeButton from './components/AnalyzeButton'
import ResultPanel from './components/ResultPanel'
import { usePredict } from './hooks/usePredict'

export default function App() {
  const [claim, setClaim] = useState('')
  const [evidence, setEvidence] = useState('')
  const [config, setConfig] = useState('hybrid_n3k3')

  const { state, startPredict, cancel } = usePredict()

  const isStreaming = state.status === 'streaming' || state.status === 'loading'
  const canAnalyze = claim.trim().length > 0 && evidence.trim().length > 0
  const showResult = state.status !== 'idle'

  function handleAnalyze() {
    if (isStreaming) {
      cancel()
      return
    }
    startPredict(claim, evidence, config)
  }

  return (
    <div className="min-h-screen bg-[#F8F9FA]">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
              <span className="text-white text-sm font-bold">V</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900 leading-none">ViMAD Demo</h1>
              <p className="text-xs text-gray-500 mt-0.5">Vietnamese Multi-Agent Debate · Fact Checking</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 py-6 space-y-4">
        {/* Input Card */}
        <section className="bg-white rounded-xl border border-gray-200 p-5">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Nhập dữ liệu</h2>
          <InputSection
            claim={claim}
            evidence={evidence}
            onClaimChange={setClaim}
            onEvidenceChange={setEvidence}
          />
        </section>

        {/* Config Card */}
        <section className="bg-white rounded-xl border border-gray-200 p-5">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Cấu hình</h2>
          <ConfigSelector selected={config} onChange={setConfig} />
        </section>

        {/* Analyze Button */}
        <AnalyzeButton
          isStreaming={isStreaming}
          onClick={handleAnalyze}
          disabled={!isStreaming && !canAnalyze}
        />

        {/* Result Card */}
        {showResult && (
          <section className="bg-white rounded-xl border border-gray-200 p-5">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Kết quả phân tích</h2>
            <ResultPanel state={state} config={config} />
          </section>
        )}
      </main>
    </div>
  )
}
