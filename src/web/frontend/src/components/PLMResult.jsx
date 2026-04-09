import VerdictBadge from './VerdictBadge'

const LABEL_ORDER = ['Support', 'Refute', 'NEI']
const BAR_COLORS = {
  Support: '#22C55E',
  Refute: '#EF4444',
  NEI: '#F59E0B',
}

export default function PLMResult({ plmResult }) {
  if (!plmResult) return null

  if (plmResult.status === 'loading') {
    return (
      <div className="flex items-center gap-3 py-4">
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-gray-500">PhoBERT đang phân tích...</span>
      </div>
    )
  }

  const { label, confidence, probabilities } = plmResult

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-gray-700">Kết quả PLM:</span>
        <VerdictBadge verdict={label} size="lg" />
        <span className="text-sm text-gray-500">
          Độ tin cậy: <strong>{(confidence * 100).toFixed(1)}%</strong>
        </span>
      </div>

      {probabilities && (
        <div className="space-y-2">
          {LABEL_ORDER.map((lbl) => {
            const pct = ((probabilities[lbl] ?? 0) * 100).toFixed(1)
            return (
              <div key={lbl} className="flex items-center gap-3">
                <span className="w-14 text-xs font-medium text-gray-600 text-right">{lbl}</span>
                <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: BAR_COLORS[lbl],
                    }}
                  />
                </div>
                <span className="w-10 text-xs text-gray-500 text-right">{pct}%</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
