const FULL_OPTIONS = [
  { value: 'full_n2k3', n: 2, k: 3 },
  { value: 'full_n2k5', n: 2, k: 5 },
  { value: 'full_n3k3', n: 3, k: 3 },
  { value: 'full_n3k5', n: 3, k: 5 },
  { value: 'full_n4k3', n: 4, k: 3 },
  { value: 'full_n4k5', n: 4, k: 5 },
]

const HYBRID_OPTIONS = [
  { value: 'hybrid_n2k3', n: 2, k: 3 },
  { value: 'hybrid_n2k5', n: 2, k: 5 },
  { value: 'hybrid_n3k3', n: 3, k: 3 },
  { value: 'hybrid_n3k5', n: 3, k: 5 },
  { value: 'hybrid_n4k3', n: 4, k: 3 },
  { value: 'hybrid_n4k5', n: 4, k: 5 },
]

function DebateOption({ option, selected, onChange, showThreshold = false }) {
  const isSelected = selected === option.value
  return (
    <label
      className={`relative cursor-pointer rounded-lg border p-3 transition-all ${
        isSelected
          ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
    >
      <input
        type="radio"
        name="config"
        value={option.value}
        checked={isSelected}
        onChange={() => onChange(option.value)}
        className="sr-only"
      />
      {showThreshold && (
        <span className="absolute top-1.5 right-1.5 text-[10px] font-medium bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full">
          t*=0.85
        </span>
      )}
      <div className={`text-sm font-semibold ${isSelected ? 'text-blue-700' : 'text-gray-800'}`}>
        N={option.n}, k={option.k}
      </div>
      <div className="text-xs text-gray-500 mt-0.5">
        {option.n} debaters · {option.k} rounds
      </div>
    </label>
  )
}

export default function ConfigSelector({ selected, onChange }) {
  return (
    <div className="space-y-5">
      {/* Group 1 — PLM Standalone */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
          PLM Standalone
        </p>
        <label
          className={`flex items-center gap-3 cursor-pointer rounded-lg border p-3 transition-all ${
            selected === 'phobert'
              ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500'
              : 'border-gray-200 bg-white hover:border-gray-300'
          }`}
        >
          <input
            type="radio"
            name="config"
            value="phobert"
            checked={selected === 'phobert'}
            onChange={() => onChange('phobert')}
            className="sr-only"
          />
          <div>
            <span className={`text-sm font-semibold ${selected === 'phobert' ? 'text-blue-700' : 'text-gray-800'}`}>
              PhoBERT ★
            </span>
            <p className="text-xs text-gray-500 mt-0.5">★ Routing Gate (M*) — F1: 85.08%</p>
          </div>
        </label>
      </div>

      {/* Group 2 — Full Debate */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
          Full Debate
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {FULL_OPTIONS.map((opt) => (
            <DebateOption key={opt.value} option={opt} selected={selected} onChange={onChange} />
          ))}
        </div>
      </div>

      {/* Group 3 — Hybrid Debate */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
          Hybrid Debate
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {HYBRID_OPTIONS.map((opt) => (
            <DebateOption
              key={opt.value}
              option={opt}
              selected={selected}
              onChange={onChange}
              showThreshold
            />
          ))}
        </div>
      </div>
    </div>
  )
}
