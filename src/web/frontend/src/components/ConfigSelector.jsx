import { useEffect, useRef, useState } from 'react'

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

function getLabel(value) {
  if (value === 'phobert') return '● PhoBERT ★ · PLM Standalone'
  const match = value.match(/^(full|hybrid)_n(\d)k(\d)$/)
  if (!match) return value
  const [, mode, n, k] = match
  const group = mode === 'full' ? 'Full Debate' : 'Hybrid Debate'
  const threshold = mode === 'hybrid' ? ' · t*=0.85' : ''
  return `N=${n}, k=${k} · ${group}${threshold}`
}

function SectionHeader({ label }) {
  return (
    <div className="flex items-center gap-2 px-3 pt-2 pb-1">
      <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
        {label}
      </span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  )
}

function DropdownItem({ value, label, subtitle, isSelected, isHybrid, isDefault, onClick }) {
  return (
    <button
      type="button"
      onClick={() => onClick(value)}
      className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-gray-100 ${
        isSelected ? 'bg-blue-50' : ''
      }`}
    >
      <span className={`w-4 text-sm flex-shrink-0 ${isSelected ? 'text-blue-600' : 'text-transparent'}`}>
        ✓
      </span>
      <span className="flex-1 min-w-0 flex items-baseline gap-3">
        <span className={`text-sm ${isSelected ? 'font-semibold text-blue-700' : 'font-medium text-gray-800'}`}>
          {label}
        </span>
        {subtitle && (
          <span className="text-xs text-gray-400">{subtitle}</span>
        )}
      </span>
      {isHybrid && (
        <span className="flex-shrink-0 text-[10px] font-medium bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full">
          t*=0.85
        </span>
      )}
      {isDefault && (
        <span className="flex-shrink-0 text-[10px] text-gray-400 ml-2">
          ← default
        </span>
      )}
    </button>
  )
}

export default function ConfigSelector({ selected, onChange }) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef(null)

  useEffect(() => {
    function handleMouseDown(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleMouseDown)
    return () => document.removeEventListener('mousedown', handleMouseDown)
  }, [])

  function handleSelect(value) {
    onChange(value)
    setIsOpen(false)
  }

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2.5 bg-white border border-gray-200 rounded-xl text-sm text-gray-800 hover:border-gray-300 transition-colors shadow-sm"
      >
        <span className="text-gray-400">⚙</span>
        <span className="flex-1 text-left font-medium truncate">{getLabel(selected)}</span>
        <span className={`text-gray-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}>
          ▾
        </span>
      </button>

      {/* Dropdown panel */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden">
          <div className="max-h-80 overflow-y-auto py-1">

            <SectionHeader label="PLM Standalone" />
            <DropdownItem
              value="phobert"
              label="PhoBERT ★"
              subtitle="Routing Gate (M*) — F1: 85.08%"
              isSelected={selected === 'phobert'}
              isHybrid={false}
              isDefault={false}
              onClick={handleSelect}
            />

            <SectionHeader label="Full Debate" />
            {FULL_OPTIONS.map((opt) => (
              <DropdownItem
                key={opt.value}
                value={opt.value}
                label={`N=${opt.n}, k=${opt.k}`}
                subtitle={`${opt.n} debaters · ${opt.k} rounds`}
                isSelected={selected === opt.value}
                isHybrid={false}
                isDefault={false}
                onClick={handleSelect}
              />
            ))}

            <SectionHeader label="Hybrid Debate" />
            {HYBRID_OPTIONS.map((opt) => (
              <DropdownItem
                key={opt.value}
                value={opt.value}
                label={`N=${opt.n}, k=${opt.k}`}
                subtitle={`${opt.n} debaters · ${opt.k} rounds`}
                isSelected={selected === opt.value}
                isHybrid
                isDefault={opt.value === 'hybrid_n3k3'}
                onClick={handleSelect}
              />
            ))}

          </div>
        </div>
      )}
    </div>
  )
}
