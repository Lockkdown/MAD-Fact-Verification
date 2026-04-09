export default function AnalyzeButton({ isStreaming, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all ${
        disabled && !isStreaming
          ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
          : isStreaming
          ? 'bg-red-100 text-red-700 border border-red-300 hover:bg-red-200'
          : 'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 shadow-sm'
      }`}
    >
      {isStreaming ? (
        <>
          <span>⏹</span>
          <span>Dừng lại</span>
        </>
      ) : (
        <>
          <span>🔍</span>
          <span>Phân tích</span>
        </>
      )}
    </button>
  )
}
