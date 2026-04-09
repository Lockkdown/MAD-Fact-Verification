export default function ThinkingAnimation({ color = '#6B7280', isJudge = false }) {
  const label = isJudge ? 'Đang tổng hợp phán quyết...' : 'Đang phân tích...'

  return (
    <div className="flex items-center gap-2 py-2">
      <div className="flex items-end gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="agent-thinking-dot block w-2 h-2 rounded-full"
            style={{
              backgroundColor: color,
              animationDelay: `${i * 0.2}s`,
            }}
          />
        ))}
      </div>
      <span className="text-sm font-medium" style={{ color }}>
        {label}
      </span>
    </div>
  )
}
