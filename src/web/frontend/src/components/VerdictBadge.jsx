const VERDICT_STYLES = {
  Support: {
    bg: '#DCFCE7',
    text: '#166534',
    border: '#BBF7D0',
    icon: '✓',
  },
  Refute: {
    bg: '#FEE2E2',
    text: '#991B1B',
    border: '#FECACA',
    icon: '✗',
  },
  NEI: {
    bg: '#FEF3C7',
    text: '#92400E',
    border: '#FDE68A',
    icon: '?',
  },
}

export default function VerdictBadge({ verdict, size = 'sm' }) {
  if (!verdict) return null
  const style = VERDICT_STYLES[verdict] ?? VERDICT_STYLES.NEI
  const isLarge = size === 'lg'

  return (
    <span
      style={{
        backgroundColor: style.bg,
        color: style.text,
        borderColor: style.border,
      }}
      className={`inline-flex items-center gap-1 border rounded-full font-semibold ${
        isLarge
          ? 'px-4 py-1.5 text-base'
          : 'px-2.5 py-0.5 text-xs'
      }`}
    >
      <span>{style.icon}</span>
      <span>{verdict}</span>
    </span>
  )
}
