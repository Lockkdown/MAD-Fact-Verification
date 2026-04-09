import { useState } from 'react'
import { AGENTS } from '../constants/agents'
import VerdictBadge from './VerdictBadge'
import ThinkingAnimation from './ThinkingAnimation'

export default function AgentCard({ agentId, round, status, verdict, reasoning, isJudge = false }) {
  const [expanded, setExpanded] = useState(false)
  const agent = AGENTS[agentId] ?? AGENTS.judge

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{
        borderColor: agent.borderColor,
        borderLeftWidth: '4px',
        borderLeftColor: agent.color,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2.5"
        style={{ backgroundColor: agent.bgColor }}
      >
        <div className="flex items-center gap-2">
          <span className="text-base">{agent.icon}</span>
          <span className="text-sm font-semibold" style={{ color: agent.color }}>
            {agent.name}
          </span>
          <span
            className="text-[10px] font-medium px-1.5 py-0.5 rounded-full border"
            style={{
              color: agent.color,
              borderColor: agent.borderColor,
              backgroundColor: 'white',
            }}
          >
            {agent.ecosystem}
          </span>
        </div>
        {round != null && (
          <span className="text-xs text-gray-400">
            {isJudge ? 'Final Judge' : `Round ${round}`}
          </span>
        )}
      </div>

      {/* Body */}
      <div className="px-4 py-3 bg-white">
        {status === 'thinking' ? (
          <ThinkingAnimation color={agent.color} isJudge={isJudge} />
        ) : (
          <>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-gray-500 font-medium">Phán quyết:</span>
              <VerdictBadge verdict={verdict} size="sm" />
            </div>

            {reasoning && (
              <div>
                <button
                  onClick={() => setExpanded((v) => !v)}
                  className="flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-gray-700 transition-colors"
                >
                  <span>{expanded ? '▲' : '▼'}</span>
                  <span>Lý luận</span>
                </button>
                {expanded && (
                  <div className="mt-2 max-h-40 overflow-y-auto">
                    <p className="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap">
                      {reasoning}
                    </p>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
