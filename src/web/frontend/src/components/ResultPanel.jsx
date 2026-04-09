import AgentCard from './AgentCard'
import PLMResult from './PLMResult'
import VerdictBadge from './VerdictBadge'
import ThinkingAnimation from './ThinkingAnimation'
import { AGENTS } from '../constants/agents'

function RoutingBanner({ routing }) {
  if (!routing) return null
  const isSkipped = routing.routedTo === 'fast_path'
  return (
    <div
      className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm border ${
        isSkipped
          ? 'bg-green-50 border-green-200 text-green-700'
          : 'bg-blue-50 border-blue-200 text-blue-700'
      }`}
    >
      <span className="font-semibold">PhoBERT Routing:</span>
      <span>
        Độ tin cậy {(routing.confidence * 100).toFixed(1)}% (ngưỡng t*={routing.threshold})
      </span>
      <span className="ml-auto font-semibold">
        {isSkipped ? '→ Fast Path (bỏ qua debate)' : '→ Gửi vào Debate Council'}
      </span>
    </div>
  )
}

function RoundSection({ round }) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
          Round {round.round}
        </span>
        {round.finished && round.isUnanimous && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-100 text-green-700 font-medium">
            Nhất trí — dừng sớm
          </span>
        )}
        <div className="flex-1 h-px bg-gray-100" />
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {round.agents.map((agent) => (
          <AgentCard
            key={`${agent.agentId}-${round.round}`}
            agentId={agent.agentId}
            round={round.round}
            status={agent.status}
            verdict={agent.verdict}
            reasoning={agent.reasoning}
          />
        ))}
      </div>
    </div>
  )
}

export default function ResultPanel({ state, config }) {
  const { status, plmResult, routing, rounds, currentRound, judgeState, finalVerdict, stats, error } = state
  const isPlmMode = config === 'phobert'

  if (status === 'idle') return null

  return (
    <div className="space-y-5">
      {/* Error */}
      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
          Lỗi: {error}
        </div>
      )}

      {/* PLM mode */}
      {isPlmMode && plmResult && <PLMResult plmResult={plmResult} />}

      {/* Hybrid routing banner */}
      {!isPlmMode && routing && <RoutingBanner routing={routing} />}

      {/* Debate rounds — completed */}
      {!isPlmMode &&
        rounds.map((round) => (
          <RoundSection key={round.round} round={round} />
        ))}

      {/* Current in-progress round */}
      {!isPlmMode && currentRound && <RoundSection round={currentRound} />}

      {/* Judge */}
      {!isPlmMode && judgeState && (
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="h-px flex-1 bg-gray-200" />
            <span className="text-sm font-semibold text-gray-500 flex items-center gap-1">
              <span>{AGENTS.judge.icon}</span>
              <span>Phán quyết cuối cùng</span>
            </span>
            <div className="h-px flex-1 bg-gray-200" />
          </div>
          <AgentCard
            agentId="judge"
            status={judgeState.status}
            verdict={judgeState.verdict}
            reasoning={judgeState.reasoning}
            isJudge
          />
        </div>
      )}

      {/* Judge thinking (before judgeState has verdict) */}
      {!isPlmMode && !judgeState && status === 'streaming' && rounds.length > 0 && !currentRound && (
        <div className="flex items-center justify-center py-4">
          <ThinkingAnimation color={AGENTS.judge.color} isJudge />
        </div>
      )}

      {/* Final verdict + stats */}
      {finalVerdict && (
        <div className="pt-3 border-t border-gray-200 flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-700">Kết quả:</span>
            <VerdictBadge verdict={finalVerdict} size="lg" />
          </div>
          {stats && (
            <div className="flex items-center gap-3 text-xs text-gray-400 ml-auto">
              {stats.roundsUsed != null && stats.roundsUsed > 0 && (
                <span>Rounds dùng: <strong className="text-gray-600">{stats.roundsUsed}</strong></span>
              )}
              {stats.totalAgentCalls != null && stats.totalAgentCalls > 0 && (
                <span>Agent calls: <strong className="text-gray-600">{stats.totalAgentCalls}</strong></span>
              )}
              <span>Config: <strong className="text-gray-600">{config}</strong></span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
