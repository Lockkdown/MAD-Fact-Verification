import { useState, useRef, useCallback } from 'react'

const INITIAL_STATE = {
  status: 'idle',
  rounds: [],
  currentRound: null,
  routing: null,
  judgeState: null,
  finalVerdict: null,
  plmResult: null,
  stats: null,
  error: null,
}

export function usePredict() {
  const [state, setState] = useState(INITIAL_STATE)
  const abortRef = useRef(null)

  const updateState = useCallback((updater) => {
    setState((prev) => (typeof updater === 'function' ? updater(prev) : { ...prev, ...updater }))
  }, [])

  const startPredict = useCallback(
    async (claim, evidence, config, useMock = true) => {
      if (abortRef.current) abortRef.current.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setState({ ...INITIAL_STATE, status: 'loading' })

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ claim, evidence, config, use_mock: useMock }),
          signal: controller.signal,
        })

        if (!res.ok) throw new Error(`HTTP ${res.status}`)

        updateState({ status: 'streaming' })

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const raw = line.slice(6).trim()
            if (!raw) continue
            try {
              const event = JSON.parse(raw)
              handleEvent(event, updateState)
            } catch {
              // ignore malformed lines
            }
          }
        }

        updateState({ status: 'done' })
      } catch (err) {
        if (err.name === 'AbortError') {
          updateState({ status: 'idle' })
        } else {
          updateState({ status: 'error', error: err.message })
        }
      }
    },
    [updateState]
  )

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    updateState({ status: 'idle' })
  }, [updateState])

  return { state, startPredict, cancel }
}

function handleEvent(event, updateState) {
  switch (event.type) {
    case 'plm_start':
      updateState((prev) => ({ ...prev, plmResult: { status: 'loading', model: event.model } }))
      break

    case 'plm_result':
      updateState((prev) => ({
        ...prev,
        plmResult: {
          status: 'done',
          label: event.label,
          confidence: event.confidence,
          probabilities: event.probabilities,
        },
      }))
      break

    case 'routing':
      updateState((prev) => ({
        ...prev,
        routing: {
          plmModel: event.plm_model,
          confidence: event.confidence,
          threshold: event.threshold,
          routedTo: event.routed_to,
        },
      }))
      break

    case 'round_start':
      updateState((prev) => ({
        ...prev,
        currentRound: { round: event.round, totalRounds: event.total_rounds, agents: [] },
      }))
      break

    case 'agent_thinking':
      updateState((prev) => {
        const round = prev.currentRound
        if (!round) return prev
        const agents = [
          ...round.agents.filter((a) => a.agentId !== event.agent_id),
          { agentId: event.agent_id, round: event.round, status: 'thinking' },
        ]
        return { ...prev, currentRound: { ...round, agents } }
      })
      break

    case 'agent_result':
      updateState((prev) => {
        const round = prev.currentRound
        if (!round) return prev
        const agents = round.agents.map((a) =>
          a.agentId === event.agent_id
            ? { ...a, status: 'done', verdict: event.verdict, reasoning: event.reasoning }
            : a
        )
        return { ...prev, currentRound: { ...round, agents } }
      })
      break

    case 'round_end':
      updateState((prev) => {
        const round = prev.currentRound
        if (!round) return prev
        const finishedRound = { ...round, isUnanimous: event.is_unanimous, finished: true }
        return {
          ...prev,
          rounds: [...prev.rounds, finishedRound],
          currentRound: null,
        }
      })
      break

    case 'judge_thinking':
      updateState((prev) => ({ ...prev, judgeState: { status: 'thinking' } }))
      break

    case 'judge_result':
      updateState((prev) => ({
        ...prev,
        judgeState: {
          status: 'done',
          verdict: event.verdict,
          reasoning: event.reasoning,
          roundsUsed: event.rounds_used,
        },
      }))
      break

    case 'final':
      updateState((prev) => ({
        ...prev,
        finalVerdict: event.label,
        stats: {
          roundsUsed: event.rounds_used,
          totalAgentCalls: event.total_agent_calls,
        },
      }))
      break

    case 'error':
      updateState((prev) => ({ ...prev, status: 'error', error: event.message }))
      break

    case 'done':
      break

    default:
      break
  }
}
