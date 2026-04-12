"""Debate service — real full panel debate via OpenRouter + PLM routing."""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

from prompts.debater_prompt import build_debater_prompt
from prompts.judge_prompt import build_judge_prompt
from services.llm_client import call_openrouter, parse_r1, parse_verdict_reasoning, make_fallback_reasoning
from services.plm_service import plm_service

logger = logging.getLogger(__name__)

AGENT_MODELS = {
    "mistral": "mistralai/mistral-small-2603",
    "gpt4o_mini": "openai/gpt-4o-mini",
    "qwen": "qwen/qwen-2.5-72b-instruct",
    "llama": "meta-llama/llama-3.3-70b-instruct",
}
JUDGE_MODEL = "deepseek/deepseek-chat"
THRESHOLD = 0.85

CONFIG_TO_AGENTS: dict[str, list[str]] = {
    "n2": ["mistral", "gpt4o_mini"],
    "n3": ["mistral", "gpt4o_mini", "qwen"],
    "n4": ["mistral", "gpt4o_mini", "qwen", "llama"],
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_config(config: str) -> tuple[str, int, int]:
    """Return (mode, n_debaters, k_rounds) from config string."""
    parts = config.split("_")
    mode = parts[0]
    nk = parts[1]
    n = int(nk[1])
    k = int(nk[3:])
    return mode, n, k


class DebateService:
    """Real full panel debate with OpenRouter + PLM routing."""

    async def run_plm(self, claim: str, evidence: str) -> AsyncGenerator[str, None]:
        """Stream PLM prediction as SSE events."""
        yield _sse({"type": "plm_start", "model": "phobert"})
        try:
            result = plm_service.predict(claim, evidence)
            yield _sse({
                "type": "plm_result",
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            })
        except Exception as exc:
            logger.error("PLM prediction failed: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "done"})

    async def run_debate(
        self, claim: str, evidence: str, config: str
    ) -> AsyncGenerator[str, None]:
        """Real debate with full panel models via OpenRouter."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            yield _sse({
                "type": "error",
                "message": "OPENROUTER_API_KEY chưa được cấu hình. Vui lòng tạo file backend/.env từ .env.example",
            })
            yield _sse({"type": "done"})
            return

        mode, n, k = _parse_config(config)
        agent_ids = CONFIG_TO_AGENTS.get(f"n{n}", ["mistral", "gpt4o_mini"])
        is_hybrid = mode == "hybrid"
        rounds_used = k

        plm_result: dict | None = None
        if is_hybrid:
            try:
                plm_result = plm_service.predict(claim, evidence)
                confidence = plm_result["confidence"]
                routed_to = "fast_path" if confidence >= THRESHOLD else "debate"
            except Exception as exc:
                logger.warning("PLM routing failed, defaulting to debate: %s", exc)
                confidence = 0.5
                routed_to = "debate"

            yield _sse({
                "type": "routing",
                "plm_model": "phobert",
                "confidence": confidence,
                "threshold": THRESHOLD,
                "routed_to": routed_to,
            })
            await asyncio.sleep(0.3)

            if routed_to == "fast_path":
                yield _sse({"type": "final", "label": plm_result["label"], "rounds_used": 0, "total_agent_calls": 1})
                yield _sse({"type": "done"})
                return

        all_rounds: list[dict] = []

        for round_num in range(1, k + 1):
            yield _sse({"type": "round_start", "round": round_num, "total_rounds": k})

            prompts = []
            for agent_id in agent_ids:
                messages = build_debater_prompt(
                    statement=claim,
                    evidence=evidence,
                    round_num=round_num,
                    prior_rounds=all_rounds,
                    debater_id=agent_id,
                )
                prompts.append(messages)

            for agent_id in agent_ids:
                yield _sse({"type": "agent_thinking", "agent_id": agent_id, "round": round_num})

            responses = await asyncio.gather(*[
                call_openrouter(msg, AGENT_MODELS[agent_id])
                for agent_id, msg in zip(agent_ids, prompts)
            ])

            round_verdicts: list[dict] = []
            for agent_id, response_text in zip(agent_ids, responses):
                if not response_text:
                    verdict = plm_result["label"] if (is_hybrid and plm_result) else "NEI"
                    reasoning = make_fallback_reasoning(agent_id, evidence, verdict)
                elif round_num == 1:
                    verdict, reasoning = parse_r1(response_text)
                else:
                    verdict, reasoning = parse_verdict_reasoning(response_text)

                round_verdicts.append({"agent_id": agent_id, "verdict": verdict, "reasoning": reasoning})
                yield _sse({
                    "type": "agent_result",
                    "agent_id": agent_id,
                    "round": round_num,
                    "verdict": verdict,
                    "reasoning": reasoning,
                })

            all_rounds.append({"round": round_num, "verdicts": round_verdicts})
            is_unanimous = len({v["verdict"] for v in round_verdicts}) == 1
            yield _sse({"type": "round_end", "round": round_num, "is_unanimous": is_unanimous})

            if is_unanimous and round_num < k:
                rounds_used = round_num
                break

        yield _sse({"type": "judge_thinking"})
        all_verdicts_flat = [v["verdict"] for r in all_rounds for v in r["verdicts"]]
        is_unanimous_final = len(set(all_verdicts_flat)) == 1
        consensus = all_rounds[-1]["verdicts"][0]["verdict"] if is_unanimous_final else None

        judge_messages = build_judge_prompt(
            statement=claim,
            evidence=evidence,
            all_rounds=all_rounds,
            is_unanimous=is_unanimous_final,
            consensus_verdict=consensus,
        )
        judge_response = await call_openrouter(judge_messages, JUDGE_MODEL)

        if not judge_response:
            final_verdict = max(set(all_verdicts_flat), key=all_verdicts_flat.count)
            judge_reasoning = make_fallback_reasoning("judge", evidence, final_verdict)
        else:
            final_verdict, judge_reasoning = parse_verdict_reasoning(judge_response)

        total_calls = sum(len(r["verdicts"]) for r in all_rounds) + 1
        yield _sse({
            "type": "judge_result",
            "verdict": final_verdict,
            "reasoning": judge_reasoning,
            "rounds_used": rounds_used,
        })
        yield _sse({
            "type": "final",
            "label": final_verdict,
            "rounds_used": rounds_used,
            "total_agent_calls": total_calls,
        })
        yield _sse({"type": "done"})


debate_service = DebateService()
