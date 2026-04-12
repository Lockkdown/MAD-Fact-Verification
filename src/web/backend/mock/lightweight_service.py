"""Lightweight debate service — uses gpt-4o-mini via OpenRouter with real prompts."""

import asyncio
import json
import logging
import random
from typing import AsyncGenerator

from prompts.debater_prompt import build_debater_prompt
from prompts.judge_prompt import build_judge_prompt
from services.llm_client import call_openrouter, parse_r1, parse_verdict_reasoning, make_fallback_reasoning

logger = logging.getLogger(__name__)

LIGHTWEIGHT_MODEL = "openai/gpt-4o-mini"

CONFIG_TO_AGENTS: dict[str, list[str]] = {
    "n2": ["mistral", "gpt4o_mini"],
    "n3": ["mistral", "gpt4o_mini", "qwen"],
    "n4": ["mistral", "gpt4o_mini", "qwen", "llama"],
}

async def call_lightweight_model(messages: list[dict]) -> str:
    """Call gpt-4o-mini via OpenRouter."""
    return await call_openrouter(messages, LIGHTWEIGHT_MODEL)


def _sse(data: dict) -> str:
    """Format a dict as SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_config(config: str) -> tuple[str, int, int]:
    """Return (mode, n_debaters, k_rounds) from config string."""
    if config == "phobert":
        return "plm", 0, 0
    parts = config.split("_")
    mode = parts[0]
    nk = parts[1]
    n = int(nk[1])
    k = int(nk[3:])
    return mode, n, k


async def generate_lightweight_plm(claim: str, evidence: str) -> AsyncGenerator[str, None]:
    """PLM mode — real PhoBERT handles this; lightweight just emits mock probabilities."""
    yield _sse({"type": "plm_start", "model": "phobert"})
    await asyncio.sleep(random.uniform(0.8, 1.2))
    probs = {"Support": round(random.uniform(0.6, 0.95), 4), "Refute": 0.0, "NEI": 0.0}
    probs["Refute"] = round(random.uniform(0.01, 1 - probs["Support"] - 0.01), 4)
    probs["NEI"] = round(1 - probs["Support"] - probs["Refute"], 4)
    label = max(probs, key=lambda k: probs[k])
    yield _sse({"type": "plm_result", "label": label, "confidence": round(probs[label], 4), "probabilities": probs})
    yield _sse({"type": "done"})


async def generate_lightweight_debate(
    claim: str, evidence: str, config: str
) -> AsyncGenerator[str, None]:
    """Generate SSE debate events using real prompts + gpt-4o-mini via OpenRouter."""
    mode, n, k = _parse_config(config)
    agent_ids = CONFIG_TO_AGENTS.get(f"n{n}", ["mistral", "gpt4o_mini"])
    is_hybrid = mode == "hybrid"
    rounds_used = k

    if is_hybrid:
        confidence = round(random.uniform(0.55, 0.99), 4)
        routed_to = "fast_path" if confidence >= 0.85 else "debate"
        yield _sse({"type": "routing", "plm_model": "phobert", "confidence": confidence,
                    "threshold": 0.85, "routed_to": routed_to})
        await asyncio.sleep(0.3)
        if routed_to == "fast_path":
            label = random.choice(["Support", "Refute", "NEI"])
            yield _sse({"type": "final", "label": label, "rounds_used": 0, "total_agent_calls": 1})
            yield _sse({"type": "done"})
            return

    all_rounds: list[dict] = []

    for round_num in range(1, k + 1):
        yield _sse({"type": "round_start", "round": round_num, "total_rounds": k})

        # Build all agent prompts for this round
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

        # Emit all thinking events immediately, then call LLMs in parallel
        for agent_id in agent_ids:
            yield _sse({"type": "agent_thinking", "agent_id": agent_id, "round": round_num})

        responses = await asyncio.gather(*[call_lightweight_model(msg) for msg in prompts])

        round_verdicts: list[dict] = []
        for agent_id, response_text in zip(agent_ids, responses):
            if not response_text:
                verdict = random.choice(["Support", "Refute", "NEI"])
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

    # Judge
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
    judge_response = await call_lightweight_model(judge_messages)

    if not judge_response:
        final_verdict = max(set(all_verdicts_flat), key=all_verdicts_flat.count)
        judge_reasoning = make_fallback_reasoning("judge", evidence, final_verdict)
    else:
        final_verdict, judge_reasoning = parse_verdict_reasoning(judge_response)

    total_calls = sum(len(r["verdicts"]) for r in all_rounds) + 1
    yield _sse({"type": "judge_result", "verdict": final_verdict, "reasoning": judge_reasoning,
                "rounds_used": rounds_used})
    yield _sse({"type": "final", "label": final_verdict, "rounds_used": rounds_used,
                "total_agent_calls": total_calls})
    yield _sse({"type": "done"})
