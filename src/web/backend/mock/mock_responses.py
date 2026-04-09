"""Backward-compat re-exports — logic đã chuyển sang lightweight_service.py."""

from mock.lightweight_service import (  # noqa: F401
    generate_lightweight_plm as generate_mock_plm,
    generate_lightweight_debate as generate_mock_debate,
)

import asyncio
import json
import random
from typing import AsyncGenerator

MOCK_VERDICTS = ["Support", "Refute", "NEI"]

MOCK_REASONINGS = {
    "mistral": (
        "After carefully analyzing the provided claim and evidence, I find that the evidence "
        "presents {stance} for the claim in question. The key phrase '{evidence_snippet}' directly "
        "addresses the core assertion. From a logical standpoint, the relationship between the "
        "claim and the evidence is {relationship}. The factual basis is well-established within "
        "the Vietnamese context, and no contradictory information has been identified. "
        "Considering the specificity of the evidence and the directness of its relationship to "
        "the claim, I am confident in my verdict. The semantic alignment between the claim and "
        "evidence leaves little room for alternative interpretations."
    ),
    "gpt4o_mini": (
        "Upon thorough examination of the claim alongside the provided evidence, the analysis "
        "reveals a {stance} relationship. The evidence states: '{evidence_snippet}', which "
        "{relationship} the claim under review. From a fact-checking perspective, the credibility "
        "of the evidence is high given the specificity of the information. Cross-referencing the "
        "key entities mentioned in both the claim and evidence shows {alignment_type} alignment. "
        "The temporal and contextual factors are consistent. Therefore, my assessment concludes "
        "with a verdict that the claim is {verdict_lower}ed by the available evidence."
    ),
    "qwen": (
        "Analyzing this Vietnamese fact-checking task, I examine the semantic relationship between "
        "the claim and the gold evidence. The evidence '{evidence_snippet}' provides {stance} "
        "information regarding the claim. The key entities and their predicates in both texts "
        "{relationship} each other. Considering the factual density of the evidence and the "
        "precision of the claim's assertions, the logical inference path leads to a {verdict} "
        "conclusion. The Vietnamese linguistic context has been accounted for, including any "
        "domain-specific terminology. My confidence in this verdict is moderate to high."
    ),
    "llama": (
        "This fact-checking analysis examines the relationship between the submitted claim and "
        "the provided evidence. Key observation: '{evidence_snippet}' — this portion of the "
        "evidence is particularly relevant to evaluating the claim's validity. The stance of the "
        "evidence toward the claim appears to be {stance}. From a reasoning chain perspective: "
        "(1) the claim makes a specific assertion, (2) the evidence addresses this assertion "
        "{relationship}, (3) therefore the verdict is {verdict}. I have considered potential "
        "ambiguities in interpretation but find the evidence sufficiently clear to render judgment."
    ),
    "judge": (
        "Having reviewed all debater arguments and the original evidence, I synthesize the "
        "following final assessment. The majority of debaters converged on a {verdict} verdict, "
        "with reasoning centered on '{evidence_snippet}'. The dissenting argument, if any, lacked "
        "sufficient grounding in the provided evidence. My role as judge is to evaluate argument "
        "quality and consistency with the evidence — not to introduce external knowledge. "
        "The winning argument demonstrates: (1) direct quotation from evidence, (2) logical "
        "inference chain, and (3) appropriate handling of ambiguity. Final verdict: {verdict}. "
        "This assessment is based solely on the provided evidence and debater arguments."
    ),
}

CONFIG_TO_AGENTS = {
    "n2": ["mistral", "gpt4o_mini"],
    "n3": ["mistral", "gpt4o_mini", "qwen"],
    "n4": ["mistral", "gpt4o_mini", "qwen", "llama"],
}


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


def _make_reasoning(agent_id: str, claim: str, evidence: str, verdict: str) -> str:
    """Fill reasoning template with actual claim/evidence content."""
    template = MOCK_REASONINGS.get(agent_id, MOCK_REASONINGS["mistral"])
    evidence_snippet = evidence[:60].strip() if evidence else "the provided evidence"
    stance_map = {"Support": "supporting", "Refute": "contradicting", "NEI": "insufficient"}
    rel_map = {"Support": "corroborates", "Refute": "contradicts", "NEI": "partially addresses"}
    return template.format(
        stance=stance_map.get(verdict, "neutral"),
        evidence_snippet=evidence_snippet,
        relationship=rel_map.get(verdict, "relates to"),
        alignment_type="strong" if verdict == "Support" else "weak",
        verdict=verdict,
        verdict_lower=verdict.lower(),
    )


def _sse(data: dict) -> str:
    """Format a dict as SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def generate_mock_plm(claim: str, evidence: str) -> AsyncGenerator[str, None]:
    """Generate mock SSE events for PLM mode."""
    yield _sse({"type": "plm_start", "model": "phobert"})
    await asyncio.sleep(random.uniform(0.8, 1.2))
    probs = {
        "Support": round(random.uniform(0.6, 0.95), 4),
        "Refute": 0.0,
        "NEI": 0.0,
    }
    probs["Refute"] = round(random.uniform(0.01, 1 - probs["Support"] - 0.01), 4)
    probs["NEI"] = round(1 - probs["Support"] - probs["Refute"], 4)
    label = max(probs, key=lambda k: probs[k])
    confidence = probs[label]
    yield _sse({
        "type": "plm_result",
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": probs,
    })
    yield _sse({"type": "done"})


async def generate_mock_debate(
    claim: str, evidence: str, config: str
) -> AsyncGenerator[str, None]:
    """Generate mock SSE events for full or hybrid debate mode."""
    mode, n, k = _parse_config(config)
    agent_ids = CONFIG_TO_AGENTS.get(f"n{n}", ["mistral", "gpt4o_mini"])

    is_hybrid = mode == "hybrid"
    rounds_used = k

    if is_hybrid:
        confidence = round(random.uniform(0.55, 0.99), 4)
        routed_to = "fast_path" if confidence >= 0.85 else "debate"
        yield _sse({
            "type": "routing",
            "plm_model": "phobert",
            "confidence": confidence,
            "threshold": 0.85,
            "routed_to": routed_to,
        })
        await asyncio.sleep(0.6)

        if routed_to == "fast_path":
            label = random.choice(MOCK_VERDICTS)
            yield _sse({"type": "final", "label": label, "rounds_used": 0, "total_agent_calls": 1})
            yield _sse({"type": "done"})
            return

    agent_verdicts_by_round: list[dict[str, str]] = []

    for round_num in range(1, k + 1):
        yield _sse({"type": "round_start", "round": round_num, "total_rounds": k})
        await asyncio.sleep(0.2)

        round_verdicts: dict[str, str] = {}
        for agent_id in agent_ids:
            yield _sse({"type": "agent_thinking", "agent_id": agent_id, "round": round_num})
            await asyncio.sleep(random.uniform(0.8, 1.2))

            verdict = random.choice(MOCK_VERDICTS)
            round_verdicts[agent_id] = verdict
            reasoning = _make_reasoning(agent_id, claim, evidence, verdict)
            yield _sse({
                "type": "agent_result",
                "agent_id": agent_id,
                "round": round_num,
                "verdict": verdict,
                "reasoning": reasoning,
            })

        agent_verdicts_by_round.append(round_verdicts)
        is_unanimous = len(set(round_verdicts.values())) == 1
        yield _sse({"type": "round_end", "round": round_num, "is_unanimous": is_unanimous})

        if is_unanimous and round_num < k:
            rounds_used = round_num
            break

    yield _sse({"type": "judge_thinking"})
    await asyncio.sleep(1.5)

    all_verdicts = [v for rd in agent_verdicts_by_round for v in rd.values()]
    final_verdict = max(set(all_verdicts), key=all_verdicts.count)
    judge_reasoning = _make_reasoning("judge", claim, evidence, final_verdict)
    total_calls = rounds_used * n + 1
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
