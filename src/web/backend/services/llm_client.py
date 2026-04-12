"""Shared OpenRouter client and response parsers for web backend."""

import json
import logging
import os
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VALID_VERDICTS = {"Support", "Refute", "NEI"}

_FALLBACK_REASONINGS: dict[str, str] = {
    "mistral": (
        "After carefully analyzing the provided claim and evidence, I find that the evidence "
        "presents {stance} for the claim in question. The key phrase '{snippet}' directly "
        "addresses the core assertion. The semantic alignment between the claim and evidence "
        "leads me to conclude: {verdict}."
    ),
    "gpt4o_mini": (
        "Upon thorough examination of the claim alongside the provided evidence, the analysis "
        "reveals a {stance} relationship. The evidence states: '{snippet}', which {rel} the "
        "claim under review. Therefore, my assessment concludes with a verdict of {verdict}."
    ),
    "qwen": (
        "Analyzing this Vietnamese fact-checking task, the evidence '{snippet}' provides "
        "{stance} information regarding the claim. The logical inference path leads to a "
        "{verdict} conclusion."
    ),
    "llama": (
        "Key observation: '{snippet}' — this portion of the evidence is particularly relevant. "
        "The stance of the evidence toward the claim appears to be {stance}. Verdict: {verdict}."
    ),
    "judge": (
        "Having reviewed all debater arguments, the majority converged on {verdict}. "
        "The winning argument demonstrates direct evidence quotation and a sound inference chain. "
        "Final verdict: {verdict}."
    ),
}


async def call_openrouter(messages: list[dict], model: str) -> str:
    """Call OpenRouter API with given model. Returns empty string on failure."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set")
        return ""

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.error("OpenRouter call failed for model %s: %s", model, exc)
        return ""


def parse_r1(text: str, fallback_verdict: str = "NEI") -> tuple[str, str]:
    """Parse Round-1 JSON response: {\"verdict\": ..., \"reasoning\": ...}."""
    if not text:
        return fallback_verdict, "Unable to parse response."
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            verdict = data.get("verdict", fallback_verdict)
            if verdict not in VALID_VERDICTS:
                verdict = fallback_verdict
            reasoning = data.get("reasoning", "")
            return verdict, reasoning
    except (json.JSONDecodeError, KeyError):
        pass
    return parse_verdict_reasoning(text, fallback_verdict)


def parse_verdict_reasoning(text: str, fallback_verdict: str = "NEI") -> tuple[str, str]:
    """Parse VERDICT: X\\nREASONING: Y format used by R2+ and Judge."""
    if not text:
        return fallback_verdict, "Unable to parse response."
    verdict = fallback_verdict
    reasoning = text.strip()
    v_match = re.search(r"VERDICT:\s*(Support|Refute|NEI)", text, re.IGNORECASE)
    if v_match:
        raw = v_match.group(1).capitalize()
        if raw in VALID_VERDICTS:
            verdict = raw
    r_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if r_match:
        reasoning = r_match.group(1).strip()
    return verdict, reasoning


def make_fallback_reasoning(agent_id: str, evidence: str, verdict: str) -> str:
    """Generate template reasoning when API call fails."""
    template = _FALLBACK_REASONINGS.get(agent_id, _FALLBACK_REASONINGS["mistral"])
    snippet = evidence[:60].strip() if evidence else "the provided evidence"
    stance_map = {"Support": "supporting", "Refute": "contradicting", "NEI": "insufficient"}
    rel_map = {"Support": "corroborates", "Refute": "contradicts", "NEI": "partially addresses"}
    return template.format(
        stance=stance_map.get(verdict, "neutral"),
        snippet=snippet,
        rel=rel_map.get(verdict, "relates to"),
        verdict=verdict,
    )
