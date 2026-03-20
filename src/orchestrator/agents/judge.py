"""JudgeAgent: Constrained Synthesizer — always called after debate ends."""

import re

from src.api.openrouter_client import LLMResponse, OpenRouterClient
from src.orchestrator.agents.prompts.judge_prompt import build_judge_prompt
from src.utils.constants import VALID_LABELS


class JudgeAgent:
    """Judge agent that resolves split verdicts by evaluating argument quality."""

    def __init__(self, model: str, client: OpenRouterClient):
        self.model = model
        self.client = client

    async def adjudicate(
        self,
        statement: str,
        evidence: str,
        all_rounds: list[dict],
        is_unanimous: bool = False,
        consensus_verdict: str | None = None,
    ) -> dict:
        """Resolve verdict. Returns structured result dict."""
        messages = build_judge_prompt(
            statement, evidence, all_rounds,
            is_unanimous=is_unanimous,
            consensus_verdict=consensus_verdict,
        )
        # Lower temperature for more deterministic adjudication
        response = await self.client.complete(self.model, messages, temperature=0.3)
        verdict, reasoning = self._parse_response(response)
        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "success": response.success,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

    def _parse_response(self, response: LLMResponse) -> tuple[str, str]:
        """Extract VERDICT and REASONING from response. Fallback to NEI on failure."""
        if not response.success:
            return "NEI", f"Judge API call failed: {response.error}"

        text = response.content.strip()
        verdict_match = re.search(r"VERDICT:\s*(Support|Refute|NEI)", text)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)

        verdict = verdict_match.group(1) if verdict_match else "NEI"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]

        if verdict not in VALID_LABELS:
            verdict = "NEI"

        return verdict, reasoning
