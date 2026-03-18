"""DebaterAgent: wraps OpenRouterClient to run one debate round and parse the response."""

import asyncio
import json
import re

from src.api.openrouter_client import LLMResponse, OpenRouterClient
from src.orchestrator.agents.prompts.debater_prompt import build_debater_prompt
from src.utils.constants import VALID_LABELS


class DebaterAgent:
    """Single debater agent backed by a configurable LLM via OpenRouter."""

    def __init__(self, agent_id: str, model: str, client: OpenRouterClient, max_retries: int = 2):
        self.agent_id = agent_id
        self.model = model
        self.client = client
        self.max_retries = max_retries

    async def debate(
        self,
        statement: str,
        evidence: str,
        round_num: int,
        prior_rounds: list[dict],
    ) -> dict:
        """Run one debate round with content-level retry. Returns structured result dict."""
        response = LLMResponse(content="", model=self.model, input_tokens=0, output_tokens=0, success=False)

        for attempt in range(self.max_retries):
            # Last retry: strip prior rounds to reduce context length
            prior = prior_rounds if attempt < self.max_retries - 1 else []
            messages = build_debater_prompt(statement, evidence, round_num, prior, self.agent_id)
            response = await self.client.complete(self.model, messages)
            if response.success and response.content.strip():
                break
            await asyncio.sleep(2 ** attempt)

        verdict, reasoning = self._parse_response(response, round_num)
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "verdict": verdict,
            "reasoning": reasoning,
            "success": response.success,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

    def _parse_response(self, response: LLMResponse, round_num: int) -> tuple[str, str]:
        """Dispatch to format-specific parser based on round number."""
        if not response.success:
            return "NEI", f"API call failed: {response.error}"
        if round_num == 1:
            return self._parse_json_response(response.content)
        return self._parse_text_response(response.content)

    def _parse_json_response(self, text: str) -> tuple[str, str]:
        """Parse R1 JSON output: {parts, verdict, reasoning}. Fallback to text parser on failure."""
        try:
            json_match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                verdict = data.get("verdict", "NEI")
                reasoning = data.get("reasoning", text[:200])
                if verdict in VALID_LABELS:
                    return verdict, reasoning
        except (json.JSONDecodeError, KeyError):
            pass
        # Graceful fallback if model ignored JSON instruction
        return self._parse_text_response(text)

    def _parse_text_response(self, text: str) -> tuple[str, str]:
        """Parse R2+ plain-text VERDICT:/REASONING: format."""
        text = text.strip()
        verdict_match = re.search(r"VERDICT:\s*(Support|Refute|NEI)", text)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
        verdict = verdict_match.group(1) if verdict_match else "NEI"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]
        if verdict not in VALID_LABELS:
            verdict = "NEI"
        return verdict, reasoning
