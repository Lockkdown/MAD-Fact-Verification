"""Orchestrator: runs one debate round, checks unanimity for early stopping, and resolves via judge."""

import asyncio

from src.orchestrator.agents.debater import DebaterAgent
from src.orchestrator.agents.judge import JudgeAgent


class Orchestrator:
    """Coordinates parallel debater calls and resolves the final verdict."""

    def __init__(self, debaters: list[DebaterAgent], judge: JudgeAgent):
        self.debaters = debaters
        self.judge = judge

    async def run_round(
        self,
        statement: str,
        evidence: str,
        round_num: int,
        prior_rounds: list[dict],
    ) -> dict:
        """Run one full round — all debaters in parallel. Returns round result dict."""
        tasks = [
            d.debate(statement, evidence, round_num, prior_rounds)
            for d in self.debaters
        ]
        verdicts = await asyncio.gather(*tasks)
        return {"round": round_num, "verdicts": list(verdicts)}

    def check_unanimous(self, round_result: dict) -> tuple[bool, str | None]:
        """Return (is_unanimous, consensus_verdict). If split, consensus is None.

        Used for early stopping only — judge is ALWAYS called regardless.
        """
        labels = [v["verdict"] for v in round_result["verdicts"]]
        unique = set(labels)
        if len(unique) == 1:
            return True, labels[0]
        return False, None

    async def resolve(
        self,
        statement: str,
        evidence: str,
        all_rounds: list[dict],
    ) -> dict:
        """Resolve final verdict. Judge is ALWAYS called regardless of unanimity."""
        if not all_rounds:
            # Judge-only mode (k=0): no debate context, call judge directly
            is_unanimous, consensus = False, None
        else:
            is_unanimous, consensus = self.check_unanimous(all_rounds[-1])
        judge_result = await self.judge.adjudicate(
            statement, evidence, all_rounds,
            is_unanimous=is_unanimous,
            consensus_verdict=consensus,
        )
        return {
            "verdict": judge_result["verdict"],
            "judge_called": True,
            "judge_reasoning": judge_result["reasoning"],
            "judge_tokens": {
                "input": judge_result["input_tokens"],
                "output": judge_result["output_tokens"],
            },
        }
