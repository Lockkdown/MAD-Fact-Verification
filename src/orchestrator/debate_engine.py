"""DebateEngine: main debate loop with early stopping and crash-safe per-sample logging."""

from src.orchestrator.orchestrator import Orchestrator
from src.outputs.metrics.debate_logger import DebateLogger


class DebateEngine:
    """Runs the full multi-round debate for a single sample."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        logger: DebateLogger,
        k_max: int,
        early_stopping: bool = True,
    ):
        self.orchestrator = orchestrator
        self.logger = logger
        self.k_max = k_max
        self.early_stopping = early_stopping

    async def run(
        self,
        sample_id: str,
        statement: str,
        evidence: str,
        gold_label: str,
        mode: str,
        routed_to_debate: bool = True,
        m_star_confidence: float | None = None,
    ) -> dict:
        """Run full debate for one sample. Logs result even on exception."""
        all_rounds: list[dict] = []
        rounds_used = 0
        unanimous_at_round: int | None = None
        result: dict = {}

        try:
            for round_num in range(1, self.k_max + 1):
                round_result = await self.orchestrator.run_round(
                    statement, evidence, round_num, all_rounds
                )
                all_rounds.append(round_result)
                rounds_used = round_num

                is_unanimous, _ = self.orchestrator.check_unanimous(round_result)
                if is_unanimous:
                    unanimous_at_round = round_num
                    if self.early_stopping:
                        break

            resolution = await self.orchestrator.resolve(
                statement, all_rounds, all_rounds[-1]
            )

            n_debaters = len(self.orchestrator.debaters)
            num_agent_calls = rounds_used * n_debaters + (1 if resolution["judge_called"] else 0)

            result = {
                "sample_id": sample_id,
                "gold_label": gold_label,
                "mode": mode,
                "n_debaters": n_debaters,
                "k_max": self.k_max,
                "routed_to_debate": routed_to_debate,
                "m_star_confidence": m_star_confidence,
                "rounds_used": rounds_used,
                "num_agent_calls": num_agent_calls,
                "unanimous_at_round": unanimous_at_round,
                "per_round_verdicts": all_rounds,
                "judge_called": resolution["judge_called"],
                "judge_reasoning": resolution["judge_reasoning"],
                "final_verdict": resolution["verdict"],
                "correct": resolution["verdict"] == gold_label,
            }

        except Exception as exc:
            result = self._make_error_result(
                sample_id, gold_label, mode, m_star_confidence, str(exc)
            )

        finally:
            self.logger.log(result)

        return result

    def _make_error_result(
        self,
        sample_id: str,
        gold_label: str,
        mode: str,
        m_star_confidence: float | None,
        error_msg: str,
    ) -> dict:
        """Safe fallback result dict logged when debate raises an exception."""
        return {
            "sample_id": sample_id,
            "gold_label": gold_label,
            "mode": mode,
            "n_debaters": len(self.orchestrator.debaters),
            "k_max": self.k_max,
            "routed_to_debate": True,
            "m_star_confidence": m_star_confidence,
            "rounds_used": 0,
            "num_agent_calls": 0,
            "unanimous_at_round": None,
            "per_round_verdicts": [],
            "judge_called": False,
            "judge_reasoning": None,
            "final_verdict": "NEI",
            "correct": False,
            "error": error_msg,
        }
