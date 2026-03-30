"""Compute and save debate experiment metrics from a completed logs.jsonl file."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]


def compute_and_save_debate_metrics(
    log_path: str,
    metrics_path: str,
    cfg: dict,
) -> None:
    """Read logs.jsonl, compute metrics, and write metrics.json."""
    samples, total_run = _load_logs(log_path)
    if not samples:
        logger.warning("No completed samples found in %s — skipping metrics.", log_path)
        return

    metrics = _compute_metrics(samples, total_run, cfg)
    _save_json(metrics_path, metrics)
    logger.info("Debate metrics saved → %s", metrics_path)


def _resolve(path_str: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_logs(log_path: str) -> tuple[list[dict], int]:
    """Load sample results from JSONL. Returns (success_samples, total_count)."""
    path = _resolve(log_path)
    if not path.exists():
        return [], 0
    results = []
    total_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total_count += 1
                if "error" not in entry:  # skip crashed samples for quality metrics
                    results.append(entry)
            except json.JSONDecodeError:
                continue
    return results, total_count


def _get_r1_majority_verdict(per_round_verdicts: list[dict]) -> str | None:
    """Extract majority verdict from round 1 debater votes."""
    if not per_round_verdicts:
        return None
    r1_verdicts = [v["verdict"] for v in per_round_verdicts[0].get("verdicts", [])]
    if not r1_verdicts:
        return None
    return max(r1_verdicts, key=r1_verdicts.count)


def _compute_verdict_flip_rate(debate_samples: list[dict]) -> float:
    """% debate-path samples whose final verdict differs from R1 majority verdict."""
    eligible = [s for s in debate_samples if s.get("per_round_verdicts")]
    if not eligible:
        return 0.0
    flipped = sum(
        1 for s in eligible
        if _get_r1_majority_verdict(s["per_round_verdicts"]) != s["final_verdict"]
    )
    return round(flipped / len(eligible), 4)


def _compute_routing_rates(samples: list[dict]) -> tuple[float, float]:
    """Return (routing_fp_rate, routing_fn_rate) for hybrid mode.

    FP: debate-path where PLM would have been correct (needless debate call).
    FN: fast-path where PLM was wrong (debate was skipped but needed).
    """
    fast_path = [s for s in samples if not s.get("routed_to_debate", True)]
    debate_path = [s for s in samples if s.get("routed_to_debate", True)]

    fn_count = sum(1 for s in fast_path if s["final_verdict"] != s["gold_label"])
    routing_fn_rate = round(fn_count / len(fast_path), 4) if fast_path else 0.0

    debate_with_verdict = [s for s in debate_path if s.get("m_star_verdict") is not None]
    fp_count = sum(1 for s in debate_with_verdict if s["m_star_verdict"] == s["gold_label"])
    routing_fp_rate = round(fp_count / len(debate_with_verdict), 4) if debate_with_verdict else 0.0

    return routing_fp_rate, routing_fn_rate


def _compute_metrics(samples: list[dict], total_run: int, cfg: dict) -> dict:
    """Compute all debate metrics from a list of completed sample results."""
    n_debaters = cfg["debate"]["panel"]["debaters"].__len__()
    k_max = cfg["debate"]["rounds"]

    gold_labels = [s["gold_label"] for s in samples]
    pred_labels = [s["final_verdict"] for s in samples]
    total = len(samples)

    macro_f1 = f1_score(gold_labels, pred_labels, labels=LABEL_NAMES, average="macro", zero_division=0)
    accuracy = sum(g == p for g, p in zip(gold_labels, pred_labels)) / total

    f1_per_label = {}
    for i, label in enumerate(LABEL_NAMES):
        f1_per_label[label] = round(
            f1_score(gold_labels, pred_labels, labels=LABEL_NAMES, average=None, zero_division=0)[i], 4
        )

    avg_agent_calls = sum(s["num_agent_calls"] for s in samples) / total
    avg_rounds_used = sum(s["rounds_used"] for s in samples) / total
    judge_called_rate = sum(1 for s in samples if s["judge_called"]) / total

    mode = cfg["debate"]["mode"]
    debate_samples = (
        [s for s in samples if s.get("routed_to_debate", True)]
        if mode == "hybrid_debate" else samples
    )
    n_debate = len(debate_samples)

    # early_stop: only over samples that actually debated (avoids fast-path 0s inflating rate)
    early_stop_rate = (
        sum(1 for s in debate_samples if s["rounds_used"] < k_max) / n_debate
        if n_debate > 0 else 0.0
    )
    # avg rounds for samples that actually entered debate (meaningful for k=5 ablation)
    avg_rounds_per_debate = (
        round(sum(s["rounds_used"] for s in debate_samples) / n_debate, 2)
        if n_debate > 0 else None
    )

    # unanimous_at_round distribution
    round_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        r = s.get("unanimous_at_round")
        if r is None:
            round_counts["never"] += 1
        else:
            round_counts[f"round_{r}"] += 1

    unanimous_rate = {
        **{f"round_{r}": round(round_counts[f"round_{r}"] / total, 4) for r in range(1, k_max + 1)},
        "never": round(round_counts["never"] / total, 4),
    }

    error_count = total_run - total
    error_rate = round(error_count / total_run, 4) if total_run > 0 else 0.0

    dsr: float | None = None
    routing_fp_rate: float | None = None
    routing_fn_rate: float | None = None
    if mode == "hybrid_debate":
        fast_path_count = sum(1 for s in samples if not s.get("routed_to_debate", True))
        dsr = round(fast_path_count / total, 4) if total > 0 else 0.0
        routing_fp_rate, routing_fn_rate = _compute_routing_rates(samples)

    verdict_flip_rate = _compute_verdict_flip_rate(debate_samples)

    return {
        "config": {
            "mode": mode,
            "n": n_debaters,
            "k": k_max,
        },
        "total_run": total_run,
        "successful_samples": total,
        "error_count": error_count,
        "error_rate": error_rate,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "f1_per_label": f1_per_label,
        "avg_agent_calls": round(avg_agent_calls, 2),
        "avg_rounds_used": round(avg_rounds_used, 2),
        "avg_rounds_per_debate": avg_rounds_per_debate,
        "dsr": dsr,
        "routing_fp_rate": routing_fp_rate,
        "routing_fn_rate": routing_fn_rate,
        "verdict_flip_rate": verdict_flip_rate,
        "unanimous_rate": unanimous_rate,
        "judge_called_rate": round(judge_called_rate, 4),
        "early_stop_rate": round(early_stop_rate, 4),
    }


def _save_json(path: str, data: dict) -> None:
    """Write dict to JSON file, creating parent dirs as needed."""
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
