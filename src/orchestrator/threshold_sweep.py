"""Virtual threshold sweep: analyze saved DEV debate logs to find optimal routing threshold."""

import json
import logging
from pathlib import Path

import yaml
from sklearn.metrics import f1_score

from src.utils.common import PROJECT_ROOT
from src.utils.constants import LABEL_NAMES, THRESHOLD_SWEEP

logger = logging.getLogger(__name__)


def load_debate_logs(log_path: str) -> dict[str, dict]:
    """Load per-sample debate JSONL → dict keyed by sample_id."""
    logs: dict[str, dict] = {}
    with open(PROJECT_ROOT / log_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            logs[str(r["sample_id"])] = r
    return logs


def load_plm_scores(scores_path: str) -> dict[str, dict]:
    """Load per-sample PLM confidence JSONL → dict keyed by sample_id.

    Each entry must have: sample_id, confidence, plm_verdict, gold_label.
    """
    scores: dict[str, dict] = {}
    with open(PROJECT_ROOT / scores_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            scores[str(r["sample_id"])] = r
    return scores


def simulate_hybrid(
    debate_logs: dict[str, dict],
    plm_scores: dict[str, dict],
    threshold: float,
) -> tuple[list[str], list[str], float]:
    """Simulate hybrid routing for one threshold value. Returns (preds, golds, dsr).

    confidence >= threshold → fast path (PLM verdict)
    confidence <  threshold → debate verdict
    """
    preds, golds = [], []
    skipped = 0
    for sid, plm in plm_scores.items():
        gold = plm["gold_label"]
        if plm["confidence"] >= threshold:
            pred = plm["plm_verdict"]
            skipped += 1
        else:
            pred = debate_logs.get(sid, {}).get("final_verdict", "NEI")
        preds.append(pred)
        golds.append(gold)
    dsr = skipped / len(plm_scores) if plm_scores else 0.0
    return preds, golds, dsr


def _compute_metrics(preds: list[str], golds: list[str]) -> tuple[float, dict]:
    """Return (macro_f1, per_label_f1_dict) for Support/Refute/NEI."""
    macro_f1 = f1_score(golds, preds, labels=LABEL_NAMES, average="macro", zero_division=0)
    per_label = f1_score(golds, preds, labels=LABEL_NAMES, average=None, zero_division=0)
    return float(macro_f1), dict(zip(LABEL_NAMES, [round(float(v), 4) for v in per_label]))


def run_threshold_sweep(
    debate_log_path: str,
    plm_scores_path: str,
    thresholds: list[float],
    output_dir: str,
) -> dict:
    """Sweep all thresholds, log results table, save JSON, and return best t*."""
    debate_logs = load_debate_logs(debate_log_path)
    plm_scores = load_plm_scores(plm_scores_path)

    results: dict[str, dict] = {}
    best_t, best_f1 = thresholds[0], -1.0

    logger.info("%-6s  %-9s  %-9s  %-9s  %-9s  %-6s", "t", "MacroF1", "Support", "Refute", "NEI", "DSR%")
    logger.info("-" * 62)

    for t in thresholds:
        preds, golds, dsr = simulate_hybrid(debate_logs, plm_scores, t)
        macro_f1, per_label = _compute_metrics(preds, golds)
        results[str(t)] = {
            "macro_f1": round(macro_f1, 4),
            "per_label_f1": per_label,
            "dsr": round(dsr, 4),
        }
        logger.info(
            "%-6.2f  %-9.4f  %-9.4f  %-9.4f  %-9.4f  %.1f%%",
            t, macro_f1,
            per_label["Support"], per_label["Refute"], per_label["NEI"],
            dsr * 100,
        )
        if macro_f1 > best_f1:
            best_f1, best_t = macro_f1, t

    logger.info("-" * 62)
    logger.info("Optimal threshold: t* = %.2f  (Macro F1 = %.4f)", best_t, best_f1)

    output = {
        "thresholds": results,
        "optimal_threshold": best_t,
        "best_macro_f1": round(best_f1, 4),
    }
    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Sweep results saved → %s/sweep_results.json", output_dir)
    return output


def run_threshold_sweep_from_config(config_path: str, plm_scores_path: str) -> dict:
    """Load sweep config from YAML and run threshold sweep."""
    resolved = Path(config_path) if Path(config_path).is_absolute() else PROJECT_ROOT / config_path
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return run_threshold_sweep(
        debate_log_path=cfg["output"]["log_path"],
        plm_scores_path=plm_scores_path,
        thresholds=THRESHOLD_SWEEP,
        output_dir=cfg["output"].get("sweep_dir", "reports/debate/sweep"),
    )
