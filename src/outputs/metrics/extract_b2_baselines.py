"""Extract B2 single-agent baselines from Round 1 verdicts in full-debate logs."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

from src.utils.common import PROJECT_ROOT
from src.utils.constants import LABEL_NAMES

logger = logging.getLogger(__name__)


def extract_r1_by_agent(log_path: str) -> dict[str, list[dict]]:
    """Load full-debate logs and collect each agent's Round 1 prediction per sample.

    Returns {agent_id: [{"gold": str, "pred": str, "model": str}]}.
    """
    agent_data: dict[str, list[dict]] = defaultdict(list)
    path = PROJECT_ROOT / log_path

    if not path.exists():
        logger.error("Log file not found: %s", path)
        return {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in entry or not entry.get("per_round_verdicts"):
                continue

            gold = entry["gold_label"]
            r1_verdicts = entry["per_round_verdicts"][0].get("verdicts", [])
            for v in r1_verdicts:
                agent_data[v["agent_id"]].append({
                    "gold": gold,
                    "pred": v["verdict"],
                    "model": v["model"],
                })

    return dict(agent_data)


def _compute_agent_metrics(samples: list[dict]) -> dict:
    """Compute macro F1 and per-label F1 for one agent's R1 predictions."""
    golds = [s["gold"] for s in samples]
    preds = [s["pred"] for s in samples]
    model = samples[0]["model"] if samples else "unknown"

    macro_f1 = f1_score(golds, preds, labels=LABEL_NAMES, average="macro", zero_division=0)
    per_label = f1_score(golds, preds, labels=LABEL_NAMES, average=None, zero_division=0)

    return {
        "model": model,
        "n_samples": len(samples),
        "macro_f1": round(float(macro_f1), 4),
        "f1_per_label": {
            label: round(float(score), 4)
            for label, score in zip(LABEL_NAMES, per_label)
        },
    }


def compute_and_save_b2_metrics(log_path: str, out_dir: str) -> dict:
    """Extract B2 single-agent metrics from R1 logs and save to JSON.

    Uses the N=4 full-debate log (all 4 debaters present). Output: b2_single_agent.json.
    """
    agent_data = extract_r1_by_agent(log_path)
    if not agent_data:
        logger.warning("No R1 data found in %s — verify log has per_round_verdicts.", log_path)
        return {}

    results: dict[str, dict] = {}
    for agent_id in sorted(agent_data):
        results[agent_id] = _compute_agent_metrics(agent_data[agent_id])
        logger.info(
            "B2 %s (%s): Macro F1 = %.4f  n=%d",
            agent_id,
            results[agent_id]["model"],
            results[agent_id]["macro_f1"],
            results[agent_id]["n_samples"],
        )

    out_path = PROJECT_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "b2_single_agent.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"source_log": log_path, "agents": results}, f, indent=2, ensure_ascii=False)

    logger.info("B2 baselines saved → %s", out_file)
    return results
