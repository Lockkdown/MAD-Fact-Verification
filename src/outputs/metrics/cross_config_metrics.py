"""Cross-config metrics aggregation: summary table, DCS, and shared log loader for MAD analysis."""

import json
import logging
from pathlib import Path

from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

CONFIG_ORDER = ["n2k3", "n2k5", "n3k3", "n3k5", "n4k3", "n4k5"]


def load_all_metrics(mode: str) -> dict[str, dict]:
    """Load metrics.json for every config under reports/debate/{mode}/."""
    base = PROJECT_ROOT / "reports" / "debate" / mode
    result: dict[str, dict] = {}
    for cfg_name in CONFIG_ORDER:
        path = base / cfg_name / "metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                result[cfg_name] = json.load(f)
        else:
            logger.warning("metrics.json not found: %s", path)
    return result


def load_all_logs(mode: str) -> dict[str, list[dict]]:
    """Load completed samples from logs.jsonl for every config under reports/debate/{mode}/."""
    base = PROJECT_ROOT / "reports" / "debate" / mode
    result: dict[str, list[dict]] = {}
    for cfg_name in CONFIG_ORDER:
        path = base / cfg_name / "logs.jsonl"
        if path.exists():
            result[cfg_name] = _load_jsonl(path)
        else:
            logger.warning("logs.jsonl not found: %s", path)
    return result


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping error and malformed lines."""
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "error" not in entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def build_summary_table(metrics_by_config: dict[str, dict]) -> list[dict]:
    """Build a flat list of per-config summary rows for the paper results table."""
    rows: list[dict] = []
    for cfg_name in CONFIG_ORDER:
        if cfg_name not in metrics_by_config:
            continue
        m = metrics_by_config[cfg_name]
        f1_per = m.get("f1_per_label", {})
        rows.append({
            "config": cfg_name,
            "mode": m.get("config", {}).get("mode"),
            "n": m.get("config", {}).get("n"),
            "k": m.get("config", {}).get("k"),
            "macro_f1": m.get("macro_f1"),
            "support_f1": f1_per.get("Support"),
            "refute_f1": f1_per.get("Refute"),
            "nei_f1": f1_per.get("NEI"),
            "avg_agent_calls": m.get("avg_agent_calls"),
            "dsr": m.get("dsr"),
            "verdict_flip_rate": m.get("verdict_flip_rate"),
            "routing_fp_rate": m.get("routing_fp_rate"),
            "routing_fn_rate": m.get("routing_fn_rate"),
            "early_stop_rate": m.get("early_stop_rate"),
            "avg_rounds_per_debate": m.get("avg_rounds_per_debate"),
        })
    return rows


def compute_dcs(debate_f1: float, judge_only_f1: float) -> float:
    """Debate Contribution Score = debate_f1 − judge_only_f1."""
    return round(debate_f1 - judge_only_f1, 4)


def run_cross_config_analysis(
    mode: str,
    judge_only_f1: float | None = None,
) -> dict:
    """Load metrics for all configs, build summary table, optionally compute DCS, and save."""
    metrics_by_config = load_all_metrics(mode)
    if not metrics_by_config:
        logger.warning("No metrics found for mode=%s — skipping analysis.", mode)
        return {}

    summary = build_summary_table(metrics_by_config)

    if judge_only_f1 is not None:
        for row in summary:
            f1 = row.get("macro_f1")
            if f1 is not None:
                row["dcs"] = compute_dcs(f1, judge_only_f1)

    out_path = PROJECT_ROOT / "reports" / "debate" / mode / "summary_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "configs": summary}, f, indent=2, ensure_ascii=False)

    logger.info("Cross-config summary saved → %s", out_path)
    return {"mode": mode, "configs": summary}
