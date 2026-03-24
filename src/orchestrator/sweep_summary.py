"""Consolidate per-config sweep_results.json files into a single threshold summary JSON."""

import json
import logging
from pathlib import Path

from src.utils.common import PROJECT_ROOT
from src.utils.constants import THRESHOLD_SWEEP

logger = logging.getLogger(__name__)

_CONFIGS = ["n2k3", "n2k5", "n3k3", "n3k5", "n4k3", "n4k5"]
_SWEEP_ROOT = "reports/debate/sweep"
_OUTPUT_DIR = "reports/debate/find_threshold"
_OUTPUT_FILE = "threshold_results.json"


def _load_sweep_result(config: str) -> dict:
    """Load sweep_results.json for a single config."""
    path = PROJECT_ROOT / _SWEEP_ROOT / config / "sweep_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Sweep result missing: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_config_entry(config: str, data: dict) -> dict:
    """Build per-config entry: t*, best F1, DSR at t*, full sweep table."""
    t_star = data["optimal_threshold"]
    t_star_key = str(t_star)
    t_star_row = data["thresholds"].get(t_star_key, {})
    return {
        "t_star": t_star,
        "best_macro_f1": data["best_macro_f1"],
        "dsr_at_t_star": t_star_row.get("dsr"),
        "per_label_f1_at_t_star": t_star_row.get("per_label_f1"),
        "full_sweep": data["thresholds"],
    }


def generate_sweep_summary(output_dir: str = _OUTPUT_DIR) -> None:
    """Read all per-config sweep results and write a consolidated summary JSON."""
    configs_data: dict[str, dict] = {}
    for config in _CONFIGS:
        try:
            raw = _load_sweep_result(config)
            configs_data[config] = _build_config_entry(config, raw)
            logger.info("  %-6s  t*=%.2f  F1=%.4f  DSR=%.1f%%",
                        config,
                        configs_data[config]["t_star"],
                        configs_data[config]["best_macro_f1"],
                        (configs_data[config]["dsr_at_t_star"] or 0) * 100)
        except FileNotFoundError as exc:
            logger.warning("%s — skipping", exc)

    summary = {
        "plm_model": "vinai/phobert-base (M*)",
        "data_split": "dev",
        "thresholds_tested": THRESHOLD_SWEEP,
        "configs": configs_data,
        "t_star_table": {
            cfg: {
                "t_star": entry["t_star"],
                "macro_f1": entry["best_macro_f1"],
                "dsr": entry["dsr_at_t_star"],
            }
            for cfg, entry in configs_data.items()
        },
    }

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / _OUTPUT_FILE
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Threshold summary saved → %s", out_file)
