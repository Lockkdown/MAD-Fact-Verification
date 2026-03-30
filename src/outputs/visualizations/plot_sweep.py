"""A1 threshold sweep visualization: dual-axis line chart of Macro F1 and DSR vs routing threshold."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt

from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

BEST_CONFIG = "n3k3"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
_COLOR_F1 = "#4C72B0"
_COLOR_DSR = "#DD8452"


def plot_threshold_sweep(out_dir: str | None = None) -> None:
    """Dual-axis line chart: Macro F1 (left) and DSR % (right) vs routing threshold.

    Shows only the best config (n3k3). Reads sweep_results.json from reports/debate/sweep/n3k3/.
    """
    out_path = _resolve_out_dir(out_dir) / "threshold_sweep.png"
    results = _load_best_config_sweep()
    if not results:
        logger.warning("sweep_results.json not found for %s — run --phase sweep first.", BEST_CONFIG)
        return

    thresholds = sorted([float(t) for t in results["thresholds"]])
    f1_vals = [results["thresholds"][str(t)]["macro_f1"] for t in thresholds]
    dsr_vals = [results["thresholds"][str(t)]["dsr"] * 100 for t in thresholds]
    opt_t = results.get("optimal_threshold")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, f1_vals, marker="o", color=_COLOR_F1,
             label="Macro F1", linewidth=2)
    ax2.plot(thresholds, dsr_vals, marker="s", color=_COLOR_DSR,
             label="DSR%", linestyle="--", linewidth=2)

    if opt_t is not None:
        ax1.axvline(x=opt_t, color="gray", linestyle=":", alpha=0.7, linewidth=1.5,
                    label=f"t* = {opt_t}")

    ax1.set_xlabel("Routing Threshold (t)")
    ax1.set_ylabel("Macro F1", color=_COLOR_F1)
    ax2.set_ylabel("Debate Skip Rate (%)", color=_COLOR_DSR)
    ax1.tick_params(axis="y", labelcolor=_COLOR_F1)
    ax2.tick_params(axis="y", labelcolor=_COLOR_DSR)
    ax1.set_xticks(THRESHOLDS)
    ax1.set_ylim(0.70, 1.0)
    ax2.set_ylim(0, 100)
    ax1.set_title(
        "Effect of Routing Threshold on Macro F1 and Debate Skip Rate",
        color="#222222",
        fontweight="semibold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=9,
        frameon=True,
    )
    ax1.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Threshold sweep chart saved → %s", out_path)


def _load_best_config_sweep() -> dict:
    """Load sweep_results.json for the best config (n3k3). Returns empty dict if missing."""
    path = PROJECT_ROOT / "reports" / "debate" / "sweep" / BEST_CONFIG / "sweep_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_out_dir(out_dir: str | None) -> Path:
    """Return resolved output directory, creating it if necessary."""
    path = Path(out_dir) if out_dir else PROJECT_ROOT / "reports" / "debate" / "sweep" / "visualizations"
    path.mkdir(parents=True, exist_ok=True)
    return path
