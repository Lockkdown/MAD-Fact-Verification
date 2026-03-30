"""Cross-config debate visualizations: config comparison, grouped F1, error propagation, conformity bias."""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from src.outputs.metrics.cross_config_metrics import load_all_logs, load_all_metrics
from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]
LABEL_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CONFIG_ORDER = ["n2k3", "n2k5", "n3k3", "n3k5", "n4k3", "n4k5"]


def generate_cross_config_plots(mode: str) -> None:
    """Load logs and metrics for mode, generate cross-config charts."""
    from src.outputs.visualizations.plot_analysis_extras import (
        plot_error_propagation_combined_table,
        plot_round_distribution,
    )

    metrics_by_config = load_all_metrics(mode)
    logs_by_config = load_all_logs(mode)

    if not metrics_by_config:
        logger.warning("No metrics found for mode=%s — skipping cross-config plots.", mode)
        return

    out_dir = PROJECT_ROOT / "reports" / "debate" / mode / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_config_comparison(metrics_by_config, out_dir / "config_comparison.png")
    plot_grouped_f1(metrics_by_config, out_dir / "grouped_f1.png")

    if logs_by_config:
        plot_conformity_bias(logs_by_config, out_dir / "conformity_bias.png")

        if mode == "full":
            full_logs = load_all_logs("full")
            hybrid_logs = load_all_logs("hybrid")
            plot_error_propagation_combined_table(
                full_logs, hybrid_logs, out_dir / "error_propagation.png"
            )
            plot_round_distribution(logs_by_config, out_dir / "round_distribution.png")

    logger.info("Cross-config plots saved → %s", out_dir)


def plot_config_comparison(metrics_by_config: dict[str, dict], out_path: Path) -> None:
    """Bar chart: Macro F1 across all 6 debate configs."""
    configs = [c for c in CONFIG_ORDER if c in metrics_by_config]
    f1_scores = [metrics_by_config[c]["macro_f1"] for c in configs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(configs, f1_scores, color="#4C72B0", edgecolor="white", width=0.6)
    ax.set_ylim(min(f1_scores) - 0.02, min(f1_scores + [1.0]) + 0.02)
    ax.set_xlabel("Config")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 Comparison Across Configs")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Config comparison chart saved → %s", out_path)


def plot_grouped_f1(metrics_by_config: dict[str, dict], out_path: Path) -> None:
    """Grouped bar chart: per-label F1 (Support / Refute / NEI) across all configs."""
    configs = [c for c in CONFIG_ORDER if c in metrics_by_config]
    n_configs = len(configs)
    n_labels = len(LABEL_NAMES)
    x = np.arange(n_configs)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, color) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
        scores = [metrics_by_config[c]["f1_per_label"].get(label, 0.0) for c in configs]
        offset = (i - n_labels // 2) * width + width / 2
        bars = ax.bar(x + offset, scores, width, label=label, color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-Label F1 Across Configs")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Grouped F1 chart saved → %s", out_path)



def plot_conformity_bias(logs_by_config: dict[str, list[dict]], out_path: Path) -> None:
    """Line chart: % agents keeping R1 verdict per round (conformity bias, k=5 configs)."""
    k5_configs = [c for c in CONFIG_ORDER if c.endswith("k5") and c in logs_by_config]
    if not k5_configs:
        logger.warning("No k=5 configs found — skipping conformity bias chart.")
        return

    PALETTE = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, color in zip(k5_configs, PALETTE):
        by_round = _compute_conformity_by_round(logs_by_config[cfg])
        rounds = sorted(by_round.keys())
        rates = [by_round[r] * 100 for r in rounds]
        ax.plot(rounds, rates, marker="o", linewidth=2, markersize=7,
                color=color, label=cfg, zorder=3)
        for r, v in zip(rounds, rates):
            offset = 8 if v >= 60 else -12
            ax.annotate(f"{v:.0f}%", xy=(r, v), xytext=(0, offset),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=color, fontweight="bold")

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="50% baseline", zorder=1)
    ax.set_xlabel("Debate Round", fontsize=10)
    ax.set_ylabel("% Agents Keeping R1 Verdict", fontsize=10)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"R{i}" for i in range(1, 6)])
    ax.set_ylim(0, 120)
    ax.set_title("Conformity Bias Across Rounds (k=5 configs)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Conformity bias chart saved → %s", out_path)


# --- private helpers ---

def _compute_conformity_by_round(samples: list[dict]) -> dict[int, float]:
    """For each round, return mean % agents that kept their R1 verdict."""
    round_keep: dict[int, list[float]] = defaultdict(list)
    for s in samples:
        per_round = s.get("per_round_verdicts", [])
        if not per_round:
            continue
        r1_verdicts = [v["verdict"] for v in per_round[0].get("verdicts", [])]
        n_agents = len(r1_verdicts)
        if n_agents == 0:
            continue
        round_keep[1].append(1.0)  # round 1 is always 100% by definition
        for round_data in per_round[1:]:
            r = round_data["round"]
            curr = [v["verdict"] for v in round_data.get("verdicts", [])]
            if len(curr) != n_agents:
                continue
            kept = sum(c == r1 for c, r1 in zip(curr, r1_verdicts)) / n_agents
            round_keep[r].append(kept)
    return {r: round(sum(vals) / len(vals), 4) for r, vals in sorted(round_keep.items()) if vals}
