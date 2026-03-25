"""Cross-config debate visualizations: config comparison, grouped F1, error propagation, conformity bias."""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.outputs.metrics.cross_config_metrics import load_all_logs, load_all_metrics
from src.utils.common import PROJECT_ROOT

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]
LABEL_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
CONFIG_ORDER = ["n2k3", "n2k5", "n3k3", "n3k5", "n4k3", "n4k5"]


def generate_cross_config_plots(mode: str) -> None:
    """Load all logs and metrics for mode, then generate all 4 cross-config charts."""
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
        plot_error_propagation(logs_by_config, out_dir / "error_propagation.png")
        plot_conformity_bias(logs_by_config, out_dir / "conformity_bias.png")

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


def plot_error_propagation(logs_by_config: dict[str, list[dict]], out_path: Path) -> None:
    """Stacked bar: per-config breakdown of ✓→✓, ✗→✓, ✓→✗, ✗→✗ (R1 majority vs judge)."""
    configs = [c for c in CONFIG_ORDER if c in logs_by_config]
    categories = ["✓→✓", "✗→✓ (fixed)", "✓→✗ (broken)", "✗→✗"]
    cat_colors = ["#55A868", "#4C72B0", "#C44E52", "#BBBBBB"]

    proportions: dict[str, list[float]] = defaultdict(list)
    for cfg in configs:
        counts = _error_propagation_counts(logs_by_config[cfg])
        total = sum(counts.values()) or 1
        for cat in categories:
            proportions[cat].append(counts.get(cat, 0) / total)

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(configs))
    for cat, color in zip(categories, cat_colors):
        vals = np.array(proportions[cat])
        ax.bar(configs, vals, bottom=bottoms, label=cat, color=color, edgecolor="white")
        bottoms += vals

    ax.set_ylabel("Proportion of Samples")
    ax.set_ylim(0, 1.12)
    ax.set_title("Error Propagation (R1 Majority → Final Verdict)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Error propagation chart saved → %s", out_path)


def plot_conformity_bias(logs_by_config: dict[str, list[dict]], out_path: Path) -> None:
    """Line chart: % agents keeping R1 verdict per round (conformity bias across configs)."""
    k5_configs = [c for c in CONFIG_ORDER if c.endswith("k5") and c in logs_by_config]
    if not k5_configs:
        logger.warning("No k=5 configs found — skipping conformity bias chart.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg in k5_configs:
        by_round = _compute_conformity_by_round(logs_by_config[cfg])
        rounds = sorted(by_round.keys())
        rates = [by_round[r] * 100 for r in rounds]
        ax.plot(rounds, rates, marker="o", label=cfg)

    ax.set_xlabel("Debate Round")
    ax.set_ylabel("% Agents Keeping R1 Verdict")
    ax.set_ylim(40, 105)
    ax.set_title("Conformity Bias Across Rounds (k=5 configs)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Conformity bias chart saved → %s", out_path)


# --- private helpers ---

def _error_propagation_counts(samples: list[dict]) -> dict[str, int]:
    """Count ✓→✓, ✗→✓, ✓→✗, ✗→✗ relative to R1 majority verdict for debate-path samples."""
    counts: dict[str, int] = {"✓→✓": 0, "✗→✓ (fixed)": 0, "✓→✗ (broken)": 0, "✗→✗": 0}
    for s in samples:
        per_round = s.get("per_round_verdicts", [])
        gold = s["gold_label"]
        final = s["final_verdict"]
        if not per_round:
            # fast-path (hybrid): no R1 — classify final only
            counts["✓→✓" if final == gold else "✗→✗"] += 1
            continue
        r1_majority = _r1_majority(per_round)
        r1_correct = r1_majority == gold
        final_correct = final == gold
        if r1_correct and final_correct:
            counts["✓→✓"] += 1
        elif not r1_correct and final_correct:
            counts["✗→✓ (fixed)"] += 1
        elif r1_correct and not final_correct:
            counts["✓→✗ (broken)"] += 1
        else:
            counts["✗→✗"] += 1
    return counts


def _r1_majority(per_round_verdicts: list[dict]) -> str | None:
    """Return the majority verdict from round 1 debater votes."""
    r1_verdicts = [v["verdict"] for v in per_round_verdicts[0].get("verdicts", [])]
    if not r1_verdicts:
        return None
    return max(set(r1_verdicts), key=r1_verdicts.count)


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
