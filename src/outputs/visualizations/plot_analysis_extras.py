"""Combined error-propagation table and round-distribution chart for MAD debate analysis."""

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CONFIG_ORDER = ["n2k3", "n2k5", "n3k3", "n3k5", "n4k3", "n4k5"]
_CAT_KEYS = ["✓→✓", "✗→✓ (fixed)", "✓→✗ (broken)", "✗→✗"]
_CAT_LABELS = ["✓→✓", "✗→✓\n(fixed)", "✓→✗\n(broken)", "✗→✗"]
_FULL_COLORS = ["#FFFFFF", "#C8E6C9", "#FFCDD2", "#EEEEEE"]
_HYBRID_COLORS = ["#F5F5F5", "#B3E5FC", "#FFE0B2", "#E8E8E8"]


def plot_error_propagation_combined_table(
    full_logs: dict[str, list[dict]],
    hybrid_logs: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Table: ✗→✓ (fixed) and ✓→✗ (broken) % normalized over debate-path samples."""
    configs = [c for c in CONFIG_ORDER if c in full_logs or c in hybrid_logs]
    full_counts = {c: _ep_counts(full_logs.get(c, [])) for c in configs}
    hybrid_counts = {c: _ep_counts(hybrid_logs.get(c, [])) for c in configs}

    # use total sample count (same denominator for both modes → fair comparison)
    full_total = {c: len(full_logs.get(c, [])) or 1 for c in configs}
    hybrid_total = {c: len(hybrid_logs.get(c, [])) or 1 for c in configs}

    def pct(counts: dict, key: str, n: int) -> str:
        c = counts.get(key, 0)
        return f"{c}  ({c / n * 100:.1f}%)"

    col_labels = ["Config", "Shift", "Full Debate", "Hybrid"]
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for cfg in configs:
        fn = full_total[cfg]
        hn = hybrid_total[cfg]
        cell_text.append([
            cfg,
            "✗→✓  (fixed)",
            pct(full_counts[cfg], "✗→✓ (fixed)", fn),
            pct(hybrid_counts[cfg], "✗→✓ (fixed)", hn),
        ])
        cell_colors.append(["#EEEEEE", "#C8E6C9", "#D4EDDA", "#D4EDDA"])

        cell_text.append([
            "",
            "✓→✗  (broken)",
            pct(full_counts[cfg], "✓→✗ (broken)", fn),
            pct(hybrid_counts[cfg], "✓→✗ (broken)", hn),
        ])
        cell_colors.append(["#FFFFFF", "#FFCDD2", "#FFEBEE", "#FFEBEE"])

    n_rows = len(configs) * 2
    fig, ax = plt.subplots(figsize=(8, n_rows * 0.42 + 1.4))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Verdict Shift After Debate  (% of 1,447 test samples)",
        fontsize=10, fontweight="bold", pad=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Error propagation table saved → %s", out_path)


def plot_round_distribution(logs_by_config: dict[str, list[dict]], out_path: Path) -> None:
    """Stacked bar: rounds_used distribution for debate-path samples (full debate)."""
    k3 = [c for c in ["n2k3", "n3k3", "n4k3"] if c in logs_by_config]
    k5 = [c for c in ["n2k5", "n3k5", "n4k5"] if c in logs_by_config]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_round_bars(axes[0], k3, logs_by_config, k_max=3, title="k = 3  (max 3 rounds)")
    _draw_round_bars(axes[1], k5, logs_by_config, k_max=5, title="k = 5  (max 5 rounds)")
    fig.suptitle("Debate Round Usage Distribution (Full Debate, per-debate-sample)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Round distribution chart saved → %s", out_path)


# --- private helpers ---

def _draw_round_bars(
    ax: plt.Axes,
    configs: list[str],
    logs_by_config: dict[str, list[dict]],
    k_max: int,
    title: str,
) -> None:
    """Stacked bars of % rounds_used distribution for given configs."""
    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    dist = {c: _rounds_used_dist(logs_by_config[c], k_max) for c in configs}

    x = np.arange(len(configs))
    bottoms = np.zeros(len(configs))
    for r in range(1, k_max + 1):
        vals = np.array([dist[c].get(r, 0.0) * 100 for c in configs])
        ax.bar(x, vals, bottom=bottoms, label=f"R{r}", color=COLORS[r - 1], edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("% of Debate Samples")
    ax.set_ylim(0, 107)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def _ep_counts(samples: list[dict]) -> dict[str, int]:
    """Count ✓→✓, ✗→✓ (fixed), ✓→✗ (broken), ✗→✗ for a sample list."""
    counts: dict[str, int] = {k: 0 for k in _CAT_KEYS}
    for s in samples:
        per_round = s.get("per_round_verdicts", [])
        gold, final = s["gold_label"], s["final_verdict"]
        if not per_round:
            counts["✓→✓" if final == gold else "✗→✗"] += 1
            continue
        r1 = _r1_majority(per_round)
        r1_ok, fin_ok = (r1 == gold), (final == gold)
        key = ("✓→✓" if r1_ok and fin_ok else
               "✗→✓ (fixed)" if not r1_ok and fin_ok else
               "✓→✗ (broken)" if r1_ok and not fin_ok else "✗→✗")
        counts[key] += 1
    return counts


def _r1_majority(per_round_verdicts: list[dict]) -> str | None:
    """Return majority verdict from round-1 debater votes (first occurrence wins on tie)."""
    votes = [v["verdict"] for v in per_round_verdicts[0].get("verdicts", [])]
    return max(votes, key=votes.count) if votes else None


def _rounds_used_dist(samples: list[dict], k_max: int) -> dict[int, float]:
    """% distribution of rounds_used for debate-path samples only."""
    debate = [s for s in samples if s.get("per_round_verdicts")]
    if not debate:
        return {}
    counts: dict[int, int] = defaultdict(int)
    for s in debate:
        counts[s.get("rounds_used", k_max)] += 1
    total = len(debate)
    return {r: counts[r] / total for r in range(1, k_max + 1)}
