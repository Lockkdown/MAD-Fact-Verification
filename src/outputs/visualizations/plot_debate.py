"""Generate per-experiment debate visualizations: confusion matrix and per-label F1 bar chart."""

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.common import PROJECT_ROOT

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]


def plot_debate_results(viz_dir: str, log_path: str) -> None:
    """Read logs.jsonl and save confusion matrix + per-label F1 bar chart to viz_dir."""
    golds, preds = _load_verdicts(log_path)
    if not golds:
        logger.warning("No completed samples in %s — skipping debate plots.", log_path)
        return

    out = PROJECT_ROOT / viz_dir
    out.mkdir(parents=True, exist_ok=True)

    _plot_confusion_matrix(out / "confusion_matrix.png", golds, preds)
    _plot_per_label_f1(out / "per_label_f1.png", golds, preds)
    logger.info("Debate plots saved → %s", out)


def _load_verdicts(log_path: str) -> tuple[list[str], list[str]]:
    """Load gold and predicted labels from logs.jsonl, skipping error entries."""
    path = PROJECT_ROOT / log_path
    if not path.exists():
        return [], []
    golds, preds = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "error" not in entry:
                    golds.append(entry["gold_label"])
                    preds.append(entry["final_verdict"])
            except (json.JSONDecodeError, KeyError):
                continue
    return golds, preds


def _plot_confusion_matrix(path: Path, golds: list[str], preds: list[str]) -> None:
    """Normalized confusion matrix heatmap for debate results."""
    cm = confusion_matrix(golds, preds, labels=LABEL_NAMES)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(LABEL_NAMES)),
        yticks=np.arange(len(LABEL_NAMES)),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix (normalized)",
    )
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_per_label_f1(path: Path, golds: list[str], preds: list[str]) -> None:
    """Horizontal bar chart showing per-label F1 scores."""
    scores = f1_score(golds, preds, labels=LABEL_NAMES, average=None, zero_division=0)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(LABEL_NAMES, scores, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Label F1")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
