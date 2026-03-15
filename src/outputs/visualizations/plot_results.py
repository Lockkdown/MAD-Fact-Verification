"""Generate confusion matrix and training curve plots for PLM experiments."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")  # non-interactive backend — no display required

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]


def save_plots(
    viz_dir: Path,
    history: list[dict],
    labels: list[int],
    preds: list[int],
) -> None:
    """Save confusion matrix, loss curve, and F1 curve to viz_dir."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(viz_dir / "confusion_matrix.png", labels, preds)
    _plot_loss_curve(viz_dir / "loss_curve.png", history)
    _plot_f1_curve(viz_dir / "f1_curve.png", history)
    logger.info("Plots saved → %s", viz_dir)


def save_test_plots(viz_dir: Path, labels: list[int], preds: list[int]) -> None:
    """Save test confusion matrix to viz_dir/confusion_matrix.png."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(viz_dir / "confusion_matrix.png", labels, preds)
    logger.info("Test plots saved → %s", viz_dir)


def _plot_confusion_matrix(path: Path, labels: list[int], preds: list[int]) -> None:
    """Normalized confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

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
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_loss_curve(path: Path, history: list[dict]) -> None:
    """Train vs dev loss per epoch."""
    epochs     = [e["epoch"]      for e in history]
    train_loss = [e["train_loss"] for e in history]
    dev_loss   = [e["dev_loss"]   for e in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax.plot(epochs, dev_loss,   label="Dev Loss",   marker="s")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training & Dev Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_f1_curve(path: Path, history: list[dict]) -> None:
    """Train vs dev macro F1 per epoch."""
    epochs   = [e["epoch"]    for e in history]
    train_f1 = [e["train_f1"] for e in history]
    dev_f1   = [e["dev_f1"]   for e in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_f1, label="Train Macro F1", marker="o")
    ax.plot(epochs, dev_f1,   label="Dev Macro F1",   marker="s")
    ax.set(xlabel="Epoch", ylabel="Macro F1", title="Training & Dev Macro F1", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
