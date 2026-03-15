"""Save PLM training results to reports/models_plm/{model_key}/metrics/."""

import json
import logging
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]


def save_best_metrics(
    metrics_dir: Path,
    history: list[dict],
    per_label_f1: dict,
    preds: list[int],
    labels: list[int],
    best_smart_score: float,
) -> None:
    """Write best_model_metrics.json, training_history.json, summary_report.txt."""
    metrics_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        digits=4,
        output_dict=True,
    )
    cm = confusion_matrix(labels, preds).tolist()
    best_epoch = max(history, key=lambda e: e["smart_score"])

    best_metrics = {
        "best_epoch":      best_epoch["epoch"],
        "best_dev_f1":     best_epoch["dev_f1"],
        "best_smart_score": best_smart_score,
        "overfit_gap":     best_epoch["train_f1"] - best_epoch["dev_f1"],
        "per_label_f1":    per_label_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    _write_json(metrics_dir / "best_model_metrics.json", best_metrics)
    _write_json(metrics_dir / "training_history.json", history)
    _write_summary(metrics_dir / "summary_report.txt", best_metrics)

    logger.info("Metrics saved → %s", metrics_dir)


def save_test_summary(
    metrics_dir: Path,
    macro_f1: float,
    accuracy: float,
    per_label_f1: dict,
) -> None:
    """Write summary_report.txt for test set evaluation."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_test_summary(metrics_dir / "summary_report.txt", macro_f1, accuracy, per_label_f1)
    logger.info("Test metrics saved → %s", metrics_dir)


def _write_json(path: Path, data) -> None:
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_summary(path: Path, m: dict) -> None:
    """Write human-readable summary for thesis table."""
    lines = [
        "=" * 52,
        "  PLM Fine-tuning Results",
        "=" * 52,
        f"  Best epoch    : {m['best_epoch']}",
        f"  Dev Macro F1  : {m['best_dev_f1']:.4f}",
        f"  Smart Score   : {m['best_smart_score']:.4f}",
        f"  Overfit gap   : {m['overfit_gap']:.4f}",
        "-" * 52,
        "  Per-label F1:",
    ]
    for label, f1 in m["per_label_f1"].items():
        lines.append(f"    {label:<10} : {f1:.4f}")
    lines += ["=" * 52, ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_test_summary(path: Path, macro_f1: float, accuracy: float, per_label_f1: dict) -> None:
    """Write human-readable test summary for thesis table."""
    lines = [
        "=" * 52,
        "  PLM Test Evaluation Results",
        "=" * 52,
        f"  Test Macro F1 : {macro_f1:.4f}",
        f"  Test Accuracy : {accuracy:.4f}",
        "-" * 52,
        "  Per-label F1:",
    ]
    for label, f1 in per_label_f1.items():
        lines.append(f"    {label:<10} : {f1:.4f}")
    lines += ["=" * 52, ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
