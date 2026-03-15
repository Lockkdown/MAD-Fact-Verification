"""Evaluate a trained PLMFactCheck checkpoint on the held-out test set."""

import logging
import platform
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

from src.data.preprocess.dataset import ViFactCheckDataset, build_dataloader
from src.models.optimizer import build_criterion
from src.models.plm_model import PLMFactCheck
from src.models.train_eval import evaluate
from src.outputs.metrics.save_metrics import save_test_summary
from src.outputs.visualizations.plot_results import save_test_plots
from src.utils.common import PROJECT_ROOT, set_seed

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]


def run_test_evaluation(cfg_path: str, device: torch.device) -> None:
    """Load best checkpoint and evaluate on test set. Saves summary_report.txt and confusion matrix."""
    resolved = Path(cfg_path) if Path(cfg_path).is_absolute() else PROJECT_ROOT / cfg_path
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    model_key       = cfg["output"]["model_key"]
    model_name      = cfg["model"]["base_model"]
    mode            = cfg["data"]["mode"]
    fp16            = cfg["training"]["fp16"]
    already_norm    = cfg["data"].get("already_normalized", False)
    num_workers     = 1 if platform.system() == "Windows" else 4

    checkpoint_path = PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / "best_model.pt"
    test_metrics_dir = PROJECT_ROOT / cfg["output"]["test_metrics_dir"]
    test_viz_dir     = PROJECT_ROOT / cfg["output"]["test_viz_dir"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run --phase train --config {cfg_path} first."
        )

    logger.info("=== Test Eval: %s | mode=%s ===", model_key, mode)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_ds = ViFactCheckDataset(
        PROJECT_ROOT / cfg["data"]["test_path"], tokenizer, model_name, mode,
        cfg["data"]["max_length"], already_norm,
    )
    test_loader = build_dataloader(
        test_ds, tokenizer, cfg["training"]["batch_size"] * 2,
        shuffle=False, workers=num_workers,
    )

    model = PLMFactCheck(
        model_name,
        cfg["model"]["num_classes"],
        cfg["model"]["dropout_rate"],
        cfg["model"]["unfreeze_last_n_layers"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion = build_criterion(cfg)

    _, macro_f1, per_label_f1, preds, labels = evaluate(
        model, test_loader, criterion, device, fp16
    )

    acc = accuracy_score(labels, preds)

    save_test_summary(test_metrics_dir, macro_f1, acc, per_label_f1)

    save_test_plots(test_viz_dir, labels, preds)

    logger.info(
        "  macro_F1=%.4f  acc=%.4f  |  Sup=%.3f  Ref=%.3f  NEI=%.3f",
        macro_f1, acc,
        per_label_f1["Support"], per_label_f1["Refute"], per_label_f1["NEI"],
    )
