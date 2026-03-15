"""Training loop for PLM fact-checking classifiers with LLRD and Smart Score checkpointing."""

import gc
import logging
import platform
from pathlib import Path

import torch
import yaml
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.preprocess.dataset import ViFactCheckDataset, build_dataloader
from src.models.optimizer import build_criterion, build_optimizer, build_scheduler
from src.models.plm_model import PLMFactCheck
from src.outputs.metrics.save_metrics import save_best_metrics
from src.outputs.visualizations.plot_results import save_plots
from src.utils.common import PROJECT_ROOT, set_seed

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Support", "Refute", "NEI"]


def train_one_epoch(
    model: PLMFactCheck,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    criterion,
    device: torch.device,
    cfg: dict,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, macro_f1)."""
    model.train()
    accumulation_steps = cfg["training"]["accumulation_steps"]
    fp16 = cfg["training"]["fp16"]
    total_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(loader, desc="  train", leave=False, ncols=90, unit="batch")
    for step, batch in enumerate(bar):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=fp16):
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
        bar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def evaluate(
    model: PLMFactCheck,
    loader,
    criterion,
    device: torch.device,
    fp16: bool,
) -> tuple[float, float, dict, list, list]:
    """Evaluate on dev/test. Returns (loss, macro_f1, per_label_f1, preds, labels)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", leave=False, ncols=90, unit="batch"):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=fp16):
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_label_f1 = {
        name: f1_score(all_labels, all_preds, average=None, zero_division=0)[i]
        for i, name in enumerate(LABEL_NAMES)
    }
    return avg_loss, macro_f1, per_label_f1, all_preds, all_labels


def _compute_smart_score(dev_f1: float, train_f1: float, tolerance: float = 0.10) -> float:
    """Smart Score = dev_f1 - max(0, overfit_gap - tolerance)."""
    penalty = max(0.0, (train_f1 - dev_f1) - tolerance)
    return dev_f1 - penalty


def run_training(cfg_path: str, device: torch.device) -> None:
    """Load config and run the full training pipeline for one PLM model."""
    resolved = Path(cfg_path) if Path(cfg_path).is_absolute() else PROJECT_ROOT / cfg_path
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    model_name  = cfg["model"]["base_model"]
    mode        = cfg["data"]["mode"]
    model_key   = cfg["output"]["model_key"]
    fp16        = cfg["training"]["fp16"]
    epochs      = cfg["training"]["epochs"]
    patience    = cfg["training"]["patience"]

    checkpoint_dir = PROJECT_ROOT / cfg["output"]["checkpoint_dir"]
    metrics_dir    = PROJECT_ROOT / cfg["output"]["train_metrics_dir"]
    viz_dir        = PROJECT_ROOT / cfg["output"]["train_viz_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    logger.info("=== Training %s | mode=%s ===", model_key, mode)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    already_normalized = cfg["data"].get("already_normalized", False)
    num_workers = 1 if platform.system() == "Windows" else 4

    train_ds = ViFactCheckDataset(
        PROJECT_ROOT / cfg["data"]["train_path"], tokenizer, model_name, mode,
        cfg["data"]["max_length"], already_normalized,
    )
    dev_ds = ViFactCheckDataset(
        PROJECT_ROOT / cfg["data"]["dev_path"], tokenizer, model_name, mode,
        cfg["data"]["max_length"], already_normalized,
    )

    batch_size   = cfg["training"]["batch_size"]
    train_loader = build_dataloader(train_ds, tokenizer, batch_size,     shuffle=True,  workers=num_workers)
    dev_loader   = build_dataloader(dev_ds,   tokenizer, batch_size * 2, shuffle=False, workers=num_workers)

    model     = PLMFactCheck(model_name, cfg["model"]["num_classes"],
                             cfg["model"]["dropout_rate"], cfg["model"]["unfreeze_last_n_layers"])
    model     = model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, train_loader, cfg)
    criterion = build_criterion(cfg)
    scaler    = GradScaler(enabled=fp16)

    best_smart_score  = -1.0
    patience_counter  = 0
    history: list[dict] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"{model_key}", ncols=90, unit="ep")
    for epoch in epoch_bar:
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device, cfg
        )
        dev_loss, dev_f1, per_label_f1, dev_preds, dev_labels = evaluate(
            model, dev_loader, criterion, device, fp16
        )
        smart_score = _compute_smart_score(dev_f1, train_f1)
        overfit_gap = train_f1 - dev_f1

        history.append({
            "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
            "train_f1": train_f1, "dev_f1": dev_f1, "smart_score": smart_score,
        })

        saved = ""
        if smart_score > best_smart_score:
            best_smart_score = smart_score
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_smart_score": best_smart_score,
                "best_dev_f1": dev_f1,
                "train_f1": train_f1,
                "overfit_gap": overfit_gap,
            }, checkpoint_path)
            saved = " ✓"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"  Early stop @ epoch {epoch}")
                break

        f1_str = " ".join(f"{k[:3]}={v:.3f}" for k, v in per_label_f1.items())
        tqdm.write(
            f"  Ep {epoch}/{epochs} | "
            f"loss {train_loss:.4f}/{dev_loss:.4f} | "
            f"F1 {train_f1:.4f}/{dev_f1:.4f} | "
            f"{f1_str} | smart={smart_score:.4f}{saved}"
        )

        gc.collect()
        torch.cuda.empty_cache()

    save_best_metrics(metrics_dir, history, per_label_f1, dev_preds, dev_labels, best_smart_score)
    save_plots(viz_dir, history, dev_labels, dev_preds)
    logger.info("Training complete → checkpoint: %s", checkpoint_path.relative_to(PROJECT_ROOT))
