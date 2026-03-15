"""LLRD optimizer, linear warmup scheduler, and label-smoothing criterion for PLM training."""

import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def get_llrd_optimizer_params(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    layer_decay: float = 0.9,
) -> list:
    """
    Layer-wise LR Decay for 12-layer base models.
    Embedding layer gets lr * decay^13 (lowest).
    Encoder layer i gets lr * decay^(12-i).
    Classifier + pooler get full lr.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    named_params = list(model.named_parameters())
    param_groups: list[dict] = []

    def _make_group(param_list: list, wd: float, lr: float) -> None:
        decay_p   = [p for n, p in param_list if not any(nd in n for nd in no_decay) and p.requires_grad]
        no_dec_p  = [p for n, p in param_list if     any(nd in n for nd in no_decay) and p.requires_grad]
        if decay_p:
            param_groups.append({"params": decay_p,  "weight_decay": wd,  "lr": lr})
        if no_dec_p:
            param_groups.append({"params": no_dec_p, "weight_decay": 0.0, "lr": lr})

    embed = [(n, p) for n, p in named_params if "embeddings" in n]
    _make_group(embed, weight_decay, learning_rate * (layer_decay ** 13))

    for i in range(12):
        layer_p = [(n, p) for n, p in named_params if f"encoder.layer.{i}." in n]
        _make_group(layer_p, weight_decay, learning_rate * (layer_decay ** (12 - i)))

    head = [(n, p) for n, p in named_params if "classifier" in n or "pooler" in n]
    _make_group(head, weight_decay, learning_rate)

    return param_groups


def build_optimizer(model: nn.Module, cfg: dict) -> AdamW:
    """Build AdamW optimizer — LLRD if layer_decay < 1.0, flat LR otherwise."""
    lr           = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    layer_decay  = cfg["training"]["layer_decay"]

    if layer_decay < 1.0:
        params = get_llrd_optimizer_params(model, lr, weight_decay, layer_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    return AdamW(params, lr=lr, weight_decay=weight_decay, eps=1e-8)


def build_scheduler(optimizer: AdamW, train_loader, cfg: dict):
    """Build linear warmup + linear decay scheduler."""
    accumulation_steps = cfg["training"]["accumulation_steps"]
    epochs             = cfg["training"]["epochs"]
    warmup_ratio       = cfg["training"]["warmup_ratio"]

    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = int(warmup_ratio * total_steps)

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def build_criterion(cfg: dict) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss with label smoothing from config."""
    return nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
