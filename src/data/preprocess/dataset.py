"""ViFactCheck Dataset class and DataLoader builder for PLM fine-tuning."""

import json
import platform
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

from src.data.preprocess.normalize import normalize_text

MODEL_CONFIGS: dict[str, dict] = {
    "vinai/phobert-base": {
        "use_pyvi": True,
        "max_length": {"gold_evidence": 256, "full_context": 512},
    },
    "xlm-roberta-base": {
        "use_pyvi": False,
        "max_length": {"gold_evidence": 256, "full_context": 512},
    },
    "bert-base-multilingual-cased": {
        "use_pyvi": False,
        "max_length": {"gold_evidence": 256, "full_context": 512},
    },
    "FPTAI/vibert-base-cased": {
        "use_pyvi": False,
        "max_length": {"gold_evidence": 256, "full_context": 512},
    },
}


def build_input_text(sample: dict, mode: str) -> tuple[str, str]:
    """Construct (text_a, text_b) input pair from a dataset sample based on evidence mode."""
    if mode == "gold_evidence":
        return sample["statement"], sample["evidence"]
    elif mode == "full_context":
        return sample["statement"], sample["context"]
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'gold_evidence' or 'full_context'.")


class ViFactCheckDataset(Dataset):
    """PyTorch Dataset for ViFactCheck JSONL files with model-specific preprocessing."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        model_name: str,
        mode: str,
        max_length: int | None = None,
        already_normalized: bool = False,
    ):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Expected one of: {list(MODEL_CONFIGS.keys())}"
            )

        self.tokenizer = tokenizer
        self.mode = mode
        self.config = MODEL_CONFIGS[model_name]
        self.max_length = max_length or self.config["max_length"][mode]
        self.use_pyvi = self.config["use_pyvi"]
        self.already_normalized = already_normalized  # skip normalize_text() if preprocessed

        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text_a, text_b = build_input_text(sample, self.mode)

        if not self.already_normalized:
            text_a = normalize_text(text_a, use_pyvi=self.use_pyvi)
            text_b = normalize_text(text_b, use_pyvi=self.use_pyvi)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
            encoding = self.tokenizer(
                text_a,
                text_b,
                truncation=True,
                padding=False,  # DataCollatorWithPadding handles batch-level padding
                max_length=self.max_length,
                return_tensors="pt",
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }


def build_dataloader(
    dataset: ViFactCheckDataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    workers: int | None = None,
) -> DataLoader:
    """Build a DataLoader with dynamic padding via DataCollatorWithPadding."""
    if workers is None:
        workers = 1 if platform.system() == "Windows" else 4  # deadlock fix on win32

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collator,
    )
