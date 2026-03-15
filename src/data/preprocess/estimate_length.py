"""Estimate optimal max_length per model × mode from training data (P95)."""

import json
import logging
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _build_input_text(sample: dict, mode: str) -> tuple[str, str]:
    """Return (text_a, text_b) pair for a sample based on experiment mode."""
    if mode == "gold_evidence":
        return sample["statement"], sample["evidence"]
    elif mode == "full_context":
        return sample["statement"], sample["context"]
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'gold_evidence' or 'full_context'.")


def estimate_max_length(
    jsonl_path: str | Path,
    tokenizer: Any,
    mode: str,
    percentile: float = 95.0,
    seed: int = 42,
    sample_size: int = 5000,
) -> int:
    """Estimate recommended max_length from P95 token length on a dataset split.

    Clips result to [128, 512] and prints P50 / P95 / max for logging.
    """
    random.seed(seed)

    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    sampled = random.sample(samples, min(sample_size, len(samples)))

    lengths = []
    for item in sampled:
        text_a, text_b = _build_input_text(item, mode)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
            tokens = tokenizer.encode(
                text_a, text_b, truncation=False, add_special_tokens=True
            )
        lengths.append(len(tokens))

    arr = np.array(lengths)
    p50 = int(np.percentile(arr, 50))
    p95 = int(np.percentile(arr, percentile))
    estimated = int(np.clip(p95, 128, 512))

    logger.info(
        "[%s | %s] P50=%d | P95=%d | max=%d → recommended max_length=%d",
        mode,
        getattr(tokenizer, "name_or_path", "tokenizer"),
        p50,
        p95,
        int(arr.max()),
        estimated,
    )
    return estimated
