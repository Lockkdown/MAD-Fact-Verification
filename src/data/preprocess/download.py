"""Download ViFactCheck from HuggingFace and export to JSONL files."""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "src" / "data" / "raw"

EXPECTED_SIZES = {"train": 5060, "dev": 723, "test": 1447}

SPLIT_MAP = {"train": "train", "dev": "dev", "test": "test"}

LABEL_MAP = {0: "Support", 1: "Refute", 2: "NEI"}


def _export_split(split_data, out_path: Path, split_name: str) -> None:
    """Write one dataset split to a JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_counter: Counter = Counter()

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(split_data):
            label_val = row.get("labels", row.get("label", -1))
            record = {
                "sample_id": str(row.get("index", idx)),
                "statement": row["Statement"],
                "evidence": row["Evidence"],
                "context": row["Context"],
                "label": int(label_val),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            label_counter[int(label_val)] += 1

    dist = {LABEL_MAP.get(k, k): v for k, v in sorted(label_counter.items())}
    logger.info("  [%s] %d samples | distribution: %s", split_name, len(split_data), dist)


def _verify_sizes(dataset) -> None:
    """Assert all three splits match expected sample counts."""
    hf_key_map = {v: k for k, v in SPLIT_MAP.items()}
    for hf_key, split_name in hf_key_map.items():
        if hf_key not in dataset:
            raise ValueError(f"Missing split '{hf_key}' in downloaded dataset.")
        actual = len(dataset[hf_key])
        expected = EXPECTED_SIZES[split_name]
        if actual != expected:
            logger.warning(
                "[%s] size mismatch: expected %d, got %d (dataset may have been updated).",
                split_name, expected, actual,
            )
        else:
            logger.info("  [%s] size verified: %d samples ✓", split_name, actual)


def download_and_export() -> None:
    """Download tranthaihoa/vifactcheck and export all splits to JSONL."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Downloading ViFactCheck dataset from HuggingFace...")

    try:
        dataset = load_dataset("tranthaihoa/vifactcheck")
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        raise

    logger.info("Verifying split sizes...")
    _verify_sizes(dataset)

    logger.info("Exporting JSONL files to %s", RAW_DIR)
    for split_name, hf_key in SPLIT_MAP.items():
        out_path = RAW_DIR / f"vifactcheck_{split_name}.jsonl"
        _export_split(dataset[hf_key], out_path, split_name)
        logger.info("  Saved → %s", out_path.relative_to(PROJECT_ROOT))

    logger.info("Download complete.")
