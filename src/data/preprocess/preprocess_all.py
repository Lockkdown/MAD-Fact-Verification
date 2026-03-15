"""Preprocess raw JSONL for all 4 PLM models and write to src/data/preprocessed/."""

import json
import logging
import sys
from pathlib import Path

from src.data.preprocess.normalize import normalize_text
from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "src" / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "src" / "data" / "preprocessed"
SPLITS = ["train", "dev", "test"]

MODEL_PREPROCESS_CONFIG: dict[str, bool] = {
    "phobert": True,   # PyVi word segmentation required
    "xlmr":    False,  # raw UTF-8 text only
    "mbert":   False,
    "vibert":  False,
}


def preprocess_split(raw_path: Path, out_path: Path, use_pyvi: bool) -> int:
    """Read raw JSONL, normalize all text fields, write preprocessed JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(raw_path, encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["statement"] = normalize_text(record["statement"], use_pyvi=use_pyvi)
            record["evidence"]  = normalize_text(record["evidence"],  use_pyvi=use_pyvi)
            record["context"]   = normalize_text(record["context"],   use_pyvi=use_pyvi)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def preprocess_model(model_key: str, use_pyvi: bool) -> None:
    """Preprocess all 3 splits for a single model."""
    for split in SPLITS:
        raw_path = RAW_DIR / f"vifactcheck_{split}.jsonl"
        if not raw_path.exists():
            logger.error("Raw file not found: %s — run --phase download first.", raw_path)
            raise FileNotFoundError(f"Missing raw file: {raw_path}")

        out_path = PREPROCESSED_DIR / model_key / f"vifactcheck_{split}.jsonl"
        count = preprocess_split(raw_path, out_path, use_pyvi)
        logger.info(
            "  [%s | %s] %d samples → %s",
            model_key, split, count,
            out_path.relative_to(PROJECT_ROOT),
        )


def preprocess_all() -> None:
    """Preprocess raw JSONL for all 4 models and write to src/data/preprocessed/."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Starting preprocessing for %d models...", len(MODEL_PREPROCESS_CONFIG))
    for model_key, use_pyvi in MODEL_PREPROCESS_CONFIG.items():
        logger.info("Model: %s | use_pyvi=%s", model_key, use_pyvi)
        preprocess_model(model_key, use_pyvi)

    logger.info("Preprocessing complete → %s", PREPROCESSED_DIR.relative_to(PROJECT_ROOT))
