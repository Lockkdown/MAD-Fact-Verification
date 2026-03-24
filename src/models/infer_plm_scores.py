"""Run PhoBERT M* inference on DEV set and save per-sample confidence scores JSONL."""

import json
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.plm_model import PLMFactCheck
from src.orchestrator.routing_gate import RoutingGate
from src.utils.common import PROJECT_ROOT
from src.utils.constants import ID2LABEL, NUM_CLASSES, PLM_CANDIDATES

logger = logging.getLogger(__name__)


def _load_dev_samples(data_path: str) -> list[dict]:
    """Load DEV JSONL → list of {id, statement, evidence, gold_label}."""
    samples = []
    with open(PROJECT_ROOT / data_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            samples.append({
                "id": str(r["sample_id"]),
                "statement": r["statement"],
                "evidence": r["evidence"],
                "gold_label": ID2LABEL[r["label"]],
            })
    return samples


def _build_gate(checkpoint_path: str, device: torch.device) -> RoutingGate:
    """Load PhoBERT M* checkpoint into RoutingGate (threshold=0.0 — unused for scoring)."""
    model_name = PLM_CANDIDATES["phobert"]
    resolved = PROJECT_ROOT / checkpoint_path
    if not resolved.exists():
        raise FileNotFoundError(f"PLM checkpoint not found: {resolved}")

    model = PLMFactCheck(
        pretrained_name=model_name,
        num_classes=NUM_CLASSES,
        dropout_rate=0.35,
        unfreeze_last_n_layers=0,
    )
    ckpt = torch.load(resolved, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return RoutingGate(plm_model=model, tokenizer=tokenizer, threshold=0.0, device=device)


def run_plm_inference(config_path: str, output_path: str, device: torch.device) -> None:
    """Run PhoBERT M* on DEV split and save per-sample confidence scores JSONL.

    Output schema per line: {sample_id, confidence, plm_verdict, gold_label}
    """
    resolved = Path(config_path) if Path(config_path).is_absolute() else PROJECT_ROOT / config_path
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    samples = _load_dev_samples(cfg["data"]["dev_path"])
    gate = _build_gate(cfg["routing"]["plm_checkpoint"], device)

    out_path = PROJECT_ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("PhoBERT M* inference on %d DEV samples → %s", len(samples), out_path)
    with open(out_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="PLM scores", unit="sample"):
            plm_verdict, confidence = gate.predict(sample["statement"], sample["evidence"])
            record = {
                "sample_id": sample["id"],
                "confidence": round(confidence, 6),
                "plm_verdict": plm_verdict,
                "gold_label": sample["gold_label"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Done. PLM scores → %s", out_path)
