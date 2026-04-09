"""Singleton PLM service — loads PhoBERT once at startup."""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_reports_env = os.environ.get("REPORTS_DIR")
if _reports_env:
    _REPORTS_DIR = Path(_reports_env)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    _REPORTS_DIR = PROJECT_ROOT / "reports"

CHECKPOINT_PATH = _REPORTS_DIR / "models_plm" / "phobert" / "checkpoints" / "best_model.pt"
PHOBERT_HF_ID = "vinai/phobert-base"
NUM_CLASSES = 3
ID2LABEL = {0: "Support", 1: "Refute", 2: "NEI"}


class PLMFactCheck(nn.Module):
    """Generic PLM classifier — must match src/models/plm_model.py exactly."""

    def __init__(self, pretrained_name: str, num_classes: int = 3, dropout_rate: float = 0.35):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden_size = config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)


class PLMService:
    """Singleton service that loads PhoBERT checkpoint once at startup."""

    def __init__(self):
        self._model: PLMFactCheck | None = None
        self._tokenizer = None
        self.is_ready: bool = False
        self.loaded_models: list[str] = []

    def load(self) -> None:
        """Load PhoBERT model and tokenizer into memory."""
        if not CHECKPOINT_PATH.exists():
            logger.warning(
                "PhoBERT checkpoint not found at %s — falling back to mock predictions.",
                CHECKPOINT_PATH,
            )
            return

        try:
            from transformers import AutoTokenizer

            logger.info("Loading PhoBERT tokenizer from %s", PHOBERT_HF_ID)
            self._tokenizer = AutoTokenizer.from_pretrained(PHOBERT_HF_ID)
            self._model = PLMFactCheck(PHOBERT_HF_ID)
            state = torch.load(CHECKPOINT_PATH, map_location="cpu")
            weights = state.get("model_state_dict", state)
            self._model.load_state_dict(weights)
            self._model.eval()
            self.is_ready = True
            self.loaded_models = ["phobert"]
            logger.info("PhoBERT loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load PhoBERT: %s — using mock predictions.", exc)
            self._model = None
            self._tokenizer = None
            self.is_ready = False

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self.is_ready = False
        self.loaded_models = []

    def predict(self, claim: str, evidence: str) -> dict:
        """Run inference for a single (claim, evidence) pair.

        Returns dict with label, confidence, probabilities.
        """
        if not self.is_ready or self._model is None:
            raise RuntimeError("PLM service not ready — use mock mode.")

        try:
            from pyvi import ViTokenizer

            segmented_claim = ViTokenizer.tokenize(claim)
            segmented_evidence = ViTokenizer.tokenize(evidence)
        except ImportError:
            logger.warning("pyvi not installed — skipping word segmentation.")
            segmented_claim = claim
            segmented_evidence = evidence

        encoding = self._tokenizer(
            segmented_claim,
            segmented_evidence,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        label_idx = int(torch.argmax(torch.tensor(probs)).item())
        label = ID2LABEL[label_idx]
        probabilities = {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return {
            "label": label,
            "confidence": round(float(probs[label_idx]), 4),
            "probabilities": probabilities,
        }


plm_service = PLMService()
