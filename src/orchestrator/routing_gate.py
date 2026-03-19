"""RoutingGate: PLM confidence-based router for hybrid debate mode."""

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from src.data.preprocess.normalize import normalize_text
from src.utils.constants import ID2LABEL


class RoutingGate:
    """Routes a sample to fast-path or debate based on PLM softmax confidence."""

    def __init__(
        self,
        plm_model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        threshold: float,
        device: torch.device,
    ):
        self.model = plm_model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.device = device
        self.model.eval()

    def predict(self, statement: str, evidence: str) -> tuple[str, float]:
        """Apply Pyvi, tokenize, and infer — returns (plm_verdict, confidence).

        Callers must pass Tier 1 clean text; Pyvi is applied internally so
        input distribution matches the PhoBERT checkpoint training data.
        """
        seg_statement = normalize_text(statement, use_pyvi=True)
        seg_evidence = normalize_text(evidence, use_pyvi=True)
        encoded = self.tokenizer(
            seg_statement, seg_evidence,
            truncation=True, max_length=256, return_tensors="pt",
        )
        result = self._route(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )
        return result["plm_verdict"], result["confidence"]

    @torch.no_grad()
    def _route(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Tensor-level inference — returns route_to_debate, plm_verdict, confidence."""
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)
        confidence_val: float = confidence.item()
        pred_label: str = ID2LABEL[pred_idx.item()]

        return {
            "route_to_debate": confidence_val < self.threshold,
            "plm_verdict": pred_label,
            "confidence": confidence_val,
        }
