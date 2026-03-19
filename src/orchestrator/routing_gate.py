"""RoutingGate: PLM confidence-based router for hybrid debate mode."""

import torch
import torch.nn.functional as F

from src.utils.constants import ID2LABEL


class RoutingGate:
    """Routes a sample to fast-path or debate based on PLM softmax confidence."""

    def __init__(self, plm_model: torch.nn.Module, threshold: float):
        self.model = plm_model
        self.threshold = threshold
        self.model.eval()

    @torch.no_grad()
    def route(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Return routing decision for one sample (batch_size=1).

        Returns dict with keys: route_to_debate, plm_verdict, confidence.
        """
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
