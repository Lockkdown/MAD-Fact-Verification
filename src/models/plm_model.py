"""PLMFactCheck — generic 3-class classifier for PhoBERT, XLM-R, mBERT, ViBERT."""

import torch.nn as nn
from transformers import AutoConfig, AutoModel


class PLMFactCheck(nn.Module):
    """
    Generic PLM classifier — 3 classes: Support(0) / Refute(1) / NEI(2).
    Compatible with vinai/phobert-base, xlm-roberta-base,
    bert-base-multilingual-cased, vinai/vibert-base-cased.
    All base variants share hidden_size=768.
    """

    def __init__(
        self,
        pretrained_name: str,
        num_classes: int = 3,
        dropout_rate: float = 0.35,
        unfreeze_last_n_layers: int = 8,
    ):
        super().__init__()
        self.config_hf = AutoConfig.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.hidden_size = self.config_hf.hidden_size  # 768 for all base variants

        self._freeze_layers(unfreeze_last_n_layers)

        # CLS → LayerNorm → Dropout → Linear(768→512) → GELU → Dropout → Linear(512→num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def _freeze_layers(self, unfreeze_last_n: int) -> None:
        """Freeze all encoder layers then unfreeze last N + pooler."""
        if unfreeze_last_n <= 0:
            return

        for param in self.bert.parameters():
            param.requires_grad = False

        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        # Valid for all 4 base variants (12 encoder layers each)
        layers = self.bert.encoder.layer
        start = max(0, len(layers) - unfreeze_last_n)
        for i in range(start, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """Forward pass — returns raw logits (no softmax)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token pooling
        return self.classifier(cls)
