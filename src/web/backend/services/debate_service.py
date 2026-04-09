"""Debate service — wraps real PLM prediction as SSE stream."""

import json
import logging
from typing import AsyncGenerator

from services.plm_service import plm_service

logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class DebateService:
    """Wraps PLM service into SSE stream format for real (non-mock) PLM predictions."""

    async def run_plm(self, claim: str, evidence: str) -> AsyncGenerator[str, None]:
        """Stream PLM prediction as SSE events."""
        yield _sse({"type": "plm_start", "model": "phobert"})
        try:
            result = plm_service.predict(claim, evidence)
            yield _sse({
                "type": "plm_result",
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            })
        except Exception as exc:
            logger.error("PLM prediction failed: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "done"})

    async def run_debate(
        self, claim: str, evidence: str, config: str
    ) -> AsyncGenerator[str, None]:
        """Real debate not implemented — falls back to error event."""
        yield _sse({
            "type": "error",
            "message": "Real debate requires API keys. Enable mock mode.",
        })
        yield _sse({"type": "done"})


debate_service = DebateService()
