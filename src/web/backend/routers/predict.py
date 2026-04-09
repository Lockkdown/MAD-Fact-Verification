"""POST /api/predict — SSE streaming endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from models.schemas import PredictRequest
from mock.lightweight_service import generate_lightweight_plm, generate_lightweight_debate
from services.plm_service import plm_service
from services.debate_service import debate_service

router = APIRouter()


def _is_plm_config(config: str) -> bool:
    return config == "phobert"


@router.post("/predict")
async def predict(request: PredictRequest):
    """Stream SSE events for PLM or debate inference."""
    use_mock = request.use_mock or not plm_service.is_ready

    if _is_plm_config(request.config):
        if use_mock:
            stream = generate_lightweight_plm(request.claim, request.evidence)
        else:
            stream = debate_service.run_plm(request.claim, request.evidence)
    else:
        if use_mock:
            stream = generate_lightweight_debate(request.claim, request.evidence, request.config)
        else:
            stream = debate_service.run_debate(request.claim, request.evidence, request.config)

    return StreamingResponse(stream, media_type="text/event-stream")
