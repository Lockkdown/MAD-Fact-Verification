"""POST /api/predict — SSE streaming endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from models.schemas import PredictRequest
from services.debate_service import debate_service

router = APIRouter()


@router.post("/predict")
async def predict(request: PredictRequest):
    """Stream SSE events for PLM or debate inference."""
    if request.config == "phobert":
        stream = debate_service.run_plm(request.claim, request.evidence)
    else:
        stream = debate_service.run_debate(request.claim, request.evidence, request.config)

    return StreamingResponse(stream, media_type="text/event-stream")
