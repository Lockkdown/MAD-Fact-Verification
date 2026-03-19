"""MAD system factory — builds DebateEngine and RoutingGate from YAML config."""

import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.api.openrouter_client import OPENROUTER_API_KEY, OpenRouterClient
from src.models.plm_model import PLMFactCheck
from src.orchestrator.agents.debater import DebaterAgent
from src.orchestrator.agents.judge import JudgeAgent
from src.orchestrator.debate_engine import DebateEngine
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.routing_gate import RoutingGate
from src.outputs.metrics.debate_logger import DebateLogger
from src.utils.common import PROJECT_ROOT
from src.utils.constants import NUM_CLASSES


def build_client() -> OpenRouterClient:
    """Return shared async OpenRouter client loaded from .env."""
    return OpenRouterClient(api_key=OPENROUTER_API_KEY)


def build_debate_engine(
    cfg: dict,
    client: OpenRouterClient,
    output_path: str,
) -> tuple[DebateEngine, DebateLogger]:
    """Build DebateEngine and DebateLogger from YAML config dict."""
    panel = cfg["debate"]["panel"]

    debaters = [
        DebaterAgent(
            agent_id=d["agent_id"],
            model=d["model"],
            client=client,
        )
        for d in panel["debaters"]
    ]

    judge = JudgeAgent(model=panel["judge"]["model"], client=client)
    orchestrator = Orchestrator(debaters=debaters, judge=judge)

    debate_logger = DebateLogger(str(PROJECT_ROOT / output_path))
    engine = DebateEngine(
        orchestrator=orchestrator,
        logger=debate_logger,
        k_max=cfg["debate"]["rounds"],
        early_stopping=cfg["debate"]["early_stopping"],
    )
    return engine, debate_logger


def build_routing_gate(cfg: dict, device: torch.device) -> RoutingGate:
    """Load PhoBERT M* checkpoint and wrap in RoutingGate.

    Raises ValueError if threshold is null — run Phase 3d sweep first.
    Raises FileNotFoundError if checkpoint is missing.
    """
    routing = cfg["routing"]
    threshold = routing["threshold"]
    if threshold is None:
        raise ValueError(
            "routing.threshold is null — run Phase 3d threshold sweep first, "
            "then fill t* into the hybrid YAML config."
        )

    model_name: str = routing["plm_model"]
    checkpoint_path = PROJECT_ROOT / routing["plm_checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PLM checkpoint not found: {checkpoint_path}")

    model = PLMFactCheck(
        pretrained_name=model_name,
        num_classes=NUM_CLASSES,
        dropout_rate=0.35,         # matches training config — dropout disabled in eval mode
        unfreeze_last_n_layers=0,  # inference only — no grad tracking needed
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return RoutingGate(plm_model=model, tokenizer=tokenizer, threshold=float(threshold), device=device)
