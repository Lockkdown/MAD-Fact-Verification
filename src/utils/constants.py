"""Shared constants for the ViFactCheck MAD project."""

SEED = 42
NUM_CLASSES = 3

LABEL2ID: dict[str, int] = {"Support": 0, "Refute": 1, "NEI": 2}
ID2LABEL: dict[int, str] = {0: "Support", 1: "Refute", 2: "NEI"}
LABEL_NAMES: list[str] = ["Support", "Refute", "NEI"]
VALID_LABELS: set[str] = {"Support", "Refute", "NEI"}

PLM_CANDIDATES: dict[str, str] = {
    "phobert": "vinai/phobert-base",
    "xlmr":    "xlm-roberta-base",
    "mbert":   "bert-base-multilingual-cased",
    "vibert":  "FPTAI/vibert-base-cased",
}

DEBATER_MODELS: list[str] = [
    "mistralai/mistral-small-2603",
    "openai/gpt-4o-mini",
    "qwen/qwen-2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]
JUDGE_MODEL: str = "deepseek/deepseek-chat"

THRESHOLD_SWEEP: list[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
OPTIMAL_THRESHOLD: float | None = None  # filled after Phase 3d dev sweep

PANEL_SIZES: list[int] = [2, 3, 4]
DEBATE_ROUNDS: list[int] = [3, 5]
