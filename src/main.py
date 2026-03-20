"""Entry point — orchestrates data pipeline phases via --phase argument."""

import argparse
import logging
import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="ViFactCheck MAD pipeline")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["download", "preprocess", "estimate", "train", "train-all", "eval", "eval-all", "debate", "sweep"],
        help="Pipeline phase to run",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment YAML config (required for --phase estimate/train)",
    )
    parser.add_argument(
        "--mode",
        default="gold_evidence",
        choices=["gold_evidence", "full_context"],
        help="Input mode for estimate phase (default: gold_evidence)",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["dev", "test"],
        help="Override data split from YAML (used with --phase debate)",
    )
    parser.add_argument(
        "--plm-scores",
        default=None,
        dest="plm_scores",
        help="Path to PLM confidence JSONL (required for --phase sweep)",
    )
    parser.add_argument(
        "--max-samples",
        default=None,
        type=int,
        dest="max_samples",
        help="Limit number of samples (smoke test). Omit for full run.",
    )
    return parser.parse_args()


def _run_download() -> None:
    """Download ViFactCheck dataset and export JSONL splits."""
    from src.data.preprocess.download import download_and_export
    download_and_export()


def _run_preprocess() -> None:
    """Preprocess raw JSONL for all 4 models → src/data/preprocessed/."""
    from src.data.preprocess.preprocess_all import preprocess_all
    preprocess_all()


def _run_estimate(config_path: str, mode: str) -> None:
    """Estimate max_length for a given model config and input mode."""
    import yaml
    from transformers import AutoTokenizer

    from src.data.preprocess.estimate_length import estimate_max_length
    from src.utils.common import PROJECT_ROOT

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["base_model"]
    train_path = PROJECT_ROOT / cfg["data"]["train_path"]

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    estimate_max_length(
        jsonl_path=train_path,
        tokenizer=tokenizer,
        mode=mode,
    )


def _get_device() -> "torch.device":
    """Return CUDA device if available, else CPU."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    return device


def _run_train(config_path: str) -> None:
    """Fine-tune a single PLM model specified by config YAML."""
    from src.models.train_eval import run_training
    run_training(config_path, _get_device())


def _run_train_all() -> None:
    """Fine-tune all 4 PLM models sequentially using their default configs."""
    from src.models.train_eval import run_training
    from src.utils.common import PROJECT_ROOT

    configs = [
        PROJECT_ROOT / "src/config/experiments/plm/phobert_gold_evidence.yaml",
        PROJECT_ROOT / "src/config/experiments/plm/xlmr_gold_evidence.yaml",
        PROJECT_ROOT / "src/config/experiments/plm/mbert_gold_evidence.yaml",
        PROJECT_ROOT / "src/config/experiments/plm/vibert_gold_evidence.yaml",
    ]
    device = _get_device()
    for cfg_path in configs:
        logger.info("--- Starting: %s ---", cfg_path.name)
        run_training(str(cfg_path), device)
    logger.info("=== All 4 models trained ===")


def _run_eval(config_path: str) -> None:
    """Evaluate best checkpoint of a single model on the test set."""
    from src.models.evaluate_test import run_test_evaluation
    run_test_evaluation(config_path, _get_device())


def _run_eval_all() -> None:
    """Evaluate all 4 PLM checkpoints on the test set sequentially."""
    from src.models.evaluate_test import run_test_evaluation
    from src.utils.common import PROJECT_ROOT

    configs = [
        PROJECT_ROOT / "src/config/experiments/plm/phobert_gold_evidence.yaml",
        PROJECT_ROOT / "src/config/experiments/plm/xlmr_gold_evidence.yaml",
        PROJECT_ROOT / "src/config/experiments/plm/mbert_gold_evidence.yaml",
        PROJECT_ROOT / "src/config/experiments/plm/vibert_gold_evidence.yaml",
    ]
    device = _get_device()
    for cfg_path in configs:
        logger.info("--- Eval: %s ---", cfg_path.name)
        run_test_evaluation(str(cfg_path), device)
    logger.info("=== All 4 models evaluated ===")


def _run_debate(config_path: str, split_override: str | None, max_samples: int | None) -> None:
    """Run full or hybrid debate experiment on configured split."""
    import asyncio
    from src.orchestrator.experiment_runner import run_debate_experiment
    asyncio.run(run_debate_experiment(config_path, _get_device(), split_override, max_samples))


def _run_sweep(config_path: str, plm_scores_path: str) -> None:
    """Run virtual threshold sweep using saved debate + PLM confidence logs."""
    from src.orchestrator.threshold_sweep import run_threshold_sweep_from_config
    run_threshold_sweep_from_config(config_path, plm_scores_path)


def main() -> None:
    """Parse args and dispatch to the correct pipeline phase."""
    args = _parse_args()

    if args.phase == "download":
        _run_download()

    elif args.phase == "preprocess":
        _run_preprocess()

    elif args.phase == "estimate":
        if not args.config:
            logger.error("--config is required for --phase estimate")
            sys.exit(1)
        _run_estimate(args.config, args.mode)

    elif args.phase == "train":
        if not args.config:
            logger.error("--config is required for --phase train")
            sys.exit(1)
        _run_train(args.config)

    elif args.phase == "train-all":
        _run_train_all()

    elif args.phase == "eval":
        if not args.config:
            logger.error("--config is required for --phase eval")
            sys.exit(1)
        _run_eval(args.config)

    elif args.phase == "eval-all":
        _run_eval_all()

    elif args.phase == "debate":
        if not args.config:
            logger.error("--config is required for --phase debate")
            sys.exit(1)
        _run_debate(args.config, args.split, args.max_samples)

    elif args.phase == "sweep":
        if not args.config:
            logger.error("--config is required for --phase sweep")
            sys.exit(1)
        if not args.plm_scores:
            logger.error("--plm-scores is required for --phase sweep")
            sys.exit(1)
        _run_sweep(args.config, args.plm_scores)


if __name__ == "__main__":
    main()