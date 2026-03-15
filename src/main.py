"""Entry point — orchestrates data pipeline phases via --phase argument."""

import argparse
import logging
import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

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
        choices=["download", "preprocess", "estimate"],
        help="Pipeline phase to run",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment YAML config (required for --phase estimate)",
    )
    parser.add_argument(
        "--mode",
        default="gold_evidence",
        choices=["gold_evidence", "full_context"],
        help="Input mode for estimate phase (default: gold_evidence)",
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


if __name__ == "__main__":
    main()