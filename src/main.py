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
        choices=["download", "preprocess", "estimate", "train", "train-all", "eval", "eval-all", "debate", "debate-all", "sweep", "plm-scores", "sweep-summary", "retry", "analyze"],
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
        "--debate-mode",
        default=None,
        choices=["full", "hybrid"],
        dest="debate_mode",
        help="Debate mode to analyze (required for --phase analyze)",
    )
    parser.add_argument(
        "--judge-f1",
        default=None,
        type=float,
        dest="judge_f1",
        help="Judge-only baseline Macro F1 for DCS computation (optional, --phase analyze)",
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
    parser.add_argument(
        "--configs-dir",
        default=None,
        dest="configs_dir",
        help="Directory containing YAML configs to run sequentially (used with --phase debate-all)",
    )
    parser.add_argument(
        "--parallel",
        default=1,
        type=int,
        dest="parallel",
        help="Number of configs to run concurrently in --phase debate-all (default: 1 = sequential)",
    )
    parser.add_argument(
        "--retry",
        default=None,
        dest="retry",
        help="Path to errors JSONL produced by DebateLogger.extract_errors() (required for --phase retry)",
    )
    parser.add_argument(
        "--output",
        default="reports/debate/sweep/plm_dev_scores.jsonl",
        dest="output",
        help="Output path for PLM scores JSONL (used with --phase plm-scores)",
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


def _run_debate_all(
    configs_dir: str,
    split_override: str | None,
    max_samples: int | None,
    parallel: int,
) -> None:
    """Run all YAML configs in a directory sequentially or with limited concurrency."""
    import asyncio
    from pathlib import Path
    from src.orchestrator.experiment_runner import run_multi_config
    from src.utils.common import PROJECT_ROOT

    config_dir_path = Path(configs_dir)
    resolved_configs_dir = (
        config_dir_path if config_dir_path.is_absolute() else PROJECT_ROOT / config_dir_path
    )
    config_paths = sorted(resolved_configs_dir.glob("*.yaml"))
    if not config_paths:
        logger.error("No YAML configs found in: %s", resolved_configs_dir)
        sys.exit(1)

    logger.info("Found %d configs in %s (parallel=%d)", len(config_paths), resolved_configs_dir, parallel)
    for p in config_paths:
        logger.info("  %s", p.name)

    asyncio.run(run_multi_config(
        config_paths=[str(p) for p in config_paths],
        device=_get_device(),
        split_override=split_override,
        max_samples=max_samples,
        max_concurrent=parallel,
    ))


def _run_sweep_summary() -> None:
    """Consolidate all per-config sweep results into a single threshold summary JSON."""
    from src.orchestrator.sweep_summary import generate_sweep_summary
    generate_sweep_summary()


def _run_plm_scores(config_path: str, output_path: str) -> None:
    """Run PhoBERT M* on DEV split and export confidence scores JSONL."""
    from src.models.infer_plm_scores import run_plm_inference
    run_plm_inference(config_path, output_path, _get_device())


def _run_sweep(config_path: str, plm_scores_path: str) -> None:
    """Run virtual threshold sweep using saved debate + PLM confidence logs."""
    from src.orchestrator.threshold_sweep import run_threshold_sweep_from_config
    run_threshold_sweep_from_config(config_path, plm_scores_path)


def _run_retry(config_path: str, errors_jsonl_path: str) -> None:
    """Retry failed samples from a previous debate run and merge results back."""
    import asyncio
    from src.orchestrator.retry_runner import run_retry_experiment
    asyncio.run(run_retry_experiment(config_path, errors_jsonl_path))


def _run_analyze(debate_mode: str, judge_f1: float | None) -> None:
    """Run cross-config analysis: summary table + 4 paper-ready charts for a debate mode."""
    from src.outputs.metrics.cross_config_metrics import run_cross_config_analysis
    from src.outputs.visualizations.plot_cross_config import generate_cross_config_plots

    run_cross_config_analysis(debate_mode, judge_only_f1=judge_f1)
    generate_cross_config_plots(debate_mode)


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

    elif args.phase == "debate-all":
        if not args.configs_dir:
            logger.error("--configs-dir is required for --phase debate-all")
            sys.exit(1)
        _run_debate_all(args.configs_dir, args.split, args.max_samples, args.parallel)

    elif args.phase == "sweep-summary":
        _run_sweep_summary()

    elif args.phase == "plm-scores":
        if not args.config:
            logger.error("--config is required for --phase plm-scores")
            sys.exit(1)
        _run_plm_scores(args.config, args.output)

    elif args.phase == "sweep":
        if not args.config:
            logger.error("--config is required for --phase sweep")
            sys.exit(1)
        if not args.plm_scores:
            logger.error("--plm-scores is required for --phase sweep")
            sys.exit(1)
        _run_sweep(args.config, args.plm_scores)

    elif args.phase == "retry":
        if not args.config:
            logger.error("--config is required for --phase retry")
            sys.exit(1)
        if not args.retry:
            logger.error("--retry is required for --phase retry")
            sys.exit(1)
        _run_retry(args.config, args.retry)

    elif args.phase == "analyze":
        if not args.debate_mode:
            logger.error("--debate-mode is required for --phase analyze")
            sys.exit(1)
        _run_analyze(args.debate_mode, args.judge_f1)


if __name__ == "__main__":
    main()