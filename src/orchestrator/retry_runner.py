"""RetryRunner: re-runs failed debate samples and merges results back into the main log."""

import asyncio
import json
import logging
import os
from pathlib import Path

import yaml

from src.orchestrator.debate_engine import DebateEngine
from src.outputs.metrics.debate_metrics import compute_and_save_debate_metrics
from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)


async def run_retry_experiment(config_path: str, errors_jsonl_path: str) -> None:
    """Load config, build engine, retry failed samples, merge and recompute metrics."""
    from src.orchestrator.experiment_runner import _load_samples
    from src.orchestrator.mad_builder import build_client, build_debate_engine

    cfg_path = Path(config_path)
    resolved = cfg_path if cfg_path.is_absolute() else PROJECT_ROOT / cfg_path
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_key = "dev_path" if cfg["data"]["split"] == "dev" else "test_path"
    dataset = _load_samples(cfg["data"][data_key])

    client = build_client()
    engine, debate_logger = build_debate_engine(cfg, client, cfg["output"]["log_path"])
    runner = RetryRunner(cfg, batch_size=cfg["data"]["batch_size"])

    try:
        await runner.run(
            error_jsonl_path=errors_jsonl_path,
            original_log_path=cfg["output"]["log_path"],
            debate_engine=engine,
            dataset=dataset,
        )
    finally:
        debate_logger.close()
        await client.close()


class RetryRunner:
    """Re-runs error samples from a debate log and merges results back atomically."""

    def __init__(self, cfg: dict, batch_size: int = 10):
        self.cfg = cfg
        self.batch_size = batch_size

    async def run(
        self,
        error_jsonl_path: str,
        original_log_path: str,
        debate_engine: DebateEngine,
        dataset: list[dict],
    ) -> None:
        """Retry failed samples, merge into original log, recompute metrics."""
        error_samples = self._load_error_samples(error_jsonl_path)
        if not error_samples:
            logger.info("No error samples found in %s — nothing to retry.", error_jsonl_path)
            return

        logger.info("Retrying %d failed sample(s)...", len(error_samples))
        dataset_by_id = {s["id"]: s for s in dataset}

        retry_results = await self._run_retries(error_samples, dataset_by_id, debate_engine)

        succeeded = sum(1 for r in retry_results if "error" not in r)
        still_failed = len(retry_results) - succeeded
        logger.info(
            "Retry complete: %d/%d succeeded, %d still failed",
            succeeded, len(error_samples), still_failed,
        )

        self._merge_into_log(original_log_path, retry_results)

        metrics_path = str(Path(original_log_path).parent / "metrics.json")
        compute_and_save_debate_metrics(
            log_path=original_log_path,
            metrics_path=metrics_path,
            cfg=self.cfg,
        )
        logger.info("Metrics recomputed → %s", metrics_path)

    # --- private helpers ---

    def _load_error_samples(self, error_jsonl_path: str) -> list[dict]:
        """Load all entries from the errors JSONL file."""
        path = _resolve(error_jsonl_path)
        if not path.exists():
            return []
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    async def _run_retries(
        self,
        error_samples: list[dict],
        dataset_by_id: dict[str, dict],
        debate_engine: DebateEngine,
    ) -> list[dict]:
        """Run debate_engine for each error sample concurrently via semaphore."""
        sem = asyncio.Semaphore(self.batch_size)

        async def retry_one(err: dict) -> dict:
            sample_id = str(err["sample_id"])
            sample = dataset_by_id.get(sample_id)
            if sample is None:
                logger.warning("sample_id=%s not found in dataset — skipping.", sample_id)
                return {**err, "retry_attempted": True}

            async with sem:
                result = await debate_engine.run(
                    sample_id=sample_id,
                    statement=sample["statement"],
                    evidence=sample["evidence"],
                    gold_label=sample["gold_label"],
                    mode=err.get("mode", self.cfg["debate"]["mode"]),
                    routed_to_debate=True,
                    m_star_confidence=err.get("m_star_confidence"),
                )

            if "error" in result:
                result["retry_attempted"] = True
            return result

        tasks = [retry_one(e) for e in error_samples]
        return list(await asyncio.gather(*tasks))

    def _merge_into_log(self, original_log_path: str, retry_results: list[dict]) -> None:
        """Replace error lines in original log with retry results. Atomic via rename."""
        orig = _resolve(original_log_path)

        # Index retry results by sample_id for O(1) lookup
        retry_by_id: dict[str, dict] = {str(r["sample_id"]): r for r in retry_results}

        # Read original log, substitute matched lines
        merged_lines: list[str] = []
        with open(orig, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    sample_id = str(entry.get("sample_id", ""))
                    if sample_id in retry_by_id:
                        merged_lines.append(
                            json.dumps(retry_by_id[sample_id], ensure_ascii=False) + "\n"
                        )
                    else:
                        merged_lines.append(stripped + "\n")
                except json.JSONDecodeError:
                    merged_lines.append(stripped + "\n")

        # Write to temp file, then rename atomically
        tmp_path = orig.with_suffix(".jsonl.merge_tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(merged_lines)

        os.replace(tmp_path, orig)  # atomic on POSIX; best-effort on Windows
        logger.info("Merged retry results into %s", orig)


def _resolve(path_str: str) -> Path:
    """Resolve path relative to PROJECT_ROOT if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p
