"""Async experiment runner for MAD debate with semaphore concurrency and checkpoint/resume."""

import asyncio
import gc
import json
import logging
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.outputs.metrics.debate_logger import DebateLogger
from src.outputs.metrics.debate_metrics import compute_and_save_debate_metrics
from src.outputs.visualizations.plot_debate import plot_debate_results
from src.orchestrator.debate_engine import DebateEngine
from src.orchestrator.mad_builder import build_client, build_debate_engine, build_routing_gate
from src.orchestrator.routing_gate import RoutingGate
from src.utils.common import PROJECT_ROOT
from src.utils.constants import ID2LABEL

logger = logging.getLogger(__name__)


def load_checkpoint(log_path: str) -> set[str]:
    """Return set of already-processed sample_ids from an existing log JSONL."""
    p = PROJECT_ROOT / log_path
    if not p.exists():
        return set()
    done: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(str(r["sample_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _load_samples(data_path: str) -> list[dict]:
    """Load debate samples from preprocessed JSONL → [{id, statement, evidence, gold_label}]."""
    samples = []
    with open(PROJECT_ROOT / data_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            samples.append({
                "id": str(r["sample_id"]),
                "statement": r["statement"],
                "evidence": r["evidence"],
                "gold_label": ID2LABEL[r["label"]],
            })
    return samples


def _make_fast_path_result(sample: dict, routing: dict, mode: str) -> dict:
    """Result dict for hybrid fast-path (high PLM confidence — debate skipped)."""
    return {
        "sample_id": sample["id"],
        "gold_label": sample["gold_label"],
        "mode": mode,
        "n_debaters": 0,
        "k_max": 0,
        "routed_to_debate": False,
        "m_star_confidence": routing["confidence"],
        "rounds_used": 0,
        "num_agent_calls": 0,
        "unanimous_at_round": None,
        "per_round_verdicts": [],
        "judge_called": False,
        "judge_reasoning": None,
        "final_verdict": routing["plm_verdict"],
        "correct": routing["plm_verdict"] == sample["gold_label"],
    }


async def run_debate_on_split(
    engine: DebateEngine,
    samples: list[dict],
    mode: str,
    gate: "RoutingGate | None" = None,
    batch_size: int = 10,
) -> list[dict]:
    """Concurrent debate runner: semaphore + random stagger per sample."""
    results: list[dict] = []
    sem = asyncio.Semaphore(batch_size)

    async def process_one(sample: dict) -> dict:
        async with sem:
            await asyncio.sleep(random.uniform(0.0, 0.5))  # stagger — avoids 429 burst

            if gate is not None:
                plm_verdict, confidence = gate.predict(sample["statement"], sample["evidence"])
                if confidence >= gate.threshold:  # fast path — high confidence
                    return _make_fast_path_result(
                        sample,
                        {"plm_verdict": plm_verdict, "confidence": confidence},
                        mode,
                    )
                m_conf: float | None = confidence
            else:
                m_conf = None

            return await engine.run(
                sample_id=str(sample["id"]),
                statement=sample["statement"],
                evidence=sample["evidence"],
                gold_label=sample["gold_label"],
                mode=mode,
                routed_to_debate=True,
                m_star_confidence=m_conf,
            )

    tasks = [process_one(s) for s in samples]
    with tqdm(total=len(samples), desc="Debating", unit="sample") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
            except BaseException as exc:  # prevent one task failure from killing the runner
                logger.error("Task raised unhandled exception — sample skipped: %s", exc)
            pbar.update(1)
            if results:
                correct_count = sum(r["correct"] for r in results)
                pbar.set_postfix_str(f"Acc: {correct_count / len(results):.1%}")
            if len(results) % 50 == 0:
                gc.collect()  # periodic memory release for long runs

    return results


async def run_debate_experiment(
    config_path: str,
    device: torch.device,
    split_override: str | None = None,
    max_samples: int | None = None,
) -> None:
    """Top-level orchestrator: load config, build engine, run debate on split."""
    resolved = Path(config_path) if Path(config_path).is_absolute() else PROJECT_ROOT / config_path
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split = split_override or cfg["data"]["split"]
    data_key = "dev_path" if split == "dev" else "test_path"
    all_samples = _load_samples(cfg["data"][data_key])

    done_ids = load_checkpoint(cfg["output"]["log_path"])
    pending = [s for s in all_samples if s["id"] not in done_ids]

    if max_samples is not None:
        pending = pending[:max_samples]

    logger.info(
        "Samples: total=%d  done=%d  pending=%d%s",
        len(all_samples), len(done_ids), len(pending),
        f"  (capped at {max_samples})" if max_samples is not None else "",
    )
    if not pending:
        logger.info("All samples already processed — recomputing metrics only.")
        compute_and_save_debate_metrics(
            log_path=cfg["output"]["log_path"],
            metrics_path=cfg["output"]["metrics_path"],
            cfg=cfg,
        )
        plot_debate_results(cfg["output"]["viz_dir"], cfg["output"]["log_path"])
        return

    mode = cfg["debate"]["mode"]
    client = build_client()
    engine, debate_logger = build_debate_engine(cfg, client, cfg["output"]["log_path"])
    gate = None

    if mode == "hybrid_debate":
        gate = build_routing_gate(cfg, device)

    logger.info(
        "=== %s | split=%s | N=%d | k=%d ===",
        mode, split,
        len(cfg["debate"]["panel"]["debaters"]),
        cfg["debate"]["rounds"],
    )
    try:
        await run_debate_on_split(
            engine=engine,
            samples=pending,
            mode=mode,
            gate=gate,
            batch_size=cfg["data"]["batch_size"],
        )
    finally:
        debate_logger.close()
        await client.close()
        compute_and_save_debate_metrics(
            log_path=cfg["output"]["log_path"],
            metrics_path=cfg["output"]["metrics_path"],
            cfg=cfg,
        )
        plot_debate_results(cfg["output"]["viz_dir"], cfg["output"]["log_path"])
        logger.info("=== Debate complete. Logs → %s ===", cfg["output"]["log_path"])


async def run_multi_config(
    config_paths: list[str],
    device: torch.device,
    split_override: str | None = None,
    max_samples: int | None = None,
    max_concurrent: int = 1,
) -> None:
    """Run multiple debate configs sequentially or concurrently.

    max_concurrent=1  → sequential (safe, no rate-limit risk)
    max_concurrent=2+ → N configs share the event loop (faster, more API load)
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _run_one(cfg_path: str) -> None:
        async with sem:
            config_name = Path(cfg_path).stem
            logger.info(">>> Starting config: %s", config_name)
            await run_debate_experiment(cfg_path, device, split_override, max_samples)
            logger.info("<<< Done config: %s", config_name)

    tasks = [_run_one(p) for p in config_paths]
    await asyncio.gather(*tasks)

    logger.info("=== All %d configs complete ===", len(config_paths))
