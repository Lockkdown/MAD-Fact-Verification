"""DebateLogger: per-sample JSONL logger with append mode and crash-safe flushing."""

import json
import logging
from pathlib import Path

from src.utils.common import PROJECT_ROOT

logger = logging.getLogger(__name__)


class DebateLogger:
    """Appends one JSON line per sample to a JSONL file. Flushes after every write."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")

    def log(self, result: dict) -> None:
        """Append one sample result as a JSON line and flush immediately."""
        if self._file.closed:
            return  # in-flight tasks may call this after close() — silently skip
        self._file.write(json.dumps(result, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "DebateLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @staticmethod
    def extract_errors(log_path: str) -> int:
        """Write error samples to <log_stem>_errors.jsonl. Returns count written."""
        src_path = Path(log_path)
        src = src_path if src_path.is_absolute() else PROJECT_ROOT / src_path
        if not src.exists():
            return 0
        errors = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "error" in entry:
                        errors.append(entry)
                except json.JSONDecodeError:
                    continue
        if not errors:
            return 0
        out_path = src.parent / f"{src.stem}_errors.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in errors:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Extracted %d error sample(s) → %s", len(errors), out_path)
        return len(errors)
