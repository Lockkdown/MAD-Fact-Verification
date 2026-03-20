"""DebateLogger: per-sample JSONL logger with append mode and crash-safe flushing."""

import json
from pathlib import Path


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
