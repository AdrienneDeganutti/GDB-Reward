from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[{_ts()}] WARNING: {msg}", file=sys.stderr, flush=True)


class JsonlLogger:
    """Append structured step records to ``<dir>/<name>.jsonl`` and echo to stdout."""

    def __init__(self, out_dir: str | Path, name: str = "log"):
        self.path = Path(out_dir) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")

    def log(self, record: dict[str, Any], echo: Optional[str] = None) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if echo is not None:
            info(echo)

    def close(self) -> None:
        self._fh.close()