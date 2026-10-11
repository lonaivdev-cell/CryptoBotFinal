"""The decision log behind GET /decisions: an in-memory ring plus an append-only JSONL file.

The file is what lets "what did the bot do overnight?" survive a restart. It is trimmed to the
last MAX_KEEP decisions when it grows past MAX_BYTES, so it cannot fill the state directory.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("DecisionLog")

MAX_KEEP = 1000
MAX_BYTES = 4 * 1024 * 1024
_FIELDS = ("at", "coin", "side", "amount", "price", "simulated", "reason")


def _valid(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and isinstance(row.get("at"), int)
        and isinstance(row.get("coin"), str)
        and row.get("side") in ("buy", "sell")
        and isinstance(row.get("amount"), (int, float))
        and isinstance(row.get("simulated"), bool)
    )


class DecisionLog:
    def __init__(self, path: Optional[str] = None, keep: int = MAX_KEEP) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._rows: "deque[Dict[str, Any]]" = deque(maxlen=keep)
        if path:
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:  # type: ignore[arg-type]
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if _valid(row):
                        self._rows.append({k: row.get(k) for k in _FIELDS})
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("Could not read the decision log %s: %s", self.path, exc)

    def extend(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows = [{k: r.get(k) for k in _FIELDS} for r in rows]
        if not rows:
            return
        with self._lock:
            self._rows.extend(rows)
            if not self.path:
                return
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                if os.path.getsize(self.path) > MAX_BYTES:
                    self._rewrite_locked()
            except OSError as exc:
                log.warning("Could not append to the decision log %s: %s", self.path, exc)

    def _rewrite_locked(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in self._rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(tmp, self.path)  # type: ignore[arg-type]

    def latest(self, limit: int) -> List[Dict[str, Any]]:
        """The newest `limit` decisions, newest first."""
        with self._lock:
            rows = list(self._rows)
        rows.reverse()
        return [dict(r) for r in rows[: max(0, limit)]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)
