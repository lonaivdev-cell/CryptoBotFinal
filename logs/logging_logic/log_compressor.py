# logs/logging_logic/log_compressor.py
from __future__ import annotations

import atexit
import logging
import os
import threading
from datetime import datetime
from typing import Optional, Type
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler


# ────────────────────────────────────────────────────────────────────
# Hybrid score compressor (collapses many "Hybrid Scores {}")
# ────────────────────────────────────────────────────────────────────

class HybridScoreCompressor:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = False
        self.start_time: Optional[str] = None
        self.last_time: Optional[str] = None

    @staticmethod
    def _fmt_ts(record: logging.LogRecord) -> str:
        return datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    def process(self, record: logging.LogRecord) -> Optional[str]:
        """
        Returns:
          - None to suppress this record (another "{}" in a streak),
          - or a (possibly prefixed) message to emit.
        """
        msg = record.getMessage()
        now = self._fmt_ts(record)

        with self._lock:
            if msg.strip() == "Hybrid Scores {}":
                if not self.active:
                    self.active = True
                    self.start_time = now
                self.last_time = now
                return None  # suppress repeated empties

            # if breaking a streak, emit summary + the current message
            if self.active:
                summary = f"from {self.start_time} until {self.last_time}: Hybrid Scores {{}}"
                # reset
                self.active = False
                self.start_time = None
                self.last_time = None
                return f"{summary}\n{msg}"

        # normal message
        return msg

    def flush(self) -> Optional[str]:
        with self._lock:
            if self.active:
                summary = f"from {self.start_time} until {self.last_time}: Hybrid Scores {{}}"
                self.active = False
                self.start_time = None
                self.last_time = None
                return summary
        return None


class HybridScoreFilter(logging.Filter):
    def __init__(self, compressor: HybridScoreCompressor):
        super().__init__()
        self.compressor = compressor

    def filter(self, record: logging.LogRecord) -> bool:
        new_msg = self.compressor.process(record)
        if new_msg is None:
            return False
        record.msg = new_msg
        record.args = ()
        return True


# ────────────────────────────────────────────────────────────────────
# Handlers that flush the compressor on rollover/close
# ────────────────────────────────────────────────────────────────────

class CompressorTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, *args, compressor: HybridScoreCompressor, **kwargs):
        super().__init__(*args, **kwargs)
        self._compressor = compressor

    def doRollover(self):
        # flush current streak before rotating
        summary = self._compressor.flush()
        if summary:
            logging.getLogger(__name__).info(summary)
        super().doRollover()


class CompressorRotatingFileHandler(RotatingFileHandler):
    def __init__(self, *args, compressor: HybridScoreCompressor, **kwargs):
        super().__init__(*args, **kwargs)
        self._compressor = compressor

    def doRollover(self):
        summary = self._compressor.flush()
        if summary:
            logging.getLogger(__name__).info(summary)
        super().doRollover()


# ────────────────────────────────────────────────────────────────────
# Public setup function
# ────────────────────────────────────────────────────────────────────

def setup_logging(
    *,
    level: str = "INFO",
    log_dir: str = "logs",
    filename: str = "bot.log",
    rotate_when: str = "midnight",   # time-based rotation; ignored if max_bytes is set
    backup_count: int = 7,
    max_bytes: Optional[int] = None, # size-based rotation if set (e.g., 5_000_000)
    qt_queue_handler_cls: Optional[Type[logging.Handler]] = None,
    qt_log_queue: Optional["queue.Queue[str]"] = None,
) -> None:
    """
    Configure root logging:
      - Console + rotating file (time- or size-based)
      - Optional Qt queue handler (GUI)
      - HybridScoreCompressor across *all* handlers
      - One-time init (clears existing handlers to avoid duplicates)
    """
    # Resolve level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure directory
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    # Prevent duplicate handlers if setup_logging is called again
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            h.close()
        root.handlers.clear()

    root.setLevel(log_level)

    # Shared compressor & filter
    compressor = HybridScoreCompressor()
    filt = HybridScoreFilter(compressor)

    # use a compact formatter: "12:38:45 message"
    compact_fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(compact_fmt)  # << compact
    console.addFilter(filt)
    root.addHandler(console)

    # File handler (time- or size-rotating) with compressor-aware rollover
    if max_bytes and max_bytes > 0:
        file_handler = CompressorRotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
            compressor=compressor
        )
    else:
        file_handler = CompressorTimedRotatingFileHandler(
            log_path, when=rotate_when, backupCount=backup_count, encoding="utf-8", utc=False,
            compressor=compressor
        )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(compact_fmt)  # << compact
    file_handler.addFilter(filt)
    root.addHandler(file_handler)

    # Optional Qt handler for GUI
    if qt_queue_handler_cls is not None and qt_log_queue is not None:
        try:
            qt_handler = qt_queue_handler_cls(qt_log_queue)
            qt_handler.setLevel(log_level)
            qt_handler.setFormatter(compact_fmt)
            qt_handler.addFilter(filt)
            root.addHandler(qt_handler)
        except Exception as e:
            # keep logging alive even if GUI handler fails
            logging.getLogger(__name__).warning("QtQueueHandler init failed: %s", e)

    # Flush any open streak on exit
    def _flush_on_exit():
        summary = compressor.flush()
        if summary:
            logging.getLogger(__name__).info(summary)

    atexit.register(_flush_on_exit)

    # Keep library logs but avoid duplicate propagation loops
    # (we want compression to see them, so propagate=True)
    logging.getLogger("LlamaHybridStrategy").propagate = True
    logging.getLogger("ExchangeWrapper").propagate = True

    logging.getLogger(__name__).info("Logging initialized → %s", log_path)
