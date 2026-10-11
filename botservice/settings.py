"""The headless service's own settings, read from the environment (``CRYPTOBOT_*``).

Trading settings (coins, thresholds, weights, the LLM) stay in ``config.py``, shared with the
desktop GUI. What is here only exists because the bot now runs as a service: where it listens,
where it keeps state, and the second live-trading switch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from botservice.safety import is_loopback_host

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PORT = 8097
DEFAULT_PAPER_USDT = 1000.0
# The agent's crypto_run_cycle waits up to 240 s by default; leave it room to answer.
DEFAULT_CYCLE_SENTIMENT_BUDGET_S = 150.0
MIN_INTERVAL_S = 5.0


class SettingsError(ValueError):
    """A setting the human must fix; the message names it and says what is wrong."""


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SettingsError(f"{name} must be a whole number, not \"{raw[:40]}\".") from None
    if not lo <= value <= hi:
        raise SettingsError(f"{name} must be between {lo} and {hi}, not {value}.")
    return value


def _float(env: Mapping[str, str], name: str, default: float, lo: float, hi: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(f"{name} must be a number, not \"{raw[:40]}\".") from None
    if not (lo <= value <= hi):
        raise SettingsError(f"{name} must be between {lo:g} and {hi:g}, not {raw}.")
    return value


@dataclass
class ServiceSettings:
    bind: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    # Optional shared secret. When set, every request needs "Authorization: Bearer <token>".
    # Off by default: the API is loopback-only, cannot enable live trading, and the agent's
    # plugin does not send one unless configured to.
    token: Optional[str] = None
    state_dir: str = field(default_factory=lambda: os.path.join(PROJECT_DIR, "state"))
    log_dir: str = field(default_factory=lambda: os.path.join(PROJECT_DIR, "logs"))
    # Where the human puts the Binance keys; only ever named in messages, never read here.
    env_file: str = field(default_factory=lambda: os.path.join(PROJECT_DIR, ".env"))
    paper_usdt: float = DEFAULT_PAPER_USDT
    autostart: bool = False
    cycle_sentiment_budget_s: float = DEFAULT_CYCLE_SENTIMENT_BUDGET_S
    # None → config.TRADE_INTERVAL. Tests set it directly to run cycles quickly.
    interval_s: Optional[float] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ServiceSettings":
        env = os.environ if env is None else env
        bind = (env.get("CRYPTOBOT_CONTROL_BIND") or "127.0.0.1").strip()
        if not is_loopback_host(bind):
            raise SettingsError(
                f"CRYPTOBOT_CONTROL_BIND must be a loopback address (127.0.0.1, ::1 or localhost), not \"{bind[:60]}\": "
                f"the control API has no business on a network interface. Reach it over SSH or Meshnet from the "
                f"server itself."
            )
        state_dir = (env.get("CRYPTOBOT_STATE_DIR") or "").strip() or os.path.join(PROJECT_DIR, "state")
        autostart_raw = (env.get("CRYPTOBOT_AUTOSTART") or "").strip().lower()
        if autostart_raw not in ("", "0", "1", "true", "false", "yes", "no", "on", "off"):
            raise SettingsError(f"CRYPTOBOT_AUTOSTART must be 1 or 0, not \"{autostart_raw[:20]}\".")
        token = (env.get("CRYPTOBOT_CONTROL_TOKEN") or "").strip() or None
        if token is not None and len(token) < 16:
            raise SettingsError("CRYPTOBOT_CONTROL_TOKEN must be at least 16 characters when set (or empty for none).")
        return cls(
            bind=bind,
            port=_int(env, "CRYPTOBOT_CONTROL_PORT", DEFAULT_PORT, 0, 65535),
            token=token,
            state_dir=state_dir,
            log_dir=(env.get("CRYPTOBOT_LOG_DIR") or "").strip() or os.path.join(PROJECT_DIR, "logs"),
            env_file=(env.get("CRYPTOBOT_ENV_FILE") or "").strip() or os.path.join(PROJECT_DIR, ".env"),
            paper_usdt=_float(env, "CRYPTOBOT_PAPER_USDT", DEFAULT_PAPER_USDT, 10.0, 1e9),
            autostart=autostart_raw in ("1", "true", "yes", "on"),
            cycle_sentiment_budget_s=_float(
                env, "CRYPTOBOT_CYCLE_SENTIMENT_BUDGET_S", DEFAULT_CYCLE_SENTIMENT_BUDGET_S, 0.0, 3600.0
            ),
        )
