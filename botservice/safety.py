"""The service's safety rules, as small pure functions so they can be tested exhaustively.

Three rules live here:

1. **Placeholder keys are not keys.** The README and ``.env.example`` ship "PLACEHOLDER…" and
   "your_key_here" values; a bot that sent those to Binance would fail every private call with
   an authentication error the user would have to decode. Detected placeholders mean "public
   market data only".
2. **Live trading is a decision a human makes in the environment.** It takes both
   ``LLAMA_SENTIMENT_SIMULATE_ONLY=False`` and ``CRYPTOBOT_ALLOW_LIVE=1``, plus real keys. Two
   separate switches, because the first one alone is also read by the desktop GUI and has been
   flipped by hand before; the second exists only for this service.
3. **Secrets never leave the process.** Every error sentence the API returns and every log
   line passes through a redactor that knows the actual key values.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

# Lower-cased values that mean "nobody filled this in".
_PLACEHOLDER_WORDS = {
    "none", "null", "nil", "changeme", "change_me", "change-me", "todo", "tbd", "placeholder",
    "your_key_here", "your_secret_here", "your_api_key", "your_api_secret", "your-key-here",
    "secret", "apikey", "api_key", "key", "test", "dummy", "example",
}
# Binance keys and secrets are 64 characters; anything this short is not one.
_MIN_KEY_LENGTH = 16


def is_placeholder(value: Optional[str]) -> bool:
    """True when a key or secret value is empty or obviously not a real credential."""
    v = (value or "").strip().strip("\"'").strip()
    if not v:
        return True
    low = v.lower()
    if low in _PLACEHOLDER_WORDS:
        return True
    if low.startswith(("placeholder", "your_", "your-", "<", "${", "replace", "insert", "put_")):
        return True
    if set(low) <= set("x*.-_ #"):
        return True
    if len(v) < _MIN_KEY_LENGTH or any(c.isspace() for c in v):
        return True
    return False


def keys_configured(api_key: Optional[str], api_secret: Optional[str]) -> bool:
    return not is_placeholder(api_key) and not is_placeholder(api_secret)


_FALSE_WORDS = ("false", "0", "no", "off")


@dataclass(frozen=True)
class LiveGate:
    mode: str                     # "simulation" | "live" — what the service actually does
    live_requested: bool          # LLAMA_SENTIMENT_SIMULATE_ONLY is an explicit false word
    live_allowed: bool            # CRYPTOBOT_ALLOW_LIVE is exactly "1"
    reason: Optional[str]         # why it is simulating despite a live switch, if one is set

    @property
    def simulate_only(self) -> bool:
        return self.mode != "live"


def live_gate(env: Mapping[str, str], keys_ok: bool) -> LiveGate:
    """Decide simulation vs live from the process environment, once, at service start.

    Only the environment counts — never a request, never a file the service can write. Anything
    that is not an explicit false word keeps LLAMA_SENTIMENT_SIMULATE_ONLY on, and only the exact
    string "1" counts as CRYPTOBOT_ALLOW_LIVE ("true", "yes" do not: this switch is meant to be
    typed deliberately, not inferred).
    """
    sim_raw = (env.get("LLAMA_SENTIMENT_SIMULATE_ONLY") or "").strip().lower()
    requested = sim_raw in _FALSE_WORDS
    allowed = (env.get("CRYPTOBOT_ALLOW_LIVE") or "").strip() == "1"
    if requested and allowed and keys_ok:
        return LiveGate("live", True, True, None)
    reason: Optional[str] = None
    if requested and allowed:
        reason = ("Live trading is enabled in the environment, but the Binance keys are still placeholders, "
                  "so the bot stays in simulation.")
    elif requested:
        reason = ("LLAMA_SENTIMENT_SIMULATE_ONLY=False is set, but CRYPTOBOT_ALLOW_LIVE is not 1, "
                  "so the bot stays in simulation.")
    elif allowed:
        reason = ("CRYPTOBOT_ALLOW_LIVE=1 is set, but LLAMA_SENTIMENT_SIMULATE_ONLY is not False, "
                  "so the bot stays in simulation.")
    return LiveGate("simulation", requested, allowed, reason)


def is_loopback_host(host: str) -> bool:
    """True for "localhost", 127.0.0.0/8 and ::1 (brackets allowed)."""
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


REDACTED = "[redacted]"


class Redactor:
    """Replaces known secret values in text. Values shorter than 6 characters are ignored:
    a placeholder like "x" must not turn every "x" in an error message into [redacted]."""

    def __init__(self, secrets: Iterable[Optional[str]] = ()) -> None:
        self._secrets: List[str] = []
        for s in secrets:
            self.add(s)

    def add(self, secret: Optional[str]) -> None:
        s = (secret or "").strip()
        if len(s) >= 6 and s not in self._secrets:
            self._secrets.append(s)
            # Longest first, so a secret that contains another is replaced whole.
            self._secrets.sort(key=len, reverse=True)

    def __call__(self, text: object) -> str:
        out = str(text)
        for s in self._secrets:
            if s in out:
                out = out.replace(s, REDACTED)
        return out


class RedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from the fully formatted message (args included)."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string is the logging module's problem, not ours
            return True
        clean = self.redactor(message)
        if record.exc_info and not record.exc_text:
            # Format the traceback now so the redacted text is what handlers print.
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redactor(record.exc_text)
        if clean != message or record.args:
            record.msg, record.args = clean, ()
        return True


def sentence(text: object, redactor: Optional[Redactor] = None, limit: int = 300) -> str:
    """One line, redacted, bounded: fit for an {"error": ...} body or a status field."""
    flat = " ".join(str(text).split())
    if redactor is not None:
        flat = redactor(flat)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
