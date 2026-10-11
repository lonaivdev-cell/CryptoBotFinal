#!/usr/bin/env python3
"""CryptoBot without the GUI: the trading loop as a service, controlled over a loopback JSON API.

    python headless.py            # serve on 127.0.0.1:${CRYPTOBOT_CONTROL_PORT:-8097}
    python headless.py --check    # print the effective mode and settings, then exit

This is what deploy/lo-cryptobot.service runs on loserver, and what the Lo's Server agent's
cryptobot plugin talks to. It wires the same exchange → LLM engine → sentiment scheduler →
strategy → TradingBot chain as main.py, minus PySide6, matplotlib and the dashboard (none of
them is imported, so requirements-headless.txt is enough).

The loop does not start by itself unless CRYPTOBOT_AUTOSTART=1: POST /start (or the agent's
crypto_start tool) starts it. It simulates unless a human set both
LLAMA_SENTIMENT_SIMULATE_ONLY=False and CRYPTOBOT_ALLOW_LIVE=1 and the keys are real; see
botservice/safety.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from botservice.safety import RedactingFilter, Redactor  # noqa: E402
from botservice.settings import ServiceSettings, SettingsError  # noqa: E402

log = logging.getLogger("headless")


def _setup_logging(settings: ServiceSettings, redactor) -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    try:
        from logs.logging_logic.log_compressor import setup_logging

        setup_logging(level=level, log_dir=settings.log_dir, filename="headless.log",
                      rotate_when="midnight", backup_count=7)
    except Exception as exc:  # an unwritable log dir must not keep the bot down: journald still has stderr
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger(__name__).warning("File logging unavailable (%s); logging to stderr only", exc)
    flt = RedactingFilter(redactor)
    for handler in logging.getLogger().handlers:
        handler.addFilter(flt)
    # ccxt and urllib3 log request URLs at DEBUG; keep them out of the journal unless asked.
    for noisy in ("ccxt", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, logging.getLogger().level))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Headless CryptoBot with a loopback control API.")
    parser.add_argument("--check", action="store_true", help="print the effective mode and settings as JSON and exit")
    args = parser.parse_args(argv)

    try:
        settings = ServiceSettings.from_env()
    except SettingsError as exc:
        print(f"lo-cryptobot: {exc}", file=sys.stderr)
        return 2

    # Logging first (with the redactor armed), so even the service's start-up lines are
    # filtered; --check prints JSON only and leaves the log files alone.
    import config

    if not args.check:
        _setup_logging(settings, Redactor([config.API_KEY, config.API_SECRET, config.API_PASSWORD,
                                           config.LLAMA_API_KEY, settings.token]))
    try:
        from botservice.service import BotService

        service = BotService(settings)
    except SettingsError as exc:
        print(f"lo-cryptobot: {exc}", file=sys.stderr)
        return 2

    if args.check:
        view = service.config_view()
        view["status"] = service.status()
        print(json.dumps(view, indent=2))
        return 0

    from botservice.api import make_server

    try:
        server = make_server(service, settings.bind, settings.port)
    except SettingsError as exc:
        log.error("%s", exc)
        return 2
    except OSError as exc:
        log.error("Could not listen on %s:%s: %s", settings.bind, settings.port, exc)
        return 1

    done = threading.Event()

    def _on_signal(signum, _frame) -> None:
        log.info("Signal %s received; shutting down", signum)
        done.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # serve_forever on its own thread: shutdown() must not be called from the thread serving.
    http_thread = threading.Thread(target=server.serve_forever, name="cryptobot-api", daemon=True)
    http_thread.start()
    host, port = server.server_address[:2]
    log.info("Control API on http://%s:%s (mode=%s)", host, port, service.gate.mode)

    if settings.autostart:
        try:
            service.start()
        except Exception as exc:
            log.error("Autostart failed: %s", exc)

    try:
        while not done.wait(1.0):
            pass
    finally:
        server.shutdown()
        server.server_close()
        service.shutdown()
        log.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
