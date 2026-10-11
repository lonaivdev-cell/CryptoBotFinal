"""Headless CryptoBot: the trading loop as a service, with a loopback JSON control API.

``headless.py`` is the entry point. Nothing in this package imports PySide6, matplotlib or the
dashboard, so the service runs on a server with only ``requirements-headless.txt`` installed.

- ``safety``   placeholder-key detection, the live-trading gate, secret redaction
- ``settings`` the service's own environment (``CRYPTOBOT_*``)
- ``paper``    the paper wallet simulated trades fill against
- ``journal``  the persistent decision log behind ``GET /decisions``
- ``service``  wiring and lifecycle: exchange → LLM → scheduler → strategy → bot, on a thread
- ``api``      the HTTP control API (stdlib ``http.server``)
"""
