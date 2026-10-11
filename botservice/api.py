"""The control API: stdlib ``http.server``, JSON in and out, loopback only.

The contract (shared with the Lo's Server agent's cryptobot plugin):

    GET  /status      running, mode, keys_configured, llm, strategy, timestamps, cycles, coins
    GET  /signals     latest per-coin signal
    GET  /decisions   ?limit=1..500 (default 20), newest first
    GET  /balances    real Binance balances, or 409 while the keys are placeholders
    GET  /config      redacted settings
    GET  /paper       the paper wallet simulated trades fill against (409 in live mode)
    POST /start       {"strategy": "hybrid" | "portfolio"} (optional)
    POST /stop        {}
    POST /cycle       {} → {"decisions": [...]}

Errors are ``{"error": "<sentence>"}``. There is deliberately no endpoint, field or header that
changes simulation/live mode or touches keys; a request that tries is refused with 403 rather
than silently ignored, so whoever sent it learns that it did not work.

Who may call: anything on the server itself. The listener refuses non-loopback addresses, and
three cheap checks keep a web page in a browser on the server from driving it (DNS rebinding,
cross-site form posts): the Host header must name a loopback host, requests carrying an Origin
header (which browsers add and command-line clients do not) are refused, and when
CRYPTOBOT_CONTROL_TOKEN is set every request needs it as a bearer token.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple

from botservice.safety import is_loopback_host, sentence
from botservice.service import BadRequest, BotService, Forbidden, ServiceError
from botservice.settings import SettingsError

log = logging.getLogger("ControlAPI")

MAX_BODY_BYTES = 16 * 1024
MAX_DECISIONS_LIMIT = 500

# Request fields that would change the trading mode or credentials. Refused by name (403) so
# the caller is told plainly that this cannot be done here, whatever endpoint it was sent to.
_MODE_FIELDS = {
    "mode", "live", "live_trading", "simulate", "simulate_only", "simulation", "dry_run", "paper",
    "allow_live", "llama_sentiment_simulate_only", "cryptobot_allow_live",
}
_SECRET_HINTS = ("key", "secret", "password", "passphrase", "token")

MODE_REFUSAL = (
    "The trading mode cannot be changed over the API: simulation or live is decided only by a human, "
    "in the bot's environment (LLAMA_SENTIMENT_SIMULATE_ONLY=False and CRYPTOBOT_ALLOW_LIVE=1)."
)
KEY_REFUSAL = "Keys cannot be set over the API; they live only in the bot's .env on the server."


def _check_fields(body: Dict[str, Any], allowed: Tuple[str, ...], endpoint: str) -> None:
    for name in body:
        low = str(name).strip().lower()
        if low in _MODE_FIELDS:
            raise Forbidden(MODE_REFUSAL)
        if any(h in low for h in _SECRET_HINTS):
            raise Forbidden(KEY_REFUSAL)
    unknown = [str(k) for k in body if k not in allowed]
    if unknown:
        takes = " and ".join(f'"{a}"' for a in allowed) if allowed else "no fields"
        raise BadRequest(f"{endpoint} does not take {', '.join(repr(u)[:40] for u in unknown[:5])}; it takes {takes}.")


# ---------------------------------------------------------------------------- routes

Handler = Callable[[BotService, Dict[str, list], Optional[Dict[str, Any]]], Tuple[int, Dict[str, Any]]]


def _get_status(svc: BotService, query, body) -> Tuple[int, Dict[str, Any]]:
    return 200, svc.status()


def _get_signals(svc: BotService, query, body):
    return 200, svc.signals()


def _get_decisions(svc: BotService, query, body):
    raw = (query.get("limit") or ["20"])[-1].strip()
    try:
        limit = int(raw)
    except ValueError:
        raise BadRequest(f"limit must be a whole number from 1 to {MAX_DECISIONS_LIMIT}, not {raw[:20]!r}.") from None
    if not 1 <= limit <= MAX_DECISIONS_LIMIT:
        raise BadRequest(f"limit must be from 1 to {MAX_DECISIONS_LIMIT}, not {limit}.")
    return 200, svc.decisions(limit)


def _get_balances(svc: BotService, query, body):
    return 200, svc.balances()


def _get_config(svc: BotService, query, body):
    return 200, svc.config_view()


def _get_paper(svc: BotService, query, body):
    return 200, svc.paper()


def _post_start(svc: BotService, query, body):
    body = body or {}
    _check_fields(body, ("strategy",), "POST /start")
    return 200, svc.start(body.get("strategy"))


def _post_stop(svc: BotService, query, body):
    _check_fields(body or {}, (), "POST /stop")
    return 200, svc.stop()


def _post_cycle(svc: BotService, query, body):
    _check_fields(body or {}, (), "POST /cycle")
    return 200, {"decisions": svc.run_cycle()}


ROUTES: Dict[str, Dict[str, Handler]] = {
    "/status": {"GET": _get_status},
    "/signals": {"GET": _get_signals},
    "/decisions": {"GET": _get_decisions},
    "/balances": {"GET": _get_balances},
    "/config": {"GET": _get_config},
    "/paper": {"GET": _get_paper},
    "/start": {"POST": _post_start},
    "/stop": {"POST": _post_stop},
    "/cycle": {"POST": _post_cycle},
}
ENDPOINTS = ", ".join(f"{m} {p}" for p, ms in ROUTES.items() for m in ms)


def _host_of(header: str) -> str:
    h = header.strip()
    if h.startswith("["):
        return h[1:h.find("]")] if "]" in h else h
    if h.count(":") == 1:
        return h.split(":", 1)[0]
    return h


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "lo-cryptobot/1"
    sys_version = ""
    # Every answer carries Content-Length and the connection closes after it: simplest for the
    # stdlib clients on the other end, and no idle keep-alive threads.
    protocol_version = "HTTP/1.0"

    def version_string(self) -> str:
        return self.server_version

    @property
    def service(self) -> BotService:
        return self.server.service  # type: ignore[attr-defined]

    # ---- plumbing
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - BaseHTTPRequestHandler API
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: Dict[str, Any], extra: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code: int, message: Optional[str] = None, explain: Optional[str] = None) -> None:
        # Malformed request lines and the like: same JSON shape as every other error.
        try:
            self._send(code, {"error": sentence(message or explain or f"HTTP {code}.")})
        except Exception:
            pass

    def _check_caller(self) -> None:
        host = self.headers.get("Host")
        if host is not None and not is_loopback_host(_host_of(host)):
            raise ServiceError("This API only answers requests addressed to the loopback interface.", 403)
        if self.headers.get("Origin") is not None:
            raise ServiceError("Browser requests are not accepted; this API is for tools running on the server.", 403)
        token = self.service.settings.token
        if token:
            auth = self.headers.get("Authorization") or ""
            given = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
            if not hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
                raise ServiceError("This API needs the header Authorization: Bearer <CRYPTOBOT_CONTROL_TOKEN>.", 401)

    def _read_body(self) -> Dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise ServiceError("Send the request body with a Content-Length, not chunked.", 411)
        raw_len = self.headers.get("Content-Length") or "0"
        try:
            length = int(raw_len)
        except ValueError:
            raise BadRequest("Content-Length is not a number.") from None
        if length < 0:
            raise BadRequest("Content-Length is negative.")
        if length > MAX_BODY_BYTES:
            raise ServiceError(f"The request body is larger than {MAX_BODY_BYTES // 1024} KiB.", 413)
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise BadRequest("The request body is not valid JSON.") from None
        if not isinstance(parsed, dict):
            raise BadRequest("The request body must be a JSON object.")
        return parsed

    def _dispatch(self) -> None:
        method = self.command
        extra: Optional[Dict[str, str]] = None
        try:
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            self._check_caller()
            methods = ROUTES.get(path)
            if methods is None:
                raise ServiceError(f"There is no endpoint at {path[:80]}; this API has {ENDPOINTS}.", 404)
            handler = methods.get(method)
            if handler is None:
                extra = {"Allow": ", ".join(methods)}
                raise ServiceError(f"{path} answers {' and '.join(methods)}, not {method}.", 405)
            body = self._read_body() if method == "POST" else None
            status, payload = handler(self.service, urllib.parse.parse_qs(parts.query), body)
        except ServiceError as exc:
            status, payload = exc.status, {"error": sentence(str(exc), self.service.redactor)}
        except Exception as exc:  # a bug, not a state: log it in full, answer in one line
            log.error("Control API error on %s %s: %s", method, self.path[:120], exc, exc_info=True)
            status, payload = 500, {"error": "The control API hit an internal error; the bot's log has the details."}
        self._send(status, payload, extra)

    do_GET = _dispatch
    do_POST = _dispatch
    do_PUT = _dispatch
    do_DELETE = _dispatch
    do_PATCH = _dispatch
    do_HEAD = _dispatch
    do_OPTIONS = _dispatch


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], service: BotService, family: int) -> None:
        self.address_family = family
        self.service = service
        super().__init__(address, ControlHandler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind would reverse-resolve the address (socket.getfqdn) just to fill
        # in server_name; a slow resolver must not delay the bot's start.
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()[:2]
        self.server_name, self.server_port = str(self.server_address[0]), int(self.server_address[1])


def make_server(service: BotService, host: str = "127.0.0.1", port: int = 8097) -> ControlServer:
    """Bind the control API. Refuses anything but a loopback address, before and after binding."""
    if not is_loopback_host(host):
        raise SettingsError(f"The control API binds only to loopback addresses, not \"{host[:60]}\".")
    # "localhost" is pinned to 127.0.0.1 rather than resolved: /etc/hosts is not ours to trust.
    addr = "127.0.0.1" if host.strip().lower() == "localhost" else host.strip().strip("[]")
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET
    server = ControlServer((addr, port), service, family)
    bound = server.server_address[0]
    if not ipaddress.ip_address(bound).is_loopback:  # pragma: no cover - belt and braces
        server.server_close()
        raise SettingsError(f"The control API ended up bound to {bound}, which is not loopback; refusing to serve.")
    return server
