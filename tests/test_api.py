"""The control API end to end: a real ThreadingHTTPServer on an ephemeral loopback port, the real
BotService, fake exchange and fake model. Shapes are checked against the contract the Lo's Server
agent's cryptobot plugin consumes."""

import http.client
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from tests.fakes import REAL_LOOKING_KEY, REAL_LOOKING_SECRET, FakeClient, FakeLLM, FakeMarket, make_cfg, make_service

from botservice.api import make_server
from botservice.settings import SettingsError

STATUS_FIELDS = {"ok", "running", "mode", "keys_configured", "llm", "strategy", "started_at", "last_cycle_at",
                 "cycles", "coins", "last_error"}
SIGNAL_FIELDS = {"coin", "sentiment", "trend", "hybrid", "action", "at"}
DECISION_FIELDS = {"at", "coin", "side", "amount", "price", "simulated", "reason"}
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ApiCase(unittest.TestCase):
    cfg_overrides: dict = {}
    env: dict = {}
    token = None

    def setUp(self):
        self.client = FakeClient(trends={"BTC": 1, "ETH": -1}, balance={"USDT": 250.0, "BTC": 0.01})
        keyed = self.cfg_overrides.get("API_KEY", "")
        self.market = FakeMarket(self.client, api_key=keyed, api_secret=self.cfg_overrides.get("API_SECRET", ""))
        self.llm = FakeLLM({"BTC": 0.9, "ETH": -0.9, "ADA": 0.0})
        self.svc = make_service(cfg=make_cfg(**self.cfg_overrides), env=dict(self.env), market=self.market,
                                engine=self.llm, interval_s=0.05, token=self.token)
        self.server = make_server(self.svc, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.svc.shutdown()
        self.server.shutdown()
        self.server.server_close()

    def call(self, method, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        hdrs = {"Accept": "application/json"}
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=hdrs)
        try:
            with NO_PROXY.open(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read()), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), dict(e.headers)

    def assertError(self, result, status, *fragments):
        code, body, _ = result
        self.assertEqual(code, status, body)
        self.assertEqual(set(body), {"error"})
        self.assertIsInstance(body["error"], str)
        self.assertTrue(body["error"].endswith("."), body["error"])
        for f in fragments:
            self.assertIn(f, body["error"])


class SimulationApiTest(ApiCase):
    def test_status_shape(self):
        code, s, headers = self.call("GET", "/status")
        self.assertEqual(code, 200)
        self.assertEqual(set(s), STATUS_FIELDS)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(s["mode"], "simulation")
        self.assertIs(s["ok"], True)
        self.assertIs(s["running"], False)
        self.assertIs(s["keys_configured"], False)
        self.assertEqual(s["llm"], {"url": "http://127.0.0.1:8080/v1", "reachable": True})
        self.assertIn(s["strategy"], ("hybrid", "portfolio"))
        self.assertIsNone(s["started_at"])
        self.assertIsNone(s["last_cycle_at"])
        self.assertEqual(s["cycles"], 0)
        self.assertEqual(s["coins"], ["BTC", "ETH", "ADA"])
        self.assertIsNone(s["last_error"])

    def test_cycle_signals_decisions(self):
        code, body, _ = self.call("POST", "/cycle", {})
        self.assertEqual(code, 200)
        self.assertEqual(len(body["decisions"]), 1)
        d = body["decisions"][0]
        self.assertEqual(set(d), DECISION_FIELDS)
        self.assertEqual((d["coin"], d["side"], d["simulated"]), ("BTC", "buy", True))
        self.assertIsInstance(d["at"], int)
        self.assertIsInstance(d["amount"], float)
        self.assertIsInstance(d["price"], float)
        self.assertEqual(self.client.orders, [])

        code, body, _ = self.call("GET", "/signals")
        self.assertEqual(code, 200)
        self.assertEqual([s["coin"] for s in body["signals"]], ["BTC", "ETH", "ADA"])
        for s in body["signals"]:
            self.assertEqual(set(s), SIGNAL_FIELDS)
            self.assertIn(s["action"], ("buy", "sell", "hold"))
        self.assertEqual(body["signals"][1]["action"], "sell")

        self.call("POST", "/cycle")  # an empty body is fine too
        code, body, _ = self.call("GET", "/decisions?limit=1")
        self.assertEqual(code, 200)
        self.assertEqual(len(body["decisions"]), 1)
        code, body, _ = self.call("GET", "/decisions")
        ats = [d["at"] for d in body["decisions"]]
        self.assertEqual(ats, sorted(ats, reverse=True))  # newest first

        code, s, _ = self.call("GET", "/status")
        self.assertEqual(s["cycles"], 2)
        self.assertIsInstance(s["last_cycle_at"], int)

    def test_decisions_limit_validation(self):
        for q in ("limit=0", "limit=501", "limit=ten", "limit=-3"):
            with self.subTest(q=q):
                self.assertError(self.call("GET", f"/decisions?{q}"), 400, "limit")
        self.assertEqual(self.call("GET", "/decisions?limit=500")[0], 200)

    def test_balances_with_placeholder_keys(self):
        self.assertError(self.call("GET", "/balances"), 409, "Binance keys are still placeholders",
                         "/etc/lo-cryptobot/cryptobot.env")

    def test_paper_wallet(self):
        self.call("POST", "/cycle", {})
        code, body, _ = self.call("GET", "/paper")
        self.assertEqual(code, 200)
        self.assertAlmostEqual(body["paper"]["balances"]["BTC"], 0.03)
        self.assertIsNotNone(body["paper"]["value"])

    def test_config_is_redacted_and_complete(self):
        code, cfg, _ = self.call("GET", "/config")
        self.assertEqual(code, 200)
        for field in ("coins", "thresholds", "weights", "interval", "simulate_only", "strategy"):
            self.assertIn(field, cfg)
        self.assertIs(cfg["simulate_only"], True)
        flat = json.dumps(cfg).lower()
        for word in ("api_key", "apikey", "secret", "password"):
            self.assertNotIn(word, flat)

    def test_start_stop(self):
        code, st, _ = self.call("POST", "/start", {"strategy": "portfolio"})
        self.assertEqual(code, 200)
        self.assertEqual(set(st), STATUS_FIELDS)
        self.assertEqual((st["running"], st["mode"], st["strategy"]), (True, "simulation", "portfolio"))
        self.assertIsInstance(st["started_at"], int)
        deadline = time.time() + 5
        while time.time() < deadline and self.svc.cycles < 2:
            time.sleep(0.02)
        self.assertGreaterEqual(self.call("GET", "/status")[1]["cycles"], 2)
        self.assertError(self.call("POST", "/start", {"strategy": "hybrid"}), 409, "already running")
        code, st, _ = self.call("POST", "/stop", {})
        self.assertEqual(code, 200)
        self.assertFalse(st["running"])
        self.assertIsNone(st["started_at"])
        self.assertFalse(self.svc._loop_thread.is_alive())
        # And again, without a strategy: the last one is kept.
        code, st, _ = self.call("POST", "/start", {})
        self.assertEqual((code, st["strategy"]), (200, "portfolio"))
        self.call("POST", "/stop")

    def test_start_validation(self):
        self.assertError(self.call("POST", "/start", {"strategy": "yolo"}), 400, "strategy")
        self.assertError(self.call("POST", "/start", {"strategy": "hybrid", "coins": ["PEPE"]}), 400, "coins")
        self.assertError(self.call("POST", "/start", raw=b"{not json"), 400, "JSON")
        self.assertError(self.call("POST", "/start", raw=b"[1]"), 400, "object")
        self.assertError(self.call("POST", "/stop", {"now": True}), 400, "no fields")
        self.assertFalse(self.svc.running)

    def test_mode_cannot_be_switched_over_the_api(self):
        attempts = [
            ("/start", {"strategy": "hybrid", "mode": "live"}),
            ("/start", {"simulate_only": False}),
            ("/start", {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False"}),
            ("/start", {"CRYPTOBOT_ALLOW_LIVE": "1"}),
            ("/cycle", {"live": True}),
            ("/stop", {"Mode": "live"}),
        ]
        for path, body in attempts:
            with self.subTest(path=path, body=body):
                self.assertError(self.call("POST", path, body), 403, "cannot be changed over the API")
        for path, body in [("/start", {"api_key": "x"}), ("/start", {"BINANCE_API_SECRET": "y"})]:
            with self.subTest(path=path, body=body):
                self.assertError(self.call("POST", path, body), 403, "Keys cannot be set")
        # No such endpoints exist, under any method.
        for method, path in [("POST", "/mode"), ("POST", "/live"), ("POST", "/config"), ("PUT", "/config"),
                             ("PATCH", "/status"), ("POST", "/keys"), ("DELETE", "/start")]:
            with self.subTest(method=method, path=path):
                code, body, _ = self.call(method, path, {"mode": "live"} if method != "DELETE" else None)
                self.assertIn(code, (403, 404, 405))
        self.assertFalse(self.svc.running)
        self.assertEqual(self.call("GET", "/status")[1]["mode"], "simulation")
        self.assertEqual(self.client.orders, [])

    def test_unknown_paths_and_methods(self):
        self.assertError(self.call("GET", "/nope"), 404, "There is no endpoint", "GET /status")
        code, body, headers = self.call("GET", "/start")
        self.assertEqual(code, 405)
        self.assertEqual(headers.get("Allow"), "POST")
        self.assertEqual(self.call("GET", "/status/")[0], 200)  # a trailing slash is the same path

    def test_browser_and_rebinding_requests_are_refused(self):
        self.assertError(self.call("GET", "/status", headers={"Origin": "http://evil.example"}), 403, "Browser")
        self.assertError(self.call("POST", "/start", {}, headers={"Origin": "null"}), 403, "Browser")
        self.assertError(self.call("GET", "/status", headers={"Host": "evil.example:8097"}), 403, "loopback")
        self.assertEqual(self.call("GET", "/status", headers={"Host": "localhost:8097"})[0], 200)
        self.assertFalse(self.svc.running)

    def test_oversized_and_chunked_bodies(self):
        self.assertError(self.call("POST", "/start", raw=b" " * (17 * 1024)), 413)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/start")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(b"0\r\n\r\n")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 411)
        self.assertIn("error", json.loads(resp.read()))
        conn.close()

    def test_malformed_request_line_gets_json(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.connect()
        conn.sock.sendall(b"BREW /pot HTTP/9.9\r\n\r\n")
        chunks = []
        while True:  # the server closes after answering; headers and body may arrive apart
            chunk = conn.sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks).decode()
        conn.close()
        # Python < 3.12 answers an unparseable version as HTTP/0.9, i.e. without headers; the
        # body is the same JSON error either way.
        self.assertIn('{"error": ', data)


class LiveApiTest(ApiCase):
    cfg_overrides = {"API_KEY": REAL_LOOKING_KEY, "API_SECRET": REAL_LOOKING_SECRET}
    env = {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False", "CRYPTOBOT_ALLOW_LIVE": "1"}

    def test_live_mode_is_reported_and_orders_are_real(self):
        code, s, _ = self.call("GET", "/status")
        self.assertEqual((s["mode"], s["keys_configured"]), ("live", True))
        code, body, _ = self.call("POST", "/cycle", {})
        self.assertEqual(code, 200)
        self.assertEqual([(d["side"], d["simulated"]) for d in body["decisions"]], [("buy", False)])
        self.assertEqual(len(self.client.orders), 1)
        code, body, _ = self.call("GET", "/balances")
        self.assertEqual(code, 200)
        self.assertEqual(body, {"balances": {"USDT": {"free": 250.0, "used": 0.0, "total": 250.0},
                                             "BTC": {"free": 0.01, "used": 0.0, "total": 0.01}}})
        self.assertIs(self.call("GET", "/config")[1]["simulate_only"], False)
        self.assertError(self.call("GET", "/paper"), 409, "live")

    def test_secrets_never_appear_in_any_answer(self):
        self.client.balance_error = RuntimeError(f"denied for key {REAL_LOOKING_KEY} sig {REAL_LOOKING_SECRET}")
        seen = []
        for method, path in [("GET", "/status"), ("GET", "/config"), ("GET", "/balances"), ("GET", "/signals"),
                             ("GET", "/decisions"), ("POST", "/cycle")]:
            seen.append(json.dumps(self.call(method, path, {} if method == "POST" else None)[1]))
        text = "\n".join(seen)
        self.assertNotIn(REAL_LOOKING_KEY, text)
        self.assertNotIn(REAL_LOOKING_SECRET, text)
        self.assertIn("[redacted]", text)


class TokenApiTest(ApiCase):
    token = "a-long-enough-control-token"

    def test_token_is_required_when_set(self):
        self.assertError(self.call("GET", "/status"), 401, "Authorization")
        self.assertError(self.call("GET", "/status", headers={"Authorization": "Bearer wrong-token-value"}), 401)
        code, _, _ = self.call("GET", "/status", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(code, 200)


class BindTest(unittest.TestCase):
    def test_only_loopback_binds(self):
        svc = make_service()
        for host in ("0.0.0.0", "::", "192.168.1.5", "loserver"):
            with self.subTest(host=host):
                with self.assertRaises(SettingsError):
                    make_server(svc, host, 0)
        for host in ("127.0.0.1", "localhost"):
            server = make_server(svc, host, 0)
            self.assertEqual(server.server_address[0], "127.0.0.1")
            server.server_close()
        try:
            server = make_server(svc, "::1", 0)
        except OSError:
            self.skipTest("no IPv6 loopback here")
        self.assertEqual(server.server_address[0], "::1")
        server.server_close()


if __name__ == "__main__":
    unittest.main()
