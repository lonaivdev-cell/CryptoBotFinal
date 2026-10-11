"""headless.py as a process, config.py's environment parsing, and what the service imports."""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from tests.fakes import ROOT

HEADLESS = os.path.join(ROOT, "headless.py")
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def clean_env(**extra):
    """The test runner's environment without anything that would change the bot's behaviour."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("BINANCE_", "LLAMA_", "CRYPTOBOT_", "SENTIMENT_", "USE_PORTFOLIO", "TRADE_INTERVAL"))}
    state = tempfile.mkdtemp(prefix="headless-test-")
    env.update({
        "CRYPTOBOT_STATE_DIR": state,
        "CRYPTOBOT_LOG_DIR": os.path.join(state, "logs"),
        # Nothing listens on the discard port: the model is simply "not reachable".
        "LLAMA_API_BASE_URL": "http://127.0.0.1:9/api",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.update(extra)
    return env


class ConfigEnvTest(unittest.TestCase):
    def read_config(self, **env):
        code = ("import json, config as c; print(json.dumps({k: getattr(c, k) for k in ("
                "'LLAMA_SENTIMENT_SIMULATE_ONLY','LLAMA_API_STYLE','LLAMA_API_BASE_URL','LLAMA_DISABLE_THINKING',"
                "'SENTIMENT_COINS','SENTIMENT_TTL_SECONDS','SENTIMENT_MAX_QPS','USE_PORTFOLIO_STRATEGY',"
                "'SENTIMENT_BUY_THRESHOLD','STOP_LOSS_PCT','POSITION_MAX_DAYS')}))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean_env(**env), capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_defaults_are_the_old_hard_coded_values(self):
        env = clean_env()
        env.pop("LLAMA_API_BASE_URL")
        out = subprocess.run([sys.executable, "-c", "import json, config as c; print(json.dumps([c.LLAMA_SENTIMENT_SIMULATE_ONLY, "
                              "c.LLAMA_API_STYLE, c.LLAMA_API_BASE_URL, c.SENTIMENT_COINS, c.SENTIMENT_TTL_SECONDS, "
                              "c.SENTIMENT_MAX_QPS, c.USE_PORTFOLIO_STRATEGY, c.SENTIMENT_BUY_THRESHOLD, "
                              "c.SENTIMENT_SELL_THRESHOLD, c.STOP_LOSS_PCT, c.POSITION_MAX_DAYS, c.OPP_BASE_NOTIONAL_USD]))"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(out.stdout), [
            True, "ollama", "http://localhost:11434/api", ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX"],
            240, 0.25, True, 0.45, -0.40, 0.15, 10, 15.0,
        ])

    def test_openai_style_defaults_to_the_local_llama_server(self):
        env = clean_env(LLAMA_API_STYLE="OpenAI")
        env.pop("LLAMA_API_BASE_URL")
        out = subprocess.run([sys.executable, "-c", "import config as c; print(c.LLAMA_API_STYLE, c.LLAMA_API_BASE_URL)"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.split(), ["openai", "http://127.0.0.1:8080/v1"])

    def test_environment_overrides(self):
        c = self.read_config(LLAMA_SENTIMENT_SIMULATE_ONLY="False", SENTIMENT_COINS="btc, eth sol;x ETH",
                             SENTIMENT_TTL_SECONDS="900", SENTIMENT_MAX_QPS="0.05", USE_PORTFOLIO_STRATEGY="0",
                             SENTIMENT_BUY_THRESHOLD="0.6", LLAMA_DISABLE_THINKING="no")
        self.assertIs(c["LLAMA_SENTIMENT_SIMULATE_ONLY"], False)
        self.assertEqual(c["SENTIMENT_COINS"], ["BTC", "ETH", "SOL"])
        self.assertEqual((c["SENTIMENT_TTL_SECONDS"], c["SENTIMENT_MAX_QPS"]), (900, 0.05))
        self.assertIs(c["USE_PORTFOLIO_STRATEGY"], False)
        self.assertEqual(c["SENTIMENT_BUY_THRESHOLD"], 0.6)
        self.assertIs(c["LLAMA_DISABLE_THINKING"], False)

    def test_garbage_falls_back_to_safe_defaults(self):
        c = self.read_config(LLAMA_SENTIMENT_SIMULATE_ONLY="Flase", SENTIMENT_TTL_SECONDS="soon",
                             SENTIMENT_BUY_THRESHOLD="nan", STOP_LOSS_PCT="inf", SENTIMENT_COINS=";;;")
        self.assertIs(c["LLAMA_SENTIMENT_SIMULATE_ONLY"], True)
        self.assertEqual(c["SENTIMENT_TTL_SECONDS"], 240)
        self.assertEqual(c["SENTIMENT_BUY_THRESHOLD"], 0.45)
        self.assertEqual(c["STOP_LOSS_PCT"], 0.15)
        self.assertEqual(len(c["SENTIMENT_COINS"]), 8)


class ImportHygieneTest(unittest.TestCase):
    def test_service_needs_neither_gui_nor_ccxt(self):
        # ccxt is made unimportable; the whole service, a cycle on fakes and the API module must
        # still work, and nothing GUI-shaped may be imported along the way.
        code = r"""
import sys, json
sys.modules['ccxt'] = None
sys.modules['PySide6'] = None
sys.modules['matplotlib'] = None
import headless, botservice.api
from tests.fakes import make_service, FakeMarket, FakeClient
svc = make_service(market=FakeMarket(FakeClient(trends={'BTC': 1})))
decisions = svc.run_cycle()
real = make_service(market=None)
try:
    real._ensure_connected()
    err = None
except Exception as exc:
    err = str(exc)
loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'matplotlib', 'dashboard_enhanced', 'main')
                and sys.modules[m] is not None)
print(json.dumps({'decisions': len(decisions), 'loaded': loaded, 'err': err}))
"""
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean_env(), capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(result["loaded"], [])
        self.assertIn("ccxt is not installed", result["err"])


class HeadlessProcessTest(unittest.TestCase):
    def start(self, **env):
        proc = subprocess.Popen([sys.executable, HEADLESS], cwd=tempfile.gettempdir(),
                                env=clean_env(CRYPTOBOT_CONTROL_PORT="0", **env),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        port = None
        lines = []
        deadline = time.time() + 30
        while time.time() < deadline and port is None:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            m = re.search(r"Control API on http://127\.0\.0\.1:(\d+)", line)
            if m:
                port = int(m.group(1))
        self.assertIsNotNone(port, "".join(lines))
        return proc, port, lines

    def get(self, port, path):
        try:
            with NO_PROXY.open(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_serves_the_api_and_stops_cleanly_on_sigterm(self):
        proc, port, _ = self.start(BINANCE_API_KEY="your_key_here", BINANCE_API_SECRET="your_secret_here",
                                   LLAMA_SENTIMENT_SIMULATE_ONLY="False", CRYPTOBOT_ALLOW_LIVE="1")
        code, status = self.get(port, "/status")
        self.assertEqual(code, 200)
        self.assertEqual((status["mode"], status["keys_configured"], status["running"]), ("simulation", False, False))
        self.assertEqual(status["llm"], {"url": "http://127.0.0.1:9/api", "reachable": False})
        code, cfg = self.get(port, "/config")
        self.assertIn("placeholders", cfg["live_blocked_reason"])
        code, body = self.get(port, "/balances")
        self.assertEqual(code, 409)
        proc.send_signal(signal.SIGTERM)
        rest, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 0, rest)
        self.assertIn("Stopped", rest)
        self.assertNotIn("your_key_here", rest)

    def test_refuses_a_public_bind(self):
        out = subprocess.run([sys.executable, HEADLESS], cwd=ROOT, env=clean_env(CRYPTOBOT_CONTROL_BIND="0.0.0.0"),
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 2)
        self.assertIn("CRYPTOBOT_CONTROL_BIND must be a loopback address", out.stderr)

    def test_check_prints_the_effective_mode(self):
        out = subprocess.run([sys.executable, HEADLESS, "--check"], cwd=ROOT, capture_output=True, text=True, timeout=60,
                             env=clean_env(LLAMA_SENTIMENT_SIMULATE_ONLY="False", LLAMA_API_STYLE="openai"))
        self.assertEqual(out.returncode, 0, out.stderr)
        view = json.loads(out.stdout)
        self.assertEqual(view["mode"], "simulation")
        self.assertIn("CRYPTOBOT_ALLOW_LIVE is not 1", view["live_blocked_reason"])
        self.assertEqual(view["llm"]["style"], "openai")
        self.assertEqual(view["status"]["running"], False)

    def test_bad_llm_style_is_reported(self):
        out = subprocess.run([sys.executable, HEADLESS, "--check"], cwd=ROOT, capture_output=True, text=True, timeout=60,
                             env=clean_env(LLAMA_API_STYLE="gemini"))
        self.assertEqual(out.returncode, 2)
        self.assertIn("LLAMA_API_STYLE must be", out.stderr)


if __name__ == "__main__":
    unittest.main()
