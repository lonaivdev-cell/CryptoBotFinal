"""The desktop GUI path still works (skipped where PySide6/matplotlib are not installed)."""

import importlib.util
import json
import os
import subprocess
import sys
import unittest

from tests.fakes import ROOT

HAVE_GUI = all(importlib.util.find_spec(m) is not None for m in ("PySide6", "matplotlib"))


@unittest.skipUnless(HAVE_GUI, "PySide6 and matplotlib are not installed (pip install -r requirements.txt)")
class GuiSmokeTest(unittest.TestCase):
    def test_main_and_trade_worker_cycles(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("BINANCE_", "LLAMA_", "SENTIMENT_"))}
        env["QT_QPA_PLATFORM"] = "offscreen"
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "gui_smoke.py")], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=180)
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(lines, out.stdout[-2000:] + out.stderr[-2000:])
        result = json.loads(lines[-1][len("RESULT "):])
        self.assertEqual(result["strategy_class"], "DiversifiedPortfolioStrategy")
        self.assertEqual(result["gui_worker"], {"done": [["BTC/USDT", "buy"]]})
        self.assertEqual(result["main_bot_cycle"], [["BTC/USDT", "buy"]])
        self.assertEqual(result["orders_placed"], 0)  # simulation by default, in the GUI too


if __name__ == "__main__":
    unittest.main()
