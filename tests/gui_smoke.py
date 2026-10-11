"""Run by test_gui_smoke.py in a subprocess (offscreen Qt, patched network): main.main() wires the
desktop app as usual, then — instead of entering the event loop — one TradeWorker cycle (the
GUI's own trading path) and one cycle of main.py's bot run, and the result is printed as JSON.

Before the headless work both cycles crashed (missing strategy helpers); this keeps them working.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.chdir(tempfile.mkdtemp(prefix="gui-smoke-"))  # main.py logs to ./logs

from tests.fakes import FakeClient  # noqa: E402

import intelligence  # noqa: E402

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX"]
CLIENT = FakeClient(prices={c: 10.0 * (i + 1) for i, c in enumerate(COINS)}, trends={"BTC": 1})


def _connect(self):
    self.client = CLIENT


def _generate(self, prompt):
    return "\n".join(f"{c}: {0.9 if c == 'BTC' else 0.0}" for c in COINS if c in prompt)


intelligence.ExchangeWrapper.connect = _connect
intelligence.LlamaDecisionEngine.generate = _generate

import dashboard_enhanced  # noqa: E402
import main  # noqa: E402


def _no_network(*a, **k):
    raise RuntimeError("no network in the smoke test")


dashboard_enhanced.requests.get = _no_network
main.time.sleep = lambda s: None
results = {}


def fake_run_dashboard(bot, exchange, log_q):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    win = dashboard_enhanced.TradingApp(bot, exchange, log_q)
    win.show()
    app.processEvents()
    got = {}
    win.trade_worker.done.connect(lambda trades, bal, usd: got.update(done=[[t.symbol, t.side] for t in trades]))
    win.trade_worker.error.connect(lambda msg: got.update(error=msg))
    win.trade_worker.run_once()
    app.processEvents()
    results["gui_worker"] = got
    try:
        results["main_bot_cycle"] = [[t.symbol, t.side] for t in bot.run_cycle()]
    except Exception as exc:  # reported, not raised: the test asserts on it
        results["main_bot_cycle"] = f"{type(exc).__name__}: {exc}"
    results["strategy_class"] = type(bot.strategy).__name__
    win.start_bot()
    app.processEvents()
    win.stop_bot()
    win.update_ui(win.last_balances)
    app.processEvents()
    for thread in (win.trade_thread, win.mkt_thread):
        thread.quit()
        thread.wait(3000)
    win.close()
    app.processEvents()


main.run_dashboard = fake_run_dashboard
main.main()
results["orders_placed"] = len(CLIENT.orders)
print("RESULT " + json.dumps(results))
