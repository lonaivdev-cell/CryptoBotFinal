"""BotService: wiring, mode, start/stop/cycle, and what simulation and live each do."""

import os
import tempfile
import threading
import time
import unittest

from tests.fakes import REAL_LOOKING_KEY, REAL_LOOKING_SECRET, FakeClient, FakeLLM, FakeMarket, make_cfg, make_service

from botservice.service import Conflict, Upstream
from botservice.settings import SettingsError

LIVE_ENV = {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False", "CRYPTOBOT_ALLOW_LIVE": "1"}


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class ModeTest(unittest.TestCase):
    def test_placeholder_keys_mean_public_data_and_simulation(self):
        market = FakeMarket(FakeClient(trends={"BTC": 1}))
        svc = make_service(cfg=make_cfg(API_KEY="your_key_here", API_SECRET="your_secret_here"), env=LIVE_ENV, market=market)
        self.assertFalse(svc.keys_ok)
        self.assertEqual(svc.gate.mode, "simulation")
        self.assertIn("placeholders", svc.gate.reason)
        self.assertIsNone(svc.account)
        with self.assertRaises(Conflict) as ctx:
            svc.balances()
        message = str(ctx.exception)
        self.assertTrue(message.startswith("Binance keys are still placeholders"))
        self.assertIn("/etc/lo-cryptobot/cryptobot.env", message)
        self.assertIn("lo-cryptobot.service", message)
        # Market data still flows: a cycle runs and trades on paper.
        decisions = svc.run_cycle()
        self.assertEqual([(d["coin"], d["side"], d["simulated"]) for d in decisions], [("BTC", "buy", True)])
        self.assertEqual(market.fake.orders, [])

    def test_the_default_exchange_wrapper_gets_no_placeholder_credentials(self):
        svc = make_service(cfg=make_cfg(API_KEY="PLACEHOLDER_BINANCE_API_KEY", API_SECRET="PLACEHOLDER_BINANCE_API_SECRET"),
                           market=None)
        # market=None: BotService built its own ExchangeWrapper.
        self.assertEqual(type(svc.market).__name__, "ExchangeWrapper")
        self.assertEqual((svc.market.api_key, svc.market.api_secret, svc.market.password), ("", "", ""))
        self.assertIsNone(svc.market.client)  # nothing connects (or needs ccxt) until a cycle

    def test_live_mode_places_real_orders_through_the_exchange(self):
        client = FakeClient(trends={"BTC": 1}, balance={"USDT": 1000.0})
        market = FakeMarket(client, api_key=REAL_LOOKING_KEY, api_secret=REAL_LOOKING_SECRET)
        svc = make_service(cfg=make_cfg(API_KEY=REAL_LOOKING_KEY, API_SECRET=REAL_LOOKING_SECRET), env=LIVE_ENV, market=market)
        self.assertEqual(svc.status()["mode"], "live")
        self.assertIsNone(svc.wallet)
        decisions = svc.run_cycle()
        self.assertEqual([(d["coin"], d["side"], d["simulated"]) for d in decisions], [("BTC", "buy", False)])
        self.assertEqual(client.orders, [{"symbol": "BTC/USDT", "side": "buy", "amount": 0.03}])
        self.assertEqual(svc.balances(), {"balances": {"USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0}}})
        with self.assertRaises(Conflict):
            svc.paper()

    def test_real_keys_without_both_switches_still_simulate(self):
        client = FakeClient(trends={"BTC": 1})
        cfg = make_cfg(API_KEY=REAL_LOOKING_KEY, API_SECRET=REAL_LOOKING_SECRET)
        for env in ({}, {"CRYPTOBOT_ALLOW_LIVE": "1"}, {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False"},
                    {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False", "CRYPTOBOT_ALLOW_LIVE": "true"}):
            with self.subTest(env=env):
                svc = make_service(cfg=cfg, env=env, market=FakeMarket(client, REAL_LOOKING_KEY, REAL_LOOKING_SECRET))
                self.assertEqual(svc.status()["mode"], "simulation")
                self.assertTrue(all(d["simulated"] for d in svc.run_cycle()))
                self.assertEqual(client.orders, [])
                # Real keys still give real (read-only) balances in simulation.
                self.assertIn("USDT", svc.balances()["balances"])

    def test_balance_errors_are_redacted(self):
        client = FakeClient()
        client.balance_error = RuntimeError(f"binance {{\"apiKey\": \"{REAL_LOOKING_KEY}\"}} Invalid API-key, IP, or permissions")
        svc = make_service(cfg=make_cfg(API_KEY=REAL_LOOKING_KEY, API_SECRET=REAL_LOOKING_SECRET),
                           market=FakeMarket(client, REAL_LOOKING_KEY, REAL_LOOKING_SECRET))
        with self.assertRaises(Upstream) as ctx:
            svc.balances()
        self.assertNotIn(REAL_LOOKING_KEY, str(ctx.exception))
        self.assertIn("[redacted]", str(ctx.exception))
        self.assertIn("Invalid API-key", str(ctx.exception))

    def test_config_never_contains_keys(self):
        svc = make_service(cfg=make_cfg(API_KEY=REAL_LOOKING_KEY, API_SECRET=REAL_LOOKING_SECRET, LLAMA_API_KEY="sk-llm-secret-value"))
        text = repr(svc.config_view()) + repr(svc.status())
        for secret in (REAL_LOOKING_KEY, REAL_LOOKING_SECRET, "sk-llm-secret-value"):
            self.assertNotIn(secret, text)
        view = svc.config_view()
        for field in ("coins", "thresholds", "weights", "interval", "simulate_only", "strategy"):
            self.assertIn(field, view)

    def test_bad_llm_style_refuses_to_start(self):
        from botservice.service import BotService
        from botservice.settings import ServiceSettings

        with self.assertRaises(SettingsError) as ctx:
            BotService(ServiceSettings(state_dir=tempfile.mkdtemp()), cfg=make_cfg(LLAMA_API_STYLE="claude"), env={},
                       market=FakeMarket())
        self.assertIn("LLAMA_API_STYLE", str(ctx.exception))


class CycleTest(unittest.TestCase):
    def test_signals_and_decisions_come_from_the_strategy(self):
        svc = make_service(market=FakeMarket(FakeClient(trends={"BTC": 1, "ETH": -1})))
        self.assertEqual(svc.signals(), {"signals": []})
        svc.run_cycle()
        signals = svc.signals()["signals"]
        self.assertEqual([s["coin"] for s in signals], ["BTC", "ETH", "ADA"])
        self.assertEqual(signals[0]["action"], "buy")
        rows = svc.decisions(20)["decisions"]
        self.assertEqual(len(rows), 1)
        self.assertIn("paper fill", rows[0]["reason"])
        st = svc.status()
        self.assertEqual((st["cycles"], st["running"], st["last_error"]), (1, False, None))
        self.assertIsNotNone(st["last_cycle_at"])

    def test_stopped_cycle_refreshes_sentiment_itself(self):
        llm = FakeLLM({"BTC": 0.9, "ETH": 0.2, "ADA": -0.1})
        svc = make_service(engine=llm)
        svc.run_cycle()
        self.assertEqual(len(llm.prompts), 3)  # one per coin (SENTIMENT_BATCH_SIZE=1)
        self.assertEqual({s["coin"]: s["sentiment"] for s in svc.signals()["signals"]}, {"BTC": 0.9, "ETH": 0.2, "ADA": -0.1})
        svc.run_cycle()
        self.assertEqual(len(llm.prompts), 3)  # fresh scores are not asked for again

    def test_unreachable_model_means_trend_only(self):
        llm = FakeLLM(reachable=False)
        svc = make_service(engine=llm, market=FakeMarket(FakeClient(trends={"BTC": 1})))
        decisions = svc.run_cycle()
        self.assertEqual(llm.prompts, [])  # ping said no; nothing waited on a dead model
        self.assertEqual(decisions, [])  # trend 0.5 alone is a hybrid of 0.2, under the 0.45 bar
        self.assertTrue(all(s["sentiment"] is None for s in svc.signals()["signals"]))
        self.assertFalse(svc.status()["llm"]["reachable"])
        self.assertIsNone(svc.status()["last_error"])  # a stopped model is a state, not an error

    def test_sentiment_refresh_respects_its_budget(self):
        llm = FakeLLM({"BTC": 0.9})
        svc = make_service(engine=llm, cycle_sentiment_budget_s=0.0)
        svc.run_cycle()
        self.assertEqual(llm.prompts, [])

    def test_exchange_outage_is_reported_and_retried(self):
        market = FakeMarket(connect_error=OSError("Network is unreachable"))
        svc = make_service(market=market)
        with self.assertRaises(Upstream) as ctx:
            svc.run_cycle()
        self.assertIn("Could not connect to binance", str(ctx.exception))
        self.assertIn("Network is unreachable", svc.status()["last_error"])
        market.connect_error = None
        svc.run_cycle()
        self.assertEqual(market.connects, 2)
        self.assertIsNone(svc.status()["last_error"])

    def test_paper_holdings_survive_a_restart_and_can_be_stopped_out(self):
        state = tempfile.mkdtemp(prefix="cryptobot-restart-")
        client = FakeClient(trends={"BTC": 1})
        first = make_service(market=FakeMarket(client), state_dir=state)
        first.run_cycle()
        first.shutdown()
        self.assertTrue(os.path.exists(os.path.join(state, "paper.json")))

        client.prices["BTC"] = 24000.0
        second = make_service(market=FakeMarket(client), state_dir=state)
        self.assertEqual(len(second.decisions(10)["decisions"]), 1)  # the journal came back too
        decisions = second.run_cycle()
        self.assertEqual(decisions[0]["side"], "sell")
        self.assertIn("stop-loss", decisions[0]["reason"])
        self.assertAlmostEqual(decisions[0]["amount"], 0.03)

    def test_cycles_are_serialised(self):
        svc = make_service()
        svc._cycle_lock.acquire()
        try:
            with self.assertRaises(Conflict):
                svc.run_cycle(wait_s=0.05)
        finally:
            svc._cycle_lock.release()


class LifecycleTest(unittest.TestCase):
    def test_start_runs_cycles_on_a_thread_and_stop_ends_them(self):
        llm = FakeLLM({"BTC": 0.9, "ETH": 0.1, "ADA": 0.0})
        svc = make_service(engine=llm, interval_s=0.05)
        st = svc.start("hybrid")
        self.assertTrue(st["running"])
        self.assertEqual(st["strategy"], "hybrid")
        self.assertIsNotNone(st["started_at"])
        self.assertTrue(wait_for(lambda: svc.cycles >= 3))
        self.assertTrue(svc._scheduler is not None and svc._scheduler.is_alive())
        with self.assertRaises(Conflict) as ctx:
            svc.start("portfolio")
        self.assertIn("already running", str(ctx.exception))
        st = svc.stop()
        self.assertFalse(st["running"])
        self.assertIsNone(st["started_at"])
        self.assertFalse(svc._loop_thread.is_alive())
        self.assertTrue(wait_for(lambda: not svc._scheduler.is_alive(), 3))
        count = svc.cycles
        time.sleep(0.2)
        self.assertEqual(svc.cycles, count)
        self.assertFalse(svc.stop()["running"])  # stopping twice is fine

    def test_restart_with_another_strategy(self):
        svc = make_service(interval_s=0.05)
        svc.start("hybrid")
        svc.stop()
        st = svc.start("portfolio")
        self.assertEqual(st["strategy"], "portfolio")
        self.assertEqual(type(svc._bot.strategy).__name__, "DiversifiedPortfolioStrategy")
        svc.stop()

    def test_bad_strategy(self):
        svc = make_service()
        for bad in ("yolo", 3, ["hybrid"]):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception) as ctx:
                    svc.start(bad)
                self.assertEqual(getattr(ctx.exception, "status", None), 400)
        self.assertFalse(svc.running)

    def test_a_cycle_during_the_loop_waits_its_turn(self):
        svc = make_service(interval_s=0.05)
        svc.start()
        try:
            results = []
            threads = [threading.Thread(target=lambda: results.append(svc.run_cycle())) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(5)
            self.assertEqual(len(results), 3)
        finally:
            svc.stop()


if __name__ == "__main__":
    unittest.main()
