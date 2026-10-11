"""The strategies and the scheduler, driven by fakes — including the paths the desktop GUI uses."""

import threading
import time
import unittest

from tests.fakes import FakeClient, FakeLLM, FakeMarket

import config
from intelligence import LlamaHybridStrategy, TradeInstruction, TradingBot
from optimization_engine import RollingSentimentCache, SentimentRollingScheduler, parse_sentiment_lines
from portfolio_strategy import DiversifiedPortfolioStrategy


def connected(client=None):
    m = FakeMarket(client or FakeClient(trends={"BTC": 1, "ETH": -1, "ADA": 0}))
    m.connect()
    return m


def scored_cache(scores):
    cache = RollingSentimentCache()
    for coin, score in scores.items():
        cache.set(coin, score)
    return cache


class PaperlessBalances:
    def __init__(self, usdt=1000.0, **coins):
        self.rows = {"USDT": {"free": usdt, "used": 0.0, "total": usdt}}
        for c, a in coins.items():
            self.rows[c] = {"free": a, "used": 0.0, "total": a}


class HybridStrategyTest(unittest.TestCase):
    def make(self, cache=None, llm=None, **kw):
        kw.setdefault("cooldown_seconds", 0)
        return LlamaHybridStrategy(
            exchange=connected(kw.pop("client", None)), llama_engine=llm or FakeLLM({"BTC": 0.9}),
            sentiment_cache=cache, coins=["BTC", "ETH", "ADA"], base_currency="USDT",
            buy_threshold=0.45, sell_threshold=-0.40, simulate_only=True, **kw,
        )

    def test_the_cache_passed_in_is_used(self):
        # It used to be accepted and dropped, so every cycle made direct LLM calls.
        llm = FakeLLM({"BTC": 0.9})
        cache = scored_cache({"BTC": 0.9, "ETH": 0.1, "ADA": 0.0})
        s = self.make(cache=cache, llm=llm)
        self.assertIs(s.sentiment_cache, cache)
        trades = s.decide_trades(PaperlessBalances().rows)
        self.assertEqual(llm.prompts, [])
        self.assertEqual([(t.symbol, t.side) for t in trades], [("BTC/USDT", "buy")])
        t = trades[0]
        self.assertAlmostEqual(t.amount, 900.0 / 30000.0)
        self.assertEqual(t.ref_price, 30000.0)
        self.assertIsNone(t.price)  # still a market order
        self.assertIn("buy threshold", t.reason)

    def test_signals_are_recorded_per_coin(self):
        s = self.make(cache=scored_cache({"BTC": 0.9, "ETH": -0.9}))
        s.decide_trades(PaperlessBalances().rows)
        sig = s.last_signals
        self.assertEqual(set(sig), {"BTC", "ETH", "ADA"})
        self.assertEqual(sig["BTC"]["action"], "buy")
        self.assertEqual(sig["ETH"]["action"], "sell")
        self.assertEqual(sig["ADA"]["action"], "hold")
        self.assertIsNone(sig["ADA"]["sentiment"])  # never scored → null, not 0.0
        self.assertEqual(sig["BTC"]["sentiment"], 0.9)
        self.assertAlmostEqual(sig["BTC"]["trend"], 0.5)
        for row in sig.values():
            self.assertEqual(set(row), {"coin", "sentiment", "trend", "hybrid", "action", "at"})

    def test_placeholder_scores_are_reported_as_unknown(self):
        cache = RollingSentimentCache()
        cache.set("BTC", 0.0, placeholder=True)
        s = self.make(cache=cache)
        s.decide_trades(PaperlessBalances().rows)
        self.assertIsNone(s.last_signals["BTC"]["sentiment"])

    def test_missing_candles_do_not_crash_the_cycle(self):
        # Trend None used to raise TypeError in abs()/float() — the GUI worker's path.
        client = FakeClient()

        def no_candles(*a, **k):
            raise RuntimeError("exchange down")

        client.fetch_ohlcv = no_candles
        s = self.make(cache=scored_cache({"BTC": 0.9}), client=client)
        trades = s.decide_trades(PaperlessBalances().rows)
        self.assertIsNone(s.last_signals["BTC"]["trend"])
        self.assertEqual([t.side for t in trades], ["buy"])  # 0.9 sentiment alone clears 0.45

    def test_gui_worker_construction_without_a_cache(self):
        # dashboard_enhanced.TradeWorker builds the strategy exactly like this: no cache, an
        # extra min_trade_usd kwarg, direct LLM calls.
        llm = FakeLLM({"BTC": 0.9, "ETH": 0.1, "ADA": 0.0})
        s = LlamaHybridStrategy(
            exchange=connected(), llama_engine=llm, coins=["BTC", "ETH", "ADA"],
            sentiment_prompt=config.LLAMA_SENTIMENT_PROMPT, base_currency="USDT",
            buy_threshold=config.SENTIMENT_BUY_THRESHOLD, sell_threshold=config.SENTIMENT_SELL_THRESHOLD,
            trade_fraction=config.SENTIMENT_TRADE_FRACTION, use_convert=config.LLAMA_SENTIMENT_USE_CONVERT,
            simulate_only=True, sentiment_weight=0.6, trend_weight=0.4, require_trend_confirmation=True,
            min_trade_usd=config.MIN_TRADE_USD, min_sentiment_delta=config.MIN_SENTIMENT_DELTA,
        )
        trades = s.decide_trades(PaperlessBalances().rows)
        self.assertTrue(llm.prompts)  # direct calls, as before
        self.assertEqual([t.symbol for t in trades], ["BTC/USDT"])

    def test_sell_of_a_held_coin(self):
        s = self.make(cache=scored_cache({"BTC": 0.9, "ETH": -0.9}))
        s.positions["ETH"] = {"entry_price": 2100.0, "amount": 0.5}
        trades = s.decide_trades(PaperlessBalances(usdt=5.0, ETH=0.5).rows)
        self.assertEqual([(t.symbol, t.side, t.amount) for t in trades], [("ETH/USDT", "sell", 0.5)])
        self.assertIn("sell threshold", trades[0].reason)
        self.assertNotIn("ETH", s.positions)

    def test_convert_never_runs_in_simulation(self):
        class Convert:
            calls = 0

            def get_quote(self, *a, **k):
                Convert.calls += 1
                return {"quoteId": "q"}

            def accept_quote(self, *a, **k):
                Convert.calls += 1

        s = self.make(cache=scored_cache({"BTC": 0.9}), use_convert=True, convert_client=Convert())
        trades = s.decide_trades(PaperlessBalances().rows)
        self.assertEqual(Convert.calls, 0)
        self.assertEqual([t.side for t in trades], ["buy"])  # an ordinary instruction instead

    def test_opportunistic_cycle_defaults_to_the_strategy_mode(self):
        client = FakeClient()
        s = self.make(cache=scored_cache({}), client=client)
        s.opportunistic_coins = ["ADA"]
        s._opp_positions["ADA"] = {"entry_price": 0.4, "amount": 100.0}  # +25%: take profit
        actions = s.run_opportunistic_cycle()
        self.assertEqual([a["action"] for a in actions], ["opp_tp_sell"])
        self.assertEqual(client.orders, [])  # it used to default to simulate_only=False


class PortfolioStrategyTest(unittest.TestCase):
    def make(self, cache, client=None, **kw):
        kw.setdefault("cooldown_seconds", 0)
        return DiversifiedPortfolioStrategy(
            exchange=connected(client), llama_engine=FakeLLM(), sentiment_cache=cache,
            coins=["BTC", "ETH", "ADA"], base_currency="USDT", buy_threshold=0.45, sell_threshold=-0.40,
            simulate_only=True, stop_loss_pct=0.15, position_max_days=10, **kw,
        )

    def test_simulation_returns_the_trades(self):
        # It used to return [] whenever simulate_only was set — and crash before that on helpers
        # that did not exist (_get_sentiment_scores, executor, _compute_hybrid_score).
        s = self.make(scored_cache({"BTC": 0.9, "ETH": 0.1, "ADA": 0.0}))
        trades = s.decide_trades(PaperlessBalances().rows)
        self.assertEqual([(t.symbol, t.side) for t in trades], [("BTC/USDT", "buy")])
        self.assertIn("core", trades[0].reason)
        self.assertIn("BTC", s.entry_prices)
        self.assertEqual(set(s.last_signals), {"BTC", "ETH", "ADA"})

    def test_buys_without_trend_confirmation_too(self):
        s = self.make(scored_cache({"BTC": 0.9}), require_trend_confirmation=False)
        self.assertEqual([t.side for t in s.decide_trades(PaperlessBalances().rows)], ["buy"])

    def test_opportunistic_bucket(self):
        client = FakeClient(trends={"ADA": 1})
        s = self.make(scored_cache({"ADA": 0.95}), client=client)
        trades = s.decide_trades(PaperlessBalances().rows)
        self.assertEqual([t.symbol for t in trades], ["ADA/USDT"])
        self.assertAlmostEqual(trades[0].amount * trades[0].ref_price, 100.0)  # 10% bucket
        self.assertIn("opportunistic", trades[0].reason)

    def test_stop_loss(self):
        client = FakeClient(trends={"BTC": 1})
        s = self.make(scored_cache({"BTC": 0.9}), client=client)
        s.decide_trades(PaperlessBalances().rows)
        client.prices["BTC"] = 24000.0  # -20%
        s._price_cache.clear()
        trades = s.decide_trades(PaperlessBalances(usdt=5.0, BTC=0.03).rows)
        self.assertEqual([(t.side, t.amount) for t in trades], [("sell", 0.03)])
        self.assertIn("stop-loss", trades[0].reason)
        self.assertIn("-20.0%", trades[0].reason)


class TradingBotTest(unittest.TestCase):
    class Strategy:
        def __init__(self, trades):
            self.trades = trades
            self.ended = []

        def decide_trades(self, balances, **kw):
            return list(self.trades)

        def on_cycle_end(self, data):
            self.ended.append(data)

    def test_decisions_are_recorded_and_simulation_places_nothing(self):
        market = connected()
        t = TradeInstruction("BTC/USDT", "buy", 0.01, reason="because", ref_price=30000.0)
        bot = TradingBot(market, self.Strategy([t]), simulate_only=True)
        self.assertEqual(bot.run_cycle(), [t])
        self.assertEqual(market.fake.orders, [])
        d = bot.last_cycle_decisions[0]
        self.assertEqual({k: d[k] for k in ("coin", "side", "amount", "price", "simulated", "reason")},
                         {"coin": "BTC", "side": "buy", "amount": 0.01, "price": 30000.0, "simulated": True, "reason": "because"})
        self.assertEqual((bot.cycles, len(bot.decisions)), (1, 1))

    def test_live_orders_and_their_failures(self):
        market = connected()
        ok = TradeInstruction("BTC/USDT", "buy", 0.01, ref_price=30000.0)
        bad = TradeInstruction("NOPE/USDT", "sell", 1.0)
        orig = market.fake.create_market_order

        def create(symbol, side, amount):
            if symbol.startswith("NOPE"):
                raise RuntimeError("Invalid symbol")
            return orig(symbol, side, amount)

        market.fake.create_market_order = create
        bot = TradingBot(market, self.Strategy([ok, bad]), simulate_only=False)
        bot.run_cycle()
        self.assertEqual(market.fake.orders, [{"symbol": "BTC/USDT", "side": "buy", "amount": 0.01}])
        first, second = bot.last_cycle_decisions
        self.assertFalse(first["simulated"])
        self.assertIn("order failed: Invalid symbol", second["reason"])


class SchedulerTest(unittest.TestCase):
    def test_parse_sentiment_lines(self):
        text = "Date: 2026-10-11\nBTC: 0.42\n**ETH**: -0.3.\n- SOL: +1.7\nnonsense\nXRP: abc\n  DOGE : 0.05\nADA:-2"
        self.assertEqual(parse_sentiment_lines(text),
                         {"DATE": 1.0, "BTC": 0.42, "ETH": -0.3, "SOL": 1.0, "DOGE": 0.05, "ADA": -1.0})
        self.assertEqual(parse_sentiment_lines(None), {})

    def test_cache_placeholders(self):
        cache = RollingSentimentCache()
        cache.set("BTC", 0.0, ts=time.time() - 10, placeholder=True)
        self.assertFalse(cache.has_real_score("BTC"))
        self.assertEqual(cache.get("BTC"), 0.0)
        cache.set("BTC", 0.4)
        self.assertTrue(cache.has_real_score("BTC"))
        self.assertFalse(cache.has_real_score("ETH"))

    def make(self, llm, cache, stop):
        return SentimentRollingScheduler(
            llama_engine=llm, coins=["BTC", "ETH"], cache=cache, ttl_seconds=60, max_qps=50.0,
            batch_size=1, stop_event=stop, build_prompt_fn=config.build_sentiment_prompt,
            enable_jitter=False, adaptive_rate=False,
        )

    def test_scheduler_fills_the_cache_and_stops_promptly(self):
        cache, stop = RollingSentimentCache(), threading.Event()
        sched = self.make(FakeLLM({"BTC": 0.7, "ETH": -0.2}), cache, stop)
        # The scheduler staggers first refreshes across the TTL (ETH would wait ~18 s); age ETH's
        # placeholder so both are due now.
        cache.set("ETH", 0.0, ts=time.time() - 1000, placeholder=True)
        sched.start()
        deadline = time.time() + 5
        while time.time() < deadline and not (cache.has_real_score("BTC") and cache.has_real_score("ETH")):
            time.sleep(0.02)
        stop.set()
        sched.join(timeout=2)
        self.assertFalse(sched.is_alive())
        self.assertEqual((cache.get("BTC"), cache.get("ETH")), (0.7, -0.2))

    def test_unreachable_model_backs_off_and_still_stops(self):
        cache, stop = RollingSentimentCache(), threading.Event()
        llm = FakeLLM(reachable=False)
        sched = self.make(llm, cache, stop)
        sched.start()
        deadline = time.time() + 5
        while time.time() < deadline and not llm.prompts:
            time.sleep(0.02)
        time.sleep(0.3)  # well inside the 5 s back-off that follows the failed call
        stop.set()
        sched.join(timeout=2)
        self.assertFalse(sched.is_alive())  # the 5 s back-off sleep is interrupted by stop
        self.assertEqual(len(llm.prompts), 1)  # and nothing hammered the dead model meanwhile

    def test_a_new_scheduler_keeps_real_scores(self):
        cache = RollingSentimentCache()
        cache.set("BTC", 0.6)
        self.make(FakeLLM(), cache, threading.Event())  # seeds placeholders in __init__
        self.assertTrue(cache.has_real_score("BTC"))
        self.assertEqual(cache.get("BTC"), 0.6)
        self.assertFalse(cache.has_real_score("ETH"))


if __name__ == "__main__":
    unittest.main()
