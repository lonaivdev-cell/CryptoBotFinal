import json
import os
import tempfile
import unittest

import tests.fakes  # noqa: F401 - puts the repo root on sys.path

from botservice.journal import DecisionLog
from botservice.paper import PaperExchange, PaperWallet
from tests.fakes import FakeMarket


class PaperWalletTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="paper-test-")
        self.path = os.path.join(self.dir, "paper.json")

    def test_buy_and_sell_with_fees(self):
        w = PaperWallet("USDT", 1000.0, fee_pct=0.001, path=self.path)
        filled, note = w.fill("buy", "BTC", 0.01, 30000.0)
        self.assertEqual((filled, note), (0.01, "paper fill"))
        self.assertAlmostEqual(w.balances["USDT"], 1000.0 - 300.0 * 1.001)
        filled, note = w.fill("sell", "btc", 0.01, 33000.0)
        self.assertEqual(filled, 0.01)
        self.assertAlmostEqual(w.balances["USDT"], 1000.0 - 300.3 + 330.0 * 0.999)
        self.assertNotIn("BTC", w.balances)
        self.assertNotIn("BTC", w.avg_cost)

    def test_buys_are_capped_by_cash_and_sells_by_holdings(self):
        w = PaperWallet("USDT", 100.0, fee_pct=0.0)
        filled, note = w.fill("buy", "ETH", 1.0, 2000.0)
        self.assertAlmostEqual(filled, 0.05)
        self.assertIn("reduced to the cash available", note)
        self.assertAlmostEqual(w.balances["USDT"], 0.0)
        self.assertEqual(w.fill("buy", "ETH", 1.0, 2000.0)[0], 0.0)
        filled, note = w.fill("sell", "ETH", 3.0, 2000.0)
        self.assertAlmostEqual(filled, 0.05)
        self.assertIn("reduced to the amount held", note)
        self.assertEqual(w.fill("sell", "ETH", 1.0, 2000.0), (0.0, "not filled: the paper wallet holds no ETH"))
        self.assertEqual(w.fill("buy", "ETH", 1.0, None), (0.0, "not filled: no price"))

    def test_average_cost_and_valuation(self):
        w = PaperWallet("USDT", 1000.0, fee_pct=0.0)
        w.fill("buy", "SOL", 1.0, 100.0)
        w.fill("buy", "SOL", 1.0, 200.0)
        self.assertAlmostEqual(w.avg_cost["SOL"], 150.0)
        w.mark("SOL", 160.0)
        snap = w.snapshot()
        self.assertAlmostEqual(snap["value"], 700.0 + 320.0)
        self.assertAlmostEqual(snap["pnl"], 20.0)
        self.assertEqual(set(w.holdings()), {"SOL"})

    def test_unpriced_holding_makes_the_value_unknown(self):
        w = PaperWallet("USDT", 1000.0)
        w.balances["XRP"] = 5.0
        self.assertIsNone(w.snapshot()["value"])

    def test_persistence_round_trip_and_bad_files(self):
        w = PaperWallet("USDT", 1000.0, fee_pct=0.0, path=self.path)
        w.fill("buy", "BTC", 0.01, 30000.0)
        again = PaperWallet("USDT", 5000.0, fee_pct=0.0, path=self.path)
        self.assertAlmostEqual(again.balances["USDT"], 700.0)
        self.assertEqual(again.start_balance, 1000.0)  # the file's start, not the new default
        self.assertAlmostEqual(again.avg_cost["BTC"], 30000.0)
        for content in ("{not json", json.dumps({"base": "BUSD", "balances": {"BUSD": 5}}), json.dumps([1, 2])):
            with open(self.path, "w") as fh:
                fh.write(content)
            fresh = PaperWallet("USDT", 1000.0, path=self.path)
            self.assertEqual(fresh.balances, {"USDT": 1000.0})

    def test_paper_exchange_has_no_order_path(self):
        market = FakeMarket()
        market.connect()
        ex = PaperExchange(market, PaperWallet("USDT", 50.0))
        self.assertIs(ex.client, market.fake)
        self.assertEqual(ex.get_ticker("BTC/USDT")["last"], 30000.0)
        self.assertEqual(ex.get_balances(), {"USDT": {"free": 50.0, "used": 0.0, "total": 50.0}})
        with self.assertRaises(RuntimeError):
            ex.create_order("BTC/USDT", "buy", 1.0)
        self.assertEqual(market.fake.orders, [])


class DecisionLogTest(unittest.TestCase):
    def row(self, i, **kw):
        r = {"at": 1000 + i, "coin": "BTC", "side": "buy", "amount": 0.1 * i, "price": 1.0, "simulated": True, "reason": f"r{i}"}
        r.update(kw)
        return r

    def test_newest_first_with_limit(self):
        log = DecisionLog()
        log.extend([self.row(i) for i in range(5)])
        self.assertEqual([r["at"] for r in log.latest(3)], [1004, 1003, 1002])
        self.assertEqual(len(log.latest(100)), 5)

    def test_survives_a_restart_and_skips_junk(self):
        path = os.path.join(tempfile.mkdtemp(prefix="journal-test-"), "decisions.jsonl")
        DecisionLog(path).extend([self.row(1), self.row(2, extra="dropped")])
        with open(path, "a") as fh:
            fh.write("not json\n" + json.dumps({"at": "yesterday"}) + "\n")
        again = DecisionLog(path)
        rows = again.latest(10)
        self.assertEqual([r["at"] for r in rows], [1002, 1001])
        self.assertNotIn("extra", rows[0])

    def test_ring_is_bounded(self):
        log = DecisionLog(keep=3)
        log.extend([self.row(i) for i in range(10)])
        self.assertEqual(len(log), 3)


if __name__ == "__main__":
    unittest.main()
