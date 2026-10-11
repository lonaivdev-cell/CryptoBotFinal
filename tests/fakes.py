"""Fakes for the headless service's tests: a ccxt-shaped exchange client, an ExchangeWrapper
that connects to it, and a sentiment model. Nothing here touches the network or needs ccxt."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
from intelligence import ExchangeWrapper  # noqa: E402

# 64 characters like a real Binance key, but obviously ours.
REAL_LOOKING_KEY = "kEy0" * 16
REAL_LOOKING_SECRET = "sEcR" * 16

DEFAULT_PRICES = {"BTC": 30000.0, "ETH": 2000.0, "ADA": 0.5, "SOL": 100.0}


class FakeClient:
    """Just enough of a ccxt exchange for ExchangeWrapper and the strategies."""

    def __init__(self, prices: Optional[Dict[str, float]] = None, trends: Optional[Dict[str, int]] = None,
                 balance: Optional[Dict[str, float]] = None) -> None:
        self.prices = dict(prices or DEFAULT_PRICES)
        # +1: every candle closes higher; -1: lower; 0: flat. Missing coin → flat.
        self.trends = dict(trends or {})
        self.balance = dict(balance or {"USDT": 1000.0})
        self.markets = {f"{c}/USDT": {"symbol": f"{c}/USDT"} for c in self.prices}
        self.orders: List[Dict[str, Any]] = []
        self.balance_error: Optional[Exception] = None
        self.ohlcv_calls = 0

    def load_markets(self):
        return self.markets

    def market(self, symbol: str):
        return {"limits": {"cost": {"min": 5.0}}}

    def _coin(self, symbol: str) -> str:
        coin = symbol.split("/")[0]
        if coin not in self.prices:
            raise KeyError(f"no market {symbol}")
        return coin

    def fetch_ticker(self, symbol: str):
        p = self.prices[self._coin(symbol)]
        return {"symbol": symbol, "last": p, "close": p, "bid": p * 0.9999, "ask": p * 1.0001}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        self.ohlcv_calls += 1
        coin = self._coin(symbol)
        direction = self.trends.get(coin, 0)
        p = self.prices[coin]
        out = []
        for i in range(limit):
            close = p * (1 + direction * 0.001 * (i - limit + 1))
            out.append([i * 60_000, close, close, close, close, 10.0])
        return out

    def fetch_balance(self):
        if self.balance_error is not None:
            raise self.balance_error
        return {
            "total": dict(self.balance),
            "free": dict(self.balance),
            "used": {k: 0.0 for k in self.balance},
        }

    def create_market_order(self, symbol: str, side: str, amount: float):
        self.orders.append({"symbol": symbol, "side": side, "amount": amount})
        return {"id": str(len(self.orders))}

    def create_limit_order(self, symbol, side, amount, price):  # pragma: no cover - unused
        self.orders.append({"symbol": symbol, "side": side, "amount": amount, "price": price})
        return {"id": str(len(self.orders))}


class FakeMarket(ExchangeWrapper):
    """An ExchangeWrapper whose connect() attaches a FakeClient instead of a ccxt one."""

    def __init__(self, client: Optional[FakeClient] = None, api_key: str = "", api_secret: str = "",
                 connect_error: Optional[Exception] = None) -> None:
        super().__init__("binance", api_key, api_secret)
        self.fake = client or FakeClient()
        self.connect_error = connect_error
        self.connects = 0

    def connect(self) -> None:
        if self.client is not None:
            return
        self.connects += 1
        if self.connect_error is not None:
            raise self.connect_error
        self.client = self.fake


class FakeLLM:
    """Answers sentiment prompts with fixed per-coin scores, in the model's line format."""

    api_style = "openai"
    base_url = "http://127.0.0.1:8080/v1"

    def __init__(self, scores: Optional[Dict[str, float]] = None, reachable: bool = True) -> None:
        self.scores = dict(scores or {})
        self.reachable = reachable
        self.prompts: List[str] = []

    def _coins(self, prompt: str) -> List[str]:
        tail = prompt.rsplit("NOW RATE THESE COINS:", 1)[-1]
        return [line.strip() for line in tail.splitlines() if line.strip().isalnum()]

    def generate(self, prompt: str) -> Optional[str]:
        self.prompts.append(prompt)
        if not self.reachable:
            return None
        return "\n".join(f"{c}: {self.scores.get(c, 0.0):.2f}" for c in self._coins(prompt))

    def ping(self, timeout: float = 2.0) -> bool:
        return self.reachable


def make_cfg(**overrides: Any) -> types.SimpleNamespace:
    """A copy of config.py's settings with overrides; strategies still read module `config`
    for the few values they look up themselves (TTL, MIN_TRADE_USD), whose defaults suit tests."""
    values = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    values.update(
        API_KEY="", API_SECRET="", API_PASSWORD="",
        SENTIMENT_COINS=["BTC", "ETH", "ADA"],
        COOLDOWN_SECONDS=0,
        TRADE_INTERVAL=10,
    )
    values.update(overrides)
    ns = types.SimpleNamespace(**values)
    ns.build_sentiment_prompt = config.build_sentiment_prompt
    return ns


_DEFAULT = object()


def make_service(cfg=None, env=None, market=_DEFAULT, engine=None, state_dir=None, **settings_kw):
    """A BotService on fakes. market=None lets the service build its real ExchangeWrapper."""
    from botservice.service import BotService
    from botservice.settings import ServiceSettings

    if state_dir is None:
        state_dir = tempfile.mkdtemp(prefix="cryptobot-test-")
    settings = ServiceSettings(state_dir=state_dir, log_dir=os.path.join(state_dir, "logs"),
                               env_file="/etc/lo-cryptobot/cryptobot.env", **settings_kw)
    return BotService(
        settings,
        cfg=cfg if cfg is not None else make_cfg(),
        env=env if env is not None else {},
        market=FakeMarket() if market is _DEFAULT else market,
        engine=engine if engine is not None else FakeLLM({"BTC": 0.9, "ETH": 0.1, "ADA": 0.0}),
    )
