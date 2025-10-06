"""
Core intelligence module for CryptoBot2 — merged & optimized (hybrid‑only).

This version removes the standalone Sentiment and Trend strategies and
consolidates their logic into a single, robust **LlamaHybridStrategy**.

Main components:
- BinanceConvertClient: minimal Binance Convert API client.
- ExchangeWrapper: light CCXT wrapper for balances/tickers/orders.
- TradeInstruction & BaseStrategy: small abstractions for strategies.
- LlamaDecisionEngine: resilient Ollama/LLaMA client with retries.
- LlamaHybridStrategy: **only strategy**. Blends LLM sentiment and
  technical trend signals with safety (cooldowns, min notional checks,
  optional trailing stop, and dynamic sizing).
- fetch_live_usd_rates / update_usd_rates_batch: helpers for price refs.
- TradingBot: orchestrates cycles (fetch balances → strategy → place orders).

Notes
-----
* The strategy expects line‑based output like "BTC: 0.42" from the model.
* If you have a config.build_sentiment_prompt(coins, include_date=False)
  it will be used; otherwise a default prompt is generated.
* TradeAnalyzer is optional; if present it can auto‑tune thresholds.
* Optional WatchlistManager can be wired in if you want dynamic universes.

Drop‑ins
--------
from intelligence import ExchangeWrapper, LlamaDecisionEngine, LlamaHybridStrategy, TradingBot

exchange = ExchangeWrapper("binance", api_key, api_secret, testnet=False)
exchange.connect()
engine = LlamaDecisionEngine(model_name=config.LLAMA_MODEL_NAME, base_url=config.LLAMA_API_BASE_URL)
strategy = LlamaHybridStrategy(exchange, llama_engine=engine, coins=["BTC","ETH","SOL"], base_currency="USDT")
bot = TradingBot(exchange, strategy, base_currency="USDT", simulate_only=True)
bot.run_cycle()  # or schedule periodically
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Protocol

import ccxt  # type: ignore

try:
    import config  # type: ignore
except Exception:  # minimal fallbacks for local testing
    class _Cfg:
        LLAMA_API_BASE_URL = "http://localhost:11434/api"
        LLAMA_MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
        LLAMA_KEEP_ALIVE = "2h"
        LLAMA_OPTIONS = {}
        LLAMA_MAX_RETRIES = 3
        LLAMA_REQUEST_TIMEOUT = 45
        LLAMA_RETRY_BACKOFF = 1.6
        MIN_TRADE_USD = 10.0
    config = _Cfg()  # type: ignore

# Optional analyzer
try:
    from trade_analyzer import TradeAnalyzer  # type: ignore
except Exception:  # graceful fallback
    class TradeAnalyzer:  # type: ignore
        def __init__(self, *_, **__):
            pass
        def analyze_history(self) -> Dict[str, float]:
            return {"total_trades": 0, "win_rate": 0.0}

# Optional symbol score parser
try:
    from parsers import parse_symbol_scores as _parse_scores_ext  # type: ignore
except Exception:
    _parse_scores_ext = None


# ======================================================================
# Helpers

class SentimentCache(Protocol):
    def get(self, key: str, default: Optional[float] = None) -> Optional[float]: ...
    def get_fresh(self, key: str, ttl_seconds: int) -> Optional[float]: ...
    # If you also have .set(), add it here:
    # def set(self, key: str, value: float) -> None: ...



def parse_symbol_scores(text: Optional[str], expected_order: Optional[List[str]] = None) -> Dict[str, float]:
    """Parse lines like "BTC: 0.42" into {"BTC": 0.42}.
    Robust to extra words and whitespace. If expected_order is given,
    missing symbols are filled with 0.0 so the caller can preserve layout.
    """
    scores: Dict[str, float] = {}
    if not text:
        return {sym: 0.0 for sym in (expected_order or [])}
    # Prefer external parser if available
    if _parse_scores_ext is not None:
        try:
            parsed = _parse_scores_ext(text, expected_order=expected_order)
            if isinstance(parsed, dict) and parsed:
                return {k.upper(): float(v) for k, v in parsed.items()}
        except Exception:
            pass
    # Fallback regex parse
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        left, right = line.split(":", 1)
        sym = re.sub(r"[^A-Za-z0-9]", "", left).upper()
        if not sym:
            continue
        # extract first float in the RHS
        m = re.search(r"[-+]?\d*\.?\d+", right)
        if not m:
            continue
        try:
            val = float(m.group(0))
            # clamp to [-1, 1]
            if val > 1:
                val = 1.0
            if val < -1:
                val = -1.0
            scores[sym] = round(val, 2)
        except Exception:
            continue
    if expected_order:
        for sym in expected_order:
            scores.setdefault(sym, 0.0)
    return scores


# ======================================================================
# Binance Convert API client

class BinanceConvertClient:
    """Minimal client for Binance Convert API endpoints.

    You must enable Convert permissions on the key used here.
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.binance.com") -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._session = None
        try:
            import requests  # type: ignore
            self._requests = requests
            self._session = requests.Session()
        except Exception:
            self._requests = None

    def _sign_params(self, params: Dict[str, Any]) -> str:
        encoded = urllib.parse.urlencode(params)
        signature = hmac.new(self.api_secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}&signature={signature}"

    def _send_signed_request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        params.setdefault("timestamp", int(time.time() * 1000))
        query = self._sign_params(params)
        url = f"{self.base_url}{path}?{query}"
        headers = {"X-MBX-APIKEY": self.api_key}
        if self._session is not None:
            resp = self._session.request(method, url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        req = urllib.request.Request(url, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:  # nosec - controlled host
            return json.load(resp)

    def get_quote(
        self,
        from_asset: str,
        to_asset: str,
        from_amount: Optional[float] = None,
        to_amount: Optional[float] = None,
        wallet_type: str = "SPOT",
        valid_time: str = "10s",
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "fromAsset": from_asset,
            "toAsset": to_asset,
            "walletType": wallet_type,
            "validTime": valid_time,
        }
        if from_amount is not None:
            params["fromAmount"] = str(from_amount)
        elif to_amount is not None:
            params["toAmount"] = str(to_amount)
        else:
            raise ValueError("Either from_amount or to_amount must be provided")
        return self._send_signed_request("POST", "/sapi/v1/convert/getQuote", params)

    def accept_quote(self, quote_id: str) -> Dict[str, Any]:
        return self._send_signed_request("POST", "/sapi/v1/convert/acceptQuote", {"quoteId": quote_id})


# ======================================================================
# Exchange wrapper

class ExchangeWrapper:
    """Thin wrapper around a CCXT exchange client."""
    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        password: str = "",
        enable_rate_limit: bool = True,
        testnet: bool = False,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.password = password
        self.enable_rate_limit = enable_rate_limit
        self.testnet = testnet
        self.client: Optional[ccxt.Exchange] = None

    def connect(self) -> None:
        if self.client is not None:
            return
        klass = getattr(ccxt, self.exchange_id)
        params: Dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": self.enable_rate_limit,
        }
        if self.password:
            params["password"] = self.password
        self.client = klass(params)
        if self.testnet and hasattr(self.client, "set_sandbox_mode"):
            self.client.set_sandbox_mode(True)
            self.logger.info("Activated sandbox mode for %s", self.exchange_id)
        self.client.load_markets()
        self.logger.info("Connected to %s", self.exchange_id)

    def get_balances(self) -> Dict[str, Dict[str, float]]:
        if self.client is None:
            raise RuntimeError("Exchange not connected – call connect() first")

        # Always load markets once to ensure symbols/currencies are known
        try:
            self.client.load_markets()
        except Exception:
            pass  # don't fail if already loaded

        balances = self.client.fetch_balance()
        result: Dict[str, Dict[str, float]] = {}

        # Some exchanges (Binance) report in lower/upper mixed case
        totals = balances.get("total", {})
        frees = balances.get("free", {})
        useds = balances.get("used", {})

        for currency, total_val in totals.items():
            try:
                free_val = frees.get(currency, 0.0) or 0.0
                used_val = useds.get(currency, 0.0) or 0.0

                total_val = float(total_val or 0.0)
                free_val = float(free_val)
                used_val = float(used_val)

                # Only include if non-zero (avoids clutter)
                if total_val > 0:
                    # Always uppercase keys to avoid mismatches later
                    result[currency.upper()] = {
                        "total": total_val,
                        "free": free_val,
                        "used": used_val,
                    }
            except Exception:
                continue

        return result

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Exchange not connected – call connect() first")
        return self.client.fetch_ticker(symbol)

    def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Exchange not connected – call connect() first")
        side = side.lower()
        order_type = order_type.lower()
        assert side in {"buy", "sell"}, "side must be 'buy' or 'sell'"
        assert order_type in {"market", "limit"}, "order_type must be 'market' or 'limit'"
        if order_type == "market":
            return self.client.create_market_order(symbol, side, amount)
        if price is None:
            raise ValueError("Price must be provided for limit orders")
        return self.client.create_limit_order(symbol, side, amount, price)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        if self.client is None:
            raise RuntimeError("Exchange not connected – call connect() first")
        if symbol:
            self.client.cancel_all_orders(symbol)
            return
        for order in self.client.fetch_open_orders():
            try:
                self.client.cancel_order(order["id"], order["symbol"])
            except Exception as exc:
                self.logger.warning("Failed to cancel order %s: %s", order.get("id"), exc)


# ======================================================================
# Trade representation and base strategy

@dataclass
class TradeInstruction:
    symbol: str
    side: str  # 'buy' | 'sell'
    amount: float
    price: Optional[float] = None
    order_type: str = "market"


class BaseStrategy:
    def __init__(self, exchange: ExchangeWrapper, base_currency: str) -> None:
        self.exchange = exchange
        self.base_currency = base_currency
        self.logger = logging.getLogger(self.__class__.__name__)

    def decide_trades(self, balances: Dict[str, Dict[str, float]], **kwargs) -> List[TradeInstruction]:
        raise NotImplementedError

    def on_cycle_end(self, cycle_data: Dict[str, Any]) -> None:
        pass


# ======================================================================
# LLaMA / Ollama client

class LlamaDecisionEngine:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/api",
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
        request_timeout: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or config.LLAMA_API_BASE_URL).rstrip("/")
        self.system_prompt = system_prompt or ""
        self.max_retries = max_retries if max_retries is not None else getattr(config, "LLAMA_MAX_RETRIES", 3)
        self.request_timeout = request_timeout if request_timeout is not None else getattr(config, "LLAMA_REQUEST_TIMEOUT", 45)
        self.retry_backoff = retry_backoff if retry_backoff is not None else getattr(config, "LLAMA_RETRY_BACKOFF", 1.6)
        self.options = dict(getattr(config, "LLAMA_OPTIONS", {}))
        if options:
            self.options.update(options)
        self.keep_alive = keep_alive or getattr(config, "LLAMA_KEEP_ALIVE", "2h")
        self.logger = logging.getLogger(self.__class__.__name__)
        # optional requests session
        self._session = None
        try:
            import requests  # type: ignore
            self._requests = requests
            self._session = requests.Session()
        except Exception:
            self._requests = None

    def generate(self, prompt: str) -> Optional[str]:
        url = f"{self.base_url}/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                if self._session is not None:
                    resp = self._session.post(url, json=payload, timeout=self.request_timeout)
                    if resp.status_code == 200:
                        return resp.json().get("response", "")
                    self.logger.error("LLaMA HTTP %s: %s", resp.status_code, resp.text[:200])
                else:
                    req = urllib.request.Request(
                        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=self.request_timeout) as r:  # nosec
                        data = json.load(r)
                        return data.get("response", "")
            except Exception as exc:
                if attempt >= self.max_retries:
                    self.logger.error("LLaMA request failed: %s", exc)
                    return None
                # capped exponential backoff
                time.sleep(min(5.0, self.retry_backoff ** (attempt - 1)))

    # Friendly wrapper for structured asks if you need it later
    def ask(self, system_prompt: str, context: Dict[str, Any]) -> Any:
        prompt = f"{system_prompt}\n\nContext: {json.dumps(context, ensure_ascii=False)}\n"
        raw = self.generate(prompt)
        try:
            return json.loads(raw or "[]")
        except Exception:
            return []


# ======================================================================
# Hybrid Strategy (single prevailing strategy)

class LlamaHybridStrategy(BaseStrategy):
    """Combines LLM sentiment and technical trend for robust decisions.

    Sentiment provides fast signals; trend validates direction & risk.
    Only this class should be used — sentiment/trend standalones were removed.
    """

    def __init__(
        self,
        exchange: ExchangeWrapper,
        llama_engine: Optional[LlamaDecisionEngine] = None,
        coins: Optional[List[str]] = None,
        base_currency: str = "USDT",
            # Sentiment
        sentiment_cache=None,
        sentiment_prompt: str = "",
        buy_threshold: float = 0.50,
        sell_threshold: float = -0.40,
            # Trend - Enhanced multi-timeframe
        ma_short: int = 20,
        ma_long: int = 50,
        rsi_period: int = 14,
        timeframe: str = "1h",
        timeframes: Optional[List[str]] = None,  # NEW: multiple timeframes
        history_length: int = 100,
            # Trading
        trade_fraction: float = 0.30,
        use_convert: bool = False,
        simulate_only: bool = True,
        convert_client: Optional[BinanceConvertClient] = None,
            # Hybrid
        sentiment_weight: float = 0.6,
        trend_weight: float = 0.4,
        require_trend_confirmation: bool = True,
            # Safety
        trailing_stop_pct: float = 0.06,
        min_sentiment_delta: float = 0.12,
        cooldown_seconds: int = 180,
        **kwargs,
    ) -> None:
        super().__init__(exchange, base_currency)
        # LLM engine (resolve from kwargs or config)
        self.llama_engine = llama_engine or LlamaDecisionEngine(
            model_name=getattr(config, "LLAMA_MODEL_NAME", "llama3.1:8b-instruct-q4_K_M"),
            base_url=getattr(config, "LLAMA_API_BASE_URL", "http://localhost:11434/api"),
        )
        # Params
        self.coins = [c.upper() for c in (coins or [])]
        self.base_currency = base_currency.upper()
        self.sentiment_prompt = sentiment_prompt
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.trade_fraction = trade_fraction
        self.use_convert = use_convert
        self.simulate_only = simulate_only
        self.convert_client = convert_client
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.timeframe = timeframe
        self.history_length = max(history_length, max(ma_short, ma_long, rsi_period) + 2)
        self.sentiment_weight = sentiment_weight
        self.trend_weight = trend_weight
        self.require_trend_confirmation = require_trend_confirmation
        self.trailing_stop_pct = trailing_stop_pct
        self.min_sentiment_delta = min_sentiment_delta
        self.cooloff_seconds = cooldown_seconds

        # State
        self.positions: Dict[str, Dict[str, float]] = {}
        self.last_trade_time: Dict[str, float] = {}
        self._prev_scores: Dict[str, float] = {}

        # Opportunistic scalper settings (from config with safe defaults)
        self.opp_enabled = bool(getattr(config, "OPP_ENABLED", True))
        self.opp_base_notional_usd = float(getattr(config, "OPP_BASE_NOTIONAL_USD", 15.0))
        self.opp_tp_pct = float(getattr(config, "OPP_TP_PCT", 0.0025))
        self.opp_sl_pct = float(getattr(config, "OPP_SL_PCT", 0.0030))
        self.opp_max_spread_pct = float(getattr(config, "OPP_MAX_SPREAD_PCT", 0.0015))
        self.opp_cooldown_seconds = int(getattr(config, "OPP_COOLDOWN_SECONDS", 60))
        self.opp_max_open_positions = int(getattr(config, "OPP_MAX_OPEN_POSITIONS", 2))
        self.opp_fee_pct = float(getattr(config, "OPP_FEE_PCT", 0.001))

        # Runtime state for quick scalps
        self.opportunistic_coins = getattr(self, "opportunistic_coins", [])  # UI appends here
        self._opp_positions: Dict[str, Dict[str, float]] = {}
        self._opp_cooldown: Dict[str, int] = {}

        # Analyzer
        self.trade_analyzer = TradeAnalyzer(history_dir="history")
        self._load_historical_insights()
        self.logger.info(
            "Hybrid ready: %d coins, weights S=%.0f%% T=%.0f%%",
            len(self.coins), 100 * self.sentiment_weight, 100 * self.trend_weight,
        )
        # Multi-timeframe analysis
        self.timeframes = timeframes or ["15m", "1h", "4h"]  # short, medium, long-term
        self.trend_cache: Dict[str, Tuple[float, float]] = {}

    # ---------- historical feedback ----------
    def _load_historical_insights(self) -> None:
        stats = self.trade_analyzer.analyze_history()
        try:
            if stats.get("total_trades", 0) > 10:
                wr = float(stats.get("win_rate", 0.0))
                if wr < 0.40:
                    self.buy_threshold = min(0.80, self.buy_threshold + 0.05)
                elif wr > 0.60:
                    self.buy_threshold = max(0.35, self.buy_threshold - 0.02)
        except Exception:
            pass

    def _multi_timeframe_trend(self, coin: str) -> Optional[float]:
        """Analyze trend across multiple timeframes for robust signals."""
        scores = []
        weights = [0.2, 0.5, 0.3]  # short, medium, long-term weights

        for i, tf in enumerate(self.timeframes):
            # Check cache first (60s TTL)
            cache_key = f"{coin}_{tf}"
            cached = self.trend_cache.get(cache_key)
            if cached and (time.time() - cached[1]) < 60:
                score = cached[0]
            else:
                # Calculate fresh score
                try:
                    score = self._trend_score_for_timeframe(coin, tf)
                    if score is not None:
                        self.trend_cache[cache_key] = (score, time.time())
                except Exception as e:
                    self.logger.debug(f"Trend calc failed for {coin} {tf}: {e}")
                    score = None

            if score is not None:
                scores.append((score, weights[i]))

        if not scores:
            # Fallback to single timeframe if multi-TF fails
            self.logger.debug(f"Multi-TF failed for {coin}, using single TF")
            return self._trend_score(coin)

        # Weighted average across timeframes
        weighted_sum = sum(s * w for s, w in scores)
        weight_total = sum(w for _, w in scores)

        return weighted_sum / weight_total if weight_total > 0 else None

        # Weighted average across timeframes
        weighted_sum = sum(s * w for s, w in scores)
        weight_total = sum(w for _, w in scores)

        return weighted_sum / weight_total if weight_total > 0 else None

    def _trend_score_for_timeframe(self, coin: str, timeframe: str) -> Optional[float]:
        """Calculate trend score for specific timeframe."""
        client = self.exchange.client
        if client is None:
            return None

        try:
            symbol = f"{coin}/{self.base_currency}"
            # Adjust history length based on timeframe
            limit = self._get_limit_for_timeframe(timeframe)
            ohlcv = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if not ohlcv or len(ohlcv) < 50:
                return None

            closes = [c[4] for c in ohlcv]
            # Use same calculation logic as _trend_score
            # ... (reuse your enhanced logic here)
            return self._calculate_trend_from_closes(closes)
        except Exception:
            return None

    def _calculate_trend_from_closes(self, closes: List[float]) -> Optional[float]:
        """Calculate trend score from closing prices - unified logic."""
        if len(closes) < max(self.ma_long, self.rsi_period) + 1:
            return None

        try:
            # Calculate indicators
            sma_s = sum(closes[-self.ma_short:]) / float(self.ma_short)
            sma_l = sum(closes[-self.ma_long:]) / float(self.ma_long)
            ema_s = self._calculate_ema(closes, self.ma_short)
            ema_l = self._calculate_ema(closes, self.ma_long)

            # RSI
            gains, losses = 0.0, 0.0
            for i in range(1, self.rsi_period + 1):
                diff = closes[-i] - closes[-i - 1]
                if diff >= 0:
                    gains += diff
                else:
                    losses += -diff
            rsi = 100.0 if losses == 0 else 100.0 - (100.0 / (1.0 + (gains / max(1e-9, losses))))

            # Trend strength
            trend_strength = self._calculate_trend_strength(closes)

            # Build score
            trend = 0.0
            if sma_s > sma_l:
                trend += 0.4
            elif sma_s < sma_l:
                trend -= 0.4

            if ema_s > ema_l:
                trend += 0.3 * trend_strength
            elif ema_s < ema_l:
                trend -= 0.3 * trend_strength

            if rsi >= 70:
                trend -= 0.2
            elif rsi >= 60:
                trend += 0.3
            elif rsi <= 30:
                trend += 0.2
            elif rsi <= 40:
                trend -= 0.3

            return max(-1.0, min(1.0, trend))
        except Exception as e:
            self.logger.error(f"Error calculating trend from closes: {e}")
            return None

    def _get_limit_for_timeframe(self, timeframe: str) -> int:
        """Adjust data points based on timeframe."""
        if "m" in timeframe:  # minutes
            return 200
        elif "h" in timeframe:  # hours
            return 150
        elif "d" in timeframe:  # days
            return 100
        return 100

    # ---------- utilities ----------
    def _get_current_price(self, coin: str) -> Optional[float]:
        """Get current price with caching to reduce API calls."""
        # Add simple 5-second cache
        if not hasattr(self, '_price_cache'):
            self._price_cache: Dict[str, float] = {}
            self._price_cache_time: Dict[str, float] = {}

        cache_key = coin
        now = time.time()

        # Return cached if fresh (within 5 seconds)
        if cache_key in self._price_cache:
            cached_time = self._price_cache_time.get(cache_key, 0)
            if now - cached_time < 5:
                return self._price_cache[cache_key]

        try:
            ticker = self.exchange.get_ticker(f"{coin}/{self.base_currency}")
            px = ticker.get("last") or ticker.get("close")
            price = float(px) if px else None

            if price:
                self._price_cache[cache_key] = price
                self._price_cache_time[cache_key] = now

            return price
        except Exception as e:
            self.logger.debug(f"Price fetch failed for {coin}: {e}")
            return None

    def _can_trade(self, coin: str) -> bool:
        last = self.last_trade_time.get(coin, 0.0)
        return (time.time() - last) >= self.cooloff_seconds

    def _update_peak(self, coin: str, price: float) -> None:
        pos = self.positions.setdefault(coin, {})
        peak = pos.get("peak_price")
        if peak is None or price > peak:
            pos["peak_price"] = price

    # ---------- trend ----------
    def _trend_score(self, coin: str) -> Optional[float]:
        client = self.exchange.client
        if client is None:
            return None
        try:
            symbol = f"{coin}/{self.base_currency}"
            ohlcv = client.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.history_length)
            if not ohlcv or len(ohlcv) < max(self.ma_long, self.rsi_period) + 1:
                return None
            closes = [c[4] for c in ohlcv]

            # Enhanced: multi-timeframe trend analysis
            sma_s = sum(closes[-self.ma_short:]) / float(self.ma_short)
            sma_l = sum(closes[-self.ma_long:]) / float(self.ma_long)

            # Add EMA for faster response to price changes
            ema_s = self._calculate_ema(closes, self.ma_short)
            ema_l = self._calculate_ema(closes, self.ma_long)

            # RSI (last rsi_period deltas)
            gains, losses = 0.0, 0.0
            for i in range(1, self.rsi_period + 1):
                diff = closes[-i] - closes[-i - 1]
                if diff >= 0:
                    gains += diff
                else:
                    losses += -diff
            rsi = 100.0 if losses == 0 else 100.0 - (100.0 / (1.0 + (gains / max(1e-9, losses))))

            # Trend strength using ADX-like calculation
            trend_strength = self._calculate_trend_strength(closes)

            # Volume-weighted trend confirmation (if volume available)
            volumes = [c[5] for c in ohlcv] if len(ohlcv[0]) > 5 else None
            vol_confirmation = self._volume_trend_confirmation(closes, volumes) if volumes else 1.0

            trend = 0.0
            # SMA crossover
            if sma_s > sma_l:
                trend += 0.4
            elif sma_s < sma_l:
                trend -= 0.4

            # EMA trend (faster signal)
            if ema_s > ema_l:
                trend += 0.3 * trend_strength
            elif ema_s < ema_l:
                trend -= 0.3 * trend_strength

            # RSI momentum with overbought/oversold zones
            if rsi >= 70:
                trend -= 0.2  # overbought - bearish
            elif rsi >= 60:
                trend += 0.3
            elif rsi <= 30:
                trend += 0.2  # oversold - bullish opportunity
            elif rsi <= 40:
                trend -= 0.3

            # Apply volume confirmation
            trend *= vol_confirmation

            # clip to [-1,1]
            return max(-1.0, min(1.0, trend))
        except Exception:
            return None

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average for faster trend detection."""
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2.0 / (period + 1)
        # Start with SMA
        ema = sum(prices[:period]) / period

        # Apply EMA formula to remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_trend_strength(self, closes: List[float]) -> float:
        """Calculate trend strength (0.0 to 1.0) using directional movement."""
        if len(closes) < 14:
            return 0.5

        ups = sum(1 for i in range(1, min(14, len(closes))) if closes[-i] > closes[-i - 1])
        strength = ups / 13.0  # normalize
        return min(1.0, max(0.0, strength))

    def _volume_trend_confirmation(self, closes: List[float], volumes: List[float]) -> float:
        """Use volume to confirm trend direction (1.0 = strong, 0.5 = weak)."""
        if len(closes) < 20 or len(volumes) < 20:
            return 1.0

        # Check if volume increases on up moves
        up_vol = sum(volumes[-i] for i in range(1, 11) if closes[-i] > closes[-i - 1])
        down_vol = sum(volumes[-i] for i in range(1, 11) if closes[-i] < closes[-i - 1])

        total_vol = up_vol + down_vol
        if total_vol == 0:
            return 1.0

        # Strong volume confirmation = 1.0, weak = 0.7
        vol_ratio = max(up_vol, down_vol) / total_vol
        return 0.7 + (0.3 * vol_ratio)

    # ---------- sentiment ----------
        # ---------- sentiment ----------
    def _sentiment_prompt(self, coins: List[str]) -> str:
        # Use project‑specific builder if available
        try:
            build = getattr(config, "build_sentiment_prompt")
            return build(coins, include_date=False)  # type: ignore
        except Exception:
            pass

        # 🔹 IMPROVED: Stricter fallback prompt with examples
        lines = "\n".join(coins)
        return f"""Rate each cryptocurrency's market sentiment from -1.0 (very bearish) to +1.0 (very bullish).

        OUTPUT FORMAT (one line per coin):
        BTC: 0.75
        ETH: 0.42
        SOL: -0.15
    
        RULES:
        - EXACTLY one line per coin
        - Format: "SYMBOL: score"
        - Score between -1.0 and +1.0
        - NO explanations or extra text
    
        COINS TO RATE:
        {lines}
    
        OUTPUT:"""

    def _get_sentiment(self, coins: List[str]) -> Dict[str, float]:
        if not coins:
            return {}

        # 🔹 FIX: Check if scheduler is active first
        if getattr(self, "sentiment_cache", None) is not None:
            # Scheduler is running - NEVER call LLaMA directly
            results = {}
            for c in coins:
                score = self.sentiment_cache.get(c)
                if score is not None:
                    results[c] = float(score)
                else:
                    # Cache miss - return neutral score, scheduler will update soon
                    results[c] = 0.0
            return results

        # No scheduler - fall back to direct calls (original logic)
        prioritized = self._prioritize_coins_for_analysis(coins)
        batch_size = self._adaptive_batch_size()

        results = {}
        for i in range(0, len(prioritized), batch_size):
            batch = prioritized[i:i + batch_size]
            prompt = self._sentiment_prompt(batch)
            trend_context = self._build_trend_context(batch)
            enhanced_prompt = f"{prompt}\n\nCurrent trend signals:\n{trend_context}"

            raw = self.llama_engine.generate(enhanced_prompt)
            scores = parse_symbol_scores(raw, expected_order=batch)
            results.update(scores)

        return {c: float(results.get(c, 0.0)) for c in coins}

    def _prioritize_coins_for_analysis(self, coins: List[str]) -> List[str]:
        """Prioritize coins with strong technical signals for AI analysis."""
        scored = []
        for c in coins:
            trend = self._trend_score(c) or 0.0
            price = self._get_current_price(c) or 0.0

            # Higher priority for coins with strong trends or large moves
            priority = abs(trend) + (0.1 if price else 0)
            scored.append((c, priority))

        # Sort by priority (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def _adaptive_batch_size(self) -> int:
        """Dynamically adjust batch size based on system performance."""
        # Start with config default
        batch_size = getattr(config, "SENTIMENT_BATCH_SIZE", 1)

        # If we have historical timing data, adjust
        if hasattr(self, '_avg_inference_time'):
            # If inference is fast (<2s), increase batch
            if self._avg_inference_time < 2.0:
                batch_size = min(3, batch_size + 1)
            # If slow (>5s), decrease batch
            elif self._avg_inference_time > 5.0:
                batch_size = max(1, batch_size - 1)

        return batch_size

    def _build_trend_context(self, coins: List[str]) -> str:
        """Provide trend context to AI for better reasoning."""
        context_lines = []
        for c in coins:
            trend = self._trend_score(c)
            if trend is not None:
                direction = "bullish" if trend > 0.3 else "bearish" if trend < -0.3 else "neutral"
                context_lines.append(f"{c}: {direction} ({trend:.2f})")
        return "\n".join(context_lines)

    # ---------- trade I/O ----------
    def _execute_buy(self, coin: str, notional_usd: float, price: float, out: List[TradeInstruction]) -> bool:
        try:
            market = f"{coin}/{self.base_currency}"
            # min notional heuristic
            min_notional = 10.0
            try:
                if self.exchange.client is not None:
                    info = self.exchange.client.market(market)  # type: ignore
                    min_notional = float(info.get("limits", {}).get("cost", {}).get("min", min_notional))
            except Exception:
                pass
            if notional_usd < min_notional:
                self.logger.info("BUY %s skipped: notional $%.2f < $%.2f", coin, notional_usd, min_notional)
                return False
            qty = notional_usd / price
            if self.use_convert and self.convert_client is not None:
                try:
                    q = self.convert_client.get_quote(self.base_currency, coin, from_amount=notional_usd)
                    qid = q.get("quoteId") if isinstance(q, dict) else None
                    if qid:
                        self.convert_client.accept_quote(qid)
                        return True
                except Exception as exc:
                    self.logger.error("Convert buy failed for %s: %s", coin, exc)
            out.append(TradeInstruction(symbol=market, side="buy", amount=qty, order_type="market"))
            return True
        except Exception as exc:
            self.logger.error("BUY failed for %s: %s", coin, exc)
            return False

    def _execute_sell(self, coin: str, amount: float, reason: str, out: List[TradeInstruction]) -> bool:
        try:
            market = f"{coin}/{self.base_currency}"
            price = self._get_current_price(coin)
            if price is None:
                return False
            min_notional = 10.0
            try:
                if self.exchange.client is not None:
                    info = self.exchange.client.market(market)  # type: ignore
                    min_notional = float(info.get("limits", {}).get("cost", {}).get("min", min_notional))
            except Exception:
                pass
            if amount * price < min_notional:
                self.logger.info("SELL %s skipped: notional $%.2f < $%.2f", coin, amount * price, min_notional)
                return False
            if self.use_convert and self.convert_client is not None:
                try:
                    q = self.convert_client.get_quote(coin, self.base_currency, from_amount=amount)
                    qid = q.get("quoteId") if isinstance(q, dict) else None
                    if qid:
                        self.convert_client.accept_quote(qid)
                        return True
                except Exception as exc:
                    self.logger.error("Convert sell failed for %s: %s", coin, exc)
            out.append(TradeInstruction(symbol=market, side="sell", amount=amount, order_type="market"))
            return True
        except Exception as exc:
            self.logger.error("SELL failed for %s: %s", coin, exc)
            return False

    # ---------- core decision ----------
    def decide_trades(self, balances: Dict[str, Dict[str, float]], **kwargs) -> List[TradeInstruction]:
        """Trading logic with portfolio allocation and risk management."""
        if not self.coins:
            return []

        coins = [c for c in self.coins if c.upper() != self.base_currency]
        if not coins:
            self.logger.info("No coins configured; skipping cycle")
            return []

        # 🔹 DEBUG: Check if cache is available
        has_cache = getattr(self, "sentiment_cache", None) is not None
        self.logger.debug(f"Sentiment cache available: {has_cache}")

        # Get trend scores using parent method
        trend_scores: Dict[str, Optional[float]] = {}
        for coin in coins:
            trend_scores[coin] = self._multi_timeframe_trend(coin)

        # Get sentiment from cache or direct call (parent handles this)
        sent: Dict[str, float] = {}
        need: List[str] = []

        if getattr(self, "sentiment_cache", None):
            ttl = getattr(config, "SENTIMENT_TTL_SECONDS", 900)
            cache_hits = 0
            cache_misses = 0

            for c in coins:
                s = self.sentiment_cache.get_fresh(c, ttl)
                if s is None:
                    cache_misses += 1
                    # Only query AI if trend is significant or cache is stale
                    if abs(trend_scores.get(c, 0.0)) > 0.2:
                        need.append(c)
                else:
                    cache_hits += 1
                    sent[c] = float(s)

            # 🔹 DEBUG: Log cache performance
            self.logger.info(
                f"Sentiment cache: {cache_hits} hits, {cache_misses} misses "
                f"(need refresh: {len(need)})"
            )
        else:
            # No cache - query all coins with significant trends OR fallback to all
            self.logger.warning("No sentiment cache - using direct AI calls")
            need = [c for c in coins if abs(trend_scores.get(c, 0.0)) > 0.2]
            if not need and not trend_scores:
                need = coins[:]

        if need:
            self.logger.info(f"Fetching sentiment for: {', '.join(need)}")
            fetched = self._get_sentiment(need)
            sent.update(fetched)

            # 🔹 DEBUG: Log fetched sentiments
            if fetched:
                sent_str = ", ".join(f"{k}={v:.2f}" for k, v in fetched.items())
                self.logger.info(f"Fetched sentiments: {sent_str}")

        # Compute hybrid scores using parent logic
        hybrid_scores: Dict[str, float] = {}
        for coin in coins:
            s = float(sent.get(coin, 0.0))
            t = float(trend_scores.get(coin, 0.0))

            # 🔹 DEBUG: Log individual components
            self.logger.debug(f"{coin}: sentiment={s:.2f}, trend={t:.2f}")

            # Use parent's weighting logic
            trend_strength = abs(t)
            sentiment_strength = abs(s)

            if trend_strength > 0.7:
                w_trend, w_sent = 0.6, 0.4
            elif sentiment_strength > 0.7:
                w_trend, w_sent = 0.4, 0.6
            else:
                w_trend = self.trend_weight
                w_sent = self.sentiment_weight

            sc = round(max(-1.0, min(1.0, w_sent * s + w_trend * t)), 3)

            if (s > 0 and t < 0) or (s < 0 and t > 0):
                confidence_penalty = 0.3 * min(abs(s), abs(t))
                sc *= (1.0 - confidence_penalty)

            hybrid_scores[coin] = sc

        # 🔹 CHANGE: Log hybrid scores (sentiment+trend), not just trend
        coin_summaries = []
        for coin in coins:
            if coin in hybrid_scores:
                s = sent.get(coin, 0.0)
                t = trend_scores.get(coin, 0.0)
                h = hybrid_scores[coin]
                # Format: COIN [S:+0.50 T:+0.30 H:+0.42]
                coin_summaries.append(f"{coin} [S:{s:+.2f} T:{t:+.2f} H:{h:+.2f}]")

        if coin_summaries:
            self.logger.info(" | ".join(coin_summaries))

        trades: List[TradeInstruction] = []
        base_balance = balances.get(self.base_currency, {}).get("free", 0.0)

        # SELL logic with risk management
        for coin in list(self.positions.keys()):
            held = balances.get(coin, {}).get("free", 0.0)
            if held <= 0:
                self._cleanup_position(coin)
                continue

            current_price = self._get_current_price(coin)
            if not current_price:
                continue

            # Check stop-loss
            if self._check_stop_loss(coin, current_price):
                self._execute_sell(coin, held, trades, "stop-loss")
                self._cleanup_position(coin)
                continue

            # Check time exit
            if self._check_time_exit(coin):
                self._execute_sell(coin, held, trades, "time-limit")
                self._cleanup_position(coin)
                continue

            # Regular sell signal
            hybrid = hybrid_scores.get(coin, 0.0)
            if hybrid <= self.sell_threshold and self._can_trade(coin):
                self._execute_sell(coin, held, trades, f"hybrid={hybrid:.2f}")
                self._cleanup_position(coin)

        # BUY logic with portfolio allocation
        if base_balance < 10:
            return trades

        # Separate core and opportunistic candidates
        from portfolio_strategy import DiversifiedPortfolioStrategy
        core_coins = DiversifiedPortfolioStrategy.CORE_COINS
        core_candidates = [
            (coin, score) for coin, score in hybrid_scores.items()
            if coin in core_coins and score >= self.buy_threshold and self._can_trade(coin)
        ]

        opportunistic_candidates = [
            (coin, score) for coin, score in hybrid_scores.items()
            if coin in self.opportunistic_coins and score >= self.buy_threshold and self._can_trade(coin)
        ]

        # Allocate funds
        CORE_ALLOCATION = DiversifiedPortfolioStrategy.CORE_ALLOCATION
        OPPORTUNISTIC_ALLOCATION = DiversifiedPortfolioStrategy.OPPORTUNISTIC_ALLOCATION
        core_funds = base_balance * CORE_ALLOCATION
        opportunistic_funds = base_balance * OPPORTUNISTIC_ALLOCATION

        # Buy best core coin
        if core_candidates:
            core_candidates.sort(key=lambda x: x[1], reverse=True)
            top_coin, top_score = core_candidates[0]

            if self.require_trend_confirmation:
                trend = trend_scores.get(top_coin)
                if trend is not None and trend < 0:
                    self.logger.info(f"Skipping {top_coin}: negative trend")
                else:
                    DiversifiedPortfolioStrategy._execute_buy_with_tracking(top_coin, core_funds, trades)
            else:
                DiversifiedPortfolioStrategy._execute_buy_with_tracking(top_coin, core_funds, trades)

        # Buy best opportunistic coin (smaller position)
        if opportunistic_candidates:
            opportunistic_candidates.sort(key=lambda x: x[1], reverse=True)
            top_coin, top_score = opportunistic_candidates[0]

            trend = trend_scores.get(top_coin)
            if trend is not None and trend < 0:
                self.logger.info(f"Skipping {top_coin}: negative trend (opportunistic)")
            else:
                DiversifiedPortfolioStrategy._execute_buy_with_tracking(top_coin, opportunistic_funds, trades)

        return trades

    def _now(self):
        import time
        return int(time.time())

    def _spread_pct(self, symbol: str) -> float:
        """Estimate spread from best bid/ask."""
        try:
            t = self.exchange.get_ticker(symbol)  # should include bid/ask on CCXT
            bid, ask = float(t.get("bid") or 0), float(t.get("ask") or 0)
            if bid > 0 and ask > 0 and ask >= bid:
                return (ask - bid) / ((ask + bid) / 2.0)
        except Exception:
            pass
        return 1.0  # very large to block entry if unknown

    def _tiny_momentum(self, symbol: str) -> float:
        """
        Very light momentum proxy (no heavy candles):
        uses last vs. a cached prev price (1 cycle ago).
        You likely already cache tickers; else we keep a mini cache here.
        """
        if not hasattr(self, "_opp_prev_px"):
            self._opp_prev_px: Dict[str, float] = {}
        try:
            t = self.exchange.get_ticker(symbol)
            last = float(t.get("last") or t.get("close") or 0.0)
            prev = float(self._opp_prev_px.get(symbol, last) or last)
            self._opp_prev_px[symbol] = last
            if prev > 0 and last > 0:
                return (last - prev) / prev
        except Exception:
            return 0.0
        return 0.0

    def run_opportunistic_cycle(self, base_currency: str = "USDT", simulate_only: bool = False):
        """
        Opens/maintains tiny scalps on opportunistic_coins:
        - enter if: spread small, tiny momentum up, no cooldown, free slot
        - exit if: TP or SL hit (fee-aware)
        """
        if not self.opp_enabled:
            return []

        actions = []
        now = self._now()

        # 1) Maintain existing small positions: check TP/SL and exit
        for coin, pos in list(self._opp_positions.items()):
            symbol = f"{coin}/{base_currency}"
            try:
                t = self.exchange.get_ticker(symbol)
                last = float(t.get("last") or t.get("close") or 0.0)
                if last <= 0:
                    continue

                entry = pos["entry_price"]
                amount = pos["amount"]

                gross_pnl_pct = (last - entry) / entry
                # net after both sides fees:
                net_pnl_pct = gross_pnl_pct - (2.0 * self.opp_fee_pct)

                if net_pnl_pct >= self.opp_tp_pct:
                    # take profit
                    if not simulate_only:
                        self.exchange.create_order(symbol, "sell", amount, "market")
                    self._opp_positions.pop(coin, None)
                    self._opp_cooldown[coin] = now + self.opp_cooldown_seconds
                    self.logger.info(f"OPP: TP exit {coin} @ {last:.8f} (+{net_pnl_pct * 100:.3f}%)")
                    actions.append({"coin": coin, "action": "opp_tp_sell", "price": last, "pnl_pct": net_pnl_pct})
                    continue

                if net_pnl_pct <= -abs(self.opp_sl_pct):
                    # stop loss
                    if not simulate_only:
                        self.exchange.create_order(symbol, "sell", amount, "market")
                    self._opp_positions.pop(coin, None)
                    self._opp_cooldown[coin] = now + self.opp_cooldown_seconds
                    self.logger.info(f"OPP: SL exit {coin} @ {last:.8f} ({net_pnl_pct * 100:.3f}%)")
                    actions.append({"coin": coin, "action": "opp_sl_sell", "price": last, "pnl_pct": net_pnl_pct})
            except Exception as e:
                self.logger.error(f"OPP: maintain error {coin}: {e}")

        # 2) Consider new entries if we have slots
        open_slots = self.opp_max_open_positions - len(self._opp_positions)
        if open_slots <= 0:
            return actions

        # Base balance check
        try:
            balances = self.exchange.get_balances()
            base_free = float(balances.get(base_currency, {}).get("free", 0.0) or 0.0)
        except Exception:
            base_free = 0.0

        min_trade = float(getattr(config, "MIN_TRADE_USD", 10.0))
        buy_usd = min(self.opp_base_notional_usd, base_free) if base_free > 0 else 0.0
        if buy_usd < max(min_trade, 1.0):
            return actions  # not enough base to attempt

        # Iterate opportunistic list
        for coin in list(self.opportunistic_coins or []):
            if open_slots <= 0:
                break
            if coin in self._opp_positions:
                continue
            ready_ts = int(self._opp_cooldown.get(coin, 0))
            if now < ready_ts:
                continue

            symbol = f"{coin}/{base_currency}"
            # 2a) spread guard
            spr = self._spread_pct(symbol)
            if spr > self.opp_max_spread_pct:
                continue

            # 2b) tiny momentum up
            mom = self._tiny_momentum(symbol)
            # We want a tiny positive drift above fees + a cushion:
            if mom < (self.opp_fee_pct * 1.1):
                continue

            # 2c) compute amount to buy
            try:
                t = self.exchange.get_ticker(symbol)
                last = float(t.get("last") or t.get("close") or 0.0)
                if last <= 0.0:
                    continue
                amount = buy_usd / last
                if amount <= 0.0:
                    continue

                if not simulate_only:
                    self.exchange.create_order(symbol, "buy", amount, "market")
                self._opp_positions[coin] = {"entry_price": last, "amount": amount, "ts": now}
                open_slots -= 1
                self.logger.info(
                    f"OPP: buy {coin} {amount:.8f} @ {last:.8f} (spr={spr * 100:.2f}%, mom={mom * 100:.3f}%)")
                actions.append({"coin": coin, "action": "opp_buy", "price": last, "amount": amount})
            except Exception as e:
                self.logger.error(f"OPP: entry error {coin}: {e}")

        return actions


# ======================================================================
# Live USD rates helpers

from typing import Dict, List, Optional

def fetch_live_usd_rates(
    exchange_client,
    coins: List[str],
    base_currency: str = "USDT",
    use_rest_fallback: bool = True,
) -> Dict[str, float]:
    """
    Return a dict {COIN: usd_price} using Binance (ccxt) with robust fallbacks.

    Logic:
      1) Stablecoins → 1.0
      2) Try direct USD-stable quotes in priority order (USDT, FDUSD, BUSD, USDC, TUSD, DAI, USDP)
      3) If no direct market, route via BTC or ETH:
           COIN/BTC * BTC/USDT   or   COIN/ETH * ETH/USDT (or other USD stables)
      4) Optional REST fallback to /api/v3/ticker/price when ccxt fails
    """
    try:
        import ccxt  # for exception types, optional
    except Exception:
        ccxt = None  # we'll just catch Exception

    stable = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI"}
    base = base_currency.upper()
    quotes_order = [base, "USDT", "FDUSD", "BUSD", "USDC", "TUSD", "DAI", "USDP"]

    # 0) Ensure markets are loaded (idempotent in ccxt)
    try:
        if not getattr(exchange_client, "markets", None):
            exchange_client.load_markets()
    except Exception:
        # If load fails, we still try fetch_ticker directly.
        pass

    markets = getattr(exchange_client, "markets", {}) or {}
    available = set(markets.keys())

    # Helper: safe single ticker fetch (prefer cached fetch_tickers later if you want)
    def _fetch_price(symbol: str) -> Optional[float]:
        try:
            t = exchange_client.fetch_ticker(symbol)
            p = t.get("last") or t.get("close")
            return float(p) if p is not None else None
        except Exception:
            return None

    # Helper: try COIN/QUOTE in quotes_order
    def _direct_usd_price(coin: str) -> Optional[float]:
        for q in quotes_order:
            sym = f"{coin}/{q}"
            if sym in available:
                p = _fetch_price(sym)
                if p and p > 0:
                    # if quote isn't strictly USDT, convert quote → USDT at 1.0 if it's a USD stable
                    # (all listed here should be ~1.0 already)
                    return float(p)
        return None

    # Helper: route via BTC or ETH
    def _routed_usd_price(coin: str) -> Optional[float]:
        for bridge in ("BTC", "ETH"):
            # coin/bridge
            sym_cb = f"{coin}/{bridge}"
            if sym_cb not in available:
                continue
            p_cb = _fetch_price(sym_cb)
            if not p_cb or p_cb <= 0:
                continue

            # bridge / USD-stable (any of quotes_order)
            for q in quotes_order:
                sym_bq = f"{bridge}/{q}"
                if sym_bq not in available:
                    continue
                p_bq = _fetch_price(sym_bq)
                if p_bq and p_bq > 0:
                    return float(p_cb) * float(p_bq)
        return None

    # Optional REST fallback (Binance public)
    def _rest_usdt(symbol_no_slash: str) -> Optional[float]:
        if not use_rest_fallback:
            return None
        try:
            import requests
            r = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol_no_slash},
                timeout=6,
            )
            if r.ok:
                data = r.json()
                price = data.get("price")
                return float(price) if price is not None else None
        except Exception:
            return None
        return None

    rates: Dict[str, float] = {}

    for c in coins:
        cu = (c or "").upper().strip()
        if not cu:
            continue

        # 1) stable or base → 1.0
        if cu in stable or cu == base:
            rates[cu] = 1.0
            continue

        # 2) direct USD-stable
        price = _direct_usd_price(cu)
        if price is not None:
            rates[cu] = float(price)
            continue

        # 3) routed via BTC/ETH
        price = _routed_usd_price(cu)
        if price is not None:
            rates[cu] = float(price)
            continue

        # 4) REST: try COINUSDT, then COINFDUSD, COINBUSD, ...
        if use_rest_fallback:
            for q in quotes_order:
                sym = f"{cu}{q}"  # REST uses no slash
                price = _rest_usdt(sym)
                if price is not None:
                    # If q is not USDT, it's still a USD stable → treat ~1:1
                    rates[cu] = float(price)
                    break

        # 5) default if still missing
        if cu not in rates:
            rates[cu] = 0.0

    return rates



def update_usd_rates_batch(exchange_client, base_currency: str = "USDT") -> Dict[str, float]:
    rates: Dict[str, float] = {}
    stable = {"USDT", "USDC", "BUSD", "TUSD", "USDP", "DAI"}
    try:
        all_tickers = exchange_client.fetch_tickers()
        for symbol, t in all_tickers.items():
            if symbol.endswith(f"/{base_currency}"):
                coin = symbol.split("/")[0]
                px = t.get("last") or t.get("close")
                if px:
                    rates[coin] = float(px)
        for s in stable:
            rates[s] = 1.0
        rates[base_currency] = 1.0
    except Exception:
        pass
    return rates


# ======================================================================
# Trading bot orchestrator

class TradingBot:
    """Periodically runs a strategy; places market/convert orders.

    simulate_only=True → returns instructions but doesn't place orders.
    """
    def __init__(
        self,
        exchange: ExchangeWrapper,
        strategy: BaseStrategy,
        base_currency: str = "USDT",
        simulate_only: bool = True,
    ) -> None:
        self.exchange = exchange
        self.strategy = strategy
        self.base_currency = base_currency
        self.simulate_only = simulate_only
        self.logger = logging.getLogger(self.__class__.__name__)


    def _place(self, t: TradeInstruction) -> None:
        try:
            self.exchange.create_order(t.symbol, t.side, t.amount, t.price, t.order_type)
        except Exception as exc:
            self.logger.error("Order failed %s %s: %s", t.side, t.symbol, exc)

    def run_cycle(self) -> List[TradeInstruction]:
        balances = self.exchange.get_balances()
        trades = self.strategy.decide_trades(balances)
        if not self.simulate_only and trades:
            for t in trades:
                self._place(t)
        self.strategy.on_cycle_end({"balances": balances, "trades": trades, "at": datetime.now(timezone.utc).isoformat()})
        return trades

    def run(self):
        """Main scheduler loop with optimized timing."""
        self.log.info(
            f"Sentiment scheduler started: {len(self.coins)} coins, "
            f"target QPS={self.target_qps:.2f}, batch={self.batch}, "
            f"TTL={self.ttl}s, strategy={self.rotate}"
        )

        consecutive_errors = 0
        max_errors = 5

        while not self.stop_event.is_set():
            try:
                # 1) Check for backpressure
                if self._check_backpressure():
                    # Slow down temporarily
                    self.log.info("Applying backpressure: sleeping 5s")
                    time.sleep(5.0)
                    continue

                # 2) Select coins to process
                targets = self._select_next_batch()

                # 3) If nothing needs refresh, idle briefly
                if not targets:
                    # 🔹 DEBUG: Show why nothing was selected
                    staleness = self.cache.get_staleness_map()
                    if staleness:
                        oldest_coin = max(staleness.items(), key=lambda x: x[1])
                        self.log.debug(
                            f"No coins need refresh yet. Oldest: {oldest_coin[0]} "
                            f"({oldest_coin[1]:.0f}s old, threshold: {self.ttl * 0.8:.0f}s)"
                        )
                    time.sleep(1.0)
                    continue

                # 4) Process batch
                start_time = time.time()
                prompt = self._build_prompt(targets)

                # 🔹 DEBUG: Log what we're asking
                self.log.debug(f"Querying AI for: {', '.join(targets)}")

                # Call LLaMA (blocks on GPU)
                raw_response = self.llama.generate(prompt)

                # 🔹 DEBUG: Log raw response
                if raw_response:
                    preview = raw_response[:200].replace('\n', ' ')
                    self.log.debug(f"AI response preview: {preview}...")

                inference_duration = time.time() - start_time
                self._recent_durations.append(inference_duration)
                self._request_times.append(time.time())

                # 5) Parse and cache results
                scores = self._parse_lines(raw_response)

                cached_count = 0
                for coin in targets:
                    if coin in scores:
                        self.cache.set(coin, scores[coin])
                        cached_count += 1

                if cached_count > 0:
                    coins_str = ", ".join(f"{c}:{scores.get(c, 0):.2f}" for c in targets if c in scores)
                    self.log.info(
                        f"✅ Updated {cached_count}/{len(targets)} coins in {inference_duration:.2f}s: {coins_str}"
                    )
                else:
                    # 🔹 DEBUG: Log parse failure
                    self.log.warning(
                        f"❌ Failed to parse any scores from {len(targets)} coins. "
                        f"Raw response length: {len(raw_response or '')}"
                    )

                # Reset error counter on success
                consecutive_errors = 0

                # 6) Adaptive sleep with jitter
                sleep_time = self._calculate_adaptive_sleep()
                sleep_time = self._add_jitter(sleep_time)

                time.sleep(sleep_time)

            except Exception as e:
                consecutive_errors += 1
                self.log.error(f"Scheduler error: {e}", exc_info=True)  # ← Full traceback

                # Exponential backoff on repeated errors
                backoff = min(30.0, 2.0 ** consecutive_errors)
                self.log.warning(f"Backing off {backoff:.1f}s after error #{consecutive_errors}")
                time.sleep(backoff)

                if consecutive_errors >= max_errors:
                    self.log.critical("Too many consecutive errors, stopping scheduler")
                    break

        self.log.info("Sentiment scheduler stopped")


__all__ = [
    "BinanceConvertClient",
    "ExchangeWrapper",
    "TradeInstruction",
    "BaseStrategy",
    "LlamaDecisionEngine",
    "LlamaHybridStrategy",
    "fetch_live_usd_rates",
    "update_usd_rates_batch",
    "TradingBot",
]