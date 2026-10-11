"""Wiring and lifecycle of the headless bot: exchange → LLM engine → sentiment scheduler →
strategy → TradingBot, with the trading loop on a background thread.

The same classes the desktop GUI uses do the work; this module only decides how they are put
together when nobody is watching a window:

- **Mode is fixed at construction** from the environment (see ``safety.live_gate``). Nothing
  here, and nothing the API can send, changes it afterwards.
- **Simulation fills against a paper wallet** (``paper.py``), so simulated buys arrive and the
  strategy's sells, stop-losses and time exits get exercised. With placeholder keys the
  exchange client carries no credentials at all and only public market data is used.
- **Sentiment comes from the rolling scheduler** while the loop runs. A cycle requested while the
  loop is stopped refreshes stale scores itself, within a time budget, when the model answers.
- **Cycles are serialised**: the loop and ``POST /cycle`` never run one at the same time.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config as default_config
from intelligence import ExchangeWrapper, LlamaDecisionEngine, LlamaHybridStrategy, TradingBot
from optimization_engine import RollingSentimentCache, SentimentRollingScheduler, parse_sentiment_lines

from botservice.journal import DecisionLog
from botservice.paper import PaperExchange, PaperWallet
from botservice.safety import Redactor, keys_configured, live_gate, sentence
from botservice.settings import MIN_INTERVAL_S, ServiceSettings, SettingsError

log = logging.getLogger("BotService")

STRATEGIES = ("hybrid", "portfolio")
UNIT = "lo-cryptobot.service"
LLM_PROBE_TTL_S = 15.0
STOP_JOIN_S = 10.0


class ServiceError(Exception):
    """An answer for the API: ``status`` is the HTTP code, ``str()`` the sentence."""

    status = 500

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status


class BadRequest(ServiceError):
    status = 400


class Forbidden(ServiceError):
    status = 403


class Conflict(ServiceError):
    status = 409


class Upstream(ServiceError):
    status = 502


def display_url(url: str) -> str:
    """The URL without user-info or query string, which is where credentials hide in URLs."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "(unparseable URL)"
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _now_ms() -> int:
    return int(time.time() * 1000)


class BotService:
    def __init__(
        self,
        settings: ServiceSettings,
        *,
        cfg: Any = None,
        env: Optional[Dict[str, str]] = None,
        market: Any = None,
        engine: Any = None,
        build_engine: bool = True,
    ) -> None:
        self.settings = settings
        self.cfg = cfg if cfg is not None else default_config
        env = dict(os.environ) if env is None else env
        c = self.cfg

        key = getattr(c, "API_KEY", "") or ""
        secret = getattr(c, "API_SECRET", "") or ""
        password = getattr(c, "API_PASSWORD", "") or ""
        self.keys_ok = keys_configured(key, secret)
        self.gate = live_gate(env, self.keys_ok)
        self.simulate_only = self.gate.simulate_only
        self.redactor = Redactor([key, secret, password, getattr(c, "LLAMA_API_KEY", ""), settings.token])

        self.exchange_id = getattr(c, "EXCHANGE_ID", "binance")
        self.base_currency = str(getattr(c, "BASE_CURRENCY", "USDT")).upper()
        if market is None:
            # Placeholder keys are not passed on at all: ccxt then makes only public calls, and a
            # private call fails with "requires apiKey" instead of an authentication error.
            market = ExchangeWrapper(
                exchange_id=self.exchange_id,
                api_key=key if self.keys_ok else "",
                api_secret=secret if self.keys_ok else "",
                password=password if self.keys_ok else "",
            )
        self.market = market
        self.account = market if self.keys_ok else None

        self.engine = engine
        if engine is None and build_engine and getattr(c, "LLAMA_ENABLED", True):
            try:
                self.engine = LlamaDecisionEngine(
                    model_name=c.LLAMA_MODEL_NAME,
                    base_url=c.LLAMA_API_BASE_URL,
                    system_prompt=getattr(c, "LLAMA_SYSTEM_PROMPT", "") or "",
                    request_timeout=getattr(c, "LLAMA_REQUEST_TIMEOUT", 45),
                    max_retries=getattr(c, "LLAMA_MAX_RETRIES", 3),
                    retry_backoff=getattr(c, "LLAMA_RETRY_BACKOFF", 1.6),
                    options=getattr(c, "LLAMA_OPTIONS", None),
                    api_style=getattr(c, "LLAMA_API_STYLE", "ollama"),
                    api_key=getattr(c, "LLAMA_API_KEY", ""),
                    disable_thinking=getattr(c, "LLAMA_DISABLE_THINKING", True),
                )
            except ValueError as exc:
                raise SettingsError(str(exc)) from None
        self.llm_url = display_url(getattr(self.engine, "base_url", None) or getattr(c, "LLAMA_API_BASE_URL", ""))

        state = settings.state_dir
        self.cache = RollingSentimentCache()
        self.wallet: Optional[PaperWallet] = None
        if self.simulate_only:
            self.wallet = PaperWallet(
                base_currency=self.base_currency,
                start_balance=settings.paper_usdt,
                fee_pct=float(getattr(c, "OPP_FEE_PCT", 0.001)),
                path=os.path.join(state, "paper.json") if state else None,
            )
        self.journal = DecisionLog(os.path.join(state, "decisions.jsonl") if state else None)

        interval = settings.interval_s
        if interval is None:
            interval = max(MIN_INTERVAL_S, float(getattr(c, "TRADE_INTERVAL", 10)))
        self.interval_s = float(interval)
        self.default_strategy = "portfolio" if getattr(c, "USE_PORTFOLIO_STRATEGY", False) else "hybrid"

        self._state_lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._bot: Optional[TradingBot] = None
        self._strategy_name = self.default_strategy
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_evt: Optional[threading.Event] = None
        self._scheduler: Optional[SentimentRollingScheduler] = None
        self.started_at: Optional[int] = None
        self.last_cycle_at: Optional[int] = None
        self.cycles = 0
        self.last_error: Optional[str] = None
        self._llm_probe: Optional[tuple] = None  # (checked at monotonic, reachable)

        if self.gate.reason:
            log.warning("%s", self.gate.reason)
        log.info(
            "CryptoBot service ready: mode=%s keys=%s strategy=%s llm=%s interval=%.0fs",
            self.gate.mode, "configured" if self.keys_ok else "placeholders (public market data only)",
            self.default_strategy, self.llm_url, self.interval_s,
        )

    # ------------------------------------------------------------------ building blocks
    def _make_bot(self, name: str) -> TradingBot:
        c = self.cfg
        exchange: Any = PaperExchange(self.market, self.wallet) if self.wallet is not None else self.market
        common: Dict[str, Any] = dict(
            exchange=exchange,
            llama_engine=self.engine,
            sentiment_cache=self.cache,
            coins=list(getattr(c, "SENTIMENT_COINS", [])),
            sentiment_prompt=getattr(c, "LLAMA_SENTIMENT_PROMPT", ""),
            base_currency=self.base_currency,
            buy_threshold=getattr(c, "SENTIMENT_BUY_THRESHOLD", 0.45),
            sell_threshold=getattr(c, "SENTIMENT_SELL_THRESHOLD", -0.40),
            trade_fraction=getattr(c, "SENTIMENT_TRADE_FRACTION", 0.5),
            # Convert executes immediately, outside TradingBot and its decision record; the
            # service only ever trades through spot market orders.
            use_convert=False,
            simulate_only=self.simulate_only,
            sentiment_weight=getattr(c, "SENTIMENT_WEIGHT", 0.6),
            trend_weight=getattr(c, "TREND_WEIGHT", 0.4),
            require_trend_confirmation=getattr(c, "REQUIRE_TREND_CONFIRMATION", True),
            min_sentiment_delta=getattr(c, "MIN_SENTIMENT_DELTA", 0.12),
            cooldown_seconds=getattr(c, "COOLDOWN_SECONDS", 180),
            timeframes=list(getattr(c, "TIMEFRAMES", ["15m", "1h", "4h"])),
        )
        if name == "portfolio":
            from portfolio_strategy import DiversifiedPortfolioStrategy

            strategy: LlamaHybridStrategy = DiversifiedPortfolioStrategy(
                **common,
                stop_loss_pct=getattr(c, "STOP_LOSS_PCT", 0.15),
                position_max_days=getattr(c, "POSITION_MAX_DAYS", 10),
                enable_stop_loss=getattr(c, "ENABLE_STOP_LOSS", True),
                enable_time_exit=getattr(c, "ENABLE_TIME_EXIT", True),
            )
        else:
            strategy = LlamaHybridStrategy(**common)
        if self.wallet is not None:
            self._seed_positions(strategy)
        return TradingBot(exchange=exchange, strategy=strategy, base_currency=self.base_currency,
                          simulate_only=self.simulate_only)

    def _seed_positions(self, strategy: LlamaHybridStrategy) -> None:
        """Tell a fresh strategy about what the paper wallet already holds.

        Positions live in the strategy object, the holdings in the wallet; after a restart or a
        strategy switch the strategy would otherwise never sell (or stop out) coins it did not
        buy itself.
        """
        assert self.wallet is not None
        for coin, (amount, cost, opened_ms) in self.wallet.holdings().items():
            if coin not in strategy.coins:
                continue
            price = cost or self.wallet.marks.get(coin)
            if not price:
                continue
            opened = datetime.fromtimestamp(opened_ms / 1000, timezone.utc) if opened_ms else datetime.now(timezone.utc)
            strategy.entry_prices[coin] = price
            strategy.entry_times[coin] = opened
            strategy.positions[coin] = {"entry_price": price, "amount": amount, "entry_time": opened.isoformat()}

    def _current_bot(self) -> TradingBot:
        with self._state_lock:
            if self._bot is None:
                self._bot = self._make_bot(self._strategy_name)
            return self._bot

    def _ensure_connected(self) -> None:
        with self._connect_lock:
            if getattr(self.market, "client", None) is not None:
                return
            try:
                self.market.connect()
            except Exception as exc:
                # connect() assigns the client before load_markets() runs; drop a half-made one
                # so the next attempt starts over instead of trading on unloaded markets.
                try:
                    self.market.client = None
                except Exception:
                    pass
                raise Upstream(sentence(
                    f"Could not connect to {self.exchange_id} for market data ({type(exc).__name__}: {exc}); "
                    f"the next cycle tries again.", self.redactor)) from None

    def _scheduler_alive(self) -> bool:
        sched = self._scheduler
        return sched is not None and sched.is_alive()

    def _start_scheduler(self, stop_evt: threading.Event, coins: List[str]) -> None:
        c = self.cfg
        self._scheduler = None
        if self.engine is None or not getattr(c, "SENTIMENT_ROLLING_ENABLED", True):
            return
        sched = SentimentRollingScheduler(
            llama_engine=self.engine,
            coins=coins,
            cache=self.cache,
            ttl_seconds=int(getattr(c, "SENTIMENT_TTL_SECONDS", 240)),
            # Zero would divide by zero in the scheduler's back-pressure check.
            max_qps=max(0.001, float(getattr(c, "SENTIMENT_MAX_QPS", 0.25))),
            batch_size=int(getattr(c, "SENTIMENT_BATCH_SIZE", 1)),
            rotate_order=getattr(c, "SENTIMENT_ROTATE_ORDER", "staleness"),
            stop_event=stop_evt,
            build_prompt_fn=getattr(c, "build_sentiment_prompt", None),
            enable_jitter=getattr(c, "SENTIMENT_ENABLE_JITTER", True),
            adaptive_rate=getattr(c, "SENTIMENT_ADAPTIVE_RATE", True),
        )
        sched.name = "cryptobot-sentiment"
        sched.start()
        self._scheduler = sched

    def _refresh_sentiment_now(self, coins: List[str]) -> None:
        """Score stale coins synchronously, for a cycle run while the scheduler is not running."""
        budget = self.settings.cycle_sentiment_budget_s
        if self.engine is None or budget <= 0:
            return
        ttl = int(getattr(self.cfg, "SENTIMENT_TTL_SECONDS", 240))
        stale = [c for c in coins if not (self.cache.has_real_score(c) and self.cache.get_fresh(c, ttl) is not None)]
        if not stale:
            return
        if not self.llm_reachable(force=True):
            log.info("Sentiment model at %s is not reachable; this cycle runs on trend only", self.llm_url)
            return
        build = getattr(self.cfg, "build_sentiment_prompt", None)
        batch = max(1, int(getattr(self.cfg, "SENTIMENT_BATCH_SIZE", 1)))
        deadline = time.monotonic() + budget
        for i in range(0, len(stale), batch):
            if time.monotonic() >= deadline:
                log.info("Sentiment refresh stopped after %.0fs; %d coin(s) keep their old score",
                         budget, len(stale) - i)
                break
            chunk = stale[i:i + batch]
            prompt = build(chunk) if callable(build) else "\n".join(f"{c}:" for c in chunk)
            raw = self.engine.generate(prompt)
            if raw is None:
                break  # the model went away mid-refresh; do not spend the budget on retries
            scores = parse_sentiment_lines(raw)
            for coin in chunk:
                if coin in scores:
                    self.cache.set(coin, scores[coin])

    def llm_reachable(self, force: bool = False) -> bool:
        if self.engine is None or not hasattr(self.engine, "ping"):
            return False
        probe = self._llm_probe
        now = time.monotonic()
        if not force and probe is not None and now - probe[0] < LLM_PROBE_TTL_S:
            return probe[1]
        try:
            ok = bool(self.engine.ping())
        except Exception:
            ok = False
        self._llm_probe = (now, ok)
        return ok

    # ------------------------------------------------------------------ cycles
    def run_cycle(self, wait_s: float = 60.0) -> List[Dict[str, Any]]:
        """Run one decision cycle and return its decisions (after paper fills, in simulation)."""
        if not self._cycle_lock.acquire(timeout=max(0.0, wait_s)):
            raise Conflict(f"A decision cycle is already running and did not finish within {wait_s:.0f} s; try again shortly.")
        try:
            return self._run_cycle_locked()
        finally:
            self._cycle_lock.release()

    def _run_cycle_locked(self) -> List[Dict[str, Any]]:
        try:
            bot = self._current_bot()
            self._ensure_connected()
            if not self._scheduler_alive():
                self._refresh_sentiment_now(list(bot.strategy.coins))
            bot.run_cycle()
            decisions = [dict(d) for d in bot.last_cycle_decisions]
            if self.wallet is not None:
                decisions = [self._paper_fill(d) for d in decisions]
                self._mark_holdings(bot.strategy)
            self.journal.extend(decisions)
            with self._state_lock:
                self.cycles += 1
                self.last_cycle_at = _now_ms()
                self.last_error = None
            if decisions:
                log.info("Cycle decisions: %s", "; ".join(
                    f"{d['side']} {d['amount']:.8g} {d['coin']} ({'sim' if d['simulated'] else 'LIVE'})" for d in decisions))
            return decisions
        except ServiceError as exc:
            with self._state_lock:
                self.last_error = str(exc)
            raise
        except Exception as exc:
            message = sentence(f"The decision cycle failed: {type(exc).__name__}: {exc}", self.redactor)
            log.error("%s", message, exc_info=True)
            with self._state_lock:
                self.last_error = message
            raise ServiceError(message) from None

    def _paper_fill(self, d: Dict[str, Any]) -> Dict[str, Any]:
        assert self.wallet is not None
        filled, note = self.wallet.fill(d["side"], d["coin"], float(d["amount"]), d.get("price"))
        out = dict(d)
        if filled > 0:
            out["amount"] = filled
        out["reason"] = f"{d['reason']}; {note}" if d.get("reason") else note
        return out

    def _mark_holdings(self, strategy: LlamaHybridStrategy) -> None:
        assert self.wallet is not None
        for coin in self.wallet.holdings():
            try:
                self.wallet.mark(coin, strategy._get_current_price(coin))
            except Exception:
                continue
        self.wallet.save()

    def _loop(self, stop_evt: threading.Event) -> None:
        log.info("Trading loop running every %.0fs", self.interval_s)
        while not stop_evt.is_set():
            try:
                self.run_cycle(wait_s=self.interval_s)
            except Conflict:
                pass  # a POST /cycle holds the lock; its result is this tick's result
            except ServiceError as exc:
                log.warning("%s", exc)
            except Exception:  # pragma: no cover - run_cycle wraps everything
                log.exception("Trading loop error")
            if stop_evt.wait(self.interval_s):
                break
        log.info("Trading loop stopped")

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        thread, evt = self._loop_thread, self._stop_evt
        return thread is not None and thread.is_alive() and evt is not None and not evt.is_set()

    def _validate_strategy(self, strategy: Any) -> str:
        if strategy is None or strategy == "":
            return self._strategy_name if self._bot is not None else self.default_strategy
        if not isinstance(strategy, str) or strategy not in STRATEGIES:
            raise BadRequest(f"strategy must be \"hybrid\" or \"portfolio\", not {str(strategy)[:40]!r}.")
        return strategy

    def start(self, strategy: Any = None) -> Dict[str, Any]:
        name = self._validate_strategy(strategy)
        with self._state_lock:
            thread = self._loop_thread
            if thread is not None and thread.is_alive():
                if self._stop_evt is not None and self._stop_evt.is_set():
                    raise Conflict("The trading loop is still finishing its last cycle after a stop; try again in a few seconds.")
                raise Conflict(f"The trading loop is already running ({self._strategy_name}); stop it first to switch strategy.")
            if self._bot is None or self._strategy_name != name:
                self._bot = self._make_bot(name)
                self._strategy_name = name
            stop_evt = threading.Event()
            self._stop_evt = stop_evt
            self._start_scheduler(stop_evt, list(self._bot.strategy.coins))
            self._loop_thread = threading.Thread(target=self._loop, args=(stop_evt,), name="cryptobot-loop", daemon=True)
            self.started_at = _now_ms()
            self._loop_thread.start()
        log.info("Trading loop started: strategy=%s mode=%s", name, self.gate.mode)
        return self.status()

    def stop(self, join_s: float = STOP_JOIN_S) -> Dict[str, Any]:
        with self._state_lock:
            thread, evt, sched = self._loop_thread, self._stop_evt, self._scheduler
            if thread is None or not thread.is_alive() or evt is None or evt.is_set():
                return self.status()
            evt.set()  # stops the scheduler too: they share the event
            self.started_at = None
        thread.join(timeout=join_s)
        if sched is not None:
            sched.join(timeout=1.0)
        if thread.is_alive():
            log.info("Stop requested; the loop exits when its current cycle ends")
        return self.status()

    def shutdown(self) -> None:
        self.stop(join_s=5.0)
        if self.wallet is not None:
            self.wallet.save()

    # ------------------------------------------------------------------ views (API bodies)
    def status(self) -> Dict[str, Any]:
        bot = self._bot
        running = self.running
        coins = list(bot.strategy.coins) if bot is not None else list(getattr(self.cfg, "SENTIMENT_COINS", []))
        return {
            "ok": True,
            "running": running,
            "mode": self.gate.mode,
            "keys_configured": self.keys_ok,
            "llm": {"url": self.llm_url, "reachable": self.llm_reachable()},
            "strategy": self._strategy_name,
            "started_at": self.started_at if running else None,
            "last_cycle_at": self.last_cycle_at,
            "cycles": self.cycles,
            "coins": coins,
            "last_error": self.last_error,
        }

    def signals(self) -> Dict[str, Any]:
        bot = self._bot
        if bot is None:
            return {"signals": []}
        latest = dict(getattr(bot.strategy, "last_signals", {}) or {})
        order = [c for c in bot.strategy.coins if c in latest] + sorted(c for c in latest if c not in bot.strategy.coins)
        return {"signals": [dict(latest[c]) for c in order]}

    def decisions(self, limit: int = 20) -> Dict[str, Any]:
        return {"decisions": self.journal.latest(limit)}

    def balances(self) -> Dict[str, Any]:
        if not self.keys_ok or self.account is None:
            raise Conflict(
                f"Binance keys are still placeholders, so balances cannot be read: put real BINANCE_API_KEY and "
                f"BINANCE_API_SECRET values in {self.settings.env_file} and restart {UNIT}."
            )
        try:
            self._ensure_connected()
            raw = self.account.get_balances()
        except ServiceError:
            raise
        except Exception as exc:
            raise Upstream(sentence(f"Binance did not return balances: {type(exc).__name__}: {exc}", self.redactor)) from None
        out: Dict[str, Dict[str, float]] = {}
        for asset, row in (raw or {}).items():
            if isinstance(row, dict):
                out[str(asset).upper()] = {
                    "free": float(row.get("free") or 0.0),
                    "used": float(row.get("used") or 0.0),
                    "total": float(row.get("total") or 0.0),
                }
        return {"balances": out}

    def paper(self) -> Dict[str, Any]:
        if self.wallet is None:
            raise Conflict("The bot is trading live, so there is no paper wallet; GET /balances shows the real account.")
        return {"paper": self.wallet.snapshot()}

    def config_view(self) -> Dict[str, Any]:
        """Settings worth showing. Built field by field from an allow-list: keys never appear."""
        from portfolio_strategy import DiversifiedPortfolioStrategy

        c = self.cfg
        bot = self._bot
        strategy = bot.strategy if bot is not None else None

        def live(attr: str, cfg_name: str, default: Any) -> Any:
            # The GUI's settings dialog can retune a running strategy; show what is in effect.
            if strategy is not None and hasattr(strategy, attr):
                return getattr(strategy, attr)
            return getattr(c, cfg_name, default)

        return {
            "exchange": self.exchange_id,
            "base_currency": self.base_currency,
            "coins": list(strategy.coins) if strategy is not None else list(getattr(c, "SENTIMENT_COINS", [])),
            "strategy": self._strategy_name,
            "strategies": list(STRATEGIES),
            "simulate_only": self.simulate_only,
            "mode": self.gate.mode,
            "live_requested": self.gate.live_requested,
            "live_allowed": self.gate.live_allowed,
            "live_blocked_reason": self.gate.reason,
            "keys_configured": self.keys_ok,
            "interval": self.interval_s,
            "thresholds": {
                "buy": live("buy_threshold", "SENTIMENT_BUY_THRESHOLD", 0.45),
                "sell": live("sell_threshold", "SENTIMENT_SELL_THRESHOLD", -0.40),
            },
            "weights": {
                "sentiment": live("sentiment_weight", "SENTIMENT_WEIGHT", 0.6),
                "trend": live("trend_weight", "TREND_WEIGHT", 0.4),
            },
            "require_trend_confirmation": live("require_trend_confirmation", "REQUIRE_TREND_CONFIRMATION", True),
            "trade_fraction": live("trade_fraction", "SENTIMENT_TRADE_FRACTION", 0.5),
            "min_trade_usd": getattr(c, "MIN_TRADE_USD", 5.0),
            "cooldown_s": live("cooloff_seconds", "COOLDOWN_SECONDS", 180),
            "timeframes": list(getattr(c, "TIMEFRAMES", [])),
            "risk": {
                "stop_loss_pct": getattr(c, "STOP_LOSS_PCT", 0.15),
                "position_max_days": getattr(c, "POSITION_MAX_DAYS", 10),
                "enable_stop_loss": getattr(c, "ENABLE_STOP_LOSS", True),
                "enable_time_exit": getattr(c, "ENABLE_TIME_EXIT", True),
                "portfolio_core_coins": list(DiversifiedPortfolioStrategy.CORE_COINS),
                "portfolio_core_allocation": DiversifiedPortfolioStrategy.CORE_ALLOCATION,
            },
            "sentiment": {
                "rolling_enabled": bool(getattr(c, "SENTIMENT_ROLLING_ENABLED", True)),
                "ttl_s": getattr(c, "SENTIMENT_TTL_SECONDS", 240),
                "max_qps": getattr(c, "SENTIMENT_MAX_QPS", 0.25),
                "batch_size": getattr(c, "SENTIMENT_BATCH_SIZE", 1),
                "cycle_refresh_budget_s": self.settings.cycle_sentiment_budget_s,
            },
            "llm": {
                "style": getattr(self.engine, "api_style", getattr(c, "LLAMA_API_STYLE", "ollama")),
                "url": self.llm_url,
                "model": getattr(c, "LLAMA_MODEL_NAME", ""),
                "max_tokens": (getattr(c, "LLAMA_OPTIONS", {}) or {}).get("num_predict"),
            },
            "paper": (
                {"start_balance": self.wallet.start_balance, "fee_pct": self.wallet.fee_pct}
                if self.wallet is not None else None
            ),
        }
