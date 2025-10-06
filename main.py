from __future__ import annotations

import os, sys, time, logging, queue, importlib.util
import threading
from typing import Optional, Any

import config
from config import *
from intelligence import *  # ExchangeWrapper, TradingBot, LlamaDecisionEngine, LlamaHybridStrategy
from logs.logging_logic.log_compressor import setup_logging
from optimization_engine import RollingSentimentCache, SentimentRollingScheduler


# ── colors + console helper ─────────────────────────────────────────
class Colors:
    WHITE = '\033[97m'; RED = '\033[91m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; BLUE = '\033[94m'; MAGENTA = '\033[95m';  CYAN = '\033[96m'; RESET = '\033[0m';  BOLD = '\033[1m'


def print_trade_action(message: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S", time.localtime())
    mapping = {
        "BUY": (Colors.GREEN, "🟢 BUY"),
        "SELL": (Colors.RED, "🔴 SELL"),
        "SENTIMENT": (Colors.CYAN, "📊 SENTIMENT"),
        "ERROR": (Colors.RED, "❌ ERROR"),
        "WARNING": (Colors.YELLOW, "⚠️ WARNING"),
        "CRASH": (Colors.MAGENTA, "🚨 CRASH ALERT"),
        "INFO": (Colors.WHITE, "ℹ️"),
    }
    color, prefix = mapping.get(level, mapping["INFO"])
    print(f"{color}{Colors.BOLD}[{ts}] {prefix}: {message}{Colors.RESET}")


# ── robust GUI import (works with Shift+F10) ────────────────────────
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

try:
    from dashboard_enhanced import TradingApp, QtQueueHandler, run_dashboard  # type: ignore
except Exception as e:
    dash_path = os.path.join(CUR_DIR, "dashboard_enhanced.py")
    try:
        spec = importlib.util.spec_from_file_location("dashboard_enhanced", dash_path)
        dash = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(dash)  # type: ignore
        TradingApp = getattr(dash, "TradingApp")
        QtQueueHandler = getattr(dash, "QtQueueHandler")
        run_dashboard = getattr(dash, "run_dashboard")
    except Exception as imp_err:
        print_trade_action(f"CRITICAL: GUI import failed: {imp_err}", "ERROR")
        sys.exit(1)


# ── main flow ───────────────────────────────────────────────────────
def main() -> None:
    print_trade_action("🚀 CryptoBot2 Starting…")

    # 1) queue for GUI logs
    log_q: "queue.Queue[str]" = queue.Queue()

    # 2) logging (named args!)
    setup_logging(
        level=LOG_LEVEL,
        log_dir="logs",
        filename="bot.log",
        rotate_when="midnight",  # or: max_bytes=5_000_000
        backup_count=7,
        qt_queue_handler_cls=QtQueueHandler,
        qt_log_queue=log_q,
    )

    # 3) exchange
    try:
        exchange = ExchangeWrapper(
            exchange_id=EXCHANGE_ID,
            api_key=API_KEY,
            api_secret=API_SECRET,
            password=API_PASSWORD,
        )
        exchange.connect()  # ← IMPORTANT: Actually connect!
        logging.getLogger("ExchangeWrapper").info("Connected to %s", EXCHANGE_ID)
        print_trade_action(f"✅ Connected to {EXCHANGE_ID}")
    except Exception as e:
        print_trade_action(f"Exchange initialization failed: {e}", "ERROR")
        return

    # 4) LLaMA engine
    llama_engine: Optional[LlamaDecisionEngine] = None
    if LLAMA_ENABLED:
        try:
            llama_engine = LlamaDecisionEngine(
                model_name=LLAMA_MODEL_NAME,
                base_url=LLAMA_API_BASE_URL,
                system_prompt=LLAMA_SYSTEM_PROMPT or "",
                request_timeout=LLAMA_REQUEST_TIMEOUT,
                max_retries=LLAMA_MAX_RETRIES,
                retry_backoff=LLAMA_RETRY_BACKOFF,
                options=LLAMA_OPTIONS,
            )
            print_trade_action(f"✅ LLaMA engine initialized: {LLAMA_MODEL_NAME}")
        except Exception as e:
            print_trade_action(f"LLaMA initialization failed: {e}", "ERROR")
            llama_engine = None  # Continue without AI if it fails

        # 5) Sentiment optimization (cache + scheduler)
        sent_cache = RollingSentimentCache()
        scheduler: Optional[SentimentRollingScheduler] = None
        stop_evt = threading.Event()

        if SENTIMENT_ROLLING_ENABLED and llama_engine:
            try:
                # Build prompt function from config if available
                build_prompt_fn = getattr(config, "build_sentiment_prompt", None)

                # 🔹 FIX: Reduce TTL for faster initial updates (testing)
                ttl_seconds = getattr(config, "SENTIMENT_TTL_SECONDS", 180)  # Use 3 min instead of 15

                scheduler = SentimentRollingScheduler(
                    llama_engine=llama_engine,
                    coins=SENTIMENT_COINS,
                    cache=sent_cache,
                    ttl_seconds=ttl_seconds,  # ← Use reduced TTL
                    max_qps=SENTIMENT_MAX_QPS,
                    batch_size=SENTIMENT_BATCH_SIZE,
                    rotate_order=SENTIMENT_ROTATE_ORDER,
                    stop_event=stop_evt,
                    build_prompt_fn=build_prompt_fn,
                    enable_jitter=getattr(config, "SENTIMENT_ENABLE_JITTER", True),
                    adaptive_rate=getattr(config, "SENTIMENT_ADAPTIVE_RATE", True),
                )
                scheduler.start()

                # 🔹 FIX: Wait a moment for first query to complete
                print_trade_action("Waiting for initial sentiment data...", "INFO")
                time.sleep(5)  # Give scheduler time to make first request

                # 🔹 DEBUG: Check if any scores were cached
                snapshot = sent_cache.snapshot()
                if snapshot:
                    sample = list(snapshot.items())[:3]
                    print_trade_action(
                        f"Initial sentiment: {', '.join(f'{k}={v:.2f}' for k, v in sample)}",
                        "SENTIMENT"
                    )
                else:
                    print_trade_action(
                        "⚠️ No sentiment scores cached yet (scheduler may be warming up)",
                        "WARNING"
                    )

                print_trade_action(
                    f"✅ Sentiment scheduler started: {len(SENTIMENT_COINS)} coins, "
                    f"QPS={SENTIMENT_MAX_QPS:.2f}, TTL={ttl_seconds}s",
                    "SENTIMENT"
                )
            except Exception as e:
                print_trade_action(f"Scheduler initialization failed: {e}", "WARNING")
                logging.exception("Scheduler error")
                scheduler = None

    # 6) Strategy selection (basic vs portfolio)
    use_portfolio_strategy = getattr(config, "USE_PORTFOLIO_STRATEGY", False)

    try:
        if use_portfolio_strategy:
            # Import portfolio strategy
            from portfolio_strategy import DiversifiedPortfolioStrategy

            strategy = DiversifiedPortfolioStrategy(
                exchange=exchange,
                llama_engine=llama_engine,
                sentiment_cache=sent_cache,  # ← Links to scheduler
                coins=SENTIMENT_COINS,
                sentiment_prompt=LLAMA_SENTIMENT_PROMPT,
                base_currency=BASE_CURRENCY,
                buy_threshold=SENTIMENT_BUY_THRESHOLD,
                sell_threshold=SENTIMENT_SELL_THRESHOLD,
                trade_fraction=SENTIMENT_TRADE_FRACTION,
                use_convert=LLAMA_SENTIMENT_USE_CONVERT,
                simulate_only=LLAMA_SENTIMENT_SIMULATE_ONLY,
                sentiment_weight=getattr(config, "SENTIMENT_WEIGHT", 0.6),
                trend_weight=getattr(config, "TREND_WEIGHT", 0.4),
                require_trend_confirmation=getattr(config, "REQUIRE_TREND_CONFIRMATION", True),
                min_sentiment_delta=getattr(config, "MIN_SENTIMENT_DELTA", 0.12),
                # Portfolio-specific params
                stop_loss_pct=getattr(config, "STOP_LOSS_PCT", 0.20),
                position_max_days=getattr(config, "POSITION_MAX_DAYS", 14),
                enable_stop_loss=getattr(config, "ENABLE_STOP_LOSS", True),
                enable_time_exit=getattr(config, "ENABLE_TIME_EXIT", True),
            )
            print_trade_action("✅ Portfolio Strategy initialized (with risk management)")
        else:
            # Basic hybrid strategy
            strategy = LlamaHybridStrategy(
                exchange=exchange,
                llama_engine=llama_engine,
                sentiment_cache=sent_cache,  # ← Links to scheduler
                coins=SENTIMENT_COINS,
                sentiment_prompt=LLAMA_SENTIMENT_PROMPT,
                base_currency=BASE_CURRENCY,
                buy_threshold=SENTIMENT_BUY_THRESHOLD,
                sell_threshold=SENTIMENT_SELL_THRESHOLD,
                trade_fraction=SENTIMENT_TRADE_FRACTION,
                use_convert=LLAMA_SENTIMENT_USE_CONVERT,
                simulate_only=LLAMA_SENTIMENT_SIMULATE_ONLY,
                sentiment_weight=getattr(config, "SENTIMENT_WEIGHT", 0.6),
                trend_weight=getattr(config, "TREND_WEIGHT", 0.4),
                require_trend_confirmation=getattr(config, "REQUIRE_TREND_CONFIRMATION", True),
                min_sentiment_delta=getattr(config, "MIN_SENTIMENT_DELTA", 0.12),
            )
            print_trade_action("✅ Hybrid Strategy initialized")

        logging.getLogger("Strategy").info(
            "Strategy ready: %d coins, weights S=%d%% T=%d%%",
            len(SENTIMENT_COINS),
            int(getattr(config, "SENTIMENT_WEIGHT", 0.6) * 100),
            int(getattr(config, "TREND_WEIGHT", 0.4) * 100),
        )
    except Exception as e:
        print_trade_action(f"Strategy init failed: {e}", "ERROR")
        logging.exception("Strategy initialization error")
        return

    # 7) Trading bot
    try:
        bot = TradingBot(
            exchange=exchange,
            strategy=strategy,
            base_currency=BASE_CURRENCY,
            simulate_only=LLAMA_SENTIMENT_SIMULATE_ONLY,
        )

        mode = "SIMULATION" if LLAMA_SENTIMENT_SIMULATE_ONLY else "LIVE TRADING"
        print_trade_action(
            f"✅ Bot initialized in {mode} mode (interval: {TRADE_INTERVAL}s)",
            "INFO" if LLAMA_SENTIMENT_SIMULATE_ONLY else "WARNING"
        )
    except Exception as e:
        print_trade_action(f"Bot initialization failed: {e}", "ERROR")
        logging.exception("Bot initialization error")
        return

    # 8) Start GUI
    print_trade_action("🚀 Ready! Click 'Start Bot' to begin trading")

    try:
        run_dashboard(bot, exchange, log_q)
    except KeyboardInterrupt:
        print_trade_action("👋 Shutting down gracefully...")
    except Exception as e:
        print_trade_action(f"Dashboard crashed: {e}", "CRASH")
        logging.exception("Dashboard error")
    finally:
        # Cleanup
        if scheduler:
            print_trade_action("Stopping sentiment scheduler...")
            stop_evt.set()
            scheduler.join(timeout=5.0)
        print_trade_action("✅ Shutdown complete")


if __name__ == "__main__":
    main()