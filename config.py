"""
CryptoBot2 configuration — hybrid-only (LlamaHybridStrategy)
Safe env handling + settings that match the merged intelligence/main.
"""
from datetime import datetime, timezone
import os

# Load environment variables from .env at project root (optional)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass


# ──────────────────────────────────────────────────────────────
# Environment helpers. Every default below is the value this file had before it read the
# environment, so an empty environment behaves exactly as the hard-coded config did. A value
# that does not parse falls back to the default instead of crashing the bot at import time.
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    # NaN and infinity compare false against every threshold, which would silently disable
    # the strategy rather than fail.
    return value if value == value and value not in (float("inf"), float("-inf")) else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_coins(name: str, default: list) -> list:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)
    coins = []
    for part in raw.replace(",", " ").replace(";", " ").split():
        coin = part.strip().upper()
        # Symbols end up in "COIN/USDT" market names and in prompts; anything that is not a
        # plain ticker is dropped rather than passed through.
        if 2 <= len(coin) <= 15 and coin.isalnum() and coin not in coins:
            coins.append(coin)
    return coins or list(default)


# ──────────────────────────────────────────────────────────────
# Exchange
EXCHANGE_ID   = os.getenv("EXCHANGE_ID", "binance")
API_KEY       = os.getenv("BINANCE_API_KEY", "")
API_SECRET    = os.getenv("BINANCE_API_SECRET", "")
API_PASSWORD  = os.getenv("BINANCE_API_PASSWORD", "")
BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USDT")
BINANCE_API_BASE_URL = os.getenv("BINANCE_API_BASE_URL", "https://api.binance.com")

# ──────────────────────────────────────────────────────────────
# Bot core / GUI
TRADE_INTERVAL = int(os.getenv("TRADE_INTERVAL", "10"))  # seconds between strategy ticks
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

# ──────────────────────────────────────────────────────────────
# LLaMA / Ollama
LLAMA_ENABLED = True
LLAMA_MODEL_NAME      = os.getenv("LLAMA_MODEL_NAME", "llama3.1:8b-instruct-q4_K_M").split("=", 1)[-1].strip()
# "ollama" (the default: POST {base}/generate) or "openai" (POST {base}/chat/completions, as
# served by llama.cpp's llama-server, vLLM, LM Studio…). On loserver the bot shares the
# llama-server the Lo's Server agent uses: LLAMA_API_STYLE=openai,
# LLAMA_API_BASE_URL=http://127.0.0.1:8080/v1.
LLAMA_API_STYLE       = os.getenv("LLAMA_API_STYLE", "ollama").strip().lower() or "ollama"
LLAMA_API_BASE_URL    = os.getenv("LLAMA_API_BASE_URL", "").strip() or (
    "http://127.0.0.1:8080/v1" if LLAMA_API_STYLE == "openai" else "http://localhost:11434/api"
)
# Bearer token for an OpenAI-style server started with --api-key; empty means none is sent.
LLAMA_API_KEY         = os.getenv("LLAMA_API_KEY", "").strip()
# Qwen3-family models think before answering unless told not to, and a 32-token budget is all
# thinking and no score. Sent as chat_template_kwargs.enable_thinking=false (OpenAI style only).
LLAMA_DISABLE_THINKING = _env_bool("LLAMA_DISABLE_THINKING", True)
LLAMA_SYSTEM_PROMPT   = os.getenv("LLAMA_SYSTEM_PROMPT", "").strip()
LLAMA_KEEP_ALIVE      = os.getenv("LLAMA_KEEP_ALIVE", "2h")
LLAMA_REQUEST_TIMEOUT = int(os.getenv("LLAMA_REQUEST_TIMEOUT", "45"))
LLAMA_MAX_RETRIES     = int(os.getenv("LLAMA_MAX_RETRIES", "3"))
LLAMA_RETRY_BACKOFF   = float(os.getenv("LLAMA_RETRY_BACKOFF", "1.6"))

# Passed to Ollama /generate "options"
LLAMA_OPTIONS = {
    "raw": True,
    "num_predict": int(os.getenv("LLAMA_NUM_PREDICT", "32")),
    "temperature": float(os.getenv("LLAMA_TEMPERATURE", "0.7")),  # 🔹 INCREASED from 0.4
    "top_p": float(os.getenv("LLAMA_TOP_P", "0.9")),  # 🔹 INCREASED from 0.85
    "top_k": int(os.getenv("LLAMA_TOP_K", "50")),  # 🔹 INCREASED from 40
    "repeat_penalty": float(os.getenv("LLAMA_REPEAT_PENALTY", "1.2")),  # 🔹 INCREASED from 1.1
    "num_ctx": int(os.getenv("LLAMA_NUM_CTX", "4096")),
    "mirostat": 0,
    "seed": -1,  # 🔹 ADD THIS: Random seed each time
}

# Strict, parse-friendly prompt. intelligence.py will call build_sentiment_prompt(coins, include_date=True)
LLAMA_SENTIMENT_PROMPT = """You are a crypto sentiment engine.
Return ONLY lines in the exact format:

SYMBOL: SCORE

Rules:
- One line per ticker provided below, same order.
- Output EXACTLY {n} lines, one per ticker, in the SAME ORDER as listed.
- SCORE is a number between -1.0 and 1.0 (use at most 2 decimals).
- No extra words, no headers, no dates, no explanations. Only the lines.

Tickers: {tickers_section}

Example:
BTC: 0.12
ETH: -0.34

Now output:
"""

HYBRID_SENTIMENT_WEIGHT = 0.7
HYBRID_TREND_WEIGHT     = 0.4
HYBRID_REQUIRE_TREND_CONFIRMATION = False

# Trade thresholds / sizing
SENTIMENT_BUY_THRESHOLD   = _env_float("SENTIMENT_BUY_THRESHOLD", 0.45)
SENTIMENT_SELL_THRESHOLD  = _env_float("SENTIMENT_SELL_THRESHOLD", -0.40)
SENTIMENT_TRADE_FRACTION  = 0.50
# Min trade value
MIN_TRADE_USD             = float(os.getenv("MIN_TRADE_USD", "5.0"))

# ═══════════════════════════════════════════════════════════════════
# STRATEGY SELECTION
# ═══════════════════════════════════════════════════════════════════
USE_PORTFOLIO_STRATEGY = _env_bool("USE_PORTFOLIO_STRATEGY", True)  # Set True to enable risk management features

# Portfolio strategy settings (only used if USE_PORTFOLIO_STRATEGY=True)
STOP_LOSS_PCT = _env_float("STOP_LOSS_PCT", 0.15)  # Exit position if down 15%
POSITION_MAX_DAYS = _env_int("POSITION_MAX_DAYS", 10)  # Auto-exit opportunistic positions after 10 days
ENABLE_STOP_LOSS = True  # Enable stop-loss protection
ENABLE_TIME_EXIT = True  # Enable time-based exits

# ═══════════════════════════════════════════════════════════════════
# SENTIMENT OPTIMIZATION (Rolling Scheduler)
# ═══════════════════════════════════════════════════════════════════
SENTIMENT_ROLLING_ENABLED = True  # Enable background AI scheduler
SENTIMENT_TTL_SECONDS = _env_int("SENTIMENT_TTL_SECONDS", 240)  # 🔹 CHANGED: 4 minutes (was 900 = 15 min)
SENTIMENT_MAX_QPS = _env_float("SENTIMENT_MAX_QPS", 0.25)  # 🔹 CHANGED: Faster (was 0.25)
SENTIMENT_BATCH_SIZE = 1  # Coins per request (1 = safer, 2-3 = faster)
SENTIMENT_ROTATE_ORDER = "staleness"  # "staleness" or "round_robin"
SENTIMENT_ENABLE_JITTER = True  # Add random timing jitter
SENTIMENT_ADAPTIVE_RATE = True  # Adapt QPS based on GPU speed

# Sentiment coins to track
SENTIMENT_COINS = _env_coins("SENTIMENT_COINS", ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX"])

# ═══════════════════════════════════════════════════════════════════
# HYBRID STRATEGY WEIGHTS
# ═══════════════════════════════════════════════════════════════════
SENTIMENT_WEIGHT = _env_float("SENTIMENT_WEIGHT", 0.6)  # Weight for AI sentiment (60%)
TREND_WEIGHT = _env_float("TREND_WEIGHT", 0.4)  # Weight for technical trend (40%)
REQUIRE_TREND_CONFIRMATION = True  # Block buys if trend is negative
MIN_SENTIMENT_DELTA = 0.12  # Minimum change to log (reduces noise)

# ═══════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYSIS
# ═══════════════════════════════════════════════════════════════════
TIMEFRAMES = ["15m", "1h", "4h"]  # Multiple timeframes for trend analysis


# Optional: Custom prompt builder function
def build_sentiment_prompt(coins: list, include_date: bool = True, market_data: dict = None) -> str:
    """
    Build a strict sentiment prompt that forces structured output.

    Args:
        coins: List of coin symbols
        include_date: Whether to include current date context
        market_data: Optional dict with real-time data like {"BTC": {"price_change_24h": 2.5, "volume_change": 15}}

    Returns:
        Formatted prompt string that enforces line-by-line output
    """
    from datetime import datetime
    import random

    date_context = ""
    if include_date:
        now = datetime.now()
        date_context = f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # 🔹 ADD MARKET DATA CONTEXT
    market_context = ""
    if market_data:
        market_context = "Recent Market Data (24h):\n"
        for coin in coins:
            if coin in market_data:
                data = market_data[coin]
                price_change = data.get('price_change_24h', 0)
                volume_change = data.get('volume_change_24h', 0)
                market_context += f"- {coin}: Price {price_change:+.2f}%, Volume {volume_change:+.1f}%\n"
        market_context += "\n"

    coin_list = "\n".join(coins)

    # 🔹 ADD RANDOM SEED TO PREVENT CACHING
    request_id = random.randint(1000, 9999)

    # 🔹 CRITICAL: Use strict format with examples
    return f"""{date_context}{market_context}Rate each cryptocurrency's market sentiment on a scale from -1.0 to +1.0.

REQUEST ID: {request_id}

RULES:
- Output EXACTLY one line per coin
- Format: "COIN: score" (e.g., "BTC: 0.65")
- Score must be between -1.0 (very bearish) and +1.0 (very bullish)
- Use 2 decimal places
- NO explanations, NO extra text, NO summaries
- Consider: Price trends, volume, market momentum, news sentiment

EXAMPLE OUTPUT:
BTC: 0.75
ETH: 0.62
SOL: -0.23

NOW RATE THESE COINS:
{coin_list}"""


# ───────────────── Opportunistic scalper (quick trades) ─────────────────
OPP_ENABLED = True                       # master switch
OPP_BASE_NOTIONAL_USD = _env_float("OPP_BASE_NOTIONAL_USD", 15.0)  # how much USDT to use per quick trade
OPP_TP_PCT = 0.0025                      # +0.25% take-profit (fee-aware)
OPP_SL_PCT = 0.0030                      # −0.30% stop-loss (tight)
OPP_MAX_SPREAD_PCT = 0.0015              # 0.15% max bid/ask spread to enter
OPP_COOLDOWN_SECONDS = 60                # per-coin cooldown after exit
OPP_MAX_OPEN_POSITIONS = 2               # cap concurrent quick positions
OPP_FEE_PCT = 0.001                      # taker fee (0.10%) — set your real taker fee!


# Risk
TRAILING_STOP_PCT = 0.06
COOLDOWN_SECONDS  = 180

# Execution route
# Convert API is optional; disable by default to avoid 400s unless your API key
# has Convert permissions and the endpoint is fully enabled for your account.
LLAMA_SENTIMENT_USE_CONVERT   = True
# Master safety switch: True = do not actually place orders. Only an explicit false word
# ("False", "0", "no", "off") in the environment turns it off; a typo keeps simulation on.
# The headless service (headless.py) additionally requires CRYPTOBOT_ALLOW_LIVE=1 and real keys.
LLAMA_SENTIMENT_SIMULATE_ONLY = _env_bool("LLAMA_SENTIMENT_SIMULATE_ONLY", True)

# GUI defaults (optional)
GUI_SHOW_TOTAL_LINE = os.getenv("GUI_SHOW_TOTAL_LINE", "True").lower() == "true"
GUI_DEFAULT_THEME   = os.getenv("GUI_DEFAULT_THEME", "dark")
