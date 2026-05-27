# CryptoBotFinal

A Python crypto trading bot for Binance that combines a **local LLM-driven sentiment engine** with **multi-timeframe technical analysis**, wrapped in a PySide6 desktop dashboard.

The strategy is a weighted hybrid: an Ollama-hosted LLaMA model rates market sentiment per coin on a continuous −1.0 to +1.0 scale, and that score is blended with a technical trend signal across the 15m / 1h / 4h timeframes. An optional portfolio mode adds diversified allocation, stop-loss protection, and time-based position exits.

> ⚠️ **Disclaimer**
> This is a portfolio / showcase project. The default configuration runs in **simulation mode** (`LLAMA_SENTIMENT_SIMULATE_ONLY = True`) and places no real orders. If you flip it to live trading, you do so at your own risk — this software is provided as-is, with no warranty of profitability or correctness. Don't trade with money you can't afford to lose.

## Features

- **Local-only AI** — sentiment scoring runs against an Ollama server on `localhost:11434`. No third-party AI APIs, no keys leaving your machine.
- **Hybrid signal blending** — configurable weights between AI sentiment and technical trend (defaults: 60% / 40%).
- **Rolling sentiment scheduler** — a background thread keeps a TTL cache of fresh sentiment scores, with adaptive QPS that backs off when the GPU is slow, jitter to avoid synchronized bursts, and staleness-aware coin rotation.
- **Two strategies, drop-in:**
  - `LlamaHybridStrategy` — basic hybrid signal across the configured coin list.
  - `DiversifiedPortfolioStrategy` — splits funds into a core allocation (90% across BTC/ETH/XRP/BNB/SOL/DOGE) plus an opportunistic bucket (10%), with stop-loss and time-based exits.
- **PySide6 GUI** — live tables of balances, sentiment scores, and signal breakdown; matplotlib equity / sentiment charts; queued logging stream wired into the dashboard.
- **Binance integration via `ccxt`** — spot orders by default; optional Binance Convert route.
- **Simulation mode** — drives the full pipeline without placing orders, for safe iteration.

## Repo layout

| File | Role |
|---|---|
| `main.py` | Entry point. Wires logging → exchange → LLaMA engine → sentiment scheduler → strategy → bot → GUI. |
| `config.py` | All tunables. Reads secrets from `.env` via `python-dotenv`; everything else has a sensible default. |
| `intelligence.py` | Core domain: `ExchangeWrapper` (ccxt + signed Binance requests), `BinanceConvertClient`, `LlamaDecisionEngine`, `LlamaHybridStrategy`, `TradingBot`, sentiment-line parser. |
| `portfolio_strategy.py` | `DiversifiedPortfolioStrategy` — subclass of the hybrid strategy with allocation + risk-management. |
| `optimization_engine.py` | `RollingSentimentCache` (TTL store) and `SentimentRollingScheduler` (background QPS-aware rotator). |
| `dashboard_enhanced.py` | PySide6 + matplotlib UI. |
| `trade_analyzer.py` | Post-hoc analysis helpers. |
| `parsers.py` | Misc parsing utilities. |
| `logs/logging_logic/log_compressor.py` | Rotating log handler with gzip compression of rotated files. |
| `test.py` | Standalone smoke test for the sentiment scheduler. |

## Requirements

- **Python 3.11+** (developed on 3.13).
- **A running [Ollama](https://ollama.ai) instance** with the model from `LLAMA_MODEL_NAME` pulled. The default is `llama3.1:8b-instruct-q4_K_M`. Lighter alternatives like `llama3.2:3b-instruct-q4_K_M` work too.
- **A Binance account + API key** if you intend to run against live data. For simulation only, an API key is still recommended so balance and ticker calls succeed.
- A GPU is helpful for the LLM but not required — the scheduler is built to absorb slow inference via backpressure.

## Setup

```bash
git clone https://github.com/lonaivdev-cell/CryptoBotFinal.git
cd CryptoBotFinal

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```dotenv
# Binance — read-only keys are enough for simulation mode
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_API_PASSWORD=          # leave empty unless your exchange requires it
BASE_CURRENCY=USDT

# Local LLM
LLAMA_API_BASE_URL=http://localhost:11434/api
LLAMA_MODEL_NAME=llama3.1:8b-instruct-q4_K_M
```

Pull the model and verify Ollama is responding:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
curl http://localhost:11434/api/tags
```

## Running

```bash
python main.py
```

The PySide6 dashboard launches. **The bot does not trade automatically on startup** — you click **Start Bot** in the GUI once you've reviewed the live sentiment / trend readout.

To smoke-test the sentiment scheduler in isolation (no GUI, no exchange):

```bash
python test.py
```

## Configuration highlights

All values in `config.py` are overridable via environment variables of the same name. Some worth knowing:

| Setting | Default | Purpose |
|---|---|---|
| `LLAMA_SENTIMENT_SIMULATE_ONLY` | `True` | Master safety switch. Set to `False` to place real orders. |
| `SENTIMENT_COINS` | `["BTC","ETH","SOL","BNB","XRP","DOGE","ADA","AVAX"]` | The watchlist. |
| `SENTIMENT_BUY_THRESHOLD` / `SELL_THRESHOLD` | `0.45` / `-0.40` | Hybrid score thresholds to act. |
| `SENTIMENT_WEIGHT` / `TREND_WEIGHT` | `0.6` / `0.4` | Hybrid blend. |
| `SENTIMENT_TTL_SECONDS` | `240` | How long a cached sentiment is considered fresh. |
| `SENTIMENT_MAX_QPS` | `0.25` | Upper bound on LLM requests / sec. |
| `USE_PORTFOLIO_STRATEGY` | `True` | Use the diversified portfolio strategy instead of the basic hybrid. |
| `STOP_LOSS_PCT` | `0.15` | Per-position drawdown that triggers exit (portfolio mode). |
| `POSITION_MAX_DAYS` | `10` | Auto-exit opportunistic positions after this many days. |
| `OPP_BASE_NOTIONAL_USD` | `15.0` | USDT per opportunistic scalp. |
| `MIN_TRADE_USD` | `5.0` | Minimum trade notional. |

## How the hybrid signal is built

For each coin on each tick:

1. **Sentiment score** is read from the rolling cache (background scheduler keeps it fresh). If stale, the strategy may issue a direct LLM call.
2. **Trend score** is computed from technical indicators across the configured `TIMEFRAMES` (15m / 1h / 4h).
3. The two are blended: `hybrid = SENTIMENT_WEIGHT * sentiment + TREND_WEIGHT * trend`.
4. `REQUIRE_TREND_CONFIRMATION` (default `True`) blocks buys whose trend component is negative even if sentiment is bullish.
5. If `hybrid >= SENTIMENT_BUY_THRESHOLD`, the bot emits a buy `TradeInstruction`; symmetrically for sell.
6. In portfolio mode, instructions are gated by the core/opportunistic allocation rules, stop-loss, and time-exit checks.

## Status

This is a **personal showcase project** built to explore local-LLM-assisted strategies. It is not actively maintained for production use. Issues and PRs are welcome as portfolio conversation starters, but expect no SLA on responses.

## License

[MIT](LICENSE) — see `LICENSE` for the full text.
