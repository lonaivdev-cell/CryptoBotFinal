# CryptoBotFinal

A Python crypto trading bot for Binance that combines a **local LLM-driven sentiment engine** with **multi-timeframe technical analysis**, wrapped in a PySide6 desktop dashboard.

The strategy is a weighted hybrid: an Ollama-hosted LLaMA model rates market sentiment per coin on a continuous −1.0 to +1.0 scale, and that score is blended with a technical trend signal across the 15m / 1h / 4h timeframes. An optional portfolio mode adds diversified allocation, stop-loss protection, and time-based position exits.

> ⚠️ **Disclaimer**
> This is a portfolio / showcase project. The default configuration runs in **simulation mode** (`LLAMA_SENTIMENT_SIMULATE_ONLY = True`) and places no real orders. If you flip it to live trading, you do so at your own risk — this software is provided as-is, with no warranty of profitability or correctness. Don't trade with money you can't afford to lose.

## Features

- **Local-only AI** — sentiment scoring runs against an Ollama server on `localhost:11434`, or any OpenAI-compatible server such as llama.cpp's `llama-server` (`LLAMA_API_STYLE=openai`). No third-party AI APIs, no keys leaving your machine.
- **Hybrid signal blending** — configurable weights between AI sentiment and technical trend (defaults: 60% / 40%).
- **Rolling sentiment scheduler** — a background thread keeps a TTL cache of fresh sentiment scores, with adaptive QPS that backs off when the GPU is slow, jitter to avoid synchronized bursts, and staleness-aware coin rotation.
- **Two strategies, drop-in:**
  - `LlamaHybridStrategy` — basic hybrid signal across the configured coin list.
  - `DiversifiedPortfolioStrategy` — splits funds into a core allocation (90% across BTC/ETH/XRP/BNB/SOL/DOGE) plus an opportunistic bucket (10%), with stop-loss and time-based exits.
- **PySide6 GUI** — live tables of balances, sentiment scores, and signal breakdown; matplotlib equity / sentiment charts; queued logging stream wired into the dashboard.
- **Binance integration via `ccxt`** — spot orders by default; optional Binance Convert route.
- **Simulation mode** — drives the full pipeline without placing orders, for safe iteration.
- **Headless service** — `headless.py` runs the bot without the GUI behind a loopback JSON control API, as a hardened systemd service; the Lo's Server agent manages it from chat (see [below](#headless-service--los-server)).

## Repo layout

| File | Role |
|---|---|
| `main.py` | Entry point. Wires logging → exchange → LLaMA engine → sentiment scheduler → strategy → bot → GUI. |
| `headless.py` | Entry point without the GUI: the same chain on a background thread, plus the control API on `127.0.0.1:8097`. |
| `botservice/` | The headless service: safety rules (placeholder keys, live gate, redaction), settings, paper wallet, decision log, wiring, HTTP API. Imports no GUI code. |
| `deploy/` | `lo-cryptobot.service` (systemd), `install.sh`, the server's settings template and [deploy/README.md](deploy/README.md). |
| `tests/` | `unittest` suite with a fake exchange and fake model — no network, no ccxt needed. |
| `config.py` | All tunables. Reads secrets from `.env` via `python-dotenv`; everything else has a sensible default. |
| `intelligence.py` | Core domain: `ExchangeWrapper` (ccxt + signed Binance requests), `BinanceConvertClient`, `LlamaDecisionEngine` (Ollama or OpenAI-style), `LlamaHybridStrategy`, `TradingBot`, sentiment-line parser. |
| `portfolio_strategy.py` | `DiversifiedPortfolioStrategy` — subclass of the hybrid strategy with allocation + risk-management. |
| `optimization_engine.py` | `RollingSentimentCache` (TTL store) and `SentimentRollingScheduler` (background QPS-aware rotator). |
| `dashboard_enhanced.py` | PySide6 + matplotlib UI. |
| `trade_analyzer.py` | Post-hoc analysis helpers. |
| `parsers.py` | Misc parsing utilities. |
| `logs/logging_logic/log_compressor.py` | Rotating log handler with gzip compression of rotated files. |
| `test.py` | Standalone smoke test for the sentiment scheduler (needs a running model). |
| `.env.example` | Every setting with its default; copy to `.env`. |

## Requirements

- **Python 3.11+** (developed on 3.13).
- **A running [Ollama](https://ollama.ai) instance** with the model from `LLAMA_MODEL_NAME` pulled. The default is `llama3.1:8b-instruct-q4_K_M`. Lighter alternatives like `llama3.2:3b-instruct-q4_K_M` work too. Or an OpenAI-compatible server (`LLAMA_API_STYLE=openai`, `LLAMA_API_BASE_URL=http://127.0.0.1:8080/v1` for a local `llama-server`).
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

Create a `.env` in the project root from the example, which documents every setting:

```bash
cp .env.example .env
```

The essentials:

```dotenv
# Binance — read-only keys are enough for simulation mode. The shipped PLACEHOLDER values
# are recognised as such: the headless service then uses public market data only.
BINANCE_API_KEY=PLACEHOLDER_BINANCE_API_KEY
BINANCE_API_SECRET=PLACEHOLDER_BINANCE_API_SECRET
BINANCE_API_PASSWORD=          # leave empty unless your exchange requires it
BASE_CURRENCY=USDT

# Local LLM (Ollama by default; LLAMA_API_STYLE=openai for llama-server and friends)
LLAMA_API_STYLE=ollama
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

The settings below are read from the environment (or `.env`) under the same name; unset means the default shown, and a value that does not parse falls back to it. The rest of `config.py` is edited in place.

| Setting | Default | Purpose |
|---|---|---|
| `LLAMA_SENTIMENT_SIMULATE_ONLY` | `True` | Master safety switch. Only an explicit `False`/`0`/`no`/`off` places real orders (the headless service also needs `CRYPTOBOT_ALLOW_LIVE=1`). |
| `LLAMA_API_STYLE` | `ollama` | `ollama` (`{base}/generate`) or `openai` (`{base}/chat/completions`). |
| `LLAMA_API_BASE_URL` | `http://localhost:11434/api` | `http://127.0.0.1:8080/v1` by default when the style is `openai`. |
| `LLAMA_API_KEY` | empty | Bearer token for an OpenAI-style server started with `--api-key`. |
| `LLAMA_DISABLE_THINKING` | `1` | Ask Qwen3-style models not to think first (`chat_template_kwargs`); `<think>` blocks are stripped either way. |
| `SENTIMENT_COINS` | `BTC,ETH,SOL,BNB,XRP,DOGE,ADA,AVAX` | The watchlist (comma-separated in the environment). |
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

## Headless service & Lo's Server

On Lo's home server the bot runs without its GUI as `lo-cryptobot.service`, and the Lo's
Server agent (the AI tab of the Lo's Server app) manages it from chat: "how is the crypto bot
doing?", "what would it buy right now?", "start it on the hybrid strategy", "run one cycle".

```bash
sudo deploy/install.sh            # on the server: user, venv, settings, systemd unit
python headless.py                # or by hand: control API on 127.0.0.1:8097
python headless.py --check        # print the effective mode and settings, then exit
```

- **What it runs** — `headless.py` wires the same exchange → LLM engine → sentiment scheduler →
  strategy → `TradingBot` chain as `main.py`, on a background thread, without importing
  PySide6, matplotlib or the dashboard; `requirements-headless.txt` (ccxt, requests,
  python-dotenv) is all it needs.
- **Control API** (loopback only, JSON, errors as `{"error": sentence}`): `GET /status`,
  `/signals`, `/decisions?limit=`, `/balances`, `/config`, `/paper`; `POST /start
  {"strategy": "hybrid"|"portfolio"}`, `/stop`, `/cycle`. Full reference in
  [deploy/README.md](deploy/README.md#control-api).
- **Simulation by default, and the API cannot change that.** Live trading needs
  `LLAMA_SENTIMENT_SIMULATE_ONLY=False` *and* `CRYPTOBOT_ALLOW_LIVE=1` set by a human in the
  bot's environment, plus real keys. `/status` reports the mode in effect; every decision says
  whether it was simulated.
- **Paper wallet** — simulated trades fill against a paper balance (`CRYPTOBOT_PAPER_USDT`,
  1000 by default) kept across restarts, so sells, stop-losses and time exits actually happen in
  simulation and `GET /paper` shows the result.
- **Placeholder keys** — while `BINANCE_API_KEY`/`SECRET` are placeholders the bot uses public
  market data only and `/balances` says where to put real keys. Keys are never logged or
  returned, and never reach the agent.
- **Shared model** — with `LLAMA_API_STYLE=openai` it talks to the same `llama-server` as the
  agent; when that is down, the bot decides on trend alone and `/status` says the model is not
  reachable.

[deploy/README.md](deploy/README.md) covers installation, how the agent's tools map to the API,
and the safety model in full. Tests: `python3 -m unittest discover -s tests -t .` (no network,
no ccxt needed; the GUI smoke test runs where PySide6 is installed).

## Status

This is a **personal showcase project** built to explore local-LLM-assisted strategies. It is not actively maintained for production use. Issues and PRs are welcome as portfolio conversation starters, but expect no SLA on responses.

## License

[MIT](LICENSE) — see `LICENSE` for the full text.
