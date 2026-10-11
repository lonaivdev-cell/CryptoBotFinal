# lo-cryptobot — CryptoBot as a service on Lo's Server

`headless.py` runs CryptoBot without the desktop GUI: the same exchange → sentiment model →
scheduler → strategy → bot chain as `main.py`, on a background thread, controlled through a
small JSON API on `127.0.0.1:8097`. On loserver it runs as `lo-cryptobot.service`, and the
Lo's Server agent (the app's AI tab) manages it through its `cryptobot` plugin.

**It simulates.** With the shipped settings it places no orders, ever: trades fill in a paper
wallet. Live trading takes two switches a human sets on the server, plus real keys — see
[Safety model](#safety-model).

## Install on loserver

```bash
git clone https://github.com/lonaivdev-cell/CryptoBotFinal.git ~/CryptoBotFinal
cd ~/CryptoBotFinal
sudo deploy/install.sh
```

`install.sh` is idempotent — run it again after a `git pull` to deploy new code. It:

1. creates the system user `cryptobot` (no home, no shell, no supplementary groups);
2. copies the headless code to `/opt/lo-cryptobot/app` (root-owned, read-only to the bot — the
   GUI files are not installed);
3. creates `/opt/lo-cryptobot/venv` and installs `requirements-headless.txt` into it (ccxt,
   requests, python-dotenv — no PySide6, matplotlib or pandas);
4. creates `/etc/lo-cryptobot/cryptobot.env` from `deploy/cryptobot.env.example` the first
   time (never overwrites it), mode `0600 root:root`;
5. installs and starts `lo-cryptobot.service`, then checks that `GET /status` answers.

It needs Python 3.10+ with `venv` (`sudo apt install python3-venv`) and network access to PyPI.

### By hand, instead

```bash
sudo useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin cryptobot
sudo mkdir -p /opt/lo-cryptobot && sudo cp -R . /opt/lo-cryptobot/app      # or a subset, as install.sh does
sudo python3 -m venv /opt/lo-cryptobot/venv
sudo /opt/lo-cryptobot/venv/bin/pip install -r /opt/lo-cryptobot/app/requirements-headless.txt
sudo install -d -m 0755 /etc/lo-cryptobot
sudo install -m 0600 deploy/cryptobot.env.example /etc/lo-cryptobot/cryptobot.env
sudo install -m 0644 deploy/lo-cryptobot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now lo-cryptobot
```

### Check it

```bash
systemctl status lo-cryptobot
journalctl -u lo-cryptobot -f
curl -s http://127.0.0.1:8097/status
sudo -u cryptobot env $(sudo grep -v '^#' /etc/lo-cryptobot/cryptobot.env | xargs) \
    /opt/lo-cryptobot/venv/bin/python /opt/lo-cryptobot/app/headless.py --check   # effective settings, as JSON
```

## Configure

Everything lives in `/etc/lo-cryptobot/cryptobot.env` (edit with sudo, then
`sudo systemctl restart lo-cryptobot`). The shipped file is tuned for loserver:

- **Binance keys** — `BINANCE_API_KEY` / `BINANCE_API_SECRET` ship as `PLACEHOLDER_…`. While
  they are placeholders (empty, `PLACEHOLDER…`, `your_key_here`, anything shorter than 16
  characters), the bot uses public market data only — prices and candles through ccxt without
  credentials — simulates every trade, and `GET /balances` answers 409 saying so. Read-only keys
  are enough for balances; live trading needs spot trading enabled on the key.
- **Sentiment model** — `LLAMA_API_STYLE=openai`, `LLAMA_API_BASE_URL=http://127.0.0.1:8080/v1`:
  the same llama-server the agent uses. It is part of the scraper's compose stack and is not
  started at boot; while it is down `/status` says `"reachable": false` and the bot decides on
  the technical trend alone. Ollama still works (`LLAMA_API_STYLE=ollama`, the default when
  unset).
- **Pace** — one cycle a minute (`TRADE_INTERVAL=60`) and sentiment refreshed every 15 minutes
  per coin at most 0.05 requests/s (`SENTIMENT_TTL_SECONDS=900`, `SENTIMENT_MAX_QPS=0.05`):
  the model runs on CPU at ~9.5 tokens/s and the agent needs it too. The scheduler only runs
  while the trading loop does.
- Every other setting, with its default, is documented in `.env.example` at the repo root.

The unit sets `CRYPTOBOT_STATE_DIR`, `CRYPTOBOT_LOG_DIR` and `CRYPTOBOT_ENV_FILE` itself; do not
put them in the `.env` (it would win, and point the bot at directories the sandbox keeps
read-only).

## How the Lo's Server agent manages it

The agent's `cryptobot` plugin (`server/agent/lo_agent/plugins/cryptobot` in Grafanafix) is a
client of this API at `LO_AGENT_CRYPTOBOT_URL` (default `http://127.0.0.1:8097`):

| Agent tool | Endpoint |
|---|---|
| `crypto_status` | `GET /status` |
| `crypto_signals` | `GET /signals` |
| `crypto_decisions` | `GET /decisions?limit=N` |
| `crypto_balances` | `GET /balances` |
| `crypto_config` | `GET /config` |
| `crypto_start` | `POST /start {"strategy": …}` |
| `crypto_stop` | `POST /stop` |
| `crypto_run_cycle` | `POST /cycle` |

So from the chat Lo can ask how the bot is doing, what it would buy, what it decided overnight,
start or stop the loop, or run one cycle now. What the agent cannot do: switch to live trading,
see or set keys (they never reach the agent; its user cannot read the `.env`), or start/stop the
service itself (it has no sudo) — if the bot is down it says so and names the `systemctl`
command for Lo to run.

## Control API

Loopback only, JSON in and out, errors as `{"error": "<sentence>"}`.

| | |
|---|---|
| `GET /status` | `{"ok", "running", "mode": "simulation"\|"live", "keys_configured", "llm": {"url", "reachable"}, "strategy": "hybrid"\|"portfolio", "started_at": ms\|null, "last_cycle_at": ms\|null, "cycles", "coins", "last_error": str\|null}` |
| `GET /signals` | `{"signals": [{"coin", "sentiment": float\|null, "trend": float\|null, "hybrid", "action": "buy"\|"sell"\|"hold", "at": ms}]}` — the latest cycle's, per coin. `sentiment` is null until the model has actually scored the coin. |
| `GET /decisions?limit=20` | `{"decisions": [{"at", "coin", "side", "amount", "price", "simulated", "reason"}]}`, newest first, `limit` 1–500. `amount` is in the coin (filled amount for paper fills), `price` the price the strategy saw. Kept across restarts. |
| `GET /balances` | `{"balances": {"USDT": {"free", "used", "total"}, …}}` from Binance, or 409 while the keys are placeholders. |
| `GET /config` | Redacted settings: coins, thresholds, weights, interval, simulate_only, strategy, risk, sentiment pacing, model URL, and why live trading is blocked if a switch is set. Never keys. |
| `GET /paper` | The paper wallet: balances, average costs, last prices, value and P&L against the starting balance (409 in live mode). |
| `POST /start` | `{"strategy": "hybrid"\|"portfolio"}` (optional; defaults to the last one, else `USE_PORTFOLIO_STRATEGY`). Starts the loop and the sentiment scheduler; answers with the status. 409 if already running. |
| `POST /stop` | Stops both; answers with the status. Stopping a stopped bot is fine. |
| `POST /cycle` | Runs one decision cycle now and answers `{"decisions": [...]}`. Works whether or not the loop runs; with the loop stopped it first scores stale coins itself (up to `CRYPTOBOT_CYCLE_SENTIMENT_BUDGET_S`, 150 s), if the model answers. |

```bash
curl -s 127.0.0.1:8097/status
curl -s -X POST -H 'Content-Type: application/json' -d '{"strategy":"hybrid"}' 127.0.0.1:8097/start
curl -s '127.0.0.1:8097/decisions?limit=5'
```

## Safety model

- **Simulation is the default and the API cannot leave it.** The mode is decided once, when
  the service starts, from its environment: live trading needs `LLAMA_SENTIMENT_SIMULATE_ONLY=False`
  **and** `CRYPTOBOT_ALLOW_LIVE=1` **and** real keys. Only an explicit false word (`False`, `0`,
  `no`, `off`) turns the first switch off — a typo keeps simulation on — and only the exact
  string `1` counts for the second. There is no endpoint that changes the mode; a request body
  that so much as names a mode field (`mode`, `live`, `simulate_only`, …) or a key field is
  refused with 403. `/status` and every decision say which mode is in effect.
- **Simulated trades never reach Binance.** In simulation the strategy sees a paper exchange:
  market data from Binance, balances from the paper wallet, and an order method that raises.
  The Binance Convert route (which bypasses the bot) is disabled in simulation and never used by
  the service. With placeholder keys the ccxt client carries no credentials at all.
- **Keys stay in one root-only file.** `/etc/lo-cryptobot/cryptobot.env` is `0600 root:root`;
  systemd hands the values to the bot's environment. They are never logged, never returned:
  `/config` is built from an allow-list, and every error sentence and log line passes through a
  redactor that knows the actual key values. The agent never receives them.
- **Loopback only, and not from a browser.** The API refuses to bind to anything but
  127.0.0.0/8 or ::1, refuses requests whose `Host` is not a loopback name (DNS rebinding) and
  requests carrying an `Origin` header (a web page). `CRYPTOBOT_CONTROL_TOKEN` adds a bearer
  token if any other local user should be kept out.
- **The process cannot do much else either.** The unit runs as `cryptobot` with no
  capabilities, `NoNewPrivileges`, a read-only file system except its state and log
  directories, no access to `/home`, private `/tmp` and devices, a system-call filter and memory
  limits (`systemd-analyze security lo-cryptobot`).

## State, logs, updates

- `/var/lib/lo-cryptobot/paper.json` — the paper wallet. Delete it (with the service stopped) to
  start over with `CRYPTOBOT_PAPER_USDT`.
- `/var/lib/lo-cryptobot/decisions.jsonl` — the decision log behind `/decisions` (trimmed
  automatically to the last 1000).
- `/var/log/lo-cryptobot/headless.log` (rotated daily, 7 kept) and the journal.
- Update: `git pull && sudo deploy/install.sh`. Uninstall: `sudo systemctl disable --now
  lo-cryptobot`, then remove `/etc/systemd/system/lo-cryptobot.service`, `/opt/lo-cryptobot`,
  `/etc/lo-cryptobot`, `/var/lib/lo-cryptobot`, `/var/log/lo-cryptobot` and the `cryptobot` user.

## Running it by hand (development)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-headless.txt
cp .env.example .env            # placeholders are fine
.venv/bin/python headless.py    # state in ./state, logs in ./logs
```

Tests need nothing installed (no network, no ccxt, fakes throughout):

```bash
python3 -m unittest discover -s tests -t .
```
