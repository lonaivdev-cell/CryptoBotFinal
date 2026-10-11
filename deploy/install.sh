#!/usr/bin/env bash
# Installs, or updates, the headless crypto bot (lo-cryptobot.service) on loserver.
#
#   sudo deploy/install.sh
#
# Idempotent: run it again after a `git pull` to deploy new code. The settings file
# /etc/lo-cryptobot/cryptobot.env is created once from deploy/cryptobot.env.example and never
# overwritten (it holds the Binance keys); only its ownership and mode are re-asserted.
#
# Layout:
#   /opt/lo-cryptobot/app      the code (root-owned, read-only to the bot), replaced on update
#   /opt/lo-cryptobot/venv     Python packages from requirements-headless.txt, kept across updates
#   /etc/lo-cryptobot/cryptobot.env   keys and settings, 0600 root:root
#   /var/lib/lo-cryptobot      paper wallet and decision log (owned by the bot)
#   /var/log/lo-cryptobot      rotating log file (the journal has the same lines)
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "install.sh: run it with sudo — it creates a system user and installs a systemd unit." >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
PREFIX=/opt/lo-cryptobot
APP="$PREFIX/app"
VENV="$PREFIX/venv"
ETC=/etc/lo-cryptobot
ENV_FILE="$ETC/cryptobot.env"
UNIT=/etc/systemd/system/lo-cryptobot.service
BOT_USER=cryptobot
PYTHON="${PYTHON:-/usr/bin/python3}"

say() { printf '==> %s\n' "$*"; }

# ---------------------------------------------------------------------------- prerequisites

if [ ! -x "$PYTHON" ] || ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "install.sh: the bot needs Python 3.10 or newer at $PYTHON (set PYTHON=/path/to/python3 to use another)." >&2
    exit 1
fi
if ! "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
    echo "install.sh: Python's venv module is missing; install it first (Ubuntu: sudo apt install python3-venv)." >&2
    exit 1
fi
if [ ! -f "$SRC/headless.py" ] || [ ! -d "$SRC/botservice" ]; then
    echo "install.sh: cannot find the bot's code next to this script (expected $SRC/headless.py)." >&2
    exit 1
fi
NOLOGIN="$(command -v nologin || echo /usr/sbin/nologin)"

# ---------------------------------------------------------------------------- the user

if id -u "$BOT_USER" >/dev/null 2>&1; then
    say "user $BOT_USER exists"
else
    say "creating system user $BOT_USER (no home, no login shell)"
    useradd --system --user-group --no-create-home --home-dir /nonexistent \
        --shell "$NOLOGIN" --comment "Lo's Server crypto bot" "$BOT_USER"
fi
# No supplementary groups at all: in particular not docker (root in all but name) and not the
# agent's group. -G replaces the list, so this also undoes any group added by hand.
usermod -G "" -s "$NOLOGIN" "$BOT_USER"
passwd -l "$BOT_USER" >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------- the code

say "installing code to $APP (root-owned, read-only to $BOT_USER)"
mkdir -p "$PREFIX"
chmod 0755 "$PREFIX"
stage="$(mktemp -d "$PREFIX/.app.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
# The headless service's files only: no GUI, no tests, never a .env or local state.
for f in headless.py config.py intelligence.py optimization_engine.py portfolio_strategy.py \
         parsers.py trade_analyzer.py __init__.py requirements-headless.txt README.md LICENSE .env.example; do
    [ -f "$SRC/$f" ] && cp "$SRC/$f" "$stage/"
done
cp -R "$SRC/botservice" "$stage/"
mkdir -p "$stage/logs/logging_logic" "$stage/deploy"
cp "$SRC/logs/logging_logic/log_compressor.py" "$stage/logs/logging_logic/"
cp "$SRC/deploy/README.md" "$SRC/deploy/lo-cryptobot.service" "$SRC/deploy/cryptobot.env.example" "$stage/deploy/"
find "$stage" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$stage" -name '*.pyc' -delete
chown -R root:root "$stage"
chmod -R u=rwX,go=rX "$stage"
chmod 0755 "$stage"
if [ -d "$APP" ]; then
    rm -rf "$APP.old"
    mv "$APP" "$APP.old"
fi
mv "$stage" "$APP"
trap - EXIT
rm -rf "$APP.old"

# ---------------------------------------------------------------------------- the venv

if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    say "creating the virtualenv at $VENV"
    rm -rf "$VENV"
    "$PYTHON" -m venv "$VENV"
fi
say "installing Python packages (ccxt, requests, python-dotenv) into $VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet --upgrade -r "$APP/requirements-headless.txt"
chown -R root:root "$VENV"
chmod -R go-w "$VENV"
# Compiled here, as root: the unit makes /opt read-only and sets PYTHONDONTWRITEBYTECODE.
"$VENV/bin/python" -m compileall -q "$APP" >/dev/null

# ---------------------------------------------------------------------------- settings

mkdir -p "$ETC"
chown root:root "$ETC"
chmod 0755 "$ETC"
if [ -f "$ENV_FILE" ]; then
    say "keeping $ENV_FILE"
else
    say "creating $ENV_FILE from deploy/cryptobot.env.example (Binance keys are PLACEHOLDERS)"
    cp "$SRC/deploy/cryptobot.env.example" "$ENV_FILE"
fi
# systemd reads it as root before starting the bot; nobody else needs to.
chown root:root "$ENV_FILE"
chmod 0600 "$ENV_FILE"

# ---------------------------------------------------------------------------- the unit

say "installing $UNIT"
install -m 0644 -o root -g root "$SRC/deploy/lo-cryptobot.service" "$UNIT"
systemctl daemon-reload
systemctl enable lo-cryptobot.service >/dev/null
if systemctl is-active --quiet lo-cryptobot.service; then
    say "restarting lo-cryptobot.service with the new code"
    systemctl restart lo-cryptobot.service
else
    say "starting lo-cryptobot.service"
    systemctl start lo-cryptobot.service
fi

# ---------------------------------------------------------------------------- check

status=""
for _ in $(seq 1 20); do
    if status="$("$VENV/bin/python" - <<'PY' 2>/dev/null
import json, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://127.0.0.1:8097/status", timeout=3) as r:
    s = json.load(r)
print(f"mode={s['mode']} keys_configured={s['keys_configured']} running={s['running']} llm_reachable={s['llm']['reachable']}")
PY
)"; then
        break
    fi
    sleep 1
done
if [ -n "$status" ]; then
    say "control API answers on 127.0.0.1:8097: $status"
else
    echo "install.sh: the service did not answer on 127.0.0.1:8097 within 20 s;" \
         "see: systemctl status lo-cryptobot; journalctl -u lo-cryptobot -n 50" >&2
    exit 1
fi
say "done. The trading loop is stopped until something POSTs /start (the agent's crypto_start)."
say "Keys and the live switches live in $ENV_FILE (sudo to edit, then: sudo systemctl restart lo-cryptobot)."
