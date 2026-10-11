"""The paper wallet simulated trades fill against.

A simulating bot that reads its real Binance balance (or no balance at all, with placeholder
keys) decides the same trade every cycle and never sees the consequence of one: its buys never
arrive, so its sells, stop-losses and time exits never fire. The headless service therefore
simulates against a wallet of its own: it starts with CRYPTOBOT_PAPER_USDT of the base currency,
each simulated decision fills at the price the strategy saw (minus the configured taker fee),
and the result is saved under the state directory so a restart picks up where it left off.

The wallet is never consulted in live mode, and PaperExchange has no order path at all.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("PaperWallet")

# Below this a holding is rounding dust from fee arithmetic, not a position.
DUST = 1e-12


class PaperWallet:
    def __init__(
        self,
        base_currency: str = "USDT",
        start_balance: float = 1000.0,
        fee_pct: float = 0.001,
        path: Optional[str] = None,
    ) -> None:
        self.base = base_currency.upper()
        self.start_balance = float(start_balance)
        self.fee_pct = max(0.0, float(fee_pct))
        self.path = path
        self._lock = threading.RLock()
        self.balances: Dict[str, float] = {self.base: self.start_balance}
        self.avg_cost: Dict[str, float] = {}
        self.opened_at: Dict[str, int] = {}  # epoch ms of the first buy of the current holding
        self.marks: Dict[str, float] = {}  # last seen price per coin, for valuation
        self.created_at = int(time.time() * 1000)
        if path:
            self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:  # type: ignore[arg-type]
                data = json.load(fh)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("Paper wallet at %s is unreadable (%s); starting a fresh one", self.path, exc)
            return
        if not isinstance(data, dict) or str(data.get("base", "")).upper() != self.base:
            log.warning("Paper wallet at %s is for another base currency; starting a fresh one", self.path)
            return

        def floats(key: str) -> Dict[str, float]:
            raw = data.get(key)
            out: Dict[str, float] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v >= 0:
                        out[str(k).upper()] = float(v)
            return out

        balances = floats("balances")
        if self.base not in balances:
            log.warning("Paper wallet at %s has no %s balance; starting a fresh one", self.path, self.base)
            return
        self.balances = balances
        self.avg_cost = floats("avg_cost")
        self.marks = floats("marks")
        self.opened_at = {k: int(v) for k, v in floats("opened_at").items()}
        start = data.get("start_balance")
        if isinstance(start, (int, float)) and not isinstance(start, bool) and start > 0:
            self.start_balance = float(start)
        created = data.get("created_at")
        if isinstance(created, int) and created > 0:
            self.created_at = created

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {
                "base": self.base,
                "start_balance": self.start_balance,
                "created_at": self.created_at,
                "balances": self.balances,
                "avg_cost": self.avg_cost,
                "opened_at": self.opened_at,
                "marks": self.marks,
            }
        directory = os.path.dirname(os.path.abspath(self.path))
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".paper.", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not save the paper wallet to %s: %s", self.path, exc)

    # ---------- the exchange-shaped view ----------
    def as_exchange_balances(self) -> Dict[str, Dict[str, float]]:
        """Balances in ExchangeWrapper.get_balances() shape; the base currency always present."""
        with self._lock:
            out = {
                asset: {"free": amount, "used": 0.0, "total": amount}
                for asset, amount in self.balances.items()
                if amount > DUST
            }
            out.setdefault(self.base, {"free": 0.0, "used": 0.0, "total": 0.0})
            return out

    def holdings(self) -> Dict[str, Tuple[float, Optional[float], Optional[int]]]:
        """{coin: (amount, average cost, opened at ms)} for every non-base holding."""
        with self._lock:
            return {
                c: (a, self.avg_cost.get(c), self.opened_at.get(c))
                for c, a in self.balances.items()
                if c != self.base and a > DUST
            }

    def mark(self, coin: str, price: Optional[float]) -> None:
        if price and price > 0:
            with self._lock:
                self.marks[coin.upper()] = float(price)

    # ---------- fills ----------
    def fill(self, side: str, coin: str, amount: float, price: Optional[float]) -> Tuple[float, str]:
        """Apply one simulated market order. Returns (amount filled, note for the decision)."""
        coin = coin.upper()
        if not price or price <= 0:
            return 0.0, "not filled: no price"
        if amount <= 0:
            return 0.0, "not filled: zero amount"
        with self._lock:
            self.marks[coin] = float(price)
            if side == "buy":
                cash = self.balances.get(self.base, 0.0)
                cost = amount * price * (1.0 + self.fee_pct)
                note = "paper fill"
                if cost > cash:
                    amount = cash / (price * (1.0 + self.fee_pct))
                    cost = cash
                    note = "paper fill, reduced to the cash available"
                if amount <= DUST:
                    return 0.0, f"not filled: no {self.base} left in the paper wallet"
                held = self.balances.get(coin, 0.0)
                prev_cost = self.avg_cost.get(coin, price)
                self.avg_cost[coin] = (held * prev_cost + amount * price) / (held + amount)
                self.balances[coin] = held + amount
                self.balances[self.base] = max(0.0, cash - cost)
                self.opened_at.setdefault(coin, int(time.time() * 1000))
            elif side == "sell":
                held = self.balances.get(coin, 0.0)
                if held <= DUST:
                    return 0.0, f"not filled: the paper wallet holds no {coin}"
                note = "paper fill"
                if amount > held:
                    amount = held
                    note = "paper fill, reduced to the amount held"
                proceeds = amount * price * (1.0 - self.fee_pct)
                left = held - amount
                if left <= DUST:
                    self.balances.pop(coin, None)
                    self.avg_cost.pop(coin, None)
                    self.opened_at.pop(coin, None)
                else:
                    self.balances[coin] = left
                self.balances[self.base] = self.balances.get(self.base, 0.0) + proceeds
            else:
                return 0.0, f"not filled: unknown side {side!r}"
        self.save()
        return amount, note

    # ---------- reporting ----------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            balances = {a: v for a, v in self.balances.items() if v > DUST or a == self.base}
            value: Optional[float] = balances.get(self.base, 0.0)
            unpriced = []
            for coin, amount in balances.items():
                if coin == self.base:
                    continue
                px = self.marks.get(coin)
                if px is None:
                    unpriced.append(coin)
                else:
                    value += amount * px  # type: ignore[operator]
            if unpriced:
                value = None
            return {
                "base": self.base,
                "balances": balances,
                "avg_cost": {c: self.avg_cost[c] for c in balances if c in self.avg_cost},
                "marks": {c: self.marks[c] for c in balances if c in self.marks},
                "value": round(value, 8) if value is not None else None,
                "start_balance": self.start_balance,
                "pnl": round(value - self.start_balance, 8) if value is not None else None,
                "fee_pct": self.fee_pct,
                "created_at": self.created_at,
            }


class PaperExchange:
    """ExchangeWrapper-shaped view for a simulating strategy: market data from the real
    exchange, balances from the paper wallet, and no way to place an order."""

    def __init__(self, market: Any, wallet: PaperWallet) -> None:
        self.market = market
        self.wallet = wallet

    @property
    def client(self) -> Any:
        return self.market.client

    def connect(self) -> None:
        self.market.connect()

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.market.get_ticker(symbol)

    def get_balances(self) -> Dict[str, Dict[str, float]]:
        return self.wallet.as_exchange_balances()

    def create_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("The bot is simulating: orders are filled in the paper wallet, never sent to the exchange.")

    def cancel_all_orders(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("The bot is simulating: there are no exchange orders to cancel.")
