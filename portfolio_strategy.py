"""Portfolio diversification strategy with risk management."""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

from intelligence import LlamaHybridStrategy, TradeInstruction


class DiversifiedPortfolioStrategy(LlamaHybridStrategy):
    """Hybrid strategy with portfolio allocation and risk management."""

    # Core portfolio (90-95% of funds)
    CORE_COINS = ["BTC", "ETH", "XRP", "BNB", "SOL", "DOGE"]
    CORE_ALLOCATION = 0.90  # 90% to core

    # Opportunistic coins (5-10% of funds)
    OPPORTUNISTIC_COINS = []  # Set dynamically or via config
    OPPORTUNISTIC_ALLOCATION = 0.10  # 10% to high-risk

    def __init__(self, *args, **kwargs):
        # Extract portfolio-specific params
        self.stop_loss_pct = kwargs.pop('stop_loss_pct', 0.20)  # 20% loss triggers sell
        self.position_max_days = kwargs.pop('position_max_days', 14)  # Exit after 14 days
        self.enable_stop_loss = kwargs.pop('enable_stop_loss', True)
        self.enable_time_exit = kwargs.pop('enable_time_exit', True)

        super().__init__(*args, **kwargs)

        # Track entry prices and times for stop-loss
        self.entry_prices: Dict[str, float] = {}
        self.entry_times: Dict[str, datetime] = {}

        # Separate core and opportunistic coins
        self.core_coins = [c for c in self.coins if c in self.CORE_COINS]
        self.opportunistic_coins = [c for c in self.coins if c not in self.CORE_COINS]

        self.logger.info(
            f"Portfolio strategy: {len(self.core_coins)} core coins ({self.CORE_ALLOCATION:.0%}), "
            f"{len(self.opportunistic_coins)} opportunistic ({self.OPPORTUNISTIC_ALLOCATION:.0%})"
        )

    def _check_stop_loss(self, coin: str, current_price: float) -> bool:
        """Check if position should be stopped out."""
        if not self.enable_stop_loss or coin not in self.entry_prices:
            return False

        entry_price = self.entry_prices[coin]
        loss_pct = (current_price - entry_price) / entry_price

        if loss_pct <= -self.stop_loss_pct:
            self.logger.warning(
                f"STOP-LOSS triggered for {coin}: "
                f"entry=${entry_price:.4f}, current=${current_price:.4f}, "
                f"loss={loss_pct:.1%}"
            )
            return True

        return False

    def _check_time_exit(self, coin: str) -> bool:
        """Check if position should be exited due to time limit."""
        if not self.enable_time_exit or coin not in self.entry_times:
            return False

        # Only apply time exits to opportunistic coins
        if coin in self.core_coins:
            return False

        entry_time = self.entry_times[coin]
        days_held = (datetime.now(timezone.utc) - entry_time).days

        if days_held >= self.position_max_days:
            self.logger.info(
                f"TIME EXIT for {coin}: held {days_held} days (max {self.position_max_days})"
            )
            return True

        return False

    def decide_trades(self, balances: Dict[str, Dict[str, float]], **kwargs) -> List[TradeInstruction]:
        """Trading logic with portfolio allocation and risk management."""
        if not self.coins:
            return []

        # Get sentiment and trend scores
        sentiment_scores = self._get_sentiment_scores(self.coins)
        if not sentiment_scores:
            self.logger.info("No sentiment data available")
            return []

        # Get trend scores
        from concurrent.futures import as_completed
        futures = {
            self.executor.submit(self._get_trend_score, coin): coin
            for coin in sentiment_scores.keys()
        }
        trend_scores: Dict[str, Optional[float]] = {}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                trend_scores[coin] = fut.result()
            except Exception:
                trend_scores[coin] = None

        # Compute hybrid scores
        hybrid_scores: Dict[str, float] = {}
        for coin, sentiment in sentiment_scores.items():
            trend = trend_scores.get(coin)
            hybrid = self._compute_hybrid_score(coin, sentiment, trend)
            hybrid_scores[coin] = hybrid

        # Log scores
        coin_summaries = []
        for coin in self.coins:
            if coin in hybrid_scores:
                trend = trend_scores.get(coin, 0.0)
                coin_summaries.append(f"{coin} {trend:+.2f}")
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
            return trades if not self.simulate_only else []

        # Separate core and opportunistic candidates
        core_candidates = [
            (coin, score) for coin, score in hybrid_scores.items()
            if coin in self.core_coins and score >= self.buy_threshold and self._can_trade(coin)
        ]

        opportunistic_candidates = [
            (coin, score) for coin, score in hybrid_scores.items()
            if coin in self.opportunistic_coins and score >= self.buy_threshold and self._can_trade(coin)
        ]

        # Allocate funds
        core_funds = base_balance * self.CORE_ALLOCATION
        opportunistic_funds = base_balance * self.OPPORTUNISTIC_ALLOCATION

        # Buy best core coin
        if core_candidates:
            core_candidates.sort(key=lambda x: x[1], reverse=True)
            top_coin, top_score = core_candidates[0]

            if self.require_trend_confirmation:
                trend = trend_scores.get(top_coin)
                if trend is not None and trend < 0:
                    self.logger.info(f"Skipping {top_coin}: negative trend")
                else:
                    self._execute_buy_with_tracking(top_coin, core_funds, trades)

        # Buy best opportunistic coin (smaller position)
        if opportunistic_candidates:
            opportunistic_candidates.sort(key=lambda x: x[1], reverse=True)
            top_coin, top_score = opportunistic_candidates[0]

            # More aggressive for opportunistic (always check trend)
            trend = trend_scores.get(top_coin)
            if trend is not None and trend < 0:
                self.logger.info(f"Skipping {top_coin}: negative trend (opportunistic)")
            else:
                self._execute_buy_with_tracking(top_coin, opportunistic_funds, trades)

        return trades if not self.simulate_only else []

    def _execute_buy_with_tracking(self, coin: str, amount: float, trades: List[TradeInstruction]) -> None:
        """Execute buy and track entry price/time."""
        trades: list[TradeInstruction] = []

        current_price = self._get_current_price(coin)
        if not current_price:
            return

        min_trade = getattr(__import__('config'), "MIN_TRADE_USD", 10.0)
        if amount < min_trade:
            return

        success = self._execute_buy(coin, amount, current_price, trades)
        if success:
            # Track entry for risk management
            self.entry_prices[coin] = current_price
            self.entry_times[coin] = datetime.now(timezone.utc)

            self.positions[coin] = {
                "entry_price": current_price,
                "amount": amount / current_price,
                "entry_time": self.entry_times[coin].isoformat()
            }
            self.last_trade_time[coin] = time.time()

            coin_type = "CORE" if coin in self.core_coins else "OPPORTUNISTIC"
            self.logger.info(f"[{coin_type}] Bought {coin} at ${current_price:.4f}")

    def _cleanup_position(self, coin: str) -> None:
        """Clean up tracking data after exit."""
        if coin in self.positions:
            del self.positions[coin]
        if coin in self.entry_prices:
            del self.entry_prices[coin]
        if coin in self.entry_times:
            del self.entry_times[coin]