"""Portfolio diversification strategy with risk management."""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

import config
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

    def _get_sentiment_scores(self, coins: List[str]) -> Dict[str, float]:
        """Sentiment for every coin: the rolling cache when there is one, else direct LLM calls.

        A coin with no score yet gets 0.0 (neutral) so it is still judged on its trend; the
        signals record says which scores were real.
        """
        coins = [c for c in coins if c.upper() != self.base_currency]
        cache = getattr(self, "sentiment_cache", None)
        if cache is None:
            return self._get_sentiment(coins)
        ttl = getattr(config, "SENTIMENT_TTL_SECONDS", 900)
        scores: Dict[str, float] = {}
        for coin in coins:
            value = cache.get_fresh(coin, ttl)
            if value is None:
                # Stale beats nothing: the scheduler refreshes stalest-first, so an old score
                # is about to be replaced, and it is still the model's last word on the coin.
                value = cache.get(coin)
            scores[coin] = float(value) if value is not None else 0.0
        return scores

    def decide_trades(self, balances: Dict[str, Dict[str, float]], **kwargs) -> List[TradeInstruction]:
        """Trading logic with portfolio allocation and risk management."""
        if not self.coins:
            return []

        # Get sentiment and trend scores
        sentiment_scores = self._get_sentiment_scores(self.coins)
        if not sentiment_scores:
            self.logger.info("No sentiment data available")
            return []

        # Trend scores, one coin after another: ccxt's synchronous client and its rate limiter
        # are not meant to be shared across threads, and the 60 s trend cache makes most of
        # these lookups free anyway.
        trend_scores: Dict[str, Optional[float]] = {}
        for coin in sentiment_scores.keys():
            try:
                trend_scores[coin] = self._multi_timeframe_trend(coin)
            except Exception:
                trend_scores[coin] = None

        # Compute hybrid scores
        hybrid_scores: Dict[str, float] = {}
        for coin, sentiment in sentiment_scores.items():
            trend = trend_scores.get(coin)
            hybrid = self._compute_hybrid_score(coin, sentiment, trend)
            hybrid_scores[coin] = hybrid
        self._record_signals(list(sentiment_scores.keys()), sentiment_scores, trend_scores, hybrid_scores)

        # Log scores
        coin_summaries = []
        for coin in self.coins:
            if coin in hybrid_scores:
                trend = trend_scores.get(coin) or 0.0
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
                entry = self.entry_prices.get(coin)
                drop = f" ({(current_price - entry) / entry:+.1%} from entry)" if entry else ""
                self._execute_sell(coin, held, f"stop-loss{drop}", trades)
                self._cleanup_position(coin)
                continue

            # Check time exit
            if self._check_time_exit(coin):
                self._execute_sell(coin, held, f"time-limit (held {self.position_max_days}+ days)", trades)
                self._cleanup_position(coin)
                continue

            # Regular sell signal
            hybrid = hybrid_scores.get(coin, 0.0)
            if hybrid <= self.sell_threshold and self._can_trade(coin):
                self._execute_sell(coin, held, f"hybrid={hybrid:+.2f} <= sell threshold {self.sell_threshold:+.2f}", trades)
                self._cleanup_position(coin)

        # BUY logic with portfolio allocation. Trades are returned in simulation too:
        # TradingBot is what declines to place them, and returning nothing here made a
        # simulating bot indistinguishable from one that never wanted to trade.
        if base_balance < 10:
            return trades

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
            reason = f"hybrid={top_score:+.2f} >= buy threshold {self.buy_threshold:+.2f} (core)"

            trend = trend_scores.get(top_coin)
            if self.require_trend_confirmation and trend is not None and trend < 0:
                self.logger.info(f"Skipping {top_coin}: negative trend")
            else:
                # (Without trend confirmation this used to buy nothing at all.)
                self._execute_buy_with_tracking(top_coin, core_funds, trades, reason=reason)

        # Buy best opportunistic coin (smaller position)
        if opportunistic_candidates:
            opportunistic_candidates.sort(key=lambda x: x[1], reverse=True)
            top_coin, top_score = opportunistic_candidates[0]

            # More aggressive for opportunistic (always check trend)
            trend = trend_scores.get(top_coin)
            if trend is not None and trend < 0:
                self.logger.info(f"Skipping {top_coin}: negative trend (opportunistic)")
            else:
                self._execute_buy_with_tracking(
                    top_coin, opportunistic_funds, trades,
                    reason=f"hybrid={top_score:+.2f} >= buy threshold {self.buy_threshold:+.2f} (opportunistic)",
                )

        return trades

    # _execute_buy_with_tracking and _cleanup_position live in LlamaHybridStrategy. The copy
    # that was here shadowed `trades` with a fresh local list, so every buy it queued was
    # thrown away.