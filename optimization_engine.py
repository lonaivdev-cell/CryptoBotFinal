"""
Optimized sentiment scheduling engine for uniform GPU usage.

Key improvements:
1. Request spreading - Distributes coin updates evenly across TTL window
2. Adaptive rate limiting - Accounts for actual inference time
3. Priority queue - Updates stalest coins first
4. Jittered timing - Prevents request clustering
5. Backpressure handling - Adapts to slow GPU responses
6. Pure background refresh - No blocking calls, only cached returns
"""
import threading
import time
import random
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging


class RollingSentimentCache:
    """Thread-safe store of {coin: (score, ts)} with staleness tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, float]] = {}  # coin -> {"score": float, "ts": float}

    def set(self, coin: str, score: float, ts: Optional[float] = None):
        """Store score with timestamp."""
        with self._lock:
            self._data[coin.upper()] = {
                "score": float(score),
                "ts": ts if ts is not None else time.time()
            }

    def get(self, coin: str) -> Optional[float]:
        """Get score regardless of age."""
        with self._lock:
            row = self._data.get(coin.upper())
            return None if not row else float(row["score"])

    def get_with_age(self, coin: str) -> Optional[Tuple[float, float]]:
        """Get score and age together. Returns (score, age_seconds) or None."""
        with self._lock:
            row = self._data.get(coin.upper())
            if not row:
                return None
            age = time.time() - float(row["ts"])
            return (float(row["score"]), age)

    def get_fresh(self, coin: str, ttl_seconds: int) -> Optional[float]:
        """Get score only if within TTL."""
        with self._lock:
            row = self._data.get(coin.upper())
            if not row:
                return None
            age = time.time() - float(row["ts"])
            if age > ttl_seconds:
                return None
            return float(row["score"])

    def get_age(self, coin: str) -> Optional[float]:
        """Get age of cached score in seconds."""
        with self._lock:
            row = self._data.get(coin.upper())
            if not row:
                return None
            return time.time() - float(row["ts"])

    def snapshot(self) -> Dict[str, float]:
        """Get all current scores."""
        with self._lock:
            return {k: float(v["score"]) for k, v in self._data.items()}

    def get_staleness_map(self) -> Dict[str, float]:
        """Get age of each coin's cache."""
        with self._lock:
            now = time.time()
            return {k: now - v["ts"] for k, v in self._data.items()}


class SentimentRollingScheduler(threading.Thread):
    """
    Optimized scheduler that spreads LLaMA requests uniformly over time.

    Key features:
    - Priority queue: Updates stalest coins first
    - Adaptive timing: Accounts for actual GPU inference time
    - Request spreading: Evenly distributes updates across TTL window
    - Jitter: Prevents request clustering
    - Backpressure: Slows down if GPU is overloaded
    """

    def __init__(
            self,
            llama_engine,
            coins: List[str],
            cache: RollingSentimentCache,
            ttl_seconds: int = 900,
            max_qps: float = 0.25,
            batch_size: int = 3,  # Increased from 1 to 3 for better efficiency
            rotate_order: str = "staleness",  # "staleness", "round_robin", or "volatility"
            stop_event: Optional[threading.Event] = None,
            build_prompt_fn=None,
            enable_jitter: bool = True,
            adaptive_rate: bool = True,
    ):
        super().__init__(daemon=True)
        self.log = logging.getLogger(self.__class__.__name__)
        self.llama = llama_engine
        self.coins = [c.upper() for c in coins]
        self.cache = cache
        self.ttl = ttl_seconds
        self.target_qps = max_qps
        self.batch = max(1, batch_size)
        self.rotate = rotate_order
        self.stop_event = stop_event or threading.Event()
        self.build_prompt_fn = build_prompt_fn
        self.enable_jitter = enable_jitter
        self.adaptive_rate = adaptive_rate

        # Timing control
        self.base_sleep = 1.0 / max_qps if max_qps > 0 else 4.0
        self.current_sleep = self.base_sleep

        # Request spreading: Aim to refresh each coin once per TTL
        self.refresh_interval = ttl_seconds / len(coins) if coins else ttl_seconds

        # Performance tracking
        self._recent_durations = deque(maxlen=10)  # Last 10 inference times
        self._request_times = deque(maxlen=20)  # Last 20 request timestamps

        # Round-robin state
        self._idx = 0

        # Initialize cache with placeholder scores to spread initial updates
        self._initialize_staggered_cache()

    def _initialize_staggered_cache(self):
        """Initialize cache with staggered timestamps to spread first refresh cycle."""
        if not self.coins:
            return

        now = time.time()
        stagger_interval = self.ttl / len(self.coins)

        for i, coin in enumerate(self.coins):
            # Set initial timestamp so coins expire at different times
            initial_ts = now - (self.ttl - (i * stagger_interval))
            self.cache.set(coin, 0.0, initial_ts)

        self.log.info(
            f"Initialized {len(self.coins)} coins with staggered refresh (spread over {self.ttl}s)"
        )

    def _calculate_adaptive_sleep(self) -> float:
        """Calculate sleep time based on recent inference durations."""
        if not self.adaptive_rate or not self._recent_durations:
            return self.current_sleep

        # Average recent inference time
        avg_duration = sum(self._recent_durations) / len(self._recent_durations)

        # Target: maintain target_qps accounting for inference time
        # sleep = (1/target_qps) - inference_time
        ideal_sleep = max(0.1, self.base_sleep - avg_duration)

        # Smooth adaptation (exponential moving average)
        alpha = 0.3
        self.current_sleep = (alpha * ideal_sleep) + ((1 - alpha) * self.current_sleep)

        return self.current_sleep

    def _add_jitter(self, sleep_time: float) -> float:
        """Add random jitter to prevent request clustering."""
        if not self.enable_jitter:
            return sleep_time

        # Add ±20% jitter
        jitter_range = sleep_time * 0.2
        jitter = random.uniform(-jitter_range, jitter_range)
        return max(0.1, sleep_time + jitter)

    def _check_backpressure(self) -> bool:
        """Detect if GPU is overloaded based on request timing."""
        if len(self._request_times) < 5:
            return False

        # Check if recent requests are taking too long
        recent = list(self._request_times)[-5:]
        intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        # If actual interval > 2x target, we're backed up
        target_interval = 1.0 / self.target_qps
        if avg_interval > target_interval * 2.0:
            self.log.warning(
                f"Backpressure detected: avg interval {avg_interval:.2f}s "
                f"vs target {target_interval:.2f}s"
            )
            return True

        return False

    def _select_next_coins_by_staleness(self, max_coins: int) -> List[str]:
        """Select coins with oldest cache entries (priority queue approach)."""
        staleness = self.cache.get_staleness_map()

        # Add coins not in cache yet
        for coin in self.coins:
            if coin not in staleness:
                staleness[coin] = float('inf')

        # Sort by staleness (oldest first)
        sorted_coins = sorted(staleness.items(), key=lambda x: x[1], reverse=True)

        # Return top N stalest coins
        return [coin for coin, _ in sorted_coins[:max_coins]]

    def _select_next_coins_round_robin(self, max_coins: int) -> List[str]:
        """Simple round-robin selection."""
        if not self.coins:
            return []

        n = len(self.coins)
        start = self._idx
        end = min(start + max_coins, n)
        batch = self.coins[start:end]
        self._idx = 0 if end >= n else end

        return batch

    def _select_next_batch(self) -> List[str]:
        """Select next coins to process based on rotation strategy."""
        if self.rotate == "staleness":
            candidates = self._select_next_coins_by_staleness(self.batch)
        else:  # round_robin (or volatility if you implement it)
            candidates = self._select_next_coins_round_robin(self.batch)

        # Only return coins that need refresh (age > 80% of TTL)
        refresh_threshold = self.ttl * 0.8
        needs_refresh = []

        for coin in candidates:
            age = self.cache.get_age(coin)
            if age is None or age > refresh_threshold:
                needs_refresh.append(coin)

        return needs_refresh[:self.batch]  # Enforce batch limit

    def _build_prompt(self, batch: List[str]) -> str:
        """Build sentiment prompt for given coins."""
        if self.build_prompt_fn:
            return self.build_prompt_fn(batch)

        # Minimal strict prompt for per-coin lines (keeps outputs tiny)
        lines = "\n".join(batch)
        return (
            "Return EXACTLY one line per ticker in 'TICKER: SCORE' form, "
            "SCORE in [-1,1] with 2 decimals. No extra text.\n\n"
            f"Tickers:\n{lines}\n"
        )

    def _parse_lines(self, text: str) -> Dict[str, float]:
        """Parse LLaMA response into {coin: score} dict."""
        out: Dict[str, float] = {}
        if not text:
            return out

        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            sym = parts[0].strip().upper()
            try:
                score = float(parts[1].strip())
                # Clamp to [-1, 1]
                score = max(-1.0, min(1.0, score))
                out[sym] = score
            except (ValueError, AttributeError):
                continue

        return out

    def run(self):
        """Main scheduler loop with optimized timing."""
        self.log.info(
            f"Sentiment scheduler started: {len(self.coins)} coins, "
            f"target QPS={self.target_qps:.2f}, batch={self.batch}, "
            f"TTL={self.ttl}s, strategy={self.rotate}"
        )

        consecutive_errors = 0
        max_errors = 5

        while not self.stop_event.is_set():
            try:
                # 1) Check for backpressure
                if self._check_backpressure():
                    # Slow down temporarily
                    self.log.info("Applying backpressure: sleeping 5s")
                    time.sleep(5.0)
                    continue

                # 2) Select coins to process
                targets = self._select_next_batch()

                # 3) If nothing needs refresh, idle briefly
                if not targets:
                    time.sleep(1.0)
                    continue

                # 4) Process batch
                start_time = time.time()
                prompt = self._build_prompt(targets)

                # Call LLaMA (blocks on GPU)
                raw_response = self.llama.generate(prompt)

                inference_duration = time.time() - start_time
                self._recent_durations.append(inference_duration)
                self._request_times.append(time.time())

                # 5) Parse and cache results
                scores = self._parse_lines(raw_response)

                cached_count = 0
                for coin in targets:
                    if coin in scores:
                        self.cache.set(coin, scores[coin])
                        cached_count += 1

                if cached_count > 0:
                    # Log with age information for visibility
                    coins_with_age = []
                    for c in targets:
                        if c in scores:
                            age_info = self.cache.get_age(c)
                            age_str = f"{age_info:.0f}s" if age_info else "new"
                            coins_with_age.append(f"{c}:{scores[c]:.2f}({age_str})")

                    self.log.debug(
                        f"Updated {cached_count}/{len(targets)} coins in {inference_duration:.2f}s: "
                        f"{', '.join(coins_with_age)}"
                    )

                # Reset error counter on success
                consecutive_errors = 0

                # 6) Adaptive sleep with jitter
                sleep_time = self._calculate_adaptive_sleep()
                sleep_time = self._add_jitter(sleep_time)

                time.sleep(sleep_time)

            except Exception as e:
                consecutive_errors += 1
                self.log.error(f"Scheduler error: {e}", exc_info=False)

                # Exponential backoff on repeated errors
                backoff = min(30.0, 2.0 ** consecutive_errors)
                self.log.warning(f"Backing off {backoff:.1f}s after error #{consecutive_errors}")
                time.sleep(backoff)

                if consecutive_errors >= max_errors:
                    self.log.critical("Too many consecutive errors, stopping scheduler")
                    break

        self.log.info("Sentiment scheduler stopped")

    def get_stats(self) -> Dict[str, any]:
        """Get current scheduler statistics."""
        staleness = self.cache.get_staleness_map()

        return {
            "coins_tracked": len(self.coins),
            "coins_cached": len(staleness),
            "current_sleep": self.current_sleep,
            "avg_inference_time": (
                sum(self._recent_durations) / len(self._recent_durations)
                if self._recent_durations else 0.0
            ),
            "stalest_coin_age": max(staleness.values()) if staleness else 0,
            "freshest_coin_age": min(staleness.values()) if staleness else 0,
            "recent_request_count": len(self._request_times),
            "is_running": self.is_alive(),
        }

    def get_scores_with_ages(self) -> Dict[str, Tuple[float, float]]:
        """Get all scores with their ages for monitoring/debugging."""
        result = {}
        for coin in self.coins:
            data = self.cache.get_with_age(coin)
            if data:
                result[coin] = data
        return result

    def get_all_scores_with_age(self) -> Dict[str, Tuple[float, float]]:
        """Get all scores with their ages for monitoring. Returns {coin: (score, age_seconds)}."""
        staleness = self.cache.get_staleness_map()
        snapshot = self.cache.snapshot()

        result = {}
        for coin in snapshot:
            score = snapshot[coin]
            age = staleness.get(coin, 0.0)
            result[coin] = (score, age)

        return result

    def request_graceful_stop(self):
        """Signal scheduler to stop gracefully after current batch."""
        self.log.info("Graceful stop requested for sentiment scheduler")
        self.stop_event.set()


class SentimentService:
    """
    Service that provides sentiment scores from cache only.
    The background scheduler keeps the cache fresh.
    """

    def __init__(self, cache: RollingSentimentCache, scheduler: Optional[SentimentRollingScheduler] = None):
        self.cache = cache
        self.scheduler = scheduler
        self.log = logging.getLogger(self.__class__.__name__)

    def get_scores(self, coins: List[str], default: float = 0.0) -> Dict[str, float]:
        """
        Get sentiment scores for given coins from cache ONLY.

        This method NEVER blocks on AI calls. It returns whatever is in cache.
        If a coin isn't cached, it returns the default value.
        The background scheduler handles refreshing stale entries.

        Args:
            coins: List of coin symbols to get scores for
            default: Default score to return if coin not in cache

        Returns:
            Dict mapping coin symbols to sentiment scores [-1, 1]
        """
        result = {}

        for coin in coins:
            score = self.cache.get(coin.upper())
            if score is not None:
                result[coin.upper()] = score
            else:
                # Not in cache yet - return default
                result[coin.upper()] = default
                self.log.debug(f"Coin {coin} not yet cached, returning default {default}")

        return result

    def get_score(self, coin: str, default: float = 0.0) -> float:
        """
        Get sentiment score for a single coin from cache ONLY.

        Args:
            coin: Coin symbol
            default: Default score if not in cache

        Returns:
            Sentiment score [-1, 1] or default
        """
        score = self.cache.get(coin.upper())
        return score if score is not None else default

    def get_scores_with_staleness(self, coins: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get scores with staleness information for monitoring.

        Returns:
            Dict mapping coins to {"score": float, "age_seconds": float}
        """
        result = {}

        for coin in coins:
            data = self.cache.get_with_age(coin.upper())
            if data:
                score, age = data
                result[coin.upper()] = {"score": score, "age_seconds": age}

        return result

    def get_scheduler_stats(self) -> Optional[Dict]:
        """Get statistics from the background scheduler if available."""
        if self.scheduler:
            return self.scheduler.get_stats()
        return None

    def format_scores_with_age(self, coins: List[str]) -> str:
        """
        Format scores with age information for logging/display.
        Example: "BTC: 0.75 (45s), ETH: -0.20 (120s)"
        """
        staleness_data = self.get_scores_with_staleness(coins)

        parts = []
        for coin in coins:
            coin_upper = coin.upper()
            if coin_upper in staleness_data:
                data = staleness_data[coin_upper]
                score = data["score"]
                age = data["age_seconds"]
                parts.append(f"{coin_upper}: {score:+.2f} ({age:.0f}s)")
            else:
                parts.append(f"{coin_upper}: N/A")

        return ", ".join(parts)