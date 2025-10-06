#!/usr/bin/env python3
"""Test the sentiment scheduler in isolation."""

import time
import logging
from intelligence import LlamaDecisionEngine
from optimization_engine import RollingSentimentCache, SentimentRollingScheduler
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)


def test_scheduler():
    print("🔍 Testing sentiment scheduler...")

    # Create cache and engine
    cache = RollingSentimentCache()
    engine = LlamaDecisionEngine(
        model_name=config.LLAMA_MODEL_NAME,
        base_url=config.LLAMA_API_BASE_URL,
        request_timeout=30,
    )

    coins = ["BTC", "ETH", "SOL"]

    # Create scheduler
    scheduler = SentimentRollingScheduler(
        llama_engine=engine,
        coins=coins,
        cache=cache,
        ttl_seconds=60,  # Short TTL for testing
        max_qps=0.5,
        batch_size=1,
        build_prompt_fn=getattr(config, "build_sentiment_prompt", None),
    )

    print("✅ Starting scheduler...")
    scheduler.start()

    # Wait for first update
    print("⏳ Waiting 10 seconds for scheduler to populate cache...")
    time.sleep(10)

    # Check cache
    snapshot = cache.snapshot()
    print(f"\n📊 Cache snapshot after 10s:")
    if snapshot:
        for coin, score in snapshot.items():
            print(f"   {coin}: {score:.2f}")
        print("\n✅ SUCCESS! Scheduler is working!")
        scheduler.stop_event.set()
        scheduler.join(timeout=5)
        return True
    else:
        print("\n❌ FAIL: Cache is empty after 10s")
        print("   Possible causes:")
        print("   1. Ollama timeout (increase LLAMA_REQUEST_TIMEOUT)")
        print("   2. Model too slow (try llama3.2:3b-instruct-q4_K_M)")
        print("   3. Scheduler error (check logs above)")
        scheduler.stop_event.set()
        scheduler.join(timeout=5)
        return False


if __name__ == "__main__":
    import sys

    success = test_scheduler()
    sys.exit(0 if success else 1)