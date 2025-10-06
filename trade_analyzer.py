import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class TradeAnalyzer:
    """Analyzes historical trades to identify successful patterns."""

    def __init__(self, history_dir: str):
        self.history_dir = Path(history_dir)
        self.successful_patterns = []
        self.failed_patterns = []

    def analyze_history(self) -> Dict[str, any]:
        """Parse all history files and extract trade outcomes."""
        trades = []

        for file in self.history_dir.glob("run_*.jsonl"):
            with open(file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'orders' in record and record['orders']:
                            trades.append(record)
                    except:
                        continue

        if not trades:
            return {"total_trades": 0, "insights": []}

        # Analyze profitability
        profitable = []
        losses = []

        for i in range(len(trades) - 1):
            current = trades[i]
            next_trade = trades[i + 1]

            # Extract balances before/after
            profit = self._calculate_profit(current, next_trade)

            if profit > 0:
                profitable.append((current, profit))
            else:
                losses.append((current, profit))

        insights = self._extract_insights(profitable, losses)

        return {
            "total_trades": len(trades),
            "profitable": len(profitable),
            "losses": len(losses),
            "win_rate": len(profitable) / len(trades) if trades else 0,
            "insights": insights
        }

    def _calculate_profit(self, before: dict, after: dict) -> float:
        """Calculate USD profit between two trade records."""
        # Sum total USD value before and after
        before_usd = sum(
            bal.get('total', 0) * bal.get('usd_rate', 1)
            for bal in before.get('balances', {}).values()
        )
        after_usd = sum(
            bal.get('total', 0) * bal.get('usd_rate', 1)
            for bal in after.get('balances', {}).values()
        )
        return after_usd - before_usd

    def _extract_insights(self, profitable: list, losses: list) -> List[str]:
        """Extract actionable patterns from trade outcomes."""
        insights = []

        # Analyze profitable trades
        if profitable:
            avg_profit = sum(p[1] for p in profitable) / len(profitable)
            insights.append(f"Average profit per winning trade: ${avg_profit:.2f}")

            # Find common patterns in profitable trades
            # (You'd extract sentiment/trend scores from the records here)

        if losses:
            avg_loss = sum(abs(l[1]) for l in losses) / len(losses)
            insights.append(f"Average loss per losing trade: ${avg_loss:.2f}")

        return insights