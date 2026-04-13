"""
Memory Agent
Retrieves historical prediction performance from the database and produces
a structured memory context that other agents can use for confidence calibration.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class MemoryAgent:
    """Recalls historical prediction performance to enable self-calibration."""

    # Minimum number of tracked predictions required before calibration kicks in.
    MIN_PREDICTIONS_FOR_CALIBRATION = 3

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _compute_accuracy(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute directional accuracy and bias from matched prediction-outcome pairs."""
        if not predictions:
            return {}

        correct = 0
        total = 0
        realized_returns: List[float] = []
        regime_buckets: Dict[str, Dict[str, Any]] = {}

        for row in predictions:
            real_ret = row.get("realized_return")
            if real_ret is None:
                continue

            # Determine predicted direction from action or probability_up
            action = (row.get("action") or "hold").lower()
            prob_up = row.get("probability_up")
            if action == "buy":
                pred_direction = 1
            elif action == "sell":
                pred_direction = -1
            elif prob_up is not None:
                pred_direction = 1 if prob_up >= 0.5 else -1
            else:
                continue  # Cannot determine direction

            total += 1
            realized_returns.append(real_ret)

            # Directional accuracy
            real_direction = 1 if real_ret >= 0 else -1
            if pred_direction == real_direction:
                correct += 1

            # Group by regime if available
            regime = row.get("regime", "unknown")
            if regime not in regime_buckets:
                regime_buckets[regime] = {"correct": 0, "total": 0}
            regime_buckets[regime]["total"] += 1
            if pred_direction == real_direction:
                regime_buckets[regime]["correct"] += 1

        if total == 0:
            return {}

        accuracy = correct / total
        avg_realized = sum(realized_returns) / len(realized_returns)

        # Separate wins and losses for Kelly calculation
        wins = [r for r in realized_returns if r > 0]
        losses = [r for r in realized_returns if r < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0  # positive value

        regime_performance = {}
        for regime, stats in regime_buckets.items():
            regime_performance[regime] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "count": stats["total"],
            }

        return {
            "prediction_count": total,
            "directional_accuracy": round(accuracy, 4),
            "avg_realized_return": round(avg_realized, 6),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "regime_performance": regime_performance,
        }

    def _build_last_prediction_info(self, predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract info about the most recent tracked prediction."""
        if not predictions:
            return None
        latest = predictions[-1]
        real_ret = latest.get("realized_return")
        if real_ret is None:
            return None
        action = (latest.get("action") or "hold").lower()
        prob_up = latest.get("probability_up")
        if action == "buy":
            pred_dir = 1
        elif action == "sell":
            pred_dir = -1
        elif prob_up is not None:
            pred_dir = 1 if prob_up >= 0.5 else -1
        else:
            pred_dir = 0
        real_dir = 1 if real_ret >= 0 else -1
        return {
            "date": latest.get("predicted_at", "unknown"),
            "action": latest.get("action", "unknown"),
            "probability_up": prob_up,
            "realized_return": real_ret,
            "correct_direction": pred_dir == real_dir,
        }

    def compute_track_record_factor(self, memory: Dict[str, Any]) -> float:
        """
        Compute a track-record factor for RiskAgent position sizing.

        Returns a float in [0.3, 1.1]:
          - no history       -> 0.7 (conservative default)
          - accuracy < 40%   -> 0.3
          - accuracy 40-60%  -> 0.6
          - accuracy > 60%   -> 1.0
          - accuracy > 75%   -> 1.1
        """
        count = memory.get("prediction_count", 0)
        if count < self.MIN_PREDICTIONS_FOR_CALIBRATION:
            return 0.7

        accuracy = memory.get("directional_accuracy", 0.5)
        if accuracy < 0.40:
            return 0.3
        elif accuracy < 0.60:
            return 0.6
        elif accuracy < 0.75:
            return 1.0
        else:
            return 1.1

    def recall(self, stock_symbol: str, storage: Any) -> Dict[str, Any]:
        """
        Retrieve historical prediction performance for a stock symbol.

        Args:
            stock_symbol: The ticker to look up.
            storage: A Storage instance with query methods.

        Returns:
            A dict with agent metadata, memory stats, calibration factors, and summary.
        """
        ticker = stock_symbol.upper().strip()

        try:
            tracked = storage.get_tracked_predictions(ticker)
        except Exception as exc:
            return {
                "agent": "memory",
                "stock_symbol": ticker,
                "status": "error",
                "memory": {},
                "track_record_factor": 0.7,
                "summary": f"Memory recall failed: {exc}",
            }

        if not tracked:
            return {
                "agent": "memory",
                "stock_symbol": ticker,
                "status": "no_history",
                "memory": {
                    "prediction_count": 0,
                },
                "track_record_factor": 0.7,
                "summary": f"No tracked prediction history for {ticker}. Using conservative defaults.",
            }

        memory = self._compute_accuracy(tracked)
        last_pred = self._build_last_prediction_info(tracked)
        if last_pred:
            memory["last_prediction"] = last_pred

        track_record_factor = self.compute_track_record_factor(memory)

        count = memory.get("prediction_count", 0)
        accuracy = memory.get("directional_accuracy", 0)

        summary = (
            f"{ticker}: {count} tracked predictions, "
            f"{accuracy:.0%} directional accuracy. "
            f"Track record factor: {track_record_factor}."
        )

        return {
            "agent": "memory",
            "stock_symbol": ticker,
            "status": "success",
            "memory": memory,
            "track_record_factor": track_record_factor,
            "summary": summary,
        }
