"""
Multi-Dimensional Backtest Evaluator

Computes comprehensive evaluation metrics from backtest results:
- Overall performance metrics (17+ indicators)
- Per-regime breakdown (directly informs regime_risk_budget.json tuning)
- Rolling window stability analysis
- Signal quality analysis
- Exit analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .engine import BacktestResult


@dataclass
class BacktestReport:
    """Complete backtest evaluation report."""

    overall: Dict[str, Any]
    per_regime: Dict[str, Dict[str, Any]]
    rolling: Dict[str, Any]
    signal_quality: Dict[str, Any]
    exit_analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "per_regime": self.per_regime,
            "rolling": self.rolling,
            "signal_quality": self.signal_quality,
            "exit_analysis": self.exit_analysis,
        }

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        o = self.overall
        return (
            f"Backtest Report: {o.get('total_trades', 0)} trades, "
            f"return={o.get('total_return', 0):.2%}, "
            f"Sharpe={o.get('sharpe_ratio', 0):.2f}, "
            f"Sortino={o.get('sortino_ratio', 0):.2f}, "
            f"MaxDD={o.get('max_drawdown', 0):.2%}, "
            f"hit_rate={o.get('hit_rate', 0):.2%}, "
            f"alpha={o.get('alpha', 0):.2%}"
        )


class BacktestEvaluator:
    """Computes multi-dimensional evaluation metrics from backtest results."""

    def evaluate(self, result: BacktestResult) -> BacktestReport:
        """Run full evaluation suite."""
        return BacktestReport(
            overall=self._compute_overall_metrics(result),
            per_regime=self._compute_per_regime_metrics(result),
            rolling=self._compute_rolling_metrics(result),
            signal_quality=self._compute_signal_quality(result),
            exit_analysis=self._compute_exit_analysis(result),
        )

    # ── Overall Metrics ──────────────────────────────────────────────────

    def _compute_overall_metrics(self, result: BacktestResult) -> Dict[str, Any]:
        """Compute aggregate performance metrics."""
        trade_log = result.trade_log
        equity = result.equity_curve

        # Filter executed trades (position_size != 0)
        executed = [t for t in trade_log if t.get("position_size", 0) != 0]
        returns = [t["net_return"] for t in executed]

        if not equity.empty:
            total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
            trading_days = len(equity)
        else:
            total_return = 0.0
            trading_days = 1

        ann_factor = 252.0 / max(trading_days, 1)

        # Annualized return
        ann_return = (1 + total_return) ** ann_factor - 1.0 if total_return > -1.0 else -1.0

        # Sharpe Ratio
        mean_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns, ddof=0)) if returns else 1.0
        sharpe = (
            (mean_ret / std_ret * np.sqrt(252.0 / result.horizon_days))
            if std_ret > 0
            else 0.0
        )

        # Sortino Ratio (downside deviation only)
        downside_returns = [r for r in returns if r < 0]
        downside_std = (
            float(np.std(downside_returns, ddof=0)) if downside_returns else 1.0
        )
        sortino = (
            (mean_ret / downside_std * np.sqrt(252.0 / result.horizon_days))
            if downside_std > 0
            else 0.0
        )

        # Max Drawdown & Duration
        max_dd, max_dd_duration = self._max_drawdown_with_duration(equity)

        # Calmar Ratio
        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0.0

        # Hit Rate
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        hit_rate = len(wins) / len(returns) if returns else 0.0

        # Profit Factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Avg Win / Avg Loss
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean([abs(r) for r in losses])) if losses else 0.0

        # Expectancy
        loss_rate = 1.0 - hit_rate
        expectancy = hit_rate * avg_win - loss_rate * avg_loss

        # Buy & Hold return
        if not result.benchmark_curve.empty:
            bh_return = float(
                result.benchmark_curve.iloc[-1] / result.benchmark_curve.iloc[0] - 1.0
            )
        else:
            bh_return = 0.0

        # Alpha vs benchmark
        alpha = total_return - bh_return

        return {
            "total_trades": len(executed),
            "rejected_trades": len(trade_log) - len(executed),
            "total_return": round(total_return, 6),
            "annualized_return": round(ann_return, 6),
            "buy_and_hold_return": round(bh_return, 6),
            "alpha": round(alpha, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown": round(max_dd, 6),
            "max_drawdown_duration_days": max_dd_duration,
            "hit_rate": round(hit_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "expectancy": round(expectancy, 6),
            "avg_trade_return": round(mean_ret, 6),
            "horizon_days": result.horizon_days,
            "transaction_cost_bps": result.params.get("transaction_cost_bps", 0),
            "slippage_bps": result.params.get("slippage_bps", 0),
        }

    @staticmethod
    def _max_drawdown_with_duration(equity: pd.Series) -> Tuple[float, int]:
        """Compute max drawdown and its duration in trading days.

        Returns:
            (max_drawdown, max_drawdown_duration_days)
            max_drawdown is negative (e.g., -0.15 for 15% drawdown).
        """
        if equity.empty or len(equity) < 2:
            return 0.0, 0

        values = equity.values.astype(float)
        cummax = np.maximum.accumulate(values)
        drawdowns = values / cummax - 1.0

        max_dd = float(np.min(drawdowns))

        # Compute duration of the longest drawdown
        in_drawdown = drawdowns < 0
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, max_duration

    # ── Per-Regime Breakdown ─────────────────────────────────────────────

    def _compute_per_regime_metrics(
        self, result: BacktestResult
    ) -> Dict[str, Dict[str, Any]]:
        """Compute performance metrics broken down by regime state.

        This directly informs regime_risk_budget.json parameter tuning:
        - Regime with low hit_rate → lower max_position
        - Regime with high avg_loss → tighter stop_multiplier
        - Regime with high reject_rate → confidence_floor may be too low
        """
        executed = [t for t in result.trade_log if t.get("position_size", 0) != 0]

        regime_groups: Dict[str, List[Dict[str, Any]]] = {}
        for trade in executed:
            state = trade.get("regime_state", "unknown")
            regime_groups.setdefault(state, []).append(trade)

        per_regime: Dict[str, Dict[str, Any]] = {}
        total_return_sum = (
            sum(t["net_return"] for t in executed) if executed else 1.0
        )

        for state, trades in regime_groups.items():
            returns = [t["net_return"] for t in trades]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]

            regime_return = sum(returns)
            contribution = (
                regime_return / total_return_sum if total_return_sum != 0 else 0.0
            )

            avg_pos = float(np.mean([abs(t["position_size"]) for t in trades]))

            stop_hits = sum(
                1 for t in trades if t.get("exit_reason") == "stop_loss"
            )
            tp_hits = sum(
                1 for t in trades if t.get("exit_reason") == "take_profit"
            )

            per_regime[state] = {
                "trade_count": len(trades),
                "hit_rate": round(len(wins) / len(returns), 4) if returns else 0.0,
                "avg_return": round(float(np.mean(returns)), 6) if returns else 0.0,
                "total_return": round(regime_return, 6),
                "contribution_to_total": round(contribution, 4),
                "avg_position_size": round(avg_pos, 4),
                "avg_win": round(float(np.mean(wins)), 6) if wins else 0.0,
                "avg_loss": (
                    round(float(np.mean([abs(r) for r in losses])), 6)
                    if losses
                    else 0.0
                ),
                "stop_loss_hit_rate": (
                    round(stop_hits / len(trades), 4) if trades else 0.0
                ),
                "take_profit_hit_rate": (
                    round(tp_hits / len(trades), 4) if trades else 0.0
                ),
            }

        return per_regime

    # ── Rolling Window Stability ─────────────────────────────────────────

    def _compute_rolling_metrics(
        self,
        result: BacktestResult,
        window_size: int = 40,
    ) -> Dict[str, Any]:
        """Compute rolling-window metrics to detect strategy degradation.

        A declining rolling Sharpe or hit_rate indicates the strategy
        may be losing edge over time.
        """
        executed = [t for t in result.trade_log if t.get("position_size", 0) != 0]

        if len(executed) < window_size:
            return {"window_size": window_size, "insufficient_data": True}

        rolling_sharpe: List[float] = []
        rolling_hit_rate: List[float] = []
        rolling_dates: List[str] = []

        for i in range(window_size, len(executed) + 1):
            window = executed[i - window_size : i]
            returns = [t["net_return"] for t in window]

            mean_r = float(np.mean(returns))
            std_r = float(np.std(returns, ddof=0))
            sharpe = (
                (mean_r / std_r * np.sqrt(252.0 / result.horizon_days))
                if std_r > 0
                else 0.0
            )
            hit = sum(1 for r in returns if r > 0) / len(returns)

            rolling_sharpe.append(round(sharpe, 4))
            rolling_hit_rate.append(round(hit, 4))
            rolling_dates.append(str(window[-1].get("date", "")))

        # Determine trend
        sharpe_trend = "stable"
        if len(rolling_sharpe) >= 2:
            if rolling_sharpe[-1] < rolling_sharpe[0] * 0.7:
                sharpe_trend = "declining"
            elif rolling_sharpe[-1] > rolling_sharpe[0] * 1.3:
                sharpe_trend = "improving"

        return {
            "window_size": window_size,
            "insufficient_data": False,
            "rolling_sharpe": rolling_sharpe,
            "rolling_hit_rate": rolling_hit_rate,
            "rolling_dates": rolling_dates,
            "sharpe_trend": sharpe_trend,
        }

    # ── Signal Quality Analysis ──────────────────────────────────────────

    def _compute_signal_quality(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze the quality of signals produced by the agent pipeline.

        Helps identify whether RiskAgent is rejecting too many or too few trades,
        and whether alignment/uncertainty filters are calibrated correctly.
        """
        all_signals = result.trade_log
        executed = [t for t in all_signals if t.get("position_size", 0) != 0]
        rejected = [t for t in all_signals if t.get("position_size", 0) == 0]

        # Rejection breakdown
        reject_reasons: Dict[str, int] = {}
        for t in rejected:
            reason = t.get("reject_reason", "unknown")
            if reason:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        # Alignment distribution
        alignments = [
            t.get("signal_alignment", 0.5)
            for t in all_signals
            if t.get("signal_alignment") is not None
        ]

        # Kelly utilization (actual position / kelly suggestion)
        kelly_utils: List[float] = []
        for t in executed:
            kelly = t.get("kelly_fraction", 0)
            actual = abs(t.get("position_size", 0))
            if kelly and kelly > 0:
                kelly_utils.append(actual / kelly)

        return {
            "total_signals": len(all_signals),
            "executed_trades": len(executed),
            "rejected_trades": len(rejected),
            "rejection_rate": (
                round(len(rejected) / len(all_signals), 4) if all_signals else 0.0
            ),
            "reject_reasons": reject_reasons,
            "alignment_mean": (
                round(float(np.mean(alignments)), 4) if alignments else 0.0
            ),
            "alignment_std": (
                round(float(np.std(alignments)), 4) if alignments else 0.0
            ),
            "kelly_utilization_mean": (
                round(float(np.mean(kelly_utils)), 4) if kelly_utils else 0.0
            ),
        }

    # ── Exit Analysis ────────────────────────────────────────────────────

    def _compute_exit_analysis(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze how trades were exited (horizon vs stop_loss vs take_profit).

        High stop_loss rate suggests stops are too tight or signals are poor.
        Low take_profit rate suggests targets are too ambitious.
        """
        executed = [t for t in result.trade_log if t.get("position_size", 0) != 0]

        exit_counts: Dict[str, int] = {
            "horizon": 0,
            "stop_loss": 0,
            "take_profit": 0,
        }
        exit_returns: Dict[str, List[float]] = {
            "horizon": [],
            "stop_loss": [],
            "take_profit": [],
        }

        for t in executed:
            reason = t.get("exit_reason", "horizon")
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            exit_returns.setdefault(reason, []).append(t["net_return"])

        total = len(executed) or 1

        # Determine stop-loss effectiveness
        sl_rate = exit_counts["stop_loss"] / total
        if sl_rate > 0.4:
            sl_effectiveness = "too_tight"
        elif sl_rate < 0.05:
            sl_effectiveness = "too_loose"
        else:
            sl_effectiveness = "appropriate"

        return {
            "exit_distribution": {
                k: round(v / total, 4) for k, v in exit_counts.items()
            },
            "exit_avg_return": {
                k: round(float(np.mean(v)), 6) if v else 0.0
                for k, v in exit_returns.items()
            },
            "stop_loss_effectiveness": sl_effectiveness,
        }
