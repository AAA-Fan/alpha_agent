"""
Backtest Agent
Runs a lightweight walk-forward simulation of the heuristic forecast policy.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.yfinance_cache import get_historical_data


class BacktestAgent:
    """Evaluates a simple forecast-driven strategy on recent history."""

    def __init__(
        self,
        lookback_days: int | None = None,
        horizon_days: int | None = None,
        transaction_cost_bps: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.lookback_days = lookback_days or int(os.getenv("BACKTEST_LOOKBACK_DAYS", "320"))
        self.horizon_days = horizon_days or int(os.getenv("BACKTEST_HORIZON_DAYS", "5"))
        self.transaction_cost_bps = transaction_cost_bps or float(
            os.getenv("BACKTEST_TRANSACTION_COST_BPS", "5")
        )
        self.verbose = verbose

    def _sigmoid(self, value: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-value))

    def _prepare_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.sort_index().copy()
        close = pd.to_numeric(frame["Close"], errors="coerce")
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        returns = close.pct_change()

        frame["momentum_5"] = close.pct_change(5)
        frame["momentum_20"] = close.pct_change(20)
        frame["sma_20_ratio"] = (close / close.rolling(20).mean()) - 1
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        frame["macd_hist"] = macd - macd_signal
        frame["volatility_20"] = returns.rolling(20).std() * np.sqrt(252)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        frame["rsi_14"] = 100 - (100 / (1 + rs))

        volume_mean_20 = volume.rolling(20).mean()
        volume_std_20 = volume.rolling(20).std().replace(0, np.nan)
        frame["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20

        score = (
            3.0 * frame["momentum_5"].fillna(0)
            + 2.0 * frame["momentum_20"].fillna(0)
            + 1.5 * frame["sma_20_ratio"].fillna(0)
            + 2.0 * frame["macd_hist"].fillna(0)
            + 0.2 * frame["volume_zscore_20"].clip(-3, 3).fillna(0)
            + 0.8 * ((frame["rsi_14"].fillna(50) - 50.0) / 50.0)
            - 1.0 * (frame["volatility_20"].fillna(0.25) - 0.2).clip(lower=0)
        )
        frame["probability_up"] = self._sigmoid(score)
        frame["future_return"] = close.shift(-self.horizon_days) / close - 1
        frame["position"] = np.select(
            [frame["probability_up"] >= 0.55, frame["probability_up"] <= 0.45],
            [1, -1],
            default=0,
        )
        frame["raw_strategy_return"] = frame["position"] * frame["future_return"]
        frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
        frame["cost"] = frame["turnover"] * (self.transaction_cost_bps / 10000.0)
        frame["strategy_return"] = frame["raw_strategy_return"] - frame["cost"]
        return frame

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        if drawdown.empty:
            return 0.0
        return float(drawdown.min())

    def analyze(self, stock_symbol: str) -> Dict[str, Any]:
        """Run a no-training-data backtest over recent history."""
        ticker = stock_symbol.upper().strip()
        try:
            data = get_historical_data(
                ticker,
                interval="daily",
                days=self.lookback_days + self.horizon_days + 60,
            )
            if data.empty or len(data) < 100:
                return {
                    "agent": "backtest",
                    "stock_symbol": ticker,
                    "status": "warning",
                    "metrics": {},
                    "summary": "Backtest skipped due to insufficient history.",
                }

            frame = self._prepare_frame(data)
            valid = frame.dropna(subset=["future_return", "strategy_return", "probability_up"])
            if valid.empty:
                return {
                    "agent": "backtest",
                    "stock_symbol": ticker,
                    "status": "warning",
                    "metrics": {},
                    "summary": "Backtest produced no valid evaluation rows.",
                }

            trades = valid[valid["position"] != 0]
            equity = (1.0 + valid["strategy_return"]).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)
            max_drawdown = self._max_drawdown(equity)
            avg_return = float(valid["strategy_return"].mean())
            std_return = float(valid["strategy_return"].std(ddof=0))
            sharpe = 0.0 if std_return == 0 else avg_return / std_return * math.sqrt(252.0 / self.horizon_days)

            start_idx = valid.index[0]
            end_idx = valid.index[-1]
            close = pd.to_numeric(data["Close"], errors="coerce")
            buy_hold_return = float(close.loc[end_idx] / close.loc[start_idx] - 1.0)

            metrics = {
                "evaluation_rows": int(len(valid)),
                "trades": int(len(trades)),
                "hit_rate": float((trades["strategy_return"] > 0).mean()) if len(trades) else 0.0,
                "avg_trade_return": float(trades["strategy_return"].mean()) if len(trades) else 0.0,
                "strategy_total_return": total_return,
                "buy_and_hold_return": buy_hold_return,
                "max_drawdown": max_drawdown,
                "sharpe_approx": float(sharpe),
                "horizon_days": int(self.horizon_days),
                "transaction_cost_bps": float(self.transaction_cost_bps),
            }
            summary = (
                f"{ticker} backtest ({self.horizon_days}d horizon): "
                f"strategy_return={total_return:.2%}, hit_rate={metrics['hit_rate']:.2%}, "
                f"max_drawdown={max_drawdown:.2%}."
            )
            return {
                "agent": "backtest",
                "stock_symbol": ticker,
                "status": "success",
                "metrics": metrics,
                "summary": summary,
            }
        except Exception as exc:
            return {
                "agent": "backtest",
                "stock_symbol": ticker,
                "status": "error",
                "metrics": {},
                "summary": f"Backtest failed: {exc}",
            }
