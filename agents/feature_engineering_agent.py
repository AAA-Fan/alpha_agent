"""
Feature Engineering Agent
Builds quantitative features used by forecasting, regime, and risk components.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.yfinance_cache import get_historical_data


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class FeatureEngineeringAgent:
    """Computes derived market features from historical OHLCV data."""

    def __init__(
        self,
        lookback_days: int | None = None,
        rsi_period: int = 14,
        verbose: bool = False,
    ) -> None:
        self.lookback_days = lookback_days or int(os.getenv("FEATURE_LOOKBACK_DAYS", "220"))
        self.rsi_period = rsi_period
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _build_features(self, data: pd.DataFrame) -> Dict[str, float | int | str | None]:
        frame = data.sort_index().copy()
        if len(frame) < 60:
            raise ValueError("Insufficient history for feature engineering; need at least 60 bars.")

        close = pd.to_numeric(frame["Close"], errors="coerce")
        high = pd.to_numeric(frame["High"], errors="coerce")
        low = pd.to_numeric(frame["Low"], errors="coerce")
        open_ = pd.to_numeric(frame["Open"], errors="coerce")
        volume = pd.to_numeric(frame["Volume"], errors="coerce")

        returns = close.pct_change()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        rsi_14 = _compute_rsi(close, period=self.rsi_period)
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_daily_20 = returns.rolling(20).std()
        momentum_5 = close.pct_change(5)
        momentum_20 = close.pct_change(20)

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        volume_mean_20 = volume.rolling(20).mean()
        volume_std_20 = volume.rolling(20).std().replace(0, np.nan)
        volume_z_20 = (volume - volume_mean_20) / volume_std_20

        rolling_max_60 = close.rolling(60).max()
        drawdown_60 = (close / rolling_max_60) - 1

        overnight_gap = (open_ / prev_close) - 1
        intraday_return = (close / open_) - 1

        latest = {
            "current_price": _safe_float(close.iloc[-1]),
            "sma_20_ratio": _safe_float((close.iloc[-1] / sma_20.iloc[-1]) - 1),
            "sma_50_ratio": _safe_float((close.iloc[-1] / sma_50.iloc[-1]) - 1),
            "macd": _safe_float(macd.iloc[-1]),
            "macd_signal": _safe_float(macd_signal.iloc[-1]),
            "macd_hist": _safe_float(macd_hist.iloc[-1]),
            "rsi_14": _safe_float(rsi_14.iloc[-1]),
            "volatility_20": _safe_float(vol_20.iloc[-1]),
            "daily_volatility_20": _safe_float(vol_daily_20.iloc[-1]),
            "atr_14": _safe_float(atr_14.iloc[-1]),
            "momentum_5": _safe_float(momentum_5.iloc[-1]),
            "momentum_20": _safe_float(momentum_20.iloc[-1]),
            "return_1d": _safe_float(returns.iloc[-1]),
            "return_5d": _safe_float(close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 5 else None),
            "volume_zscore_20": _safe_float(volume_z_20.iloc[-1]),
            "drawdown_60": _safe_float(drawdown_60.iloc[-1]),
            "overnight_gap": _safe_float(overnight_gap.iloc[-1]),
            "intraday_return": _safe_float(intraday_return.iloc[-1]),
            "data_points": int(len(frame)),
            "as_of": frame.index[-1].strftime("%Y-%m-%d"),
        }
        return latest

    def analyze(self, stock_symbol: str) -> Dict[str, Any]:
        """Compute engineered features for a stock ticker."""
        ticker = stock_symbol.upper().strip()
        try:
            self._log(f"[feature_engineering] fetching data for {ticker}")
            data = get_historical_data(ticker, interval="daily", days=self.lookback_days + 60)
            if data.empty:
                return {
                    "agent": "feature_engineering",
                    "stock_symbol": ticker,
                    "status": "error",
                    "features": {},
                    "summary": f"No historical data available for {ticker}.",
                }

            features = self._build_features(data)
            summary = (
                f"{ticker} features as of {features.get('as_of')}: "
                f"momentum_5={features.get('momentum_5')}, "
                f"rsi_14={features.get('rsi_14')}, "
                f"volatility_20={features.get('volatility_20')}."
            )
            return {
                "agent": "feature_engineering",
                "stock_symbol": ticker,
                "status": "success",
                "features": features,
                "summary": summary,
            }
        except Exception as exc:
            return {
                "agent": "feature_engineering",
                "stock_symbol": ticker,
                "status": "error",
                "features": {},
                "summary": f"Feature engineering failed: {exc}",
            }
