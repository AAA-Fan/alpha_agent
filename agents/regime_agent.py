"""
Regime Agent
Classifies current market regime from engineered features.
"""

from __future__ import annotations

import os
from typing import Any, Dict


class RegimeAgent:
    """Classifies trend and volatility regime."""

    def __init__(
        self,
        high_vol_threshold: float | None = None,
        low_vol_threshold: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.high_vol_threshold = high_vol_threshold or float(
            os.getenv("REGIME_HIGH_VOL_THRESHOLD", "0.35")
        )
        self.low_vol_threshold = low_vol_threshold or float(
            os.getenv("REGIME_LOW_VOL_THRESHOLD", "0.16")
        )
        self.verbose = verbose

    def _trend_score(self, features: Dict[str, Any]) -> float:
        score = 0.0
        sma_20_ratio = features.get("sma_20_ratio")
        sma_50_ratio = features.get("sma_50_ratio")
        momentum_20 = features.get("momentum_20")
        macd_hist = features.get("macd_hist")

        if isinstance(sma_20_ratio, (int, float)):
            score += 2.0 * float(sma_20_ratio)
        if isinstance(sma_50_ratio, (int, float)):
            score += 1.5 * float(sma_50_ratio)
        if isinstance(momentum_20, (int, float)):
            score += 1.2 * float(momentum_20)
        if isinstance(macd_hist, (int, float)):
            score += 0.8 * float(macd_hist)
        return score

    def _classify_trend(self, score: float) -> str:
        if score > 0.02:
            return "uptrend"
        if score < -0.02:
            return "downtrend"
        return "sideways"

    def _classify_volatility(self, annualized_volatility: float | None) -> str:
        if annualized_volatility is None:
            return "unknown"
        if annualized_volatility >= self.high_vol_threshold:
            return "high"
        if annualized_volatility <= self.low_vol_threshold:
            return "low"
        return "normal"

    def _build_state(self, trend: str, volatility: str) -> str:
        if trend == "uptrend" and volatility in {"low", "normal"}:
            return "trending_up"
        if trend == "downtrend" and volatility in {"low", "normal"}:
            return "trending_down"
        if volatility == "high" and trend == "sideways":
            return "choppy_high_vol"
        if volatility == "high":
            return "high_vol_trend"
        return "range_bound"

    def analyze(
        self,
        stock_symbol: str,
        feature_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Infer market regime from latest engineered features."""
        ticker = stock_symbol.upper().strip()
        feature_analysis = feature_analysis or {}
        features = feature_analysis.get("features", {})

        if not features:
            return {
                "agent": "regime",
                "stock_symbol": ticker,
                "status": "error",
                "regime": {},
                "summary": "Regime analysis requires feature data but none was provided.",
            }

        trend_score = self._trend_score(features)
        trend = self._classify_trend(trend_score)
        annualized_vol = features.get("volatility_20")
        vol_regime = self._classify_volatility(annualized_vol)
        state = self._build_state(trend, vol_regime)

        confidence = min(1.0, max(0.05, abs(trend_score) * 8.0))
        if vol_regime == "high":
            confidence = max(0.05, confidence * 0.85)

        regime = {
            "trend": trend,
            "volatility_regime": vol_regime,
            "state": state,
            "trend_score": float(trend_score),
            "confidence": float(confidence),
        }
        summary = (
            f"{ticker} regime: {state} (trend={trend}, volatility={vol_regime}, "
            f"confidence={confidence:.2f})."
        )
        return {
            "agent": "regime",
            "stock_symbol": ticker,
            "status": "success",
            "regime": regime,
            "summary": summary,
        }
