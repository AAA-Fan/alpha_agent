"""
Forecast Agent
Produces probabilistic return forecasts using a trained config when available,
otherwise falls back to a deterministic heuristic model.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict


class ForecastAgent:
    """Generates directional probability forecasts and confidence intervals."""

    def __init__(
        self,
        model_path: str | None = None,
        horizon_days: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path or os.getenv("FORECAST_MODEL_PATH", "data/forecast_model.json")
        self.horizon_days = horizon_days or int(os.getenv("FORECAST_HORIZON_DAYS", "5"))
        self.buy_threshold = float(os.getenv("FORECAST_BUY_THRESHOLD", "0.55"))
        self.sell_threshold = float(os.getenv("FORECAST_SELL_THRESHOLD", "0.45"))
        self.verbose = verbose

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _load_model_config(self) -> Dict[str, Any] | None:
        if not os.path.exists(self.model_path):
            return None
        try:
            with open(self.model_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
            return None
        except Exception:
            return None

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _heuristic_score(self, features: Dict[str, Any], regime: Dict[str, Any]) -> float:
        momentum_5 = self._coerce_float(features.get("momentum_5"))
        momentum_20 = self._coerce_float(features.get("momentum_20"))
        macd_hist = self._coerce_float(features.get("macd_hist"))
        sma_20_ratio = self._coerce_float(features.get("sma_20_ratio"))
        volatility_20 = self._coerce_float(features.get("volatility_20"), default=0.25)
        volume_zscore = self._coerce_float(features.get("volume_zscore_20"))
        rsi_14 = self._coerce_float(features.get("rsi_14"), default=50.0)

        # Normalize raw feature magnitudes into a bounded scoring scale.
        score = 0.0
        score += 3.0 * momentum_5
        score += 2.0 * momentum_20
        score += 1.5 * sma_20_ratio
        score += 2.0 * macd_hist
        score += 0.2 * max(min(volume_zscore, 3.0), -3.0)
        score += 0.8 * ((rsi_14 - 50.0) / 50.0)
        score -= 1.0 * max(0.0, volatility_20 - 0.2)

        trend = regime.get("trend")
        volatility_regime = regime.get("volatility_regime")
        if trend == "uptrend":
            score += 0.25
        elif trend == "downtrend":
            score -= 0.25

        if volatility_regime == "high":
            score *= 0.85
        return score

    def _score_from_model(self, features: Dict[str, Any], config: Dict[str, Any]) -> float:
        coefficients = config.get("coefficients", {})
        means = config.get("feature_means", {})
        stds = config.get("feature_stds", {})
        intercept = self._coerce_float(config.get("intercept"))
        score = intercept
        for name, weight in coefficients.items():
            raw = self._coerce_float(features.get(name))
            mean = self._coerce_float(means.get(name))
            std = self._coerce_float(stds.get(name), default=1.0)
            scaled = (raw - mean) / std if std not in {0.0, -0.0} else raw - mean
            score += self._coerce_float(weight) * scaled
        return float(score)

    def analyze(
        self,
        stock_symbol: str,
        feature_analysis: Dict[str, Any] | None = None,
        regime_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate a probability forecast for the configured horizon."""
        ticker = stock_symbol.upper().strip()
        feature_analysis = feature_analysis or {}
        regime_analysis = regime_analysis or {}

        features = feature_analysis.get("features", {})
        regime = regime_analysis.get("regime", {})
        if not features:
            return {
                "agent": "forecast",
                "stock_symbol": ticker,
                "status": "error",
                "forecast": {},
                "summary": "Forecasting requires engineered features but none were provided.",
            }

        config = self._load_model_config()
        if config:
            score = self._score_from_model(features, config)
            model_source = "trained_config"
        else:
            score = self._heuristic_score(features, regime)
            model_source = "heuristic_fallback"

        probability_up = self._sigmoid(score)
        confidence = min(0.99, max(0.01, abs(probability_up - 0.5) * 2.0))

        volatility_20 = self._coerce_float(features.get("volatility_20"), default=0.25)
        daily_vol = volatility_20 / math.sqrt(252.0)
        expected_move = daily_vol * math.sqrt(max(self.horizon_days, 1))
        predicted_return = (probability_up - 0.5) * 2.0 * expected_move

        ci_half_width = expected_move * (1.0 + (1.0 - confidence))
        ci_lower = predicted_return - ci_half_width
        ci_upper = predicted_return + ci_half_width

        if probability_up >= self.buy_threshold:
            action = "buy"
        elif probability_up <= self.sell_threshold:
            action = "sell"
        else:
            action = "hold"

        forecast = {
            "model_source": model_source,
            "horizon_days": int(self.horizon_days),
            "score": float(score),
            "probability_up": float(probability_up),
            "confidence": float(confidence),
            "predicted_return": float(predicted_return),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
            },
            "action": action,
        }
        summary = (
            f"{ticker} forecast ({model_source}): action={action}, "
            f"prob_up={probability_up:.2f}, predicted_return={predicted_return:.4f} "
            f"over {self.horizon_days}d."
        )
        return {
            "agent": "forecast",
            "stock_symbol": ticker,
            "status": "success",
            "forecast": forecast,
            "summary": summary,
        }
