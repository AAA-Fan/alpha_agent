"""
Risk Agent
Transforms forecast confidence and volatility into a practical risk plan.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict


class RiskAgent:
    """Builds position sizing and risk-control guidance."""

    def __init__(
        self,
        target_annual_volatility: float | None = None,
        max_position_size: float | None = None,
        risk_reward_ratio: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.target_annual_volatility = target_annual_volatility or float(
            os.getenv("RISK_TARGET_ANNUAL_VOL", "0.12")
        )
        self.max_position_size = max_position_size or float(
            os.getenv("RISK_MAX_POSITION_SIZE", "1.0")
        )
        self.risk_reward_ratio = risk_reward_ratio or float(
            os.getenv("RISK_REWARD_RATIO", "2.0")
        )
        self.verbose = verbose

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def analyze(
        self,
        stock_symbol: str,
        forecast_analysis: Dict[str, Any] | None = None,
        regime_analysis: Dict[str, Any] | None = None,
        feature_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Convert model outputs into a risk-adjusted execution plan."""
        ticker = stock_symbol.upper().strip()
        forecast_analysis = forecast_analysis or {}
        regime_analysis = regime_analysis or {}
        feature_analysis = feature_analysis or {}

        forecast = forecast_analysis.get("forecast", {})
        regime = regime_analysis.get("regime", {})
        features = feature_analysis.get("features", {})
        if not forecast:
            return {
                "agent": "risk_management",
                "stock_symbol": ticker,
                "status": "error",
                "risk_plan": {},
                "summary": "Risk management requires a forecast but none was provided.",
            }

        volatility_20 = self._coerce_float(features.get("volatility_20"), default=0.25)
        confidence = self._coerce_float(forecast.get("confidence"), default=0.2)
        action = str(forecast.get("action", "hold")).lower()
        probability_up = self._coerce_float(forecast.get("probability_up"), default=0.5)
        predicted_return = self._coerce_float(forecast.get("predicted_return"), default=0.0)
        horizon_days = int(self._coerce_float(forecast.get("horizon_days"), default=5))

        base_size = self.target_annual_volatility / max(volatility_20, 0.05)
        base_size = min(max(base_size, 0.0), self.max_position_size)

        confidence_scale = min(max(confidence, 0.0), 1.0)
        size = base_size * confidence_scale

        vol_regime = regime.get("volatility_regime")
        if vol_regime == "high":
            size *= 0.6
        elif vol_regime == "low":
            size *= 1.1

        direction = 0
        if action == "buy":
            direction = 1
        elif action == "sell":
            direction = -1
        else:
            size *= 0.25

        position_size = max(-self.max_position_size, min(self.max_position_size, direction * size))

        daily_vol = volatility_20 / math.sqrt(252.0)
        stop_loss_pct = min(0.08, max(0.01, daily_vol * 2.5))
        take_profit_pct = stop_loss_pct * self.risk_reward_ratio

        risk_flags = []
        if confidence < 0.2:
            risk_flags.append("low_signal_confidence")
        if volatility_20 >= 0.4:
            risk_flags.append("elevated_volatility")
        if action == "hold":
            risk_flags.append("no_strong_edge")
        if abs(predicted_return) < 0.002:
            risk_flags.append("small_expected_move")
        if (action == "buy" and probability_up < 0.55) or (action == "sell" and probability_up > 0.45):
            risk_flags.append("action_probability_mismatch")

        risk_plan = {
            "position_size_fraction": float(position_size),
            "max_holding_days": int(max(horizon_days, 1)),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "risk_flags": risk_flags,
            "execution_notes": (
                "Use limit entries near VWAP and avoid opening new positions during major macro/earnings events."
            ),
        }
        summary = (
            f"{ticker} risk plan: size={position_size:.2f}, stop={stop_loss_pct:.3f}, "
            f"take_profit={take_profit_pct:.3f}, flags={len(risk_flags)}."
        )
        return {
            "agent": "risk_management",
            "stock_symbol": ticker,
            "status": "success",
            "risk_plan": risk_plan,
            "summary": summary,
        }
