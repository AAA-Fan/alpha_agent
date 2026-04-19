"""
Risk Agent
Transforms forecast confidence and volatility into a practical risk plan.

Uses signal alignment (with high-confidence override), Kelly position sizing,
uncertainty-aware adjustments, and dynamic stop losses.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

# ── Kelly defaults (used when MemoryAgent has insufficient data) ─────────
DEFAULT_AVG_WIN = 0.09
DEFAULT_AVG_LOSS = 0.02
MIN_KELLY_SAMPLES = 20

# ── Minimum effective position threshold ─────────────────────────────────
MIN_POSITION_THRESHOLD = 0.03

# ── Signal alignment thresholds ──────────────────────────────────────────
ALIGNMENT_REJECT_THRESHOLD = 0.4   # below this → reject trade
# ALIGNMENT_REDUCE_THRESHOLD removed: data shows alignment 0.4-0.7 trades are
# profitable on average; halving position only reduces gains without improving
# risk-adjusted returns. See ablation analysis (v1b_B vs v1b_D).
HIGH_CONFIDENCE_OVERRIDE = 0.35    # |prob - 0.5| > this → override reject (prob > 0.85 or < 0.15)

# ── Uncertainty thresholds ───────────────────────────────────────────────
UNCERTAINTY_HIGH_THRESHOLD = 0.15   # above this → reject trade
UNCERTAINTY_MODERATE_THRESHOLD = 0.10  # above this → reduce position by 30%

# ── Regime direction score mapping ───────────────────────────────────────
REGIME_DIRECTION_SCORE: Dict[str, float] = {
    "strong_rally":  +1.0,
    "trending_up":   +0.7,
    "bottoming_out": +0.3,
    "range_bound":    0.0,
    "coiling":        0.0,
    "choppy":        -0.2,
    "topping_out":   -0.3,
    "trending_down": -0.7,
    "capitulation":  -1.0,
}


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

    # ── Helpers ──────────────────────────────────────────────────────────

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _compute_signal_alignment(self, regime_state: str, probability_up: float) -> float:
        """Compute alignment score between Regime and Forecast signals.

        Returns:
            alignment ∈ [0, 1]: 1.0 = fully aligned, 0.0 = fully opposed.
        """
        regime_signal = REGIME_DIRECTION_SCORE.get(regime_state, 0.0)
        forecast_signal = (probability_up - 0.5) * 2.0
        alignment = 1.0 - abs(regime_signal - forecast_signal) / 2.0
        return alignment

    def _compute_kelly_position(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute 1/4 Kelly position size (legacy, kept for reference).

        Returns:
            Quarter Kelly fraction. Returns 0.0 if expected value is negative.
        """
        if avg_loss <= 0:
            return 0.0

        b = avg_win / avg_loss  # odds ratio
        q = 1.0 - win_rate
        full_kelly = (win_rate * b - q) / b

        if full_kelly <= 0:
            return 0.0  # Negative expected value → don't trade

        quarter_kelly = full_kelly * 0.25
        return min(quarter_kelly, self.max_position_size)

    def _compute_prediction_kelly(
        self,
        probability_up: float,
        action: str,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute Kelly fraction using current prediction probability.

        Uses the *predicted* probability as p (not historical win rate),
        combined with historical avg_win/avg_loss as the odds ratio b.
        Position size scales continuously with signal strength.

        Full Kelly is used (not 1/4 Kelly) because conformal prediction
        filtering already removes the most uncertain signals.
        """
        if avg_loss <= 0:
            return 0.0
        b = avg_win / avg_loss  # odds ratio

        if action == "buy":
            p = probability_up
        elif action == "sell":
            p = 1.0 - probability_up
        else:
            p = max(probability_up, 1.0 - probability_up)

        q = 1.0 - p
        full_kelly = (p * b - q) / b
        if full_kelly <= 0:
            return 0.0
        return min(full_kelly, self.max_position_size)

    # ── Main entry point ─────────────────────────────────────────────────

    def analyze(
        self,
        stock_symbol: str,
        forecast_analysis: Dict[str, Any] | None = None,
        regime_analysis: Dict[str, Any] | None = None,
        feature_analysis: Dict[str, Any] | None = None,
        memory_analysis: Dict[str, Any] | None = None,
        macro_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert model outputs into a risk-adjusted execution plan."""
        ticker = stock_symbol.upper().strip()
        forecast_analysis = forecast_analysis or {}
        regime_analysis = regime_analysis or {}
        feature_analysis = feature_analysis or {}
        memory_analysis = memory_analysis or {}
        macro_features = macro_features or {}

        forecast = forecast_analysis.get("forecast", {})
        regime = regime_analysis.get("regime", {})
        features = feature_analysis.get("features", {})
        macro_feats = macro_features.get("macro_features", {})
        fund_feats = macro_features.get("fundamental_features", {})
        memory = memory_analysis.get("memory", {})

        if not forecast:
            # Degraded mode: return the most conservative risk plan
            degraded_plan = {
                "position_size_fraction": 0.0,
                "track_record_factor": 1.0,
                "max_holding_days": 1,
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.02,
                "risk_flags": [
                    "no_forecast_available",
                    "degraded_mode",
                    "no_strong_edge",
                ],
                "execution_notes": (
                    "DEGRADED: No forecast data available. Zero position recommended. "
                    "Do not open new positions until the pipeline recovers."
                ),
                "signal_alignment": 0.0,
                "kelly_fraction": 0.0,
                "regime_state": "unknown",
                "reject_reason": "no_forecast_available",
            }
            return {
                "agent": "risk_management",
                "stock_symbol": ticker,
                "status": "degraded",
                "degraded_reason": "No forecast data available; using maximum-conservative risk plan.",
                "risk_plan": degraded_plan,
                "summary": f"{ticker} risk plan: DEGRADED — no forecast. Zero position, tightest stops.",
            }

        # ── Extract key inputs ───────────────────────────────────────────
        action = str(forecast.get("action", "hold")).lower()
        probability_up = self._coerce_float(forecast.get("probability_up"), default=0.5)
        volatility_20 = self._coerce_float(features.get("volatility_20"), default=0.25)
        horizon_days = int(self._coerce_float(forecast.get("horizon_days"), default=5))
        regime_state = regime.get("state", "range_bound")
        regime_confidence = self._coerce_float(regime.get("confidence"), default=0.5)

        risk_flags: list[str] = []
        reject_reason: str | None = None
        kelly_fraction = 0.0  # Will be updated if Kelly calculation runs
        position_size = 0.0  # Initialize position size

        # ── ① Signal alignment ──────────────────────────────────────────
        alignment = self._compute_signal_alignment(regime_state, probability_up)

        if alignment < ALIGNMENT_REJECT_THRESHOLD:
            # Check if model confidence is high enough to override rejection
            model_confidence = abs(probability_up - 0.5)
            if model_confidence >= HIGH_CONFIDENCE_OVERRIDE:
                # High-confidence override: allow trade but flag it
                risk_flags.append("signal_conflict_overridden")
            else:
                # Severe conflict + low confidence → reject trade
                position_size = 0.0
                reject_reason = "signal_conflict"
                risk_flags.append("signal_conflict")

        # ── Strong rally regime: prohibit short selling ──────────────────────
        if reject_reason is None and regime_state == "strong_rally" and action == "sell":
            position_size = 0.0
            reject_reason = "strong_rally_no_short"
            risk_flags.append("strong_rally_no_short")

        if reject_reason is None:
            # ── ② 1/4 Kelly position sizing ─────────────────────────────
            prediction_count = memory.get("prediction_count", 0)
            if prediction_count >= MIN_KELLY_SAMPLES:
                avg_win = memory.get("avg_win", DEFAULT_AVG_WIN)
                avg_loss = memory.get("avg_loss", DEFAULT_AVG_LOSS)
            else:
                avg_win = DEFAULT_AVG_WIN
                avg_loss = DEFAULT_AVG_LOSS

            kelly_fraction = self._compute_prediction_kelly(
                probability_up=probability_up,
                action=action,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )

            if kelly_fraction <= 0:
                position_size = 0.0
                reject_reason = "negative_expected_value"
                risk_flags.append("negative_expected_value")
            else:
                position_size = kelly_fraction

                # ── ③ Uncertainty-aware adjustment ──────────────────────
                # Conformal prediction set and tree dispersion are two
                # independent uncertainty signals — handle them separately.
                pred_set = forecast.get("prediction_set") or []
                tree_dispersion = self._coerce_float(
                    forecast.get("uncertainty"), default=None
                )

                if len(pred_set) == 2:
                    # Both {up, down} in conformal set → ambiguous
                    position_size = 0.0
                    reject_reason = "conformal_ambiguous"
                    risk_flags.append("conformal_ambiguous")
                elif len(pred_set) == 0 and pred_set is not None:
                    # Empty conformal set → also reject
                    position_size = 0.0
                    reject_reason = "conformal_empty"
                    risk_flags.append("conformal_empty")
                elif tree_dispersion is not None and tree_dispersion > UNCERTAINTY_HIGH_THRESHOLD:
                    # High tree dispersion → halve position (not reject)
                    position_size *= 0.5
                    risk_flags.append("high_tree_dispersion")

        # ── ⑥ Direction ─────────────────────────────────────────────────
        direction = 0
        if action == "buy":
            direction = 1
        elif action == "sell":
            direction = -1
        else:
            position_size *= 0.25
            if "no_strong_edge" not in risk_flags:
                risk_flags.append("no_strong_edge")

        position_size = direction * position_size

        # ── ⑦ Macro/Fundamental adjustments (DISABLED) ──────────────────
        # Ablation analysis showed macro-based position compression reduces
        # total return significantly (-56%) without proportional risk improvement.
        # Macro features still contribute via LightGBM probability prediction.
        # VIX, yield spread, financial health flags are kept for informational
        # purposes only — they no longer compress position size.
        vix_level = self._coerce_float(macro_feats.get("vix_level"), default=0.0)
        debt_to_equity = self._coerce_float(fund_feats.get("debt_to_equity"), default=0.0)

        # ── ⑧ Track record factor ───────────────────────────────────────
        track_record_factor = float(memory_analysis.get("track_record_factor", 1.0))
        position_size *= track_record_factor
        if track_record_factor < 0.7:
            risk_flags.append("poor_historical_track_record")

        # Clamp to max position
        position_size = max(-self.max_position_size, min(self.max_position_size, position_size))

        # ── ⑨ Minimum position threshold ────────────────────────────────
        if abs(position_size) > 0 and abs(position_size) < MIN_POSITION_THRESHOLD:
            position_size = 0.0
            reject_reason = reject_reason or "position_too_small"
            risk_flags.append("position_too_small")

        # ── ⑩ Dynamic stop loss ─────────────────────────────────────────
        daily_vol = volatility_20 / math.sqrt(252.0)
        base_stop = daily_vol * 2.5
        stop_loss_pct = min(0.08, max(0.01, base_stop))

        # VIX-based and debt-based stop widening DISABLED.
        # Macro features influence the model probability instead.

        take_profit_pct = stop_loss_pct * self.risk_reward_ratio

        # ── Additional risk flags ────────────────────────────────────────
        if volatility_20 >= 0.4:
            risk_flags.append("elevated_volatility")
        if (action == "buy" and probability_up < 0.55) or (action == "sell" and probability_up > 0.45):
            risk_flags.append("action_probability_mismatch")
        if vix_level > 30 and "elevated_vix" not in risk_flags:
            risk_flags.append("elevated_vix")
        # Uncertainty-based risk flags (for info, even if position was already rejected)
        if forecast.get("is_uncertain") and "high_forecast_uncertainty" not in risk_flags:
            risk_flags.append("model_internally_uncertain")
        pred_set_final = forecast.get("prediction_set") or []
        if len(pred_set_final) == 2 and "high_forecast_uncertainty" not in risk_flags:
            risk_flags.append("conformal_ambiguous")

        # ── Build output ─────────────────────────────────────────────────
        risk_plan = {
            "position_size_fraction": float(position_size),
            "track_record_factor": float(track_record_factor),
            "max_holding_days": int(max(horizon_days, 1)),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "risk_flags": risk_flags,
            "execution_notes": (
                "Use limit entries near VWAP and avoid opening new positions "
                "during major macro/earnings events."
            ),
            # Enhanced output fields
            "signal_alignment": round(alignment, 4),
            "kelly_fraction": round(kelly_fraction, 4),
            "regime_state": regime_state,
            "reject_reason": reject_reason,
        }

        summary = (
            f"{ticker} risk plan: size={position_size:.2f}, stop={stop_loss_pct:.3f}, "
            f"take_profit={take_profit_pct:.3f}, alignment={alignment:.2f}, "
            f"kelly={kelly_fraction:.3f}, flags={len(risk_flags)}."
        )

        return {
            "agent": "risk_management",
            "stock_symbol": ticker,
            "status": "success",
            "risk_plan": risk_plan,
            "summary": summary,
        }
