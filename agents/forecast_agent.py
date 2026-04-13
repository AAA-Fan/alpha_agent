"""
Forecast Agent
Produces probabilistic return forecasts using a trained LightGBM model when
available, falling back to a Ridge config or deterministic heuristic model.

Model loading priority (3-tier fallback):
  1. data/forecast_model.lgb  → LightGBM model (best)
     V2: data/forecast_model_v2.lgb → Cross-sectional model
  2. data/forecast_model.json → Legacy Ridge model (acceptable)
  3. _heuristic_score()       → Hand-tuned heuristic (last resort)
"""

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Macro/Fundamental feature columns (must match training pipeline) ─────
from utils.macro_fundamental_provider import (
    MACRO_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    ALL_MACRO_FUNDAMENTAL_COLUMNS,
)

# ── Regime constants (must match training pipeline) ──────────────────────

ALL_REGIME_STATES = [
    "strong_rally",
    "trending_up",
    "topping_out",
    "range_bound",
    "coiling",
    "choppy",
    "trending_down",
    "bottoming_out",
    "capitulation",
]

HEALTH_MAP = {"accelerating": 0, "steady": 1, "decelerating": 2, "exhausted": 3}

# Compressed regime encoding (must match training pipeline)
_STATE_DIRECTION = {
    "strong_rally": 2,
    "trending_up": 1,
    "topping_out": 0,
    "range_bound": 0,
    "coiling": 0,
    "choppy": -1,
    "trending_down": -1,
    "bottoming_out": 0,
    "capitulation": -2,
}

_VOL_REGIME_ORD = {
    "low": 0,
    "normal": 1,
    "high": 2,
    "extreme": 3,
    "unknown": 1,
}


class ForecastAgent:
    """Generates directional probability forecasts and confidence intervals."""

    def __init__(
        self,
        model_path: str | None = None,
        horizon_days: int | None = None,
        verbose: bool = False,
        cross_section_service=None,
    ) -> None:
        self.model_path = model_path or os.getenv("FORECAST_MODEL_PATH", "data/forecast_model.json")
        self.horizon_days = horizon_days or int(os.getenv("FORECAST_HORIZON_DAYS", "5"))
        self.buy_threshold = float(os.getenv("FORECAST_BUY_THRESHOLD", "0.55"))
        self.sell_threshold = float(os.getenv("FORECAST_SELL_THRESHOLD", "0.45"))
        self.verbose = verbose
        self.cross_section_service = cross_section_service

        # LightGBM model (Phase 1 / V2)
        self.lgb_model = None
        self.lgb_meta: Optional[Dict[str, Any]] = None
        self.calibrator = None  # Isotonic Regression calibrator
        self._is_v2 = False  # Whether loaded model is V2 cross-sectional
        self._current_ticker = ""  # Set per-prediction for V2
        self._load_lgb_model()
        self._load_calibrator()

    def _load_lgb_model(self) -> None:
        """Try to load LightGBM model and metadata."""
        lgb_path = os.getenv("FORECAST_LGB_MODEL_PATH", "data/forecast_model.lgb")
        meta_path = os.getenv("FORECAST_LGB_META_PATH", "data/forecast_model_meta.json")

        if not os.path.exists(lgb_path) or not os.path.exists(meta_path):
            if self.verbose:
                print(f"[forecast] LightGBM model not found at {lgb_path}")
            return

        try:
            import lightgbm as lgb
            self.lgb_model = lgb.Booster(model_file=lgb_path)
            with open(meta_path, "r") as f:
                self.lgb_meta = json.load(f)
            if self.verbose:
                version = self.lgb_meta.get("version", "?")
                model_type = self.lgb_meta.get("model_type", "lightgbm")
                n_features = len(self.lgb_meta.get("feature_columns", []))
                n_regime = len(self.lgb_meta.get("regime_features", []))
                n_rank = len(self.lgb_meta.get("rank_feature_columns", []))
                n_cat = len(self.lgb_meta.get("categorical_features", []))
                self._is_v2 = model_type == "cross_sectional"
                print(
                    f"[forecast] LightGBM model loaded (v{version}, type={model_type}, "
                    f"{n_features} base + {n_regime} regime + {n_rank} rank + {n_cat} categorical)"
                )
            else:
                self._is_v2 = self.lgb_meta.get("model_type") == "cross_sectional"
        except Exception as exc:
            if self.verbose:
                print(f"[forecast] Failed to load LightGBM model: {exc}")
            self.lgb_model = None
            self.lgb_meta = None

    def _load_calibrator(self) -> None:
        """Load Isotonic Regression calibrator if available."""
        # Prefer path from metadata, fallback to env/default
        calibrator_path = None
        if self.lgb_meta:
            calibrator_path = self.lgb_meta.get("calibrator_path")
        if not calibrator_path:
            calibrator_path = os.getenv("FORECAST_CALIBRATOR_PATH", "data/forecast_calibrator.pkl")

        if not os.path.exists(calibrator_path):
            if self.verbose:
                print(f"[forecast] Calibrator not found at {calibrator_path}")
            return

        try:
            with open(calibrator_path, "rb") as f:
                self.calibrator = pickle.load(f)
            if self.verbose:
                print(f"[forecast] Isotonic calibrator loaded from {calibrator_path}")
        except Exception as exc:
            if self.verbose:
                print(f"[forecast] Failed to load calibrator: {exc}")
            self.calibrator = None

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

    # ── Regime feature extraction (Phase 2) ──────────────────────────────

    def _build_regime_feature_vector(self, regime: Dict[str, Any]) -> Dict[str, float]:
        """Extract compressed regime features for LightGBM inference (Phase 2).

        Converts the regime dict from RegimeAgent into ordinal-encoded features
        matching the training pipeline's compressed encoding.
        """
        state = regime.get("state", "range_bound")
        vol_regime = regime.get("volatility_regime", "normal")

        row: Dict[str, float] = {
            # Ordinal direction encoding (-2 to +2)
            "regime_direction": float(_STATE_DIRECTION.get(state, 0)),
            # Ordinal volatility encoding (0-3)
            "regime_volatility_ord": float(_VOL_REGIME_ORD.get(vol_regime, 1)),
            # Continuous features
            "trend_strength": float(regime.get("trend_strength", 0.0)),
            "vol_expanding": float(int(regime.get("vol_expanding", False))),
            # Label-encoded momentum health
            "momentum_health_enc": float(
                HEALTH_MAP.get(regime.get("momentum_health", "steady"), 1)
            ),
        }
        return row

    # ── Macro/Fundamental feature extraction ─────────────────────────────

    def _build_macro_fundamental_vector(
        self,
        macro_features: Dict[str, Any],
        fundamental_features: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract macro/fundamental features for LightGBM inference.

        Converts the provider output into a flat dict matching the training
        pipeline's column names.
        """
        row: Dict[str, float] = {}
        for col in MACRO_FEATURE_COLUMNS:
            row[col] = self._coerce_float(macro_features.get(col))
        for col in FUNDAMENTAL_FEATURE_COLUMNS:
            row[col] = self._coerce_float(fundamental_features.get(col))
        return row

    # ── LightGBM scoring (Phase 1 + 2 + macro/fundamental) ───────────────

    def _score_from_lgb(
        self,
        features: Dict[str, Any],
        regime: Dict[str, Any],
        macro_features: Dict[str, Any] | None = None,
        fundamental_features: Dict[str, Any] | None = None,
    ) -> float:
        """Score using LightGBM model. Returns probability of upward move."""
        feature_cols: List[str] = self.lgb_meta.get("feature_columns", [])
        regime_cols: List[str] = self.lgb_meta.get("regime_features", [])
        macro_fund_cols: List[str] = self.lgb_meta.get("macro_fundamental_features", [])
        rank_cols: List[str] = self.lgb_meta.get("rank_feature_columns", [])
        cat_cols: List[str] = self.lgb_meta.get("categorical_features", [])

        # Build base feature vector
        row: Dict[str, float] = {}
        for col in feature_cols:
            row[col] = self._coerce_float(features.get(col))

        # Add regime features if Phase 2 model
        if regime_cols:
            regime_features = self._build_regime_feature_vector(regime)
            for col in regime_cols:
                row[col] = regime_features.get(col, 0.0)

        # Add macro/fundamental features if model was trained with them
        if macro_fund_cols:
            mf_vector = self._build_macro_fundamental_vector(
                macro_features or {}, fundamental_features or {},
            )
            for col in macro_fund_cols:
                row[col] = mf_vector.get(col, 0.0)

        # V2: Add cross-sectional rank features
        if rank_cols and self.cross_section_service:
            cs_features = self.cross_section_service.get_cross_sectional_features(
                target_ticker=self._current_ticker,
                target_features=features,
            )
            for col in rank_cols:
                row[col] = cs_features.get(col, 0.5)  # Default median rank
        elif rank_cols:
            # No service available, use default median ranks
            for col in rank_cols:
                row[col] = 0.5

        # V2: Add categorical features (sector_code, industry_code)
        if cat_cols and self.cross_section_service:
            for col in cat_cols:
                if col == "sector_code":
                    row[col] = float(self.cross_section_service.get_sector_code(
                        self._current_ticker
                    ))
                elif col == "industry_code":
                    row[col] = float(self.cross_section_service.get_industry_code(
                        self._current_ticker
                    ))
                else:
                    row[col] = 0.0
        elif cat_cols:
            for col in cat_cols:
                row[col] = 0.0

        all_cols = feature_cols + regime_cols + macro_fund_cols + rank_cols + cat_cols
        X = pd.DataFrame([row])[all_cols]
        raw_probability_up = float(self.lgb_model.predict(X)[0])

        # Apply Isotonic calibration if available
        calibrated = False
        if self.calibrator is not None:
            try:
                probability_up = float(self.calibrator.predict([raw_probability_up])[0])
                calibrated = True
            except Exception:
                probability_up = raw_probability_up
        else:
            probability_up = raw_probability_up

        # Soft-clamp: prevent Isotonic calibrator from over-polarizing
        # The calibrator maps to [0.01, 0.99] which destroys ranking within
        # 20-day windows (constant output → IC=NaN). Blend calibrated output
        # with raw probability to preserve ordering while still benefiting
        # from calibration.
        if calibrated:
            _BLEND_ALPHA = 0.7  # 70% calibrated, 30% raw
            probability_up = _BLEND_ALPHA * probability_up + (1 - _BLEND_ALPHA) * raw_probability_up
            # Hard floor/ceiling to avoid degenerate 0/1
            probability_up = max(0.02, min(0.98, probability_up))

        # Store for uncertainty quantification
        self._last_raw_prob = raw_probability_up
        self._last_calibrated = calibrated
        self._last_X = X

        return probability_up

    # ── Legacy scoring methods ───────────────────────────────────────────

    def _heuristic_score(self, features: Dict[str, Any], regime: Dict[str, Any]) -> float:
        momentum_5 = self._coerce_float(features.get("momentum_5"))
        momentum_20 = self._coerce_float(features.get("momentum_20"))
        macd_hist = self._coerce_float(features.get("macd_hist"))
        sma_20_ratio = self._coerce_float(features.get("sma_20_ratio"))
        volatility_20 = self._coerce_float(features.get("volatility_20"), default=0.25)
        volume_zscore = self._coerce_float(features.get("volume_zscore_20"))
        rsi_14 = self._coerce_float(features.get("rsi_14"), default=50.0)

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

    # ── Main analysis entry point ────────────────────────────────────────

    def analyze(
        self,
        stock_symbol: str,
        feature_analysis: Dict[str, Any] | None = None,
        regime_analysis: Dict[str, Any] | None = None,
        macro_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a probability forecast for the configured horizon.

        Args:
            stock_symbol: Ticker symbol.
            feature_analysis: Output from FeatureEngineeringAgent.
            regime_analysis: Output from RegimeAgent.
            macro_features: Output from MacroFundamentalFeatureProvider
                            (optional; degrades gracefully if absent).
        """
        ticker = stock_symbol.upper().strip()
        self._current_ticker = ticker  # Set for V2 cross-sectional features
        feature_analysis = feature_analysis or {}
        regime_analysis = regime_analysis or {}
        macro_features = macro_features or {}

        features = feature_analysis.get("features", {})
        regime = regime_analysis.get("regime", {})
        macro_feats = macro_features.get("macro_features", {})
        fund_feats = macro_features.get("fundamental_features", {})
        if not features:
            # Degraded mode: return a conservative neutral forecast instead of error
            degraded_forecast = {
                "model_source": "degraded_fallback",
                "horizon_days": int(self.horizon_days),
                "probability_up": 0.5,
                "action": "hold",
            }
            return {
                "agent": "forecast",
                "stock_symbol": ticker,
                "status": "degraded",
                "degraded_reason": "No feature data available; using conservative neutral forecast.",
                "forecast": degraded_forecast,
                "summary": f"{ticker} forecast: DEGRADED — no features. Defaulting to HOLD (prob_up=0.50, confidence=0.05).",
            }

        # ── 3-tier model selection ───────────────────────────────────────
        # Reset per-prediction state
        self._last_raw_prob = None
        self._last_calibrated = False
        self._last_X = None

        if self.lgb_model and self.lgb_meta:
            # Tier 1: LightGBM (best)
            probability_up = self._score_from_lgb(
                features, regime, macro_feats, fund_feats,
            )
            model_source = "lightgbm"
        else:
            config = self._load_model_config()
            if config:
                # Tier 2: Legacy Ridge model
                score = self._score_from_model(features, config)
                probability_up = self._sigmoid(score)
                model_source = "trained_config"
            else:
                # Tier 3: Hand-tuned heuristic (last resort)
                score = self._heuristic_score(features, regime)
                probability_up = self._sigmoid(score)
                model_source = "heuristic_fallback"

        if probability_up >= self.buy_threshold:
            action = "buy"
        elif probability_up <= self.sell_threshold:
            action = "sell"
        else:
            action = "hold"

        # ── Uncertainty quantification (LightGBM only) ────────────────
        raw_prob = self._last_raw_prob if self._last_raw_prob is not None else probability_up
        uncertainty_info = self._compute_uncertainty(
            raw_prob=raw_prob,
            calibrated_prob=probability_up,
        )

        forecast = {
            "model_source": model_source,
            "horizon_days": int(self.horizon_days),
            "probability_up": float(probability_up),
            "raw_probability_up": float(self._last_raw_prob) if self._last_raw_prob is not None else float(probability_up),
            "calibrated": bool(self._last_calibrated),
            "action": action,
            **uncertainty_info,
        }
        # Build summary with uncertainty info
        unc_str = ""
        if uncertainty_info.get("uncertainty") is not None:
            unc = uncertainty_info["uncertainty"]
            unc_str = f", uncertainty={unc:.4f}"
            if uncertainty_info.get("is_uncertain"):
                unc_str += " [HIGH]"

        summary = (
            f"{ticker} forecast ({model_source}): action={action}, "
            f"prob_up={probability_up:.2f}{unc_str} "
            f"over {self.horizon_days}d."
        )
        return {
            "agent": "forecast",
            "stock_symbol": ticker,
            "status": "success",
            "forecast": forecast,
            "summary": summary,
        }

    # ── Uncertainty quantification ───────────────────────────────────────

    def _compute_uncertainty(
        self,
        raw_prob: float,
        calibrated_prob: float,
    ) -> Dict[str, Any]:
        """Compute prediction uncertainty using tree ensemble dispersion + conformal.

        Two complementary methods:
        1. **Tree ensemble dispersion**: Split LightGBM trees into groups,
           compute per-group predictions, measure std. High std = model
           internally disagrees = uncertain.  Uses *raw* (pre-calibration)
           probability space so the dispersion reflects genuine model
           disagreement without calibration distortion.
        2. **Conformal prediction set**: Using pre-computed nonconformity
           score quantiles, build a prediction set {up}, {down}, or {up, down}.
           Uses *raw* (pre-calibration) probability to match the raw-based
           conformal quantiles stored in meta (conformal_probability_type=raw).
           This avoids overlap with tree dispersion caused by soft-clamp
           pulling calibrated probabilities toward the center.

        Args:
            raw_prob: Pre-calibration probability from LightGBM.
            calibrated_prob: Post-calibration probability (same as raw_prob
                             when no calibrator is loaded).

        Returns a dict with uncertainty fields to merge into the forecast.
        """
        result: Dict[str, Any] = {
            "uncertainty": None,
            "prediction_range": None,
            "prediction_set": None,
            "is_uncertain": False,
        }

        if not self.lgb_model or self._last_X is None:
            return result

        # ── Method 1: Tree ensemble dispersion ───────────────────────
        try:
            n_trees = self.lgb_model.num_trees()
            n_groups = min(5, n_trees)  # Split into up to 5 groups
            if n_groups >= 2:
                group_size = n_trees // n_groups
                group_preds = []
                for g in range(n_groups):
                    start = g * group_size
                    n_iter = group_size if g < n_groups - 1 else n_trees - start
                    pred = self.lgb_model.predict(
                        self._last_X,
                        start_iteration=start,
                        num_iteration=n_iter,
                    )
                    group_preds.append(float(pred[0]))

                uncertainty = float(np.std(group_preds))
                pred_min = float(min(group_preds))
                pred_max = float(max(group_preds))

                result["uncertainty"] = round(uncertainty, 6)
                result["prediction_range"] = [round(pred_min, 4), round(pred_max, 4)]

                # Threshold: if tree groups disagree by more than 0.15, flag as uncertain
                if uncertainty > 0.15:
                    result["is_uncertain"] = True
        except Exception:
            pass  # Degrade gracefully

        # ── Method 2: Conformal prediction set ───────────────────────
        if self.lgb_meta:
            quantiles = self.lgb_meta.get("conformal_scores_quantiles", {})
            threshold = quantiles.get("q90")  # 90% coverage level
            if threshold is not None:
                prediction_set = []
                # Check if "up" is in the prediction set
                # nonconformity score for "up" = 1 - prob_up
                # Use raw_prob (not calibrated) to match raw-based quantiles
                if (1.0 - raw_prob) <= threshold:
                    prediction_set.append("up")
                # Check if "down" is in the prediction set
                # nonconformity score for "down" = prob_up
                if raw_prob <= threshold:
                    prediction_set.append("down")

                result["prediction_set"] = prediction_set

                # If both labels are in the set, prediction is uncertain
                if len(prediction_set) == 2:
                    result["is_uncertain"] = True
                # If neither label is in the set (very rare), also uncertain
                elif len(prediction_set) == 0:
                    result["is_uncertain"] = True

        return result
