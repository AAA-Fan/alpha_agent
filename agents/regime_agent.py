"""
Regime Agent
Classifies current market regime using a 3-dimension multi-level framework:
  1. Trend       — 5 levels (strong_uptrend → strong_downtrend)
  2. Volatility  — 4 levels + expansion flag
  3. Momentum    — 4 health states (accelerating → exhausted)

Composite state is one of 9 actionable regimes.

Dimension 4 (optional, from MacroFundamentalFeatureProvider):
  4. Macro Regime — 3 states (risk_on / neutral / risk_off)
     Uses VIX level, yield curve, SPY momentum, and financial health.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class RegimeAgent:
    """Multi-dimensional market regime classifier."""

    # ── configurable thresholds ──────────────────────────────────────────

    TRANSITION_PROB_THRESHOLD = 0.03  # reject transitions below this probability

    def __init__(
        self,
        high_vol_threshold: float | None = None,
        low_vol_threshold: float | None = None,
        extreme_vol_threshold: float | None = None,
        verbose: bool = False,
        transition_matrix_path: str | None = None,
    ) -> None:
        self.extreme_vol_threshold = extreme_vol_threshold or float(
            os.getenv("REGIME_EXTREME_VOL_THRESHOLD", "0.50")
        )
        self.high_vol_threshold = high_vol_threshold or float(
            os.getenv("REGIME_HIGH_VOL_THRESHOLD", "0.35")
        )
        self.low_vol_threshold = low_vol_threshold or float(
            os.getenv("REGIME_LOW_VOL_THRESHOLD", "0.16")
        )
        self.verbose = verbose

        # Load transition probability matrix
        matrix_path = transition_matrix_path or os.getenv(
            "REGIME_TRANSITION_MATRIX_PATH",
            os.path.join(os.path.dirname(__file__), "..", "data", "regime_transition_matrix.json"),
        )
        self._transition_matrix = self._load_transition_matrix(matrix_path)

    @staticmethod
    def _load_transition_matrix(path: str) -> Dict[str, Dict[str, float]]:
        """Load transition matrix from JSON file, returning empty dict on failure."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}  # No smoothing if matrix unavailable

    # ── Regime smoothing (stateless) ─────────────────────────────────────

    def _smooth_regime_sequence(
        self,
        raw_regimes: List[str],
    ) -> Tuple[List[str], List[bool]]:
        """Apply Markov transition smoothing to a raw regime sequence.

        Returns:
            smoothed: List of smoothed regime states.
            was_smoothed: List of booleans indicating if each day was smoothed.
        """
        if not raw_regimes or not self._transition_matrix:
            return list(raw_regimes), [False] * len(raw_regimes)

        smoothed = [raw_regimes[0]]
        was_smoothed = [False]

        for i in range(1, len(raw_regimes)):
            prev_smoothed = smoothed[-1]
            new_raw = raw_regimes[i]

            if prev_smoothed == new_raw:
                # Same state → no smoothing needed
                smoothed.append(new_raw)
                was_smoothed.append(False)
            else:
                # Check transition probability
                prob = self._transition_matrix.get(prev_smoothed, {}).get(new_raw, 1.0)
                if prob < self.TRANSITION_PROB_THRESHOLD:
                    # Low probability transition → reject (noise)
                    smoothed.append(prev_smoothed)
                    was_smoothed.append(True)
                else:
                    # Acceptable transition → accept
                    smoothed.append(new_raw)
                    was_smoothed.append(False)

        return smoothed, was_smoothed

    @staticmethod
    def _count_days_in_current_regime(smoothed_regimes: List[str]) -> int:
        """Count consecutive days of the current (last) regime state."""
        if not smoothed_regimes:
            return 0
        current = smoothed_regimes[-1]
        days = 1
        for i in range(len(smoothed_regimes) - 2, -1, -1):
            if smoothed_regimes[i] == current:
                days += 1
            else:
                break
        return days

    # ── Dimension 1: Trend ───────────────────────────────────────────────

    def _trend_score(self, features: Dict[str, Any]) -> float:
        """Weighted composite trend score from moving-average & momentum features."""
        score = 0.0
        weights = {
            "sma_20_ratio": 2.0,
            "sma_50_ratio": 1.5,
            "momentum_20": 1.2,
            "macd_hist": 0.8,
        }
        for key, w in weights.items():
            val = features.get(key)
            if isinstance(val, (int, float)):
                score += w * float(val)
        return score

    def _classify_trend(self, score: float, features: Dict[str, Any]) -> str:
        """5-level trend classification using score + short-term momentum confirmation."""
        momentum_5 = features.get("momentum_5")
        m5 = float(momentum_5) if isinstance(momentum_5, (int, float)) else 0.0

        if score > 0.05 and m5 > 0:
            return "strong_uptrend"
        if score > 0.02:
            return "uptrend"
        if score < -0.05 and m5 < 0:
            return "strong_downtrend"
        if score < -0.02:
            return "downtrend"
        return "sideways"

    def _trend_strength(self, score: float) -> float:
        """Normalize trend score to 0-1 strength indicator."""
        # Map |score| through a soft-clamp: 0.10 → ~1.0
        return min(1.0, abs(score) / 0.10)

    # ── Dimension 2: Volatility ──────────────────────────────────────────

    def _classify_volatility(self, annualized_vol: float | None) -> str:
        """4-level volatility classification."""
        if annualized_vol is None:
            return "unknown"
        if annualized_vol >= self.extreme_vol_threshold:
            return "extreme"
        if annualized_vol >= self.high_vol_threshold:
            return "high"
        if annualized_vol <= self.low_vol_threshold:
            return "low"
        return "normal"

    def _is_vol_expanding(self, features: Dict[str, Any]) -> bool:
        """Detect whether volatility is expanding (current vol > 5d-ago vol * 1.2).

        Uses momentum_5 of volatility as a proxy: if daily_volatility_20 is
        significantly above its recent level we flag expansion.  Since we only
        have a single snapshot, we approximate by comparing annualized vol to
        the ATR-implied vol.
        """
        vol_20 = features.get("volatility_20")
        atr_14 = features.get("atr_14")
        price = features.get("current_price")
        if not all(isinstance(v, (int, float)) for v in [vol_20, atr_14, price]):
            return False
        if float(price) <= 0:
            return False
        # ATR-implied daily vol (annualized)
        atr_daily_pct = float(atr_14) / float(price)
        atr_annualized = atr_daily_pct * (252 ** 0.5)
        # If rolling vol exceeds ATR-implied vol by 20%+, vol is expanding
        return float(vol_20) > atr_annualized * 1.20

    # ── Dimension 3: Momentum Health ─────────────────────────────────────

    def _classify_momentum_health(
        self, features: Dict[str, Any], trend: str
    ) -> str:
        """4-level momentum health classification.

        - accelerating: m5 & m20 same direction, |m5| > |m20| (trend strengthening)
        - steady:       m5 & m20 same direction, |m5| <= |m20|
        - decelerating: m5 & m20 opposite direction
        - exhausted:    RSI extreme + m5 diverges from trend
        """
        m5 = features.get("momentum_5")
        m20 = features.get("momentum_20")
        rsi = features.get("rsi_14")

        m5_f = float(m5) if isinstance(m5, (int, float)) else 0.0
        m20_f = float(m20) if isinstance(m20, (int, float)) else 0.0
        rsi_f = float(rsi) if isinstance(rsi, (int, float)) else 50.0

        # Check exhaustion first (highest priority)
        trend_is_up = trend in ("uptrend", "strong_uptrend")
        trend_is_down = trend in ("downtrend", "strong_downtrend")

        if rsi_f > 75 and trend_is_up and m5_f < 0:
            return "exhausted"
        if rsi_f < 25 and trend_is_down and m5_f > 0:
            return "exhausted"

        # Same direction check
        same_direction = (m5_f >= 0 and m20_f >= 0) or (m5_f < 0 and m20_f < 0)

        if same_direction:
            if abs(m5_f) > abs(m20_f):
                return "accelerating"
            return "steady"
        return "decelerating"

    # ── RSI zone ─────────────────────────────────────────────────────────

    @staticmethod
    def _rsi_zone(rsi: float | None) -> str:
        if rsi is None:
            return "neutral"
        if float(rsi) >= 70:
            return "overbought"
        if float(rsi) <= 30:
            return "oversold"
        return "neutral"

    # ── Drawdown severity ────────────────────────────────────────────────

    @staticmethod
    def _drawdown_severity(drawdown_60: float | None) -> str:
        if drawdown_60 is None:
            return "none"
        dd = float(drawdown_60)
        if dd <= -0.20:
            return "severe"
        if dd <= -0.10:
            return "moderate"
        if dd <= -0.03:
            return "mild"
        return "none"

    # ── Volume anomaly ───────────────────────────────────────────────────

    @staticmethod
    def _volume_anomaly(volume_zscore: float | None) -> bool:
        if volume_zscore is None:
            return False
        return abs(float(volume_zscore)) > 2.0

    # ── Composite state (9 regimes) ──────────────────────────────────────

    def _build_state(
        self,
        trend: str,
        vol_regime: str,
        momentum_health: str,
        drawdown_severity: str,
    ) -> str:
        """Map 3 dimensions into one of 9 composite regime states."""

        high_vol = vol_regime in ("high", "extreme")
        low_normal_vol = vol_regime in ("low", "normal")

        # 1. capitulation: strong down + high/extreme vol + severe drawdown
        if (
            trend == "strong_downtrend"
            and high_vol
            and drawdown_severity in ("severe", "moderate")
        ):
            return "capitulation"

        # 2. strong_rally: strong up + not exhausted + not extreme vol
        if (
            trend == "strong_uptrend"
            and momentum_health != "exhausted"
            and vol_regime != "extreme"
        ):
            return "strong_rally"

        # 3. topping_out: uptrend + momentum fading
        if trend in ("uptrend", "strong_uptrend") and momentum_health in (
            "decelerating",
            "exhausted",
        ):
            return "topping_out"

        # 4. trending_up: uptrend + healthy momentum
        if trend in ("uptrend", "strong_uptrend") and momentum_health in (
            "accelerating",
            "steady",
        ):
            return "trending_up"

        # 5. bottoming_out: downtrend + momentum fading
        if trend in ("downtrend", "strong_downtrend") and momentum_health in (
            "decelerating",
            "exhausted",
        ):
            return "bottoming_out"

        # 6. trending_down: downtrend + healthy bearish momentum
        if trend in ("downtrend", "strong_downtrend") and momentum_health in (
            "accelerating",
            "steady",
        ):
            return "trending_down"

        # 7. choppy: sideways + high/extreme vol
        if trend == "sideways" and high_vol:
            return "choppy"

        # 8. coiling: sideways + low vol (compression before breakout)
        if trend == "sideways" and vol_regime == "low":
            return "coiling"

        # 9. range_bound: sideways + normal vol (default)
        return "range_bound"

    # ── Dimension 4: Macro Regime (optional) ────────────────────────────

    def _classify_macro_regime(
        self, macro_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """3-level macro regime classification using macro/fundamental features.

        Returns a dict with:
          - macro_regime: 'risk_on' | 'neutral' | 'risk_off'
          - macro_score: float (-1 to +1, negative = risk-off)
          - yield_curve_inverted: bool
        """
        if not macro_features:
            return {
                "macro_regime": "neutral",
                "macro_score": 0.0,
                "yield_curve_inverted": False,
            }

        score = 0.0
        signals = 0

        # VIX level: low VIX = risk-on, high VIX = risk-off
        vix = macro_features.get("vix_level")
        vix_pct = macro_features.get("vix_percentile_1y")
        if isinstance(vix_pct, (int, float)):
            # percentile < 0.3 → risk-on (+1), > 0.7 → risk-off (-1)
            score += (0.5 - float(vix_pct)) * 2.0
            signals += 1
        elif isinstance(vix, (int, float)):
            vix_f = float(vix)
            if vix_f < 15:
                score += 0.5
            elif vix_f > 25:
                score -= 0.5
            elif vix_f > 35:
                score -= 1.0
            signals += 1

        # Yield curve: inverted = risk-off
        spread = macro_features.get("yield_spread_10y2y")
        yield_inverted = False
        if isinstance(spread, (int, float)):
            spread_f = float(spread)
            yield_inverted = spread_f < 0
            if spread_f < -0.2:
                score -= 0.8
            elif spread_f < 0:
                score -= 0.4
            elif spread_f > 1.0:
                score += 0.3
            signals += 1

        # SPY momentum: positive = risk-on
        spy_mom = macro_features.get("spy_momentum_20d")
        if isinstance(spy_mom, (int, float)):
            spy_f = float(spy_mom)
            score += max(-1.0, min(1.0, spy_f * 10.0))  # Scale: 10% move → ±1.0
            signals += 1

        # Rate change: rising rates = mild headwind
        rate_chg = macro_features.get("rate_change_3m")
        if isinstance(rate_chg, (int, float)):
            rate_f = float(rate_chg)
            if rate_f > 0.25:
                score -= 0.3
            elif rate_f < -0.25:
                score += 0.3
            signals += 1

        # CPI YoY: low inflation = risk-on, high inflation = risk-off
        cpi_yoy = macro_features.get("cpi_yoy")
        if isinstance(cpi_yoy, (int, float)):
            cpi_f = float(cpi_yoy)
            if cpi_f > 5.5:
                score -= 1.0   # Very high inflation: strongly unfavorable
            elif cpi_f > 4.0:
                score -= 0.5   # High inflation: unfavorable
            elif cpi_f < 2.5:
                score += 0.5   # Low inflation: favorable
            signals += 1

        # Unemployment Rate: low = risk-on, high = risk-off
        unemp = macro_features.get("unemployment_rate")
        if isinstance(unemp, (int, float)):
            unemp_f = float(unemp)
            if unemp_f > 8.0:
                score -= 1.0   # Very weak labor market
            elif unemp_f > 6.0:
                score -= 0.5   # Weak labor market
            elif unemp_f < 4.5:
                score += 0.5   # Healthy labor market
            signals += 1

        # Normalize score
        if signals > 0:
            score = score / signals
        score = max(-1.0, min(1.0, score))

        if score > 0.2:
            macro_regime = "risk_on"
        elif score < -0.2:
            macro_regime = "risk_off"
        else:
            macro_regime = "neutral"

        return {
            "macro_regime": macro_regime,
            "macro_score": round(float(score), 4),
            "yield_curve_inverted": yield_inverted,
        }

    # ── Dual-window regime transition detection ─────────────────────────

    def _detect_regime_transition(
        self, features: Dict[str, Any], current_state: str
    ) -> Dict[str, Any]:
        """Detect regime transition by comparing short-term vs long-term signals.

        Short-term proxy: momentum_5 + return_5d → short-term direction
        Long-term proxy: sma_50_ratio + momentum_20 → long-term direction
        """
        m5 = self._safe_float(features.get("momentum_5"), 0.0)
        r5d = self._safe_float(features.get("return_5d"), 0.0)
        sma50 = self._safe_float(features.get("sma_50_ratio"), 0.0)
        m20 = self._safe_float(features.get("momentum_20"), 0.0)

        # Short-term direction: average of momentum_5 and return_5d
        short_score = (m5 + r5d) / 2.0
        short_dir = "up" if short_score > 0.005 else ("down" if short_score < -0.005 else "neutral")

        # Long-term direction: average of sma_50_ratio and momentum_20
        long_score = (sma50 + m20) / 2.0
        long_dir = "up" if long_score > 0.005 else ("down" if long_score < -0.005 else "neutral")

        transitioning = (
            (short_dir != long_dir)
            and (short_dir != "neutral")
            and (long_dir != "neutral")
        )

        from_state = None
        to_state = None
        prob = 0.0

        if transitioning:
            from_state = current_state
            if short_dir == "down" and long_dir == "up":
                to_state = "topping_out"
            elif short_dir == "up" and long_dir == "down":
                to_state = "bottoming_out"
            else:
                to_state = current_state
            prob = self._transition_matrix.get(from_state, {}).get(to_state, 0.0)

        return {
            "transitioning": transitioning,
            "from": from_state,
            "to": to_state,
            "transition_probability": round(prob, 4),
            "short_term_direction": short_dir,
            "long_term_direction": long_dir,
        }

    # ── Confidence scoring ───────────────────────────────────────────────

    CONFIDENCE_CAP = 0.85
    TREND_SCORE_FULL = 0.12  # trend_score at which base confidence reaches cap

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        """Extract a scalar float, returning *default* for None / non-numeric."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _compute_confidence(
        self,
        trend_score: float,
        vol_regime: str,
        momentum_health: str,
        features: Dict[str, Any] | None = None,
    ) -> float:
        """Composite confidence with 0.85 cap and multi-timeframe consistency."""
        features = features or {}

        # Step 1: Base confidence (linear, capped at CONFIDENCE_CAP)
        base = min(
            self.CONFIDENCE_CAP,
            abs(trend_score) / self.TREND_SCORE_FULL * self.CONFIDENCE_CAP,
        )
        base = max(0.05, base)

        # Step 2: Multi-timeframe consistency adjustment
        m5 = self._safe_float(features.get("momentum_5"), 0.0)
        sma50_ratio = self._safe_float(features.get("sma_50_ratio"), 0.0)
        trend_sign = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)
        short_sign = 1 if m5 > 0 else (-1 if m5 < 0 else 0)
        long_sign = 1 if sma50_ratio > 0 else (-1 if sma50_ratio < 0 else 0)

        if trend_sign != 0 and short_sign == trend_sign and long_sign == trend_sign:
            # All 3 timeframes agree → boost
            base = min(self.CONFIDENCE_CAP, base * 1.15)
        elif trend_sign != 0 and (short_sign == -trend_sign or long_sign == -trend_sign):
            # At least one timeframe contradicts → penalize
            base *= 0.60

        # Step 3: Signal conflict penalty
        if momentum_health in ("decelerating", "exhausted"):
            base *= 0.75

        # Step 4: Volatility penalty
        if vol_regime == "extreme":
            base *= 0.70
        elif vol_regime == "high":
            base *= 0.85

        # Step 5: Hard cap
        return float(min(self.CONFIDENCE_CAP, max(0.05, base)))

    # ── Public API ───────────────────────────────────────────────────────

    def analyze(
        self,
        stock_symbol: str,
        feature_analysis: Dict[str, Any] | None = None,
        macro_features: Optional[Dict[str, Any]] = None,
        feature_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Infer market regime from latest engineered features.

        Args:
            stock_symbol: Ticker symbol.
            feature_analysis: Output from FeatureEngineeringAgent.
            macro_features: Output from MacroFundamentalFeatureProvider
                            (optional; degrades gracefully if absent).
            feature_history: Last ~60 days of features (time-ascending).
                             Enables stateless smoothing and days_in_current_regime.
                             If None, degrades to single-day mode (backward compatible).
        """
        ticker = stock_symbol.upper().strip()
        feature_analysis = feature_analysis or {}
        macro_features = macro_features or {}
        features = feature_analysis.get("features", {})
        macro_feats = macro_features.get("macro_features", {})

        if not features:
            # Degraded mode: return a neutral default regime
            default_regime = {
                "trend": "sideways",
                "trend_strength": 0.0,
                "volatility_regime": "normal",
                "vol_expanding": False,
                "momentum_health": "steady",
                "rsi_zone": "neutral",
                "drawdown_severity": "none",
                "volume_anomaly": False,
                "state": "range_bound",
                "trend_score": 0.0,
                "confidence": 0.05,
                "macro_regime": "neutral",
                "macro_score": 0.0,
                "yield_curve_inverted": False,
            }
            return {
                "agent": "regime",
                "stock_symbol": ticker,
                "status": "degraded",
                "degraded_reason": "No feature data available; using neutral default regime.",
                "regime": default_regime,
                "summary": (
                    f"{ticker} regime: DEGRADED — no feature data. "
                    "Defaulting to range_bound (sideways, normal vol, confidence=0.05)."
                ),
            }

        # ── Dimension 1: Trend ───────────────────────────────────────────
        trend_score = self._trend_score(features)
        trend = self._classify_trend(trend_score, features)
        trend_strength = self._trend_strength(trend_score)

        # ── Dimension 2: Volatility ──────────────────────────────────────
        annualized_vol = features.get("volatility_20")
        vol_regime = self._classify_volatility(annualized_vol)
        vol_expanding = self._is_vol_expanding(features)

        # ── Dimension 3: Momentum Health ─────────────────────────────────
        momentum_health = self._classify_momentum_health(features, trend)

        # ── Auxiliary signals ────────────────────────────────────────────
        rsi_zone = self._rsi_zone(features.get("rsi_14"))
        dd_severity = self._drawdown_severity(features.get("drawdown_60"))
        vol_anomaly = self._volume_anomaly(features.get("volume_zscore_20"))

        # ── Composite state ──────────────────────────────────────────────
        state = self._build_state(trend, vol_regime, momentum_health, dd_severity)

        # ── Smoothing via feature_history ───────────────────────────────────
        regime_smoothed = False
        days_in_current_regime = 1

        if feature_history and len(feature_history) >= 2:
            # Compute raw regime for each historical day
            raw_regimes: List[str] = []
            for hist_features in feature_history:
                h_ts = self._trend_score(hist_features)
                h_trend = self._classify_trend(h_ts, hist_features)
                h_vol = self._classify_volatility(hist_features.get("volatility_20"))
                h_mom = self._classify_momentum_health(hist_features, h_trend)
                h_dd = self._drawdown_severity(hist_features.get("drawdown_60"))
                h_state = self._build_state(h_trend, h_vol, h_mom, h_dd)
                raw_regimes.append(h_state)

            # Apply Markov smoothing
            smoothed_regimes, smoothed_flags = self._smooth_regime_sequence(raw_regimes)

            # Today's smoothed state and metadata
            state = smoothed_regimes[-1]
            regime_smoothed = smoothed_flags[-1]
            days_in_current_regime = self._count_days_in_current_regime(smoothed_regimes)

        # ── Dimension 4: Macro Regime (optional) ─────────────────────
        macro_info = self._classify_macro_regime(macro_feats)

        # ── Dual-window transition detection ─────────────────────────────
        regime_transition = self._detect_regime_transition(features, state)
        # ── Confidence ───────────────────────────────────────────────────
        confidence = self._compute_confidence(
            trend_score, vol_regime, momentum_health, features
        )

        # Adjust confidence based on macro alignment (respects 0.85 cap)
        macro_score = macro_info["macro_score"]
        trend_is_up = trend in ("uptrend", "strong_uptrend")
        trend_is_down = trend in ("downtrend", "strong_downtrend")
        if (trend_is_up and macro_score > 0.2) or (trend_is_down and macro_score < -0.2):
            # Macro confirms trend → boost confidence
            confidence = min(self.CONFIDENCE_CAP, confidence * 1.10)
        elif (trend_is_up and macro_score < -0.3) or (trend_is_down and macro_score > 0.3):
            # Macro contradicts trend → reduce confidence
            confidence *= 0.85
        confidence = min(self.CONFIDENCE_CAP, max(0.05, confidence))

        regime = {
            "trend": trend,
            "trend_strength": float(trend_strength),
            "volatility_regime": vol_regime,
            "vol_expanding": bool(vol_expanding),
            "momentum_health": momentum_health,
            "rsi_zone": rsi_zone,
            "drawdown_severity": dd_severity,
            "volume_anomaly": bool(vol_anomaly),
            "state": state,
            "trend_score": float(trend_score),
            "confidence": float(confidence),
            "macro_regime": macro_info["macro_regime"],
            "macro_score": macro_info["macro_score"],
            "yield_curve_inverted": macro_info["yield_curve_inverted"],
            "regime_smoothed": regime_smoothed,
            "days_in_current_regime": days_in_current_regime,
            "regime_transition": regime_transition,
        }
        macro_tag = f", macro={macro_info['macro_regime']}[{macro_info['macro_score']:.2f}]"
        summary = (
            f"{ticker} regime: {state} "
            f"(trend={trend}[{trend_strength:.2f}], vol={vol_regime}"
            f"{'↑' if vol_expanding else ''}, "
            f"momentum={momentum_health}, rsi={rsi_zone}, "
            f"drawdown={dd_severity}{macro_tag}, confidence={confidence:.2f})."
        )
        return {
            "agent": "regime",
            "stock_symbol": ticker,
            "status": "success",
            "regime": regime,
            "summary": summary,
        }
