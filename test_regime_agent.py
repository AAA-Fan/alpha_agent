"""Tests for the multi-dimensional RegimeAgent."""
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from agents.regime_agent import RegimeAgent


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_features(overrides=None):
    """Build a feature_analysis dict with sensible defaults."""
    base = {
        "current_price": 150,
        "sma_20_ratio": 0,
        "sma_50_ratio": 0,
        "momentum_20": 0,
        "momentum_5": 0,
        "macd_hist": 0,
        "rsi_14": 50,
        "volatility_20": 0.22,
        "atr_14": 2,
        "volume_zscore_20": 0,
        "drawdown_60": -0.01,
    }
    if overrides:
        base.update(overrides)
    return {"features": base}


# ── Original smoke tests (converted to pytest) ──────────────────────────

_SMOKE_CASES = [
    ("degraded", None, "range_bound"),
    ("strong_rally", _make_features({
        "sma_20_ratio": 0.03, "sma_50_ratio": 0.04, "momentum_20": 0.05,
        "momentum_5": 0.03, "macd_hist": 0.02, "rsi_14": 65,
        "volatility_20": 0.14, "atr_14": 2.5,
    }), "strong_rally"),
    ("capitulation", _make_features({
        "current_price": 100, "sma_20_ratio": -0.05, "sma_50_ratio": -0.08,
        "momentum_20": -0.10, "momentum_5": -0.06, "macd_hist": -0.03,
        "rsi_14": 22, "volatility_20": 0.55, "atr_14": 5,
        "volume_zscore_20": 3, "drawdown_60": -0.25,
    }), "capitulation"),
    ("topping_out", _make_features({
        "sma_20_ratio": 0.02, "sma_50_ratio": 0.03, "momentum_20": 0.04,
        "momentum_5": -0.01, "macd_hist": 0.01, "rsi_14": 78,
    }), "topping_out"),
    ("trending_up", _make_features({
        "sma_20_ratio": 0.008, "sma_50_ratio": 0.008, "momentum_20": 0.008,
        "momentum_5": 0.008, "macd_hist": 0.004,
    }), "trending_up"),
    ("coiling", _make_features({
        "sma_20_ratio": 0.005, "sma_50_ratio": -0.003, "momentum_20": 0.002,
        "momentum_5": 0.001, "macd_hist": 0.001, "volatility_20": 0.12, "atr_14": 1,
    }), "coiling"),
    ("choppy", _make_features({
        "sma_20_ratio": 0.005, "sma_50_ratio": -0.005, "momentum_20": 0.001,
        "momentum_5": -0.002, "macd_hist": 0, "volatility_20": 0.40,
        "atr_14": 3.5, "volume_zscore_20": 2.5, "drawdown_60": -0.05,
    }), "choppy"),
    ("bottoming_out", _make_features({
        "sma_20_ratio": -0.02, "sma_50_ratio": -0.03, "momentum_20": -0.04,
        "momentum_5": 0.01, "macd_hist": -0.01, "rsi_14": 28, "volatility_20": 0.25,
    }), "bottoming_out"),
    ("trending_down", _make_features({
        "sma_20_ratio": -0.03, "sma_50_ratio": -0.04, "momentum_20": -0.05,
        "momentum_5": -0.03, "macd_hist": -0.02, "rsi_14": 38,
    }), "trending_down"),
    ("range_bound", _make_features({
        "sma_20_ratio": 0.005, "sma_50_ratio": -0.003, "momentum_20": 0.002,
        "momentum_5": 0.001, "macd_hist": 0.001,
    }), "range_bound"),
]


@pytest.fixture
def agent():
    """Create a RegimeAgent with no transition matrix (for confidence tests)."""
    return RegimeAgent()


class TestRegimeSmoke:
    """Original smoke tests: verify each of the 9 regime states is reachable."""

    @pytest.mark.parametrize("name,feat,expected_state", _SMOKE_CASES, ids=[c[0] for c in _SMOKE_CASES])
    def test_regime_state(self, agent, name, feat, expected_state):
        result = agent.analyze("AAPL", feat)
        assert result["regime"]["state"] == expected_state

    @pytest.mark.parametrize("name,feat,expected_state", _SMOKE_CASES, ids=[c[0] for c in _SMOKE_CASES])
    def test_required_fields(self, agent, name, feat, expected_state):
        result = agent.analyze("AAPL", feat)
        required = [
            "trend", "volatility_regime", "state", "confidence", "trend_score",
            "trend_strength", "momentum_health", "vol_expanding", "rsi_zone",
            "drawdown_severity", "volume_anomaly",
        ]
        for k in required:
            assert k in result["regime"], f"Missing field: {k}"


# ── Task 8.1: Confidence cap & multi-timeframe consistency ──────────────

CONFIDENCE_CAP = 0.85


class TestConfidenceCap:
    """Test that _compute_confidence() respects the 0.85 cap and
    applies multi-timeframe consistency adjustments."""

    def test_strong_trend_capped_at_085(self, agent):
        """Scenario 1: Very strong trend_score should NOT produce confidence > 0.85."""
        features = _make_features({
            "sma_20_ratio": 0.05, "sma_50_ratio": 0.06,
            "momentum_20": 0.08, "momentum_5": 0.05,
            "macd_hist": 0.03, "rsi_14": 65,
            "volatility_20": 0.18, "atr_14": 2,
        })
        result = agent.analyze("AAPL", features)
        confidence = result["regime"]["confidence"]
        assert confidence <= CONFIDENCE_CAP, (
            f"Confidence {confidence} exceeds cap {CONFIDENCE_CAP}"
        )

    def test_all_timeframes_agree_bullish_boost(self, agent):
        """Scenario 2: momentum_5 > 0, trend_score > 0, sma_50_ratio > 0
        should get ×1.15 boost (still capped at 0.85)."""
        # Moderate trend so base is well below cap, boost is visible
        features = _make_features({
            "sma_20_ratio": 0.015, "sma_50_ratio": 0.02,  # long-term positive
            "momentum_20": 0.02, "momentum_5": 0.015,      # short-term positive
            "macd_hist": 0.005, "rsi_14": 55,
            "volatility_20": 0.20, "atr_14": 2,
        })
        result = agent.analyze("AAPL", features)
        confidence = result["regime"]["confidence"]

        # Also compute what confidence would be WITHOUT the boost
        # by using features where short-term contradicts (no boost, no penalty)
        features_neutral = _make_features({
            "sma_20_ratio": 0.015, "sma_50_ratio": 0.02,
            "momentum_20": 0.02, "momentum_5": 0.0,  # neutral short-term
            "macd_hist": 0.005, "rsi_14": 55,
            "volatility_20": 0.20, "atr_14": 2,
        })
        result_neutral = agent.analyze("AAPL", features_neutral)
        confidence_neutral = result_neutral["regime"]["confidence"]

        # Boosted confidence should be higher than neutral
        assert confidence > confidence_neutral, (
            f"All-agree confidence ({confidence}) should be > neutral ({confidence_neutral})"
        )
        # And still capped
        assert confidence <= CONFIDENCE_CAP

    def test_timeframe_contradiction_penalty(self, agent):
        """Scenario 3: momentum_5 < 0 (short-term bearish) but trend_score > 0
        (medium-term bullish) should get ×0.60 penalty."""
        # Positive medium-term trend
        features_aligned = _make_features({
            "sma_20_ratio": 0.02, "sma_50_ratio": 0.02,
            "momentum_20": 0.03, "momentum_5": 0.02,  # aligned
            "macd_hist": 0.01, "rsi_14": 55,
            "volatility_20": 0.20, "atr_14": 2,
        })
        # Same but short-term contradicts
        features_contra = _make_features({
            "sma_20_ratio": 0.02, "sma_50_ratio": 0.02,
            "momentum_20": 0.03, "momentum_5": -0.02,  # contradicts!
            "macd_hist": 0.01, "rsi_14": 55,
            "volatility_20": 0.20, "atr_14": 2,
        })
        result_aligned = agent.analyze("AAPL", features_aligned)
        result_contra = agent.analyze("AAPL", features_contra)

        conf_aligned = result_aligned["regime"]["confidence"]
        conf_contra = result_contra["regime"]["confidence"]

        # Contradicted confidence should be significantly lower
        assert conf_contra < conf_aligned * 0.80, (
            f"Contradicted confidence ({conf_contra}) should be < 80% of "
            f"aligned ({conf_aligned})"
        )

    def test_signal_conflict_penalty_decelerating(self, agent):
        """momentum_health=decelerating should apply ×0.75 penalty."""
        # Create features that produce decelerating momentum:
        # m5 and m20 in opposite directions
        features_steady = _make_features({
            "sma_20_ratio": 0.02, "sma_50_ratio": 0.02,
            "momentum_20": 0.03, "momentum_5": 0.02,  # same direction → steady/accelerating
            "macd_hist": 0.01, "rsi_14": 55,
            "volatility_20": 0.20, "atr_14": 2,
        })
        features_decel = _make_features({
            "sma_20_ratio": 0.02, "sma_50_ratio": 0.02,
            "momentum_20": 0.03, "momentum_5": -0.01,  # opposite → decelerating
            "macd_hist": 0.01, "rsi_14": 55,
            "volatility_20": 0.20, "atr_14": 2,
        })
        result_steady = agent.analyze("AAPL", features_steady)
        result_decel = agent.analyze("AAPL", features_decel)

        conf_steady = result_steady["regime"]["confidence"]
        conf_decel = result_decel["regime"]["confidence"]

        # Decelerating should be penalized
        assert conf_decel < conf_steady, (
            f"Decelerating confidence ({conf_decel}) should be < "
            f"steady confidence ({conf_steady})"
        )

    def test_signal_conflict_penalty_exhausted(self, agent):
        """momentum_health=exhausted should apply ×0.75 penalty."""
        # Exhausted: RSI > 75, trend is up, but m5 < 0
        features_exhausted = _make_features({
            "sma_20_ratio": 0.02, "sma_50_ratio": 0.02,
            "momentum_20": 0.03, "momentum_5": -0.01,
            "macd_hist": 0.01, "rsi_14": 78,  # overbought + m5 < 0 → exhausted
            "volatility_20": 0.20, "atr_14": 2,
        })
        result = agent.analyze("AAPL", features_exhausted)
        regime = result["regime"]

        # Verify momentum_health is exhausted
        assert regime["momentum_health"] == "exhausted", (
            f"Expected exhausted, got {regime['momentum_health']}"
        )
        # Confidence should be well below cap due to penalty
        assert regime["confidence"] <= CONFIDENCE_CAP

    def test_confidence_never_exceeds_cap_with_macro_boost(self, agent):
        """Even with macro alignment boost (×1.10), confidence must stay ≤ 0.85."""
        features = _make_features({
            "sma_20_ratio": 0.04, "sma_50_ratio": 0.05,
            "momentum_20": 0.06, "momentum_5": 0.04,
            "macd_hist": 0.02, "rsi_14": 60,
            "volatility_20": 0.15, "atr_14": 2,
        })
        # Macro features that confirm bullish trend (macro_score > 0.2)
        macro = {"macro_features": {
            "vix_percentile_1y": 0.1,  # low VIX → risk_on
            "yield_spread_10y2y": 1.5,  # positive spread → risk_on
            "spy_momentum_20d": 0.05,   # positive → risk_on
        }}
        result = agent.analyze("AAPL", features, macro_features=macro)
        confidence = result["regime"]["confidence"]
        assert confidence <= CONFIDENCE_CAP, (
            f"Confidence {confidence} exceeds cap {CONFIDENCE_CAP} even with macro boost"
        )

    def test_confidence_minimum_floor(self, agent):
        """Confidence should never go below 0.05."""
        features = _make_features({
            "sma_20_ratio": 0.0, "sma_50_ratio": 0.0,
            "momentum_20": 0.0, "momentum_5": 0.0,
            "macd_hist": 0.0, "rsi_14": 50,
            "volatility_20": 0.55, "atr_14": 5,  # extreme vol
        })
        result = agent.analyze("AAPL", features)
        confidence = result["regime"]["confidence"]
        assert confidence >= 0.05, f"Confidence {confidence} below floor 0.05"


# ── Task 9.2: Regime smoothing & days_in_current_regime ─────────────────

# A sample transition matrix for testing
_TEST_TRANSITION_MATRIX = {
    "strong_rally":   {"strong_rally": 0.70, "trending_up": 0.15, "topping_out": 0.08, "range_bound": 0.03, "coiling": 0.01, "choppy": 0.01, "trending_down": 0.01, "bottoming_out": 0.005, "capitulation": 0.005},
    "trending_up":    {"strong_rally": 0.05, "trending_up": 0.75, "topping_out": 0.10, "range_bound": 0.05, "coiling": 0.02, "choppy": 0.01, "trending_down": 0.01, "bottoming_out": 0.005, "capitulation": 0.005},
    "topping_out":    {"strong_rally": 0.02, "trending_up": 0.10, "topping_out": 0.50, "range_bound": 0.20, "coiling": 0.05, "choppy": 0.05, "trending_down": 0.05, "bottoming_out": 0.02, "capitulation": 0.01},
    "range_bound":    {"strong_rally": 0.02, "trending_up": 0.10, "topping_out": 0.08, "range_bound": 0.50, "coiling": 0.10, "choppy": 0.08, "trending_down": 0.05, "bottoming_out": 0.05, "capitulation": 0.02},
    "coiling":        {"strong_rally": 0.05, "trending_up": 0.15, "topping_out": 0.05, "range_bound": 0.15, "coiling": 0.40, "choppy": 0.05, "trending_down": 0.10, "bottoming_out": 0.03, "capitulation": 0.02},
    "choppy":         {"strong_rally": 0.02, "trending_up": 0.05, "topping_out": 0.05, "range_bound": 0.15, "coiling": 0.08, "choppy": 0.40, "trending_down": 0.10, "bottoming_out": 0.10, "capitulation": 0.05},
    "trending_down":  {"strong_rally": 0.005, "trending_up": 0.01, "topping_out": 0.01, "range_bound": 0.05, "coiling": 0.02, "choppy": 0.05, "trending_down": 0.75, "bottoming_out": 0.08, "capitulation": 0.025},
    "bottoming_out":  {"strong_rally": 0.02, "trending_up": 0.10, "topping_out": 0.02, "range_bound": 0.15, "coiling": 0.05, "choppy": 0.08, "trending_down": 0.10, "bottoming_out": 0.35, "capitulation": 0.03},
    "capitulation":   {"strong_rally": 0.01, "trending_up": 0.02, "topping_out": 0.01, "range_bound": 0.05, "coiling": 0.02, "choppy": 0.10, "trending_down": 0.15, "bottoming_out": 0.20, "capitulation": 0.44},
}


@pytest.fixture
def agent_with_matrix(tmp_path):
    """Create a RegimeAgent with a test transition matrix."""
    import json
    matrix_file = tmp_path / "test_matrix.json"
    matrix_file.write_text(json.dumps(_TEST_TRANSITION_MATRIX))
    return RegimeAgent(transition_matrix_path=str(matrix_file))


class TestRegimeSmoothing:
    """Test _smooth_regime_sequence() and _count_days_in_current_regime()."""

    def test_low_probability_transition_rejected(self, agent_with_matrix):
        """Scenario 6: trending_up → capitulation (prob=0.005 < 0.03) should be rejected."""
        raw = ["trending_up", "trending_up", "trending_up", "capitulation"]
        smoothed, flags = agent_with_matrix._smooth_regime_sequence(raw)

        # The capitulation should be rejected (smoothed to trending_up)
        assert smoothed[-1] == "trending_up", (
            f"Expected trending_up (smoothed), got {smoothed[-1]}"
        )
        assert flags[-1] is True, "Last day should be marked as smoothed"

    def test_high_probability_transition_accepted(self, agent_with_matrix):
        """Scenario 5: trending_up → topping_out (prob=0.10 > 0.03) should be accepted."""
        raw = ["trending_up", "trending_up", "topping_out"]
        smoothed, flags = agent_with_matrix._smooth_regime_sequence(raw)

        assert smoothed[-1] == "topping_out", (
            f"Expected topping_out (accepted), got {smoothed[-1]}"
        )
        assert flags[-1] is False, "Last day should NOT be marked as smoothed"

    def test_same_state_no_smoothing(self, agent_with_matrix):
        """Same state transitions should never be smoothed."""
        raw = ["trending_up", "trending_up", "trending_up"]
        smoothed, flags = agent_with_matrix._smooth_regime_sequence(raw)

        assert smoothed == raw
        assert all(f is False for f in flags)

    def test_empty_sequence(self, agent_with_matrix):
        """Empty sequence should return empty."""
        smoothed, flags = agent_with_matrix._smooth_regime_sequence([])
        assert smoothed == []
        assert flags == []

    def test_single_element(self, agent_with_matrix):
        """Single element should return as-is."""
        smoothed, flags = agent_with_matrix._smooth_regime_sequence(["trending_up"])
        assert smoothed == ["trending_up"]
        assert flags == [False]

    def test_count_days_in_current_regime(self, agent_with_matrix):
        """Test _count_days_in_current_regime() counts consecutive days correctly."""
        # 3 consecutive trending_up at the end
        seq = ["range_bound", "trending_up", "trending_up", "trending_up"]
        days = agent_with_matrix._count_days_in_current_regime(seq)
        assert days == 3

    def test_count_days_single_day(self, agent_with_matrix):
        """Single day should return 1."""
        days = agent_with_matrix._count_days_in_current_regime(["trending_up"])
        assert days == 1

    def test_count_days_all_same(self, agent_with_matrix):
        """All same state should return full length."""
        seq = ["trending_up"] * 10
        days = agent_with_matrix._count_days_in_current_regime(seq)
        assert days == 10

    def test_count_days_empty(self, agent_with_matrix):
        """Empty sequence should return 0."""
        days = agent_with_matrix._count_days_in_current_regime([])
        assert days == 0

    def test_no_matrix_no_smoothing(self):
        """Without transition matrix, no smoothing should occur."""
        agent_no_matrix = RegimeAgent()
        raw = ["trending_up", "trending_up", "capitulation"]
        smoothed, flags = agent_no_matrix._smooth_regime_sequence(raw)
        # Without matrix, should return raw as-is
        assert smoothed == raw
        assert all(f is False for f in flags)


# ── Task 10.1: Dual-window regime transition detection ──────────────────

class TestRegimeTransitionDetection:
    """Test _detect_regime_transition() dual-window signal detection."""

    def test_short_bearish_long_bullish_transitioning(self, agent_with_matrix):
        """Scenario 7: Short-term bearish + long-term bullish → transitioning to topping_out."""
        features = {
            "momentum_5": -0.02,   # short-term bearish
            "return_5d": -0.015,   # short-term bearish
            "sma_50_ratio": 0.03,  # long-term bullish
            "momentum_20": 0.02,   # long-term bullish
        }
        result = agent_with_matrix._detect_regime_transition(features, "trending_up")
        assert result["transitioning"] is True
        assert result["to"] == "topping_out"
        assert result["short_term_direction"] == "down"
        assert result["long_term_direction"] == "up"

    def test_short_bullish_long_bearish_transitioning(self, agent_with_matrix):
        """Short-term bullish + long-term bearish → transitioning to bottoming_out."""
        features = {
            "momentum_5": 0.02,    # short-term bullish
            "return_5d": 0.015,    # short-term bullish
            "sma_50_ratio": -0.03, # long-term bearish
            "momentum_20": -0.02,  # long-term bearish
        }
        result = agent_with_matrix._detect_regime_transition(features, "trending_down")
        assert result["transitioning"] is True
        assert result["to"] == "bottoming_out"
        assert result["short_term_direction"] == "up"
        assert result["long_term_direction"] == "down"

    def test_aligned_directions_not_transitioning(self, agent_with_matrix):
        """Both short and long term bullish → not transitioning."""
        features = {
            "momentum_5": 0.02,
            "return_5d": 0.015,
            "sma_50_ratio": 0.03,
            "momentum_20": 0.02,
        }
        result = agent_with_matrix._detect_regime_transition(features, "trending_up")
        assert result["transitioning"] is False

    def test_neutral_short_term_not_transitioning(self, agent_with_matrix):
        """Neutral short-term → not transitioning."""
        features = {
            "momentum_5": 0.001,   # near zero
            "return_5d": -0.001,   # near zero
            "sma_50_ratio": 0.03,
            "momentum_20": 0.02,
        }
        result = agent_with_matrix._detect_regime_transition(features, "trending_up")
        assert result["transitioning"] is False

    def test_transition_probability_from_matrix(self, agent_with_matrix):
        """Transition probability should come from the loaded matrix."""
        features = {
            "momentum_5": -0.02,
            "return_5d": -0.015,
            "sma_50_ratio": 0.03,
            "momentum_20": 0.02,
        }
        result = agent_with_matrix._detect_regime_transition(features, "trending_up")
        assert result["transition_probability"] > 0  # Should look up from matrix


# ── Task 10.2: Macro regime enhanced with CPI and unemployment ──────────

class TestMacroRegimeEnhanced:
    """Test _classify_macro_regime() with CPI and unemployment signals."""

    def test_high_cpi_reduces_score(self, agent):
        """High CPI (>4.0) should reduce macro_score (risk_off signal)."""
        macro_only_cpi = {"cpi_yoy": 6.0}
        result = agent._classify_macro_regime(macro_only_cpi)
        assert result["macro_score"] < 0, (
            f"High CPI should produce negative score, got {result['macro_score']}"
        )

    def test_low_cpi_increases_score(self, agent):
        """Low CPI (<2.5) should increase macro_score (risk_on signal)."""
        macro_only_cpi = {"cpi_yoy": 1.5}
        result = agent._classify_macro_regime(macro_only_cpi)
        assert result["macro_score"] > 0, (
            f"Low CPI should produce positive score, got {result['macro_score']}"
        )

    def test_low_unemployment_increases_score(self, agent):
        """Low unemployment (<4.5) should increase macro_score."""
        macro_only_unemp = {"unemployment_rate": 3.5}
        result = agent._classify_macro_regime(macro_only_unemp)
        assert result["macro_score"] > 0, (
            f"Low unemployment should produce positive score, got {result['macro_score']}"
        )

    def test_high_unemployment_reduces_score(self, agent):
        """High unemployment (>6.0) should reduce macro_score."""
        macro_only_unemp = {"unemployment_rate": 7.0}
        result = agent._classify_macro_regime(macro_only_unemp)
        assert result["macro_score"] < 0, (
            f"High unemployment should produce negative score, got {result['macro_score']}"
        )

    def test_cpi_and_unemployment_cancel_out(self, agent):
        """High CPI + low unemployment should partially cancel out."""
        macro = {"cpi_yoy": 6.0, "unemployment_rate": 3.5}
        result = agent._classify_macro_regime(macro)
        # CPI contributes -0.5, unemployment contributes +0.5 → near zero
        assert abs(result["macro_score"]) < 0.3, (
            f"CPI and unemployment should partially cancel, got {result['macro_score']}"
        )

    def test_six_signals_equal_weight(self, agent):
        """All 6 signals should be equally weighted (score / signals)."""
        # All signals present and all risk_on
        macro = {
            "vix_percentile_1y": 0.1,    # risk_on
            "yield_spread_10y2y": 1.5,   # risk_on
            "spy_momentum_20d": 0.05,    # risk_on
            "rate_change_3m": -0.5,      # risk_on
            "cpi_yoy": 1.5,             # risk_on
            "unemployment_rate": 3.5,    # risk_on
        }
        result = agent._classify_macro_regime(macro)
        assert result["macro_regime"] == "risk_on"
        assert result["macro_score"] > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
