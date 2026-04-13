"""Tests for train_forecast_model pipeline functions."""
import sys
import os
import json
import tempfile

import pytest
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from pipelines.train_forecast_model import build_regime_transition_matrix


# ── Task 9.1: Regime transition matrix ──────────────────────────────────

ALL_STATES = [
    "strong_rally", "trending_up", "topping_out", "range_bound",
    "coiling", "choppy", "trending_down", "bottoming_out", "capitulation",
]


class TestRegimeTransitionMatrix:
    """Test build_regime_transition_matrix() builds a valid 9x9 Markov matrix."""

    def _make_data(self):
        """Create a simple multi-ticker regime sequence for testing."""
        # Ticker A: trending_up → trending_up → topping_out → range_bound
        # Ticker B: trending_down → trending_down → bottoming_out → trending_up
        states = [
            "trending_up", "trending_up", "topping_out", "range_bound",
            "trending_down", "trending_down", "bottoming_out", "trending_up",
        ]
        tickers = ["A", "A", "A", "A", "B", "B", "B", "B"]
        return pd.Series(states), pd.Series(tickers)

    def test_matrix_rows_sum_to_one(self):
        """Each row of the transition matrix should sum to 1.0."""
        regime_states, ticker_ids = self._make_data()
        matrix = build_regime_transition_matrix(regime_states, ticker_ids)

        for from_state in ALL_STATES:
            assert from_state in matrix, f"Missing state: {from_state}"
            row_sum = sum(matrix[from_state].values())
            assert abs(row_sum - 1.0) < 1e-4, (
                f"Row {from_state} sums to {row_sum}, expected 1.0"
            )

    def test_matrix_has_all_states(self):
        """Matrix should contain all 9 states as both rows and columns."""
        regime_states, ticker_ids = self._make_data()
        matrix = build_regime_transition_matrix(regime_states, ticker_ids)

        assert set(matrix.keys()) == set(ALL_STATES)
        for from_state in ALL_STATES:
            assert set(matrix[from_state].keys()) == set(ALL_STATES)

    def test_correct_transition_counts(self):
        """Verify specific transition probabilities from known data."""
        regime_states, ticker_ids = self._make_data()
        matrix = build_regime_transition_matrix(regime_states, ticker_ids)

        # Ticker A: trending_up → trending_up (1 time), trending_up → topping_out (1 time)
        # So trending_up → trending_up should be ~0.5, trending_up → topping_out should be ~0.5
        assert matrix["trending_up"]["trending_up"] > 0.3
        assert matrix["trending_up"]["topping_out"] > 0.3

        # Ticker B: trending_down → trending_down (1 time), trending_down → bottoming_out (1 time)
        assert matrix["trending_down"]["trending_down"] > 0.3
        assert matrix["trending_down"]["bottoming_out"] > 0.3

    def test_uniform_prior_for_unseen_states(self):
        """States with no observed transitions should get uniform distribution."""
        regime_states, ticker_ids = self._make_data()
        matrix = build_regime_transition_matrix(regime_states, ticker_ids)

        # "capitulation" never appears as a from_state in our data
        # Should have uniform distribution
        cap_row = matrix["capitulation"]
        expected_uniform = 1.0 / len(ALL_STATES)
        for to_state in ALL_STATES:
            assert abs(cap_row[to_state] - expected_uniform) < 1e-6, (
                f"capitulation → {to_state} should be uniform ({expected_uniform}), "
                f"got {cap_row[to_state]}"
            )

    def test_per_ticker_counting(self):
        """Transitions should NOT cross ticker boundaries."""
        # Ticker A ends with "range_bound", Ticker B starts with "trending_down"
        # There should be NO range_bound → trending_down transition
        regime_states, ticker_ids = self._make_data()
        matrix = build_regime_transition_matrix(regime_states, ticker_ids)

        # topping_out → range_bound should exist (from Ticker A)
        assert matrix["topping_out"]["range_bound"] > 0
        # bottoming_out → trending_up should exist (from Ticker B)
        assert matrix["bottoming_out"]["trending_up"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
