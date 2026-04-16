"""Temperature Scaling calibrator for probability calibration.

This module provides a shared TemperatureScalingCalibrator class that can be
used by both the training pipeline and inference scripts (forecast_agent,
debug_stage scripts, etc.).
"""

from __future__ import annotations

import numpy as np


class TemperatureScalingCalibrator:
    """Temperature Scaling calibrator.

    Temperature Scaling learns a single parameter T that rescales the logits:
        calibrated_prob = sigmoid(logit / T)
    where logit = log(p / (1-p)).

    Key properties:
    - **Preserves ranking**: since sigmoid is monotonic and T > 0, the ordering
      of probabilities is unchanged.  IC and L/S spread are unaffected.
    - **Robust extrapolation**: only 1 parameter, so it generalises well to
      OOS raw probabilities outside the training range.
    - T > 1 -> probabilities are "softened" (pushed toward 0.5)
    - T < 1 -> probabilities are "sharpened" (pushed toward 0 or 1)

    The `.predict()` interface mirrors IsotonicRegression so it can be used
    as a drop-in replacement.
    """

    def __init__(self):
        self.temperature: float = 1.0

    @staticmethod
    def _to_logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "TemperatureScalingCalibrator":
        """Find optimal temperature T that minimises NLL on the calibration set."""
        from scipy.optimize import minimize_scalar

        logits = self._to_logit(raw_probs)

        def nll(T):
            if T <= 0:
                return 1e10
            scaled = self._sigmoid(logits / T)
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self

    def predict(self, raw_probs) -> np.ndarray:
        raw_probs = np.asarray(raw_probs, dtype=float)
        logits = self._to_logit(raw_probs)
        calibrated = self._sigmoid(logits / self.temperature)
        return np.clip(calibrated, 0.01, 0.99)
