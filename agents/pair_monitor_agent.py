"""
Pair Monitor Agent
Monitors momentum divergence across ledger pairs and surfaces lagging/leading signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import os

import pandas as pd
import numpy as np

from utils.yfinance_cache import get_historical_data, get_intraday_data


@dataclass(frozen=True)
class PairSignal:
    symbol_a: str
    symbol_b: str
    leading: str
    lagging: str
    momentum_a: float
    momentum_b: float
    divergence: float
    z_score: float
    similarity: float | None
    window: int
    confidence: float


class PairMonitorAgent:
    """Monitors ledger pairs for short-term divergence using spread z-scores."""

    def __init__(
        self,
        short_window: int | None = None,
        divergence_threshold: float | None = None,
        zscore_window: int | None = None,
        interval: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.short_window = short_window or int(os.getenv("PAIR_MONITOR_WINDOW", "5"))
        self.divergence_threshold = self._resolve_divergence_threshold(divergence_threshold)
        self.zscore_window = zscore_window or int(os.getenv("PAIR_ZSCORE_WINDOW", "20"))
        self.interval = interval or os.getenv("PAIR_MONITOR_INTERVAL", "daily")
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _resolve_divergence_threshold(self, override: float | None) -> float:
        if override is not None:
            return override
        if os.getenv("PAIR_ZSCORE_THRESHOLD"):
            return float(os.getenv("PAIR_ZSCORE_THRESHOLD", "1.5"))
        return float(os.getenv("PAIR_DIVERGENCE_THRESHOLD", "1.5"))

    def _fetch_series(self, symbol: str, bars: int) -> pd.Series | None:
        interval_key = self.interval.lower()
        if interval_key in {"daily", "1d", "day"}:
            data = get_historical_data(symbol, interval="daily", days=bars)
        else:
            data = get_intraday_data(symbol, interval=self.interval, outputsize="compact")
            if not data.empty:
                data = data.tail(bars)
        if data.empty or "Close" not in data.columns:
            return None
        close = pd.to_numeric(data["Close"], errors="coerce").dropna()
        return close if not close.empty else None

    def _compute_momentum(self, series: pd.Series) -> float | None:
        if len(series) < self.short_window + 1:
            return None
        start = series.iloc[-(self.short_window + 1)]
        end = series.iloc[-1]
        if start == 0:
            return None
        return float((end - start) / start)

    def _compute_zscore(self, series_a: pd.Series, series_b: pd.Series) -> float | None:
        window = max(self.zscore_window, 3)
        if len(series_a) < window or len(series_b) < window:
            return None
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if len(aligned) < window:
            return None
        aligned = aligned.tail(window)
        series_a = pd.to_numeric(aligned.iloc[:, 0], errors="coerce").astype(float)
        series_b = pd.to_numeric(aligned.iloc[:, 1], errors="coerce").astype(float)
        if (series_a <= 0).any() or (series_b <= 0).any():
            return None
        spread = np.log(series_a) - np.log(series_b)
        mean = spread.mean()
        std = spread.std(ddof=0)
        if std == 0 or pd.isna(std):
            return None
        z_score = (spread.iloc[-1] - mean) / std
        return float(z_score)

    def _score_confidence(self, divergence: float, similarity: float | None) -> float:
        base = min(abs(divergence) / max(self.divergence_threshold, 1e-6), 2.0)
        similarity_score = similarity if similarity is not None else 0.5
        confidence = base * similarity_score
        return float(min(max(confidence, 0.0), 1.0))

    def monitor_pairs(
        self, pairs: List[Dict[str, Any]], focus_symbol: str | None = None
    ) -> Dict[str, Any]:
        signals: List[PairSignal] = []
        errors: Dict[str, str] = {}

        focus_symbol = focus_symbol.upper() if focus_symbol else None

        for pair in pairs:
            symbol_a = pair.get("symbol_a")
            symbol_b = pair.get("symbol_b")
            similarity = pair.get("similarity")
            if not symbol_a or not symbol_b:
                continue

            if focus_symbol and focus_symbol not in {symbol_a, symbol_b}:
                continue

            try:
                bars_needed = max(self.short_window + 1, self.zscore_window)
                series_a = self._fetch_series(symbol_a, bars_needed)
                series_b = self._fetch_series(symbol_b, bars_needed)
                if series_a is None or series_b is None:
                    errors[f"{symbol_a}/{symbol_b}"] = "insufficient data"
                    continue
                momentum_a = self._compute_momentum(series_a)
                momentum_b = self._compute_momentum(series_b)
                z_score = self._compute_zscore(series_a, series_b)
                if momentum_a is None or momentum_b is None or z_score is None:
                    errors[f"{symbol_a}/{symbol_b}"] = "insufficient data"
                    continue

                divergence = z_score
                if abs(divergence) < self.divergence_threshold:
                    continue
                if divergence > 0:
                    leading, lagging = symbol_a, symbol_b
                else:
                    leading, lagging = symbol_b, symbol_a

                confidence = self._score_confidence(divergence, similarity)
                signals.append(
                    PairSignal(
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        leading=leading,
                        lagging=lagging,
                        momentum_a=momentum_a,
                        momentum_b=momentum_b,
                        divergence=divergence,
                        z_score=z_score,
                        similarity=similarity,
                        window=self.short_window,
                        confidence=confidence,
                    )
                )
            except Exception as exc:
                errors[f"{symbol_a}/{symbol_b}"] = str(exc)

        summary_lines = []
        for signal in signals:
            summary_lines.append(
                f"{signal.leading} leading {signal.lagging} with z-score {signal.z_score:.2f} over "
                f"{signal.window} {self.interval} bars (confidence {signal.confidence:.2f})."
            )

        return {
            "agent": "pair_monitor",
            "status": "success" if signals or not errors else "warning",
            "signals": [signal.__dict__ for signal in signals],
            "summary": "\n".join(summary_lines) if summary_lines else "No actionable divergences detected.",
            "errors": errors,
        }

    def analyze(self, pairs: List[Dict[str, Any]], focus_symbol: str | None = None) -> Dict[str, Any]:
        return self.monitor_pairs(pairs, focus_symbol=focus_symbol)
