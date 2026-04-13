#!/usr/bin/env python3
"""
Standalone script to build the 9x9 regime transition matrix.

This script downloads historical data for a set of tickers, computes base
features, derives the raw regime state for each row, and builds a global
Markov transition matrix.  The result is saved to
``data/regime_transition_matrix.json``.

Usage:
    python scripts/build_regime_matrix.py                  # use sp500_top100.json
    python scripts/build_regime_matrix.py AAPL,MSFT,GOOGL  # custom tickers
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipelines.train_forecast_model import (  # noqa: E402
    download_training_data,
    compute_base_features,
    build_regime_transition_matrix,
    # Internal helpers for computing raw regime state per row
    _trend_score_from_row,
    _classify_trend,
    _classify_volatility,
    _classify_momentum_health,
    _drawdown_severity,
    _build_state,
)

# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "regime_transition_matrix.json")


def _compute_raw_regime_states(frame: pd.DataFrame) -> pd.Series:
    """Compute the raw regime state string for every row in *frame*.

    This mirrors the first half of ``compute_regime_features`` but only
    returns the state label (e.g. 'trending_up') without adding feature
    columns.
    """

    def _safe_float(val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            if pd.notna(val):
                return float(val)
        except (ValueError, TypeError):
            pass
        return default

    states: List[str] = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        m5 = _safe_float(row.get("momentum_5"), 0.0)
        m20 = _safe_float(row.get("momentum_20"), 0.0)
        rsi = _safe_float(row.get("rsi_14"), 50.0)
        vol_20 = _safe_float(row.get("volatility_20"), 0.25)

        ts = _trend_score_from_row(row)
        trend = _classify_trend(ts, m5)
        vol_regime = _classify_volatility(vol_20)
        mom_health = _classify_momentum_health(m5, m20, rsi, trend)
        dd_sev = _drawdown_severity(_safe_float(row.get("drawdown_60"), 0.0))
        state = _build_state(trend, vol_regime, mom_health, dd_sev)
        states.append(state)

    return pd.Series(states, index=frame.index)


def main() -> None:
    load_dotenv()

    # ── Parse tickers ────────────────────────────────────────────────────
    tickers: List[str] = []
    if len(sys.argv) > 1:
        # Accept comma-separated tickers from CLI
        tickers = [t.strip().upper() for t in sys.argv[1].split(",") if t.strip()]

    if not tickers:
        # Fall back to sp500_top100.json
        tickers_path = os.path.join(_PROJECT_ROOT, "data", "sp500_top100.json")
        if os.path.exists(tickers_path):
            with open(tickers_path, "r") as f:
                ticker_data = json.load(f)
            tickers = ticker_data.get("tickers", [])
            print(f"Loaded {len(tickers)} tickers from {tickers_path}")
        else:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
            print(f"No ticker source found, using defaults: {tickers}")

    lookback_years = int(os.getenv("TRAIN_LOOKBACK_YEARS", "5"))

    print("=" * 60)
    print("  Regime Transition Matrix Builder")
    print("=" * 60)
    print(f"  Tickers:  {len(tickers)}")
    print(f"  Lookback: {lookback_years} years")
    print(f"  Output:   {OUTPUT_PATH}")
    print()

    # ── Step 1: Download data ────────────────────────────────────────────
    print("[Step 1] Downloading historical data ...")
    raw_data = download_training_data(tickers)
    if not raw_data:
        raise SystemExit("No training data downloaded. Check API key and network.")

    # ── Step 2: Compute base features & regime states ────────────────────
    print("\n[Step 2] Computing base features and regime states ...")
    all_states: List[pd.Series] = []
    all_tickers: List[pd.Series] = []
    total_rows = 0

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)

    for ticker, data in raw_data.items():
        data = data[data.index >= cutoff]
        if len(data) < 60:
            print(f"  [skip] {ticker}: only {len(data)} rows (need >= 60)")
            continue

        frame = compute_base_features(data)
        # Drop rows with NaN in key columns needed for regime classification
        key_cols = ["momentum_5", "momentum_20", "rsi_14", "volatility_20", "drawdown_60"]
        available_cols = [c for c in key_cols if c in frame.columns]
        frame = frame.dropna(subset=available_cols)

        if len(frame) < 10:
            print(f"  [skip] {ticker}: only {len(frame)} rows after dropna")
            continue

        states = _compute_raw_regime_states(frame)
        all_states.append(states)
        all_tickers.append(pd.Series([ticker] * len(states), index=frame.index))
        total_rows += len(states)
        print(f"  [ok] {ticker}: {len(states)} rows")

    if not all_states:
        raise SystemExit("No valid data to build transition matrix.")

    regime_states = pd.concat(all_states)
    ticker_ids = pd.concat(all_tickers)

    print(f"\n  Total: {total_rows} rows across {len(all_states)} tickers")

    # ── Step 3: Build transition matrix ──────────────────────────────────
    print("\n[Step 3] Building 9x9 transition matrix ...")
    matrix = build_regime_transition_matrix(regime_states, ticker_ids)

    # Print summary
    print("\n  Transition matrix summary:")
    for from_state, row in matrix.items():
        top_transitions = sorted(row.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{s}={p:.3f}" for s, p in top_transitions)
        print(f"    {from_state:20s} → {top_str}")

    # ── Step 4: Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"\n✅ Transition matrix saved to {OUTPUT_PATH}")
    print(f"   Matrix size: 9 × 9 ({sum(1 for r in matrix.values() for _ in r)} cells)")


if __name__ == "__main__":
    main()
