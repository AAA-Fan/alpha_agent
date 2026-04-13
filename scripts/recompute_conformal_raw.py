#!/usr/bin/env python3
"""
Recompute conformal prediction quantiles using RAW probabilities (no calibrator).

This script:
1. Loads the training data and replicates the feature engineering pipeline
2. Runs Purged K-Fold CV to get OOF predictions (raw LightGBM output)
3. Computes conformal nonconformity scores WITHOUT calibration
4. Updates the meta JSON with new quantiles under "conformal_scores_quantiles_raw"
   AND replaces the main "conformal_scores_quantiles" with raw-based values

Usage:
    python scripts/recompute_conformal_raw.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipelines.train_forecast_model import (
    download_training_data,
    compute_base_features,
    compute_regime_features,
    compute_sample_weights,
    compute_conformal_scores,
    train_lgb_model,
    PurgedKFold,
    BASE_FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    MACRO_FUNDAMENTAL_FEATURE_COLUMNS,
    LGB_PARAMS,
    N_SEEDS,
)
from utils.macro_fundamental_provider import (
    MacroFundamentalFeatureProvider,
    ALL_MACRO_FUNDAMENTAL_COLUMNS,
)


def main():
    load_dotenv()

    meta_path = "data/forecast_model_meta.json"
    print("=" * 60)
    print("  Recompute Conformal Scores with RAW Probabilities")
    print("=" * 60)

    # Load existing meta
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"\n  Current conformal quantiles (calibrated-based):")
    old_q = meta.get("conformal_scores_quantiles", {})
    for k, v in old_q.items():
        print(f"    {k}: {v:.6f}")

    # ── Step 1: Rebuild training data ────────────────────────────────────
    tickers = meta.get("tickers", ["AAPL"])
    feature_columns = meta.get("feature_columns", [])
    regime_cols = meta.get("regime_features", [])
    macro_fund_cols_meta = meta.get("macro_fundamental_features", [])

    print(f"\n[Step 1] Loading training data for {tickers} ...")
    raw_data = download_training_data(tickers)
    if not raw_data:
        raise SystemExit("No training data. Check API key.")

    # Download SPY for excess return labels
    spy_data_dict = download_training_data(["SPY"])
    spy_data = spy_data_dict.get("SPY")

    # ── Step 2: Feature engineering ──────────────────────────────────────
    print("\n[Step 2] Computing features ...")
    all_frames = []
    lookback_years = 5
    horizon_days = meta.get("target_horizon_days", 5)

    for ticker, data in raw_data.items():
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        data = data[data.index >= cutoff]
        if len(data) < 60:
            print(f"  [skip] {ticker}: only {len(data)} rows")
            continue

        fm = compute_base_features(data)

        # Excess return labels
        close = pd.to_numeric(data["Close"], errors="coerce")
        if spy_data is not None and not spy_data.empty:
            spy_close = pd.to_numeric(spy_data["Close"], errors="coerce")
            spy_fwd = spy_close.shift(-horizon_days) / spy_close - 1.0
            stock_fwd = close.shift(-horizon_days) / close - 1.0
            excess = stock_fwd - spy_fwd.reindex(stock_fwd.index, method="nearest")
            fm["label"] = (excess > 0).astype(float)
        else:
            fwd = close.shift(-horizon_days) / close - 1.0
            fm["label"] = (fwd > 0).astype(float)

        fm.dropna(subset=["label"], inplace=True)

        # Regime features
        if regime_cols:
            fm = compute_regime_features(fm)

        all_frames.append(fm)

    if not all_frames:
        raise SystemExit("No valid training frames.")

    feature_matrix = pd.concat(all_frames)
    feature_matrix.sort_index(inplace=True)
    print(f"  Total samples: {len(feature_matrix)}")

    # ── Step 2.5: Macro/Fundamental features ─────────────────────────────
    if macro_fund_cols_meta:
        print("\n[Step 2.5] Adding macro/fundamental features ...")
        for col in ALL_MACRO_FUNDAMENTAL_COLUMNS:
            if col not in feature_matrix.columns:
                feature_matrix[col] = np.nan

        try:
            provider = MacroFundamentalFeatureProvider()
            hist_data = provider.get_historical_features(
                tickers[0],
                start_date=feature_matrix.index.min().strftime("%Y-%m-%d"),
                end_date=feature_matrix.index.max().strftime("%Y-%m-%d"),
            )
            if hist_data is not None and not hist_data.empty:
                for col in ALL_MACRO_FUNDAMENTAL_COLUMNS:
                    if col in hist_data.columns:
                        feature_matrix[col] = hist_data[col].reindex(feature_matrix.index, method="ffill")
                print(f"  Macro/fund features loaded")
            else:
                print(f"  [warn] No historical macro data, leaving as NaN")
        except Exception as exc:
            print(f"  [warn] Macro data load failed: {exc}, leaving as NaN")

    # ── Step 3: Build X, y ───────────────────────────────────────────────
    all_cols = feature_columns + regime_cols + macro_fund_cols_meta
    # Ensure all columns exist
    for col in all_cols:
        if col not in feature_matrix.columns:
            feature_matrix[col] = 0.0 if col in regime_cols else np.nan

    X = feature_matrix[all_cols].values.astype(np.float64)
    y = feature_matrix["label"].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)

    sample_weights = compute_sample_weights(feature_matrix.index, half_life_days=252)

    print(f"\n[Step 3] Data ready: X={X.shape}, y={y.shape}")
    print(f"  Class balance: {y.mean():.3f} positive")

    # ── Step 4: Run CV to get OOF predictions ────────────────────────────
    print(f"\n[Step 4] Running Purged K-Fold CV to get OOF predictions ...")
    _, _, oof_predictions = train_lgb_model(
        X, y, sample_weights,
        feature_names=all_cols,
        n_splits=10,
        horizon_days=horizon_days,
        embargo_days=10,
    )

    # ── Step 5: Compute conformal scores with RAW prob (no calibrator) ───
    print(f"\n[Step 5] Computing conformal scores with RAW probabilities ...")
    conformal_raw = compute_conformal_scores(oof_predictions, y, calibrator=None)

    if not conformal_raw:
        raise SystemExit("Failed to compute conformal scores.")

    raw_q = conformal_raw["conformal_scores_quantiles"]
    print(f"\n  New conformal quantiles (raw-based):")
    for k, v in raw_q.items():
        print(f"    {k}: {v:.6f}")

    print(f"\n  Comparison:")
    print(f"    {'Quantile':<10} {'Calibrated':<15} {'Raw':<15} {'Delta':<10}")
    for k in raw_q:
        old_val = old_q.get(k, 0)
        new_val = raw_q[k]
        print(f"    {k:<10} {old_val:<15.6f} {new_val:<15.6f} {new_val - old_val:+.6f}")

    # ── Step 6: Update meta file ─────────────────────────────────────────
    print(f"\n[Step 6] Updating meta file: {meta_path}")

    # Save old calibrated quantiles for reference
    meta["conformal_scores_quantiles_calibrated"] = old_q

    # Replace main quantiles with raw-based
    meta["conformal_scores_quantiles"] = raw_q
    meta["conformal_n_samples"] = conformal_raw["conformal_n_samples"]
    meta["conformal_score_mean"] = conformal_raw["conformal_score_mean"]
    meta["conformal_score_std"] = conformal_raw["conformal_score_std"]
    meta["conformal_probability_type"] = "raw"

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✅ Meta updated with raw-based conformal quantiles")
    print(f"  Old calibrated quantiles saved as 'conformal_scores_quantiles_calibrated'")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
