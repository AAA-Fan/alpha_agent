#!/usr/bin/env python3
"""
Stage 0: Pure Signal IC (Information Coefficient) Analysis

Goal: Verify whether the LightGBM model has any predictive power at all,
      BEFORE involving any Agent pipeline.

What it does:
  1. Load historical OHLCV data for a ticker (or multiple tickers)
  2. Walk forward day-by-day, compute features, score with LightGBM
  3. Record predicted_probability_up vs actual_5d_return
  4. Compute:
     - Daily IC  = rank_corr(predicted_prob, actual_5d_return)
     - Mean IC, IC_IR (mean/std), IC hit rate (% of days IC > 0)
     - Long-short spread: avg return of top-quintile vs bottom-quintile signals
     - Calibration: predicted prob vs actual win rate in bins

Usage:
    python scripts/debug_stage0_signal_ic.py --ticker AAPL --start 2023-01-01 --end 2025-12-31
    python scripts/debug_stage0_signal_ic.py --ticker AAPL,MSFT,GOOGL --start 2023-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv


# ── Feature computation (standalone, no Agent dependency) ────────────────

def compute_features(data: pd.DataFrame) -> dict:
    """Compute all base + interaction features from OHLCV data.

    This replicates the training pipeline's feature computation exactly,
    including the 5 interaction features that FeatureEngineeringAgent misses.
    """
    if len(data) < 60:
        return {}

    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    open_ = pd.to_numeric(data["Open"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")
    returns = close.pct_change()

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))

    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    vol_daily_20 = returns.rolling(20).std()
    momentum_5 = close.pct_change(5)
    momentum_20 = close.pct_change(20)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    volume_z_20 = (volume - vol_mean) / vol_std

    rolling_max_60 = close.rolling(60).max()
    drawdown_60 = (close / rolling_max_60) - 1

    overnight_gap = (open_ / prev_close) - 1
    intraday_return = (close / open_) - 1

    def _safe(val):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0.0
        return float(val)

    f = {
        "momentum_5": _safe(momentum_5.iloc[-1]),
        "momentum_20": _safe(momentum_20.iloc[-1]),
        "sma_20_ratio": _safe((close.iloc[-1] / sma_20.iloc[-1]) - 1),
        "sma_50_ratio": _safe((close.iloc[-1] / sma_50.iloc[-1]) - 1),
        "macd_hist": _safe(macd_hist.iloc[-1]),
        "rsi_14": _safe(rsi_14.iloc[-1]),
        "volatility_20": _safe(vol_20.iloc[-1]),
        "daily_volatility_20": _safe(vol_daily_20.iloc[-1]),
        "atr_14": _safe(atr_14.iloc[-1]),
        "volume_zscore_20": _safe(volume_z_20.iloc[-1]),
        "drawdown_60": _safe(drawdown_60.iloc[-1]),
        "overnight_gap": _safe(overnight_gap.iloc[-1]),
        "intraday_return": _safe(intraday_return.iloc[-1]),
        "return_1d": _safe(returns.iloc[-1]),
        "return_5d": _safe(close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 5 else 0),
    }

    # ── Interaction features (CRITICAL: these are missing in FeatureEngineeringAgent!) ──
    f["momentum_trend_align"] = f["momentum_5"] * f["sma_20_ratio"]
    f["rsi_deviation"] = (f["rsi_14"] - 50.0) / 50.0
    vol_z_clipped = max(-3, min(3, f["volume_zscore_20"]))
    f["vol_confirmed_momentum"] = f["momentum_5"] * vol_z_clipped
    m5_clipped = max(-0.1, min(0.1, f["momentum_5"]))
    f["mean_reversion"] = f["drawdown_60"] * m5_clipped
    safe_vol = f["daily_volatility_20"] if f["daily_volatility_20"] != 0 else None
    if safe_vol:
        f["vol_adj_momentum_5"] = max(-5, min(5, f["momentum_5"] / safe_vol))
    else:
        f["vol_adj_momentum_5"] = 0.0

    return f


def run_signal_ic_analysis(
    ticker: str,
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Walk forward and collect (date, predicted_prob, actual_5d_return) tuples."""

    import lightgbm as lgb
    import pickle

    # Load model
    lgb_path = os.getenv("FORECAST_LGB_MODEL_PATH", "data/forecast_model_v2.lgb")
    meta_path = os.getenv("FORECAST_LGB_META_PATH", "data/forecast_model_v2_meta.json")

    model = lgb.Booster(model_file=lgb_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load calibrator
    calibrator = None
    cal_path = meta.get("calibrator_path", "data/forecast_calibrator_v2.pkl")
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

    feature_cols = meta.get("feature_columns", [])
    regime_cols = meta.get("regime_features", [])
    macro_fund_cols = meta.get("macro_fundamental_features", [])
    rank_cols = meta.get("rank_feature_columns", [])
    cat_cols = meta.get("categorical_features", [])
    all_cols = feature_cols + regime_cols + macro_fund_cols + rank_cols + cat_cols

    # Load historical data
    from utils.yfinance_cache import get_historical_data
    data = get_historical_data(ticker, interval="daily", outputsize="full")
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # Clip to range with buffer
    buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=120)
    end_ts = pd.Timestamp(end_date)
    data = data.loc[buffer_start:end_ts]

    close_prices = pd.to_numeric(data["Close"], errors="coerce")
    dates = data.index

    # Find start index
    start_idx = dates.searchsorted(pd.Timestamp(start_date))
    start_idx = max(start_idx, 60)  # Need at least 60 bars for features
    horizon = meta.get("target_horizon_days", 5)

    records = []
    total_steps = len(dates) - start_idx - horizon
    step = 0

    for t in range(start_idx, len(dates) - horizon):
        step += 1
        current_date = dates[t]

        # Compute features from data up to day t (no look-ahead)
        data_slice = data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            continue

        # Build feature row
        row = {}
        for col in feature_cols:
            row[col] = features.get(col, 0.0)

        # Regime features: use defaults (neutral) since we're bypassing RegimeAgent
        for col in regime_cols:
            row[col] = 0.0  # neutral defaults

        # Macro/fundamental features: use defaults
        for col in macro_fund_cols:
            row[col] = 0.0

        # Rank features: use median (0.5) since we're testing single-ticker
        for col in rank_cols:
            row[col] = 0.5

        # Categorical features: use 0
        for col in cat_cols:
            row[col] = 0.0

        X = pd.DataFrame([row])[all_cols]
        raw_prob = float(model.predict(X)[0])

        # Apply calibration
        if calibrator is not None:
            try:
                prob = float(calibrator.predict([raw_prob])[0])
            except Exception:
                prob = raw_prob
        else:
            prob = raw_prob

        # Soft-clamp: prevent Isotonic calibrator from over-polarizing
        # Blend calibrated output with raw probability to preserve ranking
        if calibrator is not None:
            _BLEND_ALPHA = 0.7  # 70% calibrated, 30% raw
            prob = _BLEND_ALPHA * prob + (1 - _BLEND_ALPHA) * raw_prob
            prob = max(0.02, min(0.98, prob))

        # Actual 5-day forward return
        future_close = close_prices.iloc[t + horizon]
        current_close = close_prices.iloc[t]
        actual_5d_return = (future_close / current_close) - 1.0

        records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted_prob_up": prob,
            "raw_prob_up": raw_prob,
            "actual_5d_return": float(actual_5d_return),
            "actual_direction": 1 if actual_5d_return > 0 else 0,
        })

        if verbose and step % 50 == 0:
            print(f"  [{ticker}] Step {step}/{total_steps}: date={current_date.strftime('%Y-%m-%d')}, prob={prob:.4f}, actual_5d={actual_5d_return:+.4f}")

    df = pd.DataFrame(records)
    print(f"  [{ticker}] Collected {len(df)} signal-return pairs")
    return df


def analyze_ic(df: pd.DataFrame, ticker: str) -> dict:
    """Compute IC metrics from signal-return pairs."""
    if df.empty or len(df) < 10:
        print(f"  [{ticker}] Not enough data for IC analysis")
        return {}

    # ── 1. Rank IC (Spearman correlation) ────────────────────────────────
    # Daily IC: rank_corr(predicted_prob, actual_5d_return)
    overall_ic, overall_pval = stats.spearmanr(df["predicted_prob_up"], df["actual_5d_return"])

    # Rolling IC (20-day windows)
    rolling_ics = []
    window = 20
    for i in range(window, len(df)):
        chunk = df.iloc[i - window:i]
        ic, _ = stats.spearmanr(chunk["predicted_prob_up"], chunk["actual_5d_return"])
        if not np.isnan(ic):
            rolling_ics.append(ic)

    mean_ic = np.mean(rolling_ics) if rolling_ics else overall_ic
    std_ic = np.std(rolling_ics) if rolling_ics else 0.0
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    ic_hit_rate = np.mean([1 for ic in rolling_ics if ic > 0]) / len(rolling_ics) if rolling_ics else 0.0

    # ── 2. Directional accuracy ──────────────────────────────────────────
    df["predicted_direction"] = (df["predicted_prob_up"] > 0.5).astype(int)
    directional_accuracy = (df["predicted_direction"] == df["actual_direction"]).mean()

    # ── 3. Quintile analysis ─────────────────────────────────────────────
    df["quintile"] = pd.qcut(df["predicted_prob_up"], q=5, labels=False, duplicates="drop")
    quintile_returns = df.groupby("quintile")["actual_5d_return"].mean()
    long_short_spread = 0.0
    if len(quintile_returns) >= 2:
        long_short_spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]

    # ── 4. Calibration analysis ──────────────────────────────────────────
    n_bins = 10
    df["prob_bin"] = pd.cut(df["predicted_prob_up"], bins=n_bins, labels=False, duplicates="drop")
    calibration = df.groupby("prob_bin").agg(
        mean_predicted=("predicted_prob_up", "mean"),
        actual_win_rate=("actual_direction", "mean"),
        count=("actual_direction", "count"),
    ).reset_index()

    # ── 5. Signal distribution ───────────────────────────────────────────
    buy_signals = (df["predicted_prob_up"] > 0.55).sum()
    sell_signals = (df["predicted_prob_up"] < 0.45).sum()
    hold_signals = len(df) - buy_signals - sell_signals

    # ── 6. Raw prob distribution ─────────────────────────────────────────
    raw_mean = df["raw_prob_up"].mean()
    raw_std = df["raw_prob_up"].std()
    raw_min = df["raw_prob_up"].min()
    raw_max = df["raw_prob_up"].max()

    results = {
        "ticker": ticker,
        "n_samples": len(df),
        "overall_ic": round(float(overall_ic), 6),
        "overall_ic_pval": round(float(overall_pval), 6),
        "mean_rolling_ic": round(float(mean_ic), 6),
        "std_rolling_ic": round(float(std_ic), 6),
        "ic_ir": round(float(ic_ir), 4),
        "ic_hit_rate": round(float(ic_hit_rate), 4),
        "directional_accuracy": round(float(directional_accuracy), 4),
        "long_short_spread_5d": round(float(long_short_spread), 6),
        "buy_signals": int(buy_signals),
        "sell_signals": int(sell_signals),
        "hold_signals": int(hold_signals),
        "raw_prob_mean": round(float(raw_mean), 6),
        "raw_prob_std": round(float(raw_std), 6),
        "raw_prob_range": [round(float(raw_min), 6), round(float(raw_max), 6)],
        "quintile_avg_returns": {str(k): round(float(v), 6) for k, v in quintile_returns.items()},
        "calibration": calibration.to_dict(orient="records"),
    }

    return results


def print_report(results: dict) -> None:
    """Pretty-print the IC analysis report."""
    ticker = results.get("ticker", "?")

    print(f"\n{'='*60}")
    print(f"  Stage 0: Signal IC Report — {ticker}")
    print(f"{'='*60}")

    print(f"\n📊 Sample Size: {results['n_samples']} signal-return pairs")

    # IC metrics
    print(f"\n── Information Coefficient (IC) ──")
    ic = results["overall_ic"]
    ic_ir = results["ic_ir"]
    ic_hit = results["ic_hit_rate"]

    # Color-code IC quality
    if abs(ic) > 0.05:
        ic_quality = "✅ GOOD (|IC| > 0.05)"
    elif abs(ic) > 0.02:
        ic_quality = "⚠️  WEAK (0.02 < |IC| < 0.05)"
    else:
        ic_quality = "❌ NO SIGNAL (|IC| < 0.02)"

    print(f"  Overall IC:     {ic:+.6f}  (p-value: {results['overall_ic_pval']:.4f})")
    print(f"  Mean Rolling IC: {results['mean_rolling_ic']:+.6f}")
    print(f"  IC_IR:          {ic_ir:+.4f}  {'✅ > 0.5' if abs(ic_ir) > 0.5 else '❌ < 0.5'}")
    print(f"  IC Hit Rate:    {ic_hit:.2%}  {'✅ > 50%' if ic_hit > 0.5 else '❌ ≤ 50%'}")
    print(f"  Quality:        {ic_quality}")

    # Directional accuracy
    print(f"\n── Directional Accuracy ──")
    acc = results["directional_accuracy"]
    print(f"  Accuracy:       {acc:.2%}  {'✅ > 52%' if acc > 0.52 else '❌ ≤ 52%'}")

    # Long-short spread
    print(f"\n── Quintile Analysis (Long-Short) ──")
    spread = results["long_short_spread_5d"]
    print(f"  L/S Spread (5d): {spread:+.6f}  {'✅ > 0' if spread > 0 else '❌ ≤ 0'}")
    print(f"  Quintile avg returns:")
    for q, ret in sorted(results["quintile_avg_returns"].items()):
        bar = "█" * max(1, int(abs(ret) * 5000))
        sign = "+" if ret > 0 else ""
        print(f"    Q{q}: {sign}{ret:.6f}  {bar}")

    # Signal distribution
    print(f"\n── Signal Distribution ──")
    total = results["n_samples"]
    print(f"  Buy  (prob > 0.55): {results['buy_signals']:4d} ({results['buy_signals']/total:.1%})")
    print(f"  Hold (0.45~0.55):   {results['hold_signals']:4d} ({results['hold_signals']/total:.1%})")
    print(f"  Sell (prob < 0.45): {results['sell_signals']:4d} ({results['sell_signals']/total:.1%})")

    # Raw probability distribution
    print(f"\n── Raw Probability Distribution ──")
    print(f"  Mean: {results['raw_prob_mean']:.6f}")
    print(f"  Std:  {results['raw_prob_std']:.6f}")
    print(f"  Range: [{results['raw_prob_range'][0]:.6f}, {results['raw_prob_range'][1]:.6f}]")

    # Calibration
    print(f"\n── Calibration (Predicted Prob vs Actual Win Rate) ──")
    for row in results.get("calibration", []):
        pred = row.get("mean_predicted", 0)
        actual = row.get("actual_win_rate", 0)
        count = row.get("count", 0)
        gap = actual - pred
        print(f"    Predicted={pred:.3f} | Actual={actual:.3f} | Gap={gap:+.3f} | n={count}")

    # Overall verdict
    print(f"\n{'─'*60}")
    issues = []
    if abs(ic) < 0.02:
        issues.append("IC near zero → model has no predictive power")
    if acc <= 0.50:
        issues.append("Directional accuracy ≤ 50% → worse than coin flip")
    if spread <= 0:
        issues.append("Long-short spread ≤ 0 → no monotonic signal")
    if results["raw_prob_std"] < 0.01:
        issues.append("Raw prob std < 0.01 → model outputs are nearly constant (degenerate)")
    if results["buy_signals"] == 0 and results["sell_signals"] == 0:
        issues.append("No buy/sell signals → model always outputs ~0.5 (no conviction)")

    if not issues:
        print("  ✅ VERDICT: Model shows predictive signal. Proceed to Stage 1.")
    else:
        print("  ❌ VERDICT: Model has fundamental issues:")
        for issue in issues:
            print(f"     • {issue}")
        print("  → Fix the model before proceeding to Stage 1.")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 0: Pure Signal IC Analysis")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker(s), comma-separated")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    print(f"\n{'='*60}")
    print(f"  Stage 0: Pure Signal IC Analysis")
    print(f"  Model: {os.getenv('FORECAST_LGB_MODEL_PATH', 'data/forecast_model_v2.lgb')}")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'='*60}")

    all_results = {}
    all_dfs = []

    for ticker in tickers:
        print(f"\n── Processing {ticker} ──")
        t0 = time.time()
        df = run_signal_ic_analysis(ticker, args.start, args.end, verbose=args.verbose)
        elapsed = time.time() - t0
        print(f"  [{ticker}] Completed in {elapsed:.1f}s")

        if not df.empty:
            results = analyze_ic(df, ticker)
            all_results[ticker] = results
            all_dfs.append(df)
            print_report(results)

    # Cross-ticker summary if multiple tickers
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  Cross-Ticker Summary")
        print(f"{'='*60}")
        print(f"  {'Ticker':8s} | {'IC':>10s} | {'IC_IR':>8s} | {'Accuracy':>10s} | {'L/S Spread':>12s}")
        print(f"  {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10} | {'-'*12}")
        for ticker, r in all_results.items():
            print(
                f"  {ticker:8s} | {r['overall_ic']:+10.6f} | {r['ic_ir']:+8.4f} | "
                f"{r['directional_accuracy']:10.2%} | {r['long_short_spread_5d']:+12.6f}"
            )

        # Pooled IC across all tickers
        if all_dfs:
            pooled = pd.concat(all_dfs, ignore_index=True)
            pooled_ic, pooled_pval = stats.spearmanr(pooled["predicted_prob_up"], pooled["actual_5d_return"])
            pooled_acc = (pooled["predicted_prob_up"].gt(0.5).astype(int) == pooled["actual_direction"]).mean()
            print(f"\n  Pooled IC:       {pooled_ic:+.6f} (p={pooled_pval:.4f})")
            print(f"  Pooled Accuracy: {pooled_acc:.2%}")

    # Save results
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stage0_signal_ic_{args.start}_{args.end}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[✓] Results saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
