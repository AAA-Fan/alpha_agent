#!/usr/bin/env python3
"""
Walk-Forward Backtest with Rolling Window Retraining

Implements a realistic walk-forward backtest where the model is retrained
every month using a rolling 5-year training window, then predicts the next
month's trades. This eliminates look-ahead bias and closely mirrors how
the system would operate in live trading.

Flow for each month:
  1. Train LightGBM on [T-5y, T) data
  2. Fit calibrator + conformal scores on OOF predictions
  3. Run backtest on [T, T+1m) using the freshly trained model
  4. Slide window forward by 1 month, repeat

Usage:
    python scripts/run_walk_forward_backtest.py --ticker AAPL --start 2025-01-01 --end 2025-12-31
    python scripts/run_walk_forward_backtest.py --ticker AAPL --start 2025-01-01 --end 2025-12-31 --train-years 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

# ── Import training pipeline functions ───────────────────────────────────
from pipelines.train_forecast_model import (
    download_training_data,
    compute_base_features,
    compute_regime_features,
    build_labels,
    compute_sample_weights,
    train_lgb_model,
    fit_isotonic_calibrator,
    compute_conformal_scores,
    BASE_FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    ALL_MACRO_FUNDAMENTAL_COLUMNS,
    LGB_PARAMS,
)

# ── Import agents and backtest components ────────────────────────────────
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.regime_agent import RegimeAgent
from agents.forecast_agent import ForecastAgent
from agents.risk_agent import RiskAgent
from utils.macro_fundamental_provider import (
    MacroFundamentalFeatureProvider,
    MACRO_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
)
from utils.yfinance_cache import get_historical_data
from backtest.engine import BacktestEngine, BacktestResult
from backtest.evaluator import BacktestEvaluator, BacktestReport


# ═══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest with rolling window retraining.",
    )
    parser.add_argument(
        "--ticker", type=str, required=True,
        help="Stock ticker (e.g., AAPL)",
    )
    parser.add_argument(
        "--start", type=str, required=True,
        help="Backtest start date (YYYY-MM-DD). First prediction month.",
    )
    parser.add_argument(
        "--end", type=str, required=True,
        help="Backtest end date (YYYY-MM-DD). Last prediction month end.",
    )
    parser.add_argument(
        "--train-years", type=int, default=5,
        help="Rolling training window size in years (default: 5)",
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Holding period in days (default: 5)",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=5.0,
        help="Transaction cost in basis points (default: 5.0)",
    )
    parser.add_argument(
        "--slippage-bps", type=float, default=5.0,
        help="Slippage in basis points (default: 5.0)",
    )
    parser.add_argument(
        "--warmup", type=int, default=60,
        help="Warmup days for feature calculation (default: 60)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of CV folds for training (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/backtest_results",
        help="Output directory for reports (default: data/backtest_results)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Monthly Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train_model_for_window(
    ticker: str,
    raw_data: Dict[str, pd.DataFrame],
    spy_data: Optional[pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    horizon_days: int = 5,
    n_splits: int = 5,
    embargo_days: int = 10,
    half_life: int = 252,
    verbose: bool = False,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any], Dict[str, Any]]:
    """Train a LightGBM model on the given time window.

    Args:
        ticker: Stock symbol.
        raw_data: Pre-downloaded OHLCV data dict.
        spy_data: SPY data for excess return labels.
        train_start: Training window start date.
        train_end: Training window end date (exclusive for labels).
        horizon_days: Prediction horizon in days.
        n_splits: Number of CV folds.
        embargo_days: Embargo days for purged CV.
        half_life: Sample weight half-life in days.
        verbose: Verbose output.

    Returns:
        (lgb_model, calibrator, meta_dict, cv_metrics)
    """
    data = raw_data.get(ticker)
    if data is None or data.empty:
        return None, None, {}, {}

    # Filter to training window
    data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
    if len(data) < 120:
        if verbose:
            print(f"    [skip] Only {len(data)} rows in training window (need >= 120)")
        return None, None, {}, {}

    # Step 1: Compute base features
    # Use absolute return labels (spy_data=None) to match the label semantic
    # with long/short trading decisions (validated in WF Stage 1-3 debug scripts)
    frame = compute_base_features(data)
    frame = build_labels(frame, horizon_days=horizon_days, spy_data=None)
    frame["ticker"] = ticker

    # Drop rows with NaN in required columns
    required_cols = BASE_FEATURE_COLUMNS + ["label"]
    frame = frame.dropna(subset=required_cols)

    if len(frame) < 100:
        if verbose:
            print(f"    [skip] Only {len(frame)} valid rows after feature computation")
        return None, None, {}, {}

    # Step 2: Compute regime features
    regime_tm_path = os.getenv(
        "REGIME_TRANSITION_MATRIX_PATH",
        "data/regime_transition_matrix.json",
    )
    regime_transition_matrix = None
    if os.path.exists(regime_tm_path):
        with open(regime_tm_path, "r") as f:
            regime_transition_matrix = json.load(f)

    frame = compute_regime_features(frame, transition_matrix=regime_transition_matrix)
    feature_columns = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS

    # Step 3: Fetch historical macro/fundamental features
    macro_fund_cols_used: List[str] = []
    provider = MacroFundamentalFeatureProvider(verbose=False)
    fm_start = frame.index.min().to_pydatetime()
    fm_end = frame.index.max().to_pydatetime()

    try:
        mf_hist_df = provider.extract_historical(
            stock_symbol=ticker,
            start_date=fm_start,
            end_date=fm_end,
        )

        _INTERMEDIATE_COLUMNS = [
            "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
            "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
        ]
        _ALL_MF_COLS = ALL_MACRO_FUNDAMENTAL_COLUMNS + _INTERMEDIATE_COLUMNS

        if mf_hist_df is not None and not mf_hist_df.empty:
            frame = frame.sort_index()
            mf_hist_df = mf_hist_df.sort_index()
            frame.index = pd.to_datetime(frame.index)
            mf_hist_df.index = pd.to_datetime(mf_hist_df.index)

            fm_reset = frame.reset_index()
            mf_reset = mf_hist_df.reset_index()
            fm_reset = fm_reset.rename(columns={fm_reset.columns[0]: "_merge_date"})
            mf_reset = mf_reset.rename(columns={mf_reset.columns[0]: "_merge_date"})

            merged = pd.merge_asof(
                fm_reset.sort_values("_merge_date"),
                mf_reset.sort_values("_merge_date"),
                on="_merge_date",
                direction="backward",
            )
            merged = merged.set_index("_merge_date")
            merged.index.name = frame.index.name

            for col in _ALL_MF_COLS:
                if col in merged.columns:
                    frame[col] = merged[col].values

            # Compute price-dependent features
            close = pd.to_numeric(frame["Close"], errors="coerce")

            if "_ttm_eps" in frame.columns:
                ttm_eps = frame["_ttm_eps"]
                valid = ttm_eps.notna() & (ttm_eps.abs() > 0.01)
                frame.loc[valid, "pe_ratio"] = close[valid] / ttm_eps[valid]

            if "_total_equity" in frame.columns and "_shares_outstanding" in frame.columns:
                equity = frame["_total_equity"]
                shares = frame["_shares_outstanding"]
                valid = equity.notna() & shares.notna() & (shares > 0)
                bvps = equity[valid] / shares[valid]
                bvps_valid = bvps.abs() > 0.01
                frame.loc[bvps_valid.index[bvps_valid], "pb_ratio"] = (
                    close[bvps_valid.index[bvps_valid]] / bvps[bvps_valid]
                )

            if "_ttm_revenue" in frame.columns and "_shares_outstanding" in frame.columns:
                ttm_rev = frame["_ttm_revenue"]
                shares = frame["_shares_outstanding"]
                valid = ttm_rev.notna() & shares.notna() & (shares > 0) & (ttm_rev > 0)
                rps = ttm_rev[valid] / shares[valid]
                frame.loc[valid, "ps_ratio"] = close[valid] / rps

            if all(c in frame.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                shares = frame["_shares_outstanding"]
                ttm_ebitda = frame["_ttm_ebitda"]
                total_liab = frame["_total_liabilities"].fillna(0)
                cash = frame["_cash"].fillna(0)
                valid = shares.notna() & (shares > 0) & ttm_ebitda.notna() & (ttm_ebitda.abs() > 0)
                market_cap = close[valid] * shares[valid]
                ev = market_cap + total_liab[valid] - cash[valid]
                frame.loc[valid, "ev_ebitda"] = ev / ttm_ebitda[valid]

            # Beta
            try:
                spy_cache_file = Path("data/training_cache/SPY_daily.csv")
                if spy_cache_file.exists():
                    spy_df = pd.read_csv(spy_cache_file, index_col=0, parse_dates=True)
                else:
                    spy_df = get_historical_data("SPY", interval="daily", outputsize="full")
                if not spy_df.empty:
                    spy_close = pd.to_numeric(spy_df["Close"], errors="coerce")
                    spy_returns = spy_close.pct_change()
                    stock_returns = close.pct_change()
                    aligned = pd.DataFrame({"stock": stock_returns, "spy": spy_returns}).dropna()
                    if len(aligned) > 60:
                        rolling_cov = aligned["stock"].rolling(252, min_periods=60).cov(aligned["spy"])
                        rolling_var = aligned["spy"].rolling(252, min_periods=60).var()
                        rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
                        frame["beta"] = rolling_beta.reindex(frame.index).ffill()
            except Exception:
                pass

            # PEG ratio
            if "pe_ratio" in frame.columns and "earnings_growth_yoy" in frame.columns:
                pe = frame["pe_ratio"]
                eg = frame["earnings_growth_yoy"]
                eg_pct = eg * 100
                valid = pe.notna() & eg_pct.notna() & (eg_pct.abs() > 1.0) & (pe > 0)
                frame.loc[valid, "peg_ratio"] = pe[valid] / eg_pct[valid]

            # Drop intermediate columns
            for col in _INTERMEDIATE_COLUMNS:
                if col in frame.columns:
                    frame.drop(columns=[col], inplace=True)
        else:
            for col in ALL_MACRO_FUNDAMENTAL_COLUMNS:
                frame[col] = np.nan
    except Exception as exc:
        if verbose:
            print(f"    [warn] Macro/fundamental fetch failed: {exc}")
        for col in ALL_MACRO_FUNDAMENTAL_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan

    # Determine which macro/fund columns have variance
    from utils.macro_fundamental_provider import ALL_MACRO_FUNDAMENTAL_COLUMNS as MF_COLS
    for col in MF_COLS:
        if col in frame.columns:
            col_std = frame[col].std()
            if col_std is not None and col_std > 1e-10:
                macro_fund_cols_used.append(col)

    feature_columns = feature_columns + macro_fund_cols_used

    # Step 4: Sample weights
    sample_weights = compute_sample_weights(frame.index, half_life_days=half_life)

    # Step 5: Train
    X = frame[feature_columns].values.astype(np.float64)
    y = frame["label"].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)

    if verbose:
        print(f"    Training: X={X.shape}, y={y.shape}, pos_rate={y.mean():.3f}")

    try:
        final_model, cv_metrics, oof_predictions = train_lgb_model(
            X, y, sample_weights,
            feature_names=feature_columns,
            n_splits=n_splits,
            horizon_days=horizon_days,
            embargo_days=embargo_days,
        )
    except Exception as exc:
        if verbose:
            print(f"    [error] Training failed: {exc}")
        return None, None, {}, {}

    # Step 6: Calibration + conformal
    calibrator = fit_isotonic_calibrator(oof_predictions, y)
    conformal_info = compute_conformal_scores(oof_predictions, y, calibrator)

    # Build meta dict
    meta = {
        "version": 4,
        "model_type": "lightgbm",
        "label_type": "absolute_return",
        "benchmark": "SPY",
        "calibrated": calibrator is not None,
        **conformal_info,
        "target_horizon_days": horizon_days,
        "tickers": [ticker],
        "feature_columns": BASE_FEATURE_COLUMNS,
        "regime_features": REGIME_FEATURE_COLUMNS,
        "macro_fundamental_features": macro_fund_cols_used,
        "rank_feature_columns": [],
        "categorical_features": [],
        "cv_metrics": cv_metrics,
        "training_info": {
            "total_samples": len(frame),
            "date_range": [
                frame.index.min().strftime("%Y-%m-%d"),
                frame.index.max().strftime("%Y-%m-%d"),
            ],
            "class_balance": {
                "positive": round(float(y.mean()), 4),
                "negative": round(float(1 - y.mean()), 4),
            },
        },
    }

    return final_model, calibrator, meta, cv_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Backtest Engine (Monthly Slices)
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest_for_month(
    ticker: str,
    lgb_model: Any,
    calibrator: Any,
    meta: Dict[str, Any],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    horizon_days: int = 5,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    warmup_days: int = 60,
    verbose: bool = False,
) -> BacktestResult:
    """Run backtest for a single month using the provided model.

    Temporarily saves the model/meta/calibrator to disk so that
    ForecastAgent can load them via its standard path-based loading.
    """
    # Create temporary directory for this month's model files
    with tempfile.TemporaryDirectory(prefix="wf_backtest_") as tmpdir:
        tmp_model_path = os.path.join(tmpdir, "forecast_model.lgb")
        tmp_meta_path = os.path.join(tmpdir, "forecast_model_meta.json")
        tmp_cal_path = os.path.join(tmpdir, "forecast_calibrator.pkl")

        # Save model files
        lgb_model.save_model(tmp_model_path)
        meta_copy = {**meta, "model_path": tmp_model_path, "calibrator_path": tmp_cal_path}
        with open(tmp_meta_path, "w") as f:
            json.dump(meta_copy, f, indent=2, default=str)
        if calibrator is not None:
            with open(tmp_cal_path, "wb") as f:
                pickle.dump(calibrator, f)

        # Override environment variables for this month
        old_model_path = os.environ.get("FORECAST_LGB_MODEL_PATH")
        old_meta_path = os.environ.get("FORECAST_LGB_META_PATH")
        old_cal_path = os.environ.get("FORECAST_CALIBRATOR_PATH")

        os.environ["FORECAST_LGB_MODEL_PATH"] = tmp_model_path
        os.environ["FORECAST_LGB_META_PATH"] = tmp_meta_path
        os.environ["FORECAST_CALIBRATOR_PATH"] = tmp_cal_path

        try:
            # Initialize agents with the temporary model
            feature_agent = FeatureEngineeringAgent(verbose=verbose)
            regime_agent = RegimeAgent(verbose=verbose)
            forecast_agent = ForecastAgent(verbose=verbose)
            risk_agent = RiskAgent(verbose=verbose)
            macro_fund_provider = MacroFundamentalFeatureProvider(verbose=verbose)

            # Initialize backtest engine
            engine = BacktestEngine(
                feature_agent=feature_agent,
                regime_agent=regime_agent,
                forecast_agent=forecast_agent,
                risk_agent=risk_agent,
                macro_fund_provider=macro_fund_provider,
                horizon_days=horizon_days,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                verbose=verbose,
            )

            # Run backtest for this month
            result = engine.run(
                ticker=ticker,
                start_date=month_start.strftime("%Y-%m-%d"),
                end_date=month_end.strftime("%Y-%m-%d"),
                warmup_days=warmup_days,
            )

            return result

        finally:
            # Restore original environment variables
            if old_model_path is not None:
                os.environ["FORECAST_LGB_MODEL_PATH"] = old_model_path
            elif "FORECAST_LGB_MODEL_PATH" in os.environ:
                del os.environ["FORECAST_LGB_MODEL_PATH"]
            if old_meta_path is not None:
                os.environ["FORECAST_LGB_META_PATH"] = old_meta_path
            elif "FORECAST_LGB_META_PATH" in os.environ:
                del os.environ["FORECAST_LGB_META_PATH"]
            if old_cal_path is not None:
                os.environ["FORECAST_CALIBRATOR_PATH"] = old_cal_path
            elif "FORECAST_CALIBRATOR_PATH" in os.environ:
                del os.environ["FORECAST_CALIBRATOR_PATH"]


# ═══════════════════════════════════════════════════════════════════════════
# Result Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_monthly_results(
    monthly_results: List[Dict[str, Any]],
    ticker: str,
    start_date: str,
    end_date: str,
    horizon_days: int,
    params: Dict[str, Any],
) -> Tuple[BacktestResult, List[Dict[str, Any]]]:
    """Aggregate monthly backtest results into a single BacktestResult.

    Returns:
        (aggregated_result, monthly_summaries)
    """
    all_trades: List[Dict[str, Any]] = []
    monthly_summaries: List[Dict[str, Any]] = []

    for month_info in monthly_results:
        result: BacktestResult = month_info["result"]
        cv_metrics = month_info.get("cv_metrics", {})
        train_window = month_info.get("train_window", "")

        # Collect trades
        all_trades.extend(result.trade_log)

        # Monthly summary
        executed = [t for t in result.trade_log if t.get("position_size", 0) != 0]
        returns = [t["net_return"] for t in executed]
        monthly_return = sum(returns) if returns else 0.0
        hit_rate = (
            sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
        )

        monthly_summaries.append({
            "month": month_info["month"],
            "train_window": train_window,
            "cv_auc": cv_metrics.get("mean_auc", 0.0),
            "total_trades": len(result.trade_log),
            "executed_trades": len(executed),
            "monthly_return": round(monthly_return, 6),
            "hit_rate": round(hit_rate, 4),
        })

    # Build aggregated equity curve
    equity = 1.0
    equity_curve: Dict[str, float] = {}
    for trade in all_trades:
        date = trade["date"]
        net_return = trade.get("net_return", 0.0)
        equity *= 1.0 + net_return
        equity_curve[str(date)] = equity

    if equity_curve:
        eq_series = pd.Series(equity_curve, name="equity")
        eq_series.index = pd.to_datetime(eq_series.index)
    else:
        eq_series = pd.Series(dtype=float, name="equity")

    # Build benchmark curve
    try:
        spy_data = get_historical_data("SPY", interval="daily", outputsize="full")
        if not isinstance(spy_data.index, pd.DatetimeIndex):
            spy_data.index = pd.to_datetime(spy_data.index)
        spy_data = spy_data.sort_index()
        spy_data = spy_data.loc[start_date:end_date]
        if not spy_data.empty:
            spy_close = pd.to_numeric(spy_data["Close"], errors="coerce")
            benchmark = spy_close / spy_close.iloc[0]
            benchmark.name = "benchmark"
        else:
            benchmark = pd.Series(dtype=float, name="benchmark")
    except Exception:
        benchmark = pd.Series(dtype=float, name="benchmark")

    aggregated = BacktestResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        horizon_days=horizon_days,
        trade_log=all_trades,
        equity_curve=eq_series,
        benchmark_curve=benchmark,
        params=params,
        warnings=["Walk-forward backtest with monthly retraining (no look-ahead bias)."],
    )

    return aggregated, monthly_summaries


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()
    args = parse_args()

    ticker = args.ticker.strip().upper()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    train_years = args.train_years

    print("=" * 70)
    print("  Walk-Forward Backtest with Rolling Window Retraining")
    print("=" * 70)
    print(f"  Ticker:          {ticker}")
    print(f"  Backtest period: {args.start} → {args.end}")
    print(f"  Training window: {train_years} years (rolling)")
    print(f"  Retrain freq:    monthly")
    print(f"  Horizon:         {args.horizon}d")
    print(f"  Cost:            {args.cost_bps}bps | Slippage: {args.slippage_bps}bps")
    print(f"  CV folds:        {args.cv_folds}")
    print()

    # ── Step 1: Generate monthly schedule ────────────────────────────────
    # Each month: train on [month_start - train_years, month_start), predict [month_start, month_end)
    months: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_date.replace(day=1)  # Align to month start
    while current < end_date:
        month_end = current + relativedelta(months=1) - pd.Timedelta(days=1)
        month_end = min(month_end, end_date)
        months.append((current, month_end))
        current = current + relativedelta(months=1)

    print(f"  Monthly schedule: {len(months)} months")
    for i, (ms, me) in enumerate(months):
        print(f"    Month {i+1}: {ms.strftime('%Y-%m-%d')} → {me.strftime('%Y-%m-%d')}")
    print()

    # ── Step 2: Download all required data upfront ───────────────────────
    # We need data from (earliest_train_start) to (end_date)
    earliest_train_start = months[0][0] - pd.DateOffset(years=train_years)
    print(f"[Step 1] Downloading data ({earliest_train_start.strftime('%Y-%m-%d')} → {args.end}) ...")

    raw_data = download_training_data([ticker])
    if ticker not in raw_data or raw_data[ticker].empty:
        print(f"[ERROR] No data available for {ticker}")
        sys.exit(1)

    # Download SPY for excess return labels
    spy_data_dict = download_training_data(["SPY"])
    spy_data = spy_data_dict.get("SPY")
    if spy_data is not None and not spy_data.empty:
        print(f"  SPY data: {len(spy_data)} rows")
    else:
        print("  [warn] SPY data not available, will use absolute return labels")
        spy_data = None

    print(f"  {ticker} data: {len(raw_data[ticker])} rows")
    print()

    # ── Step 3: Walk-forward loop ────────────────────────────────────────
    monthly_results: List[Dict[str, Any]] = []
    total_start_time = time.time()

    for month_idx, (month_start, month_end) in enumerate(months):
        train_start = month_start - pd.DateOffset(years=train_years)
        train_end = month_start - pd.Timedelta(days=1)  # Exclusive: don't include prediction month

        print(f"\n{'='*70}")
        print(f"  Month {month_idx + 1}/{len(months)}: {month_start.strftime('%Y-%m-%d')} → {month_end.strftime('%Y-%m-%d')}")
        print(f"  Training window: {train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
        print(f"{'='*70}")

        month_time = time.time()

        # ── 3a: Train model for this window ──────────────────────────
        print(f"\n  [Train] Training model on {train_years}-year window ...")
        lgb_model, calibrator, meta, cv_metrics = train_model_for_window(
            ticker=ticker,
            raw_data=raw_data,
            spy_data=spy_data,
            train_start=train_start,
            train_end=train_end,
            horizon_days=args.horizon,
            n_splits=args.cv_folds,
            verbose=args.verbose,
        )

        if lgb_model is None:
            print(f"  [SKIP] Training failed for this window, skipping month")
            continue

        cv_auc = cv_metrics.get("mean_auc", 0.0)
        print(f"  [Train] CV AUC: {cv_auc:.4f}")

        # ── 3b: Run backtest for this month ──────────────────────────
        print(f"\n  [Backtest] Running backtest for {month_start.strftime('%Y-%m')} ...")
        try:
            result = run_backtest_for_month(
                ticker=ticker,
                lgb_model=lgb_model,
                calibrator=calibrator,
                meta=meta,
                month_start=month_start,
                month_end=month_end,
                horizon_days=args.horizon,
                transaction_cost_bps=args.cost_bps,
                slippage_bps=args.slippage_bps,
                warmup_days=args.warmup,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"  [ERROR] Backtest failed: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

        # Monthly summary
        executed = [t for t in result.trade_log if t.get("position_size", 0) != 0]
        returns = [t["net_return"] for t in executed]
        monthly_return = sum(returns) if returns else 0.0
        hit_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0

        elapsed = time.time() - month_time
        print(f"\n  [Result] Month {month_start.strftime('%Y-%m')}:")
        print(f"    Trades: {len(executed)} executed / {len(result.trade_log)} total")
        print(f"    Return: {monthly_return:+.4f} ({monthly_return*100:+.2f}%)")
        print(f"    Hit rate: {hit_rate:.2%}")
        print(f"    CV AUC: {cv_auc:.4f}")
        print(f"    Time: {elapsed:.1f}s")

        monthly_results.append({
            "month": month_start.strftime("%Y-%m"),
            "result": result,
            "cv_metrics": cv_metrics,
            "train_window": f"{train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}",
        })

    # ── Step 4: Aggregate results ────────────────────────────────────────
    if not monthly_results:
        print("\n[ERROR] No monthly results generated. Check data availability.")
        sys.exit(1)

    print(f"\n\n{'='*70}")
    print("  Aggregating Results")
    print(f"{'='*70}")

    aggregated, monthly_summaries = aggregate_monthly_results(
        monthly_results=monthly_results,
        ticker=ticker,
        start_date=args.start,
        end_date=args.end,
        horizon_days=args.horizon,
        params={
            "transaction_cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
            "warmup_days": args.warmup,
            "train_years": train_years,
            "retrain_frequency": "monthly",
            "cv_folds": args.cv_folds,
        },
    )

    # ── Step 5: Evaluate ─────────────────────────────────────────────────
    evaluator = BacktestEvaluator()
    report = evaluator.evaluate(aggregated)

    # ── Step 6: Print results ────────────────────────────────────────────
    total_elapsed = time.time() - total_start_time

    print(f"\n{'='*70}")
    print("  Walk-Forward Backtest Results")
    print(f"{'='*70}")
    print(f"  Ticker:          {ticker}")
    print(f"  Period:          {args.start} → {args.end}")
    print(f"  Training:        {train_years}y rolling, monthly retrain")
    print(f"  Total time:      {total_elapsed:.1f}s")
    print()

    # Overall metrics
    o = report.overall
    print(f"  {'─'*50}")
    print(f"  Overall Performance:")
    print(f"    Total return:      {o.get('total_return', 0):+.4f} ({o.get('total_return', 0)*100:+.2f}%)")
    print(f"    Annualized return: {o.get('annualized_return', 0):+.4f} ({o.get('annualized_return', 0)*100:+.2f}%)")
    print(f"    Buy & Hold (SPY):  {o.get('buy_and_hold_return', 0):+.4f} ({o.get('buy_and_hold_return', 0)*100:+.2f}%)")
    print(f"    Alpha:             {o.get('alpha', 0):+.4f} ({o.get('alpha', 0)*100:+.2f}%)")
    print(f"    Sharpe ratio:      {o.get('sharpe_ratio', 0):.4f}")
    print(f"    Sortino ratio:     {o.get('sortino_ratio', 0):.4f}")
    print(f"    Calmar ratio:      {o.get('calmar_ratio', 0):.4f}")
    print(f"    Max drawdown:      {o.get('max_drawdown', 0):.4f} ({o.get('max_drawdown', 0)*100:.2f}%)")
    print(f"    Hit rate:          {o.get('hit_rate', 0):.4f} ({o.get('hit_rate', 0)*100:.2f}%)")
    print(f"    Profit factor:     {o.get('profit_factor', 0):.4f}")
    print(f"    Total trades:      {o.get('total_trades', 0)}")
    print(f"    Rejected trades:   {o.get('rejected_trades', 0)}")
    print()

    # Monthly breakdown
    print(f"  {'─'*50}")
    print(f"  Monthly Breakdown:")
    print(f"  {'Month':<10} {'Train Window':<30} {'CV AUC':>8} {'Trades':>7} {'Return':>10} {'Hit Rate':>10}")
    print(f"  {'─'*10} {'─'*30} {'─'*8} {'─'*7} {'─'*10} {'─'*10}")
    for ms in monthly_summaries:
        print(
            f"  {ms['month']:<10} {ms['train_window']:<30} "
            f"{ms['cv_auc']:>8.4f} {ms['executed_trades']:>7d} "
            f"{ms['monthly_return']:>+10.4f} {ms['hit_rate']:>10.2%}"
        )
    print()

    # Per-regime breakdown
    if report.per_regime:
        print(f"  {'─'*50}")
        print("  Per-Regime Breakdown:")
        for state, metrics in sorted(report.per_regime.items()):
            print(
                f"    {state:20s} | trades={metrics['trade_count']:3d} | "
                f"hit_rate={metrics['hit_rate']:.2%} | "
                f"avg_ret={metrics['avg_return']:+.4f} | "
                f"contribution={metrics['contribution_to_total']:+.2%}"
            )
        print()

    # Exit analysis
    exit_info = report.exit_analysis
    print(f"  {'─'*50}")
    print("  Exit Analysis:")
    for reason, pct in exit_info.get("exit_distribution", {}).items():
        avg_ret = exit_info.get("exit_avg_return", {}).get(reason, 0)
        print(f"    {reason:15s} | {pct:.1%} of trades | avg_return={avg_ret:+.4f}")
    print(f"    Stop-loss effectiveness: {exit_info.get('stop_loss_effectiveness', 'n/a')}")
    print()

    # Signal quality
    sq = report.signal_quality
    print(f"  {'─'*50}")
    print("  Signal Quality:")
    print(f"    Total signals: {sq['total_signals']} | Executed: {sq['executed_trades']} | "
          f"Rejected: {sq['rejected_trades']} ({sq['rejection_rate']:.1%})")
    if sq.get("reject_reasons"):
        for reason, count in sq["reject_reasons"].items():
            print(f"      - {reason}: {count}")
    print()

    # ── Step 7: Save outputs ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"wf_{ticker}_{args.start}_{args.end}"

    # Save report JSON
    report_path = output_dir / f"{base_name}_report.json"
    report_dict = report.to_dict()
    report_dict["walk_forward"] = {
        "train_years": train_years,
        "retrain_frequency": "monthly",
        "monthly_summaries": monthly_summaries,
    }
    report_dict["warnings"] = aggregated.warnings
    report_dict["params"] = aggregated.params
    report_dict["ticker"] = ticker
    report_dict["start_date"] = args.start
    report_dict["end_date"] = args.end
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"  [✓] Report saved: {report_path}")

    # Save trade log CSV
    trades_path = output_dir / f"{base_name}_trades.csv"
    if aggregated.trade_log:
        trades_df = pd.DataFrame(aggregated.trade_log)
        if "risk_flags" in trades_df.columns:
            trades_df["risk_flags"] = trades_df["risk_flags"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else str(x)
            )
        trades_df.to_csv(trades_path, index=False)
        print(f"  [✓] Trade log saved: {trades_path}")

    # Save equity curve CSV
    equity_path = output_dir / f"{base_name}_equity.csv"
    if not aggregated.equity_curve.empty:
        eq_df = pd.DataFrame({
            "date": aggregated.equity_curve.index,
            "equity": aggregated.equity_curve.values,
        })
        eq_df.to_csv(equity_path, index=False)
        print(f"  [✓] Equity curve saved: {equity_path}")

    # Save monthly summary CSV
    monthly_path = output_dir / f"{base_name}_monthly.csv"
    monthly_df = pd.DataFrame(monthly_summaries)
    monthly_df.to_csv(monthly_path, index=False)
    print(f"  [✓] Monthly summary saved: {monthly_path}")

    print(f"\n{'='*70}")
    print("  Walk-Forward Backtest Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
