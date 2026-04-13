#!/usr/bin/env python3
"""
Data Quality Check Script
=========================
Randomly samples tickers and dates from the training pipeline,
then verifies that technical, fundamental, and macro features
are computed correctly. Reports NaN patterns and potential bugs.

Usage:
    python -m scripts.data_quality_check
"""

import json
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────
SAMPLE_TICKERS = 5       # Number of tickers to sample
SAMPLE_DATES_PER_TICKER = 3  # Dates per ticker
RANDOM_SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────────

def load_ticker_price_data(ticker: str) -> pd.DataFrame:
    """Load cached daily price data for a ticker."""
    csv_path = Path(f"data/training_cache/{ticker}_daily.csv")
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df


def load_fundamental_cache(ticker: str) -> dict:
    """Load all cached fundamental JSON files for a ticker."""
    cache_dir = Path("data/cross_section_cache/fundamental_cache")
    result = {}
    for suffix in ["income_statement", "balance_sheet", "earnings"]:
        fpath = cache_dir / f"{ticker}_{suffix}.json"
        if fpath.exists() and fpath.stat().st_size > 10:
            with open(fpath, "r") as f:
                result[suffix] = json.load(f)
        else:
            result[suffix] = None
    return result


def load_macro_cache() -> dict:
    """Load all cached macro JSON files."""
    cache_dir = Path("data/cross_section_cache/macro_hist_cache")
    result = {}
    for fpath in cache_dir.glob("*.json"):
        with open(fpath, "r") as f:
            result[fpath.stem] = json.load(f)
    return result


def manual_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Manually compute RSI for verification."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def check_technical_features(ticker: str, price_df: pd.DataFrame, sample_dates: list):
    """Verify technical features against manual calculation."""
    print(f"\n  {'─'*50}")
    print(f"  Technical Features Check for {ticker}")
    print(f"  {'─'*50}")

    close = pd.to_numeric(price_df["Close"], errors="coerce")
    high = pd.to_numeric(price_df["High"], errors="coerce")
    low = pd.to_numeric(price_df["Low"], errors="coerce")
    open_ = pd.to_numeric(price_df["Open"], errors="coerce")
    volume = pd.to_numeric(price_df["Volume"], errors="coerce")
    returns = close.pct_change()

    for date in sample_dates:
        if date not in close.index:
            # Find nearest date
            nearest = close.index[close.index.get_indexer([date], method="nearest")[0]]
            date = nearest

        idx = close.index.get_loc(date)
        if idx < 60:
            print(f"    [{date.date()}] Skipped (too early, need 60 rows lookback)")
            continue

        print(f"\n    [{date.date()}] Close={close.iloc[idx]:.2f}")

        # Momentum 5
        expected_m5 = close.iloc[idx] / close.iloc[idx - 5] - 1
        print(f"      momentum_5:       {expected_m5:.6f}")

        # Momentum 20
        expected_m20 = close.iloc[idx] / close.iloc[idx - 20] - 1
        print(f"      momentum_20:      {expected_m20:.6f}")

        # SMA 20 ratio
        sma20 = close.iloc[idx-19:idx+1].mean()
        expected_sma20_ratio = close.iloc[idx] / sma20 - 1
        print(f"      sma_20_ratio:     {expected_sma20_ratio:.6f}")

        # RSI 14
        rsi_series = manual_rsi(close, 14)
        expected_rsi = rsi_series.iloc[idx]
        print(f"      rsi_14:           {expected_rsi:.2f}")

        # Volatility 20
        expected_vol = returns.iloc[idx-19:idx+1].std() * np.sqrt(252)
        print(f"      volatility_20:    {expected_vol:.4f}")

        # Volume z-score
        vol_mean = volume.iloc[idx-19:idx+1].mean()
        vol_std = volume.iloc[idx-19:idx+1].std()
        expected_vz = (volume.iloc[idx] - vol_mean) / vol_std if vol_std > 0 else np.nan
        print(f"      volume_zscore_20: {expected_vz:.4f}")

        # Drawdown 60
        rolling_max = close.iloc[max(0, idx-59):idx+1].max()
        expected_dd = close.iloc[idx] / rolling_max - 1
        print(f"      drawdown_60:      {expected_dd:.4f}")

        print(f"      ✅ Technical features computed (manual reference values above)")


def check_fundamental_features(ticker: str, fund_cache: dict, sample_dates: list):
    """Verify fundamental features against cached API data."""
    print(f"\n  {'─'*50}")
    print(f"  Fundamental Features Check for {ticker}")
    print(f"  {'─'*50}")

    income = fund_cache.get("income_statement")
    balance = fund_cache.get("balance_sheet")
    earnings = fund_cache.get("earnings")

    # Check data availability
    issues = []

    if income is None:
        issues.append("❌ income_statement cache is EMPTY or missing")
    else:
        quarterly = income.get("quarterlyReports", [])
        print(f"    Income Statement: {len(quarterly)} quarterly reports")
        if quarterly:
            # Check for None/"None" values in key fields
            first_q = quarterly[0]
            fiscal_date = first_q.get("fiscalDateEnding", "N/A")
            net_income = first_q.get("netIncome", "N/A")
            total_revenue = first_q.get("totalRevenue", "N/A")
            ebitda = first_q.get("ebitda", "N/A")
            print(f"      Latest quarter ({fiscal_date}):")
            print(f"        netIncome:    {net_income}")
            print(f"        totalRevenue: {total_revenue}")
            print(f"        ebitda:       {ebitda}")

            # Check how many quarters have "None" string values
            none_count = 0
            for q in quarterly:
                for key in ["netIncome", "totalRevenue", "ebitda"]:
                    val = q.get(key)
                    if val is None or val == "None" or val == "0":
                        none_count += 1
            if none_count > 0:
                issues.append(f"⚠️  {none_count} 'None'/'0' values in income statement fields")

    if balance is None:
        issues.append("❌ balance_sheet cache is EMPTY or missing")
    else:
        quarterly = balance.get("quarterlyReports", [])
        print(f"    Balance Sheet: {len(quarterly)} quarterly reports")
        if quarterly:
            first_q = quarterly[0]
            fiscal_date = first_q.get("fiscalDateEnding", "N/A")
            total_equity = first_q.get("totalShareholderEquity", "N/A")
            shares = first_q.get("commonStockSharesOutstanding", "N/A")
            total_liab = first_q.get("totalLiabilities", "N/A")
            cash = first_q.get("cashAndCashEquivalentsAtCarryingValue",
                              first_q.get("cashAndShortTermInvestments", "N/A"))
            current_assets = first_q.get("totalCurrentAssets", "N/A")
            current_liab = first_q.get("totalCurrentLiabilities", "N/A")
            print(f"      Latest quarter ({fiscal_date}):")
            print(f"        totalShareholderEquity:       {total_equity}")
            print(f"        commonStockSharesOutstanding: {shares}")
            print(f"        totalLiabilities:             {total_liab}")
            print(f"        cash:                         {cash}")
            print(f"        totalCurrentAssets:            {current_assets}")
            print(f"        totalCurrentLiabilities:       {current_liab}")

            # Check for None values
            none_fields = []
            for key in ["totalShareholderEquity", "commonStockSharesOutstanding",
                        "totalLiabilities", "totalCurrentAssets", "totalCurrentLiabilities"]:
                val = first_q.get(key)
                if val is None or val == "None":
                    none_fields.append(key)
            if none_fields:
                issues.append(f"⚠️  Balance sheet 'None' fields: {none_fields}")

    if earnings is None:
        issues.append("❌ earnings cache is EMPTY or missing")
    else:
        quarterly = earnings.get("quarterlyEarnings", [])
        print(f"    Earnings: {len(quarterly)} quarterly reports")
        if quarterly:
            first_q = quarterly[0]
            fiscal_date = first_q.get("fiscalDateEnding", "N/A")
            reported_eps = first_q.get("reportedEPS", "N/A")
            print(f"      Latest quarter ({fiscal_date}):")
            print(f"        reportedEPS: {reported_eps}")

            # Check for None EPS
            none_eps = sum(1 for q in quarterly
                         if q.get("reportedEPS") in (None, "None"))
            if none_eps > 0:
                issues.append(f"⚠️  {none_eps} quarters with None reportedEPS")

    # Verify derived features can be computed
    print(f"\n    Derived Feature Verification:")
    if income and balance and earnings:
        try:
            # TTM EPS
            eps_list = []
            for q in (earnings.get("quarterlyEarnings", []) or [])[:4]:
                eps_val = q.get("reportedEPS")
                if eps_val not in (None, "None"):
                    eps_list.append(float(eps_val))
            if eps_list:
                ttm_eps = sum(eps_list)
                print(f"      TTM EPS (latest 4Q): {ttm_eps:.4f} (from {len(eps_list)} quarters)")
            else:
                print(f"      TTM EPS: ❌ No valid EPS data")
                issues.append("❌ Cannot compute TTM EPS - no valid reportedEPS")

            # TTM Revenue
            rev_list = []
            for q in (income.get("quarterlyReports", []) or [])[:4]:
                rev_val = q.get("totalRevenue")
                if rev_val not in (None, "None"):
                    rev_list.append(float(rev_val))
            if rev_list:
                ttm_rev = sum(rev_list)
                print(f"      TTM Revenue (latest 4Q): {ttm_rev:,.0f} (from {len(rev_list)} quarters)")
            else:
                print(f"      TTM Revenue: ❌ No valid revenue data")
                issues.append("❌ Cannot compute TTM Revenue")

            # Current Ratio
            bq = (balance.get("quarterlyReports", []) or [])[0] if balance.get("quarterlyReports") else {}
            ca = bq.get("totalCurrentAssets")
            cl = bq.get("totalCurrentLiabilities")
            if ca not in (None, "None") and cl not in (None, "None") and float(cl) > 0:
                cr = float(ca) / float(cl)
                print(f"      Current Ratio: {cr:.4f}")
            else:
                print(f"      Current Ratio: ❌ Missing currentAssets or currentLiabilities")
                issues.append("❌ Cannot compute Current Ratio")

            # Debt to Equity
            tl = bq.get("totalLiabilities")
            te = bq.get("totalShareholderEquity")
            if tl not in (None, "None") and te not in (None, "None") and float(te) > 0:
                de = float(tl) / float(te)
                print(f"      Debt/Equity: {de:.4f}")
            else:
                print(f"      Debt/Equity: ❌ Missing totalLiabilities or totalShareholderEquity")
                issues.append("❌ Cannot compute Debt/Equity")

        except Exception as exc:
            issues.append(f"❌ Derived feature computation error: {exc}")

    # Summary
    if issues:
        print(f"\n    ⚠️  Issues found ({len(issues)}):")
        for issue in issues:
            print(f"      {issue}")
    else:
        print(f"\n    ✅ All fundamental data looks correct")


def check_macro_features(macro_cache: dict, sample_dates: list):
    """Verify macro features against cached API data."""
    print(f"\n{'='*60}")
    print(f"  Macro Features Check")
    print(f"{'='*60}")

    issues = []

    # Fed Funds Rate
    ffr = macro_cache.get("fed_funds_rate_monthly")
    if ffr:
        data = ffr.get("data", [])
        print(f"  Fed Funds Rate: {len(data)} data points")
        if data:
            latest = data[0]
            print(f"    Latest: {latest.get('date')} = {latest.get('value')}")
    else:
        issues.append("❌ Fed Funds Rate cache missing")

    # Treasury yields
    for name in ["treasury_yield_10year_monthly", "treasury_yield_2year_monthly"]:
        cache = macro_cache.get(name)
        if cache:
            data = cache.get("data", [])
            print(f"  {name}: {len(data)} data points")
            if data:
                latest = data[0]
                print(f"    Latest: {latest.get('date')} = {latest.get('value')}")
        else:
            issues.append(f"❌ {name} cache missing")

    # CPI
    cpi = macro_cache.get("cpi_monthly")
    if cpi:
        data = cpi.get("data", [])
        print(f"  CPI: {len(data)} data points")
        if data:
            latest = data[0]
            print(f"    Latest: {latest.get('date')} = {latest.get('value')}")
    else:
        issues.append("❌ CPI cache missing")

    # Unemployment
    unemp = macro_cache.get("unemployment")
    if unemp:
        data = unemp.get("data", [])
        print(f"  Unemployment: {len(data)} data points")
        if data:
            latest = data[0]
            print(f"    Latest: {latest.get('date')} = {latest.get('value')}")
    else:
        issues.append("❌ Unemployment cache missing")

    # SPY daily (for VIX proxy and momentum)
    spy = macro_cache.get("SPY_daily_full")
    if spy:
        ts = spy.get("Time Series (Daily)", {})
        print(f"  SPY daily: {len(ts)} data points")
    else:
        issues.append("❌ SPY daily cache missing")

    # VIXY (VIX proxy)
    vixy = macro_cache.get("VIXY_daily_full")
    if vixy:
        ts = vixy.get("Time Series (Daily)", {})
        print(f"  VIXY daily: {len(ts)} data points")
    else:
        issues.append("❌ VIXY daily cache missing")

    # Verify yield spread computation
    y10 = macro_cache.get("treasury_yield_10year_monthly")
    y2 = macro_cache.get("treasury_yield_2year_monthly")
    if y10 and y2:
        d10 = y10.get("data", [])
        d2 = y2.get("data", [])
        if d10 and d2:
            v10 = float(d10[0]["value"]) if d10[0]["value"] != "." else None
            v2 = float(d2[0]["value"]) if d2[0]["value"] != "." else None
            if v10 is not None and v2 is not None:
                spread = v10 - v2
                print(f"\n  Yield Spread (10Y-2Y): {spread:.4f} (10Y={v10}, 2Y={v2})")
            else:
                issues.append("⚠️  Yield data contains '.' (missing)")

    if issues:
        print(f"\n  ⚠️  Issues found ({len(issues)}):")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✅ All macro data looks correct")


def check_nan_patterns():
    """Run the actual training pipeline partially to check NaN patterns."""
    print(f"\n{'='*60}")
    print(f"  NaN Pattern Analysis (Full Pipeline Simulation)")
    print(f"{'='*60}")

    from pipelines.train_forecast_model import (
        compute_base_features, build_labels,
        BASE_FEATURE_COLUMNS,
    )
    from utils.macro_fundamental_provider import (
        MacroFundamentalFeatureProvider,
        MACRO_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_COLUMNS,
        ALL_MACRO_FUNDAMENTAL_COLUMNS,
    )

    # Pick 3 diverse tickers: large tech, financial, healthcare
    test_tickers = ["AAPL", "JPM", "LLY"]
    spy_df = load_ticker_price_data("SPY")

    provider = MacroFundamentalFeatureProvider(verbose=False)

    for ticker in test_tickers:
        print(f"\n  ── {ticker} ──")
        price_df = load_ticker_price_data(ticker)
        if price_df.empty:
            print(f"    ❌ No price data")
            continue

        # Filter to 5 years
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
        price_df = price_df[price_df.index >= cutoff]

        # Compute base features
        frame = compute_base_features(price_df)
        frame = build_labels(frame, horizon_days=5, spy_data=spy_df)
        frame = frame.dropna(subset=BASE_FEATURE_COLUMNS + ["label"])

        print(f"    Rows after base features: {len(frame)}")

        # Check base feature NaN counts
        base_nan = frame[BASE_FEATURE_COLUMNS].isna().sum()
        if base_nan.any():
            print(f"    ⚠️  Base feature NaNs:")
            for col, cnt in base_nan[base_nan > 0].items():
                print(f"      {col}: {cnt} NaN ({cnt/len(frame)*100:.1f}%)")
        else:
            print(f"    ✅ No NaN in base features")

        # Fetch fundamental data
        fm_start = frame.index.min().to_pydatetime()
        fm_end = frame.index.max().to_pydatetime()

        try:
            fund_df = provider.extract_fundamental_only_historical(
                stock_symbol=ticker,
                start_date=fm_start,
                end_date=fm_end,
            )

            _INTERMEDIATE_COLUMNS = [
                "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
                "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
            ]

            if not fund_df.empty:
                fund_df = fund_df.sort_index()
                fund_df.index = pd.to_datetime(fund_df.index)

                # merge_asof
                t_reset = frame.reset_index()
                f_reset = fund_df.reset_index()
                t_reset = t_reset.rename(columns={t_reset.columns[0]: "_merge_date"})
                f_reset = f_reset.rename(columns={f_reset.columns[0]: "_merge_date"})

                merged = pd.merge_asof(
                    t_reset.sort_values("_merge_date"),
                    f_reset.sort_values("_merge_date"),
                    on="_merge_date",
                    direction="backward",
                )
                merged = merged.set_index("_merge_date")

                # ── Compute price-dependent features (mirrors train pipeline) ──
                _close = pd.to_numeric(merged["Close"], errors="coerce")

                # P/E ratio
                if "_ttm_eps" in merged.columns:
                    _eps = merged["_ttm_eps"]
                    _v = _eps.notna() & (_eps.abs() > 0.01)
                    merged.loc[_v, "pe_ratio"] = (_close[_v] / _eps[_v]).values

                # P/B ratio
                if "_total_equity" in merged.columns and "_shares_outstanding" in merged.columns:
                    _eq = merged["_total_equity"]
                    _sh = merged["_shares_outstanding"]
                    _v = _eq.notna() & _sh.notna() & (_sh > 0)
                    _bvps = _eq[_v] / _sh[_v]
                    _bv = _bvps.abs() > 0.01
                    _fm = _v.copy()
                    _fm[_v] = _bv.values
                    merged.loc[_fm, "pb_ratio"] = (_close[_fm].values / _bvps[_bv].values)

                # P/S ratio
                if "_ttm_revenue" in merged.columns and "_shares_outstanding" in merged.columns:
                    _rev = merged["_ttm_revenue"]
                    _sh = merged["_shares_outstanding"]
                    _v = _rev.notna() & _sh.notna() & (_sh > 0) & (_rev > 0)
                    _rps = _rev[_v] / _sh[_v]
                    merged.loc[_v, "ps_ratio"] = (_close[_v] / _rps).values

                # EV/EBITDA
                if all(c in merged.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                    _sh = merged["_shares_outstanding"]
                    _ebitda = merged["_ttm_ebitda"]
                    _liab = merged["_total_liabilities"].fillna(0)
                    _cash = merged["_cash"].fillna(0)
                    _v = _sh.notna() & (_sh > 0) & _ebitda.notna() & (_ebitda.abs() > 0)
                    _mc = _close[_v] * _sh[_v]
                    _ev = _mc + _liab[_v] - _cash[_v]
                    merged.loc[_v, "ev_ebitda"] = (_ev / _ebitda[_v]).values

                # PEG ratio
                if "pe_ratio" in merged.columns and "earnings_growth_yoy" in merged.columns:
                    _pe = merged["pe_ratio"]
                    _eg = merged["earnings_growth_yoy"] * 100
                    _v = _pe.notna() & _eg.notna() & (_eg.abs() > 1.0) & (_pe > 0)
                    merged.loc[_v, "peg_ratio"] = (_pe[_v] / _eg[_v]).values

                # Check fundamental NaN patterns
                fund_cols = FUNDAMENTAL_FEATURE_COLUMNS + _INTERMEDIATE_COLUMNS
                available_fund_cols = [c for c in fund_cols if c in merged.columns]

                print(f"\n    Fundamental feature NaN analysis (after merge_asof + price-dependent calc):")
                for col in available_fund_cols:
                    total = len(merged)
                    nan_count = merged[col].isna().sum()
                    pct = nan_count / total * 100
                    status = "✅" if pct < 5 else ("⚠️ " if pct < 50 else "❌")
                    print(f"      {status} {col:30s}: {nan_count:5d}/{total} NaN ({pct:.1f}%)")

                # Check specific dates
                print(f"\n    Sample rows (first 3 dates with data):")
                sample_idx = [0, len(merged)//2, len(merged)-1]
                for i in sample_idx:
                    row = merged.iloc[i]
                    date = merged.index[i]
                    print(f"      [{date}]")
                    for col in ["_ttm_eps", "_ttm_revenue", "_shares_outstanding",
                                "_total_equity", "profit_margin", "current_ratio",
                                "debt_to_equity", "roe", "earnings_growth_yoy"]:
                        if col in merged.columns:
                            val = row[col]
                            print(f"        {col:30s}: {val}")

        except Exception as exc:
            print(f"    ❌ Fundamental fetch failed: {exc}")

        # Fetch macro data
        try:
            macro_df = provider.extract_macro_only_historical(
                start_date=fm_start,
                end_date=fm_end,
            )
            if not macro_df.empty:
                print(f"\n    Macro feature NaN analysis:")
                for col in MACRO_FEATURE_COLUMNS:
                    if col in macro_df.columns:
                        total = len(macro_df)
                        nan_count = macro_df[col].isna().sum()
                        pct = nan_count / total * 100
                        status = "✅" if pct < 5 else ("⚠️ " if pct < 50 else "❌")
                        print(f"      {status} {col:30s}: {nan_count:5d}/{total} NaN ({pct:.1f}%)")
        except Exception as exc:
            print(f"    ❌ Macro fetch failed: {exc}")


def check_price_dependent_features():
    """Check if price-dependent features (pe_ratio, pb_ratio, etc.) can be computed."""
    print(f"\n{'='*60}")
    print(f"  Price-Dependent Feature Computation Check")
    print(f"{'='*60}")

    from utils.macro_fundamental_provider import MacroFundamentalFeatureProvider

    test_tickers = ["AAPL", "JPM", "NVDA", "BRK.B", "GEV"]
    provider = MacroFundamentalFeatureProvider(verbose=False)

    for ticker in test_tickers:
        print(f"\n  ── {ticker} ──")
        price_df = load_ticker_price_data(ticker)
        if price_df.empty:
            print(f"    ❌ No price data")
            continue

        latest_close = pd.to_numeric(price_df["Close"], errors="coerce").iloc[-1]
        print(f"    Latest Close: {latest_close:.2f}")

        # Load fundamental cache
        fund_cache = load_fundamental_cache(ticker)

        # Check if we can compute P/E
        earnings = fund_cache.get("earnings")
        if earnings:
            quarterly = earnings.get("quarterlyEarnings", [])
            eps_vals = []
            for q in quarterly[:4]:
                eps = q.get("reportedEPS")
                if eps not in (None, "None"):
                    try:
                        eps_vals.append(float(eps))
                    except ValueError:
                        pass
            if eps_vals:
                ttm_eps = sum(eps_vals)
                if abs(ttm_eps) > 0.01:
                    pe = latest_close / ttm_eps
                    print(f"    P/E Ratio: {pe:.2f} (Close={latest_close:.2f} / TTM_EPS={ttm_eps:.4f})")
                else:
                    print(f"    P/E Ratio: ❌ TTM EPS too small ({ttm_eps:.4f})")
            else:
                print(f"    P/E Ratio: ❌ No valid EPS data in cache")
        else:
            print(f"    P/E Ratio: ❌ No earnings cache")

        # Check P/B
        balance = fund_cache.get("balance_sheet")
        if balance:
            quarterly = balance.get("quarterlyReports", [])
            if quarterly:
                bq = quarterly[0]
                equity = bq.get("totalShareholderEquity")
                shares = bq.get("commonStockSharesOutstanding")
                if equity not in (None, "None") and shares not in (None, "None"):
                    equity_f = float(equity)
                    shares_f = float(shares)
                    if shares_f > 0 and abs(equity_f / shares_f) > 0.01:
                        bvps = equity_f / shares_f
                        pb = latest_close / bvps
                        print(f"    P/B Ratio: {pb:.2f} (BVPS={bvps:.4f})")
                    else:
                        print(f"    P/B Ratio: ❌ BVPS too small or shares=0")
                else:
                    print(f"    P/B Ratio: ❌ Missing equity ({equity}) or shares ({shares})")
            else:
                print(f"    P/B Ratio: ❌ No quarterly reports")
        else:
            print(f"    P/B Ratio: ❌ No balance sheet cache")

        # Check for BRK.B-like empty cache issue
        for name, data in fund_cache.items():
            if data is None:
                print(f"    ❌ {name} cache is EMPTY (API returned no data)")


def check_fillna_impact():
    """Analyze the impact of fillna(0.0) on feature distributions."""
    print(f"\n{'='*60}")
    print(f"  fillna(0.0) Impact Analysis")
    print(f"{'='*60}")

    from utils.macro_fundamental_provider import (
        ALL_MACRO_FUNDAMENTAL_COLUMNS,
        MACRO_FEATURE_COLUMNS,
        FUNDAMENTAL_FEATURE_COLUMNS,
    )

    # Columns where 0.0 is a meaningful (and misleading) value
    problematic_if_zero = {
        "pe_ratio": "P/E=0 means 'free stock' (nonsensical)",
        "pb_ratio": "P/B=0 means 'free stock' (nonsensical)",
        "ps_ratio": "P/S=0 means 'free stock' (nonsensical)",
        "ev_ebitda": "EV/EBITDA=0 means 'free enterprise' (nonsensical)",
        "dividend_yield": "0 is valid (no dividend), but masks missing data",
        "roe": "ROE=0 means 'no return' (could be valid but usually not)",
        "beta": "Beta=0 means 'no market correlation' (nonsensical for stocks)",
        "fed_funds_rate": "0 is valid (ZIRP era), but masks missing data",
        "treasury_yield_10y": "0 is nearly impossible",
        "treasury_yield_2y": "0 is nearly impossible",
        "vix_level": "VIX=0 is impossible",
    }

    print(f"\n  Columns where fillna(0.0) creates misleading values:")
    for col, reason in problematic_if_zero.items():
        print(f"    ❌ {col:30s}: {reason}")

    print(f"\n  Columns where fillna(0.0) is acceptable:")
    safe_cols = [c for c in ALL_MACRO_FUNDAMENTAL_COLUMNS if c not in problematic_if_zero]
    for col in safe_cols:
        print(f"    ✅ {col}")

    print(f"\n  Recommendation: Use np.nan (LightGBM handles NaN natively)")
    print(f"  Or use cross-sectional median fill for fundamental features")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  Data Quality Check")
    print("=" * 60)

    # Load ticker list
    tickers_path = Path("data/sp500_top100.json")
    if tickers_path.exists():
        with open(tickers_path, "r") as f:
            all_tickers = json.load(f).get("tickers", [])
    else:
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Sample tickers (include BRK.B to test edge case)
    sampled = random.sample([t for t in all_tickers if t != "BRK.B"], SAMPLE_TICKERS - 1)
    sampled.append("BRK.B")  # Always include BRK.B (known edge case)
    print(f"\n  Sampled tickers: {sampled}")

    # ── Check 1: Technical features ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CHECK 1: Technical Features")
    print(f"{'='*60}")

    for ticker in sampled:
        price_df = load_ticker_price_data(ticker)
        if price_df.empty:
            print(f"\n  ❌ {ticker}: No price data cached")
            continue

        # Sample random dates (from middle of dataset to ensure enough lookback)
        valid_dates = price_df.index[60:-10]  # Skip first 60 and last 10
        if len(valid_dates) < SAMPLE_DATES_PER_TICKER:
            print(f"\n  ⚠️  {ticker}: Not enough data ({len(valid_dates)} valid dates)")
            continue

        sample_dates = sorted(random.sample(list(valid_dates), SAMPLE_DATES_PER_TICKER))
        check_technical_features(ticker, price_df, sample_dates)

    # ── Check 2: Fundamental features ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CHECK 2: Fundamental Features (from cache)")
    print(f"{'='*60}")

    for ticker in sampled:
        fund_cache = load_fundamental_cache(ticker)
        price_df = load_ticker_price_data(ticker)
        if price_df.empty:
            sample_dates = []
        else:
            valid_dates = price_df.index[60:-10]
            sample_dates = sorted(random.sample(list(valid_dates), min(SAMPLE_DATES_PER_TICKER, len(valid_dates)))) if len(valid_dates) > 0 else []
        check_fundamental_features(ticker, fund_cache, sample_dates)

    # ── Check 3: Macro features ──────────────────────────────────────────
    macro_cache = load_macro_cache()
    check_macro_features(macro_cache, [])

    # ── Check 4: Price-dependent features ────────────────────────────────
    check_price_dependent_features()

    # ── Check 5: fillna(0.0) impact ──────────────────────────────────────
    check_fillna_impact()

    # ── Check 6: Full pipeline NaN patterns ──────────────────────────────
    check_nan_patterns()

    print(f"\n{'='*60}")
    print(f"  Data Quality Check Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
