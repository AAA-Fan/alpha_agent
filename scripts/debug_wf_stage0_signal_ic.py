#!/usr/bin/env python3
"""
Walk-Forward Stage 0: Pure Signal IC Analysis with Monthly Retraining

Goal: Verify whether the LightGBM model has predictive power in a truly
      out-of-sample setting — the model is retrained every month on a
      rolling 5-year window, then scored ONLY on the next month's data.

This eliminates look-ahead bias that exists in the original Stage 0
(which uses a single pre-trained model evaluated on data it may have seen).

Flow for each month:
  1. Train LightGBM on [T-5y, T)
  2. Walk forward day-by-day through [T, T+1m), compute features, score
  3. Record predicted_probability_up vs actual_5d_excess_return
  4. Slide window forward by 1 month, repeat

After all months:
  - Compute IC, IC_IR, directional accuracy, quintile spread, calibration
  - Show per-month IC breakdown to detect regime-dependent signal decay

Usage:
    python scripts/debug_wf_stage0_signal_ic.py --ticker AAPL --start 2025-01-01 --end 2025-12-31
    python scripts/debug_wf_stage0_signal_ic.py --ticker AAPL --start 2025-01-01 --end 2025-12-31 --train-years 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats

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
from utils.macro_fundamental_provider import (
    MacroFundamentalFeatureProvider,
    MACRO_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
)
from utils.yfinance_cache import get_historical_data

# ── Regime computation helpers (mirroring train pipeline) ────────────────

def _trend_score_from_features(features: dict) -> float:
    score = 0.0
    weights = {
        "sma_20_ratio": 2.0,
        "sma_50_ratio": 1.5,
        "momentum_20": 1.2,
        "macd_hist": 0.8,
    }
    for key, w in weights.items():
        val = features.get(key, 0.0)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            score += w * float(val)
    return score


def _classify_trend(score: float, m5: float) -> str:
    if score > 0.05 and m5 > 0:
        return "strong_uptrend"
    if score > 0.02:
        return "uptrend"
    if score < -0.05 and m5 < 0:
        return "strong_downtrend"
    if score < -0.02:
        return "downtrend"
    return "sideways"


def _trend_strength(score: float) -> float:
    return min(1.0, abs(score) / 0.10)


def _classify_volatility(annualized_vol: float) -> str:
    if annualized_vol is None or (isinstance(annualized_vol, float) and np.isnan(annualized_vol)):
        return "unknown"
    if annualized_vol >= 0.50:
        return "extreme"
    if annualized_vol >= 0.35:
        return "high"
    if annualized_vol <= 0.16:
        return "low"
    return "normal"


def _is_vol_expanding(vol_20: float, atr_14: float, price: float) -> bool:
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [vol_20, atr_14, price]):
        return False
    if price <= 0:
        return False
    atr_daily_pct = atr_14 / price
    atr_annualized = atr_daily_pct * (252 ** 0.5)
    return vol_20 > atr_annualized * 1.20


def _classify_momentum_health(m5: float, m20: float, rsi: float, trend: str) -> str:
    trend_is_up = trend in ("uptrend", "strong_uptrend")
    trend_is_down = trend in ("downtrend", "strong_downtrend")
    if rsi > 75 and trend_is_up and m5 < 0:
        return "exhausted"
    if rsi < 25 and trend_is_down and m5 > 0:
        return "exhausted"
    same_direction = (m5 >= 0 and m20 >= 0) or (m5 < 0 and m20 < 0)
    if same_direction:
        if abs(m5) > abs(m20):
            return "accelerating"
        return "steady"
    return "decelerating"


def _drawdown_severity(dd: float) -> str:
    if dd is None or (isinstance(dd, float) and np.isnan(dd)):
        return "none"
    if dd <= -0.20:
        return "severe"
    if dd <= -0.10:
        return "moderate"
    if dd <= -0.03:
        return "mild"
    return "none"


def _build_state(trend: str, vol_regime: str, momentum_health: str, dd_severity: str) -> str:
    high_vol = vol_regime in ("high", "extreme")
    if trend == "strong_downtrend" and high_vol and dd_severity in ("severe", "moderate"):
        return "capitulation"
    if trend == "strong_uptrend" and momentum_health != "exhausted" and vol_regime != "extreme":
        return "strong_rally"
    if trend in ("uptrend", "strong_uptrend") and momentum_health in ("decelerating", "exhausted"):
        return "topping_out"
    if trend in ("uptrend", "strong_uptrend") and momentum_health in ("accelerating", "steady"):
        return "trending_up"
    if trend in ("downtrend", "strong_downtrend") and momentum_health in ("decelerating", "exhausted"):
        return "bottoming_out"
    if trend in ("downtrend", "strong_downtrend") and momentum_health in ("accelerating", "steady"):
        return "trending_down"
    if trend == "sideways" and high_vol:
        return "choppy"
    if trend == "sideways" and vol_regime == "low":
        return "coiling"
    return "range_bound"


_HEALTH_MAP = {"accelerating": 0, "steady": 1, "decelerating": 2, "exhausted": 3}
_STATE_DIRECTION = {
    "strong_rally": 2, "trending_up": 1, "topping_out": 0,
    "range_bound": 0, "coiling": 0, "choppy": -1,
    "trending_down": -1, "bottoming_out": 0, "capitulation": -2,
}
_VOL_REGIME_ORD = {"low": 0, "normal": 1, "high": 2, "extreme": 3, "unknown": 1}


def compute_regime_from_features(features: dict, close_price: float) -> dict:
    """Compute regime features from base features dict (no DataFrame needed)."""
    m5 = features.get("momentum_5", 0.0)
    m20 = features.get("momentum_20", 0.0)
    rsi = features.get("rsi_14", 50.0)
    vol_20 = features.get("volatility_20", 0.25)
    atr_14 = features.get("atr_14", 0.0)
    dd_60 = features.get("drawdown_60", 0.0)

    ts = _trend_score_from_features(features)
    trend = _classify_trend(ts, m5)
    strength = _trend_strength(ts)
    vol_regime = _classify_volatility(vol_20)
    vol_exp = _is_vol_expanding(vol_20, atr_14, close_price)
    mom_health = _classify_momentum_health(m5, m20, rsi, trend)
    dd_sev = _drawdown_severity(dd_60)
    state = _build_state(trend, vol_regime, mom_health, dd_sev)

    return {
        "regime_direction": _STATE_DIRECTION.get(state, 0),
        "regime_volatility_ord": _VOL_REGIME_ORD.get(vol_regime, 1),
        "trend_strength": strength,
        "vol_expanding": int(vol_exp),
        "momentum_health_enc": _HEALTH_MAP.get(mom_health, 1),
    }


# ── Feature computation (standalone, no Agent dependency) ────────────────

def compute_features(data: pd.DataFrame) -> dict:
    """Compute all base + interaction features from OHLCV data.

    Replicates the training pipeline's feature computation exactly,
    including the 5 interaction features.
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
            return np.nan
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

    # Interaction features (must match training pipeline)
    f["momentum_trend_align"] = f["momentum_5"] * f["sma_20_ratio"]
    f["rsi_deviation"] = (f["rsi_14"] - 50.0) / 50.0
    vol_z_clipped = max(-3, min(3, f["volume_zscore_20"])) if not np.isnan(f["volume_zscore_20"]) else 0.0
    f["vol_confirmed_momentum"] = f["momentum_5"] * vol_z_clipped if not np.isnan(f["momentum_5"]) else np.nan
    m5_clipped = max(-0.1, min(0.1, f["momentum_5"])) if not np.isnan(f["momentum_5"]) else 0.0
    dd_val = f["drawdown_60"] if not np.isnan(f["drawdown_60"]) else 0.0
    f["mean_reversion"] = dd_val * m5_clipped
    safe_vol = f["daily_volatility_20"] if not np.isnan(f["daily_volatility_20"]) and f["daily_volatility_20"] != 0 else None
    if safe_vol and not np.isnan(f["momentum_5"]):
        f["vol_adj_momentum_5"] = max(-5, min(5, f["momentum_5"] / safe_vol))
    else:
        f["vol_adj_momentum_5"] = 0.0

    return f


# ═══════════════════════════════════════════════════════════════════════════
# Model Training (reused from walk-forward backtest)
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
) -> Tuple[Optional[Any], Optional[Dict], List[str]]:
    """Train a LightGBM model on the given time window.

    Returns:
        (lgb_model, meta_dict, macro_fund_cols_used)
    """
    data = raw_data.get(ticker)
    if data is None or data.empty:
        return None, None, []

    # Filter to training window
    data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
    if len(data) < 120:
        if verbose:
            print(f"    [skip] Only {len(data)} rows in training window (need >= 120)")
        return None, None, []

    # Step 1: Compute base features + labels
    frame = compute_base_features(data)
    frame = build_labels(frame, horizon_days=horizon_days, spy_data=spy_data)
    frame["ticker"] = ticker

    required_cols = BASE_FEATURE_COLUMNS + ["label"]
    frame = frame.dropna(subset=required_cols)

    if len(frame) < 100:
        if verbose:
            print(f"    [skip] Only {len(frame)} valid rows after feature computation")
        return None, None, []

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
                    spy_close_series = pd.to_numeric(spy_df["Close"], errors="coerce")
                    spy_returns = spy_close_series.pct_change()
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
    for col in ALL_MACRO_FUNDAMENTAL_COLUMNS:
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
        return None, None, []

    # Build meta dict
    meta = {
        "version": 4,
        "model_type": "lightgbm",
        "label_type": "excess_return" if spy_data is not None else "absolute_return",
        "target_horizon_days": horizon_days,
        "feature_columns": BASE_FEATURE_COLUMNS,
        "regime_features": REGIME_FEATURE_COLUMNS,
        "macro_fundamental_features": macro_fund_cols_used,
        "rank_feature_columns": [],
        "categorical_features": [],
        "cv_metrics": cv_metrics,
        "training_samples": len(frame),
        "training_date_range": [
            frame.index.min().strftime("%Y-%m-%d"),
            frame.index.max().strftime("%Y-%m-%d"),
        ],
    }

    return final_model, meta, macro_fund_cols_used


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Signal IC Collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_monthly_signals(
    ticker: str,
    lgb_model: Any,
    meta: Dict[str, Any],
    ohlcv_data: pd.DataFrame,
    spy_close: Optional[pd.Series],
    macro_fund_df: Optional[pd.DataFrame],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    verbose: bool = False,
) -> pd.DataFrame:
    """Walk forward through one month, collecting (date, prob, actual_return) pairs.

    Uses the provided model (trained on data BEFORE month_start) to score
    each day in [month_start, month_end].
    """
    feature_cols: List[str] = meta.get("feature_columns", [])
    regime_cols: List[str] = meta.get("regime_features", [])
    macro_fund_cols: List[str] = meta.get("macro_fundamental_features", [])
    rank_cols: List[str] = meta.get("rank_feature_columns", [])
    cat_cols: List[str] = meta.get("categorical_features", [])
    all_cols = feature_cols + regime_cols + macro_fund_cols + rank_cols + cat_cols
    horizon = meta.get("target_horizon_days", 5)

    close_prices = pd.to_numeric(ohlcv_data["Close"], errors="coerce")
    dates = ohlcv_data.index

    # Find date range indices
    start_idx = dates.searchsorted(month_start)
    start_idx = max(start_idx, 60)  # Need at least 60 bars for features
    end_idx = dates.searchsorted(month_end, side="right")

    records = []

    for t in range(start_idx, min(end_idx, len(dates) - horizon)):
        current_date = dates[t]
        if current_date < month_start or current_date > month_end:
            continue

        # Compute features from data up to day t (no look-ahead)
        data_slice = ohlcv_data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            continue

        # Build feature row
        row = {}
        for col in feature_cols:
            val = features.get(col)
            row[col] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan

        # Regime features
        current_close = float(close_prices.iloc[t])
        regime_feats = compute_regime_from_features(features, current_close)
        for col in regime_cols:
            row[col] = regime_feats.get(col, 0.0)

        # Macro/fundamental features
        if macro_fund_df is not None and not macro_fund_df.empty:
            valid_dates = macro_fund_df.index[macro_fund_df.index <= current_date]
            if len(valid_dates) > 0:
                nearest_date = valid_dates[-1]
                mf_row = macro_fund_df.loc[nearest_date]
                for col in macro_fund_cols:
                    val = mf_row.get(col, np.nan) if col in macro_fund_df.columns else np.nan
                    if pd.notna(val):
                        row[col] = float(val)
                    else:
                        row[col] = np.nan
            else:
                for col in macro_fund_cols:
                    row[col] = np.nan
        else:
            for col in macro_fund_cols:
                row[col] = np.nan

        # Rank features: median (single-ticker)
        for col in rank_cols:
            row[col] = 0.5

        # Categorical features
        for col in cat_cols:
            row[col] = 0.0

        X = pd.DataFrame([row])[all_cols]
        raw_prob = float(lgb_model.predict(X)[0])

        # Actual forward return (excess over SPY)
        future_close = close_prices.iloc[t + horizon]
        stock_return = (future_close / current_close) - 1.0

        spy_return = 0.0
        if spy_close is not None:
            spy_cur = spy_close.iloc[t] if t < len(spy_close) else np.nan
            spy_fut = spy_close.iloc[t + horizon] if (t + horizon) < len(spy_close) else np.nan
            if pd.notna(spy_cur) and pd.notna(spy_fut) and spy_cur > 0:
                spy_return = (spy_fut / spy_cur) - 1.0

        excess_return = stock_return - spy_return

        records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted_prob_up": raw_prob,
            "raw_prob_up": raw_prob,
            "stock_5d_return": float(stock_return),
            "spy_5d_return": float(spy_return),
            "actual_5d_return": float(excess_return),
            "actual_direction": 1 if excess_return > 0 else 0,
        })

    df = pd.DataFrame(records)
    if verbose and not df.empty:
        print(f"    [{ticker}] Collected {len(df)} signal-return pairs for {month_start.strftime('%Y-%m')}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# IC Analysis (same as original Stage 0)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_ic(df: pd.DataFrame, ticker: str) -> dict:
    """Compute IC metrics from signal-return pairs.

    In walk-forward mode, the 250 samples come from 12 different monthly
    models whose probability distributions (mean/std) differ significantly.
    Pooling them naively for a global Spearman IC is misleading because
    cross-model probability levels are not comparable.

    Primary IC metrics are therefore computed **per-month then averaged**.
    A secondary "normalised global IC" is computed after within-month rank
    normalisation so that cross-model scale differences are removed.
    """
    if df.empty or len(df) < 10:
        print(f"  [{ticker}] Not enough data for IC analysis")
        return {}

    # ── 0. Detect month column for per-model grouping ────────────────────
    if "date" in df.columns:
        df["_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    else:
        df["_month"] = "all"

    # ── 1a. Naïve global IC (kept for reference, but NOT the primary metric)
    naive_global_ic, naive_global_pval = stats.spearmanr(
        df["predicted_prob_up"], df["actual_5d_return"]
    )

    # ── 1b. Per-month IC → mean / std / IR / hit-rate (PRIMARY metrics)
    monthly_ics = []
    for _m, grp in df.groupby("_month"):
        if len(grp) < 5:
            continue
        ic_val, _ = stats.spearmanr(grp["predicted_prob_up"], grp["actual_5d_return"])
        if not np.isnan(ic_val):
            monthly_ics.append(ic_val)

    mean_monthly_ic = np.mean(monthly_ics) if monthly_ics else float(naive_global_ic)
    std_monthly_ic = np.std(monthly_ics) if monthly_ics else 0.0
    monthly_ic_ir = mean_monthly_ic / std_monthly_ic if std_monthly_ic > 0 else 0.0
    monthly_ic_hit = (
        np.mean([1 if ic > 0 else 0 for ic in monthly_ics]) if monthly_ics else 0.0
    )

    # ── 1c. Normalised global IC (within-month rank → percentile → global Spearman)
    df["_norm_prob"] = df.groupby("_month")["predicted_prob_up"].rank(pct=True)
    norm_global_ic, norm_global_pval = stats.spearmanr(
        df["_norm_prob"], df["actual_5d_return"]
    )

    # ── 2. Directional accuracy ──────────────────────────────────────────
    # Use within-month median as threshold (not fixed 0.5) because each
    # monthly model has a different probability centre.
    df["_month_median"] = df.groupby("_month")["predicted_prob_up"].transform("median")
    df["predicted_direction"] = (df["predicted_prob_up"] > df["_month_median"]).astype(int)
    directional_accuracy = (df["predicted_direction"] == df["actual_direction"]).mean()

    # Also compute fixed-0.5-threshold accuracy for reference
    df["predicted_direction_fixed"] = (df["predicted_prob_up"] > 0.5).astype(int)
    directional_accuracy_fixed = (df["predicted_direction_fixed"] == df["actual_direction"]).mean()

    # Absolute-return accuracy for reference
    if "stock_5d_return" in df.columns:
        abs_direction = (df["stock_5d_return"] > 0).astype(int)
        abs_accuracy = (df["predicted_direction"] == abs_direction).mean()
    else:
        abs_accuracy = directional_accuracy

    # ── 3. Quintile analysis (within-month normalised) ───────────────────
    df["quintile"] = df.groupby("_month")["predicted_prob_up"].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop") if len(x) >= 5 else 2
    )
    quintile_returns = df.groupby("quintile")["actual_5d_return"].mean()
    long_short_spread = 0.0
    if len(quintile_returns) >= 2:
        long_short_spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]

    # 4. Calibration analysis
    n_bins = 10
    df["prob_bin"] = pd.cut(df["predicted_prob_up"], bins=n_bins, labels=False, duplicates="drop")
    calibration = df.groupby("prob_bin").agg(
        mean_predicted=("predicted_prob_up", "mean"),
        actual_win_rate=("actual_direction", "mean"),
        count=("actual_direction", "count"),
    ).reset_index()

    # 5. Signal distribution
    buy_signals = (df["predicted_prob_up"] > 0.55).sum()
    sell_signals = (df["predicted_prob_up"] < 0.45).sum()
    hold_signals = len(df) - buy_signals - sell_signals

    # 6. Raw prob distribution
    raw_mean = df["raw_prob_up"].mean()
    raw_std = df["raw_prob_up"].std()
    raw_min = df["raw_prob_up"].min()
    raw_max = df["raw_prob_up"].max()

    # 7. Excess return statistics
    avg_excess = df["actual_5d_return"].mean()
    avg_stock = df["stock_5d_return"].mean() if "stock_5d_return" in df.columns else avg_excess
    avg_spy = df["spy_5d_return"].mean() if "spy_5d_return" in df.columns else 0.0
    pct_outperform = (df["actual_direction"] == 1).mean()

    # Clean up temporary columns
    df.drop(columns=["_month", "_norm_prob", "_month_median", "predicted_direction_fixed"], errors="ignore", inplace=True)

    results = {
        "ticker": ticker,
        "n_samples": len(df),
        "label_type": "excess_return (stock - SPY)",
        # Primary IC metrics (per-month averaged — correct for walk-forward)
        "mean_monthly_ic": round(float(mean_monthly_ic), 6),
        "std_monthly_ic": round(float(std_monthly_ic), 6),
        "monthly_ic_ir": round(float(monthly_ic_ir), 4),
        "monthly_ic_hit_rate": round(float(monthly_ic_hit), 4),
        "n_months_for_ic": len(monthly_ics),
        # Normalised global IC (within-month rank normalised)
        "norm_global_ic": round(float(norm_global_ic), 6),
        "norm_global_ic_pval": round(float(norm_global_pval), 6),
        # Naïve global IC (kept for reference — misleading in walk-forward)
        "naive_global_ic": round(float(naive_global_ic), 6),
        "naive_global_ic_pval": round(float(naive_global_pval), 6),
        # Legacy aliases for backward compatibility
        "overall_ic": round(float(mean_monthly_ic), 6),
        "overall_ic_pval": round(float(norm_global_pval), 6),
        "mean_rolling_ic": round(float(mean_monthly_ic), 6),
        "std_rolling_ic": round(float(std_monthly_ic), 6),
        "ic_ir": round(float(monthly_ic_ir), 4),
        "ic_hit_rate": round(float(monthly_ic_hit), 4),
        "directional_accuracy": round(float(directional_accuracy), 4),
        "directional_accuracy_fixed_threshold": round(float(directional_accuracy_fixed), 4),
        "abs_return_accuracy": round(float(abs_accuracy), 4),
        "long_short_spread_5d": round(float(long_short_spread), 6),
        "avg_excess_return_5d": round(float(avg_excess), 6),
        "avg_stock_return_5d": round(float(avg_stock), 6),
        "avg_spy_return_5d": round(float(avg_spy), 6),
        "pct_outperform_spy": round(float(pct_outperform), 4),
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


def analyze_monthly_ic(monthly_dfs: List[Tuple[str, pd.DataFrame]]) -> List[dict]:
    """Compute per-month IC to detect signal decay or regime dependence."""
    monthly_ic_results = []
    for month_label, df in monthly_dfs:
        if df.empty or len(df) < 5:
            monthly_ic_results.append({
                "month": month_label,
                "n_samples": len(df),
                "ic": np.nan,
                "directional_accuracy": np.nan,
            })
            continue

        ic, pval = stats.spearmanr(df["predicted_prob_up"], df["actual_5d_return"])
        # Use within-month median as threshold for directional accuracy
        month_median = df["predicted_prob_up"].median()
        pred_dir = (df["predicted_prob_up"] > month_median).astype(int)
        acc = (pred_dir == df["actual_direction"]).mean()
        # Also compute fixed-0.5 accuracy for reference
        pred_dir_fixed = (df["predicted_prob_up"] > 0.5).astype(int)
        acc_fixed = (pred_dir_fixed == df["actual_direction"]).mean()
        avg_excess = df["actual_5d_return"].mean()
        avg_prob = df["predicted_prob_up"].mean()

        monthly_ic_results.append({
            "month": month_label,
            "n_samples": len(df),
            "ic": round(float(ic), 6) if not np.isnan(ic) else np.nan,
            "ic_pval": round(float(pval), 6) if not np.isnan(pval) else np.nan,
            "directional_accuracy": round(float(acc), 4),
            "directional_accuracy_fixed": round(float(acc_fixed), 4),
            "avg_excess_return": round(float(avg_excess), 6),
            "avg_predicted_prob": round(float(avg_prob), 6),
        })

    return monthly_ic_results


# ═══════════════════════════════════════════════════════════════════════════
# Report Printing
# ═══════════════════════════════════════════════════════════════════════════

def print_report(results: dict, monthly_ic: List[dict]) -> None:
    """Pretty-print the walk-forward IC analysis report."""
    ticker = results.get("ticker", "?")

    print(f"\n{'='*70}")
    print(f"  Walk-Forward Stage 0: Signal IC Report — {ticker}")
    print(f"  Label: {results.get('label_type', 'excess_return')}")
    print(f"  Mode: Monthly retraining (no look-ahead bias)")
    print(f"{'='*70}")

    print(f"\n📊 Sample Size: {results['n_samples']} signal-return pairs (out-of-sample)")

    # IC metrics — per-month averaged (primary) + normalised global + naive global
    print(f"\n── Information Coefficient (IC) ──")
    mean_mic = results.get("mean_monthly_ic", results.get("overall_ic", 0))
    mic_ir = results.get("monthly_ic_ir", results.get("ic_ir", 0))
    mic_hit = results.get("monthly_ic_hit_rate", results.get("ic_hit_rate", 0))
    n_months_ic = results.get("n_months_for_ic", "?")
    norm_ic = results.get("norm_global_ic", 0)
    norm_pval = results.get("norm_global_ic_pval", 1)
    naive_ic = results.get("naive_global_ic", 0)
    naive_pval = results.get("naive_global_ic_pval", 1)

    if abs(mean_mic) > 0.05:
        ic_quality = "✅ GOOD (|IC| > 0.05)"
    elif abs(mean_mic) > 0.02:
        ic_quality = "⚠️  WEAK (0.02 < |IC| < 0.05)"
    else:
        ic_quality = "❌ NO SIGNAL (|IC| < 0.02)"

    print(f"  [PRIMARY] Mean Monthly IC:  {mean_mic:+.6f}  (avg of {n_months_ic} per-model ICs)")
    print(f"  [PRIMARY] Monthly IC_IR:    {mic_ir:+.4f}  {'✅ > 0.5' if abs(mic_ir) > 0.5 else '❌ < 0.5'}")
    print(f"  [PRIMARY] Monthly IC Hit:   {mic_hit:.2%}  {'✅ > 50%' if mic_hit > 0.5 else '❌ ≤ 50%'}")
    print(f"  [NORM]    Global IC (rank-normalised): {norm_ic:+.6f}  (p={norm_pval:.4f})")
    print(f"  [REF]     Naive Global IC (raw pool):  {naive_ic:+.6f}  (p={naive_pval:.4f})  ⚠️ misleading in WF")
    print(f"  Quality:  {ic_quality}")

    # Directional accuracy
    print(f"\n── Directional Accuracy (Excess Return: Outperform SPY) ──")
    acc = results["directional_accuracy"]
    acc_fixed = results.get("directional_accuracy_fixed_threshold", acc)
    abs_acc = results.get("abs_return_accuracy", acc)
    print(f"  Excess Ret Acc (month-median threshold): {acc:.2%}  {'✅ > 52%' if acc > 0.52 else '❌ ≤ 52%'}")
    print(f"  Excess Ret Acc (fixed 0.5 threshold):    {acc_fixed:.2%}  (reference — biased by prob drift)")
    print(f"  Abs Return Acc: {abs_acc:.2%}  (reference only)")

    # Excess return statistics
    print(f"\n── Excess Return Statistics ──")
    print(f"  Avg Excess Return (5d): {results.get('avg_excess_return_5d', 0):+.6f}")
    print(f"  Avg Stock Return (5d):  {results.get('avg_stock_return_5d', 0):+.6f}")
    print(f"  Avg SPY Return (5d):    {results.get('avg_spy_return_5d', 0):+.6f}")
    print(f"  % Outperform SPY:       {results.get('pct_outperform_spy', 0):.2%}")

    # Long-short spread (within-month quintiles)
    print(f"\n── Quintile Analysis (Long-Short, within-month normalised) ──")
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
    print(f"  Mean:  {results['raw_prob_mean']:.6f}")
    print(f"  Std:   {results['raw_prob_std']:.6f}")
    print(f"  Range: [{results['raw_prob_range'][0]:.6f}, {results['raw_prob_range'][1]:.6f}]")

    # Calibration
    print(f"\n── Calibration (Predicted Prob vs Actual Outperform Rate) ──")
    for row in results.get("calibration", []):
        pred = row.get("mean_predicted", 0)
        actual = row.get("actual_win_rate", 0)
        count = row.get("count", 0)
        gap = actual - pred
        print(f"    Predicted={pred:.3f} | ActualOutperform={actual:.3f} | Gap={gap:+.3f} | n={count}")

    # Per-month IC breakdown (unique to walk-forward)
    print(f"\n── Per-Month IC Breakdown (Walk-Forward) ──")
    print(f"  {'Month':<10} {'N':>5} {'IC':>10} {'p-val':>8} {'Accuracy':>10} {'AvgExcess':>12} {'AvgProb':>10}")
    print(f"  {'─'*10} {'─'*5} {'─'*10} {'─'*8} {'─'*10} {'─'*12} {'─'*10}")
    for m in monthly_ic:
        ic_str = f"{m['ic']:+.6f}" if not (isinstance(m['ic'], float) and np.isnan(m['ic'])) else "    N/A"
        pval_str = f"{m.get('ic_pval', np.nan):.4f}" if not (isinstance(m.get('ic_pval', np.nan), float) and np.isnan(m.get('ic_pval', np.nan))) else "  N/A"
        acc_str = f"{m['directional_accuracy']:.2%}" if not (isinstance(m['directional_accuracy'], float) and np.isnan(m['directional_accuracy'])) else "    N/A"
        excess_str = f"{m.get('avg_excess_return', 0):+.6f}"
        prob_str = f"{m.get('avg_predicted_prob', 0):.6f}"
        print(f"  {m['month']:<10} {m['n_samples']:>5} {ic_str:>10} {pval_str:>8} {acc_str:>10} {excess_str:>12} {prob_str:>10}")

    # IC stability analysis
    valid_ics = [m["ic"] for m in monthly_ic if not (isinstance(m["ic"], float) and np.isnan(m["ic"]))]
    if len(valid_ics) >= 2:
        ic_mean_monthly = np.mean(valid_ics)
        ic_std_monthly = np.std(valid_ics)
        ic_positive_months = sum(1 for ic in valid_ics if ic > 0)
        print(f"\n  Monthly IC Summary:")
        print(f"    Mean monthly IC:     {ic_mean_monthly:+.6f}")
        print(f"    Std monthly IC:      {ic_std_monthly:.6f}")
        print(f"    Positive IC months:  {ic_positive_months}/{len(valid_ics)} ({ic_positive_months/len(valid_ics):.0%})")

    # Overall verdict — use per-month averaged IC as primary criterion
    print(f"\n{'─'*70}")
    issues = []
    if abs(mean_mic) < 0.02:
        issues.append("Mean monthly IC near zero → model has no per-model ranking power")
    if abs(norm_ic) < 0.02 and norm_pval > 0.1:
        issues.append("Normalised global IC near zero → no signal even after removing prob drift")
    if acc <= 0.50:
        issues.append("Directional accuracy ≤ 50% (month-median threshold) → worse than coin flip")
    if spread <= 0:
        issues.append("Long-short spread ≤ 0 (within-month quintiles) → no monotonic signal")
    if results["raw_prob_std"] < 0.01:
        issues.append("Raw prob std < 0.01 → model outputs are nearly constant (degenerate)")
    if results["buy_signals"] == 0 and results["sell_signals"] == 0:
        issues.append("No buy/sell signals → model always outputs ~0.5 (no conviction)")
    if valid_ics and sum(1 for ic in valid_ics if ic > 0) / len(valid_ics) < 0.4:
        issues.append("IC positive in < 40% of months → signal is unstable")

    if not issues:
        print("  ✅ VERDICT: Model shows out-of-sample predictive signal. Proceed to WF Stage 1.")
    else:
        print("  ❌ VERDICT: Model has fundamental issues (out-of-sample):")
        for issue in issues:
            print(f"     • {issue}")
        print("  → Fix the model/features before proceeding to WF Stage 1.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Walk-Forward Stage 0: Signal IC Analysis with Monthly Retraining",
    )
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--train-years", type=int, default=5, help="Rolling training window in years (default: 5)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days (default: 5)")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for training (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)

    print(f"\n{'='*70}")
    print(f"  Walk-Forward Stage 0: Signal IC Analysis")
    print(f"{'='*70}")
    print(f"  Ticker:          {ticker}")
    print(f"  Backtest period: {args.start} → {args.end}")
    print(f"  Training window: {args.train_years} years (rolling)")
    print(f"  Retrain freq:    monthly")
    print(f"  Horizon:         {args.horizon}d")
    print(f"  CV folds:        {args.cv_folds}")
    print()

    # ── Step 1: Generate monthly schedule ────────────────────────────────
    months: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_date.replace(day=1)
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
    earliest_train_start = months[0][0] - pd.DateOffset(years=args.train_years)
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
        spy_close = pd.to_numeric(spy_data["Close"], errors="coerce")
    else:
        print("  [warn] SPY data not available, will use absolute return labels")
        spy_data = None
        spy_close = None

    ohlcv_data = raw_data[ticker]
    if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
        ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
    ohlcv_data = ohlcv_data.sort_index()
    print(f"  {ticker} data: {len(ohlcv_data)} rows")

    # Align SPY close to ticker dates
    if spy_close is not None:
        spy_close = spy_close.reindex(ohlcv_data.index, method="ffill")

    # ── Step 3: Pre-fetch macro/fundamental data (once) ──────────────────
    macro_fund_df = None
    try:
        provider = MacroFundamentalFeatureProvider(verbose=args.verbose)
        mf_start = (earliest_train_start - pd.Timedelta(days=120)).to_pydatetime()
        mf_end = end_date.to_pydatetime()
        macro_fund_df = provider.extract_historical(
            stock_symbol=ticker,
            start_date=mf_start,
            end_date=mf_end,
        )
        if macro_fund_df is not None and not macro_fund_df.empty:
            macro_fund_df = macro_fund_df.sort_index()
            macro_fund_df.index = pd.to_datetime(macro_fund_df.index)

            # Compute price-dependent features
            close_prices = pd.to_numeric(ohlcv_data["Close"], errors="coerce")
            close_for_mf = close_prices.reindex(macro_fund_df.index, method="ffill")

            _INTERMEDIATE_COLS = [
                "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
                "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
            ]

            if "_ttm_eps" in macro_fund_df.columns:
                ttm_eps = macro_fund_df["_ttm_eps"]
                valid = ttm_eps.notna() & (ttm_eps.abs() > 0.01) & close_for_mf.notna()
                macro_fund_df.loc[valid, "pe_ratio"] = close_for_mf[valid] / ttm_eps[valid]

            if "_total_equity" in macro_fund_df.columns and "_shares_outstanding" in macro_fund_df.columns:
                equity = macro_fund_df["_total_equity"]
                shares = macro_fund_df["_shares_outstanding"]
                valid = equity.notna() & shares.notna() & (shares > 0) & close_for_mf.notna()
                bvps = equity[valid] / shares[valid]
                bvps_valid = bvps.abs() > 0.01
                final_idx = bvps_valid.index[bvps_valid]
                macro_fund_df.loc[final_idx, "pb_ratio"] = close_for_mf[final_idx] / bvps[bvps_valid]

            if "_ttm_revenue" in macro_fund_df.columns and "_shares_outstanding" in macro_fund_df.columns:
                ttm_rev = macro_fund_df["_ttm_revenue"]
                shares = macro_fund_df["_shares_outstanding"]
                valid = ttm_rev.notna() & shares.notna() & (shares > 0) & (ttm_rev > 0) & close_for_mf.notna()
                rps = ttm_rev[valid] / shares[valid]
                macro_fund_df.loc[valid, "ps_ratio"] = close_for_mf[valid] / rps

            if all(c in macro_fund_df.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                shares = macro_fund_df["_shares_outstanding"]
                ttm_ebitda = macro_fund_df["_ttm_ebitda"]
                total_liab = macro_fund_df["_total_liabilities"].fillna(0)
                cash = macro_fund_df["_cash"].fillna(0)
                valid = shares.notna() & (shares > 0) & ttm_ebitda.notna() & (ttm_ebitda.abs() > 0) & close_for_mf.notna()
                market_cap = close_for_mf[valid] * shares[valid]
                ev = market_cap + total_liab[valid] - cash[valid]
                macro_fund_df.loc[valid, "ev_ebitda"] = ev / ttm_ebitda[valid]

            # Beta
            try:
                if spy_close is not None:
                    spy_returns = spy_close.pct_change()
                    stock_returns = close_prices.pct_change()
                    aligned = pd.DataFrame({"stock": stock_returns, "spy": spy_returns}).dropna()
                    if len(aligned) > 60:
                        rolling_cov = aligned["stock"].rolling(252, min_periods=60).cov(aligned["spy"])
                        rolling_var = aligned["spy"].rolling(252, min_periods=60).var()
                        rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
                        macro_fund_df["beta"] = rolling_beta.reindex(macro_fund_df.index).ffill()
            except Exception:
                pass

            # PEG ratio
            if "pe_ratio" in macro_fund_df.columns and "earnings_growth_yoy" in macro_fund_df.columns:
                pe = macro_fund_df["pe_ratio"]
                eg = macro_fund_df["earnings_growth_yoy"]
                eg_pct = eg * 100
                valid = pe.notna() & eg_pct.notna() & (eg_pct.abs() > 1.0) & (pe > 0)
                macro_fund_df.loc[valid, "peg_ratio"] = pe[valid] / eg_pct[valid]

            # Drop intermediate columns
            for col in _INTERMEDIATE_COLS:
                if col in macro_fund_df.columns:
                    macro_fund_df.drop(columns=[col], inplace=True)

            print(f"  [macro/fund] Loaded {len(macro_fund_df)} days of macro/fundamental data")
        else:
            print("  [warn] macro/fund data empty")
            macro_fund_df = None
    except Exception as exc:
        print(f"  [warn] macro/fund fetch failed: {exc}")
        macro_fund_df = None

    print()

    # ── Step 4: Walk-forward loop ────────────────────────────────────────
    all_monthly_dfs: List[Tuple[str, pd.DataFrame]] = []
    all_signal_dfs: List[pd.DataFrame] = []
    total_start_time = time.time()

    for month_idx, (month_start, month_end) in enumerate(months):
        train_start = month_start - pd.DateOffset(years=args.train_years)
        train_end = month_start - pd.Timedelta(days=1)

        print(f"{'='*70}")
        print(f"  Month {month_idx + 1}/{len(months)}: {month_start.strftime('%Y-%m-%d')} → {month_end.strftime('%Y-%m-%d')}")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
        print(f"{'='*70}")

        month_time = time.time()

        # 4a: Train model for this window
        print(f"  [Train] Training model on {args.train_years}-year window ...")
        lgb_model, meta, mf_cols_used = train_model_for_window(
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
            print(f"  [SKIP] Training failed for this window")
            all_monthly_dfs.append((month_start.strftime("%Y-%m"), pd.DataFrame()))
            continue

        cv_auc = meta.get("cv_metrics", {}).get("mean_auc", 0.0)
        n_train = meta.get("training_samples", 0)
        n_mf = len(mf_cols_used)
        print(f"  [Train] CV AUC: {cv_auc:.4f}, samples: {n_train}, macro/fund features: {n_mf}")

        # 4b: Collect signals for this month (out-of-sample)
        print(f"  [Score] Scoring out-of-sample month ...")
        month_df = collect_monthly_signals(
            ticker=ticker,
            lgb_model=lgb_model,
            meta=meta,
            ohlcv_data=ohlcv_data,
            spy_close=spy_close,
            macro_fund_df=macro_fund_df,
            month_start=month_start,
            month_end=month_end,
            verbose=args.verbose,
        )

        elapsed = time.time() - month_time
        month_label = month_start.strftime("%Y-%m")

        if not month_df.empty:
            # Quick per-month IC
            ic_val, _ = stats.spearmanr(month_df["predicted_prob_up"], month_df["actual_5d_return"])
            pred_dir = (month_df["predicted_prob_up"] > 0.5).astype(int)
            acc_val = (pred_dir == month_df["actual_direction"]).mean()
            print(f"  [Result] {month_label}: {len(month_df)} signals, IC={ic_val:+.4f}, Acc={acc_val:.2%}, Time={elapsed:.1f}s")
        else:
            print(f"  [Result] {month_label}: No signals collected, Time={elapsed:.1f}s")

        all_monthly_dfs.append((month_label, month_df))
        if not month_df.empty:
            all_signal_dfs.append(month_df)

    # ── Step 5: Aggregate and analyze ────────────────────────────────────
    total_elapsed = time.time() - total_start_time

    if not all_signal_dfs:
        print("\n[ERROR] No signal data collected across all months.")
        sys.exit(1)

    pooled_df = pd.concat(all_signal_dfs, ignore_index=True)
    print(f"\n[Step 5] Analyzing {len(pooled_df)} total out-of-sample signal-return pairs ...")
    print(f"  Total time: {total_elapsed:.1f}s")

    # Overall IC analysis
    results = analyze_ic(pooled_df, ticker)

    # Per-month IC breakdown
    monthly_ic = analyze_monthly_ic(all_monthly_dfs)

    # Add walk-forward specific info to results
    results["walk_forward"] = {
        "train_years": args.train_years,
        "retrain_frequency": "monthly",
        "n_months": len(months),
        "n_months_with_signal": len(all_signal_dfs),
        "monthly_ic": monthly_ic,
    }

    # Print report
    print_report(results, monthly_ic)

    # ── Step 6: Save results ─────────────────────────────────────────────
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"wf_stage0_signal_ic_{ticker}_{args.start}_{args.end}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[✓] Results saved: {output_path}")

    # Save raw signal data for further analysis
    signals_path = output_dir / f"wf_stage0_signals_{ticker}_{args.start}_{args.end}.csv"
    pooled_df.to_csv(signals_path, index=False)
    print(f"[✓] Raw signals saved: {signals_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
