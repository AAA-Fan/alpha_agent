#!/usr/bin/env python3
"""
Walk-Forward Stage 2: ForecastAgent + Simplified RiskAgent Backtest

Goal: Validate whether RiskAgent's core risk controls HELP or HURT
      performance compared to WF Stage 1 (ForecastAgent only),
      in a truly out-of-sample walk-forward setting.

Simplified RiskAgent (5 steps):
  ① Prediction Kelly position sizing (p=current prob, b=avg_win/avg_loss)
  ② Uncertainty filtering via Conformal Prediction Set + Tree Dispersion
  ③ Direction judgment (buy/sell/hold)
  ④ Minimum position threshold (< 3% → zero)
  ⑤ Dynamic stop-loss + take-profit (daily_vol × 2.5, R:R = 2.0)

Bypassed (Stage 3 will add these back):
  - Signal alignment — needs RegimeAgent
  - Regime risk budget — needs RegimeAgent
  - Macro/Fundamental adjustments — handled by LightGBM features
  - Track record factor — needs MemoryAgent

Key differences from WF Stage 1:
  - Model training also fits calibrator + conformal scores
  - Position size from Prediction Kelly (not fixed 100%)
  - Conformal prediction set + tree dispersion for uncertainty filtering
  - Dynamic stop-loss / take-profit checked daily during holding period
  - Minimum position threshold
  - Label: absolute return (not excess return)

Flow for each month:
  1. Train LightGBM on [T-5y, T) + fit calibrator + conformal scores
  2. Walk forward day-by-day through [T, T+1m), score with uncertainty
  3. Apply simplified RiskAgent for position sizing + risk controls
  4. Execute trades with daily stop-loss/take-profit monitoring
  5. Slide window forward by 1 month, repeat

Usage:
    python scripts/debug_wf_stage2_forecast_risk.py --ticker AAPL --start 2025-01-01 --end 2025-12-31
    python scripts/debug_wf_stage2_forecast_risk.py --ticker AAPL --start 2025-01-01 --end 2025-12-31 --threshold 0.55 --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    fit_real_isotonic_calibrator,
    compute_conformal_scores,
    BASE_FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    ALL_MACRO_FUNDAMENTAL_COLUMNS,
)
from utils.macro_fundamental_provider import (
    MacroFundamentalFeatureProvider,
    MACRO_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
)
from utils.yfinance_cache import get_historical_data

# Layer 1 adaptive thresholds — reuse the implementation already validated
# in run_walk_forward_backtest.py (same calibrated-space search logic).
from scripts.run_walk_forward_backtest import derive_adaptive_thresholds


# ── Platt Scaling calibrator ─────────────────────────────────────────────
# A sigmoid-based monotonic calibrator as an alternative to isotonic.
# Isotonic can produce sharp step plateaus at the tails (e.g. calibrated
# prob collapsing onto a single 0.98 step on long-bias ETFs such as QQQ),
# which then makes adaptive-threshold search ineffective. Platt fits a
# simple sigmoid (logistic regression on raw_prob) and always yields a
# smooth, strictly monotone mapping — no plateaus.

class _PlattCalibrator:
    """Sigmoid (Platt) calibrator with an sklearn-like `.predict(x)` API."""

    def __init__(self, lr):
        self._lr = lr

    def predict(self, x):
        arr = np.asarray(x, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(arr)[:, 1]


def fit_platt_calibrator(oof_raw, y, verbose: bool = False):
    """Fit a Platt (sigmoid) calibrator on OOF predictions."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        if verbose:
            print(f"    [Platt] sklearn import failed: {exc}")
        return None

    raw = np.asarray(oof_raw, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = ~np.isnan(raw) & ~np.isnan(yv)
    raw = raw[mask]
    yv = yv[mask]
    if raw.size < 60:
        if verbose:
            print(f"    [Platt] Only {raw.size} valid OOF samples — skip")
        return None
    if yv.min() == yv.max():
        if verbose:
            print("    [Platt] OOF labels are all one class — skip")
        return None

    lr = LogisticRegression(solver="lbfgs", max_iter=200)
    try:
        lr.fit(raw.reshape(-1, 1), yv.astype(int))
    except Exception as exc:
        if verbose:
            print(f"    [Platt] fit failed: {exc}")
        return None
    if verbose:
        coef = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
        print(f"    [Platt] fitted sigmoid: coef={coef:+.3f}, intercept={intercept:+.3f} "
              f"(n_oof={raw.size})")
    return _PlattCalibrator(lr)


# ═════════════════════════════════════════════════════════════════════════
# Regime computation helpers (mirroring train pipeline)
# ═════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# Feature computation (standalone, no Agent dependency)
# ═══════════════════════════════════════════════════════════════════════════

def compute_features(data: pd.DataFrame) -> dict:
    """Compute all base + interaction features from OHLCV data."""
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
# Model Training (with calibrator + conformal scores)
# ═══════════════════════════════════════════════════════════════════════════

def train_model_for_window(
    ticker: str,
    raw_data: Dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    horizon_days: int = 5,
    n_splits: int = 5,
    embargo_days: int = 10,
    half_life: int = 252,
    verbose: bool = False,
    calibrator_type: str = "temperature",
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict], List[str]]:
    """Train a LightGBM model on the given time window.

    Uses absolute return labels (spy_data=None).

    calibrator_type:
        - "temperature" (default): legacy 1-param Temperature Scaling.
        - "isotonic": real non-parametric sklearn IsotonicRegression
          (Stage 1 benchmark winner: NVDA −40%→−12%, AAPL +60%→+63%).
        - "none": no calibration, raw LightGBM probabilities.

    Returns:
        (lgb_model, calibrator, meta_dict, macro_fund_cols_used)
    """
    data = raw_data.get(ticker)
    if data is None or data.empty:
        return None, None, None, []

    # Filter to training window
    data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
    if len(data) < 120:
        if verbose:
            print(f"    [skip] Only {len(data)} rows in training window (need >= 120)")
        return None, None, None, []

    # Step 1: Compute base features + absolute return labels
    frame = compute_base_features(data)
    frame = build_labels(frame, horizon_days=horizon_days, spy_data=None)
    frame["ticker"] = ticker

    required_cols = BASE_FEATURE_COLUMNS + ["label"]
    frame = frame.dropna(subset=required_cols)

    if len(frame) < 100:
        if verbose:
            print(f"    [skip] Only {len(frame)} valid rows after feature computation")
        return None, None, None, []

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
        return None, None, None, [], None, None, None

    # Step 6: Fit calibrator + conformal scores
    if calibrator_type == "isotonic":
        calibrator = fit_real_isotonic_calibrator(oof_predictions, y, verbose=verbose)
    elif calibrator_type == "platt":
        calibrator = fit_platt_calibrator(oof_predictions, y, verbose=verbose)
    elif calibrator_type == "none":
        calibrator = None
    else:  # "temperature" (default / legacy)
        calibrator = fit_isotonic_calibrator(oof_predictions, y)
    conformal_info = compute_conformal_scores(oof_predictions, y, calibrator)

    # Build meta dict
    meta = {
        "version": 4,
        "model_type": "lightgbm",
        "label_type": "absolute_return",
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
        **conformal_info,
    }

    # Extract future returns aligned with oof_predictions / y for Layer 1
    future_returns_np = None
    if "future_return" in frame.columns:
        future_returns_np = frame["future_return"].values.astype(np.float64)

    return final_model, calibrator, meta, macro_fund_cols_used, oof_predictions, y, future_returns_np


# ═══════════════════════════════════════════════════════════════════════════
# Model Scoring (with uncertainty quantification)
# ═══════════════════════════════════════════════════════════════════════════

def score_features(
    features: dict,
    model,
    meta: dict,
    calibrator=None,
    regime_features: dict | None = None,
    macro_fund_row: dict | None = None,
) -> Tuple[float, float, dict]:
    """Score features with LightGBM model and return (raw_prob, prob, uncertainty_info).

    Returns:
        raw_prob: Raw LightGBM probability.
        prob: Calibrated probability (or raw if no calibrator).
        uncertainty_info: Dict with keys: uncertainty, prediction_set, is_uncertain.
    """
    feature_cols = meta.get("feature_columns", [])
    regime_cols = meta.get("regime_features", [])
    macro_fund_cols = meta.get("macro_fundamental_features", [])
    rank_cols = meta.get("rank_feature_columns", [])
    cat_cols = meta.get("categorical_features", [])
    all_cols = feature_cols + regime_cols + macro_fund_cols + rank_cols + cat_cols

    row = {}
    for col in feature_cols:
        val = features.get(col)
        row[col] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan

    for col in regime_cols:
        row[col] = regime_features.get(col, 0.0) if regime_features else 0.0

    for col in macro_fund_cols:
        if macro_fund_row and col in macro_fund_row and pd.notna(macro_fund_row[col]):
            row[col] = float(macro_fund_row[col])
        else:
            row[col] = np.nan

    for col in rank_cols:
        row[col] = 0.5
    for col in cat_cols:
        row[col] = 0.0

    X = pd.DataFrame([row])[all_cols]
    raw_prob = float(model.predict(X)[0])

    # Apply calibration if available
    prob = raw_prob
    if calibrator is not None:
        try:
            prob = float(calibrator.predict([raw_prob])[0])
            prob = max(0.02, min(0.98, prob))
        except Exception:
            prob = raw_prob

    # ── Uncertainty quantification ───────────────────────────────────
    uncertainty_info = {
        "uncertainty": None,
        "prediction_set": None,
        "is_uncertain": False,
    }

    # Method 1: Tree ensemble dispersion
    try:
        n_trees = model.num_trees()
        n_groups = min(5, n_trees)
        if n_groups >= 2:
            group_size = n_trees // n_groups
            group_preds = []
            for g in range(n_groups):
                start = g * group_size
                n_iter = group_size if g < n_groups - 1 else n_trees - start
                pred = model.predict(X, start_iteration=start, num_iteration=n_iter)
                group_preds.append(float(pred[0]))
            uncertainty_info["uncertainty"] = float(np.std(group_preds))
            if uncertainty_info["uncertainty"] > 0.15:
                uncertainty_info["is_uncertain"] = True
    except Exception:
        pass

    # Method 2: Conformal prediction set (uses calibrated prob to match
    # training-time conformal scores which are computed on calibrated probs)
    quantiles = meta.get("conformal_scores_quantiles", {})
    threshold = quantiles.get("q90")
    if threshold is not None:
        prediction_set = []
        if (1.0 - prob) <= threshold:
            prediction_set.append("up")
        if prob <= threshold:
            prediction_set.append("down")
        uncertainty_info["prediction_set"] = prediction_set
        if len(prediction_set) == 2:
            uncertainty_info["is_uncertain"] = True
        elif len(prediction_set) == 0:
            uncertainty_info["is_uncertain"] = True

    return raw_prob, prob, uncertainty_info


# ═══════════════════════════════════════════════════════════════════════════
# Simplified RiskAgent (inline)
# ═══════════════════════════════════════════════════════════════════════════

# Kelly defaults
KELLY_AVG_WIN = 0.09
KELLY_AVG_LOSS = 0.02

# Position limits
MAX_POSITION_SIZE = 1.0
MIN_POSITION_THRESHOLD = 0.03

# Stop-loss / take-profit (vol-bucket adaptive)
#   high-vol (vol_20 >= 0.35) -> 3.5x daily_vol  (NVDA, TSLA)
#   medium-vol                -> 2.5x daily_vol  (AAPL, default)
#   low-vol  (vol_20 <  0.20) -> 2.0x daily_vol  (KO, defensive)
RISK_REWARD_RATIO = 2.0
STOP_VOL_MULTIPLIER_HIGH = 3.5
STOP_VOL_MULTIPLIER_MID = 2.5
STOP_VOL_MULTIPLIER_LOW = 2.0
STOP_VOL_HIGH_THR = 0.35
STOP_VOL_LOW_THR = 0.20
STOP_MIN = 0.01
STOP_MAX = 0.12  # raised from 0.08 so the high-vol bucket actually takes effect

# Probability-conviction Kelly scaling (B2).
# See agents/risk_agent.py for the full rationale. Anchored on Layer-1
# per-month adaptive buy/sell thresholds (not hard-coded 0.55/0.45).
try:
    PROB_CONVICTION_ALPHA = float(os.getenv("RISK_PROB_CONVICTION_ALPHA", "3.0"))
except (TypeError, ValueError):
    PROB_CONVICTION_ALPHA = 3.0
FALLBACK_BUY_THRESHOLD = 0.55
FALLBACK_SELL_THRESHOLD = 0.45


def compute_probability_conviction(
    action: str,
    probability_up: float,
    buy_threshold: float,
    sell_threshold: float,
    alpha: float = PROB_CONVICTION_ALPHA,
) -> float:
    """Return multiplier >= 1.0 based on distance beyond the Layer-1 cutoff."""
    if action == "buy":
        edge = max(0.0, probability_up - buy_threshold)
    elif action == "sell":
        edge = max(0.0, sell_threshold - probability_up)
    else:
        return 1.0
    return 1.0 + alpha * edge


def compute_prediction_kelly(
    probability_up: float,
    action: str,
    avg_win: float = KELLY_AVG_WIN,
    avg_loss: float = KELLY_AVG_LOSS,
) -> float:
    """Compute Kelly fraction using current prediction probability."""
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss

    if action == "buy":
        p = probability_up
    elif action == "sell":
        p = 1.0 - probability_up
    else:
        p = max(probability_up, 1.0 - probability_up)

    q = 1.0 - p
    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly, MAX_POSITION_SIZE)


def adaptive_prediction_set(
    p_cal: float,
    buy_threshold: float,
    sell_threshold: float,
    buy_disabled: bool = False,
    short_disabled: bool = False,
) -> list:
    """Build conformal prediction set aligned with Layer-1 adaptive thresholds.

    Instead of using the training-time OOF q90 (which is too wide for high-
    volatility tickers like NVDA and rejects most valid signals), we derive
    ambiguity directly from the per-month buy/sell thresholds that Layer-1
    already validated to hit precision>=55%. This keeps the conformal filter
    semantically aligned with the decision thresholds:

      * If Layer-1 enabled buying for this month and p_cal >= buy_threshold
        -> "up" enters the set (model is confident enough to buy).
      * If Layer-1 enabled shorting and p_cal <= sell_threshold
        -> "down" enters the set.
      * If a direction is disabled by Layer-1, it NEVER enters the set,
        which prevents the RiskAgent from silently reviving it.

    Resulting behaviour:
      |set| == 0 : prob in Layer-1 neutral zone -> will be rejected (reject_reason=conformal_empty)
      |set| == 1 : one-sided confidence -> normal trade
      |set| == 2 : mathematically impossible when buy_threshold > sell_threshold
                   (Layer-1 guarantees this), kept as a safety net.
    """
    pred_set: list = []
    if (not buy_disabled) and p_cal >= buy_threshold:
        pred_set.append("up")
    if (not short_disabled) and p_cal <= sell_threshold:
        pred_set.append("down")
    return pred_set


def simplified_risk_plan(
    action: str,
    probability_up: float,
    volatility_20: float,
    horizon_days: int = 5,
    uncertainty_info: dict | None = None,
    buy_threshold: float = FALLBACK_BUY_THRESHOLD,
    sell_threshold: float = FALLBACK_SELL_THRESHOLD,
    full_position: bool = False,
) -> dict:
    """Simplified RiskAgent: Prediction Kelly + conformal uncertainty + stop-loss.

    Steps:
      ① Prediction Kelly position sizing
      ①b Probability-conviction scaling (Layer-1-anchored)
            OR, if ``full_position=True``: skip Kelly*conviction and
            lock the position at MAX_POSITION_SIZE whenever a trade is
            taken. The negative-EV / conformal / stop-TP guards stay
            fully active; only the fractional Kelly sizing is bypassed.
      ② Uncertainty filtering (conformal + tree dispersion)
      ③ Direction judgment
      ④ Minimum position threshold
      ⑤ Dynamic stop-loss / take-profit
    """
    risk_flags = []
    reject_reason = None

    # ── ① Prediction Kelly position sizing ──────────────────────────────
    kelly = compute_prediction_kelly(
        probability_up=probability_up,
        action=action,
    )

    if kelly <= 0:
        return {
            "position_size": 0.0,
            "kelly_fraction": 0.0,
            "stop_loss_pct": STOP_MIN,
            "take_profit_pct": STOP_MIN * RISK_REWARD_RATIO,
            "max_holding_days": horizon_days,
            "reject_reason": "negative_expected_value",
            "risk_flags": ["negative_expected_value"],
        }

    # ①b Position sizing: Kelly * prob_conviction  OR  full-position lock.
    if full_position:
        # Full-position mode: bypass fractional Kelly and conviction scaling.
        # Negative-EV was already rejected above; Conformal, high tree
        # dispersion, stop-loss / take-profit and min-position guards
        # below remain fully active.
        position_size = MAX_POSITION_SIZE
        prob_conviction = 1.0
        risk_flags.append("kelly_locked_full")
    else:
        # Probability-conviction scaling (Layer-1-anchored).
        prob_conviction = compute_probability_conviction(
            action=action,
            probability_up=probability_up,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        position_size = min(kelly * prob_conviction, MAX_POSITION_SIZE)
        if prob_conviction > 1.0:
            risk_flags.append("prob_conviction_boost")

    # ── ② Uncertainty-aware filtering (Conformal Prediction) ─────────────
    if uncertainty_info is not None:
        prediction_set = uncertainty_info.get("prediction_set") or []
        tree_dispersion = uncertainty_info.get("uncertainty")

        if len(prediction_set) == 2:
            position_size = 0.0
            reject_reason = "conformal_ambiguous"
            risk_flags.append("conformal_ambiguous")
        elif len(prediction_set) == 0:
            position_size = 0.0
            reject_reason = "conformal_empty"
            risk_flags.append("conformal_empty")
        elif tree_dispersion is not None and tree_dispersion > 0.15:
            position_size *= 0.5
            risk_flags.append("high_tree_dispersion")

    # ── ③ Direction ──────────────────────────────────────────────────────
    # Align with agents/risk_agent.py (production path):
    # hold → direction = 0 → no trade. Do NOT take a reverse quarter-Kelly
    # position based on prob, which would silently bypass Layer-1 adaptive
    # thresholds (e.g. sell_threshold=-0.5 means "shorts disabled").
    if action == "buy":
        direction = 1
    elif action == "sell":
        direction = -1
    else:
        direction = 0
        position_size = 0.0
        if reject_reason is None:
            reject_reason = "no_strong_edge"
        risk_flags.append("no_strong_edge")

    position_size = direction * position_size

    # Clamp
    position_size = max(-MAX_POSITION_SIZE, min(MAX_POSITION_SIZE, position_size))

    # ── ④ Minimum position threshold ────────────────────────────────────
    if abs(position_size) > 0 and abs(position_size) < MIN_POSITION_THRESHOLD:
        position_size = 0.0
        reject_reason = reject_reason or "position_too_small"
        risk_flags.append("position_too_small")

    # ── ⑤ Dynamic stop-loss / take-profit (vol-bucket adaptive) ─────────
    daily_vol = volatility_20 / math.sqrt(252.0)
    if volatility_20 >= STOP_VOL_HIGH_THR:
        stop_multiplier = STOP_VOL_MULTIPLIER_HIGH
    elif volatility_20 < STOP_VOL_LOW_THR:
        stop_multiplier = STOP_VOL_MULTIPLIER_LOW
    else:
        stop_multiplier = STOP_VOL_MULTIPLIER_MID
    base_stop = daily_vol * stop_multiplier
    stop_loss_pct = min(STOP_MAX, max(STOP_MIN, base_stop))
    take_profit_pct = stop_loss_pct * RISK_REWARD_RATIO

    return {
        "position_size": float(position_size),
        "kelly_fraction": float(kelly),
        "stop_loss_pct": float(stop_loss_pct),
        "take_profit_pct": float(take_profit_pct),
        "max_holding_days": int(horizon_days),
        "reject_reason": reject_reason,
        "risk_flags": risk_flags,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Monthly Backtest Engine (Stage 2: with RiskAgent + stop-loss/take-profit)
# ═══════════════════════════════════════════════════════════════════════════

def run_month_backtest_stage2(
    lgb_model,
    calibrator,
    meta: dict,
    ohlcv_data: pd.DataFrame,
    spy_close: Optional[pd.Series],
    macro_fund_df: Optional[pd.DataFrame],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    equity_start: float = 1.0,
    horizon: int = 5,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.50,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    verbose: bool = False,
    buy_disabled: bool = False,
    short_disabled: bool = False,
    use_adaptive_conformal: bool = True,
    full_position: bool = False,
) -> Tuple[List[dict], float, dict, dict]:
    """Run Stage 2 backtest for a single month (with RiskAgent).

    Returns:
        (trade_log, final_equity, equity_curve_dict, risk_stats)
    """
    dates = ohlcv_data.index
    open_prices = pd.to_numeric(ohlcv_data["Open"], errors="coerce").values
    high_prices = pd.to_numeric(ohlcv_data["High"], errors="coerce").values
    low_prices = pd.to_numeric(ohlcv_data["Low"], errors="coerce").values
    close_prices = pd.to_numeric(ohlcv_data["Close"], errors="coerce").values

    cost_per_trade = (cost_bps + slippage_bps) / 10000.0 * 2

    start_idx = dates.searchsorted(month_start)
    start_idx = max(start_idx, 60)
    end_idx = dates.searchsorted(month_end, side="right")

    trade_log = []
    equity = equity_start
    equity_curve = {}
    risk_stats = {
        "total_signals": 0,
        "executed_trades": 0,
        "rejected_negative_ev": 0,
        "rejected_conformal_ambiguous": 0,
        "rejected_conformal_empty": 0,
        "rejected_position_too_small": 0,
        "stopped_out": 0,
        "took_profit": 0,
        "horizon_exit": 0,
        "high_tree_dispersion": 0,
        # Adaptive conformal vs legacy q90 comparison counters
        "conformal_mode": "adaptive" if use_adaptive_conformal else "q90",
        "adaptive_vs_q90_overrides": 0,   # adaptive passes but q90 would ambig
        "adaptive_vs_q90_extra_reject": 0, # adaptive rejects but q90 would pass
    }

    t = start_idx
    while t < min(end_idx, len(dates) - horizon):
        current_date = dates[t]
        if current_date < month_start or current_date > month_end:
            t += 1
            continue

        # Compute features
        data_slice = ohlcv_data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            t += horizon
            continue

        current_close = float(close_prices[t])
        regime_feats = compute_regime_from_features(features, current_close)

        # Look up macro/fund features
        mf_row = None
        if macro_fund_df is not None and not macro_fund_df.empty:
            valid_dates = macro_fund_df.index[macro_fund_df.index <= current_date]
            if len(valid_dates) > 0:
                nearest_date = valid_dates[-1]
                mf_row = macro_fund_df.loc[nearest_date].to_dict()

        # Score with uncertainty
        raw_prob, prob, uncertainty_info = score_features(
            features, lgb_model, meta, calibrator,
            regime_features=regime_feats,
            macro_fund_row=mf_row,
        )

        # ── Adaptive conformal override ──────────────────────────────────
        # Replace the training-time q90-based prediction_set with one that is
        # aligned with Layer-1's per-month buy/sell thresholds. Keep the
        # original q90 set for diagnostics. See adaptive_prediction_set() doc.
        if use_adaptive_conformal and uncertainty_info is not None:
            q90_set = uncertainty_info.get("prediction_set") or []
            adaptive_set = adaptive_prediction_set(
                p_cal=prob,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                buy_disabled=buy_disabled,
                short_disabled=short_disabled,
            )
            uncertainty_info["prediction_set_q90"] = q90_set
            uncertainty_info["prediction_set"] = adaptive_set
            # Track disagreements between the two schemes
            q90_ambig = (len(q90_set) != 1)
            adp_ambig = (len(adaptive_set) != 1)
            if q90_ambig and not adp_ambig:
                risk_stats["adaptive_vs_q90_overrides"] += 1
            elif (not q90_ambig) and adp_ambig:
                risk_stats["adaptive_vs_q90_extra_reject"] += 1

        # Decision: threshold
        if prob > buy_threshold:
            action = "buy"
        elif prob < sell_threshold:
            action = "sell"
        else:
            action = "hold"

        risk_stats["total_signals"] += 1

        # Get risk plan
        risk_plan = simplified_risk_plan(
            action=action,
            probability_up=prob,
            volatility_20=features.get("volatility_20", 0.25),
            horizon_days=horizon,
            uncertainty_info=uncertainty_info,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            full_position=full_position,
        )

        position_size = risk_plan["position_size"]
        stop_loss_pct = risk_plan["stop_loss_pct"]
        take_profit_pct = risk_plan["take_profit_pct"]
        reject_reason = risk_plan["reject_reason"]

        # Track rejections
        if reject_reason == "negative_expected_value":
            risk_stats["rejected_negative_ev"] += 1
        elif reject_reason == "conformal_ambiguous":
            risk_stats["rejected_conformal_ambiguous"] += 1
        elif reject_reason == "conformal_empty":
            risk_stats["rejected_conformal_empty"] += 1
        elif reject_reason == "position_too_small":
            risk_stats["rejected_position_too_small"] += 1
        if "high_tree_dispersion" in risk_plan.get("risk_flags", []):
            risk_stats["high_tree_dispersion"] += 1

        # Execute trade
        if abs(position_size) > 0 and t + 1 < len(dates):
            entry_price = open_prices[t + 1]
            if np.isnan(entry_price) or entry_price <= 0:
                t += horizon
                continue

            direction = 1.0 if position_size > 0 else -1.0
            abs_position = abs(position_size)

            # ── Daily stop-loss / take-profit check ──────────────────
            exit_price = None
            exit_reason = "horizon"
            exit_idx = None

            entry_idx = t + 1
            for day_offset in range(1, horizon + 1):
                check_idx = entry_idx + day_offset
                if check_idx >= len(dates):
                    break

                day_high = high_prices[check_idx]
                day_low = low_prices[check_idx]

                if np.isnan(day_high) or np.isnan(day_low):
                    continue

                if direction > 0:  # Long
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                    tp_price = entry_price * (1.0 + take_profit_pct)
                    if day_low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                        exit_idx = check_idx
                        break
                    elif day_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        exit_idx = check_idx
                        break
                else:  # Short
                    stop_price = entry_price * (1.0 + stop_loss_pct)
                    tp_price = entry_price * (1.0 - take_profit_pct)
                    if day_high >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop_loss"
                        exit_idx = check_idx
                        break
                    elif day_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        exit_idx = check_idx
                        break

            # Horizon exit
            if exit_price is None:
                exit_idx = min(entry_idx + horizon, len(dates) - 1)
                exit_price = close_prices[exit_idx]
                exit_reason = "horizon"

            if np.isnan(exit_price) or exit_price <= 0:
                t += horizon
                continue

            # Calculate return (scaled by position size)
            raw_return_per_unit = (exit_price / entry_price - 1.0) * direction
            raw_return = raw_return_per_unit * abs_position
            net_return = raw_return - cost_per_trade * abs_position

            equity *= (1.0 + net_return)
            risk_stats["executed_trades"] += 1

            if exit_reason == "stop_loss":
                risk_stats["stopped_out"] += 1
            elif exit_reason == "take_profit":
                risk_stats["took_profit"] += 1
            else:
                risk_stats["horizon_exit"] += 1

            # SPY return for comparison
            spy_return = 0.0
            if spy_close is not None:
                spy_entry = spy_close.iloc[entry_idx] if entry_idx < len(spy_close) else np.nan
                spy_exit = spy_close.iloc[exit_idx] if exit_idx < len(spy_close) else np.nan
                if pd.notna(spy_entry) and pd.notna(spy_exit) and spy_entry > 0:
                    spy_return = (spy_exit / spy_entry) - 1.0

            trade = {
                "date": current_date.strftime("%Y-%m-%d"),
                "action": action,
                "probability_up": round(prob, 6),
                "raw_probability_up": round(raw_prob, 6),
                "direction": direction,
                "position_size": round(abs_position, 4),
                "kelly_fraction": round(risk_plan["kelly_fraction"], 4),
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "exit_reason": exit_reason,
                "stop_loss_pct": round(stop_loss_pct, 4),
                "take_profit_pct": round(take_profit_pct, 4),
                "raw_return": round(float(raw_return), 6),
                "net_return": round(float(net_return), 6),
                "spy_return": round(float(spy_return), 6),
                "excess_return": round(float(raw_return_per_unit - spy_return), 6),
                "equity": round(float(equity), 6),
                "risk_flags": risk_plan["risk_flags"],
                "tree_dispersion": round(uncertainty_info.get("uncertainty", 0) or 0, 6),
                "prediction_set": uncertainty_info.get("prediction_set", []),
            }
            trade_log.append(trade)
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity
        else:
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

        t += horizon

    return trade_log, equity, equity_curve, risk_stats


def run_month_backtest_stage1(
    lgb_model,
    meta: dict,
    ohlcv_data: pd.DataFrame,
    spy_close: Optional[pd.Series],
    macro_fund_df: Optional[pd.DataFrame],
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    equity_start: float = 1.0,
    horizon: int = 5,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.50,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    verbose: bool = False,
    calibrator: Any = None,
    buy_disabled: bool = False,
    short_disabled: bool = False,
) -> Tuple[List[dict], float, dict]:
    """Run Stage 1 backtest for a single month (no RiskAgent, for comparison).

    NOTE: S1 must use the **same calibrated-probability space** as S2 so that
    Layer-1 adaptive thresholds (which are derived on calibrated OOF probs)
    are compared against a consistent decision variable. Otherwise S1 trades
    are judged in raw-prob space while the threshold lives in calibrated space,
    and the S1 vs S2 comparison becomes meaningless.
    """
    dates = ohlcv_data.index
    open_prices = pd.to_numeric(ohlcv_data["Open"], errors="coerce").values
    close_prices = pd.to_numeric(ohlcv_data["Close"], errors="coerce").values

    cost_per_trade = (cost_bps + slippage_bps) / 10000.0 * 2

    start_idx = dates.searchsorted(month_start)
    start_idx = max(start_idx, 60)
    end_idx = dates.searchsorted(month_end, side="right")

    trade_log = []
    equity = equity_start
    equity_curve = {}

    t = start_idx
    while t < min(end_idx, len(dates) - horizon):
        current_date = dates[t]
        if current_date < month_start or current_date > month_end:
            t += 1
            continue

        data_slice = ohlcv_data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            t += horizon
            continue

        current_close = float(close_prices[t])
        regime_feats = compute_regime_from_features(features, current_close)

        mf_row = None
        if macro_fund_df is not None and not macro_fund_df.empty:
            valid_dates = macro_fund_df.index[macro_fund_df.index <= current_date]
            if len(valid_dates) > 0:
                mf_row = macro_fund_df.loc[valid_dates[-1]].to_dict()

        # Score (use calibrator so S1 and S2 live in the same probability space).
        raw_prob, prob, _ = score_features(
            features, lgb_model, meta, calibrator,
            regime_features=regime_feats,
            macro_fund_row=mf_row,
        )

        # Apply Layer-1 buy/short disable flags (same logic as S2 entry gate)
        if prob > buy_threshold and not buy_disabled:
            action = "buy"
            direction = 1.0
        elif prob < sell_threshold and not short_disabled:
            action = "sell"
            direction = -1.0
        else:
            action = "hold"
            direction = 0.0

        if direction != 0.0 and t + 1 < len(dates):
            entry_price = open_prices[t + 1]
            exit_idx = min(t + 1 + horizon, len(dates) - 1)
            exit_price = close_prices[exit_idx]

            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price <= 0:
                t += horizon
                continue

            raw_return = (exit_price / entry_price - 1.0) * direction
            net_return = raw_return - cost_per_trade
            equity *= (1.0 + net_return)

            spy_return = 0.0
            if spy_close is not None:
                spy_entry = spy_close.iloc[t + 1] if (t + 1) < len(spy_close) else np.nan
                spy_exit = spy_close.iloc[exit_idx] if exit_idx < len(spy_close) else np.nan
                if pd.notna(spy_entry) and pd.notna(spy_exit) and spy_entry > 0:
                    spy_return = (spy_exit / spy_entry) - 1.0

            trade = {
                "date": current_date.strftime("%Y-%m-%d"),
                "action": action,
                "probability_up": round(prob, 6),
                "raw_probability_up": round(float(raw_prob), 6),
                "direction": direction,
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "raw_return": round(float(raw_return), 6),
                "net_return": round(float(net_return), 6),
                "spy_return": round(float(spy_return), 6),
                "excess_return": round(float(raw_return * direction - spy_return), 6),
                "equity": round(float(equity), 6),
            }
            trade_log.append(trade)
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity
        else:
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

        t += horizon

    return trade_log, equity, equity_curve


# ═══════════════════════════════════════════════════════════════════════════
# Analysis & Reporting
# ═══════════════════════════════════════════════════════════════════════════

def analyze_backtest(all_trades: List[dict], benchmark_return: float) -> dict:
    """Compute comprehensive metrics from aggregated trade log."""
    if not all_trades:
        return {"error": "No trades executed"}

    returns = [t["net_return"] for t in all_trades]
    buy_trades = [t for t in all_trades if t["action"] == "buy"]
    sell_trades = [t for t in all_trades if t["action"] == "sell"]

    equity = 1.0
    for r in returns:
        equity *= (1.0 + r)
    total_return = equity - 1.0

    n_trades = len(all_trades)
    n_buy = len(buy_trades)
    n_sell = len(sell_trades)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    hit_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean([abs(r) for r in losses])) if losses else 0.0

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=0)) if len(returns) > 1 else 1.0
    horizon = 5
    sharpe = (mean_ret / std_ret * np.sqrt(252.0 / horizon)) if std_ret > 0 else 0.0

    downside = [r for r in returns if r < 0]
    downside_std = float(np.std(downside, ddof=0)) if downside else 1.0
    sortino = (mean_ret / downside_std * np.sqrt(252.0 / horizon)) if downside_std > 0 else 0.0

    equity_vals = [1.0]
    for r in returns:
        equity_vals.append(equity_vals[-1] * (1 + r))
    equity_arr = np.array(equity_vals)
    cummax = np.maximum.accumulate(equity_arr)
    drawdowns = equity_arr / cummax - 1.0
    max_dd = float(np.min(drawdowns))

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss
    alpha = total_return - benchmark_return

    buy_returns = [t["net_return"] for t in buy_trades]
    sell_returns = [t["net_return"] for t in sell_trades]
    buy_hit = sum(1 for r in buy_returns if r > 0) / len(buy_returns) if buy_returns else 0.0
    sell_hit = sum(1 for r in sell_returns if r > 0) / len(sell_returns) if sell_returns else 0.0
    buy_avg = float(np.mean(buy_returns)) if buy_returns else 0.0
    sell_avg = float(np.mean(sell_returns)) if sell_returns else 0.0

    probs = [t["probability_up"] for t in all_trades]
    actuals = [t["raw_return"] * t.get("direction", 1.0) for t in all_trades]
    if len(probs) > 5:
        trade_ic, trade_ic_pval = stats.spearmanr(probs, actuals)
    else:
        trade_ic, trade_ic_pval = 0.0, 1.0

    excess_returns = [t.get("excess_return", 0.0) for t in all_trades]
    avg_excess = float(np.mean(excess_returns)) if excess_returns else 0.0
    pct_outperform = sum(1 for e in excess_returns if e > 0) / len(excess_returns) if excess_returns else 0.0

    max_consec_wins = 0
    max_consec_losses = 0
    cur_wins = 0
    cur_losses = 0
    for r in returns:
        if r > 0:
            cur_wins += 1
            cur_losses = 0
            max_consec_wins = max(max_consec_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_consec_losses = max(max_consec_losses, cur_losses)

    return {
        "total_return": round(total_return, 6),
        "benchmark_return": round(benchmark_return, 6),
        "alpha": round(alpha, 6),
        "n_trades": n_trades,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "hit_rate": round(hit_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "avg_trade_return": round(mean_ret, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 6),
        "trade_ic": round(float(trade_ic), 6),
        "trade_ic_pval": round(float(trade_ic_pval), 6),
        "buy_hit_rate": round(buy_hit, 4),
        "buy_avg_return": round(buy_avg, 6),
        "sell_hit_rate": round(sell_hit, 4),
        "sell_avg_return": round(sell_avg, 6),
        "avg_excess_return": round(avg_excess, 6),
        "pct_outperform_spy": round(pct_outperform, 4),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
    }


def print_comparison_report(
    s1_metrics: dict,
    s2_metrics: dict,
    s2_risk_stats: dict,
    monthly_perf: List[dict],
    config: dict,
) -> None:
    """Pretty-print Stage 1 vs Stage 2 comparison report."""
    ticker = config["ticker"]

    print(f"\n{'='*80}")
    print(f"  Walk-Forward Stage 2: ForecastAgent + RiskAgent — {ticker}")
    print(f"  Mode: Monthly retraining (absolute return label, no look-ahead bias)")
    print(f"{'='*80}")

    print(f"\n📋 Configuration:")
    print(f"  Period:          {config['start_date']} → {config['end_date']}")
    print(f"  Training window: {config['train_years']} years (rolling)")
    print(f"  Retrain freq:    monthly")
    print(f"  Horizon:         {config['horizon']}d")
    print(f"  Thresholds:      buy > {config['buy_threshold']}, sell < {config['sell_threshold']}")
    print(f"  Costs:           {config['cost_bps']}bps + {config['slippage_bps']}bps slippage")
    print(f"  Label:           absolute return")

    # ── Stage 1 vs Stage 2 comparison ────────────────────────────────────
    def _delta(s2_val, s1_val, fmt="+.2%", higher_better=True):
        d = s2_val - s1_val
        if fmt == "+.2%":
            s = f"{d:+.2%}"
        elif fmt == "+.4f":
            s = f"{d:+.4f}"
        else:
            s = f"{d:+.6f}"
        if higher_better:
            icon = "🔥" if d > 0.001 else ("✅" if d >= 0 else "❌")
        else:
            icon = "🔥" if d < -0.001 else ("✅" if d <= 0 else "❌")
        return f"{s} {icon}"

    print(f"\n── Stage 1 vs Stage 2 Comparison ──")
    print(f"  {'Metric':<25} {'Stage 1':>12} {'Stage 2':>12} {'Delta':>18}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*18}")

    rows = [
        ("Strategy Return", "total_return", "+.2%", True),
        ("Alpha vs SPY", "alpha", "+.2%", True),
        ("Sharpe Ratio", "sharpe_ratio", "+.4f", True),
        ("Sortino Ratio", "sortino_ratio", "+.4f", True),
        ("Max Drawdown", "max_drawdown", "+.2%", False),
        ("Hit Rate", "hit_rate", "+.2%", True),
        ("Profit Factor", "profit_factor", "+.4f", True),
        ("Avg Trade Return", "avg_trade_return", "+.6f", True),
        ("Avg Win", "avg_win", "+.6f", True),
        ("Avg Loss", "avg_loss", "+.6f", False),
        ("Trade IC", "trade_ic", "+.6f", True),
        ("Buy Hit Rate", "buy_hit_rate", "+.2%", True),
        ("Buy Avg Return", "buy_avg_return", "+.6f", True),
        ("Sell Hit Rate", "sell_hit_rate", "+.2%", True),
        ("Sell Avg Return", "sell_avg_return", "+.6f", True),
        ("% Outperform SPY", "pct_outperform_spy", "+.2%", True),
    ]

    for label, key, fmt, higher_better in rows:
        s1_val = s1_metrics.get(key, 0)
        s2_val = s2_metrics.get(key, 0)
        if fmt == "+.2%":
            s1_str = f"{s1_val:+.2%}"
            s2_str = f"{s2_val:+.2%}"
        elif fmt == "+.4f":
            s1_str = f"{s1_val:+.4f}"
            s2_str = f"{s2_val:+.4f}"
        else:
            s1_str = f"{s1_val:+.6f}"
            s2_str = f"{s2_val:+.6f}"
        delta = _delta(s2_val, s1_val, fmt, higher_better)
        print(f"  {label:<25} {s1_str:>12} {s2_str:>12} {delta:>18}")

    print(f"\n  Trade counts: Stage 1 = {s1_metrics['n_trades']}, Stage 2 = {s2_metrics['n_trades']}")
    print(f"    Stage 1: Buy={s1_metrics['n_buy']}, Sell={s1_metrics['n_sell']}")
    print(f"    Stage 2: Buy={s2_metrics['n_buy']}, Sell={s2_metrics['n_sell']}")

    # ── Risk Stats ───────────────────────────────────────────────────────
    print(f"\n── RiskAgent Statistics ──")
    total_sig = s2_risk_stats.get("total_signals", 0)
    executed = s2_risk_stats.get("executed_trades", 0)
    rejected = total_sig - executed
    print(f"  Total signals:           {total_sig}")
    print(f"  Executed trades:         {executed} ({executed/max(total_sig,1):.1%})")
    print(f"  Rejected trades:         {rejected} ({rejected/max(total_sig,1):.1%})")
    print(f"    Negative EV:           {s2_risk_stats.get('rejected_negative_ev', 0)}")
    print(f"    Conformal ambiguous:   {s2_risk_stats.get('rejected_conformal_ambiguous', 0)}")
    print(f"    Conformal empty:       {s2_risk_stats.get('rejected_conformal_empty', 0)}")
    print(f"    Position too small:    {s2_risk_stats.get('rejected_position_too_small', 0)}")
    print(f"  High tree dispersion:    {s2_risk_stats.get('high_tree_dispersion', 0)} (position halved)")
    print(f"\n  Exit reasons:")
    print(f"    Horizon exit:          {s2_risk_stats.get('horizon_exit', 0)}")
    print(f"    Stop-loss:             {s2_risk_stats.get('stopped_out', 0)}")
    print(f"    Take-profit:           {s2_risk_stats.get('took_profit', 0)}")

    # ── Per-month breakdown ──────────────────────────────────────────────
    print(f"\n── Per-Month Performance (Walk-Forward) ──")
    print(f"  {'Month':<10} {'S1 Ret':>10} {'S2 Ret':>10} {'S2 Trades':>10} {'S2 Hit':>8} {'CV AUC':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for m in monthly_perf:
        s1_ret = m.get("s1_return", 0)
        s2_ret = m.get("s2_return", 0)
        n_t = m.get("s2_trades", 0)
        hit = m.get("s2_hit_rate", 0)
        auc = m.get("cv_auc", 0)
        s1_str = f"{s1_ret:+.4%}" if m.get("s1_trades", 0) > 0 else "    N/A"
        s2_str = f"{s2_ret:+.4%}" if n_t > 0 else "    N/A"
        hit_str = f"{hit:.2%}" if n_t > 0 else "  N/A"
        auc_str = f"{auc:.4f}" if auc > 0 else "  N/A"
        print(f"  {m['month']:<10} {s1_str:>10} {s2_str:>10} {n_t:>10} {hit_str:>8} {auc_str:>8}")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    s1_ret = s1_metrics["total_return"]
    s2_ret = s2_metrics["total_return"]
    s1_sharpe = s1_metrics["sharpe_ratio"]
    s2_sharpe = s2_metrics["sharpe_ratio"]
    s1_dd = s1_metrics["max_drawdown"]
    s2_dd = s2_metrics["max_drawdown"]

    improvements = []
    regressions = []

    if s2_ret > s1_ret:
        improvements.append(f"Return: {s1_ret:+.2%} → {s2_ret:+.2%}")
    else:
        regressions.append(f"Return: {s1_ret:+.2%} → {s2_ret:+.2%}")

    if s2_sharpe > s1_sharpe:
        improvements.append(f"Sharpe: {s1_sharpe:.4f} → {s2_sharpe:.4f}")
    else:
        regressions.append(f"Sharpe: {s1_sharpe:.4f} → {s2_sharpe:.4f}")

    if s2_dd > s1_dd:  # Less negative = better
        improvements.append(f"Max DD: {s1_dd:.2%} → {s2_dd:.2%}")
    else:
        regressions.append(f"Max DD: {s1_dd:.2%} → {s2_dd:.2%}")

    if len(improvements) >= 2:
        print("  ✅ VERDICT: RiskAgent HELPS performance:")
        for imp in improvements:
            print(f"     ✅ {imp}")
        for reg in regressions:
            print(f"     ⚠️  {reg}")
        print("  → Proceed to WF Stage 3 (full Orchestrator).")
    elif len(improvements) == 1 and s2_dd > s1_dd:
        print("  ⚠️  VERDICT: RiskAgent provides mixed results:")
        for imp in improvements:
            print(f"     ✅ {imp}")
        for reg in regressions:
            print(f"     ⚠️  {reg}")
        print("  → RiskAgent improves risk but may reduce return. Consider tuning.")
    else:
        print("  ❌ VERDICT: RiskAgent HURTS performance:")
        for reg in regressions:
            print(f"     ❌ {reg}")
        for imp in improvements:
            print(f"     ✅ {imp}")
        print("  → Debug RiskAgent parameters before proceeding.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Walk-Forward Stage 2: ForecastAgent + RiskAgent Backtest",
    )
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--train-years", type=int, default=5, help="Rolling training window in years (default: 5)")
    parser.add_argument("--horizon", type=int, default=5, help="Holding period in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Buy/sell threshold (default: 0.50)")
    # Layer 1 adaptive thresholds (per-month, per-ticker, calibrated-space)
    parser.add_argument(
        "--adaptive-thresholds",
        dest="adaptive_thresholds",
        action="store_true",
        default=True,
        help="Layer 1: per-month per-ticker adaptive buy/sell thresholds derived from "
             "OOF *calibrated* probabilities. Overrides --threshold each month. "
             "Enabled by default (pass --no-adaptive-thresholds to disable).",
    )
    parser.add_argument(
        "--no-adaptive-thresholds",
        dest="adaptive_thresholds",
        action="store_false",
        help="Disable Layer 1 adaptive thresholds and fall back to the static --threshold value.",
    )
    parser.add_argument("--buy-min-precision", type=float, default=0.55,
                        help="Min empirical up-rate for a buy-threshold bin (Layer 1). Default: 0.55")
    parser.add_argument("--sell-min-precision", type=float, default=0.55,
                        help="Min empirical down-rate for a sell-threshold bin (Layer 1). Default: 0.55")
    parser.add_argument("--adaptive-min-support", type=int, default=30,
                        help="Min samples required in a threshold bin (Layer 1). Default: 30")
    parser.add_argument("--adaptive-fallback-buy", type=float, default=0.55,
                        help="Fallback buy threshold when OOF too small (Layer 1). Default: 0.55")
    parser.add_argument("--adaptive-fallback-sell", type=float, default=0.45,
                        help="Fallback sell threshold when OOF too small (Layer 1). Default: 0.45")
    # Adaptive conformal prediction set (binds ambiguity to Layer-1 thresholds)
    parser.add_argument(
        "--adaptive-conformal",
        action="store_true",
        default=True,
        help="Replace training-time q90-based conformal set with one aligned to "
             "Layer-1 adaptive buy/sell thresholds. Enabled by default (pass "
             "--no-adaptive-conformal to disable).",
    )
    parser.add_argument(
        "--no-adaptive-conformal",
        dest="adaptive_conformal",
        action="store_false",
        help="Fallback to the legacy q90-based conformal prediction set.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for training (default: 5)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost bps (default: 5.0)")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps (default: 5.0)")
    parser.add_argument(
        "--calibrator",
        type=str,
        default="isotonic",
        choices=["temperature", "isotonic", "platt", "none"],
        help="Probability calibrator: 'isotonic' (default, sklearn IsotonicRegression — Stage 1 benchmark winner), "
             "'platt' (sigmoid via LogisticRegression on raw_prob — smooth monotone, avoids isotonic step plateaus), "
             "'temperature' (legacy 1-param T-scaling), 'none' (raw LightGBM prob)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Disable SHORT side entirely (sell_threshold forced to -1.0 so prob < sell is never true).",
    )
    parser.add_argument(
        "--full-position",
        action="store_true",
        help="Kelly-lock switch: when enabled, every trade that passes the "
             "EV / conformal / tree-dispersion / track-record guards is "
             "sized at MAX_POSITION_SIZE instead of Kelly*conviction. "
             "Useful for low/mid-volatility names (QQQ, AAPL) where "
             "fractional Kelly is the dominant Stage2 drag. High-vol names "
             "(NVDA, TSLA) usually prefer fractional Kelly.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    buy_threshold = args.threshold
    sell_threshold = 1.0 - args.threshold

    print(f"\n{'='*80}")
    print(f"  Walk-Forward Stage 2: ForecastAgent + RiskAgent Backtest")
    print(f"{'='*80}")
    print(f"  Ticker:          {ticker}")
    print(f"  Backtest period: {args.start} → {args.end}")
    print(f"  Training window: {args.train_years} years (rolling)")
    print(f"  Retrain freq:    monthly")
    print(f"  Horizon:         {args.horizon}d")
    if args.adaptive_thresholds:
        print(f"  Thresholds:      ADAPTIVE (Layer 1, calibrated-space, per-month)")
        print(f"                   min_precision buy={args.buy_min_precision} sell={args.sell_min_precision}, "
              f"min_support={args.adaptive_min_support}")
    else:
        print(f"  Thresholds:      buy > {buy_threshold}, sell < {sell_threshold} (static)")
    print(f"  Costs:           {args.cost_bps}bps + {args.slippage_bps}bps slippage")
    print(f"  Label:           absolute return")
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

    # Download SPY for benchmark (not for labels — we use absolute return)
    spy_data_dict = download_training_data(["SPY"])
    spy_data = spy_data_dict.get("SPY")
    spy_close = None
    if spy_data is not None and not spy_data.empty:
        print(f"  SPY data: {len(spy_data)} rows (benchmark only, not for labels)")
        spy_close = pd.to_numeric(spy_data["Close"], errors="coerce")
    else:
        print("  [warn] SPY data not available")

    ohlcv_data = raw_data[ticker]
    if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
        ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
    ohlcv_data = ohlcv_data.sort_index()
    print(f"  {ticker} data: {len(ohlcv_data)} rows")

    if spy_close is not None:
        spy_close = spy_close.reindex(ohlcv_data.index, method="ffill")

    # ── Step 3: Pre-fetch macro/fundamental data ─────────────────────────
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
                equity_col = macro_fund_df["_total_equity"]
                shares = macro_fund_df["_shares_outstanding"]
                valid = equity_col.notna() & shares.notna() & (shares > 0) & close_for_mf.notna()
                bvps = equity_col[valid] / shares[valid]
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

            if "pe_ratio" in macro_fund_df.columns and "earnings_growth_yoy" in macro_fund_df.columns:
                pe = macro_fund_df["pe_ratio"]
                eg = macro_fund_df["earnings_growth_yoy"]
                eg_pct = eg * 100
                valid = pe.notna() & eg_pct.notna() & (eg_pct.abs() > 1.0) & (pe > 0)
                macro_fund_df.loc[valid, "peg_ratio"] = pe[valid] / eg_pct[valid]

            for col in _INTERMEDIATE_COLS:
                if col in macro_fund_df.columns:
                    macro_fund_df.drop(columns=[col], inplace=True)

            print(f"  [macro/fund] Loaded {len(macro_fund_df)} days of macro/fundamental data")
        else:
            macro_fund_df = None
    except Exception as exc:
        print(f"  [warn] macro/fund fetch failed: {exc}")
        macro_fund_df = None

    print()

    # ── Step 4: Walk-forward loop ────────────────────────────────────────
    s1_all_trades: List[dict] = []
    s2_all_trades: List[dict] = []
    monthly_perf: List[dict] = []
    s1_equity = 1.0
    s2_equity = 1.0
    s2_total_risk_stats: Dict[str, int] = {}
    total_start_time = time.time()

    for month_idx, (month_start, month_end) in enumerate(months):
        train_start = month_start - pd.DateOffset(years=args.train_years)
        train_end = month_start - pd.Timedelta(days=1)

        print(f"{'='*80}")
        print(f"  Month {month_idx + 1}/{len(months)}: {month_start.strftime('%Y-%m-%d')} → {month_end.strftime('%Y-%m-%d')}")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
        print(f"{'='*80}")

        month_time = time.time()

        # 4a: Train model (absolute return label)
        print(f"  [Train] Training model on {args.train_years}-year window (absolute return label) ...")
        lgb_model, calibrator, meta, mf_cols_used, oof_preds, y_train, future_returns_train = train_model_for_window(
            ticker=ticker,
            raw_data=raw_data,
            train_start=train_start,
            train_end=train_end,
            horizon_days=args.horizon,
            n_splits=args.cv_folds,
            verbose=args.verbose,
            calibrator_type=args.calibrator,
        )

        if lgb_model is None:
            print(f"  [SKIP] Training failed for this window")
            monthly_perf.append({
                "month": month_start.strftime("%Y-%m"),
                "s1_trades": 0, "s2_trades": 0,
                "s1_return": 0.0, "s2_return": 0.0,
                "s2_hit_rate": 0.0, "cv_auc": 0.0,
            })
            continue

        cv_auc = meta.get("cv_metrics", {}).get("mean_auc", 0.0)
        n_train = meta.get("training_samples", 0)
        conformal_q90 = meta.get("conformal_scores_quantiles", {}).get("q90", "N/A")
        print(f"  [Train] CV AUC: {cv_auc:.4f}, samples: {n_train}, conformal q90: {conformal_q90}")

        # 4a+: Layer 1 — derive per-month adaptive thresholds on calibrated probs
        month_buy_th = buy_threshold
        month_sell_th = sell_threshold
        month_adaptive_info: Optional[Dict[str, Any]] = None
        if args.adaptive_thresholds and oof_preds is not None and y_train is not None:
            month_adaptive_info = derive_adaptive_thresholds(
                oof_raw=oof_preds,
                y=y_train,
                future_returns=future_returns_train,
                calibrator=calibrator,
                buy_min_precision=args.buy_min_precision,
                sell_min_precision=args.sell_min_precision,
                min_support=args.adaptive_min_support,
                fallback_buy=args.adaptive_fallback_buy,
                fallback_sell=args.adaptive_fallback_sell,
                verbose=True,
            )
            month_buy_th = month_adaptive_info["buy_threshold"]
            month_sell_th = month_adaptive_info["sell_threshold"]

        # Resolve disabled flags from adaptive info (default = enabled)
        month_buy_disabled = bool(month_adaptive_info["buy_disabled"]) if month_adaptive_info else False
        month_short_disabled = bool(month_adaptive_info["short_disabled"]) if month_adaptive_info else False

        # Long-only override: kill SHORT side entirely
        if getattr(args, "long_only", False):
            month_sell_th = -1.0
            month_short_disabled = True

        # 4b: Run Stage 1 backtest (no RiskAgent, for comparison)
        #     S1 now uses the same calibrator + disable flags as S2 so that
        #     the S1 vs S2 comparison isolates ONLY the RiskAgent effect.
        print(f"  [S1] Running forecast-only backtest ...")
        s1_trades, s1_equity, s1_eq_curve = run_month_backtest_stage1(
            lgb_model=lgb_model,
            meta=meta,
            ohlcv_data=ohlcv_data,
            spy_close=spy_close,
            macro_fund_df=macro_fund_df,
            month_start=month_start,
            month_end=month_end,
            equity_start=s1_equity,
            horizon=args.horizon,
            buy_threshold=month_buy_th,
            sell_threshold=month_sell_th,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            verbose=args.verbose,
            calibrator=calibrator,
            buy_disabled=month_buy_disabled,
            short_disabled=month_short_disabled,
        )

        # 4c: Run Stage 2 backtest (with RiskAgent)
        print(f"  [S2] Running forecast + risk backtest ...")
        s2_trades, s2_equity, s2_eq_curve, month_risk_stats = run_month_backtest_stage2(
            lgb_model=lgb_model,
            calibrator=calibrator,
            meta=meta,
            ohlcv_data=ohlcv_data,
            spy_close=spy_close,
            macro_fund_df=macro_fund_df,
            month_start=month_start,
            month_end=month_end,
            equity_start=s2_equity,
            horizon=args.horizon,
            buy_threshold=month_buy_th,
            sell_threshold=month_sell_th,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            verbose=args.verbose,
            buy_disabled=month_buy_disabled,
            short_disabled=month_short_disabled,
            use_adaptive_conformal=args.adaptive_conformal,
            full_position=getattr(args, "full_position", False),
        )

        # Accumulate risk stats
        for k, v in month_risk_stats.items():
            if isinstance(v, (int, float)):
                s2_total_risk_stats[k] = s2_total_risk_stats.get(k, 0) + v
            else:
                # Non-numeric fields (e.g. conformal_mode) — record once
                s2_total_risk_stats.setdefault(k, v)

        elapsed = time.time() - month_time
        month_label = month_start.strftime("%Y-%m")

        # Compute month returns
        s1_month_ret = 0.0
        if s1_trades:
            r = 1.0
            for t in s1_trades:
                r *= (1.0 + t["net_return"])
            s1_month_ret = r - 1.0

        s2_month_ret = 0.0
        s2_hit = 0.0
        if s2_trades:
            r = 1.0
            for t in s2_trades:
                r *= (1.0 + t["net_return"])
            s2_month_ret = r - 1.0
            s2_hit = sum(1 for t in s2_trades if t["net_return"] > 0) / len(s2_trades)

        print(f"  [Result] {month_label}: S1={len(s1_trades)} trades ({s1_month_ret:+.4%}), "
              f"S2={len(s2_trades)} trades ({s2_month_ret:+.4%}), "
              f"S1 eq={s1_equity:.4f}, S2 eq={s2_equity:.4f}, Time={elapsed:.1f}s")

        s1_all_trades.extend(s1_trades)
        s2_all_trades.extend(s2_trades)
        month_perf_record = {
            "month": month_label,
            "s1_trades": len(s1_trades),
            "s2_trades": len(s2_trades),
            "s1_return": round(s1_month_ret, 6),
            "s2_return": round(s2_month_ret, 6),
            "s2_hit_rate": round(s2_hit, 4),
            "cv_auc": round(cv_auc, 4),
        }
        if month_adaptive_info is not None:
            month_perf_record["adaptive_buy_threshold"] = month_adaptive_info["buy_threshold"]
            month_perf_record["adaptive_sell_threshold"] = month_adaptive_info["sell_threshold"]
            month_perf_record["adaptive_buy_disabled"] = month_adaptive_info["buy_disabled"]
            month_perf_record["adaptive_short_disabled"] = month_adaptive_info["short_disabled"]
            month_perf_record["adaptive_mode"] = month_adaptive_info.get("mode", "adaptive")
        monthly_perf.append(month_perf_record)

    # ── Step 5: Compute benchmark return ─────────────────────────────────
    total_elapsed = time.time() - total_start_time
    benchmark_return = 0.0
    try:
        if spy_data is not None:
            spy_slice = spy_data.loc[str(start_date):str(end_date)]
            if not spy_slice.empty:
                spy_c = pd.to_numeric(spy_slice["Close"], errors="coerce")
                benchmark_return = float(spy_c.iloc[-1] / spy_c.iloc[0] - 1.0)
    except Exception:
        pass

    # ── Step 6: Analyze and report ───────────────────────────────────────
    if not s2_all_trades:
        print("\n[ERROR] No Stage 2 trades executed across all months.")
        sys.exit(1)

    print(f"\n[Step 5] Analyzing S1={len(s1_all_trades)}, S2={len(s2_all_trades)} trades ...")
    print(f"  Total time: {total_elapsed:.1f}s")

    s1_metrics = analyze_backtest(s1_all_trades, benchmark_return)
    s2_metrics = analyze_backtest(s2_all_trades, benchmark_return)

    config = {
        "ticker": ticker,
        "start_date": args.start,
        "end_date": args.end,
        "train_years": args.train_years,
        "horizon": args.horizon,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "adaptive_thresholds": args.adaptive_thresholds,
        "buy_min_precision": args.buy_min_precision,
        "sell_min_precision": args.sell_min_precision,
        "adaptive_min_support": args.adaptive_min_support,
        "adaptive_fallback_buy": args.adaptive_fallback_buy,
        "adaptive_fallback_sell": args.adaptive_fallback_sell,
    }

    print_comparison_report(s1_metrics, s2_metrics, s2_total_risk_stats, monthly_perf, config)

    # ── Step 7: Save results ─────────────────────────────────────────────
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"wf_stage2_{ticker}_{args.start}_{args.end}_t{args.threshold}_cal-{args.calibrator}"
    if args.adaptive_thresholds:
        base_name += "_adaptive"
    if getattr(args, "long_only", False):
        base_name += "_longonly"
    if getattr(args, "full_position", False):
        base_name += "_fullpos"

    # Save report JSON
    report_path = output_dir / f"{base_name}_report.json"
    save_data = {
        "config": config,
        "stage1_metrics": s1_metrics,
        "stage2_metrics": s2_metrics,
        "risk_stats": s2_total_risk_stats,
        "monthly_performance": monthly_perf,
        "walk_forward": {
            "train_years": args.train_years,
            "retrain_frequency": "monthly",
            "label_type": "absolute_return",
            "n_months": len(months),
        },
    }
    with open(report_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[✓] Report saved: {report_path}")

    # Save Stage 2 trade log CSV
    trades_path = output_dir / f"{base_name}_trades.csv"
    if s2_all_trades:
        trades_df = pd.DataFrame(s2_all_trades)
        if "risk_flags" in trades_df.columns:
            trades_df["risk_flags"] = trades_df["risk_flags"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else str(x)
            )
        if "prediction_set" in trades_df.columns:
            trades_df["prediction_set"] = trades_df["prediction_set"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else str(x)
            )
        trades_df.to_csv(trades_path, index=False)
        print(f"[✓] Stage 2 trade log saved: {trades_path}")

    # Save Stage 1 trade log CSV (for comparison)
    s1_trades_path = output_dir / f"{base_name}_s1_trades.csv"
    if s1_all_trades:
        pd.DataFrame(s1_all_trades).to_csv(s1_trades_path, index=False)
        print(f"[✓] Stage 1 trade log saved: {s1_trades_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
