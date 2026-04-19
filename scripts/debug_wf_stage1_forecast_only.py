#!/usr/bin/env python3
"""
Walk-Forward Stage 1: ForecastAgent Only Backtest with Monthly Retraining

Goal: Validate ForecastAgent signal quality in a truly out-of-sample backtest,
      with monthly model retraining on a rolling window.

Key differences from original Stage 1:
  - Model is retrained every month on a rolling 5-year window
  - Each month's trades use ONLY the model trained on data BEFORE that month
  - Eliminates look-ahead bias completely

Rules (same as original Stage 1):
  - Only FeatureEngineeringAgent + ForecastAgent logic (standalone)
  - RegimeAgent is bypassed (regime features computed from base features)
  - RiskAgent is bypassed (no filtering, no Kelly sizing)
  - Fixed position size: 100% of capital per trade
  - No stop-loss / take-profit — pure horizon exit
  - Execution: Signal at Close[t] → Enter at Open[t+1] → Exit at Close[t+horizon]
  - Transaction cost & slippage still applied for realism

Flow for each month:
  1. Train LightGBM on [T-5y, T)
  2. Walk forward day-by-day through [T, T+1m), score and trade
  3. Record trades, equity changes
  4. Slide window forward by 1 month, repeat

Usage:
    python scripts/debug_wf_stage1_forecast_only.py --ticker AAPL --start 2025-01-01 --end 2025-12-31
    python scripts/debug_wf_stage1_forecast_only.py --ticker AAPL --start 2025-01-01 --end 2025-12-31 --threshold 0.55 --verbose
"""

from __future__ import annotations

import argparse
import json
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


# ═══════════════════════════════════════════════════════════════════════════
# Feature computation (standalone, no Agent dependency)
# ═══════════════════════════════════════════════════════════════════════════

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
            return np.nan  # Let LightGBM handle missing values natively
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
# Model Training (reused from walk-forward stage 0)
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
        "label_type": "absolute_return" if spy_data is None else "excess_return",
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
# Model Scoring (standalone, no Agent dependency)
# ═══════════════════════════════════════════════════════════════════════════

def score_features(
    features: dict,
    model,
    meta: dict,
    regime_features: dict | None = None,
    macro_fund_row: dict | None = None,
) -> float:
    """Score features with LightGBM model and return raw probability.

    Uses raw probability directly (no calibrator) to avoid over-smoothing
    and cross-model probability drift issues in walk-forward setting.
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

    # Regime features: use real values if provided, else neutral defaults
    for col in regime_cols:
        row[col] = regime_features.get(col, 0.0) if regime_features else 0.0

    # Macro/fund features: use real values if provided, else NaN
    for col in macro_fund_cols:
        if macro_fund_row and col in macro_fund_row and pd.notna(macro_fund_row[col]):
            row[col] = float(macro_fund_row[col])
        else:
            row[col] = np.nan

    for col in rank_cols:
        row[col] = 0.5  # median rank (single-ticker)
    for col in cat_cols:
        row[col] = 0.0

    X = pd.DataFrame([row])[all_cols]
    raw_prob = float(model.predict(X)[0])

    return raw_prob


# ═══════════════════════════════════════════════════════════════════════════
# Monthly Backtest Engine (simplified — no RegimeAgent, no RiskAgent)
# ═══════════════════════════════════════════════════════════════════════════

def run_month_backtest(
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
) -> Tuple[List[dict], float, dict]:
    """Run forecast-only backtest for a single month.

    Trading rules:
      - prob > buy_threshold  → BUY  (long 100%)
      - prob < sell_threshold → SELL (short 100%)
      - otherwise             → HOLD (flat, no trade)
      - No stop-loss / take-profit — exit at horizon
      - Entry at Open[t+1], Exit at Close[t+horizon]

    Returns:
        (trade_log, final_equity, equity_curve_dict)
    """
    macro_fund_cols = meta.get("macro_fundamental_features", [])
    dates = ohlcv_data.index
    open_prices = pd.to_numeric(ohlcv_data["Open"], errors="coerce").values
    close_prices = pd.to_numeric(ohlcv_data["Close"], errors="coerce").values

    cost_per_trade = (cost_bps + slippage_bps) / 10000.0 * 2  # round-trip

    # Find date range indices
    start_idx = dates.searchsorted(month_start)
    start_idx = max(start_idx, 60)  # Need at least 60 bars for features
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

        # Compute features (no look-ahead)
        data_slice = ohlcv_data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            t += horizon
            continue

        # Compute real regime features
        current_close = float(close_prices[t])
        regime_feats = compute_regime_from_features(features, current_close)

        # Look up macro/fund features by date (backward fill)
        mf_row = None
        if macro_fund_df is not None and not macro_fund_df.empty:
            valid_dates = macro_fund_df.index[macro_fund_df.index <= current_date]
            if len(valid_dates) > 0:
                nearest_date = valid_dates[-1]
                mf_row = macro_fund_df.loc[nearest_date].to_dict()

        # Score
        prob = score_features(
            features, lgb_model, meta,
            regime_features=regime_feats,
            macro_fund_row=mf_row,
        )

        # Decision: simple threshold
        if prob > buy_threshold:
            action = "buy"
            direction = 1.0
        elif prob < sell_threshold:
            action = "sell"
            direction = -1.0
        else:
            action = "hold"
            direction = 0.0

        # Execute trade
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

            # Compute SPY return for the same period (for excess return tracking)
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

        t += horizon  # Step forward by horizon (non-overlapping trades)

    return trade_log, equity, equity_curve


# ═══════════════════════════════════════════════════════════════════════════
# Analysis & Reporting
# ═══════════════════════════════════════════════════════════════════════════

def analyze_backtest(all_trades: List[dict], benchmark_return: float) -> dict:
    """Compute comprehensive metrics from aggregated trade log."""
    if not all_trades:
        return {"error": "No trades executed"}

    returns = [t["net_return"] for t in all_trades]
    raw_returns = [t["raw_return"] for t in all_trades]
    buy_trades = [t for t in all_trades if t["action"] == "buy"]
    sell_trades = [t for t in all_trades if t["action"] == "sell"]

    # Final equity
    equity = 1.0
    for r in returns:
        equity *= (1.0 + r)
    total_return = equity - 1.0

    n_trades = len(all_trades)
    n_buy = len(buy_trades)
    n_sell = len(sell_trades)

    # Win/loss
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    hit_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    # Avg win/loss
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean([abs(r) for r in losses])) if losses else 0.0

    # Sharpe & Sortino
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=0)) if len(returns) > 1 else 1.0
    horizon = 5  # default
    sharpe = (mean_ret / std_ret * np.sqrt(252.0 / horizon)) if std_ret > 0 else 0.0

    downside = [r for r in returns if r < 0]
    downside_std = float(np.std(downside, ddof=0)) if downside else 1.0
    sortino = (mean_ret / downside_std * np.sqrt(252.0 / horizon)) if downside_std > 0 else 0.0

    # Max drawdown
    equity_vals = [1.0]
    for r in returns:
        equity_vals.append(equity_vals[-1] * (1 + r))
    equity_arr = np.array(equity_vals)
    cummax = np.maximum.accumulate(equity_arr)
    drawdowns = equity_arr / cummax - 1.0
    max_dd = float(np.min(drawdowns))

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss

    # Alpha
    alpha = total_return - benchmark_return

    # Buy vs Sell breakdown
    buy_returns = [t["net_return"] for t in buy_trades]
    sell_returns = [t["net_return"] for t in sell_trades]
    buy_hit = sum(1 for r in buy_returns if r > 0) / len(buy_returns) if buy_returns else 0.0
    sell_hit = sum(1 for r in sell_returns if r > 0) / len(sell_returns) if sell_returns else 0.0
    buy_avg = float(np.mean(buy_returns)) if buy_returns else 0.0
    sell_avg = float(np.mean(sell_returns)) if sell_returns else 0.0

    # Trade IC (prob vs actual return direction)
    probs = [t["probability_up"] for t in all_trades]
    actuals = [t["raw_return"] * t["direction"] for t in all_trades]
    if len(probs) > 5:
        trade_ic, trade_ic_pval = stats.spearmanr(probs, actuals)
    else:
        trade_ic, trade_ic_pval = 0.0, 1.0

    # Excess return analysis
    excess_returns = [t.get("excess_return", 0.0) for t in all_trades]
    avg_excess = float(np.mean(excess_returns)) if excess_returns else 0.0
    pct_outperform = sum(1 for e in excess_returns if e > 0) / len(excess_returns) if excess_returns else 0.0

    # Consecutive wins/losses
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


def analyze_monthly_performance(monthly_results: List[dict]) -> List[dict]:
    """Compute per-month performance breakdown."""
    summaries = []
    for m in monthly_results:
        trades = m.get("trades", [])
        month_label = m["month"]
        cv_auc = m.get("cv_auc", 0.0)
        n_train = m.get("n_train", 0)

        if not trades:
            summaries.append({
                "month": month_label,
                "n_trades": 0,
                "return": 0.0,
                "hit_rate": 0.0,
                "avg_prob": 0.0,
                "cv_auc": cv_auc,
                "n_train": n_train,
            })
            continue

        returns = [t["net_return"] for t in trades]
        month_return = 1.0
        for r in returns:
            month_return *= (1.0 + r)
        month_return -= 1.0

        hit = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
        avg_prob = float(np.mean([t["probability_up"] for t in trades]))
        n_buy = sum(1 for t in trades if t["action"] == "buy")
        n_sell = sum(1 for t in trades if t["action"] == "sell")

        # Excess return
        excess_rets = [t.get("excess_return", 0.0) for t in trades]
        avg_excess = float(np.mean(excess_rets)) if excess_rets else 0.0

        summaries.append({
            "month": month_label,
            "n_trades": len(trades),
            "n_buy": n_buy,
            "n_sell": n_sell,
            "return": round(float(month_return), 6),
            "hit_rate": round(hit, 4),
            "avg_prob": round(avg_prob, 6),
            "avg_excess_return": round(avg_excess, 6),
            "cv_auc": round(cv_auc, 4),
            "n_train": n_train,
        })

    return summaries


def print_report(metrics: dict, monthly_perf: List[dict], config: dict) -> None:
    """Pretty-print Walk-Forward Stage 1 backtest report."""
    ticker = config["ticker"]

    print(f"\n{'='*70}")
    print(f"  Walk-Forward Stage 1: ForecastAgent Only Backtest — {ticker}")
    print(f"  Mode: Monthly retraining (no look-ahead bias)")
    print(f"{'='*70}")

    print(f"\n📋 Configuration:")
    print(f"  Period:          {config['start_date']} → {config['end_date']}")
    print(f"  Training window: {config['train_years']} years (rolling)")
    print(f"  Retrain freq:    monthly")
    print(f"  Horizon:         {config['horizon']}d")
    print(f"  Thresholds:      buy > {config['buy_threshold']}, sell < {config['sell_threshold']}")
    print(f"  Costs:           {config['cost_bps']}bps + {config['slippage_bps']}bps slippage")

    print(f"\n── Performance ──")
    tr = metrics["total_return"]
    br = metrics["benchmark_return"]
    alpha = metrics["alpha"]
    print(f"  Strategy Return: {tr:+.2%}  {'✅' if tr > 0 else '❌'}")
    print(f"  Benchmark (SPY): {br:+.2%}")
    print(f"  Alpha:           {alpha:+.2%}  {'✅' if alpha > 0 else '❌'}")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:+.4f}  {'✅ > 1' if metrics['sharpe_ratio'] > 1 else '⚠️' if metrics['sharpe_ratio'] > 0 else '❌'}")
    print(f"  Sortino Ratio:   {metrics['sortino_ratio']:+.4f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:+.2%}")
    print(f"  Profit Factor:   {metrics['profit_factor']:.4f}  {'✅ > 1' if metrics['profit_factor'] > 1 else '❌'}")
    print(f"  Expectancy:      {metrics['expectancy']:+.6f}")

    print(f"\n── Trade Statistics ──")
    print(f"  Total Trades:    {metrics['n_trades']}")
    print(f"    Buy:           {metrics['n_buy']} ({metrics['n_buy']/max(metrics['n_trades'],1):.1%})")
    print(f"    Sell:          {metrics['n_sell']} ({metrics['n_sell']/max(metrics['n_trades'],1):.1%})")
    print(f"  Hit Rate:        {metrics['hit_rate']:.2%}  {'✅ > 50%' if metrics['hit_rate'] > 0.5 else '❌'}")
    print(f"  Avg Win:         {metrics['avg_win']:+.4%}")
    print(f"  Avg Loss:        {metrics['avg_loss']:+.4%}")
    print(f"  Avg Trade:       {metrics['avg_trade_return']:+.4%}")

    print(f"\n── Buy vs Sell Breakdown ──")
    print(f"  Buy  trades: hit_rate={metrics['buy_hit_rate']:.2%}, avg_return={metrics['buy_avg_return']:+.4%}")
    print(f"  Sell trades: hit_rate={metrics['sell_hit_rate']:.2%}, avg_return={metrics['sell_avg_return']:+.4%}")

    print(f"\n── Signal Quality ──")
    print(f"  Trade IC:            {metrics['trade_ic']:+.6f} (p={metrics['trade_ic_pval']:.4f})")
    print(f"  Avg Excess Return:   {metrics['avg_excess_return']:+.6f}")
    print(f"  % Outperform SPY:    {metrics['pct_outperform_spy']:.2%}")
    print(f"  Max Consec Wins:     {metrics['max_consecutive_wins']}")
    print(f"  Max Consec Losses:   {metrics['max_consecutive_losses']}")

    # Per-month breakdown (unique to walk-forward)
    print(f"\n── Per-Month Performance (Walk-Forward) ──")
    print(f"  {'Month':<10} {'Trades':>6} {'B/S':>7} {'Return':>10} {'HitRate':>8} {'AvgProb':>8} {'AvgExcess':>10} {'CV_AUC':>8} {'NTrain':>7}")
    print(f"  {'─'*10} {'─'*6} {'─'*7} {'─'*10} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*7}")
    for m in monthly_perf:
        n_t = m["n_trades"]
        bs = f"{m.get('n_buy', 0)}/{m.get('n_sell', 0)}"
        ret_str = f"{m['return']:+.4%}" if n_t > 0 else "    N/A"
        hit_str = f"{m['hit_rate']:.2%}" if n_t > 0 else "  N/A"
        prob_str = f"{m['avg_prob']:.4f}" if n_t > 0 else "  N/A"
        excess_str = f"{m.get('avg_excess_return', 0):+.6f}" if n_t > 0 else "      N/A"
        auc_str = f"{m['cv_auc']:.4f}" if m['cv_auc'] > 0 else "  N/A"
        print(f"  {m['month']:<10} {n_t:>6} {bs:>7} {ret_str:>10} {hit_str:>8} {prob_str:>8} {excess_str:>10} {auc_str:>8} {m['n_train']:>7}")

    # Monthly return summary
    monthly_returns = [m["return"] for m in monthly_perf if m["n_trades"] > 0]
    if monthly_returns:
        positive_months = sum(1 for r in monthly_returns if r > 0)
        print(f"\n  Monthly Summary:")
        print(f"    Mean monthly return:  {np.mean(monthly_returns):+.4%}")
        print(f"    Std monthly return:   {np.std(monthly_returns):.4%}")
        print(f"    Positive months:      {positive_months}/{len(monthly_returns)} ({positive_months/len(monthly_returns):.0%})")
        print(f"    Best month:           {max(monthly_returns):+.4%}")
        print(f"    Worst month:          {min(monthly_returns):+.4%}")

    # Verdict
    print(f"\n{'─'*70}")
    issues = []
    if tr <= 0:
        issues.append("Negative total return")
    if alpha < -0.05:
        issues.append(f"Large negative alpha ({alpha:+.2%}) vs benchmark")
    if metrics["hit_rate"] < 0.45:
        issues.append(f"Low hit rate ({metrics['hit_rate']:.2%})")
    if metrics["sharpe_ratio"] < 0:
        issues.append(f"Negative Sharpe ({metrics['sharpe_ratio']:.4f})")
    if metrics["max_drawdown"] < -0.20:
        issues.append(f"Large max drawdown ({metrics['max_drawdown']:.2%})")
    if metrics["profit_factor"] < 1.0:
        issues.append(f"Profit factor < 1 ({metrics['profit_factor']:.4f})")
    if monthly_returns and positive_months / len(monthly_returns) < 0.4:
        issues.append(f"Profitable in < 40% of months")

    if not issues:
        print("  ✅ VERDICT: ForecastAgent signals are profitable (out-of-sample). Proceed to WF Stage 2.")
    elif len(issues) <= 2 and tr > 0:
        print("  ⚠️  VERDICT: ForecastAgent signals are marginally profitable:")
        for issue in issues:
            print(f"     • {issue}")
        print("  → Consider tuning thresholds before WF Stage 2.")
    else:
        print("  ❌ VERDICT: ForecastAgent signals have issues (out-of-sample):")
        for issue in issues:
            print(f"     • {issue}")
        print("  → Debug signal quality before proceeding.")


def print_trade_summary(trades: list, n: int = 10) -> None:
    """Print first/last N trades for inspection."""
    if not trades:
        return

    print(f"\n── Sample Trades (first {min(n, len(trades))}) ──")
    print(f"  {'Date':12s} | {'Action':6s} | {'Prob':>7s} | {'Entry':>9s} | {'Exit':>9s} | {'Return':>9s} | {'Excess':>9s} | {'Equity':>8s}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*8}")
    for t in trades[:n]:
        print(
            f"  {t['date']:12s} | {t['action']:6s} | {t['probability_up']:7.4f} | "
            f"{t['entry_price']:9.2f} | {t['exit_price']:9.2f} | "
            f"{t['net_return']:+9.4f} | {t.get('excess_return', 0):+9.4f} | {t['equity']:8.4f}"
        )

    if len(trades) > n:
        print(f"  ... ({len(trades) - n} more trades) ...")
        print(f"\n── Last {min(n, len(trades))} Trades ──")
        print(f"  {'Date':12s} | {'Action':6s} | {'Prob':>7s} | {'Entry':>9s} | {'Exit':>9s} | {'Return':>9s} | {'Excess':>9s} | {'Equity':>8s}")
        print(f"  {'-'*12} | {'-'*6} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*8}")
        for t in trades[-n:]:
            print(
                f"  {t['date']:12s} | {t['action']:6s} | {t['probability_up']:7.4f} | "
                f"{t['entry_price']:9.2f} | {t['exit_price']:9.2f} | "
                f"{t['net_return']:+9.4f} | {t.get('excess_return', 0):+9.4f} | {t['equity']:8.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Walk-Forward Stage 1: ForecastAgent Only Backtest with Monthly Retraining",
    )
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--train-years", type=int, default=5, help="Rolling training window in years (default: 5)")
    parser.add_argument("--horizon", type=int, default=5, help="Holding period in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Buy/sell threshold (default: 0.50)")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for training (default: 5)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost bps (default: 5.0)")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps (default: 5.0)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)
    buy_threshold = args.threshold
    sell_threshold = 1.0 - args.threshold  # symmetric

    print(f"\n{'='*70}")
    print(f"  Walk-Forward Stage 1: ForecastAgent Only Backtest")
    print(f"{'='*70}")
    print(f"  Ticker:          {ticker}")
    print(f"  Backtest period: {args.start} → {args.end}")
    print(f"  Training window: {args.train_years} years (rolling)")
    print(f"  Retrain freq:    monthly")
    print(f"  Horizon:         {args.horizon}d")
    print(f"  Thresholds:      buy > {buy_threshold}, sell < {sell_threshold}")
    print(f"  Costs:           {args.cost_bps}bps + {args.slippage_bps}bps slippage")
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

    # Download SPY for excess return labels + benchmark
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
    all_trades: List[dict] = []
    monthly_results: List[dict] = []
    equity = 1.0
    all_equity_curve: dict = {}
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
        # Use absolute return labels (spy_data=None) instead of excess return
        # to avoid label-semantic mismatch with long/short trading decisions
        print(f"  [Train] Training model on {args.train_years}-year window (absolute return label) ...")
        lgb_model, meta, mf_cols_used = train_model_for_window(
            ticker=ticker,
            raw_data=raw_data,
            spy_data=None,
            train_start=train_start,
            train_end=train_end,
            horizon_days=args.horizon,
            n_splits=args.cv_folds,
            verbose=args.verbose,
        )

        if lgb_model is None:
            print(f"  [SKIP] Training failed for this window")
            monthly_results.append({
                "month": month_start.strftime("%Y-%m"),
                "trades": [],
                "cv_auc": 0.0,
                "n_train": 0,
            })
            continue

        cv_auc = meta.get("cv_metrics", {}).get("mean_auc", 0.0)
        n_train = meta.get("training_samples", 0)
        n_mf = len(mf_cols_used)
        print(f"  [Train] CV AUC: {cv_auc:.4f}, samples: {n_train}, macro/fund features: {n_mf}")

        # 4b: Run backtest for this month (out-of-sample)
        print(f"  [Trade] Running forecast-only backtest ...")
        month_trades, equity, month_equity_curve = run_month_backtest(
            lgb_model=lgb_model,
            meta=meta,
            ohlcv_data=ohlcv_data,
            spy_close=spy_close,
            macro_fund_df=macro_fund_df,
            month_start=month_start,
            month_end=month_end,
            equity_start=equity,
            horizon=args.horizon,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            verbose=args.verbose,
        )

        elapsed = time.time() - month_time
        month_label = month_start.strftime("%Y-%m")

        # Compute month return
        if month_trades:
            month_ret = 1.0
            for t in month_trades:
                month_ret *= (1.0 + t["net_return"])
            month_ret -= 1.0
            n_buy = sum(1 for t in month_trades if t["action"] == "buy")
            n_sell = sum(1 for t in month_trades if t["action"] == "sell")
            hit = sum(1 for t in month_trades if t["net_return"] > 0) / len(month_trades)
            print(f"  [Result] {month_label}: {len(month_trades)} trades (B:{n_buy}/S:{n_sell}), "
                  f"return={month_ret:+.4%}, hit={hit:.2%}, equity={equity:.4f}, Time={elapsed:.1f}s")
        else:
            print(f"  [Result] {month_label}: No trades, equity={equity:.4f}, Time={elapsed:.1f}s")

        all_trades.extend(month_trades)
        all_equity_curve.update(month_equity_curve)
        monthly_results.append({
            "month": month_label,
            "trades": month_trades,
            "cv_auc": cv_auc,
            "n_train": n_train,
        })

    # ── Step 5: Compute benchmark return ─────────────────────────────────
    total_elapsed = time.time() - total_start_time
    benchmark_return = 0.0
    try:
        spy_slice = spy_data.loc[str(start_date):str(end_date)] if spy_data is not None else None
        if spy_slice is not None and not spy_slice.empty:
            spy_c = pd.to_numeric(spy_slice["Close"], errors="coerce")
            benchmark_return = float(spy_c.iloc[-1] / spy_c.iloc[0] - 1.0)
    except Exception:
        pass

    # ── Step 6: Analyze and report ───────────────────────────────────────
    if not all_trades:
        print("\n[ERROR] No trades executed across all months.")
        sys.exit(1)

    print(f"\n[Step 5] Analyzing {len(all_trades)} total trades across {len(months)} months ...")
    print(f"  Total time: {total_elapsed:.1f}s")

    metrics = analyze_backtest(all_trades, benchmark_return)
    monthly_perf = analyze_monthly_performance(monthly_results)

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
    }

    print_report(metrics, monthly_perf, config)
    print_trade_summary(all_trades)

    # ── Step 7: Save results ─────────────────────────────────────────────
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"wf_stage1_{ticker}_{args.start}_{args.end}_t{args.threshold}"

    # Save report JSON
    report_path = output_dir / f"{base_name}_report.json"
    save_data = {
        "config": config,
        "metrics": metrics,
        "monthly_performance": monthly_perf,
        "walk_forward": {
            "train_years": args.train_years,
            "retrain_frequency": "monthly",
            "n_months": len(months),
            "n_months_with_trades": sum(1 for m in monthly_results if m["trades"]),
        },
    }
    with open(report_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[✓] Report saved: {report_path}")

    # Save trade log CSV
    trades_path = output_dir / f"{base_name}_trades.csv"
    if all_trades:
        pd.DataFrame(all_trades).to_csv(trades_path, index=False)
        print(f"[✓] Trade log saved: {trades_path}")

    # Save equity curve CSV
    equity_path = output_dir / f"{base_name}_equity.csv"
    if all_equity_curve:
        eq_df = pd.DataFrame(
            list(all_equity_curve.items()),
            columns=["date", "equity"],
        )
        eq_df.to_csv(equity_path, index=False)
        print(f"[✓] Equity curve saved: {equity_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
