#!/usr/bin/env python3
"""
Stage 1: ForecastAgent Only Backtest

Goal: Validate ForecastAgent signal quality in a real backtest framework,
      WITHOUT RegimeAgent or RiskAgent interference.

Rules:
  - Only FeatureEngineeringAgent + ForecastAgent are used
  - RegimeAgent is bypassed (neutral regime defaults)
  - RiskAgent is bypassed (no filtering, no Kelly sizing)
  - Fixed position size: 100% of capital per trade
  - No stop-loss / take-profit — pure horizon exit
  - Execution: Signal at Close[t] → Enter at Open[t+1] → Exit at Close[t+horizon]
  - Transaction cost & slippage still applied for realism

Usage:
    python scripts/debug_stage1_forecast_only.py --ticker AAPL --start 2023-01-01 --end 2025-12-31
    python scripts/debug_stage1_forecast_only.py --ticker AAPL --start 2023-01-01 --end 2025-12-31 --verbose
    python scripts/debug_stage1_forecast_only.py --ticker AAPL --start 2023-01-01 --end 2025-12-31 --threshold 0.55
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


# ── Regime computation helpers (mirroring train pipeline) ────────────────

def _trend_score_from_features(features: dict) -> float:
    """Compute trend score from feature dict (mirrors training pipeline)."""
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
# 1. Feature computation (same as Stage 0 — standalone, no Agent dependency)
# ═══════════════════════════════════════════════════════════════════════════

def compute_features(data: pd.DataFrame) -> dict:
    """Compute all base + interaction features from OHLCV data.

    Replicates the training pipeline exactly, including the 5 interaction
    features that were previously missing in FeatureEngineeringAgent.
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

    # Interaction features (must match training pipeline)
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


# ═══════════════════════════════════════════════════════════════════════════
# 2. Model scoring (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    """Load LightGBM model, metadata, and calibrator."""
    import lightgbm as lgb
    import pickle

    lgb_path = os.getenv("FORECAST_LGB_MODEL_PATH", "data/forecast_model.lgb")
    meta_path = os.getenv("FORECAST_LGB_META_PATH", "data/forecast_model_meta.json")

    model = lgb.Booster(model_file=lgb_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    calibrator = None
    cal_path = meta.get("calibrator_path", "data/forecast_calibrator.pkl")
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

    print(f"  Model: {lgb_path}")
    print(f"  Meta:  {meta_path}")
    print(f"  Calibrator: {cal_path} ({'loaded' if calibrator else 'not found'})")

    return model, meta, calibrator


def score_features(
    features: dict,
    model,
    meta: dict,
    calibrator,
    regime_features: dict | None = None,
    macro_fund_row: dict | None = None,
    return_uncertainty: bool = False,
) -> tuple:
    """Score features with LightGBM model and return (raw_prob, calibrated_prob).

    If return_uncertainty=True, returns (raw_prob, prob, uncertainty_info) where
    uncertainty_info is a dict with keys: uncertainty, prediction_set, is_uncertain.

    Regime and macro/fundamental features are now passed in from the caller
    (computed from real data), with fallback to neutral defaults.
    """
    feature_cols = meta.get("feature_columns", [])
    regime_cols = meta.get("regime_features", [])
    macro_fund_cols = meta.get("macro_fundamental_features", [])
    rank_cols = meta.get("rank_feature_columns", [])
    cat_cols = meta.get("categorical_features", [])
    all_cols = feature_cols + regime_cols + macro_fund_cols + rank_cols + cat_cols

    row = {}
    for col in feature_cols:
        row[col] = features.get(col, 0.0)
    # Regime features: use real values if provided, else neutral defaults
    for col in regime_cols:
        row[col] = regime_features.get(col, 0.0) if regime_features else 0.0
    # Macro/fund features: use real values if provided, else NaN (LightGBM handles natively)
    for col in macro_fund_cols:
        if macro_fund_row and col in macro_fund_row and pd.notna(macro_fund_row[col]):
            row[col] = float(macro_fund_row[col])
        else:
            row[col] = np.nan
    for col in rank_cols:
        row[col] = 0.5  # median rank
    for col in cat_cols:
        row[col] = 0.0

    X = pd.DataFrame([row])[all_cols]
    raw_prob = float(model.predict(X)[0])

    # Apply calibration with soft-clamp
    if calibrator is not None:
        try:
            cal_prob = float(calibrator.predict([raw_prob])[0])
        except Exception:
            cal_prob = raw_prob
        # Soft-clamp: 70% calibrated + 30% raw to prevent over-polarization
        prob = 0.7 * cal_prob + 0.3 * raw_prob
        prob = max(0.02, min(0.98, prob))
    else:
        prob = raw_prob

    if not return_uncertainty:
        return raw_prob, prob

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

    # Method 2: Conformal prediction set (uses raw_prob to avoid soft-clamp bias)
    # The soft-clamp calibrator pulls extreme probs toward center, which makes
    # conformal filtering overlap with proxy uncertainty. Using raw_prob gives
    # a more independent uncertainty signal.
    quantiles = meta.get("conformal_scores_quantiles", {})
    threshold = quantiles.get("q90")
    if threshold is not None:
        prediction_set = []
        # "up" in set ⟺ 1 - raw_prob ≤ threshold ⟺ raw_prob ≥ 1 - threshold
        if (1.0 - raw_prob) <= threshold:
            prediction_set.append("up")
        # "down" in set ⟺ raw_prob ≤ threshold
        if raw_prob <= threshold:
            prediction_set.append("down")
        uncertainty_info["prediction_set"] = prediction_set
        # Both labels → ambiguous → uncertain
        if len(prediction_set) == 2:
            uncertainty_info["is_uncertain"] = True
        # Neither label → very rare → also uncertain
        elif len(prediction_set) == 0:
            uncertainty_info["is_uncertain"] = True

    return raw_prob, prob, uncertainty_info


# ═══════════════════════════════════════════════════════════════════════════
# 3. Backtest engine (simplified — no RegimeAgent, no RiskAgent)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage1_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    model,
    meta: dict,
    calibrator,
    horizon: int = 5,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.50,
    verbose: bool = False,
) -> dict:
    """Run ForecastAgent-only walk-forward backtest.

    Trading rules:
      - prob > buy_threshold  → BUY  (long 100%)
      - prob < sell_threshold → SELL (short 100%)
      - otherwise             → HOLD (flat, no trade)
      - No stop-loss / take-profit — exit at horizon
      - Entry at Open[t+1], Exit at Close[t+horizon]
    """
    from utils.yfinance_cache import get_historical_data

    # Load data
    data = get_historical_data(ticker, interval="daily", outputsize="full")
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=120)
    end_ts = pd.Timestamp(end_date)
    data = data.loc[buffer_start:end_ts]

    dates = data.index
    open_prices = pd.to_numeric(data["Open"], errors="coerce").values
    close_prices_arr = pd.to_numeric(data["Close"], errors="coerce").values
    close_prices_series = pd.to_numeric(data["Close"], errors="coerce")

    start_idx = dates.searchsorted(pd.Timestamp(start_date))
    start_idx = max(start_idx, 60)
    total_days = len(dates)

    cost_per_trade = (cost_bps + slippage_bps) / 10000.0 * 2  # round-trip

    # ── Pre-fetch macro/fundamental historical data (once) ──
    macro_fund_cols = meta.get("macro_fundamental_features", [])
    macro_fund_df = None
    if macro_fund_cols:
        try:
            from utils.macro_fundamental_provider import MacroFundamentalFeatureProvider
            provider = MacroFundamentalFeatureProvider(verbose=verbose)
            mf_start = (pd.Timestamp(start_date) - pd.Timedelta(days=120)).to_pydatetime()
            mf_end = pd.Timestamp(end_date).to_pydatetime()
            macro_fund_df = provider.extract_historical(
                stock_symbol=ticker,
                start_date=mf_start,
                end_date=mf_end,
            )
            if macro_fund_df is not None and not macro_fund_df.empty:
                macro_fund_df = macro_fund_df.sort_index()
                macro_fund_df.index = pd.to_datetime(macro_fund_df.index)

                # Compute price-dependent features
                _INTERMEDIATE_COLS = [
                    "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
                    "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
                ]
                close_for_mf = close_prices_series.reindex(macro_fund_df.index, method="ffill")

                # P/E ratio
                if "_ttm_eps" in macro_fund_df.columns:
                    ttm_eps = macro_fund_df["_ttm_eps"]
                    valid = ttm_eps.notna() & (ttm_eps.abs() > 0.01) & close_for_mf.notna()
                    macro_fund_df.loc[valid, "pe_ratio"] = close_for_mf[valid] / ttm_eps[valid]

                # P/B ratio
                if "_total_equity" in macro_fund_df.columns and "_shares_outstanding" in macro_fund_df.columns:
                    equity_col = macro_fund_df["_total_equity"]
                    shares = macro_fund_df["_shares_outstanding"]
                    valid = equity_col.notna() & shares.notna() & (shares > 0) & close_for_mf.notna()
                    bvps = equity_col[valid] / shares[valid]
                    bvps_valid = bvps.abs() > 0.01
                    final_idx = bvps_valid.index[bvps_valid]
                    macro_fund_df.loc[final_idx, "pb_ratio"] = close_for_mf[final_idx] / bvps[bvps_valid]

                # P/S ratio
                if "_ttm_revenue" in macro_fund_df.columns and "_shares_outstanding" in macro_fund_df.columns:
                    ttm_rev = macro_fund_df["_ttm_revenue"]
                    shares = macro_fund_df["_shares_outstanding"]
                    valid = ttm_rev.notna() & shares.notna() & (shares > 0) & (ttm_rev > 0) & close_for_mf.notna()
                    rps = ttm_rev[valid] / shares[valid]
                    macro_fund_df.loc[valid, "ps_ratio"] = close_for_mf[valid] / rps

                # EV/EBITDA
                if all(c in macro_fund_df.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                    shares = macro_fund_df["_shares_outstanding"]
                    ttm_ebitda = macro_fund_df["_ttm_ebitda"]
                    total_liab = macro_fund_df["_total_liabilities"].fillna(0)
                    cash = macro_fund_df["_cash"].fillna(0)
                    valid = shares.notna() & (shares > 0) & ttm_ebitda.notna() & (ttm_ebitda.abs() > 0) & close_for_mf.notna()
                    market_cap = close_for_mf[valid] * shares[valid]
                    ev = market_cap + total_liab[valid] - cash[valid]
                    macro_fund_df.loc[valid, "ev_ebitda"] = ev / ttm_ebitda[valid]

                # Beta (rolling 252d)
                try:
                    spy_cache_file = Path("data/training_cache/SPY_daily.csv")
                    if spy_cache_file.exists():
                        spy_df = pd.read_csv(spy_cache_file, index_col=0, parse_dates=True)
                    else:
                        spy_df = get_historical_data("SPY", interval="daily", outputsize="full")
                    if not spy_df.empty:
                        spy_close = pd.to_numeric(spy_df["Close"], errors="coerce")
                        spy_returns = spy_close.pct_change()
                        stock_returns = close_prices_series.pct_change()
                        aligned = pd.DataFrame({"stock": stock_returns, "spy": spy_returns}).dropna()
                        if len(aligned) > 60:
                            rolling_cov = aligned["stock"].rolling(252, min_periods=60).cov(aligned["spy"])
                            rolling_var = aligned["spy"].rolling(252, min_periods=60).var()
                            rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
                            macro_fund_df["beta"] = rolling_beta.reindex(macro_fund_df.index).ffill()
                except Exception as exc:
                    if verbose:
                        print(f"  [warn] beta computation failed: {exc}")

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

                n_mf = len(macro_fund_df)
                n_mf_cols = sum(1 for c in macro_fund_cols if c in macro_fund_df.columns)
                print(f"  [macro/fund] Loaded {n_mf} days, {n_mf_cols}/{len(macro_fund_cols)} features available")
            else:
                print("  [warn] macro/fund data empty, using NaN")
                macro_fund_df = None
        except Exception as exc:
            print(f"  [warn] macro/fund fetch failed: {exc}, using NaN")
            macro_fund_df = None

    trade_log = []
    equity = 1.0
    equity_curve = {}
    step = 0
    total_steps = (total_days - start_idx - horizon) // horizon

    t = start_idx
    while t < total_days - horizon:
        step += 1
        current_date = dates[t]

        # Compute features (no look-ahead)
        data_slice = data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            t += horizon
            continue

        # Compute real regime features
        current_close = float(close_prices_arr[t])
        regime_feats = compute_regime_from_features(features, current_close)

        # Look up macro/fund features by date (backward fill)
        mf_row = None
        if macro_fund_df is not None and not macro_fund_df.empty:
            valid_dates = macro_fund_df.index[macro_fund_df.index <= current_date]
            if len(valid_dates) > 0:
                nearest_date = valid_dates[-1]
                mf_row = macro_fund_df.loc[nearest_date].to_dict()

        # Score with real regime + macro/fund features
        raw_prob, prob = score_features(
            features, model, meta, calibrator,
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
        if direction != 0.0 and t + 1 < total_days:
            entry_price = open_prices[t + 1]
            exit_idx = min(t + 1 + horizon, total_days - 1)
            exit_price = close_prices_arr[exit_idx]

            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price <= 0:
                t += horizon
                continue

            raw_return = (exit_price / entry_price - 1.0) * direction
            net_return = raw_return - cost_per_trade

            equity *= (1.0 + net_return)

            trade = {
                "date": current_date.strftime("%Y-%m-%d"),
                "action": action,
                "probability_up": round(prob, 6),
                "raw_probability_up": round(raw_prob, 6),
                "direction": direction,
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "raw_return": round(float(raw_return), 6),
                "net_return": round(float(net_return), 6),
                "equity": round(float(equity), 6),
            }
            trade_log.append(trade)
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

            if verbose and step % 20 == 0:
                print(
                    f"  [{ticker}] Step {step}/{total_steps}: "
                    f"date={current_date.strftime('%Y-%m-%d')}, "
                    f"action={action}, prob={prob:.4f}, "
                    f"ret={net_return:+.4f}, equity={equity:.4f}"
                )
        else:
            # Hold — record for analysis but no equity change
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

            if verbose and step % 50 == 0:
                print(
                    f"  [{ticker}] Step {step}/{total_steps}: "
                    f"date={current_date.strftime('%Y-%m-%d')}, "
                    f"action=hold, prob={prob:.4f}, equity={equity:.4f}"
                )

        t += horizon

    # Build benchmark (SPY buy-and-hold)
    try:
        spy_data = get_historical_data("SPY", interval="daily", outputsize="full")
        if not isinstance(spy_data.index, pd.DatetimeIndex):
            spy_data.index = pd.to_datetime(spy_data.index)
        spy_data = spy_data.sort_index()
        spy_slice = spy_data.loc[start_date:end_date]
        spy_close = pd.to_numeric(spy_slice["Close"], errors="coerce")
        benchmark_return = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1.0)
    except Exception:
        benchmark_return = 0.0

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "horizon": horizon,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "final_equity": equity,
        "benchmark_return": benchmark_return,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Analysis & reporting
# ═══════════════════════════════════════════════════════════════════════════

def analyze_backtest(result: dict) -> dict:
    """Compute comprehensive metrics from backtest result."""
    trades = result["trade_log"]
    if not trades:
        return {"error": "No trades executed"}

    returns = [t["net_return"] for t in trades]
    raw_returns = [t["raw_return"] for t in trades]
    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]

    # Basic metrics
    total_return = result["final_equity"] - 1.0
    n_trades = len(trades)
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
    horizon = result["horizon"]
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
    alpha = total_return - result["benchmark_return"]

    # Buy vs Sell breakdown
    buy_returns = [t["net_return"] for t in buy_trades]
    sell_returns = [t["net_return"] for t in sell_trades]
    buy_hit = sum(1 for r in buy_returns if r > 0) / len(buy_returns) if buy_returns else 0.0
    sell_hit = sum(1 for r in sell_returns if r > 0) / len(sell_returns) if sell_returns else 0.0
    buy_avg = float(np.mean(buy_returns)) if buy_returns else 0.0
    sell_avg = float(np.mean(sell_returns)) if sell_returns else 0.0

    # IC of trade signals (prob vs actual return)
    probs = [t["probability_up"] for t in trades]
    actuals = [t["raw_return"] * t["direction"] for t in trades]  # unsigned return
    if len(probs) > 5:
        trade_ic, trade_ic_pval = stats.spearmanr(probs, actuals)
    else:
        trade_ic, trade_ic_pval = 0.0, 1.0

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
        "benchmark_return": round(result["benchmark_return"], 6),
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
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
    }


def print_report(result: dict, metrics: dict) -> None:
    """Pretty-print Stage 1 backtest report."""
    ticker = result["ticker"]

    print(f"\n{'='*60}")
    print(f"  Stage 1: ForecastAgent Only Backtest — {ticker}")
    print(f"{'='*60}")

    print(f"\n📋 Configuration:")
    print(f"  Period:     {result['start_date']} → {result['end_date']}")
    print(f"  Horizon:    {result['horizon']}d")
    print(f"  Thresholds: buy > {result['buy_threshold']}, sell < {result['sell_threshold']}")
    print(f"  Costs:      {result['cost_bps']}bps + {result['slippage_bps']}bps slippage")

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
    print(f"  Trade IC:        {metrics['trade_ic']:+.6f} (p={metrics['trade_ic_pval']:.4f})")
    print(f"  Max Consec Wins: {metrics['max_consecutive_wins']}")
    print(f"  Max Consec Loss: {metrics['max_consecutive_losses']}")

    # Verdict
    print(f"\n{'─'*60}")
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

    if not issues:
        print("  ✅ VERDICT: ForecastAgent signals are profitable. Proceed to Stage 2.")
    elif len(issues) <= 2 and tr > 0:
        print("  ⚠️  VERDICT: ForecastAgent signals are marginally profitable:")
        for issue in issues:
            print(f"     • {issue}")
        print("  → Consider tuning thresholds before Stage 2.")
    else:
        print("  ❌ VERDICT: ForecastAgent signals have issues:")
        for issue in issues:
            print(f"     • {issue}")
        print("  → Debug signal quality before proceeding.")


def print_trade_summary(trades: list, n: int = 10) -> None:
    """Print first/last N trades for inspection."""
    if not trades:
        return

    print(f"\n── Sample Trades (first {min(n, len(trades))}) ──")
    print(f"  {'Date':12s} | {'Action':6s} | {'Prob':>7s} | {'Entry':>9s} | {'Exit':>9s} | {'Return':>9s} | {'Equity':>8s}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*8}")
    for t in trades[:n]:
        print(
            f"  {t['date']:12s} | {t['action']:6s} | {t['probability_up']:7.4f} | "
            f"{t['entry_price']:9.2f} | {t['exit_price']:9.2f} | "
            f"{t['net_return']:+9.4f} | {t['equity']:8.4f}"
        )

    if len(trades) > n:
        print(f"  ... ({len(trades) - n} more trades) ...")
        print(f"\n── Last {min(n, len(trades))} Trades ──")
        print(f"  {'Date':12s} | {'Action':6s} | {'Prob':>7s} | {'Entry':>9s} | {'Exit':>9s} | {'Return':>9s} | {'Equity':>8s}")
        print(f"  {'-'*12} | {'-'*6} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*8}")
        for t in trades[-n:]:
            print(
                f"  {t['date']:12s} | {t['action']:6s} | {t['probability_up']:7.4f} | "
                f"{t['entry_price']:9.2f} | {t['exit_price']:9.2f} | "
                f"{t['net_return']:+9.4f} | {t['equity']:8.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 1: ForecastAgent Only Backtest")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="Holding period in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Buy/sell threshold (default: 0.50)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost bps (default: 5.0)")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps (default: 5.0)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    print(f"\n{'='*60}")
    print(f"  Stage 1: ForecastAgent Only Backtest")
    print(f"  Ticker:    {ticker}")
    print(f"  Period:    {args.start} → {args.end}")
    print(f"  Horizon:   {args.horizon}d")
    print(f"  Threshold: {args.threshold}")
    print(f"  Model:     {os.getenv('FORECAST_LGB_MODEL_PATH', 'data/forecast_model.lgb')}")
    print(f"{'='*60}")

    # Load model
    print(f"\n── Loading model ──")
    model, meta, calibrator = load_model()

    # Run backtest
    print(f"\n── Running backtest ──")
    t0 = time.time()
    result = run_stage1_backtest(
        ticker=ticker,
        start_date=args.start,
        end_date=args.end,
        model=model,
        meta=meta,
        calibrator=calibrator,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        buy_threshold=args.threshold,
        sell_threshold=1.0 - args.threshold,  # symmetric: buy > 0.55, sell < 0.45
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print(f"\n[✓] Backtest completed in {elapsed:.1f}s")

    # Analyze
    metrics = analyze_backtest(result)

    # Print report
    print_report(result, metrics)
    print_trade_summary(result["trade_log"])

    # Save results
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"stage1_{ticker}_{args.start}_{args.end}_t{args.threshold}"

    # Save report JSON
    report_path = output_dir / f"{base_name}_report.json"
    save_data = {
        "config": {
            "ticker": ticker,
            "start_date": args.start,
            "end_date": args.end,
            "horizon": args.horizon,
            "buy_threshold": args.threshold,
            "sell_threshold": 1.0 - args.threshold,
            "cost_bps": args.cost_bps,
            "slippage_bps": args.slippage_bps,
            "model_path": os.getenv("FORECAST_LGB_MODEL_PATH", ""),
        },
        "metrics": metrics,
    }
    with open(report_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[✓] Report saved: {report_path}")

    # Save trade log CSV
    trades_path = output_dir / f"{base_name}_trades.csv"
    if result["trade_log"]:
        pd.DataFrame(result["trade_log"]).to_csv(trades_path, index=False)
        print(f"[✓] Trade log saved: {trades_path}")

    # Save equity curve CSV
    equity_path = output_dir / f"{base_name}_equity.csv"
    if result["equity_curve"]:
        eq_df = pd.DataFrame(
            list(result["equity_curve"].items()),
            columns=["date", "equity"],
        )
        eq_df.to_csv(equity_path, index=False)
        print(f"[✓] Equity curve saved: {equity_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
