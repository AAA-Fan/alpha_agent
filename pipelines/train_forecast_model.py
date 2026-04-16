#!/usr/bin/env python3
"""
Train a LightGBM binary classifier for ForecastAgent.

Phase 1: LightGBM with 15 base features + Purged K-Fold CV
Phase 2: + 5 Regime features (compressed ordinal encoding)
Phase 2.5: + 24 Macro/Fundamental features (from MacroFundamentalFeatureProvider)
Phase 3 (V2): Cross-sectional model — 100 tickers, cross-sectional rank features,
              dual-layer industry encoding, date-based PurgedKFold.

Data source: Alpha Vantage Premium (TIME_SERIES_DAILY, outputsize=full)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.isotonic import IsotonicRegression  # kept for backward compat with old .pkl files
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import project modules
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.yfinance_cache import get_historical_data  # noqa: E402
from utils.macro_fundamental_provider import (  # noqa: E402
    MacroFundamentalFeatureProvider,
    MACRO_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    ALL_MACRO_FUNDAMENTAL_COLUMNS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_FEATURE_COLUMNS = [
    "momentum_5",
    "momentum_20",
    "sma_20_ratio",
    "sma_50_ratio",
    "macd_hist",
    "rsi_14",
    "volatility_20",
    "daily_volatility_20",
    "atr_14",
    "volume_zscore_20",
    "drawdown_60",
    "overnight_gap",
    "intraday_return",
    "return_1d",
    "return_5d",
    # Interaction / derived features
    "momentum_trend_align",
    "rsi_deviation",
    "vol_confirmed_momentum",
    "mean_reversion",
    "vol_adj_momentum_5",
]

ALL_REGIME_STATES = [
    "strong_rally",
    "trending_up",
    "topping_out",
    "range_bound",
    "coiling",
    "choppy",
    "trending_down",
    "bottoming_out",
    "capitulation",
]

REGIME_FEATURE_COLUMNS_ONEHOT = [
    # 9 one-hot state features (kept for reference, not used in training)
    *[f"regime_{s}" for s in ALL_REGIME_STATES],
    # 3 continuous / encoded features
    "trend_strength",
    "vol_expanding",
    "momentum_health_enc",
]

# Compressed regime features: replace 9 sparse one-hot with 2 ordinal encodings
# This reduces sparsity and works better with small sample sizes
REGIME_FEATURE_COLUMNS = [
    "regime_direction",       # -1=bearish, 0=neutral, 1=bullish (ordinal)
    "regime_volatility_ord",  # 0=low, 1=normal, 2=high, 3=extreme (ordinal)
    "trend_strength",         # 0.0-1.0 (continuous)
    "vol_expanding",          # 0/1 (binary)
    "momentum_health_enc",    # 0-3 (ordinal)
]

# Macro/Fundamental feature columns for training
# At training time, we fetch *historical* time-series data so each row sees the
# macro/fundamental values that were known at that point in time. This allows the
# model to learn how technical signals behave under different macro environments.
# At inference time, the latest snapshot is used (via extract()).
MACRO_FUNDAMENTAL_FEATURE_COLUMNS = ALL_MACRO_FUNDAMENTAL_COLUMNS

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 16,              # Moderate complexity for ~1200 samples
    "max_depth": 4,                # Slightly deeper to capture interactions
    "learning_rate": 0.01,         # Slow LR + more rounds = better generalisation
    "n_estimators": 800,           # More rounds; early stopping will cut
    "min_child_samples": 25,       # Lower threshold to allow finer splits
    "min_data_in_leaf": 25,        # Same as min_child_samples
    "subsample": 0.75,             # Slightly more data per tree
    "subsample_freq": 1,           # Apply subsampling every iteration
    "colsample_bytree": 0.7,       # Use more features per tree
    "reg_alpha": 0.1,              # Lighter L1 to let model learn
    "reg_lambda": 1.0,             # Moderate L2
    "min_gain_to_split": 0.001,    # Lower bar for splits
    "path_smooth": 5.0,            # Less aggressive smoothing
    "extra_trees": True,           # Randomised splits for better generalisation
    "random_state": 42,
    "verbose": -1,
    "feature_pre_filter": False,
}

# V2 cross-sectional model parameters – tuned for low signal-to-noise ratio
# Key principle: moderate complexity + extra_trees for built-in regularisation
LGB_PARAMS_V2 = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 24,              # Moderate complexity
    "max_depth": 5,                # Moderate depth
    "learning_rate": 0.02,         # Stable learning rate
    "n_estimators": 800,           # More rounds; early stopping will cut
    "min_child_samples": 150,      # Balanced threshold
    "min_data_in_leaf": 150,       # Same as min_child_samples
    "subsample": 0.65,             # Row subsampling
    "subsample_freq": 1,           # Apply subsampling every iteration
    "colsample_bytree": 0.5,       # Aggressive column subsampling
    "reg_alpha": 0.3,              # Moderate L1 regularisation
    "reg_lambda": 1.5,             # Moderate L2 regularisation
    "min_gain_to_split": 0.003,    # Lower split threshold to capture weak signals
    "path_smooth": 8.0,            # Smoothing for noisy labels
    "extra_trees": True,           # Extremely randomised trees for better generalisation
    "random_state": 42,
    "verbose": -1,
    "feature_pre_filter": False,
}

# ── Cross-sectional model constants ──────────────────────────────────────

# GICS sector mapping (from Alpha Vantage OVERVIEW Sector field)
SECTOR_MAP = {
    # Alpha Vantage OVERVIEW returns UPPER CASE sector names
    "TECHNOLOGY": 0,
    "HEALTHCARE": 1,
    "FINANCIAL SERVICES": 2,
    "CONSUMER CYCLICAL": 3,
    "COMMUNICATION SERVICES": 4,
    "INDUSTRIALS": 5,
    "CONSUMER DEFENSIVE": 6,
    "ENERGY": 7,
    "UTILITIES": 8,
    "REAL ESTATE": 9,
    "BASIC MATERIALS": 10,
    # Title-case aliases (for compatibility / documentation)
    "Technology": 0,
    "Healthcare": 1,
    "Financials": 2,
    "Financial Services": 2,
    "Consumer Discretionary": 3,
    "Consumer Cyclical": 3,
    "Communication Services": 4,
    "Industrials": 5,
    "Consumer Staples": 6,
    "Consumer Defensive": 6,
    "Energy": 7,
    "Utilities": 8,
    "Real Estate": 9,
    "Materials": 10,
    "Basic Materials": 10,
}

# Columns to compute cross-sectional rank for
RANK_COLUMNS_TECH = [
    "momentum_5", "momentum_20", "sma_20_ratio", "sma_50_ratio",
    "macd_hist", "rsi_14", "volatility_20", "daily_volatility_20",
    "atr_14", "volume_zscore_20", "drawdown_60", "overnight_gap",
    "intraday_return", "return_1d", "return_5d",
    "momentum_trend_align", "rsi_deviation", "vol_confirmed_momentum",
    "mean_reversion", "vol_adj_momentum_5",
]

RANK_COLUMNS_FUND = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "dividend_yield", "roe", "profit_margin",
    "revenue_growth_yoy", "earnings_growth_yoy", "beta",
]

# Early stopping – moderate patience for weak-signal financial data
USE_EARLY_STOPPING = True
EARLY_STOPPING_ROUNDS = 80

# Number of random seeds to average for more stable CV
N_SEEDS = 5


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Download
# ═══════════════════════════════════════════════════════════════════════════

def download_training_data(
    tickers: List[str],
    cache_dir: str = "data/training_cache",
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV data via Alpha Vantage Premium and cache locally.

    Uses get_historical_data() which internally calls Alpha Vantage
    TIME_SERIES_DAILY. The full history is fetched (Alpha Vantage returns
    up to 20 years for premium accounts).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        csv_file = cache_path / f"{ticker}_daily.csv"

        # Use cache if fresh (< 24 hours old)
        if csv_file.exists():
            age_hours = (time.time() - csv_file.stat().st_mtime) / 3600
            if age_hours < 24:
                print(f"  [cache] {ticker}: loading from {csv_file}")
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                result[ticker] = df
                continue

        print(f"  [download] {ticker}: fetching from Alpha Vantage Premium ...")
        try:
            # get_historical_data fetches full history with outputsize=full
            df = get_historical_data(ticker, interval="daily", days=None, outputsize="full")
            if df.empty:
                print(f"  [warn] {ticker}: no data returned")
                continue
            # Save to cache
            df.to_csv(csv_file)
            result[ticker] = df
            print(f"  [ok] {ticker}: {len(df)} rows cached")
        except Exception as exc:
            print(f"  [error] {ticker}: {exc}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature Engineering (vectorized, all rows)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_base_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 base features for every row (vectorized).

    Mirrors FeatureEngineeringAgent._build_features() but returns a full
    DataFrame instead of a single-row dict.
    """
    frame = data.sort_index().copy()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    open_ = pd.to_numeric(frame["Open"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")

    returns = close.pct_change()

    # Momentum
    frame["momentum_5"] = close.pct_change(5)
    frame["momentum_20"] = close.pct_change(20)

    # Moving average ratios
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    frame["sma_20_ratio"] = (close / sma_20) - 1
    frame["sma_50_ratio"] = (close / sma_50) - 1

    # MACD histogram
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = macd - macd_signal

    # RSI
    frame["rsi_14"] = _compute_rsi(close, period=14)

    # Volatility
    frame["volatility_20"] = returns.rolling(20).std() * np.sqrt(252)
    frame["daily_volatility_20"] = returns.rolling(20).std()

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = tr.rolling(14).mean()

    # Volume z-score
    volume_mean_20 = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std().replace(0, np.nan)
    frame["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20

    # Drawdown
    rolling_max_60 = close.rolling(60).max()
    frame["drawdown_60"] = (close / rolling_max_60) - 1

    # Overnight gap & intraday return
    frame["overnight_gap"] = (open_ / prev_close) - 1
    frame["intraday_return"] = (close / open_) - 1

    # Returns
    frame["return_1d"] = returns
    frame["return_5d"] = close / close.shift(5) - 1

    # ── Interaction / derived features (boost signal-to-noise) ────────
    # Momentum × trend alignment
    frame["momentum_trend_align"] = frame["momentum_5"] * frame["sma_20_ratio"]
    # RSI deviation from neutral (captures overbought/oversold intensity)
    frame["rsi_deviation"] = (frame["rsi_14"] - 50.0) / 50.0
    # Volume-confirmed momentum (high volume + momentum = stronger signal)
    frame["vol_confirmed_momentum"] = frame["momentum_5"] * frame["volume_zscore_20"].clip(-3, 3)
    # Mean reversion signal (drawdown × momentum reversal)
    frame["mean_reversion"] = frame["drawdown_60"] * frame["momentum_5"].clip(-0.1, 0.1)
    # Volatility-adjusted momentum (momentum normalized by volatility)
    safe_vol = frame["daily_volatility_20"].replace(0, np.nan)
    frame["vol_adj_momentum_5"] = frame["momentum_5"] / safe_vol
    frame["vol_adj_momentum_5"] = frame["vol_adj_momentum_5"].clip(-5, 5)

    return frame


# ═══════════════════════════════════════════════════════════════════════════
# 3. Regime Feature Engineering (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

def _trend_score_from_row(row: pd.Series) -> float:
    """Compute trend score from a feature row (mirrors RegimeAgent._trend_score)."""
    score = 0.0
    weights = {
        "sma_20_ratio": 2.0,
        "sma_50_ratio": 1.5,
        "momentum_20": 1.2,
        "macd_hist": 0.8,
    }
    for key, w in weights.items():
        val = row.get(key)
        if pd.notna(val):
            score += w * float(val)
    return score


def _classify_trend(score: float, m5: float) -> str:
    """5-level trend classification (mirrors RegimeAgent._classify_trend)."""
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
    """Normalize trend score to 0-1 (mirrors RegimeAgent._trend_strength)."""
    return min(1.0, abs(score) / 0.10)


def _classify_volatility(
    annualized_vol: float,
    extreme_thresh: float = 0.50,
    high_thresh: float = 0.35,
    low_thresh: float = 0.16,
) -> str:
    """4-level volatility classification (mirrors RegimeAgent._classify_volatility)."""
    if pd.isna(annualized_vol):
        return "unknown"
    if annualized_vol >= extreme_thresh:
        return "extreme"
    if annualized_vol >= high_thresh:
        return "high"
    if annualized_vol <= low_thresh:
        return "low"
    return "normal"


def _is_vol_expanding(vol_20: float, atr_14: float, price: float) -> bool:
    """Detect vol expansion (mirrors RegimeAgent._is_vol_expanding)."""
    if pd.isna(vol_20) or pd.isna(atr_14) or pd.isna(price) or price <= 0:
        return False
    atr_daily_pct = atr_14 / price
    atr_annualized = atr_daily_pct * (252 ** 0.5)
    return vol_20 > atr_annualized * 1.20


def _classify_momentum_health(m5: float, m20: float, rsi: float, trend: str) -> str:
    """4-level momentum health (mirrors RegimeAgent._classify_momentum_health)."""
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
    """Drawdown severity (mirrors RegimeAgent._drawdown_severity)."""
    if pd.isna(dd):
        return "none"
    if dd <= -0.20:
        return "severe"
    if dd <= -0.10:
        return "moderate"
    if dd <= -0.03:
        return "mild"
    return "none"


def _build_state(trend: str, vol_regime: str, momentum_health: str, dd_severity: str) -> str:
    """Map 3 dimensions into one of 9 composite regime states (mirrors RegimeAgent._build_state)."""
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


HEALTH_MAP = {"accelerating": 0, "steady": 1, "decelerating": 2, "exhausted": 3}


# Mapping from composite state to directional ordinal
_STATE_DIRECTION = {
    "strong_rally": 2,
    "trending_up": 1,
    "topping_out": 0,
    "range_bound": 0,
    "coiling": 0,
    "choppy": -1,
    "trending_down": -1,
    "bottoming_out": 0,
    "capitulation": -2,
}

_VOL_REGIME_ORD = {
    "low": 0,
    "normal": 1,
    "high": 2,
    "extreme": 3,
    "unknown": 1,
}


def compute_regime_features(
    frame: pd.DataFrame,
    transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Compute compressed regime features for every row.

    Instead of 9 sparse one-hot state features, we use:
      - regime_direction: ordinal (-2 to +2) encoding market direction
      - regime_volatility_ord: ordinal (0-3) encoding volatility level
      - trend_strength: continuous 0-1
      - vol_expanding: binary 0/1
      - momentum_health_enc: ordinal 0-3

    Total: 5 regime features (down from 12), much less sparse.

    If *transition_matrix* is provided, regime_direction is computed from
    the smoothed state sequence (per ticker) rather than the raw state.
    """
    TRANSITION_PROB_THRESHOLD = 0.03

    regime_rows: List[Dict[str, Any]] = []
    raw_states: List[str] = []  # track raw state for smoothing

    def _safe_float(val, default: float = 0.0) -> float:
        """Extract a scalar float from a value that may be scalar or Series."""
        if val is None:
            return default
        try:
            if pd.notna(val):
                return float(val)
        except (ValueError, TypeError):
            pass
        return default

    for i in range(len(frame)):
        row = frame.iloc[i]
        m5 = _safe_float(row.get("momentum_5"), 0.0)
        m20 = _safe_float(row.get("momentum_20"), 0.0)
        rsi = _safe_float(row.get("rsi_14"), 50.0)
        vol_20 = _safe_float(row.get("volatility_20"), 0.25)
        atr_14_val = _safe_float(row.get("atr_14"), 0.0)
        price = _safe_float(row.get("Close"), 0.0)
        dd_60 = _safe_float(row.get("drawdown_60"), 0.0)

        ts = _trend_score_from_row(row)
        trend = _classify_trend(ts, m5)
        strength = _trend_strength(ts)
        vol_regime = _classify_volatility(vol_20)
        vol_exp = _is_vol_expanding(vol_20, atr_14_val, price)
        mom_health = _classify_momentum_health(m5, m20, rsi, trend)
        dd_sev = _drawdown_severity(dd_60)
        state = _build_state(trend, vol_regime, mom_health, dd_sev)

        raw_states.append(state)

        r: Dict[str, Any] = {
            "regime_direction": _STATE_DIRECTION.get(state, 0),
            "regime_volatility_ord": _VOL_REGIME_ORD.get(vol_regime, 1),
            "trend_strength": strength,
            "vol_expanding": int(vol_exp),
            "momentum_health_enc": HEALTH_MAP.get(mom_health, 1),
        }
        regime_rows.append(r)

    # Apply smoothing per ticker if transition_matrix is provided
    if transition_matrix and "ticker" in frame.columns:
        for ticker in frame["ticker"].unique():
            mask = frame["ticker"] == ticker
            indices = [i for i in range(len(frame)) if mask.iloc[i]]
            ticker_raw = [raw_states[i] for i in indices]

            # Smooth the sequence (same logic as RegimeAgent._smooth_regime_sequence)
            if len(ticker_raw) >= 2:
                smoothed = [ticker_raw[0]]
                for j in range(1, len(ticker_raw)):
                    prev = smoothed[-1]
                    curr = ticker_raw[j]
                    if prev == curr:
                        smoothed.append(curr)
                    else:
                        prob = transition_matrix.get(prev, {}).get(curr, 1.0)
                        if prob < TRANSITION_PROB_THRESHOLD:
                            smoothed.append(prev)
                        else:
                            smoothed.append(curr)

                # Update regime_direction based on smoothed state
                for k, idx in enumerate(indices):
                    regime_rows[idx]["regime_direction"] = _STATE_DIRECTION.get(
                        smoothed[k], 0
                    )

    regime_df = pd.DataFrame(regime_rows, index=frame.index)
    return pd.concat([frame, regime_df], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Regime Transition Matrix Construction
# ═══════════════════════════════════════════════════════════════════════════

ALL_REGIME_STATES = [
    "strong_rally", "trending_up", "topping_out", "range_bound",
    "coiling", "choppy", "trending_down", "bottoming_out", "capitulation",
]


def build_regime_transition_matrix(
    regime_states: pd.Series,
    ticker_ids: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Build global 9x9 Markov transition matrix from all tickers' regime sequences.

    For each ticker, iterate through consecutive days and count
    (from_state, to_state) pairs.  Normalize each row to sum to 1.0.

    Args:
        regime_states: Per-row regime state (e.g. 'trending_up').
        ticker_ids: Ticker identifier for each row (same length as regime_states).

    Returns:
        Nested dict: matrix[from_state][to_state] = probability.
    """
    counts: Dict[str, Dict[str, int]] = {
        s: {t: 0 for t in ALL_REGIME_STATES} for s in ALL_REGIME_STATES
    }

    for ticker in ticker_ids.unique():
        mask = ticker_ids == ticker
        states = regime_states[mask].tolist()
        for i in range(1, len(states)):
            prev, curr = states[i - 1], states[i]
            if prev in counts and curr in counts[prev]:
                counts[prev][curr] += 1

    # Normalize: each row sums to 1.0
    matrix: Dict[str, Dict[str, float]] = {}
    for from_state in ALL_REGIME_STATES:
        row_sum = sum(counts[from_state].values())
        if row_sum > 0:
            matrix[from_state] = {
                to_state: round(count / row_sum, 6)
                for to_state, count in counts[from_state].items()
            }
        else:
            # Uniform prior if no data for this state
            matrix[from_state] = {
                to_state: round(1.0 / len(ALL_REGIME_STATES), 6)
                for to_state in ALL_REGIME_STATES
            }

    return matrix


# ═══════════════════════════════════════════════════════════════════════════
# 4. Label Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_labels(
    frame: pd.DataFrame,
    horizon_days: int = 5,
    spy_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add binary label based on excess return relative to SPY.

    If *spy_data* is provided the label is:
        label = 1  if  stock_return_Nd − SPY_return_Nd > 0   (outperforms)
        label = 0  otherwise

    This eliminates the systematic positive-class bias that arises when
    using absolute return (``future_return > 0``) on stocks with a long-term
    upward drift (e.g. AAPL positive ratio ≈ 54 %).

    When *spy_data* is ``None`` or empty the function falls back to the
    original absolute-return label so that training still works without SPY.
    """
    close = pd.to_numeric(frame["Close"], errors="coerce")
    stock_return = close.shift(-horizon_days) / close - 1

    if spy_data is not None and not spy_data.empty:
        # ── Excess return label (relative to SPY) ────────────────────
        spy_close = pd.to_numeric(spy_data["Close"], errors="coerce")
        spy_return = spy_close.shift(-horizon_days) / spy_close - 1

        # Build a small DataFrame of SPY returns for merge_asof alignment
        spy_ret_df = spy_return.rename("spy_return").to_frame()
        spy_ret_df.index = pd.to_datetime(spy_ret_df.index)

        frame_idx_name = frame.index.name or "index"
        frame.index = pd.to_datetime(frame.index)

        aligned = pd.merge_asof(
            frame[[]]  # empty DF, keep index only
                .reset_index()
                .rename(columns={frame_idx_name: "_date"})
                .sort_values("_date"),
            spy_ret_df
                .reset_index()
                .rename(columns={spy_ret_df.index.name or "index": "_date"})
                .sort_values("_date"),
            on="_date",
            direction="nearest",
            tolerance=pd.Timedelta("2D"),  # allow weekend / holiday alignment
        ).set_index("_date")

        frame["future_return"] = stock_return
        frame["spy_future_return"] = aligned["spy_return"].values
        frame["excess_return"] = stock_return - aligned["spy_return"].values
        frame["label"] = (frame["excess_return"] > 0).astype(int)
        frame["label_type"] = "excess_return"
        print(f"    Label type: excess return (vs SPY)")
    else:
        # ── Fallback: absolute return label ──────────────────────────
        print("    [warn] SPY data not available, falling back to absolute return labels")
        frame["future_return"] = stock_return
        frame["label"] = (frame["future_return"] > 0).astype(int)
        frame["label_type"] = "absolute_return"

    return frame


def assign_cross_sectional_labels(
    feature_matrix: pd.DataFrame,
    upper_quantile: float = 0.6,
    lower_quantile: float = 0.4,
) -> pd.DataFrame:
    """Re-assign labels based on cross-sectional ranking of excess returns.

    For each date, rank all tickers by excess_return:
      - Top (1 - upper_quantile) fraction  → label = 1  (clear outperformers)
      - Bottom lower_quantile fraction     → label = 0  (clear underperformers)
      - Middle noise band                  → dropped (NaN label)

    This removes ~20 % of ambiguous samples whose excess return is near zero,
    dramatically improving label quality and model signal-to-noise ratio.
    """
    n_before = len(feature_matrix)

    def _label_group(group: pd.DataFrame) -> pd.Series:
        er = group["excess_return"]
        if len(er) < 5:
            # Too few tickers on this date – keep original > 0 labels
            return group["label"]
        upper = er.quantile(upper_quantile)
        lower = er.quantile(lower_quantile)
        labels = pd.Series(np.nan, index=group.index)
        labels[er >= upper] = 1.0
        labels[er <= lower] = 0.0
        return labels

    feature_matrix["label"] = (
        feature_matrix.groupby(feature_matrix.index, group_keys=False)
        .apply(_label_group)
    )

    # Drop noise-band samples
    feature_matrix = feature_matrix.dropna(subset=["label"]).copy()
    feature_matrix["label"] = feature_matrix["label"].astype(int)
    n_after = len(feature_matrix)
    n_dropped = n_before - n_after
    pct_dropped = n_dropped / n_before * 100 if n_before > 0 else 0

    print(f"  Cross-sectional labels: {n_after}/{n_before} samples retained "
          f"({n_dropped} noise-band samples dropped, {pct_dropped:.1f}%)")
    print(f"  Class balance: {feature_matrix['label'].mean():.3f}")

    return feature_matrix


# ═══════════════════════════════════════════════════════════════════════════
# 4b. Cross-Sectional Feature Engineering (Phase 3 / V2)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cross_sectional_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional rank features grouped by date.

    For each date, ranks all tickers' indicator values and normalizes
    to [0, 1] percentile. Both raw and rank features are kept.
    """
    rank_cols = RANK_COLUMNS_TECH + RANK_COLUMNS_FUND

    for col in rank_cols:
        if col in feature_matrix.columns:
            feature_matrix[f"{col}_rank"] = (
                feature_matrix.groupby(feature_matrix.index)[col]
                .rank(pct=True, method="average")
            )

    # Relative strength: individual return - cross-sectional median
    for period in [5, 20]:
        col = f"momentum_{period}"
        if col in feature_matrix.columns:
            median = feature_matrix.groupby(feature_matrix.index)[col].transform("median")
            feature_matrix[f"rs_{period}d"] = feature_matrix[col] - median

    return feature_matrix


def fetch_overview_data(
    tickers: List[str],
    cache_dir: str = "data/cross_section_cache/overview_cache",
) -> Dict[str, dict]:
    """Fetch Alpha Vantage OVERVIEW data for all tickers (with caching).

    Returns:
        {ticker: {"Sector": ..., "Industry": ..., ...}}
    """
    import requests

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        print("  [warn] ALPHAVANTAGE_API_KEY not set, using empty industry data")
        return {t: {"Sector": "Unknown", "Industry": "Unknown"} for t in tickers}

    result: Dict[str, dict] = {}

    for ticker in tickers:
        cache_file = cache_path / f"{ticker}_overview.json"

        # Use cache if exists (overview data rarely changes)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    result[ticker] = json.load(f)
                continue
            except Exception:
                pass

        try:
            url = "https://www.alphavantage.co/query"
            params = {"function": "OVERVIEW", "symbol": ticker, "apikey": api_key}
            resp = requests.get(url, params=params, timeout=30)
            overview = resp.json()
            if overview and "Sector" in overview:
                result[ticker] = overview
                with open(cache_file, "w") as f:
                    json.dump(overview, f, indent=2)
                print(f"  [overview] {ticker}: {overview.get('Sector', 'N/A')} / {overview.get('Industry', 'N/A')}")
            else:
                print(f"  [warn] {ticker}: OVERVIEW returned no sector data")
                result[ticker] = {"Sector": "Unknown", "Industry": "Unknown"}
            # Rate limiting
            time.sleep(0.8)
        except Exception as exc:
            print(f"  [error] {ticker}: OVERVIEW failed: {exc}")
            result[ticker] = {"Sector": "Unknown", "Industry": "Unknown"}

    return result


def build_industry_map(overview_data: Dict[str, dict]) -> Dict[str, int]:
    """Build industry -> integer mapping from OVERVIEW data.

    Args:
        overview_data: {ticker: {"Sector": ..., "Industry": ...}}

    Returns:
        {industry_name: integer_code}
    """
    industries = sorted(set(
        info.get("Industry", "Unknown")
        for info in overview_data.values()
    ))
    return {name: idx for idx, name in enumerate(industries)}


def assign_industry_codes(
    feature_matrix: pd.DataFrame,
    overview_data: Dict[str, dict],
    industry_map: Dict[str, int],
) -> pd.DataFrame:
    """Assign sector_code and industry_code columns to the feature matrix.

    Args:
        feature_matrix: DataFrame with 'ticker' column.
        overview_data: {ticker: {"Sector": ..., "Industry": ...}}
        industry_map: {industry_name: integer_code}
    """
    sector_codes = []
    industry_codes = []

    for ticker in feature_matrix["ticker"]:
        info = overview_data.get(ticker, {})
        sector = info.get("Sector", "Unknown")
        industry = info.get("Industry", "Unknown")
        sector_codes.append(SECTOR_MAP.get(sector, -1))
        industry_codes.append(industry_map.get(industry, -1))

    feature_matrix["sector_code"] = sector_codes
    feature_matrix["industry_code"] = industry_codes

    return feature_matrix


# ═══════════════════════════════════════════════════════════════════════════
# 5. Sample Weights
# ═══════════════════════════════════════════════════════════════════════════

def compute_sample_weights(dates: pd.DatetimeIndex, half_life_days: int = 252) -> np.ndarray:
    """Exponential decay weights: recent samples get higher weight."""
    days_ago = (dates.max() - dates).days.values.astype(float)
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return weights / weights.sum() * len(weights)  # Normalize to mean=1


# ═══════════════════════════════════════════════════════════════════════════
# 6. Purged K-Fold
# ═══════════════════════════════════════════════════════════════════════════

class PurgedKFold:
    """Purged K-Fold cross-validation for time series.

    Standard K-Fold but with a purge gap between train and validation sets
    to prevent label leakage from overlapping prediction windows.

    V2 (cross-sectional): splits by unique dates so that all tickers on the
    same day stay in the same fold, preventing cross-sectional information
    leakage.
    """

    def __init__(self, n_splits: int = 5, horizon_days: int = 5, embargo_days: int = 10):
        self.n_splits = n_splits
        self.horizon_days = horizon_days
        self.embargo_days = embargo_days
        self.gap = horizon_days + embargo_days

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              dates: Optional[np.ndarray] = None):
        """Yield (train_idx, val_idx) tuples with purged gap.

        Args:
            X: Feature matrix.
            y: Labels (unused, for API compatibility).
            dates: Array of date values aligned with X rows. If provided,
                   splitting is done by unique dates so all rows sharing
                   the same date are kept together (V2 cross-sectional mode).
                   If None, falls back to row-index splitting (V1 mode).
        """
        if dates is not None:
            yield from self._split_by_date(X, dates)
        else:
            yield from self._split_by_index(X)

    def _split_by_index(self, X: np.ndarray):
        """V1 mode: split by row index position."""
        n = len(X)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else n

            train_mask = np.ones(n, dtype=bool)
            train_mask[val_start:val_end] = False

            # Remove gap before validation
            purge_start = max(0, val_start - self.gap)
            train_mask[purge_start:val_start] = False

            # Remove gap after validation
            purge_end = min(n, val_end + self.gap)
            train_mask[val_end:purge_end] = False

            train_idx = np.where(train_mask)[0]
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx

    def _split_by_date(self, X: np.ndarray, dates: np.ndarray):
        """V2 mode: split by unique dates (cross-sectional safe).

        All rows sharing the same date are kept in the same fold.
        Train indices are shuffled to avoid bagging bias.
        """
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        fold_size = n_dates // self.n_splits

        for i in range(self.n_splits):
            val_date_start = i * fold_size
            val_date_end = val_date_start + fold_size if i < self.n_splits - 1 else n_dates

            val_dates = set(unique_dates[val_date_start:val_date_end])

            # Purge gap: remove dates near validation boundary
            gap_start = max(0, val_date_start - self.gap)
            gap_end = min(n_dates, val_date_end + self.gap)
            purge_dates = set(unique_dates[gap_start:val_date_start]) | \
                          set(unique_dates[val_date_end:gap_end])

            train_mask = np.array([
                d not in val_dates and d not in purge_dates for d in dates
            ])
            val_mask = np.array([d in val_dates for d in dates])

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            # Shuffle train indices to avoid bagging bias
            np.random.shuffle(train_idx)

            yield train_idx, val_idx


# ═══════════════════════════════════════════════════════════════════════════
# 7. Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train_lgb_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    horizon_days: int = 5,
    embargo_days: int = 10,
    dates: Optional[np.ndarray] = None,
    categorical_feature: Optional[List[str]] = None,
    lgb_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """Train LightGBM with Purged K-Fold CV and multi-seed averaging.

    For each fold, trains N_SEEDS models with different random seeds and
    averages their predictions. This stabilizes the noisy AUC estimates
    on small validation sets.

    Args:
        dates: If provided, PurgedKFold splits by date (V2 cross-sectional).
        categorical_feature: List of categorical feature names for LightGBM.
        lgb_params: LightGBM parameters (defaults to LGB_PARAMS if None).

    Returns:
        (final_booster, cv_metrics, oof_predictions)
        oof_predictions: out-of-fold predicted probabilities for calibration.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

    params = lgb_params or LGB_PARAMS
    cat_feature = categorical_feature or "auto"

    pkf = PurgedKFold(n_splits=n_splits, horizon_days=horizon_days, embargo_days=embargo_days)

    fold_metrics: List[Dict[str, Any]] = []
    best_iterations: List[int] = []

    # Collect out-of-fold predictions for Isotonic calibration & Conformal scores
    oof_predictions = np.full(len(y), np.nan)

    split_mode = "date-based" if dates is not None else "index-based"
    print(f"\n{'='*60}")
    print(f"  Purged K-Fold Cross-Validation ({n_splits} folds, gap={horizon_days + embargo_days}d, {N_SEEDS} seeds, {split_mode})")
    print(f"{'='*60}")

    for fold_i, (train_idx, val_idx) in enumerate(pkf.split(X, y, dates=dates)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weights[train_idx]

        # Multi-seed averaging: train N_SEEDS models and average predictions
        seed_preds = []
        seed_best_iters = []

        for seed_offset in range(N_SEEDS):
            seed = params["random_state"] + seed_offset * 1000
            seed_params = {**params, "random_state": seed, "bagging_seed": seed, "feature_fraction_seed": seed}

            train_set = lgb.Dataset(X_train, label=y_train, weight=w_train,
                                    feature_name=feature_names,
                                    categorical_feature=cat_feature)
            val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                                  categorical_feature=cat_feature, reference=train_set)

            if USE_EARLY_STOPPING:
                callbacks = [
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
                booster = lgb.train(
                    seed_params,
                    train_set,
                    num_boost_round=seed_params["n_estimators"],
                    valid_sets=[val_set],
                    valid_names=["val"],
                    callbacks=callbacks,
                )
                seed_best_iters.append(booster.best_iteration)
            else:
                booster = lgb.train(
                    seed_params,
                    train_set,
                    num_boost_round=seed_params["n_estimators"],
                    valid_sets=[val_set],
                    valid_names=["val"],
                    callbacks=[lgb.log_evaluation(period=0)],
                )
                seed_best_iters.append(seed_params["n_estimators"])

            seed_preds.append(booster.predict(X_val))

        # Average predictions across seeds
        val_pred = np.mean(seed_preds, axis=0)
        avg_best_iter_fold = int(np.mean(seed_best_iters))

        # Store OOF predictions for this fold
        oof_predictions[val_idx] = val_pred

        auc = roc_auc_score(y_val, val_pred)
        acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
        ll = log_loss(y_val, val_pred)

        fold_metrics.append({
            "fold": fold_i + 1,
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "best_iteration": avg_best_iter_fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
        })
        best_iterations.append(avg_best_iter_fold)

        print(f"  Fold {fold_i + 1}: AUC={auc:.4f}  Acc={acc:.4f}  LogLoss={ll:.4f}  BestIter={avg_best_iter_fold}  (train={len(train_idx)}, val={len(val_idx)})")

    # Aggregate CV metrics
    mean_auc = np.mean([m["auc"] for m in fold_metrics])
    std_auc = np.std([m["auc"] for m in fold_metrics])
    mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
    mean_ll = np.mean([m["log_loss"] for m in fold_metrics])
    avg_best_iter = int(np.mean(best_iterations))

    print(f"\n  CV Summary: AUC={mean_auc:.4f}\u00b1{std_auc:.4f}  Acc={mean_acc:.4f}  LogLoss={mean_ll:.4f}")
    print(f"  Avg best iteration: {avg_best_iter}")

    cv_metrics = {
        "mean_auc": round(float(mean_auc), 4),
        "std_auc": round(float(std_auc), 4),
        "mean_accuracy": round(float(mean_acc), 4),
        "mean_log_loss": round(float(mean_ll), 4),
        "fold_details": fold_metrics,
    }

    # Train final model on all data
    final_num_rounds = max(avg_best_iter, 100)
    print(f"\n  Training final model on all {len(X)} samples (num_boost_round={final_num_rounds}) ...")
    full_train_set = lgb.Dataset(X, label=y, weight=sample_weights,
                                 feature_name=feature_names,
                                 categorical_feature=cat_feature)
    final_booster = lgb.train(
        params,
        full_train_set,
        num_boost_round=final_num_rounds,
    )

    return final_booster, cv_metrics, oof_predictions

# ═══════════════════════════════════════════════════════════════════════════
# 8. Probability Calibration & Uncertainty Quantification
# ═══════════════════════════════════════════════════════════════════════════

from utils.calibrator import TemperatureScalingCalibrator  # noqa: E402


def fit_isotonic_calibrator(
    oof_predictions: np.ndarray,
    y: np.ndarray,
) -> Optional["TemperatureScalingCalibrator"]:
    """Fit a Temperature Scaling calibrator from out-of-fold predictions.

    Temperature Scaling is preferred over Isotonic Regression and Platt Scaling
    because:
    - It preserves the ranking of raw probabilities (IC unchanged)
    - It has only 1 parameter → minimal overfitting risk
    - It extrapolates gracefully when OOS raw probabilities exceed the
      training range — a common occurrence in financial time-series

    Only valid (non-NaN) OOF predictions are used.
    """
    valid_mask = ~np.isnan(oof_predictions)
    n_valid = valid_mask.sum()
    if n_valid < 50:
        print(f"  [warn] Only {n_valid} valid OOF predictions, skipping calibration")
        return None

    raw = oof_predictions[valid_mask]
    labels = y[valid_mask]

    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(raw, labels)

    # Report calibration effect
    calibrated = calibrator.predict(raw)
    print(f"  Temperature Scaling calibration fitted on {n_valid} OOF samples")
    print(f"    Temperature T = {calibrator.temperature:.4f}  ({'softening' if calibrator.temperature > 1 else 'sharpening'})")
    print(f"    Raw prob range:        [{raw.min():.4f}, {raw.max():.4f}]")
    print(f"    Calibrated prob range:  [{calibrated.min():.4f}, {calibrated.max():.4f}]")
    print(f"    Raw mean:      {raw.mean():.4f}  →  Calibrated mean: {calibrated.mean():.4f}")

    # Report extrapolation behavior at extreme values
    test_extremes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    extreme_cal = calibrator.predict(test_extremes)
    print(f"    Extrapolation check: {dict(zip(test_extremes.round(1), extreme_cal.round(4)))}")

    return calibrator


def compute_conformal_scores(
    oof_predictions: np.ndarray,
    y: np.ndarray,
    calibrator: Optional[IsotonicRegression] = None,
) -> Dict[str, Any]:
    """Compute conformal nonconformity scores from OOF predictions.

    The nonconformity score measures how "surprising" each prediction is:
      - For positive samples (y=1): score = 1 - prob_up
      - For negative samples (y=0): score = prob_up

    Higher scores = worse predictions. The quantiles of these scores are
    used at inference time to build prediction sets with coverage guarantees.

    If a calibrator is provided, scores are computed on calibrated probabilities
    for better coverage accuracy.
    """
    valid_mask = ~np.isnan(oof_predictions)
    n_valid = valid_mask.sum()
    if n_valid < 50:
        print(f"  [warn] Only {n_valid} valid OOF predictions, skipping conformal scores")
        return {}

    probs = oof_predictions[valid_mask].copy()
    labels = y[valid_mask]

    # Use calibrated probabilities if available (better coverage)
    if calibrator is not None:
        probs = calibrator.predict(probs)

    # Compute nonconformity scores
    scores = np.where(labels == 1, 1.0 - probs, probs)

    # Compute quantiles for different confidence levels
    quantiles = {}
    for alpha_name, alpha in [("q80", 0.80), ("q85", 0.85), ("q90", 0.90), ("q95", 0.95)]:
        quantiles[alpha_name] = float(np.quantile(scores, alpha))

    print(f"  Conformal scores computed on {n_valid} OOF samples")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    Quantiles: q80={quantiles['q80']:.4f}  q85={quantiles['q85']:.4f}  "
          f"q90={quantiles['q90']:.4f}  q95={quantiles['q95']:.4f}")

    return {
        "conformal_scores_quantiles": quantiles,
        "conformal_n_samples": int(n_valid),
        "conformal_score_mean": float(scores.mean()),
        "conformal_score_std": float(scores.std()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()

    # ── Configuration ────────────────────────────────────────────────────
    model_version = os.getenv("FORECAST_MODEL_VERSION", "v1")
    is_v2 = model_version == "v2"

    if is_v2:
        # V2: Load tickers from sp500_top100.json
        tickers_source = os.getenv("TRAIN_TICKERS_SOURCE", "sp500_top100")
        tickers_path = os.getenv("TRAIN_TICKERS_PATH", "data/sp500_top100.json")
        if tickers_source == "sp500_top100" and os.path.exists(tickers_path):
            with open(tickers_path, "r") as f:
                ticker_data = json.load(f)
            tickers = ticker_data.get("tickers", [])
        else:
            tickers_env = os.getenv("TRAIN_TICKERS", "AAPL")
            tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
    else:
        tickers_env = os.getenv("TRAIN_TICKERS", "AAPL")
        tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
    if not tickers:
        tickers = ["AAPL"]

    lookback_years = int(os.getenv("TRAIN_LOOKBACK_YEARS", "5"))
    horizon_days = int(os.getenv("FORECAST_HORIZON_DAYS", "5"))
    n_splits = int(os.getenv("TRAIN_CV_FOLDS", "10"))
    embargo_days = int(os.getenv("TRAIN_EMBARGO_DAYS", "10"))
    half_life = int(os.getenv("TRAIN_SAMPLE_WEIGHT_HALFLIFE", "252"))
    enable_regime = os.getenv("TRAIN_ENABLE_REGIME", "true").lower() in ("true", "1", "yes")
    enable_macro_fund = os.getenv("TRAIN_ENABLE_MACRO_FUND", "true").lower() in ("true", "1", "yes")
    train_end_date_str = os.getenv("TRAIN_END_DATE", "")
    train_end_date: Optional[pd.Timestamp] = None
    if train_end_date_str:
        try:
            train_end_date = pd.Timestamp(train_end_date_str)
            print(f"  [config] TRAIN_END_DATE={train_end_date.date()}")
        except Exception:
            print(f"  [warn] Invalid TRAIN_END_DATE='{train_end_date_str}', ignoring")

    if is_v2:
        model_path = os.getenv("FORECAST_LGB_MODEL_PATH", "data/forecast_model_v2.lgb")
        meta_path = os.getenv("FORECAST_LGB_META_PATH", "data/forecast_model_v2_meta.json")
        calibrator_path_str = os.getenv("FORECAST_CALIBRATOR_PATH", "data/forecast_calibrator_v2.pkl")
    else:
        model_path = os.getenv("FORECAST_LGB_MODEL_PATH", "data/forecast_model.lgb")
        meta_path = os.getenv("FORECAST_LGB_META_PATH", "data/forecast_model_meta.json")
        calibrator_path_str = os.getenv("FORECAST_CALIBRATOR_PATH", "data/forecast_calibrator.pkl")

    print("=" * 60)
    print(f"  LightGBM Forecast Model Training Pipeline {'(V2 Cross-Sectional)' if is_v2 else '(V1 Single-Ticker)'}")
    print("=" * 60)
    print(f"  Model version:  {model_version}")
    print(f"  Tickers:        {len(tickers)} tickers" if is_v2 else f"  Tickers:        {tickers}")
    print(f"  Lookback:       {lookback_years} years")
    print(f"  Horizon:        {horizon_days} days")
    print(f"  CV Folds:       {n_splits}")
    print(f"  Embargo:        {embargo_days} days")
    print(f"  Half-life:      {half_life} days")
    print(f"  Regime features: {'ON' if enable_regime else 'OFF'}")
    print(f"  Macro/Fund features: {'ON' if enable_macro_fund else 'OFF'}")
    print(f"  Train end date: {train_end_date.date() if train_end_date else 'now (latest)'}")
    print()

    # ── Step 1: Download data ────────────────────────────────────────────
    print("[Step 1] Downloading training data ...")
    raw_data = download_training_data(tickers)
    if not raw_data:
        raise SystemExit("No training data downloaded. Check API key and network.")

    # Download SPY data for excess-return labels
    print("  Downloading SPY data for excess return labels ...")
    spy_data_dict = download_training_data(["SPY"])
    spy_data: Optional[pd.DataFrame] = spy_data_dict.get("SPY")
    if spy_data is not None and not spy_data.empty:
        print(f"  [ok] SPY: {len(spy_data)} rows")
    else:
        print("  [warn] SPY data not available, will use absolute return labels")
        spy_data = None

    # ── Step 2: Feature engineering ──────────────────────────────────────
    print("\n[Step 2] Computing base features ...")
    all_frames: List[pd.DataFrame] = []

    # V2: Fetch industry info for all tickers
    overview_data: Dict[str, dict] = {}
    industry_map: Dict[str, int] = {}
    if is_v2:
        print("\n[Step 2a] Fetching OVERVIEW data for industry encoding ...")
        try:
            overview_data = fetch_overview_data(tickers)
            industry_map = build_industry_map(overview_data)
            print(f"  Sectors: {len(set(info.get('Sector', 'Unknown') for info in overview_data.values()))} unique")
            print(f"  Industries: {len(industry_map)} unique")
        except Exception as exc:
            print(f"  [warn] OVERVIEW fetch failed: {exc}, using empty industry data")

    for ticker, data in raw_data.items():
        # Filter to lookback window
        anchor = train_end_date if train_end_date else pd.Timestamp.now()
        cutoff = anchor - pd.DateOffset(years=lookback_years)
        data = data[data.index >= cutoff]
        if train_end_date:
            data = data[data.index <= train_end_date]

        if len(data) < 60:
            print(f"  [skip] {ticker}: only {len(data)} rows (need >= 60)")
            continue

        frame = compute_base_features(data)
        frame = build_labels(frame, horizon_days=horizon_days, spy_data=spy_data)
        frame["ticker"] = ticker

        # Drop rows with NaN in features or label
        required_cols = BASE_FEATURE_COLUMNS + ["label"]
        frame = frame.dropna(subset=required_cols)

        print(f"  [ok] {ticker}: {len(frame)} rows after feature computation")
        all_frames.append(frame)

    if not all_frames:
        raise SystemExit("No valid feature rows generated.")

    feature_matrix = pd.concat(all_frames, axis=0).sort_index()
    print(f"\n  Total feature matrix: {len(feature_matrix)} rows, {feature_matrix['ticker'].nunique()} tickers")

    # V2: Assign industry codes
    if is_v2 and overview_data:
        print("\n[Step 2b] Assigning industry codes ...")
        feature_matrix = assign_industry_codes(feature_matrix, overview_data, industry_map)
        n_valid_sector = (feature_matrix["sector_code"] >= 0).sum()
        n_valid_industry = (feature_matrix["industry_code"] >= 0).sum()
        print(f"  sector_code: {n_valid_sector}/{len(feature_matrix)} valid")
        print(f"  industry_code: {n_valid_industry}/{len(feature_matrix)} valid")

    # ── Step 3: Regime features (Phase 2) ────────────────────────────────
    if enable_regime:
        print("\n[Step 3] Computing regime features (Phase 2) ...")
        # Load transition matrix for regime smoothing if available
        regime_tm_path = os.getenv(
            "REGIME_TRANSITION_MATRIX_PATH",
            "data/regime_transition_matrix.json",
        )
        regime_transition_matrix: Optional[Dict[str, Dict[str, float]]] = None
        if os.path.exists(regime_tm_path):
            with open(regime_tm_path, "r") as f:
                regime_transition_matrix = json.load(f)
            print(f"  Loaded transition matrix from {regime_tm_path} ({len(regime_transition_matrix)} states)")
        else:
            print("  No transition matrix found, using raw regime states")
        feature_matrix = compute_regime_features(feature_matrix, transition_matrix=regime_transition_matrix)
        feature_columns = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
        print(f"  Feature dimensions: {len(feature_columns)} ({len(BASE_FEATURE_COLUMNS)} base + {len(REGIME_FEATURE_COLUMNS)} regime)")
    else:
        feature_columns = BASE_FEATURE_COLUMNS
        print(f"\n[Step 3] Regime features SKIPPED (Phase 1 only)")
        print(f"  Feature dimensions: {len(feature_columns)}")

    # ── Step 3.5: Macro/Fundamental features (historical time-series) ────
    macro_fund_cols_used: List[str] = []
    if enable_macro_fund:
        print("\n[Step 3.5] Fetching historical macro/fundamental features ...")
        provider = MacroFundamentalFeatureProvider(verbose=True)

        # Determine date range from the feature matrix
        fm_start = feature_matrix.index.min().to_pydatetime()
        fm_end = feature_matrix.index.max().to_pydatetime()
        print(f"  Date range: {fm_start.date()} to {fm_end.date()}")

        if is_v2:
            # V2: Fetch macro data once (shared), then per-ticker fundamentals
            print("  [V2] Fetching macro features (shared across all tickers) ...")
            macro_only_df = provider.extract_macro_only_historical(
                start_date=fm_start,
                end_date=fm_end,
            )

            # Merge macro features into feature_matrix (same for all tickers)
            feature_matrix = feature_matrix.sort_index()
            feature_matrix.index = pd.to_datetime(feature_matrix.index)

            if not macro_only_df.empty:
                macro_only_df = macro_only_df.sort_index()
                macro_only_df.index = pd.to_datetime(macro_only_df.index)

                fm_reset = feature_matrix.reset_index()
                m_reset = macro_only_df.reset_index()
                fm_reset = fm_reset.rename(columns={fm_reset.columns[0]: "_merge_date"})
                m_reset = m_reset.rename(columns={m_reset.columns[0]: "_merge_date"})

                merged_macro = pd.merge_asof(
                    fm_reset.sort_values("_merge_date"),
                    m_reset.sort_values("_merge_date"),
                    on="_merge_date",
                    direction="backward",
                )
                merged_macro = merged_macro.set_index("_merge_date")
                merged_macro.index.name = feature_matrix.index.name

                from utils.macro_fundamental_provider import MACRO_FEATURE_COLUMNS as _MACRO_COLS
                for col in _MACRO_COLS:
                    if col in merged_macro.columns:
                        feature_matrix[col] = merged_macro[col].values
                print(f"  [V2] Macro features merged into feature_matrix")

            # Process each ticker's fundamentals independently (no macro re-fetch)
            unique_tickers = feature_matrix["ticker"].unique()
            print(f"  [V2] Fetching per-ticker fundamentals for {len(unique_tickers)} tickers ...")
            ticker_fund_cache: Dict[str, pd.DataFrame] = {}

            for t_idx, t in enumerate(unique_tickers):
                if (t_idx + 1) % 10 == 0 or t_idx == 0:
                    print(f"    Processing ticker {t_idx + 1}/{len(unique_tickers)}: {t} ...")
                try:
                    fund_df = provider.extract_fundamental_only_historical(
                        stock_symbol=t,
                        start_date=fm_start,
                        end_date=fm_end,
                    )
                    ticker_fund_cache[t] = fund_df
                except Exception as exc:
                    print(f"    [warn] {t}: fundamental fetch failed: {exc}")

            # Merge per-ticker fundamentals: for each ticker's rows, use that ticker's data
            _INTERMEDIATE_COLUMNS = [
                "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
                "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
            ]
            from utils.macro_fundamental_provider import FUNDAMENTAL_FEATURE_COLUMNS as _FUND_COLS
            _FUND_AND_INTERMEDIATE = _FUND_COLS + _INTERMEDIATE_COLUMNS

            for t in unique_tickers:
                t_mask = feature_matrix["ticker"] == t
                t_rows = feature_matrix.loc[t_mask]

                # Get this ticker's fundamental data
                fund_df = ticker_fund_cache.get(t)
                if fund_df is not None and not fund_df.empty:
                    fund_df = fund_df.sort_index()
                    fund_df.index = pd.to_datetime(fund_df.index)

                    # merge_asof for this ticker's rows
                    t_reset = t_rows.reset_index()
                    f_reset = fund_df.reset_index()
                    t_reset = t_reset.rename(columns={t_reset.columns[0]: "_merge_date"})
                    f_reset = f_reset.rename(columns={f_reset.columns[0]: "_merge_date"})

                    # Drop columns that will come from f_reset to avoid
                    # _x/_y suffix collision in merge_asof.  After the 1st
                    # ticker is processed, feature_matrix (and therefore
                    # t_rows / t_reset) already contains these columns,
                    # which would clash with the same columns in f_reset.
                    _cols_from_fund = [c for c in _FUND_AND_INTERMEDIATE if c in t_reset.columns]
                    if _cols_from_fund:
                        t_reset = t_reset.drop(columns=_cols_from_fund)

                    merged = pd.merge_asof(
                        t_reset.sort_values("_merge_date"),
                        f_reset.sort_values("_merge_date"),
                        on="_merge_date",
                        direction="backward",
                    )
                    merged = merged.set_index("_merge_date")

                    for col in _FUND_AND_INTERMEDIATE:
                        if col in merged.columns:
                            feature_matrix.loc[t_mask, col] = merged[col].values

            # Compute price-dependent features per ticker
            print("  [V2] Computing price-dependent fundamental features per ticker ...")
            close = pd.to_numeric(feature_matrix["Close"], errors="coerce")

            # P/E ratio
            if "_ttm_eps" in feature_matrix.columns:
                ttm_eps = feature_matrix["_ttm_eps"]
                valid = ttm_eps.notna() & (ttm_eps.abs() > 0.01)
                feature_matrix.loc[valid, "pe_ratio"] = (close[valid] / ttm_eps[valid]).values

            # P/B ratio
            if "_total_equity" in feature_matrix.columns and "_shares_outstanding" in feature_matrix.columns:
                equity = feature_matrix["_total_equity"]
                shares = feature_matrix["_shares_outstanding"]
                valid = equity.notna() & shares.notna() & (shares > 0)
                bvps = equity[valid] / shares[valid]
                bvps_valid = bvps.abs() > 0.01
                final_mask = valid.copy()
                final_mask[valid] = bvps_valid.values
                feature_matrix.loc[final_mask, "pb_ratio"] = (
                    close[final_mask].values / bvps[bvps_valid].values
                )

            # P/S ratio
            if "_ttm_revenue" in feature_matrix.columns and "_shares_outstanding" in feature_matrix.columns:
                ttm_rev = feature_matrix["_ttm_revenue"]
                shares = feature_matrix["_shares_outstanding"]
                valid = ttm_rev.notna() & shares.notna() & (shares > 0) & (ttm_rev > 0)
                rps = ttm_rev[valid] / shares[valid]
                feature_matrix.loc[valid, "ps_ratio"] = (close[valid] / rps).values

            # EV/EBITDA
            if all(c in feature_matrix.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                shares = feature_matrix["_shares_outstanding"]
                ttm_ebitda = feature_matrix["_ttm_ebitda"]
                total_liab = feature_matrix["_total_liabilities"].fillna(0)
                cash = feature_matrix["_cash"].fillna(0)
                valid = shares.notna() & (shares > 0) & ttm_ebitda.notna() & (ttm_ebitda.abs() > 0)
                market_cap = close[valid] * shares[valid]
                ev = market_cap + total_liab[valid] - cash[valid]
                feature_matrix.loc[valid, "ev_ebitda"] = (ev / ttm_ebitda[valid]).values

            # Beta: compute per-ticker rolling beta vs SPY
            try:
                spy_cache_file = Path("data/training_cache/SPY_daily.csv")
                if spy_cache_file.exists():
                    spy_df = pd.read_csv(spy_cache_file, index_col=0, parse_dates=True)
                else:
                    spy_df = get_historical_data("SPY", interval="daily", days=None, outputsize="full")
                if not spy_df.empty:
                    spy_close = pd.to_numeric(spy_df["Close"], errors="coerce")
                    spy_returns = spy_close.pct_change()
                    for t in unique_tickers:
                        t_mask = feature_matrix["ticker"] == t
                        t_close = pd.to_numeric(feature_matrix.loc[t_mask, "Close"], errors="coerce")
                        t_returns = t_close.pct_change()
                        aligned = pd.DataFrame({"stock": t_returns, "spy": spy_returns}).dropna()
                        if len(aligned) > 60:
                            rolling_cov = aligned["stock"].rolling(252, min_periods=60).cov(aligned["spy"])
                            rolling_var = aligned["spy"].rolling(252, min_periods=60).var()
                            rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
                            feature_matrix.loc[t_mask, "beta"] = rolling_beta.reindex(
                                feature_matrix.loc[t_mask].index
                            ).ffill().values
            except Exception as exc:
                print(f"    beta: computation failed: {exc}")

            # PEG ratio
            if "pe_ratio" in feature_matrix.columns and "earnings_growth_yoy" in feature_matrix.columns:
                pe = feature_matrix["pe_ratio"]
                eg = feature_matrix["earnings_growth_yoy"]
                eg_pct = eg * 100
                valid = pe.notna() & eg_pct.notna() & (eg_pct.abs() > 1.0) & (pe > 0)
                feature_matrix.loc[valid, "peg_ratio"] = (pe[valid] / eg_pct[valid]).values

            # Drop intermediate columns; leave NaN for LightGBM native handling
            for col in _INTERMEDIATE_COLUMNS:
                if col in feature_matrix.columns:
                    feature_matrix.drop(columns=[col], inplace=True)

            print(f"  [V2] Per-ticker fundamental features computed")

        else:
            # V1: Original single-ticker logic
            mf_hist_df = provider.extract_historical(
                stock_symbol=tickers[0] if tickers else "",
                start_date=fm_start,
                end_date=fm_end,
            )

            # Intermediate columns for price-dependent feature computation
            _INTERMEDIATE_COLUMNS = [
                "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
                "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
            ]
            _ALL_MF_COLS = ALL_MACRO_FUNDAMENTAL_COLUMNS + _INTERMEDIATE_COLUMNS

            if not mf_hist_df.empty:
                # Merge historical macro/fundamental features into feature_matrix
                # using merge_asof to align by nearest preceding date
                feature_matrix = feature_matrix.sort_index()
                mf_hist_df = mf_hist_df.sort_index()

                # Ensure both indices are DatetimeIndex for merge_asof
                feature_matrix.index = pd.to_datetime(feature_matrix.index)
                mf_hist_df.index = pd.to_datetime(mf_hist_df.index)

                # Use merge_asof: for each row in feature_matrix, find the most
                # recent macro/fundamental data point (backward direction)
                fm_reset = feature_matrix.reset_index()
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
                merged.index.name = feature_matrix.index.name

                # Copy merged macro/fundamental + intermediate columns back
                for col in _ALL_MF_COLS:
                    if col in merged.columns:
                        feature_matrix[col] = merged[col].values

                # ── Compute price-dependent features using existing Close price ──
                close = pd.to_numeric(feature_matrix["Close"], errors="coerce")
                print("  [Step 3.5a] Computing price-dependent fundamental features ...")

                # P/E ratio = Close / TTM EPS
                if "_ttm_eps" in feature_matrix.columns:
                    ttm_eps = feature_matrix["_ttm_eps"]
                    valid = ttm_eps.notna() & (ttm_eps.abs() > 0.01)
                    feature_matrix.loc[valid, "pe_ratio"] = close[valid] / ttm_eps[valid]
                    n_pe = valid.sum()
                    print(f"    pe_ratio: computed for {n_pe} rows")

                # P/B ratio = Close / (Total Equity / Shares Outstanding)
                if "_total_equity" in feature_matrix.columns and "_shares_outstanding" in feature_matrix.columns:
                    equity = feature_matrix["_total_equity"]
                    shares = feature_matrix["_shares_outstanding"]
                    valid = equity.notna() & shares.notna() & (shares > 0)
                    bvps = equity[valid] / shares[valid]
                    bvps_valid = bvps.abs() > 0.01
                    feature_matrix.loc[bvps_valid.index[bvps_valid], "pb_ratio"] = (
                        close[bvps_valid.index[bvps_valid]] / bvps[bvps_valid]
                    )
                    n_pb = bvps_valid.sum()
                    print(f"    pb_ratio: computed for {n_pb} rows")

                # P/S ratio = Close / (TTM Revenue / Shares Outstanding)
                if "_ttm_revenue" in feature_matrix.columns and "_shares_outstanding" in feature_matrix.columns:
                    ttm_rev = feature_matrix["_ttm_revenue"]
                    shares = feature_matrix["_shares_outstanding"]
                    valid = ttm_rev.notna() & shares.notna() & (shares > 0) & (ttm_rev > 0)
                    rps = ttm_rev[valid] / shares[valid]
                    feature_matrix.loc[valid, "ps_ratio"] = close[valid] / rps
                    n_ps = valid.sum()
                    print(f"    ps_ratio: computed for {n_ps} rows")

                # EV/EBITDA
                if all(c in feature_matrix.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                    shares = feature_matrix["_shares_outstanding"]
                    ttm_ebitda = feature_matrix["_ttm_ebitda"]
                    total_liab = feature_matrix["_total_liabilities"].fillna(0)
                    cash = feature_matrix["_cash"].fillna(0)
                    valid = shares.notna() & (shares > 0) & ttm_ebitda.notna() & (ttm_ebitda.abs() > 0)
                    market_cap = close[valid] * shares[valid]
                    ev = market_cap + total_liab[valid] - cash[valid]
                    feature_matrix.loc[valid, "ev_ebitda"] = ev / ttm_ebitda[valid]
                    n_ev = valid.sum()
                    print(f"    ev_ebitda: computed for {n_ev} rows")

                # Beta
                stock_returns = close.pct_change()
                try:
                    spy_cache_file = Path("data/training_cache/SPY_daily.csv")
                    if spy_cache_file.exists():
                        print("    beta: loading SPY from training cache ...")
                        spy_data = pd.read_csv(spy_cache_file, index_col=0, parse_dates=True)
                    else:
                        print("    beta: fetching SPY data (will be cached) ...")
                        spy_data = get_historical_data("SPY", interval="daily", days=None, outputsize="full")
                        if not spy_data.empty:
                            spy_cache_file.parent.mkdir(parents=True, exist_ok=True)
                            spy_data.to_csv(spy_cache_file)
                    if not spy_data.empty:
                        spy_close = pd.to_numeric(spy_data["Close"], errors="coerce")
                        spy_returns = spy_close.pct_change()
                        aligned = pd.DataFrame({"stock": stock_returns, "spy": spy_returns}).dropna()
                        if len(aligned) > 60:
                            rolling_cov = aligned["stock"].rolling(252, min_periods=60).cov(aligned["spy"])
                            rolling_var = aligned["spy"].rolling(252, min_periods=60).var()
                            rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
                            feature_matrix["beta"] = rolling_beta.reindex(feature_matrix.index)
                            feature_matrix["beta"] = feature_matrix["beta"].ffill()
                            n_beta = feature_matrix["beta"].notna().sum()
                            print(f"    beta: computed rolling 252d for {n_beta} rows")
                        else:
                            print(f"    beta: insufficient aligned data ({len(aligned)} rows)")
                    else:
                        print("    beta: SPY data not available")
                except Exception as exc:
                    print(f"    beta: computation failed: {exc}")

                # PEG ratio
                if "pe_ratio" in feature_matrix.columns and "earnings_growth_yoy" in feature_matrix.columns:
                    pe = feature_matrix["pe_ratio"]
                    eg = feature_matrix["earnings_growth_yoy"]
                    eg_pct = eg * 100
                    valid = pe.notna() & eg_pct.notna() & (eg_pct.abs() > 1.0) & (pe > 0)
                    feature_matrix.loc[valid, "peg_ratio"] = pe[valid] / eg_pct[valid]
                    n_peg = valid.sum()
                    print(f"    peg_ratio: computed for {n_peg} rows")

                # Dividend yield
                if "dividend_yield" in feature_matrix.columns and feature_matrix["dividend_yield"].isna().all():
                    print("    dividend_yield: no historical data available, leaving as NaN")

                # Drop intermediate columns; leave NaN for LightGBM native handling
                for col in _INTERMEDIATE_COLUMNS:
                    if col in feature_matrix.columns:
                        feature_matrix.drop(columns=[col], inplace=True)
            else:
                print("  [warn] Historical macro/fundamental data is empty, leaving as NaN")
                for col in ALL_MACRO_FUNDAMENTAL_COLUMNS:
                    feature_matrix[col] = np.nan

        # Only include columns that have non-zero variance
        for col in MACRO_FUNDAMENTAL_FEATURE_COLUMNS:
            if col in feature_matrix.columns:
                col_std = feature_matrix[col].std()
                if col_std > 1e-10:
                    macro_fund_cols_used.append(col)

        feature_columns = feature_columns + macro_fund_cols_used
        n_mf = len(macro_fund_cols_used)
        n_mf_total = len(MACRO_FUNDAMENTAL_FEATURE_COLUMNS)
        print(f"  Macro/Fund features with variance: {n_mf}/{n_mf_total} columns")
        print(f"  Total feature dimensions: {len(feature_columns)}")

        # Log sample values for verification
        if n_mf > 0 and len(feature_matrix) > 0:
            sample_idxs = [0, len(feature_matrix) // 2, len(feature_matrix) - 1]
            print(f"  Sample values (first/mid/last):")
            for col in macro_fund_cols_used[:5]:  # Show first 5 columns
                vals = [float(feature_matrix[col].iloc[i]) for i in sample_idxs]
                print(f"    {col}: {vals[0]:.4f} / {vals[1]:.4f} / {vals[2]:.4f}")
            if n_mf > 5:
                print(f"    ... and {n_mf - 5} more columns")
    else:
        print("\n[Step 3.5] Macro/Fundamental features SKIPPED")

    # ── Step 3.7: Cross-sectional features (V2 only) ─────────────────────
    rank_feature_columns: List[str] = []
    categorical_features: List[str] = []
    if is_v2:
        print("\n[Step 3.7] Computing cross-sectional features (V2) ...")
        feature_matrix = compute_cross_sectional_features(feature_matrix)

        # Collect rank feature columns that were actually created
        all_rank_cols = RANK_COLUMNS_TECH + RANK_COLUMNS_FUND
        for col in all_rank_cols:
            rank_col = f"{col}_rank"
            if rank_col in feature_matrix.columns:
                rank_feature_columns.append(rank_col)

        # Relative strength columns
        rs_cols = [c for c in feature_matrix.columns if c.startswith("rs_") and c.endswith("d")]
        rank_feature_columns.extend(rs_cols)

        # Add categorical features
        if "sector_code" in feature_matrix.columns:
            categorical_features.append("sector_code")
        if "industry_code" in feature_matrix.columns:
            categorical_features.append("industry_code")

        feature_columns = feature_columns + rank_feature_columns + categorical_features
        print(f"  Rank features: {len(rank_feature_columns)}")
        print(f"  Categorical features: {categorical_features}")
        print(f"  Total feature dimensions: {len(feature_columns)}")

    # ── Step 3.9: Cross-sectional label refinement (V2 only) ─────────────
    if is_v2 and "excess_return" in feature_matrix.columns:
        print("\n[Step 3.9] Refining labels with cross-sectional quantiles ...")
        feature_matrix = assign_cross_sectional_labels(
            feature_matrix, upper_quantile=0.6, lower_quantile=0.4
        )

    # ── Step 4: Sample weights ───────────────────────────────────────────
    print("\n[Step 4] Computing sample weights ...")
    sample_weights = compute_sample_weights(feature_matrix.index, half_life_days=half_life)
    print(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # ── Step 5 & 6: Purged K-Fold CV + Final model ──────────────────────
    X = feature_matrix[feature_columns].values.astype(np.float64)
    y = feature_matrix["label"].values.astype(np.float64)

    # Replace inf with NaN (LightGBM handles NaN natively, but not inf)
    X = np.where(np.isinf(X), np.nan, X)

    # V2: prepare dates array for date-based PurgedKFold
    dates_array = None
    if is_v2:
        dates_array = feature_matrix.index.values

    print(f"\n[Step 5-6] Training LightGBM ...")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Class balance: {y.mean():.3f} positive ({y.sum():.0f}/{len(y)})")

    final_model, cv_metrics, oof_predictions = train_lgb_model(
        X, y, sample_weights,
        feature_names=feature_columns,
        n_splits=n_splits,
        horizon_days=horizon_days,
        embargo_days=embargo_days,
        dates=dates_array,
        categorical_feature=categorical_features if categorical_features else None,
        lgb_params=LGB_PARAMS_V2 if is_v2 else LGB_PARAMS,
    )

    # ── Step 6.5: Platt Scaling calibration + Conformal scores ─────────────
    print(f"\n[Step 6.5] Probability calibration & uncertainty quantification ...")
    calibrator_path = Path(calibrator_path_str)

    calibrator = fit_isotonic_calibrator(oof_predictions, y)
    if calibrator is not None:
        calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"  Calibrator saved to: {calibrator_path}")
    else:
        print(f"  [warn] No calibrator fitted")

    conformal_info = compute_conformal_scores(oof_predictions, y, calibrator)
    if conformal_info:
        print(f"  Conformal prediction ready")

    # ── Step 7: Save model ───────────────────────────────────────────────
    print(f"\n[Step 7] Saving model ...")
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_file))
    print(f"  Model saved to: {model_file}")

    # ── Step 8: Save metadata ────────────────────────────────────────────
    print(f"\n[Step 8] Saving metadata ...")
    date_range = [
        feature_matrix.index.min().strftime("%Y-%m-%d"),
        feature_matrix.index.max().strftime("%Y-%m-%d"),
    ]

    # Determine label type from feature matrix
    if is_v2 and spy_data is not None:
        label_type = "cross_sectional_quantile"
    elif spy_data is not None:
        label_type = "excess_return"
    else:
        label_type = "absolute_return"

    if is_v2:
        meta = {
            "version": "2.0.0",
            "model_type": "cross_sectional",
            "model_path": str(model_path),
            "label_type": label_type,
            "benchmark": "SPY" if label_type == "excess_return" else None,
            "calibrator_path": str(calibrator_path) if calibrator is not None else None,
            "calibrated": calibrator is not None,
            **conformal_info,
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trainer": "pipelines/train_forecast_model.py",
            "target_horizon_days": horizon_days,
            "n_tickers": len(tickers),
            "ticker_source": os.getenv("TRAIN_TICKERS_PATH", "data/sp500_top100.json"),
            "tickers": tickers,
            "feature_columns": BASE_FEATURE_COLUMNS,
            "rank_feature_columns": rank_feature_columns,
            "regime_features": REGIME_FEATURE_COLUMNS if enable_regime else [],
            "macro_fundamental_features": macro_fund_cols_used if enable_macro_fund else [],
            "categorical_features": categorical_features,
            "sector_map": SECTOR_MAP,
            "industry_map": industry_map,
            "cv_metrics": cv_metrics,
            "training_info": {
                "tickers": tickers,
                "tickers_count": len(tickers),
                "total_samples": len(feature_matrix),
                "date_range": date_range,
                "class_balance": {
                    "positive": round(float(y.mean()), 4),
                    "negative": round(float(1 - y.mean()), 4),
                },
                "best_iteration": int(np.mean([f["best_iteration"] for f in cv_metrics["fold_details"]])),
                "sample_weight_half_life": half_life,
                "data_source": "alpha_vantage_premium",
                "regime_features_enabled": enable_regime,
                "regime_smoothing_enabled": regime_transition_matrix is not None if enable_regime else False,
                "macro_fundamental_features_enabled": enable_macro_fund,
                "macro_fundamental_features_count": len(macro_fund_cols_used) if enable_macro_fund else 0,
                "cross_sectional_rank_features_count": len(rank_feature_columns),
                "categorical_features_count": len(categorical_features),
            },
            "lgb_params": LGB_PARAMS_V2,
        }
    else:
        meta = {
            "version": 4,
            "model_type": "lightgbm",
            "model_path": str(model_path),
            "label_type": label_type,
            "benchmark": "SPY" if label_type == "excess_return" else None,
            "calibrator_path": str(calibrator_path) if calibrator is not None else None,
            "calibrated": calibrator is not None,
            **conformal_info,
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trainer": "pipelines/train_forecast_model.py",
            "target_horizon_days": horizon_days,
            "tickers": tickers,
            "feature_columns": BASE_FEATURE_COLUMNS,
            "regime_features": REGIME_FEATURE_COLUMNS if enable_regime else [],
            "macro_fundamental_features": macro_fund_cols_used if enable_macro_fund else [],
            "cv_metrics": cv_metrics,
            "training_info": {
                "tickers": tickers,
                "tickers_count": len(tickers),
                "total_samples": len(feature_matrix),
                "date_range": date_range,
                "class_balance": {
                    "positive": round(float(y.mean()), 4),
                    "negative": round(float(1 - y.mean()), 4),
                },
                "best_iteration": int(np.mean([f["best_iteration"] for f in cv_metrics["fold_details"]])),
                "sample_weight_half_life": half_life,
                "data_source": "alpha_vantage_premium",
                "regime_features_enabled": enable_regime,
                "regime_smoothing_enabled": regime_transition_matrix is not None if enable_regime else False,
                "macro_fundamental_features_enabled": enable_macro_fund,
                "macro_fundamental_features_count": len(macro_fund_cols_used) if enable_macro_fund else 0,
            },
        }

    meta_file = Path(meta_path)
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Metadata saved to: {meta_file}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete! {'(V2 Cross-Sectional)' if is_v2 else '(V1 Single-Ticker)'}")
    print(f"{'='*60}")
    print(f"  Tickers:       {len(tickers)}")
    print(f"  Samples:       {len(feature_matrix)}")
    feat_desc_parts = ["base"]
    if is_v2 and rank_feature_columns:
        feat_desc_parts.append(f"rank({len(rank_feature_columns)})")
    if is_v2 and categorical_features:
        feat_desc_parts.append(f"categorical({len(categorical_features)})")
    if enable_regime:
        feat_desc_parts.append("regime")
    if enable_macro_fund and macro_fund_cols_used:
        feat_desc_parts.append(f"macro/fund({len(macro_fund_cols_used)})")
    print(f"  Features:      {len(feature_columns)} ({' + '.join(feat_desc_parts)})")
    print(f"  CV AUC:        {cv_metrics['mean_auc']:.4f} ± {cv_metrics['std_auc']:.4f}")
    print(f"  CV Accuracy:   {cv_metrics['mean_accuracy']:.4f}")
    print(f"  CV Log Loss:   {cv_metrics['mean_log_loss']:.4f}")
    print(f"  Model:         {model_file}")
    print(f"  Metadata:      {meta_file}")
    if calibrator is not None:
        print(f"  Calibrator:    {calibrator_path}")
    if conformal_info:
        qs = conformal_info.get("conformal_scores_quantiles", {})
        print(f"  Conformal q90: {qs.get('q90', 'N/A')}")
    print()


if __name__ == "__main__":
    main()
