#!/usr/bin/env python3
"""
Stage 3: RegimeAgent Integration — Ablation Backtest

Goal: Validate the impact of RegimeAgent through 4 independent influence paths,
      each controlled by a separate switch for ablation testing.

Influence Paths:
  Path ④ — Regime features → LightGBM (affects probability_up)
  Path ③ — Regime stop multiplier (affects stop-loss width)
  Path ② — Regime risk budget (max_position cap + confidence_floor gate)
  Path ① — Signal alignment (regime vs forecast direction conflict)

Ablation Rounds:
  v0: Stage 2 baseline (no RegimeAgent, regime features = 0)
  v1: + Path ④ only (real regime features → LightGBM)
  v2: + Path ④ + Path ③ (+ regime stop multiplier)
  v3: + Path ④ + Path ③ + Path ② (+ risk budget)
  v4: + All 4 paths (+ signal alignment)

Usage:
    python scripts/debug_stage3_regime.py --ticker AAPL --start 2023-01-01 --end 2025-01-01
    python scripts/debug_stage3_regime.py --ticker AAPL --start 2023-01-01 --end 2025-01-01 --rounds v0,v1
    python scripts/debug_stage3_regime.py --ticker AAPL --start 2023-01-01 --end 2025-01-01 --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Reuse Stage 1 utilities
from debug_stage1_forecast_only import (
    compute_features,
    load_model,
    score_features,
    analyze_backtest,
)

# Import RegimeAgent
from agents.regime_agent import RegimeAgent

# Import MacroFundamentalFeatureProvider for real macro data
from utils.macro_fundamental_provider import (
    MacroFundamentalFeatureProvider,
    MACRO_FEATURE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════════════════
# 0. Regime feature encoding (mirrors ForecastAgent._build_regime_feature_vector)
# ═══════════════════════════════════════════════════════════════════════════

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

_HEALTH_MAP = {
    "accelerating": 0,
    "steady": 1,
    "decelerating": 2,
    "exhausted": 3,
}


def encode_regime_features(regime: Dict[str, Any]) -> Dict[str, float]:
    """Encode regime dict into numeric features matching the LightGBM training pipeline."""
    state = regime.get("state", "range_bound")
    vol_regime = regime.get("volatility_regime", "normal")
    return {
        "regime_direction": float(_STATE_DIRECTION.get(state, 0)),
        "regime_volatility_ord": float(_VOL_REGIME_ORD.get(vol_regime, 1)),
        "trend_strength": float(regime.get("trend_strength", 0.0)),
        "vol_expanding": float(int(regime.get("vol_expanding", False))),
        "momentum_health_enc": float(
            _HEALTH_MAP.get(regime.get("momentum_health", "steady"), 1)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Score features WITH real regime features (Path ④)
# ═══════════════════════════════════════════════════════════════════════════

def score_features_with_regime(
    features: dict,
    regime: Dict[str, Any],
    model,
    meta: dict,
    calibrator,
    use_real_regime: bool = True,
    return_uncertainty: bool = False,
    macro_snapshot: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Score features with LightGBM, optionally injecting real regime features.

    When use_real_regime=True (Path ④ enabled), regime features come from
    RegimeAgent's actual output instead of neutral defaults (all zeros).

    Args:
        features: Base technical features from compute_features().
        regime: Regime dict from RegimeAgent.analyze()['regime'].
        model: LightGBM Booster.
        meta: Model metadata dict.
        calibrator: Isotonic calibrator (or None).
        use_real_regime: If True, use real regime features; if False, use zeros.
        return_uncertainty: If True, return uncertainty info as 3rd element.

    Returns:
        (raw_prob, calibrated_prob) or (raw_prob, calibrated_prob, uncertainty_info)
    """
    feature_cols = meta.get("feature_columns", [])
    regime_cols = meta.get("regime_features", [])
    macro_fund_cols = meta.get("macro_fundamental_features", [])
    rank_cols = meta.get("rank_feature_columns", [])
    cat_cols = meta.get("categorical_features", [])
    all_cols = feature_cols + regime_cols + macro_fund_cols + rank_cols + cat_cols

    row = {}
    # Base features
    for col in feature_cols:
        row[col] = features.get(col, 0.0)

    # Regime features: real or neutral
    if use_real_regime and regime_cols:
        regime_feats = encode_regime_features(regime)
        for col in regime_cols:
            row[col] = regime_feats.get(col, 0.0)
    else:
        for col in regime_cols:
            row[col] = 0.0  # neutral defaults

    # Macro/fundamental: use real historical data if available
    if macro_snapshot and macro_fund_cols:
        macro_feats = macro_snapshot.get("macro_features", {})
        fund_feats = macro_snapshot.get("fundamental_features", {})
        for col in macro_fund_cols:
            val = macro_feats.get(col) if macro_feats else None
            if val is None and fund_feats:
                val = fund_feats.get(col)
            row[col] = float(val) if val is not None else 0.0
    else:
        for col in macro_fund_cols:
            row[col] = 0.0
    # Rank features: median default
    for col in rank_cols:
        row[col] = 0.5
    # Categorical: default
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

    # Method 2: Conformal prediction set (uses raw_prob)
    quantiles = meta.get("conformal_scores_quantiles", {})
    threshold = quantiles.get("q90")
    if threshold is not None:
        prediction_set = []
        if (1.0 - raw_prob) <= threshold:
            prediction_set.append("up")
        if raw_prob <= threshold:
            prediction_set.append("down")
        uncertainty_info["prediction_set"] = prediction_set
        if len(prediction_set) == 2:
            uncertainty_info["is_uncertain"] = True
        elif len(prediction_set) == 0:
            uncertainty_info["is_uncertain"] = True

    return raw_prob, prob, uncertainty_info


# ═══════════════════════════════════════════════════════════════════════════
# 2. Regime-aware Risk Plan (supports ablation switches for Paths ①②③)
# ═══════════════════════════════════════════════════════════════════════════

# Kelly defaults
KELLY_AVG_WIN = 0.03
KELLY_AVG_LOSS = 0.02

# Position limits
MAX_POSITION_SIZE = 1.0
MIN_POSITION_THRESHOLD = 0.03

# Stop-loss / take-profit
RISK_REWARD_RATIO = 2.0
STOP_VOL_MULTIPLIER = 2.5
STOP_MIN = 0.01
STOP_MAX = 0.08

# Signal alignment thresholds (Path ①)
ALIGNMENT_REJECT_THRESHOLD = 0.4
# ALIGNMENT_REDUCE_THRESHOLD removed: ablation showed half-position hurts returns
# without improving risk-adjusted metrics. See v1b_B vs v1b_D analysis.
HIGH_CONFIDENCE_OVERRIDE = 0.35    # |prob - 0.5| > this → override reject (prob > 0.85 or < 0.15)

# Regime direction scores for signal alignment
REGIME_DIRECTION_SCORE: Dict[str, float] = {
    "strong_rally":  +1.0,
    "trending_up":   +0.7,
    "bottoming_out": +0.3,
    "range_bound":    0.0,
    "coiling":        0.0,
    "choppy":        -0.2,
    "topping_out":   -0.3,
    "trending_down": -0.7,
    "capitulation":  -1.0,
}

# Default regime risk budget (fallback)
_DEFAULT_REGIME_BUDGET = {
    "max_position": 0.5,
    "stop_multiplier": 1.0,
    "confidence_floor": 0.4,
}


def _load_regime_risk_budget() -> Dict[str, Dict[str, float]]:
    """Load regime risk budget from JSON file."""
    budget_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "regime_risk_budget.json"
    )
    try:
        with open(budget_path, "r") as f:
            budget = json.load(f)
        for _state, params in budget.items():
            for key in ("max_position", "stop_multiplier", "confidence_floor"):
                if key not in params:
                    params[key] = _DEFAULT_REGIME_BUDGET[key]
        return budget
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Load once at module level
_REGIME_RISK_BUDGET = _load_regime_risk_budget()


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


def compute_signal_alignment(regime_state: str, probability_up: float) -> float:
    """Compute alignment score between Regime and Forecast signals.

    Returns alignment ∈ [0, 1]: 1.0 = fully aligned, 0.0 = fully opposed.
    """
    regime_signal = REGIME_DIRECTION_SCORE.get(regime_state, 0.0)
    forecast_signal = (probability_up - 0.5) * 2.0
    alignment = 1.0 - abs(regime_signal - forecast_signal) / 2.0
    return alignment


def regime_risk_plan(
    action: str,
    probability_up: float,
    volatility_20: float,
    regime: Dict[str, Any],
    horizon_days: int = 5,
    uncertainty_info: dict | None = None,
    macro_snapshot: Optional[Dict[str, Any]] = None,
    # Ablation switches
    enable_signal_alignment: bool = False,   # Path ①
    enable_alignment_reject: bool | None = None,   # Path ①a (override)
    enable_alignment_reduce: bool | None = None,   # Path ①b (override)
    enable_confidence_override: bool = False,       # Path ①c (high-confidence override)
    enable_risk_budget: bool = False,        # Path ②
    enable_stop_multiplier: bool = False,    # Path ③
) -> dict:
    """Regime-aware risk plan with ablation switches for Paths ①②③.

    Path ④ (regime features → LightGBM) is handled upstream in scoring,
    not in this function.

    Args:
        action: 'buy', 'sell', or 'hold'.
        probability_up: Calibrated probability of upward move.
        volatility_20: 20-day annualized volatility.
        regime: Regime dict from RegimeAgent.
        horizon_days: Holding period in days.
        uncertainty_info: Conformal prediction set + tree dispersion.
        enable_signal_alignment: Enable Path ① (signal alignment filter).
        enable_alignment_reject: Override for alignment reject (< 0.4). None = follow enable_signal_alignment.
        enable_alignment_reduce: Override for alignment reduce (< 0.7). None = follow enable_signal_alignment.
        enable_confidence_override: Enable high-confidence override for alignment reject.
        enable_risk_budget: Enable Path ② (regime risk budget cap).
        enable_stop_multiplier: Enable Path ③ (regime stop multiplier).

    Returns:
        dict with position_size, stop_loss_pct, take_profit_pct, etc.
    """
    # Resolve alignment sub-switches
    do_alignment_reject = enable_alignment_reject if enable_alignment_reject is not None else enable_signal_alignment
    do_alignment_reduce = enable_alignment_reduce if enable_alignment_reduce is not None else enable_signal_alignment
    regime_state = regime.get("state", "range_bound")
    regime_confidence = regime.get("confidence", 0.5)

    risk_flags = []
    reject_reason = None

    # ── Path ① Signal alignment (optional) ──────────────────────────────
    alignment = compute_signal_alignment(regime_state, probability_up)

    if do_alignment_reject and alignment < ALIGNMENT_REJECT_THRESHOLD:
        # Check if high-confidence override is enabled and model is confident enough
        model_confidence = abs(probability_up - 0.5)
        if enable_confidence_override and model_confidence >= HIGH_CONFIDENCE_OVERRIDE:
            # High-confidence override: allow trade but flag it
            risk_flags.append("signal_conflict_overridden")
        else:
            return {
                "position_size": 0.0,
                "kelly_fraction": 0.0,
                "stop_loss_pct": STOP_MIN,
                "take_profit_pct": STOP_MIN * RISK_REWARD_RATIO,
                "max_holding_days": horizon_days,
                "reject_reason": "signal_conflict",
                "risk_flags": ["signal_conflict"],
                "signal_alignment": round(alignment, 4),
                "regime_state": regime_state,
            }

    # ── Kelly position sizing ────────────────────────────────────────────
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
            "signal_alignment": round(alignment, 4),
            "regime_state": regime_state,
        }

    position_size = kelly

    # ── Path ② Regime risk budget (optional) ─────────────────────────────
    budget = _REGIME_RISK_BUDGET.get(regime_state, dict(_DEFAULT_REGIME_BUDGET))

    if enable_risk_budget:
        if regime_confidence < budget["confidence_floor"]:
            return {
                "position_size": 0.0,
                "kelly_fraction": round(kelly, 4),
                "stop_loss_pct": STOP_MIN,
                "take_profit_pct": STOP_MIN * RISK_REWARD_RATIO,
                "max_holding_days": horizon_days,
                "reject_reason": "below_confidence_floor",
                "risk_flags": ["below_confidence_floor"],
                "signal_alignment": round(alignment, 4),
                "regime_state": regime_state,
            }
        position_size = min(position_size, budget["max_position"])

    # ── Uncertainty filtering (same as Stage 2) ──────────────────────────
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

    # ── Path ① Signal alignment reduction (REMOVED) ──────────────────────
    # Previously halved position when alignment < 0.7.
    # Ablation showed this hurt returns without improving risk-adjusted metrics.
    # Alignment 0.4-0.7 trades are profitable (avg +0.3%, 45% hit rate).

    # ── Direction ────────────────────────────────────────────────────────
    if action == "buy":
        direction = 1
    elif action == "sell":
        direction = -1
    else:
        direction = 1 if probability_up > 0.5 else -1
        position_size *= 0.25
        risk_flags.append("no_strong_edge")

    position_size = direction * position_size
    position_size = max(-MAX_POSITION_SIZE, min(MAX_POSITION_SIZE, position_size))

    # ── Minimum position threshold ───────────────────────────────────────
    if abs(position_size) > 0 and abs(position_size) < MIN_POSITION_THRESHOLD:
        position_size = 0.0
        reject_reason = reject_reason or "position_too_small"
        risk_flags.append("position_too_small")

    # ── Dynamic stop-loss / take-profit ──────────────────────────────────
    daily_vol = volatility_20 / math.sqrt(252.0)
    base_stop = daily_vol * STOP_VOL_MULTIPLIER

    # Path ③: Apply regime stop multiplier (optional)
    if enable_stop_multiplier:
        stop_mult = budget.get("stop_multiplier", 1.0)
    else:
        stop_mult = 1.0

    stop_loss_pct = min(STOP_MAX, max(STOP_MIN, base_stop * stop_mult))

    take_profit_pct = stop_loss_pct * RISK_REWARD_RATIO

    return {
        "position_size": float(position_size),
        "kelly_fraction": float(kelly),
        "stop_loss_pct": float(stop_loss_pct),
        "take_profit_pct": float(take_profit_pct),
        "max_holding_days": int(horizon_days),
        "reject_reason": reject_reason,
        "risk_flags": risk_flags,
        "signal_alignment": round(alignment, 4),
        "regime_state": regime_state,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stage 3 Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════

def run_stage3_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    model,
    meta: dict,
    calibrator,
    regime_agent: RegimeAgent,
    macro_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    horizon: int = 5,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.50,
    verbose: bool = False,
    # Ablation switches
    enable_regime_features: bool = False,    # Path ④
    enable_signal_alignment: bool = False,   # Path ①
    enable_alignment_reject: bool | None = None,   # Path ①a (override)
    enable_alignment_reduce: bool | None = None,   # Path ①b (override)
    enable_confidence_override: bool = False,       # Path ①c (high-confidence override)
    enable_risk_budget: bool = False,        # Path ②
    enable_stop_multiplier: bool = False,    # Path ③
) -> dict:
    """Run Stage 3 backtest with RegimeAgent and ablation switches.

    Key differences from Stage 2:
      - RegimeAgent computes regime state for each trading day
      - Path ④: Real regime features injected into LightGBM scoring
      - Path ③: Regime-aware stop multiplier
      - Path ②: Regime risk budget (max_position cap + confidence_floor)
      - Path ①: Signal alignment filter
      - Path ①a: Alignment reject only (< 0.4 → reject)
      - Path ①b: Alignment reduce only (< 0.7 → half position)
      - Path ①c: High-confidence override for alignment reject
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
    high_prices = pd.to_numeric(data["High"], errors="coerce").values
    low_prices = pd.to_numeric(data["Low"], errors="coerce").values
    close_prices = pd.to_numeric(data["Close"], errors="coerce").values

    start_idx = dates.searchsorted(pd.Timestamp(start_date))
    start_idx = max(start_idx, 60)
    total_days = len(dates)

    cost_per_trade = (cost_bps + slippage_bps) / 10000.0 * 2  # round-trip

    # Pre-sort macro snapshot dates for efficient lookup
    _sorted_macro_dates = sorted(macro_snapshots.keys()) if macro_snapshots else []

    trade_log = []
    equity = 1.0
    equity_curve = {}
    risk_stats = {
        "total_signals": 0,
        "rejected_by_uncertainty": 0,
        "rejected_by_min_position": 0,
        "rejected_by_signal_conflict": 0,
        "rejected_by_confidence_floor": 0,
        "stopped_out": 0,
        "took_profit": 0,
        "horizon_exit": 0,
    }
    regime_stats = {
        "regime_counts": {},
        "regime_prob_diffs": [],  # (regime_state, prob_with_regime - prob_without)
    }
    step = 0
    total_steps = (total_days - start_idx - horizon) // horizon

    # Feature history buffer for regime smoothing (last 60 days)
    REGIME_HISTORY_WINDOW = 60

    t = start_idx
    while t < total_days - horizon:
        step += 1
        current_date = dates[t]

        # Compute base features (no look-ahead)
        data_slice = data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            t += horizon
            continue

        # ── Compute regime using RegimeAgent ─────────────────────────────
        # Build feature_analysis dict matching RegimeAgent.analyze() input
        feature_analysis = {"features": features}

        # Build feature history for regime smoothing
        feature_history = None
        history_start = max(60, t - REGIME_HISTORY_WINDOW)
        if t - history_start >= 2:
            feature_history = []
            for h_idx in range(history_start, t + 1):
                h_slice = data.iloc[:h_idx + 1]
                h_feats = compute_features(h_slice)
                if h_feats:
                    feature_history.append(h_feats)

        # ── Get macro snapshot for current date ────────────────────────
        macro_snapshot = None
        if macro_snapshots and _sorted_macro_dates:
            date_str = current_date.strftime("%Y-%m-%d")
            # Exact match first
            if date_str in macro_snapshots:
                macro_snapshot = macro_snapshots[date_str]
            else:
                # Nearest earlier date via binary search
                import bisect
                idx = bisect.bisect_right(_sorted_macro_dates, date_str) - 1
                if idx >= 0:
                    macro_snapshot = macro_snapshots[_sorted_macro_dates[idx]]

        regime_result = regime_agent.analyze(
            stock_symbol=ticker,
            feature_analysis=feature_analysis,
            macro_features=macro_snapshot,
            feature_history=feature_history,
        )
        regime = regime_result.get("regime", {})
        regime_state = regime.get("state", "range_bound")

        # Track regime distribution
        regime_stats["regime_counts"][regime_state] = (
            regime_stats["regime_counts"].get(regime_state, 0) + 1
        )

        # ── Score with regime features (Path ④) ─────────────────────────
        raw_prob, prob, uncertainty_info = score_features_with_regime(
            features=features,
            regime=regime,
            model=model,
            meta=meta,
            calibrator=calibrator,
            use_real_regime=enable_regime_features,
            return_uncertainty=True,
            macro_snapshot=macro_snapshot,
        )

        # Track probability difference for Path ④ analysis
        if enable_regime_features:
            _, prob_baseline = score_features(features, model, meta, calibrator)
            prob_diff = prob - prob_baseline
            regime_stats["regime_prob_diffs"].append(
                (regime_state, round(prob_diff, 6))
            )

        # Decision: simple threshold (same as Stage 1/2)
        if prob > buy_threshold:
            action = "buy"
        elif prob < sell_threshold:
            action = "sell"
        else:
            action = "hold"

        risk_stats["total_signals"] += 1

        # ── Get risk plan (with ablation switches for Paths ①②③) ────────
        risk_plan = regime_risk_plan(
            action=action,
            probability_up=prob,
            volatility_20=features.get("volatility_20", 0.25),
            regime=regime,
            horizon_days=horizon,
            uncertainty_info=uncertainty_info,
            macro_snapshot=macro_snapshot,
            enable_signal_alignment=enable_signal_alignment,
            enable_alignment_reject=enable_alignment_reject,
            enable_alignment_reduce=enable_alignment_reduce,
            enable_confidence_override=enable_confidence_override,
            enable_risk_budget=enable_risk_budget,
            enable_stop_multiplier=enable_stop_multiplier,
        )

        position_size = risk_plan["position_size"]
        stop_loss_pct = risk_plan["stop_loss_pct"]
        take_profit_pct = risk_plan["take_profit_pct"]
        reject_reason = risk_plan["reject_reason"]

        # Track rejections
        if reject_reason == "signal_conflict":
            risk_stats["rejected_by_signal_conflict"] += 1
        elif reject_reason == "below_confidence_floor":
            risk_stats["rejected_by_confidence_floor"] += 1
        elif reject_reason in ("conformal_ambiguous", "conformal_empty"):
            risk_stats["rejected_by_uncertainty"] += 1
        elif reject_reason == "position_too_small":
            risk_stats["rejected_by_min_position"] += 1

        # Execute trade (only if position_size != 0)
        if abs(position_size) > 0 and t + 1 < total_days:
            entry_idx = t + 1
            entry_price = open_prices[entry_idx]

            if np.isnan(entry_price) or entry_price <= 0:
                t += horizon
                continue

            direction = 1.0 if position_size > 0 else -1.0
            abs_position = abs(position_size)

            # ── Daily stop-loss / take-profit check ──────────────────────
            exit_price = None
            exit_reason = "horizon"
            exit_idx = None

            for day_offset in range(1, horizon + 1):
                check_idx = entry_idx + day_offset
                if check_idx >= total_days:
                    break

                day_high = high_prices[check_idx]
                day_low = low_prices[check_idx]
                day_close = close_prices[check_idx]

                if np.isnan(day_high) or np.isnan(day_low) or np.isnan(day_close):
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

            # If no stop/TP triggered, exit at horizon close
            if exit_price is None:
                exit_idx = min(entry_idx + horizon, total_days - 1)
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

            # Track exit reasons
            if exit_reason == "stop_loss":
                risk_stats["stopped_out"] += 1
            elif exit_reason == "take_profit":
                risk_stats["took_profit"] += 1
            else:
                risk_stats["horizon_exit"] += 1

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
                "equity": round(float(equity), 6),
                "risk_flags": risk_plan["risk_flags"],
                "regime_state": regime_state,
                "signal_alignment": risk_plan.get("signal_alignment", 0.0),
            }
            trade_log.append(trade)
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

            if verbose and step % 20 == 0:
                print(
                    f"  [{ticker}] Step {step}/{total_steps}: "
                    f"date={current_date.strftime('%Y-%m-%d')}, "
                    f"regime={regime_state}, action={action}, prob={prob:.4f}, "
                    f"pos={abs_position:.2f}, exit={exit_reason}, "
                    f"ret={net_return:+.4f}, equity={equity:.4f}"
                )
        else:
            # Rejected or hold
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

            if verbose and step % 50 == 0:
                reason = reject_reason or "hold"
                print(
                    f"  [{ticker}] Step {step}/{total_steps}: "
                    f"date={current_date.strftime('%Y-%m-%d')}, "
                    f"regime={regime_state}, action={action}, prob={prob:.4f}, "
                    f"SKIPPED ({reason}), equity={equity:.4f}"
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
        "risk_stats": risk_stats,
        "regime_stats": regime_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Analysis (extends Stage 2)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_stage3(result: dict) -> dict:
    """Compute Stage 3 specific metrics."""
    base_metrics = analyze_backtest(result)
    if "error" in base_metrics:
        return base_metrics

    trades = result["trade_log"]
    risk_stats = result.get("risk_stats", {})

    # Exit reason breakdown
    stop_trades = [t for t in trades if t.get("exit_reason") == "stop_loss"]
    tp_trades = [t for t in trades if t.get("exit_reason") == "take_profit"]
    horizon_trades = [t for t in trades if t.get("exit_reason") == "horizon"]

    stop_returns = [t["net_return"] for t in stop_trades]
    tp_returns = [t["net_return"] for t in tp_trades]
    horizon_returns = [t["net_return"] for t in horizon_trades]

    # Position size distribution
    pos_sizes = [abs(t.get("position_size", 1.0)) for t in trades]

    # Regime-specific performance
    regime_perf = {}
    for t in trades:
        rs = t.get("regime_state", "unknown")
        if rs not in regime_perf:
            regime_perf[rs] = {"returns": [], "count": 0}
        regime_perf[rs]["returns"].append(t["net_return"])
        regime_perf[rs]["count"] += 1

    for rs, data in regime_perf.items():
        rets = data["returns"]
        data["avg_return"] = round(float(np.mean(rets)), 6) if rets else 0.0
        data["hit_rate"] = round(sum(1 for r in rets if r > 0) / len(rets), 4) if rets else 0.0
        data["total_return"] = round(float(np.sum(rets)), 6)

    base_metrics.update({
        # Risk stats
        "total_signals": risk_stats.get("total_signals", 0),
        "rejected_by_uncertainty": risk_stats.get("rejected_by_uncertainty", 0),
        "rejected_by_min_position": risk_stats.get("rejected_by_min_position", 0),
        "rejected_by_signal_conflict": risk_stats.get("rejected_by_signal_conflict", 0),
        "rejected_by_confidence_floor": risk_stats.get("rejected_by_confidence_floor", 0),
        # Exit breakdown
        "n_stop_loss": len(stop_trades),
        "n_take_profit": len(tp_trades),
        "n_horizon_exit": len(horizon_trades),
        "avg_stop_loss_return": round(float(np.mean(stop_returns)), 6) if stop_returns else 0.0,
        "avg_take_profit_return": round(float(np.mean(tp_returns)), 6) if tp_returns else 0.0,
        "avg_horizon_return": round(float(np.mean(horizon_returns)), 6) if horizon_returns else 0.0,
        # Position sizing
        "avg_position_size": round(float(np.mean(pos_sizes)), 4) if pos_sizes else 0.0,
        "min_position_size": round(float(np.min(pos_sizes)), 4) if pos_sizes else 0.0,
        "max_position_size": round(float(np.max(pos_sizes)), 4) if pos_sizes else 0.0,
        # Regime performance
        "regime_performance": regime_perf,
    })

    return base_metrics


# ═══════════════════════════════════════════════════════════════════════════
# 5. Comparison Report
# ═══════════════════════════════════════════════════════════════════════════

def print_ablation_report(
    ticker: str,
    round_results: Dict[str, dict],
    round_metrics: Dict[str, dict],
) -> None:
    """Print ablation comparison table across all rounds."""

    print(f"\n{'='*80}")
    print(f"  Stage 3: RegimeAgent Ablation Report — {ticker}")
    print(f"{'='*80}")

    rounds = sorted(round_results.keys())
    if not rounds:
        print("  No rounds to report.")
        return

    # Configuration
    config = round_results[rounds[0]]
    print(f"\n📋 Configuration:")
    print(f"  Period:     {config['start_date']} → {config['end_date']}")
    print(f"  Horizon:    {config['horizon']}d")
    print(f"  Thresholds: buy > {config['buy_threshold']}, sell < {config['sell_threshold']}")

    round_labels = {
        "v0": "Stage2 baseline (no regime)",
        "v1": "+ Path④ (regime→LGB)",
        "v1b": "+ Path④① (+ signal alignment, skip②③)",
        "v2": "+ Path④③ (+ stop mult)",
        "v2b": "+ Path④② (+ risk budget, skip③)",
        "v3": "+ Path④③② (+ risk budget)",
        "v4": "Full (all 4 paths)",
    }

    # ── Main comparison table ────────────────────────────────────────────
    print(f"\n── Ablation Comparison ──")

    # Header
    header = f"  {'Metric':<22s}"
    for r in rounds:
        label = r
        header += f" | {label:>14s}"
    print(header)
    print(f"  {'-'*22}" + " | " + " | ".join([f"{'-'*14}"] * len(rounds)))

    # Rows
    metric_rows = [
        ("Total Return", "total_return", "+.2%"),
        ("Alpha", "alpha", "+.2%"),
        ("Sharpe Ratio", "sharpe_ratio", "+.4f"),
        ("Sortino Ratio", "sortino_ratio", "+.4f"),
        ("Max Drawdown", "max_drawdown", "+.2%"),
        ("Hit Rate", "hit_rate", ".2%"),
        ("Profit Factor", "profit_factor", ".4f"),
        ("Avg Trade Return", "avg_trade_return", "+.6f"),
        ("Trade IC", "trade_ic", "+.6f"),
        ("N Trades", "n_trades", "d"),
        ("N Buy", "n_buy", "d"),
        ("N Sell", "n_sell", "d"),
    ]

    for name, key, fmt in metric_rows:
        row = f"  {name:<22s}"
        for r in rounds:
            m = round_metrics[r]
            val = m.get(key, 0)
            if fmt == "d":
                cell = f"{val:d}"
            else:
                cell = f"{val:{fmt}}"
            row += f" | {cell:>14s}"
        print(row)

    # ── Risk stats (Stage 3 specific) ────────────────────────────────────
    risk_keys = [
        ("Total Signals", "total_signals"),
        ("Rej: Uncertainty", "rejected_by_uncertainty"),
        ("Rej: Min Position", "rejected_by_min_position"),
        ("Rej: Signal Conflict", "rejected_by_signal_conflict"),
        ("Rej: Conf Floor", "rejected_by_confidence_floor"),
        ("Stop-Loss Exits", "n_stop_loss"),
        ("Take-Profit Exits", "n_take_profit"),
        ("Horizon Exits", "n_horizon_exit"),
    ]

    print(f"\n── Risk Management Stats ──")
    header = f"  {'Metric':<22s}"
    for r in rounds:
        header += f" | {r:>14s}"
    print(header)
    print(f"  {'-'*22}" + " | ".join([f"{'-'*14}"] * len(rounds)))

    for name, key in risk_keys:
        row = f"  {name:<22s}"
        for r in rounds:
            m = round_metrics[r]
            val = m.get(key, 0)
            row += f" | {val:>14d}"
        print(row)

    # ── Position sizing stats ────────────────────────────────────────────
    print(f"\n── Position Sizing ──")
    for name, key in [("Avg Position", "avg_position_size"), ("Min Position", "min_position_size"), ("Max Position", "max_position_size")]:
        row = f"  {name:<22s}"
        for r in rounds:
            m = round_metrics[r]
            val = m.get(key, 0)
            row += f" | {val:>14.4f}"
        print(row)

    # ── Exit reason breakdown ────────────────────────────────────────────
    print(f"\n── Exit Reason Avg Returns ──")
    for name, key in [("Avg SL Return", "avg_stop_loss_return"), ("Avg TP Return", "avg_take_profit_return"), ("Avg Horizon Return", "avg_horizon_return")]:
        row = f"  {name:<22s}"
        for r in rounds:
            m = round_metrics[r]
            val = m.get(key, 0)
            row += f" | {val:>+14.6f}"
        print(row)

    # ── Regime distribution (for rounds with regime) ─────────────────────
    for r in rounds:
        regime_stats = round_results[r].get("regime_stats", {})
        regime_counts = regime_stats.get("regime_counts", {})
        if regime_counts:
            total = sum(regime_counts.values())
            print(f"\n── Regime Distribution ({r}) ──")
            for state, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                print(f"  {state:<20s}: {count:>4d} ({pct:5.1f}%)")

    # ── Regime-specific performance (for rounds with regime) ─────────────
    for r in rounds:
        m = round_metrics[r]
        regime_perf = m.get("regime_performance", {})
        if regime_perf:
            print(f"\n── Regime Performance ({r}) ──")
            print(f"  {'Regime':<20s} | {'Count':>6s} | {'Avg Ret':>10s} | {'Hit Rate':>9s} | {'Total Ret':>10s}")
            print(f"  {'-'*20} | {'-'*6} | {'-'*10} | {'-'*9} | {'-'*10}")
            for state, data in sorted(regime_perf.items(), key=lambda x: -x[1]["count"]):
                print(
                    f"  {state:<20s} | {data['count']:>6d} | "
                    f"{data['avg_return']:>+10.6f} | {data['hit_rate']:>9.2%} | "
                    f"{data['total_return']:>+10.6f}"
                )

    # ── Path ④ probability diff analysis ─────────────────────────────────
    for r in rounds:
        regime_stats = round_results[r].get("regime_stats", {})
        prob_diffs = regime_stats.get("regime_prob_diffs", [])
        if prob_diffs:
            print(f"\n── Path ④ Probability Impact ({r}) ──")
            # Group by regime state
            diff_by_regime = {}
            for state, diff in prob_diffs:
                if state not in diff_by_regime:
                    diff_by_regime[state] = []
                diff_by_regime[state].append(diff)

            print(f"  {'Regime':<20s} | {'Count':>6s} | {'Avg ΔProb':>10s} | {'Std ΔProb':>10s} | {'Min':>8s} | {'Max':>8s}")
            print(f"  {'-'*20} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
            for state, diffs in sorted(diff_by_regime.items(), key=lambda x: -len(x[1])):
                arr = np.array(diffs)
                print(
                    f"  {state:<20s} | {len(diffs):>6d} | "
                    f"{np.mean(arr):>+10.6f} | {np.std(arr):>10.6f} | "
                    f"{np.min(arr):>+8.4f} | {np.max(arr):>+8.4f}"
                )

            # Overall
            all_diffs = np.array([d for _, d in prob_diffs])
            print(f"\n  Overall: mean ΔProb = {np.mean(all_diffs):+.6f}, "
                  f"std = {np.std(all_diffs):.6f}, "
                  f"range = [{np.min(all_diffs):+.4f}, {np.max(all_diffs):+.4f}]")

    # ── Delta summary ────────────────────────────────────────────────────
    if len(rounds) >= 2:
        print(f"\n── Delta vs Baseline ({rounds[0]}) ──")
        baseline = round_metrics[rounds[0]]
        for r in rounds[1:]:
            m = round_metrics[r]
            label = round_labels.get(r, r)
            ret_delta = m["total_return"] - baseline["total_return"]
            sharpe_delta = m["sharpe_ratio"] - baseline["sharpe_ratio"]
            dd_delta = m["max_drawdown"] - baseline["max_drawdown"]
            hit_delta = m["hit_rate"] - baseline["hit_rate"]
            trade_delta = m["n_trades"] - baseline["n_trades"]

            print(f"\n  {r} ({label}):")
            print(f"    Return:   {ret_delta:+.2%}  {'✅' if ret_delta > 0 else '❌'}")
            print(f"    Sharpe:   {sharpe_delta:+.4f}  {'✅' if sharpe_delta > 0 else '❌'}")
            print(f"    Drawdown: {dd_delta:+.2%}  {'✅' if dd_delta > 0 else '❌'}")
            print(f"    Hit Rate: {hit_delta:+.2%}  {'✅' if hit_delta > 0 else '❌'}")
            print(f"    Trades:   {trade_delta:+d}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════

# Round configurations: which paths are enabled
ROUND_CONFIGS = {
    "v0": {
        "enable_regime_features": False,
        "enable_signal_alignment": False,
        "enable_risk_budget": False,
        "enable_stop_multiplier": False,
        "description": "Stage 2 baseline (no RegimeAgent)",
    },
    "v1": {
        "enable_regime_features": True,
        "enable_signal_alignment": False,
        "enable_risk_budget": False,
        "enable_stop_multiplier": False,
        "description": "+ Path④ (regime features → LightGBM)",
    },
    # Final strategy: Path④ + alignment reject + high-confidence override, no reduce
    "v1_final": {
        "enable_regime_features": True,
        "enable_signal_alignment": False,
        "enable_alignment_reject": True,
        "enable_alignment_reduce": False,
        "enable_confidence_override": True,
        "enable_risk_budget": False,
        "enable_stop_multiplier": False,
        "description": "Final: Path④ + reject + confidence override, no reduce/budget/stop",
    },
}


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 3: RegimeAgent Ablation Backtest")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="Holding period in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Buy/sell threshold (default: 0.55)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost bps (default: 5.0)")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps (default: 5.0)")
    parser.add_argument("--rounds", type=str, default="v0,v1",
                        help="Comma-separated rounds to run (default: v0,v1). Options: v0,v1,v2,v3,v4")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    rounds_to_run = [r.strip() for r in args.rounds.split(",")]

    # Validate rounds
    for r in rounds_to_run:
        if r not in ROUND_CONFIGS:
            print(f"ERROR: Unknown round '{r}'. Valid: {list(ROUND_CONFIGS.keys())}")
            sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  Stage 3: RegimeAgent Ablation Backtest")
    print(f"  Ticker:    {ticker}")
    print(f"  Period:    {args.start} → {args.end}")
    print(f"  Horizon:   {args.horizon}d")
    print(f"  Threshold: {args.threshold}")
    print(f"  Rounds:    {', '.join(rounds_to_run)}")
    print(f"  Model:     {os.getenv('FORECAST_LGB_MODEL_PATH', 'data/forecast_model.lgb')}")
    print(f"{'='*80}")

    # Load model
    print(f"\n── Loading model ──")
    model, meta, calibrator = load_model()

    # Initialize RegimeAgent
    print(f"\n── Initializing RegimeAgent ──")
    regime_agent = RegimeAgent(verbose=args.verbose)
    print(f"  [✓] RegimeAgent initialized")

    # Initialize MacroFundamentalFeatureProvider and load historical data
    print(f"\n── Loading historical macro/fundamental data ──")
    macro_snapshots = {}
    try:
        from datetime import datetime as _dt
        macro_provider = MacroFundamentalFeatureProvider(verbose=args.verbose)
        hist_df = macro_provider.extract_historical(
            stock_symbol=ticker,
            start_date=_dt.strptime(args.start, "%Y-%m-%d"),
            end_date=_dt.strptime(args.end, "%Y-%m-%d"),
        )
        if hist_df is not None and not hist_df.empty:
            for date_idx, row in hist_df.iterrows():
                date_key = str(date_idx)[:10]
                macro_feats = {}
                fund_feats = {}
                for col in MACRO_FEATURE_COLUMNS:
                    val = row.get(col)
                    macro_feats[col] = float(val) if pd.notna(val) else None
                for col in FUNDAMENTAL_FEATURE_COLUMNS:
                    val = row.get(col)
                    fund_feats[col] = float(val) if pd.notna(val) else None
                macro_snapshots[date_key] = {
                    "status": "success",
                    "macro_features": macro_feats,
                    "fundamental_features": fund_feats,
                }
            print(f"  [✓] Loaded {len(macro_snapshots)} daily macro snapshots")
        else:
            print(f"  [⚠] Historical macro data returned empty")
    except Exception as exc:
        print(f"  [⚠] Failed to load macro data: {exc}")
        print(f"  [⚠] Falling back to macro_features=None (same as before)")

    # Run each round
    round_results = {}
    round_metrics = {}

    for r in rounds_to_run:
        config = ROUND_CONFIGS[r]
        desc = config["description"]
        print(f"\n{'─'*80}")
        print(f"  Running {r}: {desc}")
        print(f"{'─'*80}")

        t0 = time.time()
        result = run_stage3_backtest(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
            model=model,
            meta=meta,
            calibrator=calibrator,
            regime_agent=regime_agent,
            macro_snapshots=macro_snapshots if macro_snapshots else None,
            horizon=args.horizon,
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            buy_threshold=args.threshold,
            sell_threshold=1.0 - args.threshold,
            verbose=args.verbose,
            enable_regime_features=config["enable_regime_features"],
            enable_signal_alignment=config["enable_signal_alignment"],
            enable_alignment_reject=config.get("enable_alignment_reject"),
            enable_confidence_override=config.get("enable_confidence_override", False),
            enable_alignment_reduce=config.get("enable_alignment_reduce"),
            enable_risk_budget=config["enable_risk_budget"],
            enable_stop_multiplier=config["enable_stop_multiplier"],
        )
        elapsed = time.time() - t0

        metrics = analyze_stage3(result)
        round_results[r] = result
        round_metrics[r] = metrics

        n_trades = len(result["trade_log"])
        ret = result["final_equity"] - 1.0
        print(f"  [✓] {r} done in {elapsed:.1f}s — {n_trades} trades, return={ret:+.2%}")

    # Print ablation report
    print_ablation_report(ticker, round_results, round_metrics)

    # ── Save results ─────────────────────────────────────────────────────
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"stage3_{ticker}_{args.start}_{args.end}_t{args.threshold}"

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
            "rounds_run": rounds_to_run,
        },
    }
    for r in rounds_to_run:
        # Remove non-serializable items from regime_stats
        metrics_copy = dict(round_metrics[r])
        save_data[f"{r}_metrics"] = metrics_copy
        save_data[f"{r}_risk_stats"] = round_results[r].get("risk_stats", {})
        save_data[f"{r}_regime_counts"] = round_results[r].get("regime_stats", {}).get("regime_counts", {})

    with open(report_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[✓] Report saved: {report_path}")

    # Save trade logs CSV (for each round)
    for r in rounds_to_run:
        trades = round_results[r]["trade_log"]
        if trades:
            trades_path = output_dir / f"{base_name}_{r}_trades.csv"
            trades_df = pd.DataFrame(trades)
            if "risk_flags" in trades_df.columns:
                trades_df["risk_flags"] = trades_df["risk_flags"].apply(
                    lambda x: "|".join(x) if isinstance(x, list) else str(x)
                )
            trades_df.to_csv(trades_path, index=False)
            print(f"[✓] Trade log saved: {trades_path}")

    # Save equity curves CSV (all rounds)
    all_dates = set()
    for r in rounds_to_run:
        all_dates.update(round_results[r]["equity_curve"].keys())
    all_dates = sorted(all_dates)

    eq_rows = []
    for d in all_dates:
        row = {"date": d}
        for r in rounds_to_run:
            row[f"{r}_equity"] = round_results[r]["equity_curve"].get(d, np.nan)
        eq_rows.append(row)

    equity_path = output_dir / f"{base_name}_equity.csv"
    pd.DataFrame(eq_rows).to_csv(equity_path, index=False)
    print(f"[✓] Equity curves saved: {equity_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
