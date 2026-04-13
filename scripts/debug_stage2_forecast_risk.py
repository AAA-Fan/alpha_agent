#!/usr/bin/env python3
"""
Stage 2: ForecastAgent + Simplified RiskAgent Backtest

Goal: Validate whether RiskAgent's core risk controls HELP or HURT
      performance compared to Stage 1 (ForecastAgent only).

Simplified RiskAgent (5 steps only):
  ① Prediction Kelly position sizing (p=current prob, b=avg_win/avg_loss)
  ② Uncertainty filtering via Conformal Prediction Set + Tree Dispersion
  ③ Direction judgment (buy/sell/hold)
  ④ Minimum position threshold (< 3% → zero)
  ⑤ Dynamic stop-loss + take-profit (daily_vol × 2.5, R:R = 2.0)

Bypassed (Stage 3 will add these back):
  - Signal alignment (①⑤ in full RiskAgent) — needs RegimeAgent
  - Regime risk budget (③) — needs RegimeAgent
  - Macro/Fundamental adjustments (⑦) — needs external data
  - Track record factor (⑧) — needs MemoryAgent

Usage:
    python scripts/debug_stage2_forecast_risk.py --ticker AAPL --start 2023-01-01 --end 2025-12-31
    python scripts/debug_stage2_forecast_risk.py --ticker AAPL --start 2023-01-01 --end 2025-12-31 --threshold 0.55
    python scripts/debug_stage2_forecast_risk.py --ticker AAPL --start 2023-01-01 --end 2025-12-31 --verbose
"""

from __future__ import annotations

import argparse
import json
import math
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

# Reuse Stage 1 utilities
from debug_stage1_forecast_only import (
    compute_features,
    load_model,
    score_features,
    analyze_backtest,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Simplified RiskAgent (inline — no Regime/Memory/Macro dependencies)
# ═══════════════════════════════════════════════════════════════════════════

# Kelly defaults (b = avg_win / avg_loss; p comes from prediction probability)
KELLY_AVG_WIN = 0.03
KELLY_AVG_LOSS = 0.02

# Position limits
MAX_POSITION_SIZE = 1.0
MIN_POSITION_THRESHOLD = 0.03

# Uncertainty thresholds
UNCERTAINTY_HIGH = 0.15
UNCERTAINTY_MODERATE = 0.10

# Stop-loss / take-profit
RISK_REWARD_RATIO = 2.0
STOP_VOL_MULTIPLIER = 2.5
STOP_MIN = 0.01
STOP_MAX = 0.08


def compute_prediction_kelly(
    probability_up: float,
    action: str,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Compute Kelly fraction using current prediction probability.

    Uses the *predicted* probability as p (not historical win rate),
    combined with historical avg_win/avg_loss as the odds ratio b.
    This way position size scales continuously with signal strength.

    Args:
        probability_up: Current model prediction (0-1).
        action: 'buy', 'sell', or 'hold'.
        avg_win: Historical average win magnitude.
        avg_loss: Historical average loss magnitude.

    Returns:
        Kelly fraction (0 to MAX_POSITION_SIZE).
    """
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss  # odds ratio from historical data

    # p = directional probability (how confident we are in the chosen direction)
    if action == "buy":
        p = probability_up
    elif action == "sell":
        p = 1.0 - probability_up
    else:
        # hold: use whichever direction is stronger
        p = max(probability_up, 1.0 - probability_up)

    q = 1.0 - p
    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly, MAX_POSITION_SIZE)


def simplified_risk_plan(
    action: str,
    probability_up: float,
    volatility_20: float,
    horizon_days: int = 5,
    uncertainty_info: dict | None = None,
) -> dict:
    """Simplified RiskAgent: Prediction Kelly + conformal uncertainty + stop-loss.

    Args:
        action: 'buy', 'sell', or 'hold'.
        probability_up: Calibrated probability of upward move.
        volatility_20: 20-day annualized volatility.
        horizon_days: Holding period in days.
        uncertainty_info: Dict from score_features with conformal prediction set
                          and tree ensemble dispersion.

    Returns:
        dict with keys: position_size, stop_loss_pct, take_profit_pct,
                        max_holding_days, reject_reason, risk_flags
    """
    risk_flags = []
    reject_reason = None

    # ── ① Prediction Kelly position sizing ──────────────────────────────
    # p = current prediction probability (not historical win rate)
    # b = avg_win / avg_loss (historical odds ratio)
    kelly = compute_prediction_kelly(
        probability_up=probability_up,
        action=action,
        avg_win=KELLY_AVG_WIN,
        avg_loss=KELLY_AVG_LOSS,
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

    position_size = kelly

    # ── ② Uncertainty-aware filtering (Conformal Prediction) ─────────────
    # Use model's conformal prediction set instead of proxy uncertainty.
    # If the conformal set contains both {up, down}, the model is ambiguous
    # → reject the trade. If tree ensemble dispersion is high → also reject.
    if uncertainty_info is not None:
        prediction_set = uncertainty_info.get("prediction_set") or []
        tree_uncertain = uncertainty_info.get("is_uncertain", False)
        tree_dispersion = uncertainty_info.get("uncertainty")

        if len(prediction_set) == 2:
            # Conformal: both labels in set → ambiguous prediction
            position_size = 0.0
            reject_reason = "conformal_ambiguous"
            risk_flags.append("conformal_ambiguous")
        elif len(prediction_set) == 0:
            # Conformal: neither label in set → very unusual
            position_size = 0.0
            reject_reason = "conformal_empty"
            risk_flags.append("conformal_empty")
        elif tree_dispersion is not None and tree_dispersion > 0.15:
            # Tree ensemble disagrees significantly
            position_size *= 0.5
            risk_flags.append("high_tree_dispersion")

    # ── ③ Direction ──────────────────────────────────────────────────────
    if action == "buy":
        direction = 1
    elif action == "sell":
        direction = -1
    else:
        # Hold: reduce position to 25% (weak signal)
        direction = 1 if probability_up > 0.5 else -1
        position_size *= 0.25
        risk_flags.append("no_strong_edge")

    position_size = direction * position_size

    # Clamp
    position_size = max(-MAX_POSITION_SIZE, min(MAX_POSITION_SIZE, position_size))

    # ── ④ Minimum position threshold ────────────────────────────────────
    if abs(position_size) > 0 and abs(position_size) < MIN_POSITION_THRESHOLD:
        position_size = 0.0
        reject_reason = reject_reason or "position_too_small"
        risk_flags.append("position_too_small")

    # ── ⑤ Dynamic stop-loss / take-profit ───────────────────────────────
    daily_vol = volatility_20 / math.sqrt(252.0)
    base_stop = daily_vol * STOP_VOL_MULTIPLIER
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
# 2. Stage 2 Backtest Engine (with stop-loss / take-profit intraday check)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage2_backtest(
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
    """Run ForecastAgent + simplified RiskAgent walk-forward backtest.

    Key differences from Stage 1:
      - Position size from Prediction Kelly (p=current prob, b=avg_win/avg_loss)
      - Conformal prediction set + tree dispersion for uncertainty filtering
      - Dynamic stop-loss / take-profit checked daily during holding period
      - Minimum position threshold
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

    trade_log = []
    equity = 1.0
    equity_curve = {}
    risk_stats = {
        "total_signals": 0,
        "rejected_by_uncertainty": 0,
        "rejected_by_min_position": 0,
        "stopped_out": 0,
        "took_profit": 0,
        "horizon_exit": 0,
    }
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

        # Score (with uncertainty info for conformal filtering)
        raw_prob, prob, uncertainty_info = score_features(
            features, model, meta, calibrator, return_uncertainty=True,
        )

        # Decision: simple threshold (same as Stage 1)
        if prob > buy_threshold:
            action = "buy"
        elif prob < sell_threshold:
            action = "sell"
        else:
            action = "hold"

        risk_stats["total_signals"] += 1

        # Get risk plan from simplified RiskAgent
        risk_plan = simplified_risk_plan(
            action=action,
            probability_up=prob,
            volatility_20=features.get("volatility_20", 0.25),
            horizon_days=horizon,
            uncertainty_info=uncertainty_info,
        )

        position_size = risk_plan["position_size"]
        stop_loss_pct = risk_plan["stop_loss_pct"]
        take_profit_pct = risk_plan["take_profit_pct"]
        reject_reason = risk_plan["reject_reason"]

        # Track rejections
        if reject_reason in ("conformal_ambiguous", "conformal_empty"):
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

                # Check intraday extremes for stop/TP triggers
                day_high = high_prices[check_idx]
                day_low = low_prices[check_idx]
                day_close = close_prices[check_idx]

                if np.isnan(day_high) or np.isnan(day_low) or np.isnan(day_close):
                    continue

                if direction > 0:  # Long position
                    # Stop-loss: price drops below entry * (1 - stop_loss_pct)
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                    tp_price = entry_price * (1.0 + take_profit_pct)

                    if day_low <= stop_price:
                        exit_price = stop_price  # Assume stopped at stop price
                        exit_reason = "stop_loss"
                        exit_idx = check_idx
                        break
                    elif day_high >= tp_price:
                        exit_price = tp_price  # Assume filled at TP price
                        exit_reason = "take_profit"
                        exit_idx = check_idx
                        break
                else:  # Short position
                    # Stop-loss: price rises above entry * (1 + stop_loss_pct)
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
            }
            trade_log.append(trade)
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

            if verbose and step % 20 == 0:
                print(
                    f"  [{ticker}] Step {step}/{total_steps}: "
                    f"date={current_date.strftime('%Y-%m-%d')}, "
                    f"action={action}, prob={prob:.4f}, "
                    f"pos={abs_position:.2f}, exit={exit_reason}, "
                    f"ret={net_return:+.4f}, equity={equity:.4f}"
                )
        else:
            # Rejected or hold — record equity but no trade
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

            if verbose and step % 50 == 0:
                reason = reject_reason or "hold"
                print(
                    f"  [{ticker}] Step {step}/{total_steps}: "
                    f"date={current_date.strftime('%Y-%m-%d')}, "
                    f"action={action}, prob={prob:.4f}, "
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
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stage 1 Backtest (inline for comparison — same as Stage 1 script)
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
    """Stage 1 backtest (fixed 100% position, no stop-loss, horizon exit only)."""
    from utils.yfinance_cache import get_historical_data

    data = get_historical_data(ticker, interval="daily", outputsize="full")
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=120)
    end_ts = pd.Timestamp(end_date)
    data = data.loc[buffer_start:end_ts]

    dates = data.index
    open_prices = pd.to_numeric(data["Open"], errors="coerce").values
    close_prices = pd.to_numeric(data["Close"], errors="coerce").values

    start_idx = dates.searchsorted(pd.Timestamp(start_date))
    start_idx = max(start_idx, 60)
    total_days = len(dates)

    cost_per_trade = (cost_bps + slippage_bps) / 10000.0 * 2

    trade_log = []
    equity = 1.0
    equity_curve = {}

    t = start_idx
    while t < total_days - horizon:
        current_date = dates[t]
        data_slice = data.iloc[:t + 1]
        features = compute_features(data_slice)
        if not features:
            t += horizon
            continue

        raw_prob, prob = score_features(features, model, meta, calibrator)

        if prob > buy_threshold:
            action = "buy"
            direction = 1.0
        elif prob < sell_threshold:
            action = "sell"
            direction = -1.0
        else:
            action = "hold"
            direction = 0.0

        if direction != 0.0 and t + 1 < total_days:
            entry_price = open_prices[t + 1]
            exit_idx = min(t + 1 + horizon, total_days - 1)
            exit_price = close_prices[exit_idx]

            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price <= 0:
                t += horizon
                continue

            raw_return = (exit_price / entry_price - 1.0) * direction
            net_return = raw_return - cost_per_trade

            equity *= (1.0 + net_return)

            trade_log.append({
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
            })
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity
        else:
            equity_curve[current_date.strftime("%Y-%m-%d")] = equity

        t += horizon

    # Benchmark
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
# 4. Enhanced Analysis (Stage 2 specific)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_stage2(result: dict) -> dict:
    """Compute Stage 2 specific metrics (extends Stage 1 analysis)."""
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

    base_metrics.update({
        # Risk stats
        "total_signals": risk_stats.get("total_signals", 0),
        "rejected_by_uncertainty": risk_stats.get("rejected_by_uncertainty", 0),
        "rejected_by_min_position": risk_stats.get("rejected_by_min_position", 0),
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
    })

    return base_metrics


# ═══════════════════════════════════════════════════════════════════════════
# 5. Comparison Report
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_report(
    ticker: str,
    s1_metrics: dict,
    s2_metrics: dict,
    s2_result: dict,
) -> None:
    """Print side-by-side comparison of Stage 1 vs Stage 2."""

    print(f"\n{'='*70}")
    print(f"  Stage 2: ForecastAgent + RiskAgent Backtest — {ticker}")
    print(f"{'='*70}")

    print(f"\n📋 Configuration:")
    print(f"  Period:     {s2_result['start_date']} → {s2_result['end_date']}")
    print(f"  Horizon:    {s2_result['horizon']}d")
    print(f"  Thresholds: buy > {s2_result['buy_threshold']}, sell < {s2_result['sell_threshold']}")
    print(f"  Kelly:      prediction_kelly (p=prob, b=avg_win/avg_loss), avg_win={KELLY_AVG_WIN}, avg_loss={KELLY_AVG_LOSS}")
    print(f"  Stop/TP:    vol × {STOP_VOL_MULTIPLIER}, R:R = {RISK_REWARD_RATIO}")

    # ── Side-by-side comparison ──────────────────────────────────────────
    def _fmt(v, fmt="+.2%"):
        if isinstance(v, str):
            return v
        return f"{v:{fmt}}"

    def _delta(s2_val, s1_val, higher_is_better=True):
        diff = s2_val - s1_val
        if abs(diff) < 0.0001:
            return "  ≈"
        arrow = "↑" if diff > 0 else "↓"
        good = (diff > 0) == higher_is_better
        emoji = "✅" if good else "❌"
        return f"{emoji} {arrow} {abs(diff):.2%}"

    def _delta_num(s2_val, s1_val, higher_is_better=True, fmt=".4f"):
        diff = s2_val - s1_val
        if abs(diff) < 0.0001:
            return "  ≈"
        arrow = "↑" if diff > 0 else "↓"
        good = (diff > 0) == higher_is_better
        emoji = "✅" if good else "❌"
        return f"{emoji} {arrow} {abs(diff):{fmt}}"

    print(f"\n── Performance Comparison: Stage 1 vs Stage 2 ──")
    print(f"  {'Metric':<25s} | {'Stage 1':>12s} | {'Stage 2':>12s} | {'Delta':>16s}")
    print(f"  {'-'*25} | {'-'*12} | {'-'*12} | {'-'*16}")

    rows = [
        ("Strategy Return", s1_metrics["total_return"], s2_metrics["total_return"], True),
        ("Benchmark (SPY)", s1_metrics["benchmark_return"], s2_metrics["benchmark_return"], None),
        ("Alpha", s1_metrics["alpha"], s2_metrics["alpha"], True),
        ("Sharpe Ratio", s1_metrics["sharpe_ratio"], s2_metrics["sharpe_ratio"], True),
        ("Sortino Ratio", s1_metrics["sortino_ratio"], s2_metrics["sortino_ratio"], True),
        ("Max Drawdown", s1_metrics["max_drawdown"], s2_metrics["max_drawdown"], True),
        ("Hit Rate", s1_metrics["hit_rate"], s2_metrics["hit_rate"], True),
        ("Profit Factor", s1_metrics["profit_factor"], s2_metrics["profit_factor"], True),
        ("Avg Trade Return", s1_metrics["avg_trade_return"], s2_metrics["avg_trade_return"], True),
        ("Trade IC", s1_metrics["trade_ic"], s2_metrics["trade_ic"], True),
    ]

    for name, s1_val, s2_val, higher_better in rows:
        if higher_better is None:
            delta_str = ""
        elif name in ("Sharpe Ratio", "Sortino Ratio", "Profit Factor", "Trade IC"):
            delta_str = _delta_num(s2_val, s1_val, higher_better)
        else:
            delta_str = _delta(s2_val, s1_val, higher_better)
        print(f"  {name:<25s} | {s1_val:>+12.4f} | {s2_val:>+12.4f} | {delta_str:>16s}")

    # ── Trade count comparison ───────────────────────────────────────────
    print(f"\n── Trade Statistics ──")
    print(f"  {'Metric':<25s} | {'Stage 1':>12s} | {'Stage 2':>12s}")
    print(f"  {'-'*25} | {'-'*12} | {'-'*12}")
    print(f"  {'Total Trades':<25s} | {s1_metrics['n_trades']:>12d} | {s2_metrics['n_trades']:>12d}")
    print(f"  {'Buy Trades':<25s} | {s1_metrics['n_buy']:>12d} | {s2_metrics['n_buy']:>12d}")
    print(f"  {'Sell Trades':<25s} | {s1_metrics['n_sell']:>12d} | {s2_metrics['n_sell']:>12d}")
    print(f"  {'Buy Hit Rate':<25s} | {s1_metrics['buy_hit_rate']:>12.2%} | {s2_metrics['buy_hit_rate']:>12.2%}")
    print(f"  {'Sell Hit Rate':<25s} | {s1_metrics['sell_hit_rate']:>12.2%} | {s2_metrics['sell_hit_rate']:>12.2%}")
    print(f"  {'Max Consec Wins':<25s} | {s1_metrics['max_consecutive_wins']:>12d} | {s2_metrics['max_consecutive_wins']:>12d}")
    print(f"  {'Max Consec Losses':<25s} | {s1_metrics['max_consecutive_losses']:>12d} | {s2_metrics['max_consecutive_losses']:>12d}")

    # ── Stage 2 specific: Risk Management Stats ──────────────────────────
    print(f"\n── RiskAgent Impact (Stage 2 only) ──")
    print(f"  Total signals generated:     {s2_metrics.get('total_signals', 'N/A')}")
    print(f"  Rejected (uncertainty):      {s2_metrics.get('rejected_by_uncertainty', 0)}")
    print(f"  Rejected (min position):     {s2_metrics.get('rejected_by_min_position', 0)}")
    print(f"  Trades executed:             {s2_metrics['n_trades']}")

    n_sl = s2_metrics.get("n_stop_loss", 0)
    n_tp = s2_metrics.get("n_take_profit", 0)
    n_hz = s2_metrics.get("n_horizon_exit", 0)
    total_exits = n_sl + n_tp + n_hz
    print(f"\n  Exit Reason Breakdown:")
    print(f"    Stop-loss:    {n_sl:>4d} ({n_sl/max(total_exits,1):.1%})  avg_return={s2_metrics.get('avg_stop_loss_return', 0):+.4f}")
    print(f"    Take-profit:  {n_tp:>4d} ({n_tp/max(total_exits,1):.1%})  avg_return={s2_metrics.get('avg_take_profit_return', 0):+.4f}")
    print(f"    Horizon exit: {n_hz:>4d} ({n_hz/max(total_exits,1):.1%})  avg_return={s2_metrics.get('avg_horizon_return', 0):+.4f}")

    print(f"\n  Position Sizing:")
    print(f"    Avg position:  {s2_metrics.get('avg_position_size', 0):.4f}")
    print(f"    Min position:  {s2_metrics.get('min_position_size', 0):.4f}")
    print(f"    Max position:  {s2_metrics.get('max_position_size', 0):.4f}")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")

    s2_better = []
    s2_worse = []

    if s2_metrics["total_return"] > s1_metrics["total_return"]:
        s2_better.append(f"Higher return ({s2_metrics['total_return']:+.2%} vs {s1_metrics['total_return']:+.2%})")
    else:
        s2_worse.append(f"Lower return ({s2_metrics['total_return']:+.2%} vs {s1_metrics['total_return']:+.2%})")

    if s2_metrics["sharpe_ratio"] > s1_metrics["sharpe_ratio"]:
        s2_better.append(f"Better Sharpe ({s2_metrics['sharpe_ratio']:.4f} vs {s1_metrics['sharpe_ratio']:.4f})")
    else:
        s2_worse.append(f"Worse Sharpe ({s2_metrics['sharpe_ratio']:.4f} vs {s1_metrics['sharpe_ratio']:.4f})")

    if s2_metrics["max_drawdown"] > s1_metrics["max_drawdown"]:  # less negative = better
        s2_better.append(f"Smaller drawdown ({s2_metrics['max_drawdown']:+.2%} vs {s1_metrics['max_drawdown']:+.2%})")
    else:
        s2_worse.append(f"Larger drawdown ({s2_metrics['max_drawdown']:+.2%} vs {s1_metrics['max_drawdown']:+.2%})")

    if s2_metrics["hit_rate"] > s1_metrics["hit_rate"]:
        s2_better.append(f"Higher hit rate ({s2_metrics['hit_rate']:.2%} vs {s1_metrics['hit_rate']:.2%})")
    else:
        s2_worse.append(f"Lower hit rate ({s2_metrics['hit_rate']:.2%} vs {s1_metrics['hit_rate']:.2%})")

    if s2_metrics["profit_factor"] > s1_metrics["profit_factor"]:
        s2_better.append(f"Better profit factor ({s2_metrics['profit_factor']:.4f} vs {s1_metrics['profit_factor']:.4f})")
    else:
        s2_worse.append(f"Worse profit factor ({s2_metrics['profit_factor']:.4f} vs {s1_metrics['profit_factor']:.4f})")

    if len(s2_better) > len(s2_worse):
        print(f"  ✅ VERDICT: RiskAgent is HELPING — Stage 2 outperforms Stage 1")
    elif len(s2_better) == len(s2_worse):
        print(f"  ⚠️  VERDICT: RiskAgent has MIXED impact — some metrics better, some worse")
    else:
        print(f"  ❌ VERDICT: RiskAgent is HURTING — Stage 1 outperforms Stage 2")

    if s2_better:
        print(f"\n  RiskAgent improvements:")
        for item in s2_better:
            print(f"    ✅ {item}")
    if s2_worse:
        print(f"\n  RiskAgent regressions:")
        for item in s2_worse:
            print(f"    ❌ {item}")

    # Specific recommendations
    print(f"\n  Recommendations:")
    if n_sl > 0 and s2_metrics.get("avg_stop_loss_return", 0) < -0.02:
        print(f"    • Stop-loss is working (avg loss = {s2_metrics['avg_stop_loss_return']:+.4f})")
    if n_sl > n_tp * 2:
        print(f"    • Stop-loss triggers too often ({n_sl} vs {n_tp} TP) — consider widening stops")
    if s2_metrics.get("rejected_by_uncertainty", 0) > s2_metrics.get("total_signals", 1) * 0.3:
        print(f"    • Too many signals rejected by uncertainty — consider relaxing threshold")
    if s2_metrics.get("avg_position_size", 0) < 0.05:
        print(f"    • Average position very small ({s2_metrics['avg_position_size']:.4f}) — Kelly may be too conservative")


def print_stage2_trades(trades: list, n: int = 10) -> None:
    """Print sample trades with risk details."""
    if not trades:
        return

    print(f"\n── Sample Trades (first {min(n, len(trades))}) ──")
    print(f"  {'Date':12s} | {'Act':4s} | {'Prob':>6s} | {'Pos%':>6s} | {'Entry':>8s} | {'Exit':>8s} | {'ExitBy':>7s} | {'Return':>8s} | {'Equity':>8s}")
    print(f"  {'-'*12} | {'-'*4} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*8}")
    for t in trades[:n]:
        act = t["action"][:4]
        pos = abs(t.get("position_size", 1.0))
        exit_r = t.get("exit_reason", "horz")[:7]
        print(
            f"  {t['date']:12s} | {act:4s} | {t['probability_up']:6.4f} | {pos:6.2%} | "
            f"{t['entry_price']:8.2f} | {t['exit_price']:8.2f} | {exit_r:>7s} | "
            f"{t['net_return']:+8.4f} | {t['equity']:8.4f}"
        )

    if len(trades) > n:
        print(f"  ... ({len(trades) - n} more trades) ...")
        print(f"\n── Last {min(n, len(trades))} Trades ──")
        print(f"  {'Date':12s} | {'Act':4s} | {'Prob':>6s} | {'Pos%':>6s} | {'Entry':>8s} | {'Exit':>8s} | {'ExitBy':>7s} | {'Return':>8s} | {'Equity':>8s}")
        print(f"  {'-'*12} | {'-'*4} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*8} | {'-'*8}")
        for t in trades[-n:]:
            act = t["action"][:4]
            pos = abs(t.get("position_size", 1.0))
            exit_r = t.get("exit_reason", "horz")[:7]
            print(
                f"  {t['date']:12s} | {act:4s} | {t['probability_up']:6.4f} | {pos:6.2%} | "
                f"{t['entry_price']:8.2f} | {t['exit_price']:8.2f} | {exit_r:>7s} | "
                f"{t['net_return']:+8.4f} | {t['equity']:8.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Stage 2: ForecastAgent + RiskAgent Backtest")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="Holding period in days (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Buy/sell threshold (default: 0.55)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost bps (default: 5.0)")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps (default: 5.0)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    print(f"\n{'='*70}")
    print(f"  Stage 2: ForecastAgent + Simplified RiskAgent Backtest")
    print(f"  Ticker:    {ticker}")
    print(f"  Period:    {args.start} → {args.end}")
    print(f"  Horizon:   {args.horizon}d")
    print(f"  Threshold: {args.threshold}")
    print(f"  Model:     {os.getenv('FORECAST_LGB_MODEL_PATH', 'data/forecast_model.lgb')}")
    print(f"{'='*70}")

    # Load model
    print(f"\n── Loading model ──")
    model, meta, calibrator = load_model()

    # ── Run Stage 1 (baseline) ───────────────────────────────────────────
    print(f"\n── Running Stage 1 baseline ──")
    t0 = time.time()
    s1_result = run_stage1_backtest(
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
        sell_threshold=1.0 - args.threshold,
        verbose=False,
    )
    s1_elapsed = time.time() - t0
    s1_metrics = analyze_backtest(s1_result)
    print(f"  [✓] Stage 1 done in {s1_elapsed:.1f}s — {len(s1_result['trade_log'])} trades, return={s1_result['final_equity']-1:+.2%}")

    # ── Run Stage 2 ──────────────────────────────────────────────────────
    print(f"\n── Running Stage 2 (+ RiskAgent) ──")
    t0 = time.time()
    s2_result = run_stage2_backtest(
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
        sell_threshold=1.0 - args.threshold,
        verbose=args.verbose,
    )
    s2_elapsed = time.time() - t0
    s2_metrics = analyze_stage2(s2_result)
    print(f"  [✓] Stage 2 done in {s2_elapsed:.1f}s — {len(s2_result['trade_log'])} trades, return={s2_result['final_equity']-1:+.2%}")

    # ── Print comparison report ──────────────────────────────────────────
    print_comparison_report(ticker, s1_metrics, s2_metrics, s2_result)
    print_stage2_trades(s2_result["trade_log"])

    # ── Save results ─────────────────────────────────────────────────────
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"stage2_{ticker}_{args.start}_{args.end}_t{args.threshold}"

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
            "kelly_params": {
                "type": "prediction_kelly",
                "avg_win": KELLY_AVG_WIN,
                "avg_loss": KELLY_AVG_LOSS,
            },
            "risk_params": {
                "stop_vol_multiplier": STOP_VOL_MULTIPLIER,
                "risk_reward_ratio": RISK_REWARD_RATIO,
                "uncertainty_high": UNCERTAINTY_HIGH,
                "uncertainty_moderate": UNCERTAINTY_MODERATE,
                "min_position_threshold": MIN_POSITION_THRESHOLD,
            },
        },
        "stage1_metrics": s1_metrics,
        "stage2_metrics": s2_metrics,
        "risk_stats": s2_result.get("risk_stats", {}),
    }
    with open(report_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[✓] Report saved: {report_path}")

    # Save trade log CSV
    trades_path = output_dir / f"{base_name}_trades.csv"
    if s2_result["trade_log"]:
        trades_df = pd.DataFrame(s2_result["trade_log"])
        # Convert risk_flags list to string for CSV
        if "risk_flags" in trades_df.columns:
            trades_df["risk_flags"] = trades_df["risk_flags"].apply(lambda x: "|".join(x) if isinstance(x, list) else str(x))
        trades_df.to_csv(trades_path, index=False)
        print(f"[✓] Trade log saved: {trades_path}")

    # Save equity curves CSV (both stages)
    equity_path = output_dir / f"{base_name}_equity.csv"
    all_dates = sorted(set(list(s1_result["equity_curve"].keys()) + list(s2_result["equity_curve"].keys())))
    eq_rows = []
    s1_eq = 1.0
    s2_eq = 1.0
    for d in all_dates:
        if d in s1_result["equity_curve"]:
            s1_eq = s1_result["equity_curve"][d]
        if d in s2_result["equity_curve"]:
            s2_eq = s2_result["equity_curve"][d]
        eq_rows.append({"date": d, "stage1_equity": s1_eq, "stage2_equity": s2_eq})
    pd.DataFrame(eq_rows).to_csv(equity_path, index=False)
    print(f"[✓] Equity curves saved: {equity_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
