#!/usr/bin/env python3
"""
Live Trading Assistant
=====================
Bridges the backtest system to real-world manual trading.

Usage:
  # Generate a new signal for AAPL
  python scripts/live_trading_assistant.py signal AAPL

  # Check all open positions (stop-loss / take-profit / expiry)
  python scripts/live_trading_assistant.py monitor

  # View current portfolio status
  python scripts/live_trading_assistant.py status

  # Record that you executed a trade (after reviewing the signal)
  python scripts/live_trading_assistant.py execute AAPL

  # Record that you manually closed a position
  python scripts/live_trading_assistant.py close AAPL --price 210.50

  # Generate signals for a watchlist
  python scripts/live_trading_assistant.py scan AAPL MSFT GOOGL NVDA

  # Clear expired / closed positions from ledger
  python scripts/live_trading_assistant.py cleanup
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
import pandas as pd

from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.regime_agent import RegimeAgent
from agents.forecast_agent import ForecastAgent
from agents.risk_agent import RiskAgent
from utils.macro_fundamental_provider import MacroFundamentalFeatureProvider
from utils.yfinance_cache import get_historical_data
from utils.cross_sectional_service import CrossSectionalFeatureService

# ── Position ledger file ─────────────────────────────────────────────────
POSITION_LEDGER_PATH = PROJECT_ROOT / "data" / "live_positions.json"
SIGNAL_LOG_PATH = PROJECT_ROOT / "data" / "live_signal_log.json"


# ═══════════════════════════════════════════════════════════════════════════
# Position Ledger Management
# ═══════════════════════════════════════════════════════════════════════════

def _load_positions() -> List[Dict[str, Any]]:
    """Load open positions from the ledger file."""
    if not POSITION_LEDGER_PATH.exists():
        return []
    with open(POSITION_LEDGER_PATH, "r") as f:
        return json.load(f)


def _save_positions(positions: List[Dict[str, Any]]) -> None:
    """Save positions to the ledger file."""
    POSITION_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITION_LEDGER_PATH, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def _load_signal_log() -> List[Dict[str, Any]]:
    """Load signal history log."""
    if not SIGNAL_LOG_PATH.exists():
        return []
    with open(SIGNAL_LOG_PATH, "r") as f:
        return json.load(f)


def _save_signal_log(log: List[Dict[str, Any]]) -> None:
    """Save signal history log."""
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


def _get_current_price(ticker: str) -> Optional[float]:
    """Fetch the latest closing price for a ticker."""
    try:
        data = get_historical_data(ticker, interval="daily", outputsize="compact")
        if data.empty:
            return None
        close = pd.to_numeric(data["Close"], errors="coerce")
        return float(close.iloc[-1])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Agent Pipeline (reuses existing agents)
# ═══════════════════════════════════════════════════════════════════════════

def _init_agents(verbose: bool = False):
    """Initialize the quantitative agent pipeline."""
    cs_service = None
    if os.getenv("FORECAST_MODEL_VERSION", "v1") == "v2":
        cs_service = CrossSectionalFeatureService(
            ticker_list_path=os.getenv("TRAIN_TICKERS_PATH", "data/sp500_top100.json"),
            cache_dir=os.getenv("CS_CACHE_DIR", "data/cross_section_cache"),
            cache_ttl_hours=float(os.getenv("CS_CACHE_TTL_HOURS", "24")),
            verbose=verbose,
        )

    feature_agent = FeatureEngineeringAgent(verbose=verbose)
    regime_agent = RegimeAgent(verbose=verbose)
    forecast_agent = ForecastAgent(verbose=verbose, cross_section_service=cs_service)
    risk_agent = RiskAgent(verbose=verbose)
    macro_fund_provider = MacroFundamentalFeatureProvider(verbose=verbose)

    return feature_agent, regime_agent, forecast_agent, risk_agent, macro_fund_provider


def _generate_signal(
    ticker: str,
    feature_agent: FeatureEngineeringAgent,
    regime_agent: RegimeAgent,
    forecast_agent: ForecastAgent,
    risk_agent: RiskAgent,
    macro_fund_provider: MacroFundamentalFeatureProvider,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full agent pipeline and return a structured signal."""
    ticker = ticker.upper().strip()

    # Step 1: Features
    feature_result = feature_agent.analyze(ticker)
    features = feature_result.get("features", {})

    # Step 2: Macro/Fundamental
    try:
        macro_features = macro_fund_provider.extract(ticker)
    except Exception as exc:
        if verbose:
            print(f"  [warn] Macro data fetch failed: {exc}")
        macro_features = {}

    # Step 3: Build feature history for regime smoothing
    data = get_historical_data(ticker, interval="daily", outputsize="full")
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    feature_history = []
    lookback = 60
    total_bars = len(data)
    for offset in range(min(lookback, total_bars - 60), 0, -5):
        idx = total_bars - offset
        data_slice = data.iloc[:idx]
        if len(data_slice) < 60:
            continue
        try:
            res = feature_agent.analyze(ticker, data_override=data_slice)
            fts = res.get("features", {})
            if fts:
                feature_history.append(fts)
        except Exception:
            pass

    # Step 4: Regime
    regime_result = regime_agent.analyze(
        ticker, feature_result, macro_features, feature_history=feature_history,
    )

    # Step 5: Forecast
    forecast_result = forecast_agent.analyze(
        ticker, feature_result, regime_result, macro_features,
    )

    # Step 6: Risk plan (no memory in live mode — use defaults)
    memory_result = {
        "agent": "memory",
        "status": "skipped",
        "memory": {},
        "track_record_factor": 1.0,
        "summary": "Live mode — no simulated memory.",
    }
    risk_result = risk_agent.analyze(
        ticker, forecast_result, regime_result, feature_result, memory_result, macro_features,
    )

    # Assemble signal
    forecast = forecast_result.get("forecast", {})
    risk_plan = risk_result.get("risk_plan", {})
    regime = regime_result.get("regime", {})

    current_price = _get_current_price(ticker)
    horizon_days = int(forecast.get("horizon_days", 5))

    # Stop-loss / take-profit are DYNAMICALLY computed by RiskAgent:
    #   daily_vol = volatility_20 / sqrt(252)
    #   base_stop = daily_vol * 2.5
    #   stop_loss_pct = clamp(base_stop, 0.01, 0.08)
    #   take_profit_pct = stop_loss_pct * risk_reward_ratio (default 2x)
    stop_loss_pct = risk_plan.get("stop_loss_pct", 0.05)
    take_profit_pct = risk_plan.get("take_profit_pct", 0.10)
    position_size = risk_plan.get("position_size_fraction", 0.0)
    action = forecast.get("action", "hold")

    # Extract volatility for display (shows calculation basis)
    volatility_20 = features.get("volatility_20")

    stop_loss_price = None
    take_profit_price = None
    if current_price and position_size != 0:
        if position_size > 0:  # Long
            stop_loss_price = round(current_price * (1 - stop_loss_pct), 2)
            take_profit_price = round(current_price * (1 + take_profit_pct), 2)
        else:  # Short
            stop_loss_price = round(current_price * (1 + stop_loss_pct), 2)
            take_profit_price = round(current_price * (1 - take_profit_pct), 2)

    expiry_date = (datetime.now() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")

    signal = {
        "ticker": ticker,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": current_price,
        "action": action,
        "probability_up": forecast.get("probability_up"),
        "raw_probability_up": forecast.get("raw_probability_up"),
        "calibrated": forecast.get("calibrated", False),
        "position_size_fraction": position_size,
        "direction": "LONG" if position_size > 0 else ("SHORT" if position_size < 0 else "FLAT"),
        "regime_state": regime.get("state", "unknown"),
        "regime_confidence": regime.get("confidence"),
        "signal_alignment": risk_plan.get("signal_alignment"),
        "kelly_fraction": risk_plan.get("kelly_fraction"),
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "horizon_days": horizon_days,
        "expiry_date": expiry_date,
        "risk_flags": risk_plan.get("risk_flags", []),
        "reject_reason": risk_plan.get("reject_reason"),
        "uncertainty": forecast.get("uncertainty"),
        "prediction_set": forecast.get("prediction_set"),
        "is_uncertain": forecast.get("is_uncertain", False),
        "model_source": forecast.get("model_source"),
        "volatility_20": volatility_20,
    }

    return signal


# ═══════════════════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _format_signal(signal: Dict[str, Any]) -> str:
    """Format a signal into a human-readable trading plan."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  📊 TRADING SIGNAL — {signal['ticker']}")
    lines.append(f"  Generated: {signal['timestamp']}")
    lines.append("=" * 70)

    # Rejection check
    if signal.get("reject_reason"):
        lines.append(f"\n  ❌ SIGNAL REJECTED: {signal['reject_reason']}")
        lines.append(f"     Action: {signal['action'].upper()}")
        lines.append(f"     Prob(up): {signal['probability_up']:.4f}")
        lines.append(f"     Regime: {signal['regime_state']}")
        lines.append(f"     Risk Flags: {', '.join(signal.get('risk_flags', []))}")
        lines.append("\n  → No trade recommended. Wait for next signal.")
        lines.append("=" * 70)
        return "\n".join(lines)

    # Active signal
    action = signal["action"].upper()
    direction = signal["direction"]
    emoji = "🟢" if direction == "LONG" else ("🔴" if direction == "SHORT" else "⚪")

    lines.append(f"\n  {emoji} ACTION: {action} ({direction})")
    lines.append(f"  Current Price: ${signal['current_price']:.2f}" if signal['current_price'] else "  Current Price: N/A")
    lines.append(f"  Position Size: {abs(signal['position_size_fraction']):.1%} of portfolio")

    lines.append(f"\n  📈 Model Confidence:")
    lines.append(f"     Prob(up): {signal['probability_up']:.4f}" + (" [calibrated]" if signal.get('calibrated') else " [raw]"))
    if signal.get('uncertainty') is not None:
        unc_flag = " ⚠️ HIGH" if signal.get('is_uncertain') else ""
        lines.append(f"     Uncertainty: {signal['uncertainty']:.4f}{unc_flag}")
    if signal.get('prediction_set'):
        lines.append(f"     Conformal Set: {signal['prediction_set']}")
    lines.append(f"     Model: {signal.get('model_source', 'unknown')}")

    lines.append(f"\n  🌍 Market Context:")
    lines.append(f"     Regime: {signal['regime_state']}")
    if signal.get('regime_confidence') is not None:
        lines.append(f"     Regime Confidence: {signal['regime_confidence']:.2f}")
    lines.append(f"     Signal Alignment: {signal.get('signal_alignment', 'N/A')}")
    lines.append(f"     Kelly Fraction: {signal.get('kelly_fraction', 'N/A')}")

    lines.append(f"\n  🎯 EXECUTION PLAN (dynamic, volatility-based):")
    vol20 = signal.get('volatility_20')
    if vol20 is not None:
        import math as _math
        daily_vol = vol20 / _math.sqrt(252)
        lines.append(f"     Vol20={vol20:.1%} → DailyVol={daily_vol:.2%} → SL=DailyVol×2.5, TP=SL×2")
    lines.append(f"     ┌─────────────────────────────────────────┐")
    if signal['current_price']:
        lines.append(f"     │ Entry Price:     ~${signal['current_price']:.2f} (next open)    │")
    if signal.get('stop_loss_price'):
        lines.append(f"     │ Stop Loss:        ${signal['stop_loss_price']:.2f} ({signal['stop_loss_pct']:.1%})     │")
    if signal.get('take_profit_price'):
        lines.append(f"     │ Take Profit:      ${signal['take_profit_price']:.2f} ({signal['take_profit_pct']:.1%})     │")
    lines.append(f"     │ Holding Period:   {signal['horizon_days']} trading days         │")
    lines.append(f"     │ Expiry Date:      {signal['expiry_date']}              │")
    lines.append(f"     └─────────────────────────────────────────┘")

    if signal.get("risk_flags"):
        lines.append(f"\n  ⚠️  Risk Flags: {', '.join(signal['risk_flags'])}")

    lines.append(f"\n  💡 Instructions:")
    lines.append(f"     1. Place a {'BUY' if direction == 'LONG' else 'SELL/SHORT'} order at market open")
    lines.append(f"     2. Set stop-loss at ${signal.get('stop_loss_price', 'N/A')}")
    lines.append(f"     3. Set take-profit at ${signal.get('take_profit_price', 'N/A')}")
    lines.append(f"     4. If neither triggers, close position on {signal['expiry_date']}")
    lines.append(f"     5. Run 'monitor' command daily to check status")
    lines.append("=" * 70)

    return "\n".join(lines)


def _format_position_status(pos: Dict[str, Any], current_price: Optional[float]) -> str:
    """Format a single position's status."""
    lines = []
    ticker = pos["ticker"]
    direction = pos["direction"]
    entry_price = pos["entry_price"]
    stop_loss = pos["stop_loss_price"]
    take_profit = pos["take_profit_price"]
    expiry = pos["expiry_date"]

    emoji = "🟢" if direction == "LONG" else "🔴"
    lines.append(f"  {emoji} {ticker} — {direction}")
    lines.append(f"     Entry: ${entry_price:.2f} on {pos['entry_date']}")

    if current_price:
        if direction == "LONG":
            pnl = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) / entry_price
        pnl_emoji = "📈" if pnl > 0 else "📉"
        lines.append(f"     Current: ${current_price:.2f} ({pnl_emoji} {pnl:+.2%})")

        # Check stop-loss / take-profit
        if direction == "LONG":
            if current_price <= stop_loss:
                lines.append(f"     🚨 STOP-LOSS TRIGGERED! Close position NOW (SL=${stop_loss:.2f})")
            elif current_price >= take_profit:
                lines.append(f"     🎯 TAKE-PROFIT REACHED! Close position NOW (TP=${take_profit:.2f})")
            else:
                dist_to_sl = (current_price - stop_loss) / current_price
                dist_to_tp = (take_profit - current_price) / current_price
                lines.append(f"     Stop Loss: ${stop_loss:.2f} ({dist_to_sl:.1%} away)")
                lines.append(f"     Take Profit: ${take_profit:.2f} ({dist_to_tp:.1%} away)")
        else:  # SHORT
            if current_price >= stop_loss:
                lines.append(f"     🚨 STOP-LOSS TRIGGERED! Close position NOW (SL=${stop_loss:.2f})")
            elif current_price <= take_profit:
                lines.append(f"     🎯 TAKE-PROFIT REACHED! Close position NOW (TP=${take_profit:.2f})")
            else:
                dist_to_sl = (stop_loss - current_price) / current_price
                dist_to_tp = (current_price - take_profit) / current_price
                lines.append(f"     Stop Loss: ${stop_loss:.2f} ({dist_to_sl:.1%} away)")
                lines.append(f"     Take Profit: ${take_profit:.2f} ({dist_to_tp:.1%} away)")
    else:
        lines.append(f"     Current: N/A (price fetch failed)")
        lines.append(f"     Stop Loss: ${stop_loss:.2f}")
        lines.append(f"     Take Profit: ${take_profit:.2f}")

    # Check expiry
    today = datetime.now().strftime("%Y-%m-%d")
    if today >= expiry:
        lines.append(f"     ⏰ EXPIRED! Holding period ended on {expiry}. Close position.")
    else:
        days_left = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
        lines.append(f"     Expiry: {expiry} ({days_left} days left)")

    lines.append(f"     Position Size: {abs(pos.get('position_size_fraction', 0)):.1%}")
    lines.append(f"     Regime at Entry: {pos.get('regime_state', 'N/A')}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════

def cmd_signal(args):
    """Generate a trading signal for a ticker."""
    ticker = args.ticker.upper()
    verbose = args.verbose

    print(f"\n  Generating signal for {ticker}...")
    print(f"  Running: Feature → Regime → Forecast → Risk pipeline\n")

    agents = _init_agents(verbose=verbose)
    signal = _generate_signal(ticker, *agents, verbose=verbose)

    # Display
    print(_format_signal(signal))

    # Log signal
    log = _load_signal_log()
    log.append(signal)
    # Keep last 200 signals
    if len(log) > 200:
        log = log[-200:]
    _save_signal_log(log)

    # If signal is actionable, ask if user wants to record it as a pending trade
    if signal.get("position_size_fraction", 0) != 0 and not signal.get("reject_reason"):
        print("\n  To record this as an open position after you execute the trade:")
        print(f"  python scripts/live_trading_assistant.py execute {ticker}")

    return signal


def cmd_execute(args):
    """Record that you executed a trade based on the latest signal."""
    ticker = args.ticker.upper()

    # Find the latest signal for this ticker
    log = _load_signal_log()
    latest = None
    for s in reversed(log):
        if s["ticker"] == ticker and not s.get("reject_reason") and s.get("position_size_fraction", 0) != 0:
            latest = s
            break

    if not latest:
        print(f"\n  ❌ No actionable signal found for {ticker}.")
        print(f"     Run 'signal {ticker}' first to generate one.")
        return

    # Get entry price
    if args.price:
        entry_price = args.price
    else:
        entry_price = _get_current_price(ticker)
        if entry_price is None:
            print(f"\n  ❌ Could not fetch current price for {ticker}.")
            print(f"     Use --price to specify manually.")
            return

    # Recalculate stop/take-profit based on actual entry price
    stop_loss_pct = latest["stop_loss_pct"]
    take_profit_pct = latest["take_profit_pct"]
    direction = latest["direction"]

    if direction == "LONG":
        stop_loss_price = round(entry_price * (1 - stop_loss_pct), 2)
        take_profit_price = round(entry_price * (1 + take_profit_pct), 2)
    else:
        stop_loss_price = round(entry_price * (1 + stop_loss_pct), 2)
        take_profit_price = round(entry_price * (1 - take_profit_pct), 2)

    position = {
        "ticker": ticker,
        "direction": direction,
        "entry_price": entry_price,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "expiry_date": latest["expiry_date"],
        "horizon_days": latest["horizon_days"],
        "position_size_fraction": latest["position_size_fraction"],
        "probability_up": latest["probability_up"],
        "regime_state": latest.get("regime_state"),
        "signal_alignment": latest.get("signal_alignment"),
        "risk_flags": latest.get("risk_flags", []),
        "status": "open",
    }

    positions = _load_positions()
    # Remove any existing open position for this ticker
    positions = [p for p in positions if not (p["ticker"] == ticker and p["status"] == "open")]
    positions.append(position)
    _save_positions(positions)

    print(f"\n  ✅ Position recorded for {ticker}:")
    print(f"     Direction: {direction}")
    print(f"     Entry: ${entry_price:.2f}")
    print(f"     Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct:.1%})")
    print(f"     Take Profit: ${take_profit_price:.2f} ({take_profit_pct:.1%})")
    print(f"     Expiry: {latest['expiry_date']}")
    print(f"\n  Run 'monitor' daily to check stop-loss / take-profit / expiry.")


def cmd_close(args):
    """Record that you closed a position."""
    ticker = args.ticker.upper()
    positions = _load_positions()

    open_pos = [p for p in positions if p["ticker"] == ticker and p["status"] == "open"]
    if not open_pos:
        print(f"\n  ❌ No open position found for {ticker}.")
        return

    pos = open_pos[0]
    exit_price = args.price
    if exit_price is None:
        exit_price = _get_current_price(ticker)
        if exit_price is None:
            print(f"\n  ❌ Could not fetch current price. Use --price to specify.")
            return

    # Calculate P&L
    entry_price = pos["entry_price"]
    if pos["direction"] == "LONG":
        raw_return = (exit_price - entry_price) / entry_price
    else:
        raw_return = (entry_price - exit_price) / entry_price

    position_return = raw_return * abs(pos.get("position_size_fraction", 1.0))

    # Determine exit reason
    if pos["direction"] == "LONG":
        if exit_price <= pos["stop_loss_price"]:
            exit_reason = "stop_loss"
        elif exit_price >= pos["take_profit_price"]:
            exit_reason = "take_profit"
        else:
            exit_reason = "manual"
    else:
        if exit_price >= pos["stop_loss_price"]:
            exit_reason = "stop_loss"
        elif exit_price <= pos["take_profit_price"]:
            exit_reason = "take_profit"
        else:
            exit_reason = "manual"

    # Update position
    pos["status"] = "closed"
    pos["exit_price"] = exit_price
    pos["exit_date"] = datetime.now().strftime("%Y-%m-%d")
    pos["exit_reason"] = exit_reason
    pos["raw_return"] = round(raw_return, 6)
    pos["position_return"] = round(position_return, 6)

    _save_positions(positions)

    pnl_emoji = "📈" if raw_return > 0 else "📉"
    print(f"\n  ✅ Position closed for {ticker}:")
    print(f"     Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
    print(f"     {pnl_emoji} Return: {raw_return:+.2%} (position-weighted: {position_return:+.2%})")
    print(f"     Exit Reason: {exit_reason}")


def cmd_monitor(args):
    """Monitor all open positions for stop-loss / take-profit / expiry."""
    positions = _load_positions()
    open_positions = [p for p in positions if p["status"] == "open"]

    if not open_positions:
        print("\n  📭 No open positions to monitor.")
        print("     Generate a signal with: python scripts/live_trading_assistant.py signal AAPL")
        return

    print("\n" + "=" * 70)
    print("  📋 POSITION MONITOR")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    alerts = []

    for pos in open_positions:
        ticker = pos["ticker"]
        current_price = _get_current_price(ticker)
        print()
        print(_format_position_status(pos, current_price))

        # Check for alerts
        if current_price:
            direction = pos["direction"]
            sl = pos["stop_loss_price"]
            tp = pos["take_profit_price"]

            if direction == "LONG":
                if current_price <= sl:
                    alerts.append(f"🚨 {ticker}: STOP-LOSS HIT (${current_price:.2f} ≤ ${sl:.2f})")
                elif current_price >= tp:
                    alerts.append(f"🎯 {ticker}: TAKE-PROFIT HIT (${current_price:.2f} ≥ ${tp:.2f})")
            else:
                if current_price >= sl:
                    alerts.append(f"🚨 {ticker}: STOP-LOSS HIT (${current_price:.2f} ≥ ${sl:.2f})")
                elif current_price <= tp:
                    alerts.append(f"🎯 {ticker}: TAKE-PROFIT HIT (${current_price:.2f} ≤ ${tp:.2f})")

        today = datetime.now().strftime("%Y-%m-%d")
        if today >= pos["expiry_date"]:
            alerts.append(f"⏰ {ticker}: HOLDING PERIOD EXPIRED (expiry={pos['expiry_date']})")

    if alerts:
        print("\n" + "=" * 70)
        print("  🔔 ACTION REQUIRED:")
        print("=" * 70)
        for alert in alerts:
            print(f"  {alert}")
        print(f"\n  Close positions with: python scripts/live_trading_assistant.py close <TICKER> --price <PRICE>")
    else:
        print(f"\n  ✅ All positions within normal range. No action needed.")

    print("=" * 70)


def cmd_status(args):
    """Show portfolio status summary."""
    positions = _load_positions()
    open_positions = [p for p in positions if p["status"] == "open"]
    closed_positions = [p for p in positions if p["status"] == "closed"]

    print("\n" + "=" * 70)
    print("  📊 PORTFOLIO STATUS")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n  Open Positions: {len(open_positions)}")
    print(f"  Closed Positions: {len(closed_positions)}")

    if open_positions:
        print(f"\n  ── Open Positions ──")
        total_unrealized = 0.0
        for pos in open_positions:
            ticker = pos["ticker"]
            current_price = _get_current_price(ticker)
            entry = pos["entry_price"]
            direction = pos["direction"]

            if current_price:
                if direction == "LONG":
                    pnl = (current_price - entry) / entry
                else:
                    pnl = (entry - current_price) / entry
                total_unrealized += pnl * abs(pos.get("position_size_fraction", 1.0))
                pnl_str = f"{pnl:+.2%}"
            else:
                pnl_str = "N/A"

            print(f"     {direction:5s} {ticker:6s}  Entry=${entry:.2f}  Current=${current_price:.2f if current_price else 0:.2f}  P&L={pnl_str}")

        print(f"\n     Total Unrealized (position-weighted): {total_unrealized:+.2%}")

    if closed_positions:
        print(f"\n  ── Recent Closed Positions (last 10) ──")
        total_realized = 0.0
        wins = 0
        for pos in closed_positions[-10:]:
            ret = pos.get("raw_return", 0)
            total_realized += pos.get("position_return", 0)
            if ret > 0:
                wins += 1
            emoji = "✅" if ret > 0 else "❌"
            print(
                f"     {emoji} {pos['direction']:5s} {pos['ticker']:6s}  "
                f"${pos['entry_price']:.2f}→${pos.get('exit_price', 0):.2f}  "
                f"{ret:+.2%}  ({pos.get('exit_reason', 'N/A')})"
            )

        n_closed = len(closed_positions)
        total_wins = sum(1 for p in closed_positions if p.get("raw_return", 0) > 0)
        total_pnl = sum(p.get("position_return", 0) for p in closed_positions)
        hit_rate = total_wins / n_closed if n_closed > 0 else 0

        print(f"\n     All-time: {n_closed} trades, Hit Rate={hit_rate:.1%}, Total P&L={total_pnl:+.2%}")

    print("=" * 70)


def cmd_scan(args):
    """Scan multiple tickers and generate signals."""
    tickers = [t.upper() for t in args.tickers]
    verbose = args.verbose

    print(f"\n  Scanning {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"  Running: Feature → Regime → Forecast → Risk pipeline\n")

    agents = _init_agents(verbose=verbose)

    actionable = []
    rejected = []

    for ticker in tickers:
        print(f"  Processing {ticker}...", end="", flush=True)
        try:
            signal = _generate_signal(ticker, *agents, verbose=verbose)
            if signal.get("reject_reason") or signal.get("position_size_fraction", 0) == 0:
                rejected.append(signal)
                print(f" → REJECTED ({signal.get('reject_reason', 'no edge')})")
            else:
                actionable.append(signal)
                direction = signal["direction"]
                prob = signal["probability_up"]
                size = abs(signal["position_size_fraction"])
                print(f" → {direction} (prob={prob:.3f}, size={size:.1%})")

            # Log
            log = _load_signal_log()
            log.append(signal)
            if len(log) > 200:
                log = log[-200:]
            _save_signal_log(log)
        except Exception as exc:
            print(f" → ERROR: {exc}")

    # Summary
    print("\n" + "=" * 70)
    print(f"  📊 SCAN RESULTS: {len(actionable)} actionable / {len(rejected)} rejected")
    print("=" * 70)

    if actionable:
        print("\n  🟢 Actionable Signals:")
        # Sort by absolute position size (strongest first)
        actionable.sort(key=lambda s: abs(s.get("position_size_fraction", 0)), reverse=True)
        for s in actionable:
            print(f"\n{'─' * 60}")
            print(_format_signal(s))

    if rejected:
        print(f"\n  ❌ Rejected ({len(rejected)}):")
        for s in rejected:
            reason = s.get("reject_reason", "no_edge")
            print(f"     {s['ticker']:6s}  action={s['action']:4s}  prob={s.get('probability_up', 0):.3f}  reason={reason}")


def cmd_cleanup(args):
    """Remove closed and expired positions from the ledger."""
    positions = _load_positions()
    before = len(positions)

    # Auto-close expired positions
    today = datetime.now().strftime("%Y-%m-%d")
    for pos in positions:
        if pos["status"] == "open" and today >= pos["expiry_date"]:
            current_price = _get_current_price(pos["ticker"])
            if current_price:
                entry = pos["entry_price"]
                if pos["direction"] == "LONG":
                    raw_return = (current_price - entry) / entry
                else:
                    raw_return = (entry - current_price) / entry
                pos["status"] = "closed"
                pos["exit_price"] = current_price
                pos["exit_date"] = today
                pos["exit_reason"] = "horizon_expired"
                pos["raw_return"] = round(raw_return, 6)
                pos["position_return"] = round(raw_return * abs(pos.get("position_size_fraction", 1.0)), 6)
                print(f"  ⏰ Auto-closed {pos['ticker']}: {raw_return:+.2%}")

    if args.archive:
        # Remove closed positions entirely
        positions = [p for p in positions if p["status"] == "open"]

    _save_positions(positions)
    after = len(positions)
    print(f"\n  Cleanup complete: {before} → {after} positions")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Live Trading Assistant — Manual trading with agent signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/live_trading_assistant.py signal AAPL
  python scripts/live_trading_assistant.py monitor
  python scripts/live_trading_assistant.py status
  python scripts/live_trading_assistant.py execute AAPL --price 210.50
  python scripts/live_trading_assistant.py close AAPL --price 215.00
  python scripts/live_trading_assistant.py scan AAPL MSFT GOOGL NVDA
  python scripts/live_trading_assistant.py cleanup --archive
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # signal
    p_signal = subparsers.add_parser("signal", help="Generate a trading signal")
    p_signal.add_argument("ticker", type=str, help="Stock ticker symbol")
    p_signal.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_signal.set_defaults(func=cmd_signal)

    # monitor
    p_monitor = subparsers.add_parser("monitor", help="Monitor open positions")
    p_monitor.set_defaults(func=cmd_monitor)

    # status
    p_status = subparsers.add_parser("status", help="Portfolio status summary")
    p_status.set_defaults(func=cmd_status)

    # execute
    p_execute = subparsers.add_parser("execute", help="Record a trade execution")
    p_execute.add_argument("ticker", type=str, help="Stock ticker symbol")
    p_execute.add_argument("--price", type=float, help="Actual entry price (default: current price)")
    p_execute.set_defaults(func=cmd_execute)

    # close
    p_close = subparsers.add_parser("close", help="Record closing a position")
    p_close.add_argument("ticker", type=str, help="Stock ticker symbol")
    p_close.add_argument("--price", type=float, help="Actual exit price (default: current price)")
    p_close.set_defaults(func=cmd_close)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan multiple tickers")
    p_scan.add_argument("tickers", nargs="+", type=str, help="Ticker symbols to scan")
    p_scan.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_scan.set_defaults(func=cmd_scan)

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Clean up expired positions")
    p_cleanup.add_argument("--archive", action="store_true", help="Remove closed positions from ledger")
    p_cleanup.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
