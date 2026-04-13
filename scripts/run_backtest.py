#!/usr/bin/env python3
"""
Backtest CLI Entry Point

Independent CLI for running walk-forward backtests through the real agent
pipeline (Feature → Regime → Forecast → Risk).

Usage:
    python scripts/run_backtest.py --ticker AAPL --start 2023-01-01 --end 2025-12-31
    python scripts/run_backtest.py --ticker AAPL,MSFT --start 2024-01-01 --end 2025-12-31 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.regime_agent import RegimeAgent
from agents.forecast_agent import ForecastAgent
from agents.risk_agent import RiskAgent
from utils.macro_fundamental_provider import MacroFundamentalFeatureProvider
from utils.cross_sectional_service import CrossSectionalFeatureService
from backtest.engine import BacktestEngine
from backtest.evaluator import BacktestEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest through the real agent pipeline.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker(s), comma-separated (e.g., AAPL or AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Holding period in days (default: 5)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost in basis points (default: 5.0)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Warmup days for feature calculation (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtest_results",
        help="Output directory for reports (default: data/backtest_results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def init_agents(verbose: bool = False):
    """Initialize all agents needed for the backtest pipeline."""
    feature_agent = FeatureEngineeringAgent(verbose=verbose)
    regime_agent = RegimeAgent(verbose=verbose)

    # V2: Initialize cross-sectional feature service if model version is v2
    cs_service = None
    if os.getenv("FORECAST_MODEL_VERSION", "v1") == "v2":
        cs_service = CrossSectionalFeatureService(
            ticker_list_path=os.getenv("TRAIN_TICKERS_PATH", "data/sp500_top100.json"),
            cache_dir=os.getenv("CS_CACHE_DIR", "data/cross_section_cache"),
            cache_ttl_hours=float(os.getenv("CS_CACHE_TTL_HOURS", "24")),
            verbose=verbose,
        )

    forecast_agent = ForecastAgent(verbose=verbose, cross_section_service=cs_service)
    risk_agent = RiskAgent(verbose=verbose)
    macro_fund_provider = MacroFundamentalFeatureProvider(verbose=verbose)

    return feature_agent, regime_agent, forecast_agent, risk_agent, macro_fund_provider


def run_single_backtest(
    ticker: str,
    args: argparse.Namespace,
    feature_agent: FeatureEngineeringAgent,
    regime_agent: RegimeAgent,
    forecast_agent: ForecastAgent,
    risk_agent: RiskAgent,
    macro_fund_provider: MacroFundamentalFeatureProvider,
) -> dict:
    """Run backtest for a single ticker and save results."""
    print(f"\n{'='*60}")
    print(f"  Backtest: {ticker}  [{args.start} → {args.end}]")
    print(f"  Horizon: {args.horizon}d | Cost: {args.cost_bps}bps | Slippage: {args.slippage_bps}bps")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Initialize engine
    engine = BacktestEngine(
        feature_agent=feature_agent,
        regime_agent=regime_agent,
        forecast_agent=forecast_agent,
        risk_agent=risk_agent,
        macro_fund_provider=macro_fund_provider,
        horizon_days=args.horizon,
        transaction_cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        verbose=args.verbose,
    )

    # Run backtest
    result = engine.run(
        ticker=ticker,
        start_date=args.start,
        end_date=args.end,
        warmup_days=args.warmup,
    )

    elapsed = time.time() - start_time
    print(f"\n[✓] Backtest completed in {elapsed:.1f}s")

    # Evaluate
    evaluator = BacktestEvaluator()
    report = evaluator.evaluate(result)

    # Print summary
    print(f"\n{report.summary()}")

    # Print per-regime breakdown
    if report.per_regime:
        print(f"\n{'─'*50}")
        print("Per-Regime Breakdown:")
        for state, metrics in sorted(report.per_regime.items()):
            print(
                f"  {state:20s} | trades={metrics['trade_count']:3d} | "
                f"hit_rate={metrics['hit_rate']:.2%} | "
                f"avg_ret={metrics['avg_return']:+.4f} | "
                f"contribution={metrics['contribution_to_total']:+.2%}"
            )

    # Print exit analysis
    exit_info = report.exit_analysis
    print(f"\n{'─'*50}")
    print("Exit Analysis:")
    for reason, pct in exit_info.get("exit_distribution", {}).items():
        avg_ret = exit_info.get("exit_avg_return", {}).get(reason, 0)
        print(f"  {reason:15s} | {pct:.1%} of trades | avg_return={avg_ret:+.4f}")
    print(f"  Stop-loss effectiveness: {exit_info.get('stop_loss_effectiveness', 'n/a')}")

    # Print signal quality
    sq = report.signal_quality
    print(f"\n{'─'*50}")
    print("Signal Quality:")
    print(f"  Total signals: {sq['total_signals']} | Executed: {sq['executed_trades']} | Rejected: {sq['rejected_trades']} ({sq['rejection_rate']:.1%})")
    if sq.get("reject_reasons"):
        for reason, count in sq["reject_reasons"].items():
            print(f"    - {reason}: {count}")

    # Print warnings
    if result.warnings:
        print(f"\n{'─'*50}")
        print("⚠ Warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{ticker}_{args.start}_{args.end}"

    # Save report JSON
    report_path = output_dir / f"{base_name}_report.json"
    report_dict = report.to_dict()
    report_dict["warnings"] = result.warnings
    report_dict["params"] = result.params
    report_dict["ticker"] = result.ticker
    report_dict["start_date"] = result.start_date
    report_dict["end_date"] = result.end_date
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"\n[✓] Report saved: {report_path}")

    # Save trade log CSV
    trades_path = output_dir / f"{base_name}_trades.csv"
    if result.trade_log:
        import pandas as pd

        trades_df = pd.DataFrame(result.trade_log)
        # Convert risk_flags list to string for CSV
        if "risk_flags" in trades_df.columns:
            trades_df["risk_flags"] = trades_df["risk_flags"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else str(x)
            )
        trades_df.to_csv(trades_path, index=False)
        print(f"[✓] Trade log saved: {trades_path}")

    # Save equity curve CSV
    equity_path = output_dir / f"{base_name}_equity.csv"
    if not result.equity_curve.empty:
        eq_df = pd.DataFrame({"date": result.equity_curve.index, "equity": result.equity_curve.values})
        eq_df.to_csv(equity_path, index=False)
        print(f"[✓] Equity curve saved: {equity_path}")

    return report_dict


def main():
    load_dotenv()
    args = parse_args()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    if not tickers:
        print("Error: No valid tickers provided.")
        sys.exit(1)

    print(f"\nInitializing agents...")
    feature_agent, regime_agent, forecast_agent, risk_agent, macro_fund_provider = init_agents(
        verbose=args.verbose
    )

    results = {}
    for ticker in tickers:
        try:
            report = run_single_backtest(
                ticker, args,
                feature_agent, regime_agent, forecast_agent,
                risk_agent, macro_fund_provider,
            )
            results[ticker] = report
        except Exception as exc:
            print(f"\n[✗] Backtest failed for {ticker}: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results[ticker] = {"error": str(exc)}

    # Final summary
    if len(tickers) > 1:
        print(f"\n{'='*60}")
        print("  Multi-Ticker Summary")
        print(f"{'='*60}")
        for ticker, report in results.items():
            if "error" in report:
                print(f"  {ticker}: ERROR - {report['error']}")
            else:
                overall = report.get("overall", {})
                print(
                    f"  {ticker}: return={overall.get('total_return', 0):.2%}, "
                    f"Sharpe={overall.get('sharpe_ratio', 0):.2f}, "
                    f"alpha={overall.get('alpha', 0):.2%}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
