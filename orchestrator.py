"""
Pipeline orchestrator shared by CLI and frontend dashboards.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.historical_agent import HistoricalAnalysisAgent
from agents.indicator_agent import IndicatorAnalysisAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.regime_agent import RegimeAgent
from agents.forecast_agent import ForecastAgent
from agents.risk_agent import RiskAgent
from agents.backtest_agent import BacktestAgent
from agents.ledger_agent import PairLedgerAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.pair_monitor_agent import PairMonitorAgent
from agents.supervisor_agent import SupervisorAgent
from utils.storage import Storage

ProgressCallback = Callable[[int, int, str], None]


def _emit_progress(
    callback: ProgressCallback | None,
    step: int,
    total: int,
    message: str,
) -> None:
    if callback:
        callback(step, total, message)


def _build_report_file(
    stock_symbol: str,
    final_report: str,
    historical_result: Dict[str, Any],
    indicator_result: Dict[str, Any],
    news_result: Dict[str, Any],
    pair_monitor_result: Dict[str, Any],
    feature_result: Dict[str, Any],
    regime_result: Dict[str, Any],
    forecast_result: Dict[str, Any],
    risk_result: Dict[str, Any],
    backtest_result: Dict[str, Any],
) -> str:
    output_file = f"report_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, "w") as f:
        f.write(final_report)
        f.write("\n\nDETAILED ANALYSES:\n")
        f.write("=" * 80 + "\n\n")
        f.write("HISTORICAL ANALYSIS:\n")
        f.write(historical_result.get("analysis", "") + "\n\n")
        f.write("INDICATOR ANALYSIS:\n")
        f.write(indicator_result.get("analysis", "") + "\n\n")
        f.write("NEWS SENTIMENT ANALYSIS:\n")
        f.write(news_result.get("analysis", "") + "\n")
        f.write("\nPAIR MONITOR ANALYSIS:\n")
        f.write(pair_monitor_result.get("summary", "") + "\n")
        f.write(json.dumps(pair_monitor_result.get("signals", []), indent=2) + "\n")
        f.write("\nFEATURE ENGINEERING:\n")
        f.write(feature_result.get("summary", "") + "\n")
        f.write(json.dumps(feature_result.get("features", {}), indent=2) + "\n")
        f.write("\nREGIME ANALYSIS:\n")
        f.write(regime_result.get("summary", "") + "\n")
        f.write(json.dumps(regime_result.get("regime", {}), indent=2) + "\n")
        f.write("\nFORECAST ANALYSIS:\n")
        f.write(forecast_result.get("summary", "") + "\n")
        f.write(json.dumps(forecast_result.get("forecast", {}), indent=2) + "\n")
        f.write("\nRISK PLAN:\n")
        f.write(risk_result.get("summary", "") + "\n")
        f.write(json.dumps(risk_result.get("risk_plan", {}), indent=2) + "\n")
        f.write("\nBACKTEST SNAPSHOT:\n")
        f.write(backtest_result.get("summary", "") + "\n")
        f.write(json.dumps(backtest_result.get("metrics", {}), indent=2) + "\n")
    return output_file


def run_full_analysis(
    stock_symbol: str,
    *,
    verbose: bool = False,
    persist: bool = True,
    save_report: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """
    Execute the full 10-stage financial analysis pipeline.
    """
    total_steps = 10
    symbol = stock_symbol.strip().upper()
    if not symbol:
        return {
            "status": "error",
            "error": "No stock symbol provided.",
        }

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": "OPENAI_API_KEY not found in environment variables.",
        }

    _emit_progress(progress_callback, 1, total_steps, "Initializing models and agents")
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        api_key=api_key,
    )

    historical_agent = HistoricalAnalysisAgent(llm, verbose=verbose)
    indicator_agent = IndicatorAnalysisAgent(llm, verbose=verbose)
    news_agent = NewsSentimentAgent(llm, verbose=verbose)
    ledger_agent = PairLedgerAgent(verbose=verbose)
    pair_monitor_agent = PairMonitorAgent(verbose=verbose)
    feature_agent = FeatureEngineeringAgent(verbose=verbose)
    regime_agent = RegimeAgent(verbose=verbose)
    forecast_agent = ForecastAgent(verbose=verbose)
    risk_agent = RiskAgent(verbose=verbose)
    backtest_agent = BacktestAgent(verbose=verbose)
    supervisor_agent = SupervisorAgent(llm)
    print("Initializing agents...")

    storage = None
    storage_enabled = os.getenv("STORAGE_ENABLED", "true").lower() == "true"
    if persist and storage_enabled:
        try:
            storage = Storage()
        except Exception:
            storage = None

    print("Analyzing pair ledger...")
    pair_ledger = ledger_agent.analyze()
    print("Pair ledger analyzed:", pair_ledger)

    _emit_progress(progress_callback, 2, total_steps, "Running historical analysis")
    historical_result = historical_agent.analyze(symbol)

    _emit_progress(progress_callback, 3, total_steps, "Running indicator analysis")
    indicator_result = indicator_agent.analyze(symbol)

    _emit_progress(progress_callback, 4, total_steps, "Running news sentiment analysis")
    news_result = news_agent.analyze(symbol)

    _emit_progress(progress_callback, 5, total_steps, "Running pair monitor analysis")
    if pair_ledger.get("status") == "success":
        pair_monitor_result = pair_monitor_agent.analyze(
            pair_ledger.get("pairs", []),
            focus_symbol=symbol,
        )
    else:
        pair_monitor_result = {
            "agent": "pair_monitor",
            "status": "skipped",
            "summary": "Pair ledger unavailable; monitoring skipped.",
            "signals": [],
        }

    _emit_progress(progress_callback, 6, total_steps, "Building quantitative features")
    feature_result = feature_agent.analyze(symbol)

    _emit_progress(progress_callback, 7, total_steps, "Classifying market regime")
    regime_result = regime_agent.analyze(symbol, feature_result)

    _emit_progress(progress_callback, 8, total_steps, "Generating probabilistic forecast")
    forecast_result = forecast_agent.analyze(symbol, feature_result, regime_result)

    _emit_progress(progress_callback, 9, total_steps, "Generating risk plan and backtest snapshot")
    risk_result = risk_agent.analyze(symbol, forecast_result, regime_result, feature_result)
    backtest_result = backtest_agent.analyze(symbol)

    _emit_progress(progress_callback, 10, total_steps, "Synthesizing final recommendation")
    recommendation = supervisor_agent.make_recommendation(
        historical_result,
        indicator_result,
        news_result,
        pair_monitor_result,
        symbol,
        feature_result,
        regime_result,
        forecast_result,
        risk_result,
        backtest_result,
    )
    recommendation["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_report = supervisor_agent.format_final_report(recommendation)

    output_file = None
    if save_report:
        output_file = _build_report_file(
            symbol,
            final_report,
            historical_result,
            indicator_result,
            news_result,
            pair_monitor_result,
            feature_result,
            regime_result,
            forecast_result,
            risk_result,
            backtest_result,
        )

    persisted = False
    if storage:
        try:
            storage.save_recommendation(recommendation, symbol, report_text=final_report)
            storage.save_pair_signals(
                pair_monitor_result,
                stock_symbol=symbol,
                interval=pair_monitor_agent.interval,
            )
            storage.save_prediction(forecast_result, stock_symbol=symbol)
            persisted = True
        except Exception:
            persisted = False

    return {
        "status": "success",
        "stock_symbol": symbol,
        "timestamp": recommendation.get("timestamp"),
        "final_report": final_report,
        "output_file": output_file,
        "persisted": persisted,
        "results": {
            "historical": historical_result,
            "indicator": indicator_result,
            "news": news_result,
            "pair_ledger": pair_ledger,
            "pair_monitor": pair_monitor_result,
            "feature": feature_result,
            "regime": regime_result,
            "forecast": forecast_result,
            "risk": risk_result,
            "backtest": backtest_result,
            "recommendation": recommendation,
        },
    }
