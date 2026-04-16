"""
Pipeline orchestrator shared by CLI and programmatic usage.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.historical_agent import HistoricalAnalysisAgent
from agents.indicator_agent import IndicatorAnalysisAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.regime_agent import RegimeAgent
from agents.forecast_agent import ForecastAgent
from utils.cross_sectional_service import CrossSectionalFeatureService
from agents.risk_agent import RiskAgent
from agents.ledger_agent import PairLedgerAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.pair_monitor_agent import PairMonitorAgent
from agents.supervisor_agent import SupervisorAgent
from agents.reviewer_agent import ReviewerAgent
from agents.memory_agent import MemoryAgent
from agents.fundamental_agent import FundamentalAnalysisAgent
from agents.macro_agent import MacroAnalysisAgent
from utils.macro_fundamental_provider import MacroFundamentalFeatureProvider
from utils.storage import Storage
from pipelines.track_outcomes import track_outcomes

ProgressCallback = Callable[[int, int, str], None]


def _emit_progress(
    callback: ProgressCallback | None,
    step: int,
    total: int,
    message: str,
) -> None:
    if callback:
        callback(step, total, message)


def _safe_run(
    agent_name: str,
    fn: Callable[..., Dict[str, Any]],
    *args: Any,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Safely execute an agent, returning a degraded result on failure.

    This ensures that a single agent exception never crashes the entire
    pipeline.  Downstream agents receive structurally valid (but empty)
    data so they can continue in their own degraded mode.
    """
    try:
        result = fn(*args, **kwargs)
        # If the agent itself returned an error status, promote to degraded
        if result.get("status") == "error":
            result["status"] = "degraded"
            result.setdefault("degraded_reason", result.get("summary", "Agent returned error status"))
        return result
    except Exception as exc:
        if verbose:
            print(f"[pipeline] {agent_name} failed with exception: {exc}")
        return {
            "agent": agent_name,
            "status": "degraded",
            "degraded_reason": f"Exception: {exc}",
            "summary": f"{agent_name} failed: {exc}. Downstream agents will use defaults.",
            # Provide empty but structurally valid data so downstream can continue
            "analysis": "",
            "features": {},
            "regime": {},
            "forecast": {},
            "risk_plan": {},
            "metrics": {},
            "signals": [],
            "memory": {},
            "track_record_factor": 1.0,
            "pairs": [],
            "macro_features": {},
            "fundamental_features": {},
        }


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
    memory_result: Dict[str, Any] | None = None,
    fundamental_result: Dict[str, Any] | None = None,
    macro_result: Dict[str, Any] | None = None,
) -> str:
    memory_result = memory_result or {}
    fundamental_result = fundamental_result or {}
    macro_result = macro_result or {}
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
        f.write("\nFUNDAMENTAL ANALYSIS:\n")
        f.write(fundamental_result.get("analysis", "") + "\n")
        f.write("\nMACROECONOMIC ANALYSIS:\n")
        f.write(macro_result.get("analysis", "") + "\n")
        f.write("\nMEMORY (HISTORICAL PREDICTION PERFORMANCE):\n")
        f.write(memory_result.get("summary", "No memory data available.") + "\n")
        f.write(json.dumps(memory_result.get("memory", {}), indent=2) + "\n")
    return output_file


def run_full_analysis(
    stock_symbol: str,
    *,
    verbose: bool = False,
    persist: bool = True,
    save_report: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Execute the full financial analysis pipeline with fault tolerance."""
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
model=os.getenv("OPENAI_MODEL", "gpt-4o"),
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
    memory_agent = MemoryAgent(verbose=verbose)
    fundamental_agent = FundamentalAnalysisAgent(llm, verbose=verbose)
    macro_agent = MacroAnalysisAgent(llm, verbose=verbose)
    macro_fund_provider = MacroFundamentalFeatureProvider(verbose=verbose)
    supervisor_agent = SupervisorAgent(llm)
    reviewer_llm = ChatOpenAI(
        model=os.getenv("REVIEWER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        temperature=0.2,
        api_key=api_key,
    )
    reviewer_agent = ReviewerAgent(reviewer_llm)
    _emit_progress(progress_callback, 1, total_steps, "Agents initialized, connecting storage")

    storage = None
    storage_enabled = os.getenv("STORAGE_ENABLED", "true").lower() == "true"
    if persist and storage_enabled:
        try:
            storage = Storage()
        except Exception:
            storage = None

    _emit_progress(progress_callback, 1, total_steps, "Analyzing pair ledger")

    # Step 0: Track matured predictions and recall memory
    memory_result: Dict[str, Any] = {
        "agent": "memory",
        "status": "skipped",
        "memory": {},
        "track_record_factor": 1.0,
        "summary": "Memory module skipped (storage unavailable).",
    }
    if storage:
        try:
            _emit_progress(progress_callback, 1, total_steps, "Tracking matured predictions")
            track_result = track_outcomes(storage=storage, verbose=verbose)
            if verbose and track_result.get("tracked", 0) > 0:
                print(f"[memory] Tracked {track_result['tracked']} matured prediction(s).")
        except Exception:
            pass  # Non-critical; continue even if tracking fails

        try:
            _emit_progress(progress_callback, 1, total_steps, "Recalling historical prediction performance")
            memory_result = memory_agent.recall(symbol, storage)
        except Exception:
            pass  # Non-critical; use defaults

    # ── P2: Layer 0 — parallel execution of independent agents ──────────
    _emit_progress(progress_callback, 2, total_steps, "Running independent agents in parallel (Layer 0)")

    layer0_futures: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=9, thread_name_prefix="agent") as pool:
        layer0_futures["pair_ledger"] = pool.submit(
            _safe_run, "pair_ledger", ledger_agent.analyze, verbose=verbose,
        )
        layer0_futures["historical"] = pool.submit(
            _safe_run, "historical", historical_agent.analyze, symbol, verbose=verbose,
        )
        layer0_futures["indicator"] = pool.submit(
            _safe_run, "indicator", indicator_agent.analyze, symbol, verbose=verbose,
        )
        layer0_futures["news"] = pool.submit(
            _safe_run, "news", news_agent.analyze, symbol, verbose=verbose,
        )
        layer0_futures["feature"] = pool.submit(
            _safe_run, "feature", feature_agent.analyze, symbol, verbose=verbose,
        )
        # Structured macro/fundamental features for downstream numeric agents
        # Also provides raw reports for LLM agents (fundamental + macro)
        layer0_futures["macro_fund_features"] = pool.submit(
            _safe_run, "macro_fund_features", macro_fund_provider.extract, symbol, verbose=verbose,
        )

    # Collect Layer 0 results
    pair_ledger = layer0_futures["pair_ledger"].result()
    historical_result = layer0_futures["historical"].result()
    indicator_result = layer0_futures["indicator"].result()
    news_result = layer0_futures["news"].result()
    feature_result = layer0_futures["feature"].result()
    macro_fund_features = layer0_futures["macro_fund_features"].result()

    # ── Layer 0b: LLM agents that depend on Provider data ──────────────
    # Fundamental and Macro agents now receive pre-fetched data from Provider
    # instead of calling Alpha Vantage directly. Run them in parallel.
    layer0b_futures: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-agent") as pool:
        layer0b_futures["fundamental"] = pool.submit(
            _safe_run, "fundamental", fundamental_agent.analyze, symbol,
            macro_fund_features, verbose=verbose,
        )
        layer0b_futures["macro"] = pool.submit(
            _safe_run, "macro", macro_agent.analyze, symbol,
            macro_fund_features, verbose=verbose,
        )

    fundamental_result = layer0b_futures["fundamental"].result()
    macro_result = layer0b_futures["macro"].result()

    if verbose:
        all_layer0 = {**layer0_futures, **layer0b_futures}
        degraded_agents = [
            name for name, fut in all_layer0.items()
            if fut.result().get("status") in ("degraded", "error", "skipped")
        ]
        if degraded_agents:
            print(f"[pipeline] Layer 0 degraded agents: {degraded_agents}")

    _emit_progress(progress_callback, 3, total_steps, "Layer 0 complete")

    # ── Layer 1: PairMonitor (depends on PairLedger) ────────────────────
    _emit_progress(progress_callback, 4, total_steps, "Running pair monitor analysis")
    if pair_ledger.get("status") == "success":
        pair_monitor_result = _safe_run(
            "pair_monitor",
            pair_monitor_agent.analyze,
            pair_ledger.get("pairs", []),
            focus_symbol=symbol,
            verbose=verbose,
        )
    else:
        pair_monitor_result = {
            "agent": "pair_monitor",
            "status": "degraded",
            "degraded_reason": "Pair ledger unavailable; monitoring skipped.",
            "summary": "Pair ledger unavailable; monitoring skipped.",
            "signals": [],
        }

    # ── Layer 1: Regime (depends on Feature + MacroFundFeatures) ─────────
    _emit_progress(progress_callback, 5, total_steps, "Classifying market regime")
    regime_result = _safe_run(
        "regime", regime_agent.analyze, symbol, feature_result,
        macro_fund_features, verbose=verbose,
    )

    # ── Layer 2: Forecast (depends on Feature + Regime + MacroFundFeatures) ──
    _emit_progress(progress_callback, 6, total_steps, "Generating probabilistic forecast")
    forecast_result = _safe_run(
        "forecast",
        forecast_agent.analyze,
        symbol,
        feature_result,
        regime_result,
        macro_fund_features,
        verbose=verbose,
    )

    # ── Layer 3: Risk (depends on Forecast + Regime + Feature + Memory + MacroFundFeatures) ─
    _emit_progress(progress_callback, 7, total_steps, "Generating risk plan")
    risk_result = _safe_run(
        "risk",
        risk_agent.analyze,
        symbol,
        forecast_result,
        regime_result,
        feature_result,
        memory_result,
        macro_fund_features,
        verbose=verbose,
    )

    # ── Layer 4: Supervisor (synthesizes everything) ────────────────────
    _emit_progress(progress_callback, 8, total_steps, "Synthesizing final recommendation")
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
        memory_result,
        fundamental_result,
        macro_result,
    )

    # ── Layer 4.5: Reflection Loop (Reviewer → Supervisor Revision) ─────
    draft_text = recommendation.get("recommendation", "")
    _emit_progress(progress_callback, 9, total_steps, "Reviewing recommendation (reflection loop)")
    review_result = _safe_run(
        "reviewer",
        reviewer_agent.review,
        draft_text,
        news_result,
        fundamental_result,
        macro_result,
        forecast_result,
        risk_result,
        memory_result,
        regime_result,
        verbose=verbose,
    )

    review_text = review_result.get("review", "")
    if review_result.get("status") == "success" and "ISSUES_FOUND: 0" not in review_text:
        _emit_progress(progress_callback, 10, total_steps, "Revising recommendation based on review")
        revised_text = supervisor_agent.revise_recommendation(
            draft_text, review_text, symbol
        )
        recommendation["recommendation"] = revised_text
        recommendation["review"] = review_text
        recommendation["was_revised"] = True
        if verbose:
            print("[pipeline] Recommendation revised after reviewer critique.")
    else:
        recommendation["review"] = review_text
        recommendation["was_revised"] = False
        if verbose:
            print("[pipeline] Reviewer found no issues; using original recommendation.")

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
            memory_result,
            fundamental_result,
            macro_result,
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
            storage.save_prediction(forecast_result, stock_symbol=symbol, regime_result=regime_result)
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
            "memory": memory_result,
            "fundamental": fundamental_result,
            "macro": macro_result,
            "macro_fund_features": macro_fund_features,
            "recommendation": recommendation,
        },
    }
