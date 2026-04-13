"""
Supervisor Agent
Coordinates all agents and provides a final recommendation.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class SupervisorAgent:
    """Supervisor agent that synthesizes cross-agent evidence."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_prompt()

    def _setup_prompt(self) -> None:
        """Setup the prompt template for the supervisor."""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a senior financial advisor coordinating multiple analysis agents.
Your role is to:
1. Review historical price/volume analysis.
2. Review indicator analysis (RSI and momentum signals).
3. Review news sentiment analysis.
4. Review pair-monitor divergence signals.
5. Review quantitative outputs (features, regime, probability forecast).
6. Review risk plan.
7. Review fundamental analysis (financial statements, valuation, financial health).
8. Review macroeconomic environment (interest rates, inflation, employment, GDP, yield curve).
9. Review historical prediction performance (memory) to calibrate your confidence.
10. Synthesize agreement/conflict across all evidence.
11. Produce clear, actionable recommendations.

Requirements:
- Be probabilistic, not certain.
- Explicitly call out conflicts between qualitative and quantitative signals.
- Provide BUY/SELL/HOLD with rationale.
- Provide confidence level (Low/Medium/High).
- If historical prediction performance data is available, use it to calibrate your confidence:
  * If past directional accuracy < 50%, you MUST set confidence to Low.
  * If there is a systematic bias, explicitly mention it and adjust expectations.
  * If the model performs poorly in the current market regime, flag this as a key risk.
- Provide key risks and caveats.
- Mention whether sizing should be conservative, moderate, or aggressive.
- If any agent is marked as DEGRADED, explicitly acknowledge the reduced data quality.
  * Lower your overall confidence proportionally to the number and importance of degraded agents.
  * If critical agents (Feature, Forecast, Risk) are degraded, recommend conservative sizing.
  * Clearly state which analyses are based on incomplete or default data.

Be objective and balanced."""                ),
                ("human", "{input}"),
            ]
        )

    def make_recommendation(
        self,
        historical_analysis: Dict[str, Any],
        indicator_analysis: Dict[str, Any],
        news_analysis: Dict[str, Any],
        pair_monitor_analysis: Dict[str, Any],
        stock_symbol: str,
        feature_analysis: Dict[str, Any] | None = None,
        regime_analysis: Dict[str, Any] | None = None,
        forecast_analysis: Dict[str, Any] | None = None,
        risk_analysis: Dict[str, Any] | None = None,
        memory_analysis: Dict[str, Any] | None = None,
        fundamental_analysis: Dict[str, Any] | None = None,
        macro_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Synthesize analyses and make final recommendation.
        """
        feature_analysis = feature_analysis or {}
        regime_analysis = regime_analysis or {}
        forecast_analysis = forecast_analysis or {}
        risk_analysis = risk_analysis or {}
        memory_analysis = memory_analysis or {}
        fundamental_analysis = fundamental_analysis or {}
        macro_analysis = macro_analysis or {}

        # P3: Build degraded-agents warning section
        all_analyses = {
            "Historical": historical_analysis,
            "Indicator": indicator_analysis,
            "News Sentiment": news_analysis,
            "Pair Monitor": pair_monitor_analysis,
            "Feature Engineering": feature_analysis,
            "Regime": regime_analysis,
            "Forecast": forecast_analysis,
            "Risk Management": risk_analysis,
            "Memory": memory_analysis,
            "Fundamental": fundamental_analysis,
            "Macro": macro_analysis,
        }
        degraded_lines = []
        for name, analysis in all_analyses.items():
            status = analysis.get("status", "unknown")
            if status in ("degraded", "error", "skipped"):
                reason = analysis.get("degraded_reason", analysis.get("summary", "Unknown reason"))
                degraded_lines.append(f"- {name}: [{status.upper()}] {reason}")

        degraded_section = ""
        if degraded_lines:
            degraded_section = (
                "\n\n⚠️ DEGRADED / UNAVAILABLE AGENTS:\n"
                + "\n".join(degraded_lines)
                + "\n\nPlease factor in the reduced data quality when making your recommendation. "
                "Lower your overall confidence accordingly.\n"
            )

        input_text = f"""
STOCK SYMBOL: {stock_symbol}
{degraded_section}
HISTORICAL DATA ANALYSIS:
{historical_analysis.get('analysis', 'No analysis available')}

INDICATOR DATA ANALYSIS:
{indicator_analysis.get('analysis', 'No indicator analysis available')}

NEWS SENTIMENT ANALYSIS:
{news_analysis.get('analysis', 'No analysis available')}

PAIR MONITOR ANALYSIS:
Summary: {pair_monitor_analysis.get('summary', 'No pair monitoring available')}
Signals: {json.dumps(pair_monitor_analysis.get('signals', []), indent=2)}

FEATURE ENGINEERING:
Summary: {feature_analysis.get('summary', 'No feature analysis available')}
Features: {json.dumps(feature_analysis.get('features', {}), indent=2)}

REGIME ANALYSIS:
Summary: {regime_analysis.get('summary', 'No regime analysis available')}
Regime: {json.dumps(regime_analysis.get('regime', {}), indent=2)}

FORECAST ANALYSIS:
Summary: {forecast_analysis.get('summary', 'No forecast analysis available')}
Forecast: {json.dumps(forecast_analysis.get('forecast', {}), indent=2)}

RISK MANAGEMENT ANALYSIS:
Summary: {risk_analysis.get('summary', 'No risk analysis available')}
Risk Plan: {json.dumps(risk_analysis.get('risk_plan', {}), indent=2)}

FUNDAMENTAL ANALYSIS:
{fundamental_analysis.get('analysis', 'No fundamental analysis available')}

MACROECONOMIC ANALYSIS:
{macro_analysis.get('analysis', 'No macroeconomic analysis available')}

HISTORICAL PREDICTION PERFORMANCE (MEMORY):
Summary: {memory_analysis.get('summary', 'No historical prediction performance data available.')}
Stats: {json.dumps(memory_analysis.get('memory', {}), indent=2)}
Track Record Factor: {memory_analysis.get('track_record_factor', 'N/A')}

Please provide:
1. A synthesis across all evidence.
2. Final recommendation (BUY/SELL/HOLD).
3. Confidence level (Low/Medium/High) and why.
4. Risk assessment and sizing stance (conservative/moderate/aggressive).
5. Key supporting factors and key contradicting factors.
6. Important caveats/warnings.
7. If fundamental data is available, incorporate valuation, financial health, and growth assessment into your recommendation.
8. If macro data is available, incorporate the macroeconomic environment assessment (rates, inflation, growth, yield curve) into your recommendation.
9. If memory data is available, comment on the model's historical track record and how it affects your confidence.
10. If any agents are degraded/unavailable, explicitly state how this affects your analysis quality and confidence.
"""

        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"input": input_text})
            recommendation_text = result.content if hasattr(result, "content") else str(result)

            return {
                "agent": "supervisor",
                "stock_symbol": stock_symbol,
                "recommendation": recommendation_text,
                "historical_status": historical_analysis.get("status", "unknown"),
                "indicator_status": indicator_analysis.get("status", "unknown"),
                "news_status": news_analysis.get("status", "unknown"),
                "pair_monitor_status": pair_monitor_analysis.get("status", "unknown"),
                "feature_status": feature_analysis.get("status", "unknown"),
                "regime_status": regime_analysis.get("status", "unknown"),
                "forecast_status": forecast_analysis.get("status", "unknown"),
                "risk_status": risk_analysis.get("status", "unknown"),
                "memory_status": memory_analysis.get("status", "unknown"),
                "fundamental_status": fundamental_analysis.get("status", "unknown"),
                "macro_status": macro_analysis.get("status", "unknown"),
                "status": "success",
            }
        except Exception as exc:
            return {
                "agent": "supervisor",
                "stock_symbol": stock_symbol,
                "recommendation": f"Error generating recommendation: {exc}",
                "historical_status": historical_analysis.get("status", "unknown"),
                "indicator_status": indicator_analysis.get("status", "unknown"),
                "news_status": news_analysis.get("status", "unknown"),
                "pair_monitor_status": pair_monitor_analysis.get("status", "unknown"),
                "feature_status": feature_analysis.get("status", "unknown"),
                "regime_status": regime_analysis.get("status", "unknown"),
                "forecast_status": forecast_analysis.get("status", "unknown"),
                "risk_status": risk_analysis.get("status", "unknown"),
                "memory_status": memory_analysis.get("status", "unknown"),
                "fundamental_status": fundamental_analysis.get("status", "unknown"),
                "macro_status": macro_analysis.get("status", "unknown"),
                "status": "error",
            }

    def revise_recommendation(
        self,
        draft_recommendation: str,
        review_critique: str,
        stock_symbol: str,
    ) -> str:
        """Revise the draft recommendation based on Reviewer's critique."""
        revise_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a senior financial advisor. You previously wrote a draft recommendation.
A risk-control officer has reviewed your draft and found potential issues.

Your task:
1. Read the critique carefully.
2. For each CRITICAL_ISSUE: either fix your recommendation or explain why you disagree.
3. For each WARNING: acknowledge it and adjust if appropriate.
4. Apply the suggested CONFIDENCE_ADJUSTMENT and SIZING_ADJUSTMENT if justified.
5. Add any SUGGESTED_ADDITIONS that are valid.

Output your REVISED final recommendation in the same format as the original.
Mark any changes with [REVISED] so the reader can see what was updated.

Important: Do NOT simply append the critique. Rewrite the recommendation incorporating the feedback.
Keep the same structure (synthesis, recommendation, confidence, risk assessment, etc.).""",
                ),
                (
                    "human",
                    """ORIGINAL DRAFT:
{draft}

REVIEWER CRITIQUE:
{critique}

STOCK SYMBOL: {symbol}

Please provide your revised recommendation.""",
                ),
            ]
        )

        try:
            chain = revise_prompt | self.llm
            result = chain.invoke({
                "draft": draft_recommendation,
                "critique": review_critique,
                "symbol": stock_symbol,
            })
            return result.content if hasattr(result, "content") else str(result)
        except Exception:
            # If revision fails, return the original draft unchanged
            return draft_recommendation

    def format_final_report(self, recommendation: Dict[str, Any]) -> str:
        """Format the final recommendation into a readable report."""
        stock_symbol = recommendation.get("stock_symbol", "UNKNOWN")
        rec_text = recommendation.get("recommendation", "No recommendation available")

        # Build optional review section
        review_section = ""
        if recommendation.get("was_revised"):
            review_section = f"""
REVIEWER CRITIQUE (Reflection Loop):
{'-'*80}
{recommendation.get('review', 'N/A')}

Note: The recommendation above has been [REVISED] based on the reviewer's critique.
"""
        elif recommendation.get("review"):
            review_section = f"""
REVIEWER CRITIQUE (Reflection Loop):
{'-'*80}
No issues found. Original recommendation retained.
"""

        report = f"""
{'='*80}
FINANCIAL ADVISORY REPORT
{'='*80}
Stock Symbol: {stock_symbol}
Generated: {recommendation.get('timestamp', 'N/A')}
Reviewed: {'Yes (revised)' if recommendation.get('was_revised') else 'Yes (no changes needed)' if recommendation.get('review') else 'No'}

RECOMMENDATION:
{'-'*80}
{rec_text}
{review_section}
{'='*80}
Status: {recommendation.get('status', 'unknown')}
Historical Analysis: {recommendation.get('historical_status', 'unknown')}
Indicator Analysis: {recommendation.get('indicator_status', 'unknown')}
News Sentiment: {recommendation.get('news_status', 'unknown')}
Pair Monitor: {recommendation.get('pair_monitor_status', 'unknown')}
Feature Engineering: {recommendation.get('feature_status', 'unknown')}
Regime: {recommendation.get('regime_status', 'unknown')}
Forecast: {recommendation.get('forecast_status', 'unknown')}
Risk: {recommendation.get('risk_status', 'unknown')}
Memory: {recommendation.get('memory_status', 'unknown')}
Fundamental: {recommendation.get('fundamental_status', 'unknown')}
Macro: {recommendation.get('macro_status', 'unknown')}
{'='*80}
"""
        return report
