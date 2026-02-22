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
6. Review risk plan and backtest snapshot.
7. Synthesize agreement/conflict across all evidence.
8. Produce clear, actionable recommendations.

Requirements:
- Be probabilistic, not certain.
- Explicitly call out conflicts between qualitative and quantitative signals.
- Provide BUY/SELL/HOLD with rationale.
- Provide confidence level (Low/Medium/High).
- Provide key risks and caveats.
- Mention whether sizing should be conservative, moderate, or aggressive.

Be objective and balanced.""",
                ),
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
        backtest_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Synthesize analyses and make final recommendation.
        """
        feature_analysis = feature_analysis or {}
        regime_analysis = regime_analysis or {}
        forecast_analysis = forecast_analysis or {}
        risk_analysis = risk_analysis or {}
        backtest_analysis = backtest_analysis or {}

        input_text = f"""
STOCK SYMBOL: {stock_symbol}

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

BACKTEST SNAPSHOT:
Summary: {backtest_analysis.get('summary', 'No backtest analysis available')}
Metrics: {json.dumps(backtest_analysis.get('metrics', {}), indent=2)}

Please provide:
1. A synthesis across all evidence.
2. Final recommendation (BUY/SELL/HOLD).
3. Confidence level (Low/Medium/High) and why.
4. Risk assessment and sizing stance (conservative/moderate/aggressive).
5. Key supporting factors and key contradicting factors.
6. Important caveats/warnings.
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
                "backtest_status": backtest_analysis.get("status", "unknown"),
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
                "backtest_status": backtest_analysis.get("status", "unknown"),
                "status": "error",
            }

    def format_final_report(self, recommendation: Dict[str, Any]) -> str:
        """Format the final recommendation into a readable report."""
        stock_symbol = recommendation.get("stock_symbol", "UNKNOWN")
        rec_text = recommendation.get("recommendation", "No recommendation available")

        report = f"""
{'='*80}
FINANCIAL ADVISORY REPORT
{'='*80}
Stock Symbol: {stock_symbol}
Generated: {recommendation.get('timestamp', 'N/A')}

RECOMMENDATION:
{'-'*80}
{rec_text}

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
Backtest: {recommendation.get('backtest_status', 'unknown')}
{'='*80}
"""
        return report
