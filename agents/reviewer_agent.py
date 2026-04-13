"""
Reviewer Agent
Acts as a risk-control officer that reviews the Supervisor's draft recommendation.
Identifies logical gaps, overlooked risks, and signal conflicts.
This is part of the Reflection Loop (Phase 3 collaboration enhancement).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class ReviewerAgent:
    """Reviews Supervisor's draft recommendation and provides structured critique."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_prompt()

    def _setup_prompt(self) -> None:
        """Setup the prompt template for the reviewer."""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a senior risk-control officer reviewing an investment recommendation.

Your job is NOT to agree or disagree with the recommendation. Your job is to find PROBLEMS:

1. **Overlooked Warnings**: Did the recommendation ignore any agent's critical warning?
   - Check if negative news sentiment was acknowledged.
   - Check if fundamental overvaluation was considered.
   - Check if macro headwinds were factored in.
   - Check if poor historical prediction accuracy was addressed.
   - Check if risk flags from the risk plan were mentioned.

2. **Logic Consistency**: Is the reasoning internally consistent?
   - If the recommendation is BUY but multiple agents signal bearish, flag this.
   - If confidence is High but signals are mixed, flag this.
   - If position sizing is aggressive but risk flags exist, flag this.
   - If the action contradicts the forecast probability, flag this.

3. **Missing Considerations**:
   - Are there upcoming earnings/events not mentioned?
   - Is the stop-loss appropriate for the current volatility regime?
   - Does the recommendation account for the model's historical bias?

4. **Confidence Calibration**:
   - Is the stated confidence level justified by the evidence?
   - If the model has < 50% historical accuracy, confidence should be LOW regardless.
   - If this is among the first predictions with no track record, confidence should be capped at Medium.

Output your review in this EXACT format:

ISSUES_FOUND: [number]

CRITICAL_ISSUES:
- [issue description, with specific reference to which agent's data was overlooked]

WARNINGS:
- [less severe concerns]

CONFIDENCE_ADJUSTMENT: [LOWER/KEEP/RAISE] — [reason]

SIZING_ADJUSTMENT: [MORE_CONSERVATIVE/KEEP/MORE_AGGRESSIVE] — [reason]

SUGGESTED_ADDITIONS:
- [specific points the final recommendation should address]

If no issues are found, output:
ISSUES_FOUND: 0
No critical issues or warnings detected.

Be concise and specific. Reference actual data from the agents, not generic advice.""",
                ),
                ("human", "{input}"),
            ]
        )

    def review(
        self,
        draft_recommendation: str,
        news_analysis: Dict[str, Any],
        fundamental_analysis: Dict[str, Any],
        macro_analysis: Dict[str, Any],
        forecast_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        memory_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Review the Supervisor's draft and return structured critique."""
        news_analysis = news_analysis or {}
        fundamental_analysis = fundamental_analysis or {}
        macro_analysis = macro_analysis or {}
        forecast_analysis = forecast_analysis or {}
        risk_analysis = risk_analysis or {}
        memory_analysis = memory_analysis or {}
        regime_analysis = regime_analysis or {}

        input_text = f"""
DRAFT RECOMMENDATION TO REVIEW:
{draft_recommendation}

--- RAW AGENT DATA FOR CROSS-CHECK ---

NEWS SENTIMENT:
{news_analysis.get('analysis', 'N/A')}

FUNDAMENTAL ANALYSIS:
{fundamental_analysis.get('analysis', 'N/A')}

MACROECONOMIC ANALYSIS:
{macro_analysis.get('analysis', 'N/A')}

FORECAST:
{json.dumps(forecast_analysis.get('forecast', {}), indent=2)}

RISK PLAN:
{json.dumps(risk_analysis.get('risk_plan', {}), indent=2)}

REGIME:
{json.dumps(regime_analysis.get('regime', {}), indent=2)}

HISTORICAL PREDICTION PERFORMANCE:
Summary: {memory_analysis.get('summary', 'N/A')}
Track Record Factor: {memory_analysis.get('track_record_factor', 'N/A')}

Please review the draft recommendation against the raw agent data above.
"""

        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"input": input_text})
            review_text = result.content if hasattr(result, "content") else str(result)

            return {
                "agent": "reviewer",
                "status": "success",
                "review": review_text,
            }
        except Exception as exc:
            return {
                "agent": "reviewer",
                "status": "error",
                "review": f"Review failed: {exc}",
            }
