"""
Macro Analysis Agent
Receives pre-fetched macroeconomic data from MacroFundamentalFeatureProvider
and produces a comprehensive macro-environment assessment for US equities.

NOTE: This agent does NOT call Alpha Vantage directly. All data is fetched
by MacroFundamentalFeatureProvider and passed in via analyze().
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


_SYSTEM_PROMPT = """You are a senior macroeconomist and market strategist.
Your role is to analyze the current US macroeconomic environment and assess its
implications for equity markets. You will be provided with the latest macro data.

Based on the data provided, you must:

1. Analyze the TREND of each indicator (improving, deteriorating, stable).
2. Assess the overall macro environment for US equities:
   - FAVORABLE: supportive of equity prices (low/falling rates, strong growth,
     low unemployment, contained inflation).
   - NEUTRAL: mixed signals, no clear directional bias.
   - UNFAVORABLE: headwinds for equities (rising rates, high inflation,
     weakening growth, rising unemployment).
3. Identify the TOP 3 macro risks and TOP 3 macro tailwinds.
4. Assess which equity SECTORS are most/least favored by the current macro regime.
5. Provide a concise macro outlook summary.

Be data-driven. Reference specific numbers and trends.
Format your response in clear sections with headers."""


class MacroAnalysisAgent:
    """Agent responsible for macroeconomic environment analysis.

    This agent receives pre-fetched data from MacroFundamentalFeatureProvider
    and uses an LLM to produce qualitative macro analysis. It does NOT call
    any external APIs directly.
    """

    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def analyze(
        self,
        stock_symbol: str = "",
        macro_fund_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run macroeconomic analysis using pre-fetched data.

        Args:
            stock_symbol: Optional stock symbol for sector-specific context.
            macro_fund_data: Result dict from MacroFundamentalFeatureProvider.extract().
                             Must contain 'raw_macro_report' key with formatted text.

        Returns:
            Dictionary with macro analysis results.
        """
        macro_fund_data = macro_fund_data or {}
        raw_report = macro_fund_data.get("raw_macro_report", "")

        if not raw_report:
            return {
                "agent": "macro_analysis",
                "stock_symbol": stock_symbol,
                "analysis": "No macro data available. MacroFundamentalFeatureProvider may have failed.",
                "status": "degraded",
            }

        sector_hint = (
            f" with particular attention to how it affects {stock_symbol}'s sector"
            if stock_symbol else ""
        )
        query = (
            f"Below is the latest US macroeconomic data. Provide a comprehensive "
            f"macro environment assessment{sector_hint}. Include:\n"
            f"1. Interest rate environment (Fed Funds Rate, Treasury yields, yield curve)\n"
            f"2. Inflation assessment (CPI trends)\n"
            f"3. Labor market health (unemployment, nonfarm payroll)\n"
            f"4. Economic growth (GDP, retail sales)\n"
            f"5. Energy/commodity impact (oil prices)\n"
            f"6. Overall macro verdict: FAVORABLE / NEUTRAL / UNFAVORABLE for equities\n"
            f"7. Top 3 macro risks and top 3 macro tailwinds\n"
            f"8. Sector implications\n\n"
            f"--- MACRO DATA ---\n{raw_report}"
        )

        try:
            response = self.chain.invoke({"input": query})
            output = response.content if hasattr(response, "content") else str(response)

            if not output:
                return {
                    "agent": "macro_analysis",
                    "stock_symbol": stock_symbol,
                    "analysis": "No analysis generated",
                    "status": "error",
                }
            return {
                "agent": "macro_analysis",
                "stock_symbol": stock_symbol,
                "analysis": output,
                "status": "success",
            }
        except Exception as e:
            error_msg = str(e)
            return {
                "agent": "macro_analysis",
                "stock_symbol": stock_symbol,
                "analysis": f"Error during macro analysis: {error_msg}",
                "status": "error",
            }
