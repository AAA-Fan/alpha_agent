"""
Fundamental Analysis Agent
Receives pre-fetched financial data from MacroFundamentalFeatureProvider
and produces a comprehensive fundamental analysis.

NOTE: This agent does NOT call Alpha Vantage directly. All data is fetched
by MacroFundamentalFeatureProvider and passed in via analyze().
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


_SYSTEM_PROMPT = """You are a senior equity research analyst specializing in fundamental analysis.
You will be provided with pre-fetched financial data including income statement,
balance sheet, cash flow, and valuation metrics.

Based on the data provided, you must:
1. Evaluate valuation metrics (P/E, P/B, P/S, EV/EBITDA, PEG).
2. Assess profitability (margins, ROE, ROA).
3. Analyze financial health (current ratio, debt-to-equity, cash position).
4. Evaluate growth trajectory (revenue growth, earnings growth).
5. Compute a qualitative valuation verdict: UNDERVALUED / FAIRLY VALUED / OVERVALUED.
6. Assign a financial health score: STRONG / MODERATE / WEAK.
7. Assign a growth score: HIGH GROWTH / MODERATE GROWTH / LOW GROWTH / DECLINING.

Be thorough and data-driven. Highlight both strengths and weaknesses.
Compare metrics to typical benchmarks where appropriate.
Format your response in a clear, structured manner with sections."""


class FundamentalAnalysisAgent:
    """Agent responsible for fundamental / financial-statement analysis.

    This agent receives pre-fetched data from MacroFundamentalFeatureProvider
    and uses an LLM to produce qualitative fundamental analysis. It does NOT
    call any external APIs directly.
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
        stock_symbol: str,
        macro_fund_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Run fundamental analysis using pre-fetched data.

        Args:
            stock_symbol: Stock ticker symbol to analyze.
            macro_fund_data: Result dict from MacroFundamentalFeatureProvider.extract().
                             Must contain 'raw_fundamental_report' key with formatted text.

        Returns:
            Dictionary with analysis results including valuation verdict,
            financial health score, growth score, and full analysis text.
        """
        macro_fund_data = macro_fund_data or {}
        raw_report = macro_fund_data.get("raw_fundamental_report", "")

        if not raw_report:
            return {
                "agent": "fundamental_analysis",
                "stock_symbol": stock_symbol,
                "analysis": "No fundamental data available. MacroFundamentalFeatureProvider may have failed.",
                "status": "degraded",
            }

        query = (
            f"Below is the fundamental financial data for {stock_symbol}. "
            f"Provide a comprehensive fundamental analysis. Include:\n"
            f"1. Valuation assessment (cheap / fair / expensive)\n"
            f"2. Financial health evaluation\n"
            f"3. Growth trajectory analysis\n"
            f"4. Key strengths and risks\n"
            f"5. Overall fundamental outlook\n\n"
            f"--- FUNDAMENTAL DATA ---\n{raw_report}"
        )

        try:
            response = self.chain.invoke({"input": query})
            output = response.content if hasattr(response, "content") else str(response)

            if not output:
                return {
                    "agent": "fundamental_analysis",
                    "stock_symbol": stock_symbol,
                    "analysis": "No analysis generated",
                    "status": "error",
                }
            return {
                "agent": "fundamental_analysis",
                "stock_symbol": stock_symbol,
                "analysis": output,
                "status": "success",
            }
        except Exception as e:
            error_msg = str(e)
            return {
                "agent": "fundamental_analysis",
                "stock_symbol": stock_symbol,
                "analysis": f"Error during fundamental analysis: {error_msg}",
                "status": "error",
            }
