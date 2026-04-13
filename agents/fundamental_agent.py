"""
Fundamental Analysis Agent
Fetches financial statements and valuation data from Alpha Vantage
and produces a comprehensive fundamental analysis.
"""

from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool


# ── Alpha Vantage helpers ───────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"


def _av_get(function: str, symbol: str, api_key: str, **extra) -> Dict[str, Any]:
    """Generic Alpha Vantage GET request with error handling."""
    params = {"function": function, "symbol": symbol, "apikey": api_key, **extra}
    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Alpha Vantage returns an error message in a "Note" or "Information" key
    # when rate-limited or when the API key is invalid.
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    return data


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for 'None' or missing."""
    if value is None or value == "None" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _pct(a: float | None, b: float | None) -> float | None:
    """Return a/b as a percentage, or None if not computable."""
    if a is None or b is None or b == 0:
        return None
    return (a / b) * 100


def _fmt(val: float | None, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    """Format a float for display, returning 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def _fmt_large(val: float | None) -> str:
    """Format a large number (e.g. revenue) in human-readable form."""
    if val is None:
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    return f"{sign}${abs_val:,.0f}"


# ── Tool: fetch fundamental data ───────────────────────────────────────

@tool
def fetch_fundamental_data(symbol: str) -> str:
    """
    Fetches fundamental financial data for a stock from Alpha Vantage,
    including company overview, income statement, balance sheet, and cash flow.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        Formatted string containing key fundamental metrics and financial data.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        return "Error: ALPHAVANTAGE_API_KEY not set in environment variables."

    try:
        # Fetch all four endpoints
        overview = _av_get("OVERVIEW", symbol, api_key)
        income_raw = _av_get("INCOME_STATEMENT", symbol, api_key)
        balance_raw = _av_get("BALANCE_SHEET", symbol, api_key)
        cashflow_raw = _av_get("CASH_FLOW", symbol, api_key)

        # ── Company Overview ────────────────────────────────────────
        company_name = overview.get("Name", symbol)
        sector = overview.get("Sector", "N/A")
        industry = overview.get("Industry", "N/A")
        market_cap = _safe_float(overview.get("MarketCapitalization"))
        pe_ratio = _safe_float(overview.get("PERatio"))
        peg_ratio = _safe_float(overview.get("PEGRatio"))
        pb_ratio = _safe_float(overview.get("PriceToBookRatio"))
        ps_ratio = _safe_float(overview.get("PriceToSalesRatioTTM"))
        ev_ebitda = _safe_float(overview.get("EVToEBITDA"))
        dividend_yield = _safe_float(overview.get("DividendYield"))
        eps = _safe_float(overview.get("EPS"))
        roe = _safe_float(overview.get("ReturnOnEquityTTM"))
        roa = _safe_float(overview.get("ReturnOnAssetsTTM"))
        profit_margin = _safe_float(overview.get("ProfitMargin"))
        operating_margin = _safe_float(overview.get("OperatingMarginTTM"))
        revenue_ttm = _safe_float(overview.get("RevenueTTM"))
        gross_profit_ttm = _safe_float(overview.get("GrossProfitTTM"))
        beta = _safe_float(overview.get("Beta"))
        week52_high = _safe_float(overview.get("52WeekHigh"))
        week52_low = _safe_float(overview.get("52WeekLow"))
        analyst_target = _safe_float(overview.get("AnalystTargetPrice"))
        rev_growth_yoy = _safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
        earnings_growth_yoy = _safe_float(overview.get("QuarterlyEarningsGrowthYOY"))

        # ── Income Statement (most recent annual + quarterly) ───────
        annual_income = income_raw.get("annualReports", [])
        quarterly_income = income_raw.get("quarterlyReports", [])

        latest_annual_inc = annual_income[0] if annual_income else {}
        prev_annual_inc = annual_income[1] if len(annual_income) > 1 else {}
        latest_q_inc = quarterly_income[0] if quarterly_income else {}

        annual_revenue = _safe_float(latest_annual_inc.get("totalRevenue"))
        annual_net_income = _safe_float(latest_annual_inc.get("netIncome"))
        annual_gross_profit = _safe_float(latest_annual_inc.get("grossProfit"))
        annual_operating_income = _safe_float(latest_annual_inc.get("operatingIncome"))
        annual_ebitda = _safe_float(latest_annual_inc.get("ebitda"))

        prev_annual_revenue = _safe_float(prev_annual_inc.get("totalRevenue"))
        prev_annual_net_income = _safe_float(prev_annual_inc.get("netIncome"))

        # Compute YoY growth
        revenue_growth = _pct(
            (annual_revenue - prev_annual_revenue) if annual_revenue and prev_annual_revenue else None,
            prev_annual_revenue,
        )
        net_income_growth = _pct(
            (annual_net_income - prev_annual_net_income) if annual_net_income and prev_annual_net_income else None,
            abs(prev_annual_net_income) if prev_annual_net_income else None,
        )

        # Margins
        gross_margin = _pct(annual_gross_profit, annual_revenue)
        operating_margin_calc = _pct(annual_operating_income, annual_revenue)
        net_margin = _pct(annual_net_income, annual_revenue)

        # ── Balance Sheet (most recent annual) ──────────────────────
        annual_balance = balance_raw.get("annualReports", [])
        latest_balance = annual_balance[0] if annual_balance else {}

        total_assets = _safe_float(latest_balance.get("totalAssets"))
        total_liabilities = _safe_float(latest_balance.get("totalLiabilities"))
        total_equity = _safe_float(latest_balance.get("totalShareholderEquity"))
        current_assets = _safe_float(latest_balance.get("totalCurrentAssets"))
        current_liabilities = _safe_float(latest_balance.get("totalCurrentLiabilities"))
        long_term_debt = _safe_float(latest_balance.get("longTermDebt"))
        short_term_debt = _safe_float(latest_balance.get("shortTermDebt"))
        cash_and_equiv = _safe_float(latest_balance.get("cashAndCashEquivalentsAtCarryingValue"))
        if cash_and_equiv is None:
            cash_and_equiv = _safe_float(latest_balance.get("cashAndShortTermInvestments"))

        current_ratio = (current_assets / current_liabilities) if current_assets and current_liabilities and current_liabilities != 0 else None
        debt_to_equity = (total_liabilities / total_equity) if total_liabilities and total_equity and total_equity != 0 else None
        total_debt = ((long_term_debt or 0) + (short_term_debt or 0)) or None

        # ── Cash Flow (most recent annual) ──────────────────────────
        annual_cf = cashflow_raw.get("annualReports", [])
        latest_cf = annual_cf[0] if annual_cf else {}

        operating_cf = _safe_float(latest_cf.get("operatingCashflow"))
        capex = _safe_float(latest_cf.get("capitalExpenditures"))
        free_cash_flow = (operating_cf - capex) if operating_cf is not None and capex is not None else None
        dividends_paid = _safe_float(latest_cf.get("dividendPayout"))

        # ── Build formatted output ──────────────────────────────────
        report = f"""
FUNDAMENTAL ANALYSIS FOR {symbol} ({company_name})
{'=' * 60}
Sector: {sector}  |  Industry: {industry}
Fiscal Year End: {latest_annual_inc.get('fiscalDateEnding', 'N/A')}

VALUATION METRICS:
- Market Cap: {_fmt_large(market_cap)}
- P/E Ratio (TTM): {_fmt(pe_ratio)}
- PEG Ratio: {_fmt(peg_ratio)}
- P/B Ratio: {_fmt(pb_ratio)}
- P/S Ratio (TTM): {_fmt(ps_ratio)}
- EV/EBITDA: {_fmt(ev_ebitda)}
- EPS (TTM): {_fmt(eps, prefix='$')}
- Dividend Yield: {_fmt(dividend_yield if dividend_yield and dividend_yield < 1 else (dividend_yield * 100 if dividend_yield and dividend_yield >= 1 else dividend_yield), suffix='%') if dividend_yield else 'N/A'}
- Beta: {_fmt(beta)}
- 52-Week Range: {_fmt(week52_low, prefix='$')} – {_fmt(week52_high, prefix='$')}
- Analyst Target Price: {_fmt(analyst_target, prefix='$')}

PROFITABILITY:
- Gross Margin: {_fmt(gross_margin, suffix='%')}
- Operating Margin: {_fmt(operating_margin_calc, suffix='%')}
- Net Margin: {_fmt(net_margin, suffix='%')}
- ROE (TTM): {_fmt(roe if roe and abs(roe) < 1 else None, suffix='%') if roe is None or abs(roe) < 1 else _fmt(roe * 100, suffix='%')}
- ROA (TTM): {_fmt(roa if roa and abs(roa) < 1 else None, suffix='%') if roa is None or abs(roa) < 1 else _fmt(roa * 100, suffix='%')}

INCOME STATEMENT (Annual: {latest_annual_inc.get('fiscalDateEnding', 'N/A')}):
- Revenue: {_fmt_large(annual_revenue)}
- Gross Profit: {_fmt_large(annual_gross_profit)}
- Operating Income: {_fmt_large(annual_operating_income)}
- EBITDA: {_fmt_large(annual_ebitda)}
- Net Income: {_fmt_large(annual_net_income)}

GROWTH:
- Revenue YoY Growth: {_fmt(revenue_growth, suffix='%')}
- Net Income YoY Growth: {_fmt(net_income_growth, suffix='%')}
- Quarterly Revenue Growth YoY: {_fmt(rev_growth_yoy if rev_growth_yoy and abs(rev_growth_yoy) < 1 else None, suffix='%') if rev_growth_yoy is None or abs(rev_growth_yoy) < 1 else _fmt(rev_growth_yoy * 100, suffix='%')}
- Quarterly Earnings Growth YoY: {_fmt(earnings_growth_yoy if earnings_growth_yoy and abs(earnings_growth_yoy) < 1 else None, suffix='%') if earnings_growth_yoy is None or abs(earnings_growth_yoy) < 1 else _fmt(earnings_growth_yoy * 100, suffix='%')}

BALANCE SHEET (Annual: {latest_balance.get('fiscalDateEnding', 'N/A')}):
- Total Assets: {_fmt_large(total_assets)}
- Total Liabilities: {_fmt_large(total_liabilities)}
- Shareholder Equity: {_fmt_large(total_equity)}
- Cash & Equivalents: {_fmt_large(cash_and_equiv)}
- Total Debt: {_fmt_large(total_debt)}
- Current Ratio: {_fmt(current_ratio)}
- Debt-to-Equity: {_fmt(debt_to_equity)}

CASH FLOW (Annual: {latest_cf.get('fiscalDateEnding', 'N/A')}):
- Operating Cash Flow: {_fmt_large(operating_cf)}
- Capital Expenditures: {_fmt_large(capex)}
- Free Cash Flow: {_fmt_large(free_cash_flow)}
- Dividends Paid: {_fmt_large(dividends_paid)}

LATEST QUARTER ({latest_q_inc.get('fiscalDateEnding', 'N/A')}):
- Quarterly Revenue: {_fmt_large(_safe_float(latest_q_inc.get('totalRevenue')))}
- Quarterly Net Income: {_fmt_large(_safe_float(latest_q_inc.get('netIncome')))}
"""
        return report

    except Exception as e:
        return f"Error fetching fundamental data for {symbol}: {str(e)}"


# ── Agent class ─────────────────────────────────────────────────────────

class FundamentalAnalysisAgent:
    """Agent responsible for fundamental / financial-statement analysis."""

    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tools = [fetch_fundamental_data]
        self._setup_agent()

    def _setup_agent(self) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior equity research analyst specializing in fundamental analysis.
Your role is to:
1. Fetch and review financial statements (income statement, balance sheet, cash flow).
2. Evaluate valuation metrics (P/E, P/B, P/S, EV/EBITDA, PEG).
3. Assess profitability (margins, ROE, ROA).
4. Analyze financial health (current ratio, debt-to-equity, cash position).
5. Evaluate growth trajectory (revenue growth, earnings growth).
6. Compute a qualitative valuation verdict: UNDERVALUED / FAIRLY VALUED / OVERVALUED.
7. Assign a financial health score: STRONG / MODERATE / WEAK.
8. Assign a growth score: HIGH GROWTH / MODERATE GROWTH / LOW GROWTH / DECLINING.

Be thorough and data-driven. Highlight both strengths and weaknesses.
Compare metrics to typical benchmarks where appropriate.
Format your response in a clear, structured manner with sections."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=self.verbose,
        )

    def analyze(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Run fundamental analysis for a given stock symbol.

        Args:
            stock_symbol: Stock ticker symbol to analyze.

        Returns:
            Dictionary with analysis results including valuation verdict,
            financial health score, growth score, and full analysis text.
        """
        query = (
            f"Fetch the fundamental financial data for {stock_symbol} and provide "
            f"a comprehensive fundamental analysis. Include:\n"
            f"1. Valuation assessment (cheap / fair / expensive)\n"
            f"2. Financial health evaluation\n"
            f"3. Growth trajectory analysis\n"
            f"4. Key strengths and risks\n"
            f"5. Overall fundamental outlook"
        )

        try:
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "")
            if not output or "Error" in output:
                return {
                    "agent": "fundamental_analysis",
                    "stock_symbol": stock_symbol,
                    "analysis": output or "No analysis generated",
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
                "analysis": (
                    f"Error during fundamental analysis: {error_msg}. "
                    "This may be due to Alpha Vantage API rate limits (5 calls/min on free tier). "
                    "Please try again shortly."
                ),
                "status": "error",
            }
