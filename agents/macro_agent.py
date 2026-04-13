"""
Macro Analysis Agent
Fetches macroeconomic data from Alpha Vantage and produces
a comprehensive macro-environment assessment for US equities.
"""

from __future__ import annotations

import os
import time
import requests
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool


# ── Alpha Vantage helpers ───────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"

# Delay between API calls to respect the 5 calls/min free-tier limit
_CALL_DELAY_SECONDS = 12


def _av_macro_get(function: str, api_key: str, **extra) -> Dict[str, Any]:
    """Generic Alpha Vantage GET request for macro endpoints (no symbol needed)."""
    params = {"function": function, "apikey": api_key, **extra}
    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    return data


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for 'None' or missing."""
    if value is None or value == "None" or value == "" or value == ".":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _fmt(val: float | None, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    """Format a float for display, returning 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def _extract_recent(data: List[Dict[str, str]], n: int = 6) -> List[Dict[str, Any]]:
    """Extract the most recent N data points from an Alpha Vantage time series."""
    return data[:n] if data else []


def _trend_arrow(values: List[float | None]) -> str:
    """Return a simple trend indicator based on recent values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return "→ (insufficient data)"
    if clean[0] > clean[1]:
        return "↑ (rising)"
    elif clean[0] < clean[1]:
        return "↓ (falling)"
    return "→ (flat)"


def _format_series(
    series: List[Dict[str, str]],
    value_key: str = "value",
    date_key: str = "date",
    suffix: str = "",
) -> str:
    """Format a time series into a readable table string."""
    if not series:
        return "  No data available."
    lines = []
    for dp in series:
        date = dp.get(date_key, "N/A")
        val = _safe_float(dp.get(value_key))
        lines.append(f"  {date}: {_fmt(val, suffix=suffix)}")
    return "\n".join(lines)


# ── Tool: fetch macro data ─────────────────────────────────────────────

@tool
def fetch_macro_data(dummy: str = "run") -> str:
    """
    Fetches key US macroeconomic indicators from Alpha Vantage:
    - Federal Funds Rate
    - Treasury Yield (10-Year and 2-Year)
    - CPI (Consumer Price Index)
    - Unemployment Rate
    - Nonfarm Payroll
    - Real GDP
    - Retail Sales
    - WTI Crude Oil Price

    Args:
        dummy: Unused parameter (tools require at least one argument).

    Returns:
        Formatted string containing recent macro data and trend indicators.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        return "Error: ALPHAVANTAGE_API_KEY not set in environment variables."

    sections: List[str] = []
    errors: List[str] = []

    def _fetch_and_format(
        label: str,
        function: str,
        data_key: str,
        suffix: str = "",
        n: int = 6,
        **extra_params,
    ):
        """Fetch one macro endpoint and append formatted output."""
        try:
            raw = _av_macro_get(function, api_key, **extra_params)
            series = _extract_recent(raw.get(data_key, []), n)
            values = [_safe_float(dp.get("value")) for dp in series]
            trend = _trend_arrow(values)

            section = f"{label} (Trend: {trend}):\n"
            section += _format_series(series, suffix=suffix)
            sections.append(section)
            time.sleep(_CALL_DELAY_SECONDS)
        except Exception as e:
            errors.append(f"{label}: {str(e)}")
            time.sleep(_CALL_DELAY_SECONDS)

    try:
        # 1. Federal Funds Rate (monthly)
        _fetch_and_format(
            "FEDERAL FUNDS RATE (Monthly)",
            "FEDERAL_FUNDS_RATE",
            "data",
            suffix="%",
            interval="monthly",
        )

        # 2. Treasury Yield — 10-Year (monthly)
        _fetch_and_format(
            "10-YEAR TREASURY YIELD (Monthly)",
            "TREASURY_YIELD",
            "data",
            suffix="%",
            interval="monthly",
            maturity="10year",
        )

        # 3. Treasury Yield — 2-Year (monthly)
        _fetch_and_format(
            "2-YEAR TREASURY YIELD (Monthly)",
            "TREASURY_YIELD",
            "data",
            suffix="%",
            interval="monthly",
            maturity="2year",
        )

        # 4. CPI (monthly)
        _fetch_and_format(
            "CPI — CONSUMER PRICE INDEX (Monthly)",
            "CPI",
            "data",
            interval="monthly",
        )

        # 5. Unemployment Rate (monthly)
        _fetch_and_format(
            "UNEMPLOYMENT RATE (Monthly)",
            "UNEMPLOYMENT",
            "data",
            suffix="%",
        )

        # 6. Nonfarm Payroll (monthly)
        _fetch_and_format(
            "NONFARM PAYROLL (Monthly, thousands)",
            "NONFARM_PAYROLL",
            "data",
            suffix="K",
        )

        # 7. Real GDP (quarterly)
        _fetch_and_format(
            "REAL GDP (Quarterly, billions USD)",
            "REAL_GDP",
            "data",
            suffix="B",
            interval="quarterly",
        )

        # 8. Retail Sales (monthly, millions USD)
        _fetch_and_format(
            "RETAIL SALES (Monthly, millions USD)",
            "RETAIL_SALES",
            "data",
            suffix="M",
        )

        # 9. WTI Crude Oil (monthly)
        _fetch_and_format(
            "WTI CRUDE OIL PRICE (Monthly, USD/barrel)",
            "WTI",
            "data",
            prefix="$",
            suffix="",
            interval="monthly",
        )

        # ── Compute yield curve spread (10Y - 2Y) ──────────────────
        yield_spread_section = _compute_yield_spread(api_key)
        if yield_spread_section:
            sections.insert(3, yield_spread_section)  # Insert after the two yield sections

    except Exception as e:
        errors.append(f"General error: {str(e)}")

    # ── Build final report ──────────────────────────────────────────
    report = f"""
MACROECONOMIC ENVIRONMENT REPORT (US)
{'=' * 60}
Data Source: Alpha Vantage  |  Most recent data points shown

"""
    report += "\n\n".join(sections)

    if errors:
        report += f"\n\n{'─' * 40}\n⚠️ DATA FETCH ERRORS:\n"
        for err in errors:
            report += f"  - {err}\n"

    return report


def _compute_yield_spread(api_key: str) -> str | None:
    """Compute the 10Y-2Y yield spread from already-fetched data.

    We re-fetch the two yield series (they should be cached by the OS/network
    layer within the same minute) and compute the spread.
    """
    try:
        raw_10y = _av_macro_get("TREASURY_YIELD", api_key, interval="monthly", maturity="10year")
        time.sleep(_CALL_DELAY_SECONDS)
        raw_2y = _av_macro_get("TREASURY_YIELD", api_key, interval="monthly", maturity="2year")

        data_10y = {dp["date"]: _safe_float(dp["value"]) for dp in raw_10y.get("data", [])[:6]}
        data_2y = {dp["date"]: _safe_float(dp["value"]) for dp in raw_2y.get("data", [])[:6]}

        common_dates = sorted(set(data_10y.keys()) & set(data_2y.keys()), reverse=True)[:6]
        if not common_dates:
            return None

        lines = []
        spreads = []
        for date in common_dates:
            v10 = data_10y.get(date)
            v2 = data_2y.get(date)
            if v10 is not None and v2 is not None:
                spread = v10 - v2
                spreads.append(spread)
                status = "INVERTED ⚠️" if spread < 0 else "Normal"
                lines.append(f"  {date}: {spread:+.2f}% ({status})")
            else:
                lines.append(f"  {date}: N/A")

        trend = _trend_arrow(spreads)
        section = f"YIELD CURVE SPREAD: 10Y - 2Y (Trend: {trend}):\n"
        section += "\n".join(lines)

        # Add interpretation
        if spreads and spreads[0] is not None:
            if spreads[0] < 0:
                section += "\n  ⚠️ YIELD CURVE IS INVERTED — historically a recession warning signal."
            elif spreads[0] < 0.5:
                section += "\n  ⚡ Yield curve is very flat — potential slowdown signal."
            else:
                section += "\n  ✅ Yield curve is positively sloped — normal economic expansion signal."

        return section
    except Exception:
        return None


# ── Agent class ─────────────────────────────────────────────────────────

class MacroAnalysisAgent:
    """Agent responsible for macroeconomic environment analysis."""

    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tools = [fetch_macro_data]
        self._setup_agent()

    def _setup_agent(self) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior macroeconomist and market strategist.
Your role is to analyze the current US macroeconomic environment and assess its
implications for equity markets. You will:

1. Fetch the latest macroeconomic data (interest rates, inflation, employment,
   GDP, retail sales, oil prices, yield curve).
2. Analyze the TREND of each indicator (improving, deteriorating, stable).
3. Assess the overall macro environment for US equities:
   - FAVORABLE: supportive of equity prices (low/falling rates, strong growth,
     low unemployment, contained inflation).
   - NEUTRAL: mixed signals, no clear directional bias.
   - UNFAVORABLE: headwinds for equities (rising rates, high inflation,
     weakening growth, rising unemployment).
4. Identify the TOP 3 macro risks and TOP 3 macro tailwinds.
5. Assess which equity SECTORS are most/least favored by the current macro regime.
6. Provide a concise macro outlook summary.

Be data-driven. Reference specific numbers and trends.
Format your response in clear sections with headers."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=self.verbose,
        )

    def analyze(self, stock_symbol: str = "") -> Dict[str, Any]:
        """
        Run macroeconomic analysis.

        Args:
            stock_symbol: Optional stock symbol for sector-specific context.

        Returns:
            Dictionary with macro analysis results.
        """
        sector_hint = f" with particular attention to how it affects {stock_symbol}'s sector" if stock_symbol else ""
        query = (
            f"Fetch the latest US macroeconomic data and provide a comprehensive "
            f"macro environment assessment{sector_hint}. Include:\n"
            f"1. Interest rate environment (Fed Funds Rate, Treasury yields, yield curve)\n"
            f"2. Inflation assessment (CPI trends)\n"
            f"3. Labor market health (unemployment, nonfarm payroll)\n"
            f"4. Economic growth (GDP, retail sales)\n"
            f"5. Energy/commodity impact (oil prices)\n"
            f"6. Overall macro verdict: FAVORABLE / NEUTRAL / UNFAVORABLE for equities\n"
            f"7. Top 3 macro risks and top 3 macro tailwinds\n"
            f"8. Sector implications"
        )

        try:
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "")
            if not output or "Error" in output:
                return {
                    "agent": "macro_analysis",
                    "stock_symbol": stock_symbol,
                    "analysis": output or "No analysis generated",
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
                "analysis": (
                    f"Error during macro analysis: {error_msg}. "
                    "This may be due to Alpha Vantage API rate limits (5 calls/min on free tier). "
                    "The macro agent fetches ~9 endpoints, which requires careful rate limiting. "
                    "Please try again shortly."
                ),
                "status": "error",
            }
