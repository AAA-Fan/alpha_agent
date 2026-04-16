"""
Macro & Fundamental Feature Provider
Extracts structured numeric features from Alpha Vantage for downstream agents
(RegimeAgent, ForecastAgent, RiskAgent).

This is NOT an Agent (no LLM, no decision-making). It is a lightweight data
provider that fetches key macro/fundamental numbers once and shares them across
all consumers in the pipeline.

Output contract:
    {
        "status": "success" | "degraded",
        "macro_features": { ... },          # Always present (may be empty)
        "fundamental_features": { ... },    # Always present (may be empty)
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

# ── Cache configuration ─────────────────────────────────────────────────

_CACHE_DIR = Path("data/cross_section_cache")
_MACRO_CACHE_DIR = _CACHE_DIR / "macro_hist_cache"
_FUND_CACHE_DIR = _CACHE_DIR / "fundamental_cache"
_CACHE_TTL_DAYS = int(os.getenv("MF_CACHE_TTL_DAYS", "7"))

# ── Alpha Vantage helpers ───────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"


def _av_get_cached(
    function: str,
    api_key: str,
    cache_key: str,
    cache_dir: Path,
    timeout: int = 30,
    **params,
) -> Dict[str, Any]:
    """Alpha Vantage GET with file-based caching.

    If a valid cache file exists (< _CACHE_TTL_DAYS old), return cached data.
    Otherwise fetch from API, save to cache, and return.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"

    # Check cache freshness
    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < _CACHE_TTL_DAYS:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # Cache corrupted, re-fetch

    # Fetch from API
    _throttle()
    data = _av_get(function, api_key, timeout=timeout, **params)

    # Save to cache
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except OSError:
        pass  # Non-fatal: cache write failure

    return data


def _get_api_key() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    return key


def _av_get(function: str, api_key: str, timeout: int = 30, **params) -> Dict[str, Any]:
    """Generic Alpha Vantage GET with error handling."""
    params.update({"function": function, "apikey": api_key})
    resp = requests.get(_AV_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    for err_key in ("Note", "Information", "Error Message"):
        if err_key in data:
            raise RuntimeError(f"Alpha Vantage {err_key}: {data[err_key]}")
    return data


def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, returning None for missing / unparseable values."""
    if value is None or value in ("None", "", "."):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _series_to_df(series: list, value_key: str = "value", date_key: str = "date") -> pd.DataFrame:
    """Convert Alpha Vantage time series list to a DatetimeIndex DataFrame."""
    if not series:
        return pd.DataFrame()
    rows = []
    for item in series:
        dt = pd.to_datetime(item.get(date_key), errors="coerce")
        val = _safe_float(item.get(value_key))
        if pd.notna(dt):
            rows.append({"date": dt, "value": val})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


# ── Rate limiter ────────────────────────────────────────────────────────

_last_call_ts: float = 0.0


def _throttle() -> None:
    """Best-effort rate limiting for Alpha Vantage (default 5 calls/min)."""
    global _last_call_ts
    limit = int(os.getenv("ALPHAVANTAGE_RATE_LIMIT_PER_MIN", "75"))
    if limit <= 0:
        return
    min_interval = 60.0 / float(limit)
    now = time.time()
    elapsed = now - _last_call_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_ts = time.time()


# ═══════════════════════════════════════════════════════════════════════════
# Macro Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_macro_features(api_key: str, verbose: bool = False) -> Dict[str, Optional[float]]:
    """Fetch key macro indicators and return structured numeric features.

    Features extracted:
        - fed_funds_rate:       Current Federal Funds Rate (%)
        - rate_change_3m:       3-month change in Fed Funds Rate (pp)
        - treasury_yield_10y:   10-Year Treasury Yield (%)
        - treasury_yield_2y:    2-Year Treasury Yield (%)
        - yield_spread_10y2y:   10Y - 2Y spread (pp, negative = inverted)
        - cpi_yoy:              CPI year-over-year change (%)
        - unemployment_rate:    Unemployment Rate (%)
        - vix_level:            VIX current level (from CBOE VIX proxy via AV)
        - vix_percentile_1y:    VIX percentile over last 12 months (0-1)
        - spy_momentum_20d:     SPY 20-day momentum (return)
    """
    features: Dict[str, Optional[float]] = {}

    # ── 1. Federal Funds Rate ───────────────────────────────────────
    try:
        _throttle()
        data = _av_get("FEDERAL_FUNDS_RATE", api_key, interval="monthly")
        series = data.get("data", [])
        if len(series) >= 4:
            current = _safe_float(series[0].get("value"))
            three_months_ago = _safe_float(series[3].get("value"))
            features["fed_funds_rate"] = current
            if current is not None and three_months_ago is not None:
                features["rate_change_3m"] = round(current - three_months_ago, 4)
            else:
                features["rate_change_3m"] = None
        else:
            features["fed_funds_rate"] = _safe_float(series[0].get("value")) if series else None
            features["rate_change_3m"] = None
        if verbose:
            print(f"  [macro] Fed Funds Rate: {features.get('fed_funds_rate')}")
    except Exception as exc:
        if verbose:
            print(f"  [macro] Fed Funds Rate failed: {exc}")
        features["fed_funds_rate"] = None
        features["rate_change_3m"] = None

    # ── 2. Treasury Yields (10Y and 2Y) ────────────────────────────
    for maturity, key in [("10year", "treasury_yield_10y"), ("2year", "treasury_yield_2y")]:
        try:
            _throttle()
            data = _av_get("TREASURY_YIELD", api_key, interval="monthly", maturity=maturity)
            series = data.get("data", [])
            features[key] = _safe_float(series[0].get("value")) if series else None
            if verbose:
                print(f"  [macro] {key}: {features.get(key)}")
        except Exception as exc:
            if verbose:
                print(f"  [macro] {key} failed: {exc}")
            features[key] = None

    # Yield spread
    y10 = features.get("treasury_yield_10y")
    y2 = features.get("treasury_yield_2y")
    if y10 is not None and y2 is not None:
        features["yield_spread_10y2y"] = round(y10 - y2, 4)
    else:
        features["yield_spread_10y2y"] = None

    # ── 3. CPI (year-over-year) ─────────────────────────────────────
    try:
        _throttle()
        data = _av_get("CPI", api_key, interval="monthly")
        series = data.get("data", [])
        if len(series) >= 13:
            current_cpi = _safe_float(series[0].get("value"))
            year_ago_cpi = _safe_float(series[12].get("value"))
            if current_cpi is not None and year_ago_cpi is not None and year_ago_cpi > 0:
                features["cpi_yoy"] = round((current_cpi / year_ago_cpi - 1) * 100, 2)
            else:
                features["cpi_yoy"] = None
        else:
            features["cpi_yoy"] = None
        if verbose:
            print(f"  [macro] CPI YoY: {features.get('cpi_yoy')}")
    except Exception as exc:
        if verbose:
            print(f"  [macro] CPI failed: {exc}")
        features["cpi_yoy"] = None

    # ── 4. Unemployment Rate ────────────────────────────────────────
    try:
        _throttle()
        data = _av_get("UNEMPLOYMENT", api_key)
        series = data.get("data", [])
        features["unemployment_rate"] = _safe_float(series[0].get("value")) if series else None
        if verbose:
            print(f"  [macro] Unemployment: {features.get('unemployment_rate')}")
    except Exception as exc:
        if verbose:
            print(f"  [macro] Unemployment failed: {exc}")
        features["unemployment_rate"] = None

    # ── 5. VIX (via CBOE Volatility Index — use AV global quote for ^VIX) ──
    # Alpha Vantage does not have a dedicated VIX endpoint, so we use
    # TIME_SERIES_DAILY for the VIX ETF proxy (VIXY) or direct GLOBAL_QUOTE.
    # Fallback: use SPY volatility as a VIX proxy.
    try:
        _throttle()
        data = _av_get("TIME_SERIES_DAILY", api_key, symbol="VIXY", outputsize="compact", datatype="json")
        ts_key = "Time Series (Daily)"
        ts = data.get(ts_key, {})
        if ts:
            sorted_dates = sorted(ts.keys(), reverse=True)
            # VIX level from VIXY close price (proxy)
            latest_close = _safe_float(ts[sorted_dates[0]].get("4. close"))
            features["vix_level"] = latest_close

            # VIX percentile over last ~252 trading days
            closes = []
            for d in sorted_dates[:252]:
                c = _safe_float(ts[d].get("4. close"))
                if c is not None:
                    closes.append(c)
            if latest_close is not None and len(closes) > 20:
                percentile = sum(1 for c in closes if c <= latest_close) / len(closes)
                features["vix_percentile_1y"] = round(percentile, 4)
            else:
                features["vix_percentile_1y"] = None
        else:
            features["vix_level"] = None
            features["vix_percentile_1y"] = None
        if verbose:
            print(f"  [macro] VIX level: {features.get('vix_level')}, percentile: {features.get('vix_percentile_1y')}")
    except Exception as exc:
        if verbose:
            print(f"  [macro] VIX failed: {exc}")
        features["vix_level"] = None
        features["vix_percentile_1y"] = None

    # ── 6. SPY Momentum (20-day) ────────────────────────────────────
    try:
        _throttle()
        data = _av_get("TIME_SERIES_DAILY", api_key, symbol="SPY", outputsize="compact", datatype="json")
        ts_key = "Time Series (Daily)"
        ts = data.get(ts_key, {})
        if ts:
            sorted_dates = sorted(ts.keys(), reverse=True)
            if len(sorted_dates) >= 21:
                latest_close = _safe_float(ts[sorted_dates[0]].get("4. close"))
                close_20d_ago = _safe_float(ts[sorted_dates[20]].get("4. close"))
                if latest_close is not None and close_20d_ago is not None and close_20d_ago > 0:
                    features["spy_momentum_20d"] = round(latest_close / close_20d_ago - 1, 6)
                else:
                    features["spy_momentum_20d"] = None
            else:
                features["spy_momentum_20d"] = None
        else:
            features["spy_momentum_20d"] = None
        if verbose:
            print(f"  [macro] SPY momentum 20d: {features.get('spy_momentum_20d')}")
    except Exception as exc:
        if verbose:
            print(f"  [macro] SPY momentum failed: {exc}")
        features["spy_momentum_20d"] = None

    # ── 7. Nonfarm Payroll (monthly) ────────────────────────────────
    try:
        _throttle()
        data = _av_get("NONFARM_PAYROLL", api_key)
        series = data.get("data", [])[:6]
        features["_raw_nonfarm_payroll"] = series
        if verbose:
            print(f"  [macro] Nonfarm Payroll: {len(series)} data points")
    except Exception as exc:
        if verbose:
            print(f"  [macro] Nonfarm Payroll failed: {exc}")
        features["_raw_nonfarm_payroll"] = []

    # ── 8. Real GDP (quarterly) ─────────────────────────────────────
    try:
        _throttle()
        data = _av_get("REAL_GDP", api_key, interval="quarterly")
        series = data.get("data", [])[:6]
        features["_raw_real_gdp"] = series
        if verbose:
            print(f"  [macro] Real GDP: {len(series)} data points")
    except Exception as exc:
        if verbose:
            print(f"  [macro] Real GDP failed: {exc}")
        features["_raw_real_gdp"] = []

    # ── 9. Retail Sales (monthly) ───────────────────────────────────
    try:
        _throttle()
        data = _av_get("RETAIL_SALES", api_key)
        series = data.get("data", [])[:6]
        features["_raw_retail_sales"] = series
        if verbose:
            print(f"  [macro] Retail Sales: {len(series)} data points")
    except Exception as exc:
        if verbose:
            print(f"  [macro] Retail Sales failed: {exc}")
        features["_raw_retail_sales"] = []

    # ── 10. WTI Crude Oil (monthly) ─────────────────────────────────
    try:
        _throttle()
        data = _av_get("WTI", api_key, interval="monthly")
        series = data.get("data", [])[:6]
        features["_raw_wti"] = series
        if verbose:
            print(f"  [macro] WTI Crude Oil: {len(series)} data points")
    except Exception as exc:
        if verbose:
            print(f"  [macro] WTI Crude Oil failed: {exc}")
        features["_raw_wti"] = []

    return features


# ═══════════════════════════════════════════════════════════════════════════
# Fundamental Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_fundamental_features(
    symbol: str,
    api_key: str,
    verbose: bool = False,
) -> Dict[str, Optional[float]]:
    """Fetch key fundamental metrics and return structured numeric features.

    Features extracted:
        - pe_ratio:               P/E ratio (TTM)
        - peg_ratio:              PEG ratio
        - pb_ratio:               Price-to-Book ratio
        - ps_ratio:               Price-to-Sales ratio (TTM)
        - ev_ebitda:              EV/EBITDA
        - dividend_yield:         Dividend yield (decimal, e.g. 0.006)
        - roe:                    Return on Equity (TTM, decimal)
        - profit_margin:          Profit margin (decimal)
        - revenue_growth_yoy:     Quarterly revenue growth YoY (decimal)
        - earnings_growth_yoy:    Quarterly earnings growth YoY (decimal)
        - current_ratio:          Current ratio
        - debt_to_equity:         Debt-to-equity ratio
        - beta:                   Beta
        - financial_health_score: Composite score 0-1 (derived)
    """
    features: Dict[str, Optional[float]] = {}

    try:
        _throttle()
        overview = _av_get("OVERVIEW", api_key, symbol=symbol)

        # Save raw overview data for report generation
        features["_raw_overview"] = overview

        features["pe_ratio"] = _safe_float(overview.get("PERatio"))
        features["peg_ratio"] = _safe_float(overview.get("PEGRatio"))
        features["pb_ratio"] = _safe_float(overview.get("PriceToBookRatio"))
        features["ps_ratio"] = _safe_float(overview.get("PriceToSalesRatioTTM"))
        features["ev_ebitda"] = _safe_float(overview.get("EVToEBITDA"))
        features["dividend_yield"] = _safe_float(overview.get("DividendYield"))
        features["beta"] = _safe_float(overview.get("Beta"))

        # Profitability
        roe = _safe_float(overview.get("ReturnOnEquityTTM"))
        features["roe"] = roe
        profit_margin = _safe_float(overview.get("ProfitMargin"))
        features["profit_margin"] = profit_margin

        # Growth
        rev_growth = _safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
        earn_growth = _safe_float(overview.get("QuarterlyEarningsGrowthYOY"))
        features["revenue_growth_yoy"] = rev_growth
        features["earnings_growth_yoy"] = earn_growth

        if verbose:
            print(f"  [fundamental] P/E: {features.get('pe_ratio')}, ROE: {roe}")

    except Exception as exc:
        if verbose:
            print(f"  [fundamental] Overview failed: {exc}")
        features["_raw_overview"] = {}
        for k in [
            "pe_ratio", "peg_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
            "dividend_yield", "beta", "roe", "profit_margin",
            "revenue_growth_yoy", "earnings_growth_yoy",
        ]:
            features[k] = None

    # ── Balance sheet metrics (current ratio, D/E) ──────────────────
    try:
        _throttle()
        balance = _av_get("BALANCE_SHEET", api_key, symbol=symbol)
        annual = balance.get("annualReports", [])
        latest = annual[0] if annual else {}

        current_assets = _safe_float(latest.get("totalCurrentAssets"))
        current_liabilities = _safe_float(latest.get("totalCurrentLiabilities"))
        total_liabilities = _safe_float(latest.get("totalLiabilities"))
        total_equity = _safe_float(latest.get("totalShareholderEquity"))

        if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
            features["current_ratio"] = round(current_assets / current_liabilities, 4)
        else:
            features["current_ratio"] = None

        if total_liabilities is not None and total_equity is not None and total_equity > 0:
            features["debt_to_equity"] = round(total_liabilities / total_equity, 4)
        else:
            features["debt_to_equity"] = None

        if verbose:
            print(f"  [fundamental] Current ratio: {features.get('current_ratio')}, D/E: {features.get('debt_to_equity')}")

    except Exception as exc:
        if verbose:
            print(f"  [fundamental] Balance sheet failed: {exc}")
        features["current_ratio"] = None
        features["debt_to_equity"] = None

    # ── Composite financial health score (0-1) ──────────────────────
    features["financial_health_score"] = _compute_health_score(features)

    # ── Raw data for LLM agents (income statement + cash flow) ──────
    try:
        _throttle()
        income_data = _av_get("INCOME_STATEMENT", api_key, symbol=symbol)
        features["_raw_income_statement"] = income_data
        if verbose:
            n_annual = len(income_data.get("annualReports", []))
            n_quarterly = len(income_data.get("quarterlyReports", []))
            print(f"  [fundamental] Income Statement: {n_annual} annual, {n_quarterly} quarterly")
    except Exception as exc:
        if verbose:
            print(f"  [fundamental] Income Statement failed: {exc}")
        features["_raw_income_statement"] = {}

    try:
        _throttle()
        cashflow_data = _av_get("CASH_FLOW", api_key, symbol=symbol)
        features["_raw_cash_flow"] = cashflow_data
        if verbose:
            n_annual = len(cashflow_data.get("annualReports", []))
            print(f"  [fundamental] Cash Flow: {n_annual} annual reports")
    except Exception as exc:
        if verbose:
            print(f"  [fundamental] Cash Flow failed: {exc}")
        features["_raw_cash_flow"] = {}

    return features


def _compute_health_score(features: Dict[str, Optional[float]]) -> float:
    """Compute a composite financial health score from 0 (weak) to 1 (strong).

    Scoring rubric (each component 0-1, then averaged):
        1. Profitability: ROE > 15% = 1.0, 0% = 0.5, < 0% = 0.0
        2. Leverage: D/E < 0.5 = 1.0, 0.5-2.0 linear, > 2.0 = 0.0
        3. Liquidity: Current ratio > 2.0 = 1.0, 1.0 = 0.5, < 0.5 = 0.0
        4. Growth: earnings_growth > 20% = 1.0, 0% = 0.5, < -20% = 0.0
    """
    scores = []

    # Profitability
    roe = features.get("roe")
    if roe is not None:
        # Normalize: ROE is typically in decimal form (e.g. 0.15 = 15%)
        roe_pct = roe * 100 if abs(roe) < 1 else roe
        scores.append(max(0.0, min(1.0, (roe_pct + 5) / 25)))

    # Leverage
    de = features.get("debt_to_equity")
    if de is not None:
        scores.append(max(0.0, min(1.0, 1.0 - (de - 0.5) / 1.5)))

    # Liquidity
    cr = features.get("current_ratio")
    if cr is not None:
        scores.append(max(0.0, min(1.0, cr / 2.0)))

    # Growth
    eg = features.get("earnings_growth_yoy")
    if eg is not None:
        eg_pct = eg * 100 if abs(eg) < 1 else eg
        scores.append(max(0.0, min(1.0, (eg_pct + 20) / 40)))

    if scores:
        return round(sum(scores) / len(scores), 4)
    return 0.5  # Neutral default when no data available


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation (for LLM Agents)
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(val: Optional[float], decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    """Format a float for display, returning 'N/A' if None."""
    if val is None:
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


def _fmt_large(val: Optional[float]) -> str:
    """Format a large number in human-readable form."""
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


def _format_series(series: list, suffix: str = "", prefix: str = "") -> str:
    """Format a time series into a readable table string."""
    if not series:
        return "  No data available."
    lines = []
    for dp in series:
        date = dp.get("date", "N/A")
        val = _safe_float(dp.get("value"))
        lines.append(f"  {date}: {_fmt(val, prefix=prefix, suffix=suffix)}")
    return "\n".join(lines)


def _trend_arrow(series: list) -> str:
    """Return a simple trend indicator based on recent values."""
    values = [_safe_float(dp.get("value")) for dp in series[:2]] if series else []
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return "→ (insufficient data)"
    if clean[0] > clean[1]:
        return "↑ (rising)"
    elif clean[0] < clean[1]:
        return "↓ (falling)"
    return "→ (flat)"


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return a/b as a percentage, or None if not computable."""
    if a is None or b is None or b == 0:
        return None
    return (a / b) * 100


def _build_macro_report(macro_features: Dict[str, Any]) -> str:
    """Build a formatted macro report from Provider data for LLM consumption.

    Uses both numeric features and raw time series data (_raw_* keys).
    """
    sections = []

    # 1. Federal Funds Rate
    ffr = macro_features.get("fed_funds_rate")
    rc3m = macro_features.get("rate_change_3m")
    sections.append(
        f"FEDERAL FUNDS RATE:\n"
        f"  Current: {_fmt(ffr, suffix='%')}\n"
        f"  3-Month Change: {_fmt(rc3m, suffix=' pp')}"
    )

    # 2. Treasury Yields
    y10 = macro_features.get("treasury_yield_10y")
    y2 = macro_features.get("treasury_yield_2y")
    spread = macro_features.get("yield_spread_10y2y")
    spread_status = ""
    if spread is not None:
        if spread < 0:
            spread_status = " ⚠️ INVERTED — historically a recession warning signal"
        elif spread < 0.5:
            spread_status = " ⚡ Very flat — potential slowdown signal"
        else:
            spread_status = " ✅ Positively sloped — normal expansion signal"
    sections.append(
        f"TREASURY YIELDS:\n"
        f"  10-Year: {_fmt(y10, suffix='%')}\n"
        f"  2-Year: {_fmt(y2, suffix='%')}\n"
        f"  Yield Spread (10Y-2Y): {_fmt(spread, suffix='%')}{spread_status}"
    )

    # 3. CPI
    cpi = macro_features.get("cpi_yoy")
    sections.append(f"CPI (Year-over-Year): {_fmt(cpi, suffix='%')}")

    # 4. Unemployment
    unemp = macro_features.get("unemployment_rate")
    sections.append(f"UNEMPLOYMENT RATE: {_fmt(unemp, suffix='%')}")

    # 5. VIX
    vix = macro_features.get("vix_level")
    vix_pct = macro_features.get("vix_percentile_1y")
    sections.append(
        f"VIX (Volatility Index Proxy):\n"
        f"  Level: {_fmt(vix)}\n"
        f"  1-Year Percentile: {_fmt(vix_pct, decimals=1, suffix='%') if vix_pct is not None else 'N/A'}"
    )

    # 6. SPY Momentum
    spy_mom = macro_features.get("spy_momentum_20d")
    sections.append(
        f"SPY 20-DAY MOMENTUM: {_fmt(spy_mom, decimals=4, suffix='') if spy_mom is not None else 'N/A'}"
    )

    # 7. Nonfarm Payroll (from raw data)
    nfp_series = macro_features.get("_raw_nonfarm_payroll", [])
    if nfp_series:
        trend = _trend_arrow(nfp_series)
        sections.append(
            f"NONFARM PAYROLL (Monthly, thousands) (Trend: {trend}):\n"
            + _format_series(nfp_series, suffix="K")
        )

    # 8. Real GDP (from raw data)
    gdp_series = macro_features.get("_raw_real_gdp", [])
    if gdp_series:
        trend = _trend_arrow(gdp_series)
        sections.append(
            f"REAL GDP (Quarterly, billions USD) (Trend: {trend}):\n"
            + _format_series(gdp_series, suffix="B")
        )

    # 9. Retail Sales (from raw data)
    retail_series = macro_features.get("_raw_retail_sales", [])
    if retail_series:
        trend = _trend_arrow(retail_series)
        sections.append(
            f"RETAIL SALES (Monthly, millions USD) (Trend: {trend}):\n"
            + _format_series(retail_series, suffix="M")
        )

    # 10. WTI Crude Oil (from raw data)
    wti_series = macro_features.get("_raw_wti", [])
    if wti_series:
        trend = _trend_arrow(wti_series)
        sections.append(
            f"WTI CRUDE OIL PRICE (Monthly, USD/barrel) (Trend: {trend}):\n"
            + _format_series(wti_series, prefix="$")
        )

    report = (
        f"\nMACROECONOMIC ENVIRONMENT REPORT (US)\n"
        f"{'=' * 60}\n"
        f"Data Source: Alpha Vantage  |  Most recent data points shown\n\n"
    )
    report += "\n\n".join(sections)
    return report


def _build_fundamental_report(
    symbol: str,
    fundamental_features: Dict[str, Any],
    overview_data: Dict[str, Any],
) -> str:
    """Build a formatted fundamental report from Provider data for LLM consumption.

    Uses numeric features, raw income statement, raw cash flow, and overview data.
    """
    company_name = overview_data.get("Name", symbol)
    sector = overview_data.get("Sector", "N/A")
    industry = overview_data.get("Industry", "N/A")
    market_cap = _safe_float(overview_data.get("MarketCapitalization"))
    analyst_target = _safe_float(overview_data.get("AnalystTargetPrice"))
    week52_high = _safe_float(overview_data.get("52WeekHigh"))
    week52_low = _safe_float(overview_data.get("52WeekLow"))
    eps = _safe_float(overview_data.get("EPS"))

    # From numeric features
    pe = fundamental_features.get("pe_ratio")
    peg = fundamental_features.get("peg_ratio")
    pb = fundamental_features.get("pb_ratio")
    ps = fundamental_features.get("ps_ratio")
    ev_ebitda = fundamental_features.get("ev_ebitda")
    div_yield = fundamental_features.get("dividend_yield")
    roe = fundamental_features.get("roe")
    roa = _safe_float(overview_data.get("ReturnOnAssetsTTM"))
    profit_margin = fundamental_features.get("profit_margin")
    beta = fundamental_features.get("beta")
    rev_growth = fundamental_features.get("revenue_growth_yoy")
    earn_growth = fundamental_features.get("earnings_growth_yoy")
    current_ratio = fundamental_features.get("current_ratio")
    debt_to_equity = fundamental_features.get("debt_to_equity")

    # From raw income statement
    income_data = fundamental_features.get("_raw_income_statement", {})
    annual_income = income_data.get("annualReports", [])
    quarterly_income = income_data.get("quarterlyReports", [])
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
    revenue_growth_calc = _pct(
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

    # From raw cash flow
    cashflow_data = fundamental_features.get("_raw_cash_flow", {})
    annual_cf = cashflow_data.get("annualReports", [])
    latest_cf = annual_cf[0] if annual_cf else {}
    operating_cf = _safe_float(latest_cf.get("operatingCashflow"))
    capex = _safe_float(latest_cf.get("capitalExpenditures"))
    free_cash_flow = (operating_cf - capex) if operating_cf is not None and capex is not None else None
    dividends_paid = _safe_float(latest_cf.get("dividendPayout"))

    # From raw balance sheet (already fetched in _fetch_fundamental_features)
    # We reuse the computed current_ratio and debt_to_equity from features
    # and get raw balance sheet values from overview
    total_assets_str = "N/A"
    total_liabilities_str = "N/A"
    total_equity_str = "N/A"
    cash_str = "N/A"
    total_debt_str = "N/A"

    # Format dividend yield
    div_yield_str = "N/A"
    if div_yield is not None:
        div_yield_pct = div_yield * 100 if div_yield < 1 else div_yield
        div_yield_str = f"{div_yield_pct:.2f}%"

    # Format ROE/ROA
    roe_str = "N/A"
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) < 1 else roe
        roe_str = f"{roe_pct:.2f}%"
    roa_str = "N/A"
    if roa is not None:
        roa_pct = roa * 100 if abs(roa) < 1 else roa
        roa_str = f"{roa_pct:.2f}%"

    # Format growth
    rev_growth_str = "N/A"
    if rev_growth is not None:
        rg_pct = rev_growth * 100 if abs(rev_growth) < 1 else rev_growth
        rev_growth_str = f"{rg_pct:.2f}%"
    earn_growth_str = "N/A"
    if earn_growth is not None:
        eg_pct = earn_growth * 100 if abs(earn_growth) < 1 else earn_growth
        earn_growth_str = f"{eg_pct:.2f}%"

    report = f"""
FUNDAMENTAL ANALYSIS FOR {symbol} ({company_name})
{'=' * 60}
Sector: {sector}  |  Industry: {industry}
Fiscal Year End: {latest_annual_inc.get('fiscalDateEnding', 'N/A')}

VALUATION METRICS:
- Market Cap: {_fmt_large(market_cap)}
- P/E Ratio (TTM): {_fmt(pe)}
- PEG Ratio: {_fmt(peg)}
- P/B Ratio: {_fmt(pb)}
- P/S Ratio (TTM): {_fmt(ps)}
- EV/EBITDA: {_fmt(ev_ebitda)}
- EPS (TTM): {_fmt(eps, prefix='$')}
- Dividend Yield: {div_yield_str}
- Beta: {_fmt(beta)}
- 52-Week Range: {_fmt(week52_low, prefix='$')} – {_fmt(week52_high, prefix='$')}
- Analyst Target Price: {_fmt(analyst_target, prefix='$')}

PROFITABILITY:
- Gross Margin: {_fmt(gross_margin, suffix='%')}
- Operating Margin: {_fmt(operating_margin_calc, suffix='%')}
- Net Margin: {_fmt(net_margin, suffix='%')}
- ROE (TTM): {roe_str}
- ROA (TTM): {roa_str}

INCOME STATEMENT (Annual: {latest_annual_inc.get('fiscalDateEnding', 'N/A')}):
- Revenue: {_fmt_large(annual_revenue)}
- Gross Profit: {_fmt_large(annual_gross_profit)}
- Operating Income: {_fmt_large(annual_operating_income)}
- EBITDA: {_fmt_large(annual_ebitda)}
- Net Income: {_fmt_large(annual_net_income)}

GROWTH:
- Revenue YoY Growth: {_fmt(revenue_growth_calc, suffix='%')}
- Net Income YoY Growth: {_fmt(net_income_growth, suffix='%')}
- Quarterly Revenue Growth YoY: {rev_growth_str}
- Quarterly Earnings Growth YoY: {earn_growth_str}

BALANCE SHEET:
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

def _fetch_macro_features_historical(
    api_key: str,
    start_date: datetime,
    end_date: datetime,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch historical macro indicators and return a daily-frequency DataFrame.

    Columns match MACRO_FEATURE_COLUMNS. Monthly data is forward-filled to daily.
    Daily data (VIX, SPY) is kept as-is.

    Returns:
        pd.DataFrame with DatetimeIndex and columns for each macro feature.
    """
    # Create a daily date range to reindex everything onto
    daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    result = pd.DataFrame(index=daily_idx)
    result.index.name = "date"

    # ── 1. Federal Funds Rate (monthly) ─────────────────────────────
    try:
        data = _av_get_cached(
            "FEDERAL_FUNDS_RATE", api_key,
            cache_key="fed_funds_rate_monthly",
            cache_dir=_MACRO_CACHE_DIR,
            interval="monthly",
        )
        series = data.get("data", [])
        df = _series_to_df(series)
        if not df.empty:
            df = df.rename(columns={"value": "fed_funds_rate"})
            # Compute 3-month rate change
            df["rate_change_3m"] = df["fed_funds_rate"] - df["fed_funds_rate"].shift(3)
            result = result.join(df[["fed_funds_rate", "rate_change_3m"]], how="left")
            result[["fed_funds_rate", "rate_change_3m"]] = result[["fed_funds_rate", "rate_change_3m"]].ffill()
        if verbose:
            n = result["fed_funds_rate"].notna().sum() if "fed_funds_rate" in result.columns else 0
            print(f"  [macro-hist] Fed Funds Rate: {n} daily values")
    except Exception as exc:
        if verbose:
            print(f"  [macro-hist] Fed Funds Rate failed: {exc}")

    # ── 2. Treasury Yields (10Y and 2Y, monthly) ───────────────────
    for maturity, col_name in [("10year", "treasury_yield_10y"), ("2year", "treasury_yield_2y")]:
        try:
            data = _av_get_cached(
                "TREASURY_YIELD", api_key,
                cache_key=f"treasury_yield_{maturity}_monthly",
                cache_dir=_MACRO_CACHE_DIR,
                interval="monthly", maturity=maturity,
            )
            series = data.get("data", [])
            df = _series_to_df(series)
            if not df.empty:
                df = df.rename(columns={"value": col_name})
                result = result.join(df[[col_name]], how="left")
                result[col_name] = result[col_name].ffill()
            if verbose:
                n = result[col_name].notna().sum() if col_name in result.columns else 0
                print(f"  [macro-hist] {col_name}: {n} daily values")
        except Exception as exc:
            if verbose:
                print(f"  [macro-hist] {col_name} failed: {exc}")

    # Yield spread
    if "treasury_yield_10y" in result.columns and "treasury_yield_2y" in result.columns:
        result["yield_spread_10y2y"] = result["treasury_yield_10y"] - result["treasury_yield_2y"]

    # ── 3. CPI (monthly, compute YoY) ──────────────────────────────
    try:
        data = _av_get_cached(
            "CPI", api_key,
            cache_key="cpi_monthly",
            cache_dir=_MACRO_CACHE_DIR,
            interval="monthly",
        )
        series = data.get("data", [])
        df = _series_to_df(series)
        if not df.empty:
            df = df.rename(columns={"value": "cpi_raw"})
            df = df.sort_index()
            # YoY: compare to 12 months ago
            df["cpi_yoy"] = (df["cpi_raw"] / df["cpi_raw"].shift(12) - 1) * 100
            result = result.join(df[["cpi_yoy"]], how="left")
            result["cpi_yoy"] = result["cpi_yoy"].ffill()
        if verbose:
            n = result["cpi_yoy"].notna().sum() if "cpi_yoy" in result.columns else 0
            print(f"  [macro-hist] CPI YoY: {n} daily values")
    except Exception as exc:
        if verbose:
            print(f"  [macro-hist] CPI failed: {exc}")

    # ── 4. Unemployment Rate (monthly) ─────────────────────────────
    try:
        data = _av_get_cached(
            "UNEMPLOYMENT", api_key,
            cache_key="unemployment",
            cache_dir=_MACRO_CACHE_DIR,
        )
        series = data.get("data", [])
        df = _series_to_df(series)
        if not df.empty:
            df = df.rename(columns={"value": "unemployment_rate"})
            result = result.join(df[["unemployment_rate"]], how="left")
            result["unemployment_rate"] = result["unemployment_rate"].ffill()
        if verbose:
            n = result["unemployment_rate"].notna().sum() if "unemployment_rate" in result.columns else 0
            print(f"  [macro-hist] Unemployment: {n} daily values")
    except Exception as exc:
        if verbose:
            print(f"  [macro-hist] Unemployment failed: {exc}")

    # ── 5. VIX (daily via VIXY proxy, full history) ────────────────
    try:
        data = _av_get_cached(
            "TIME_SERIES_DAILY", api_key,
            cache_key="VIXY_daily_full",
            cache_dir=_MACRO_CACHE_DIR,
            symbol="VIXY", outputsize="full", datatype="json",
        )
        ts_key = "Time Series (Daily)"
        ts = data.get(ts_key, {})
        if ts:
            rows = []
            for date_str, vals in ts.items():
                dt = pd.to_datetime(date_str, errors="coerce")
                close = _safe_float(vals.get("4. close"))
                if pd.notna(dt) and close is not None:
                    rows.append({"date": dt, "vix_level": close})
            if rows:
                vix_df = pd.DataFrame(rows).set_index("date").sort_index()
                # Compute rolling 252-day percentile
                vix_df["vix_percentile_1y"] = vix_df["vix_level"].rolling(252, min_periods=20).apply(
                    lambda x: (x <= x.iloc[-1]).sum() / len(x), raw=False
                )
                result = result.join(vix_df[["vix_level", "vix_percentile_1y"]], how="left")
                result[["vix_level", "vix_percentile_1y"]] = result[["vix_level", "vix_percentile_1y"]].ffill()
        if verbose:
            n = result["vix_level"].notna().sum() if "vix_level" in result.columns else 0
            print(f"  [macro-hist] VIX: {n} daily values")
    except Exception as exc:
        if verbose:
            print(f"  [macro-hist] VIX failed: {exc}")

    # ── 6. SPY Momentum (daily, 20-day return) ─────────────────────
    try:
        data = _av_get_cached(
            "TIME_SERIES_DAILY", api_key,
            cache_key="SPY_daily_full",
            cache_dir=_MACRO_CACHE_DIR,
            symbol="SPY", outputsize="full", datatype="json",
        )
        ts_key = "Time Series (Daily)"
        ts = data.get(ts_key, {})
        if ts:
            rows = []
            for date_str, vals in ts.items():
                dt = pd.to_datetime(date_str, errors="coerce")
                close = _safe_float(vals.get("4. close"))
                if pd.notna(dt) and close is not None:
                    rows.append({"date": dt, "spy_close": close})
            if rows:
                spy_df = pd.DataFrame(rows).set_index("date").sort_index()
                spy_df["spy_momentum_20d"] = spy_df["spy_close"] / spy_df["spy_close"].shift(20) - 1
                result = result.join(spy_df[["spy_momentum_20d"]], how="left")
                result["spy_momentum_20d"] = result["spy_momentum_20d"].ffill()
        if verbose:
            n = result["spy_momentum_20d"].notna().sum() if "spy_momentum_20d" in result.columns else 0
            print(f"  [macro-hist] SPY momentum: {n} daily values")
    except Exception as exc:
        if verbose:
            print(f"  [macro-hist] SPY momentum failed: {exc}")

    # Ensure all expected columns exist (fill missing ones with NaN)
    for col in MACRO_FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    # Filter to requested date range
    result = result.loc[start_date:end_date, MACRO_FEATURE_COLUMNS]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Historical Fundamental Feature Extraction (for training)
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_fundamental_features_historical(
    symbol: str,
    api_key: str,
    start_date: datetime,
    end_date: datetime,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch historical fundamental features from quarterly/annual reports.

    Fundamental data is reported quarterly. We align each report to its
    fiscal date end and forward-fill to daily frequency so each training
    row sees the fundamentals that were known at that point in time.

    Returns:
        pd.DataFrame with DatetimeIndex and columns for each fundamental feature.
    """
    daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
    result = pd.DataFrame(index=daily_idx)
    result.index.name = "date"

    # ── Income Statement (quarterly) → margins, growth, raw data for ratios ──
    income_rows: list[Dict[str, Any]] = []
    try:
        income_data = _av_get_cached(
            "INCOME_STATEMENT", api_key,
            cache_key=f"{symbol}_income_statement",
            cache_dir=_FUND_CACHE_DIR,
            symbol=symbol,
        )
        quarterly_reports = income_data.get("quarterlyReports", [])
        for report in quarterly_reports:
            fiscal_date = pd.to_datetime(report.get("fiscalDateEnding"), errors="coerce")
            if pd.isna(fiscal_date):
                continue
            net_income = _safe_float(report.get("netIncome"))
            total_revenue = _safe_float(report.get("totalRevenue"))
            gross_profit = _safe_float(report.get("grossProfit"))
            ebitda = _safe_float(report.get("ebitda"))

            row: Dict[str, Any] = {"date": fiscal_date}
            # Profit margin
            if net_income is not None and total_revenue is not None and total_revenue > 0:
                row["profit_margin"] = net_income / total_revenue
            # Store raw quarterly values for TTM calculations
            if net_income is not None:
                row["_q_net_income"] = net_income
            if total_revenue is not None:
                row["_q_total_revenue"] = total_revenue
            if ebitda is not None:
                row["_q_ebitda"] = ebitda
            income_rows.append(row)
        if verbose:
            print(f"  [fund-hist] Income statements: {len(quarterly_reports)} quarters")
    except Exception as exc:
        if verbose:
            print(f"  [fund-hist] Income statement failed: {exc}")

    if income_rows:
        income_df = pd.DataFrame(income_rows).set_index("date").sort_index()
        # Remove duplicate dates (keep latest)
        income_df = income_df[~income_df.index.duplicated(keep="last")]
        # Compute TTM (trailing 4 quarters) for revenue and ebitda
        if "_q_total_revenue" in income_df.columns:
            income_df["_ttm_revenue"] = income_df["_q_total_revenue"].rolling(4, min_periods=1).sum()
            # Revenue growth YoY: compare current quarter to same quarter 4 quarters ago
            income_df["revenue_growth_yoy"] = (
                income_df["_q_total_revenue"] / income_df["_q_total_revenue"].shift(4) - 1
            )
        if "_q_net_income" in income_df.columns:
            income_df["_ttm_net_income"] = income_df["_q_net_income"].rolling(4, min_periods=1).sum()
        if "_q_ebitda" in income_df.columns:
            income_df["_ttm_ebitda"] = income_df["_q_ebitda"].rolling(4, min_periods=1).sum()
        # Drop raw quarterly columns before joining
        cols_to_join = [c for c in income_df.columns if not c.startswith("_q_")]
        result = result.join(income_df[cols_to_join], how="left")

    # ── Balance Sheet (quarterly) → current ratio, D/E, raw data for ratios ──
    balance_rows: list[Dict[str, Any]] = []
    try:
        balance_data = _av_get_cached(
            "BALANCE_SHEET", api_key,
            cache_key=f"{symbol}_balance_sheet",
            cache_dir=_FUND_CACHE_DIR,
            symbol=symbol,
        )
        quarterly_reports = balance_data.get("quarterlyReports", [])
        for report in quarterly_reports:
            fiscal_date = pd.to_datetime(report.get("fiscalDateEnding"), errors="coerce")
            if pd.isna(fiscal_date):
                continue
            current_assets = _safe_float(report.get("totalCurrentAssets"))
            current_liabilities = _safe_float(report.get("totalCurrentLiabilities"))
            total_liabilities = _safe_float(report.get("totalLiabilities"))
            total_equity = _safe_float(report.get("totalShareholderEquity"))
            shares_outstanding = _safe_float(report.get("commonStockSharesOutstanding"))
            cash = _safe_float(report.get("cashAndCashEquivalentsAtCarryingValue"))
            if cash is None:
                cash = _safe_float(report.get("cashAndShortTermInvestments"))

            row: Dict[str, Any] = {"date": fiscal_date}
            if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
                row["current_ratio"] = current_assets / current_liabilities
            if total_liabilities is not None and total_equity is not None and total_equity > 0:
                row["debt_to_equity"] = total_liabilities / total_equity
            # Store raw values for price-dependent ratio calculations
            if total_equity is not None:
                row["_total_equity"] = total_equity
            if shares_outstanding is not None:
                row["_shares_outstanding"] = shares_outstanding
            if total_liabilities is not None:
                row["_total_liabilities"] = total_liabilities
            if cash is not None:
                row["_cash"] = cash
            balance_rows.append(row)
        if verbose:
            print(f"  [fund-hist] Balance sheets: {len(quarterly_reports)} quarters")
    except Exception as exc:
        if verbose:
            print(f"  [fund-hist] Balance sheet failed: {exc}")

    if balance_rows:
        balance_df = pd.DataFrame(balance_rows).set_index("date").sort_index()
        balance_df = balance_df[~balance_df.index.duplicated(keep="last")]
        for col in balance_df.columns:
            if col not in result.columns:
                result = result.join(balance_df[[col]], how="left")
            else:
                result[col] = result[col].fillna(balance_df[col])

    # ── Earnings (quarterly) → EPS for P/E, earnings growth ───────
    earnings_rows: list[Dict[str, Any]] = []
    try:
        earnings_data = _av_get_cached(
            "EARNINGS", api_key,
            cache_key=f"{symbol}_earnings",
            cache_dir=_FUND_CACHE_DIR,
            symbol=symbol,
        )
        quarterly_earnings = earnings_data.get("quarterlyEarnings", [])
        for report in quarterly_earnings:
            fiscal_date = pd.to_datetime(report.get("fiscalDateEnding"), errors="coerce")
            if pd.isna(fiscal_date):
                continue
            reported_eps = _safe_float(report.get("reportedEPS"))
            row = {"date": fiscal_date}
            if reported_eps is not None:
                row["reported_eps"] = reported_eps
            earnings_rows.append(row)
        if verbose:
            print(f"  [fund-hist] Earnings: {len(quarterly_earnings)} quarters")
    except Exception as exc:
        if verbose:
            print(f"  [fund-hist] Earnings failed: {exc}")

    if earnings_rows:
        eps_df = pd.DataFrame(earnings_rows).set_index("date").sort_index()
        eps_df = eps_df[~eps_df.index.duplicated(keep="last")]
        if "reported_eps" in eps_df.columns:
            # Trailing 4-quarter EPS for P/E calculation
            eps_df["_ttm_eps"] = eps_df["reported_eps"].rolling(4, min_periods=1).sum()
            # YoY earnings growth (compare to 4 quarters ago)
            eps_df["earnings_growth_yoy"] = (
                eps_df["reported_eps"] / eps_df["reported_eps"].shift(4) - 1
            )
            for col in ["_ttm_eps", "earnings_growth_yoy"]:
                if col not in result.columns:
                    result = result.join(eps_df[[col]], how="left")

    # ── Overview: no longer used for historical features ────────────
    # beta, peg_ratio, dividend_yield are now computed from historical data
    # in the training pipeline (see train_forecast_model.py Step 3.5).
    if verbose:
        print(f"  [fund-hist] Skipping OVERVIEW (features computed from historical data)")

    # ── Derived features ───────────────────────────────────────────
    # Forward-fill all quarterly data to daily
    for col in result.columns:
        result[col] = result[col].ffill()

    # ROE: compute from TTM net income / total equity (time-varying)
    if "_ttm_net_income" in result.columns and "_total_equity" in result.columns:
        valid_mask = result["_total_equity"].notna() & (result["_total_equity"].abs() > 0)
        result.loc[valid_mask, "roe"] = (
            result.loc[valid_mask, "_ttm_net_income"] / result.loc[valid_mask, "_total_equity"]
        )
    if "roe" not in result.columns:
        result["roe"] = np.nan

    # Revenue growth YoY (already computed in income_df if data available)
    if "revenue_growth_yoy" not in result.columns:
        result["revenue_growth_yoy"] = np.nan

    # Price-dependent features (pe_ratio, pb_ratio, ps_ratio, ev_ebitda,
    # beta, peg_ratio, dividend_yield) are set to NaN here and will be
    # computed in the training pipeline using the stock's Close price.
    for col in ["pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
                "beta", "peg_ratio", "dividend_yield"]:
        if col not in result.columns:
            result[col] = np.nan

    # Financial health score (computed row-by-row)
    health_scores = []
    for idx in result.index:
        row_dict = {col: result.at[idx, col] if col in result.columns else None
                    for col in ["roe", "debt_to_equity", "current_ratio", "earnings_growth_yoy"]}
        health_scores.append(_compute_health_score(row_dict))
    result["financial_health_score"] = health_scores

    # Ensure all expected columns exist (including intermediate columns
    # needed by the training pipeline for price-dependent calculations)
    _INTERMEDIATE_COLUMNS = [
        "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
        "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
    ]
    all_output_cols = FUNDAMENTAL_FEATURE_COLUMNS + _INTERMEDIATE_COLUMNS
    for col in all_output_cols:
        if col not in result.columns:
            result[col] = np.nan

    # Filter to requested date range and return all columns
    result = result.loc[start_date:end_date, all_output_cols]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

class MacroFundamentalFeatureProvider:
    """Extracts structured macro & fundamental features for downstream agents.

    Usage:
        provider = MacroFundamentalFeatureProvider(verbose=True)
        result = provider.extract("AAPL")
        macro = result["macro_features"]
        fundamental = result["fundamental_features"]
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def extract(self, stock_symbol: str = "") -> Dict[str, Any]:
        """Fetch macro and fundamental features.

        Args:
            stock_symbol: Ticker for fundamental data. If empty, only macro
                          features are returned.

        Returns:
            Dict with keys: status, macro_features, fundamental_features,
            degraded_reason (if applicable), summary.
        """
        symbol = stock_symbol.strip().upper()
        errors: list[str] = []

        try:
            api_key = _get_api_key()
        except RuntimeError as exc:
            return {
                "status": "degraded",
                "degraded_reason": str(exc),
                "macro_features": {},
                "fundamental_features": {},
                "summary": f"MacroFundamentalFeatureProvider: {exc}",
            }

        # ── Macro features (symbol-independent) ────────────────────
        if self.verbose:
            print("[macro_fundamental] Fetching macro features ...")
        try:
            macro_features = _fetch_macro_features(api_key, verbose=self.verbose)
        except Exception as exc:
            macro_features = {}
            errors.append(f"macro: {exc}")
            if self.verbose:
                print(f"[macro_fundamental] Macro extraction failed: {exc}")

        # ── Fundamental features (symbol-specific) ─────────────────
        fundamental_features: Dict[str, Optional[float]] = {}
        overview_data: Dict[str, Any] = {}
        if symbol:
            if self.verbose:
                print(f"[macro_fundamental] Fetching fundamental features for {symbol} ...")
            try:
                fundamental_features = _fetch_fundamental_features(
                    symbol, api_key, verbose=self.verbose,
                )
                # Reuse overview data already fetched by _fetch_fundamental_features
                overview_data = fundamental_features.pop("_raw_overview", {})
            except Exception as exc:
                fundamental_features = {}
                errors.append(f"fundamental: {exc}")
                if self.verbose:
                    print(f"[macro_fundamental] Fundamental extraction failed: {exc}")

        # ── Build raw reports for LLM agents ───────────────────────
        raw_macro_report = ""
        if macro_features:
            try:
                raw_macro_report = _build_macro_report(macro_features)
            except Exception as exc:
                if self.verbose:
                    print(f"[macro_fundamental] Macro report generation failed: {exc}")

        raw_fundamental_report = ""
        if fundamental_features and symbol:
            try:
                raw_fundamental_report = _build_fundamental_report(
                    symbol, fundamental_features, overview_data,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[macro_fundamental] Fundamental report generation failed: {exc}")

        # ── Build result ───────────────────────────────────────────
        n_macro = sum(1 for k, v in macro_features.items() if v is not None and not k.startswith("_raw"))
        n_fund = sum(1 for k, v in fundamental_features.items() if v is not None and not k.startswith("_raw"))
        total = n_macro + n_fund

        status = "success" if total > 0 else "degraded"
        summary_parts = [f"macro={n_macro}/{len(macro_features)}"]
        if symbol:
            summary_parts.append(f"fundamental={n_fund}/{len(fundamental_features)}")
        if errors:
            summary_parts.append(f"errors={len(errors)}")

        result: Dict[str, Any] = {
            "status": status,
            "macro_features": macro_features,
            "fundamental_features": fundamental_features,
            "raw_macro_report": raw_macro_report,
            "raw_fundamental_report": raw_fundamental_report,
            "summary": f"MacroFundamentalFeatureProvider: {', '.join(summary_parts)}",
        }
        if errors:
            result["degraded_reason"] = "; ".join(errors)

        if self.verbose:
            print(f"[macro_fundamental] Done: {result['summary']}")

        return result

    def extract_historical(
        self,
        stock_symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical macro & fundamental features as a time-series DataFrame.

        This method is designed for the **training pipeline** where each row
        needs the macro/fundamental values that were known at that point in time.

        For **inference**, use ``extract()`` which returns the latest snapshot.

        Args:
            stock_symbol: Ticker for fundamental data. If empty, only macro
                          features are returned.
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            pd.DataFrame with DatetimeIndex (business days) and columns
            matching ALL_MACRO_FUNDAMENTAL_COLUMNS. Missing values are
            forward-filled from the most recent available data point.
        """
        symbol = stock_symbol.strip().upper()

        try:
            api_key = _get_api_key()
        except RuntimeError as exc:
            if self.verbose:
                print(f"[macro_fundamental] extract_historical failed: {exc}")
            # Return empty DataFrame with expected columns
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            return pd.DataFrame(np.nan, index=daily_idx, columns=ALL_MACRO_FUNDAMENTAL_COLUMNS)

        # ── Macro features (symbol-independent) ────────────────────
        if self.verbose:
            print(f"[macro_fundamental] Fetching historical macro features ({start_date.date()} to {end_date.date()}) ...")
        try:
            macro_df = _fetch_macro_features_historical(
                api_key, start_date, end_date, verbose=self.verbose,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[macro_fundamental] Historical macro extraction failed: {exc}")
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            macro_df = pd.DataFrame(np.nan, index=daily_idx, columns=MACRO_FEATURE_COLUMNS)

        # ── Fundamental features (symbol-specific) ─────────────────
        if symbol:
            if self.verbose:
                print(f"[macro_fundamental] Fetching historical fundamental features for {symbol} ...")
            try:
                fund_df = _fetch_fundamental_features_historical(
                    symbol, api_key, start_date, end_date, verbose=self.verbose,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[macro_fundamental] Historical fundamental extraction failed: {exc}")
                daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
                fund_df = pd.DataFrame(np.nan, index=daily_idx, columns=FUNDAMENTAL_FEATURE_COLUMNS)
        else:
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            fund_df = pd.DataFrame(np.nan, index=daily_idx, columns=FUNDAMENTAL_FEATURE_COLUMNS)

        # ── Merge macro + fundamental ──────────────────────────────
        combined = macro_df.join(fund_df, how="outer")

        # Intermediate columns for price-dependent feature computation
        _INTERMEDIATE_COLUMNS = [
            "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
            "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
        ]

        # Ensure all expected columns exist
        all_cols = ALL_MACRO_FUNDAMENTAL_COLUMNS + _INTERMEDIATE_COLUMNS
        for col in all_cols:
            if col not in combined.columns:
                combined[col] = np.nan

        combined = combined[all_cols]

        n_macro = combined[MACRO_FEATURE_COLUMNS].notna().any(axis=0).sum()
        n_fund = combined[FUNDAMENTAL_FEATURE_COLUMNS].notna().any(axis=0).sum()
        n_intermediate = combined[_INTERMEDIATE_COLUMNS].notna().any(axis=0).sum()
        if self.verbose:
            print(
                f"[macro_fundamental] Historical extraction done: "
                f"{len(combined)} days, macro={n_macro}/{len(MACRO_FEATURE_COLUMNS)}, "
                f"fundamental={n_fund}/{len(FUNDAMENTAL_FEATURE_COLUMNS)}, "
                f"intermediate={n_intermediate}/{len(_INTERMEDIATE_COLUMNS)}"
            )

        return combined

    def extract_fundamental_only_historical(
        self,
        stock_symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch only fundamental features (no macro) for a single ticker.

        This is an optimized variant of ``extract_historical`` for the V2
        cross-sectional training pipeline where macro data is fetched once
        and shared across all tickers, while fundamental data must be
        fetched independently per ticker.

        Returns:
            pd.DataFrame with DatetimeIndex and fundamental + intermediate
            columns. Macro columns are NOT included.
        """
        symbol = stock_symbol.strip().upper()
        if not symbol:
            raise ValueError("stock_symbol is required for fundamental-only extraction")

        try:
            api_key = _get_api_key()
        except RuntimeError as exc:
            if self.verbose:
                print(f"[macro_fundamental] extract_fundamental_only failed: {exc}")
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            return pd.DataFrame(np.nan, index=daily_idx, columns=FUNDAMENTAL_FEATURE_COLUMNS)

        if self.verbose:
            print(f"[macro_fundamental] Fetching fundamental-only for {symbol} ...")

        try:
            fund_df = _fetch_fundamental_features_historical(
                symbol, api_key, start_date, end_date, verbose=self.verbose,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[macro_fundamental] Fundamental extraction failed for {symbol}: {exc}")
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            fund_df = pd.DataFrame(np.nan, index=daily_idx, columns=FUNDAMENTAL_FEATURE_COLUMNS)

        # Intermediate columns for price-dependent feature computation
        _INTERMEDIATE_COLUMNS = [
            "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
            "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
        ]

        all_cols = FUNDAMENTAL_FEATURE_COLUMNS + _INTERMEDIATE_COLUMNS
        for col in all_cols:
            if col not in fund_df.columns:
                fund_df[col] = np.nan

        fund_df = fund_df[[c for c in all_cols if c in fund_df.columns]]

        if self.verbose:
            n_fund = fund_df[FUNDAMENTAL_FEATURE_COLUMNS].notna().any(axis=0).sum()
            n_inter = fund_df[[c for c in _INTERMEDIATE_COLUMNS if c in fund_df.columns]].notna().any(axis=0).sum()
            print(
                f"[macro_fundamental] Fundamental-only done for {symbol}: "
                f"{len(fund_df)} days, fundamental={n_fund}/{len(FUNDAMENTAL_FEATURE_COLUMNS)}, "
                f"intermediate={n_inter}/{len(_INTERMEDIATE_COLUMNS)}"
            )

        return fund_df

    def extract_macro_only_historical(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch only macro features (no fundamental) as a time-series.

        This is an optimized variant for the V2 cross-sectional training
        pipeline where macro data is shared across all tickers.

        Returns:
            pd.DataFrame with DatetimeIndex and macro columns only.
        """
        try:
            api_key = _get_api_key()
        except RuntimeError as exc:
            if self.verbose:
                print(f"[macro_fundamental] extract_macro_only failed: {exc}")
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            return pd.DataFrame(np.nan, index=daily_idx, columns=MACRO_FEATURE_COLUMNS)

        if self.verbose:
            print(f"[macro_fundamental] Fetching macro-only ({start_date.date()} to {end_date.date()}) ...")

        try:
            macro_df = _fetch_macro_features_historical(
                api_key, start_date, end_date, verbose=self.verbose,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[macro_fundamental] Macro extraction failed: {exc}")
            daily_idx = pd.date_range(start=start_date, end=end_date, freq="B")
            macro_df = pd.DataFrame(np.nan, index=daily_idx, columns=MACRO_FEATURE_COLUMNS)

        for col in MACRO_FEATURE_COLUMNS:
            if col not in macro_df.columns:
                macro_df[col] = np.nan

        macro_df = macro_df[[c for c in MACRO_FEATURE_COLUMNS if c in macro_df.columns]]

        if self.verbose:
            n_macro = macro_df.notna().any(axis=0).sum()
            print(
                f"[macro_fundamental] Macro-only done: "
                f"{len(macro_df)} days, macro={n_macro}/{len(MACRO_FEATURE_COLUMNS)}"
            )

        return macro_df


# ── Convenience: feature name lists for training pipeline ───────────────

MACRO_FEATURE_COLUMNS = [
    "fed_funds_rate",
    "rate_change_3m",
    "treasury_yield_10y",
    "treasury_yield_2y",
    "yield_spread_10y2y",
    "cpi_yoy",
    "unemployment_rate",
    "vix_level",
    "vix_percentile_1y",
    "spy_momentum_20d",
]

FUNDAMENTAL_FEATURE_COLUMNS = [
    "pe_ratio",
    "peg_ratio",
    "pb_ratio",
    "ps_ratio",
    "ev_ebitda",
    "dividend_yield",
    "roe",
    "profit_margin",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "current_ratio",
    "debt_to_equity",
    "beta",
    "financial_health_score",
]

ALL_MACRO_FUNDAMENTAL_COLUMNS = MACRO_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS
