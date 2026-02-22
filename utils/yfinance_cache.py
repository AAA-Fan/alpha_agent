"""Utilities for caching Alpha Vantage requests across agents."""

from __future__ import annotations

import os
import time
from datetime import datetime
from io import StringIO
from threading import RLock
from typing import Dict, Tuple

import pandas as pd
import requests

_ALPHA_BASE_URL = "https://www.alphavantage.co/query"
_ALPHA_PERIOD_FUNCTION = {
    "daily": "TIME_SERIES_DAILY",
    "weekly": "TIME_SERIES_WEEKLY",
    "monthly": "TIME_SERIES_MONTHLY",
}
_INTERVAL_ALIASES = {
    "1d": "daily",
    "daily": "daily",
    "day": "daily",
    "d": "daily",
    "1w": "weekly",
    "1wk": "weekly",
    "weekly": "weekly",
    "week": "weekly",
    "w": "weekly",
    "1mo": "monthly",
    "monthly": "monthly",
    "month": "monthly",
    "m": "monthly",
}

_download_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_download_lock = RLock()
_intraday_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_intraday_lock = RLock()
_rate_lock = RLock()
_last_request_ts = 0.0


def _get_api_key() -> str:
    """Load Alpha Vantage API key from environment."""
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY is not configured. "
            "Add it to your environment or .env file."
        )
    return key


def _normalize_interval(interval: str) -> str:
    interval_key = interval.lower()
    if interval_key in _ALPHA_PERIOD_FUNCTION:
        return interval_key
    alias = _INTERVAL_ALIASES.get(interval_key)
    if alias:
        return alias
    raise ValueError(
        f"Unsupported Alpha Vantage period '{interval}'. "
        "Use one of daily, weekly, or monthly."
    )


def _normalize_intraday_interval(interval: str) -> str:
    """Normalize intraday interval to Alpha Vantage accepted values."""
    key = interval.lower().strip()
    aliases = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "60min": "60min",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"Unsupported intraday interval '{interval}'. "
        "Use one of 1min, 5min, 15min, 30min, 60min."
    )


def _throttle_requests() -> None:
    """Best-effort rate limiting to respect Alpha Vantage quotas."""
    limit = int(os.getenv("ALPHAVANTAGE_RATE_LIMIT_PER_MIN", "5"))
    if limit <= 0:
        return
    min_interval = 60.0 / float(limit)
    with _rate_lock:
        global _last_request_ts
        now = time.time()
        elapsed = now - _last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_ts = time.time()


def _fetch_alpha_series(symbol: str, interval: str) -> pd.DataFrame:
    """Retrieve a time series from Alpha Vantage and normalize the response."""
    _throttle_requests()
    params = {
        "function": _ALPHA_PERIOD_FUNCTION[interval],
        "symbol": symbol.upper(),
        "apikey": _get_api_key(),
        "datatype": "csv",
    }
    try:
        response = requests.get(_ALPHA_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Alpha Vantage request failed: {exc}") from exc

    payload = response.text.strip()
    if not payload:
        return pd.DataFrame()
    if payload.startswith("{"):
        # Alpha Vantage returns JSON for errors even when datatype=csv.
        raise RuntimeError(f"Alpha Vantage error: {payload}")

    frame = pd.read_csv(StringIO(payload))
    if frame.empty:
        return frame

    renamed = frame.rename(
        columns={
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    renamed["Date"] = pd.to_datetime(renamed["Date"])
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    normalized = renamed.sort_values("Date").set_index("Date")
    return normalized


def _fetch_alpha_intraday(symbol: str, interval: str, outputsize: str = "compact") -> pd.DataFrame:
    """Retrieve intraday time series from Alpha Vantage and normalize the response."""
    _throttle_requests()
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol.upper(),
        "interval": interval,
        "outputsize": outputsize,
        "apikey": _get_api_key(),
        "datatype": "csv",
    }
    try:
        response = requests.get(_ALPHA_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Alpha Vantage intraday request failed: {exc}") from exc

    payload = response.text.strip()
    if not payload:
        return pd.DataFrame()
    if payload.startswith("{"):
        raise RuntimeError(f"Alpha Vantage error: {payload}")

    frame = pd.read_csv(StringIO(payload))
    if frame.empty:
        return frame

    renamed = frame.rename(
        columns={
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    renamed["Date"] = pd.to_datetime(renamed["Date"])
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    normalized = renamed.sort_values("Date").set_index("Date")
    return normalized


def get_historical_data(symbol: str, interval: str = "daily", days: int | None = None) -> pd.DataFrame:
    """
    Cached wrapper around Alpha Vantage TIME_SERIES_* endpoints.

    Args:
        symbol: Equity ticker symbol.
        interval: One of ``daily``, ``weekly``, ``monthly`` (aliases supported).
        days: Optional number of most recent rows to return (useful for daily analysis).
    """
    normalized_interval = _normalize_interval(interval)
    key = (symbol.upper(), normalized_interval)
    with _download_lock:
        cached = _download_cache.get(key)
    if cached is None:
        data = _fetch_alpha_series(symbol, normalized_interval)
        with _download_lock:
            _download_cache[key] = data
    else:
        data = cached

    result = data.copy()
    if days is not None and days > 0:
        result = result.tail(days)
    return result


def get_intraday_data(
    symbol: str,
    interval: str = "5min",
    outputsize: str = "compact",
    ttl_seconds: int | None = None,
) -> pd.DataFrame:
    """
    Cached wrapper around Alpha Vantage TIME_SERIES_INTRADAY.

    Args:
        symbol: Equity ticker symbol.
        interval: One of 1min, 5min, 15min, 30min, 60min.
        outputsize: "compact" or "full".
        ttl_seconds: Cache TTL in seconds (default: ALPHAVANTAGE_INTRADAY_TTL or 300).
    """
    normalized_interval = _normalize_intraday_interval(interval)
    ttl = int(os.getenv("ALPHAVANTAGE_INTRADAY_TTL", "300")) if ttl_seconds is None else ttl_seconds
    key = (symbol.upper(), normalized_interval)

    with _intraday_lock:
        cached = _intraday_cache.get(key)
    if cached:
        fetched_at, data = cached
        if time.time() - fetched_at <= ttl:
            return data.copy()

    data = _fetch_alpha_intraday(symbol, normalized_interval, outputsize=outputsize)
    with _intraday_lock:
        _intraday_cache[key] = (time.time(), data)
    return data.copy()


def get_price_history(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Provide a historical slice between ``start`` and ``end`` using cached data.

    Since Alpha Vantage does not support arbitrary intervals, ``interval`` is treated
    as a hint (mapped to daily/weekly/monthly via aliases).
    """
    period = _normalize_interval(interval)
    data = get_historical_data(symbol, interval=period)
    if data.empty:
        return data

    mask = (data.index >= start) & (data.index <= end)
    return data.loc[mask].copy()


if __name__ == "__main__":
    data = get_historical_data("AAPL", "daily", days=30)
    print(data.tail())
