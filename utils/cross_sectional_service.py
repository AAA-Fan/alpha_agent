"""
Cross-Sectional Feature Service for inference.

Manages downloading all 100 tickers' daily data, computing base features,
and calculating cross-sectional rank features for a target ticker.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# GICS sector mapping (must match training pipeline)
SECTOR_MAP = {
    "Technology": 0,
    "Healthcare": 1,
    "Financials": 2,
    "Consumer Discretionary": 3,
    "Communication Services": 4,
    "Industrials": 5,
    "Consumer Staples": 6,
    "Energy": 7,
    "Utilities": 8,
    "Real Estate": 9,
    "Materials": 10,
}

# Columns to compute cross-sectional rank for (must match training pipeline)
RANK_COLUMNS_TECH = [
    "momentum_5", "momentum_20", "sma_20_ratio", "sma_50_ratio",
    "macd_hist", "rsi_14", "volatility_20", "daily_volatility_20",
    "atr_14", "volume_zscore_20", "drawdown_60", "overnight_gap",
    "intraday_return", "return_1d", "return_5d",
    "momentum_trend_align", "rsi_deviation", "vol_confirmed_momentum",
    "mean_reversion", "vol_adj_momentum_5",
]

RANK_COLUMNS_FUND = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
    "dividend_yield", "roe", "profit_margin",
    "revenue_growth_yoy", "earnings_growth_yoy", "beta",
]


class CrossSectionalFeatureService:
    """Manages cross-sectional feature computation for inference.

    Downloads all tickers' latest daily data, computes base features,
    and calculates cross-sectional rank features for a target ticker.
    Uses 24-hour caching to minimize API calls.
    """

    def __init__(
        self,
        ticker_list_path: str = "data/sp500_top100.json",
        cache_dir: str = "data/cross_section_cache",
        cache_ttl_hours: float = 24.0,
        verbose: bool = False,
    ):
        self.ticker_list_path = ticker_list_path
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_hours = cache_ttl_hours
        self.verbose = verbose

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load ticker list
        self.tickers: List[str] = []
        if os.path.exists(ticker_list_path):
            with open(ticker_list_path, "r") as f:
                data = json.load(f)
            self.tickers = data.get("tickers", [])

        # Load sector/industry maps
        self.sector_map_data: Dict[str, str] = {}  # ticker -> sector name
        self.industry_map_data: Dict[str, str] = {}  # ticker -> industry name
        self.industry_code_map: Dict[str, int] = {}  # industry name -> code
        self._load_industry_maps()

        # Cached features for all tickers (date -> DataFrame)
        self._cached_features: Optional[pd.DataFrame] = None
        self._cache_date: Optional[str] = None
        self._cache_time: float = 0.0

    def _load_industry_maps(self) -> None:
        """Load sector and industry maps from cache files."""
        sector_file = self.cache_dir / "sector_map.json"
        industry_file = self.cache_dir / "industry_map.json"

        if sector_file.exists():
            with open(sector_file, "r") as f:
                self.sector_map_data = json.load(f)

        if industry_file.exists():
            with open(industry_file, "r") as f:
                self.industry_code_map = json.load(f)

        # Build reverse map: ticker -> industry name
        overview_dir = self.cache_dir / "overview_cache"
        if overview_dir.exists():
            for ticker in self.tickers:
                cache_file = overview_dir / f"{ticker}_overview.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, "r") as f:
                            overview = json.load(f)
                        self.sector_map_data[ticker] = overview.get("Sector", "Unknown")
                        self.industry_map_data[ticker] = overview.get("Industry", "Unknown")
                    except Exception:
                        pass

    def _is_cache_fresh(self) -> bool:
        """Check if cached features are still fresh."""
        if self._cached_features is None:
            return False
        age_hours = (time.time() - self._cache_time) / 3600
        return age_hours < self.cache_ttl_hours

    def _compute_base_features_for_ticker(
        self, ticker: str, data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute base features for a single ticker from its OHLCV data.

        Returns a dict of feature_name -> value for the latest row.
        """
        if len(data) < 60:
            return {}

        close = pd.to_numeric(data["Close"], errors="coerce")
        high = pd.to_numeric(data["High"], errors="coerce")
        low = pd.to_numeric(data["Low"], errors="coerce")
        open_ = pd.to_numeric(data["Open"], errors="coerce")
        volume = pd.to_numeric(data["Volume"], errors="coerce")
        returns = close.pct_change()

        features: Dict[str, float] = {}

        # Momentum
        features["momentum_5"] = float(close.pct_change(5).iloc[-1])
        features["momentum_20"] = float(close.pct_change(20).iloc[-1])

        # Moving average ratios
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        features["sma_20_ratio"] = float((close / sma_20 - 1).iloc[-1])
        features["sma_50_ratio"] = float((close / sma_50 - 1).iloc[-1])

        # MACD histogram
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        features["macd_hist"] = float((macd - macd_signal).iloc[-1])

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        features["rsi_14"] = float(rsi.iloc[-1])

        # Volatility
        features["volatility_20"] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252))
        features["daily_volatility_20"] = float(returns.rolling(20).std().iloc[-1])

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        features["atr_14"] = float(tr.rolling(14).mean().iloc[-1])

        # Volume z-score
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std().replace(0, np.nan)
        features["volume_zscore_20"] = float(((volume - vol_mean) / vol_std).iloc[-1])

        # Drawdown
        rolling_max = close.rolling(60).max()
        features["drawdown_60"] = float((close / rolling_max - 1).iloc[-1])

        # Overnight gap & intraday return
        features["overnight_gap"] = float((open_ / prev_close - 1).iloc[-1])
        features["intraday_return"] = float((close / open_ - 1).iloc[-1])

        # Returns
        features["return_1d"] = float(returns.iloc[-1])
        features["return_5d"] = float((close / close.shift(5) - 1).iloc[-1])

        # Interaction features
        features["momentum_trend_align"] = features["momentum_5"] * features["sma_20_ratio"]
        features["rsi_deviation"] = (features["rsi_14"] - 50.0) / 50.0
        vol_z_clipped = max(-3, min(3, features["volume_zscore_20"]))
        features["vol_confirmed_momentum"] = features["momentum_5"] * vol_z_clipped
        dd_clipped = features["drawdown_60"]
        m5_clipped = max(-0.1, min(0.1, features["momentum_5"]))
        features["mean_reversion"] = dd_clipped * m5_clipped
        safe_vol = features["daily_volatility_20"] if features["daily_volatility_20"] != 0 else np.nan
        if safe_vol and not np.isnan(safe_vol):
            features["vol_adj_momentum_5"] = max(-5, min(5, features["momentum_5"] / safe_vol))
        else:
            features["vol_adj_momentum_5"] = 0.0

        # Replace NaN with 0
        for k, v in features.items():
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

        return features

    def _refresh_cache(self) -> None:
        """Download latest data for all tickers and compute features."""
        from utils.yfinance_cache import get_historical_data

        if self.verbose:
            print(f"[cross_section] Refreshing features for {len(self.tickers)} tickers ...")

        all_features: List[Dict[str, Any]] = []

        for ticker in self.tickers:
            try:
                data = get_historical_data(ticker, interval="daily", days=120)
                if data.empty or len(data) < 60:
                    continue
                features = self._compute_base_features_for_ticker(ticker, data)
                if features:
                    features["ticker"] = ticker
                    all_features.append(features)
            except Exception as exc:
                if self.verbose:
                    print(f"  [warn] {ticker}: {exc}")

        if all_features:
            self._cached_features = pd.DataFrame(all_features)
            self._cache_time = time.time()
            self._cache_date = datetime.now().strftime("%Y-%m-%d")
            if self.verbose:
                print(f"[cross_section] Cached features for {len(all_features)} tickers")

    def get_cross_sectional_features(
        self,
        target_ticker: str,
        target_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute cross-sectional rank features for the target ticker.

        1. Load/refresh cached features for all 100 tickers
        2. Compute ranks across all tickers for the current date
        3. Return rank features for the target ticker

        Returns:
            Dict with keys like "momentum_5_rank", "rsi_14_rank", etc.
        """
        if not self._is_cache_fresh():
            self._refresh_cache()

        result: Dict[str, float] = {}

        if self._cached_features is None or self._cached_features.empty:
            # Return default median ranks
            for col in RANK_COLUMNS_TECH + RANK_COLUMNS_FUND:
                result[f"{col}_rank"] = 0.5
            result["rs_5d"] = 0.0
            result["rs_20d"] = 0.0
            return result

        # Build a combined DataFrame with target ticker's features
        df = self._cached_features.copy()

        # If target ticker is not in the cache, add it
        target_in_cache = target_ticker in df["ticker"].values
        if not target_in_cache:
            target_row = {"ticker": target_ticker}
            for col in RANK_COLUMNS_TECH:
                target_row[col] = target_features.get(col, 0.0)
            df = pd.concat([df, pd.DataFrame([target_row])], ignore_index=True)

        # Compute ranks
        rank_cols = RANK_COLUMNS_TECH + RANK_COLUMNS_FUND
        for col in rank_cols:
            if col in df.columns:
                df[f"{col}_rank"] = df[col].rank(pct=True, method="average")

        # Compute relative strength
        for period in [5, 20]:
            col = f"momentum_{period}"
            if col in df.columns:
                median = df[col].median()
                df[f"rs_{period}d"] = df[col] - median

        # Extract target ticker's rank features
        target_row = df[df["ticker"] == target_ticker]
        if target_row.empty:
            for col in rank_cols:
                result[f"{col}_rank"] = 0.5
            result["rs_5d"] = 0.0
            result["rs_20d"] = 0.0
        else:
            row = target_row.iloc[0]
            for col in rank_cols:
                rank_col = f"{col}_rank"
                if rank_col in row.index:
                    val = row[rank_col]
                    result[rank_col] = float(val) if not np.isnan(val) else 0.5
                else:
                    result[rank_col] = 0.5
            for period in [5, 20]:
                rs_col = f"rs_{period}d"
                if rs_col in row.index:
                    val = row[rs_col]
                    result[rs_col] = float(val) if not np.isnan(val) else 0.0
                else:
                    result[rs_col] = 0.0

        return result

    def get_sector_code(self, ticker: str) -> int:
        """Return the GICS sector code for a ticker."""
        sector = self.sector_map_data.get(ticker, "Unknown")
        return SECTOR_MAP.get(sector, -1)

    def get_industry_code(self, ticker: str) -> int:
        """Return the sub-industry code for a ticker."""
        industry = self.industry_map_data.get(ticker, "Unknown")
        return self.industry_code_map.get(industry, -1)
