"""
Walk-Forward Backtest Engine

Agent-in-the-Loop backtest engine that replays historical data through the
real agent pipeline (Feature → Regime → Forecast → Risk) to evaluate system
performance.

Execution model: Signal at Close → Execute at Next Open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.regime_agent import RegimeAgent
from agents.forecast_agent import ForecastAgent
from agents.risk_agent import RiskAgent
from utils.macro_fundamental_provider import MacroFundamentalFeatureProvider
from utils.yfinance_cache import get_historical_data

logger = logging.getLogger(__name__)

# Maximum backtest span (years)
MAX_BACKTEST_YEARS = 5


@dataclass
class BacktestResult:
    """Container for backtest output."""

    ticker: str
    start_date: str
    end_date: str
    horizon_days: int
    trade_log: List[Dict[str, Any]]
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    params: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


class BacktestEngine:
    """Agent-in-the-Loop walk-forward backtest engine.

    Replays historical data through the real agent pipeline
    (Feature → Regime → Forecast → Risk) to evaluate system performance.
    """

    def __init__(
        self,
        feature_agent: FeatureEngineeringAgent,
        regime_agent: RegimeAgent,
        forecast_agent: ForecastAgent,
        risk_agent: RiskAgent,
        macro_fund_provider: MacroFundamentalFeatureProvider,
        horizon_days: int = 5,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 5.0,
        verbose: bool = False,
    ) -> None:
        self.feature_agent = feature_agent
        self.regime_agent = regime_agent
        self.forecast_agent = forecast_agent
        self.risk_agent = risk_agent
        self.macro_fund_provider = macro_fund_provider
        self.horizon_days = horizon_days
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ── Historical data loading ──────────────────────────────────────────

    def _load_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load and trim historical OHLCV data for the backtest period.

        Fetches full history and clips to [start_date - warmup_buffer, end_date].
        """
        self._log(f"[backtest] Loading historical data for {ticker}")
        data = get_historical_data(ticker, interval="daily", outputsize="full")

        if data.empty:
            raise ValueError(f"No historical data available for {ticker}")

        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        data = data.sort_index()

        # Clip to backtest period (with extra buffer for warmup)
        # We need extra days before start_date for feature calculation warmup
        buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=120)
        end_ts = pd.Timestamp(end_date)

        data = data.loc[buffer_start:end_ts]

        if len(data) < 80:
            raise ValueError(
                f"Insufficient data for {ticker} in [{start_date}, {end_date}]: "
                f"only {len(data)} bars (need at least 80)."
            )

        return data

    # ── Macro data historical snapshots ────────────────────────────────────

    def _init_macro_snapshots(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Pre-fetch historical macro & fundamental data for the backtest period.

        Uses MacroFundamentalFeatureProvider.extract_historical() to get
        time-varying macro/fundamental data, then builds per-date snapshots
        in the same format as the live extract() output so downstream agents
        see no interface change.

        Falls back to a single current snapshot if historical fetch fails.
        """
        self._log("[backtest] Fetching historical macro/fundamental data")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        snapshots: Dict[str, Dict[str, Any]] = {}
        self._macro_hist_available = False

        # Temporarily extend cache TTL for backtest (avoid re-fetching from API)
        import utils.macro_fundamental_provider as _mfp
        original_ttl = _mfp._CACHE_TTL_DAYS
        _mfp._CACHE_TTL_DAYS = max(original_ttl, 365)

        try:
            # Extend start date for warmup (beta needs 252 days of history)
            extended_start = start_dt - pd.Timedelta(days=120)
            hist_df = self.macro_fund_provider.extract_historical(
                stock_symbol=ticker,
                start_date=extended_start,
                end_date=end_dt,
            )

            if hist_df is not None and not hist_df.empty:
                from utils.macro_fundamental_provider import (
                    MACRO_FEATURE_COLUMNS,
                    FUNDAMENTAL_FEATURE_COLUMNS,
                )

                hist_df = hist_df.sort_index()
                hist_df.index = pd.to_datetime(hist_df.index)

                # ── Compute derived features (same as Stage 2/3) ────────
                _INTERMEDIATE_COLS = [
                    "_ttm_eps", "_ttm_revenue", "_ttm_ebitda", "_ttm_net_income",
                    "_total_equity", "_shares_outstanding", "_total_liabilities", "_cash",
                ]

                # Get close prices for price-dependent features
                price_data = get_historical_data(ticker, interval="daily", outputsize="full")
                if not isinstance(price_data.index, pd.DatetimeIndex):
                    price_data.index = pd.to_datetime(price_data.index)
                close_series = pd.to_numeric(price_data["Close"], errors="coerce")
                close_for_mf = close_series.reindex(hist_df.index, method="ffill")

                if "_ttm_eps" in hist_df.columns:
                    ttm_eps = hist_df["_ttm_eps"]
                    valid = ttm_eps.notna() & (ttm_eps.abs() > 0.01) & close_for_mf.notna()
                    hist_df.loc[valid, "pe_ratio"] = close_for_mf[valid] / ttm_eps[valid]

                if "_total_equity" in hist_df.columns and "_shares_outstanding" in hist_df.columns:
                    equity_col = hist_df["_total_equity"]
                    shares = hist_df["_shares_outstanding"]
                    valid = equity_col.notna() & shares.notna() & (shares > 0) & close_for_mf.notna()
                    bvps = equity_col[valid] / shares[valid]
                    bvps_valid = bvps.abs() > 0.01
                    final_idx = bvps_valid.index[bvps_valid]
                    hist_df.loc[final_idx, "pb_ratio"] = close_for_mf[final_idx] / bvps[bvps_valid]

                if "_ttm_revenue" in hist_df.columns and "_shares_outstanding" in hist_df.columns:
                    ttm_rev = hist_df["_ttm_revenue"]
                    shares = hist_df["_shares_outstanding"]
                    valid = ttm_rev.notna() & shares.notna() & (shares > 0) & (ttm_rev > 0) & close_for_mf.notna()
                    rps = ttm_rev[valid] / shares[valid]
                    hist_df.loc[valid, "ps_ratio"] = close_for_mf[valid] / rps

                if all(c in hist_df.columns for c in ["_ttm_ebitda", "_shares_outstanding", "_total_liabilities", "_cash"]):
                    shares = hist_df["_shares_outstanding"]
                    ttm_ebitda = hist_df["_ttm_ebitda"]
                    total_liab = hist_df["_total_liabilities"].fillna(0)
                    cash = hist_df["_cash"].fillna(0)
                    valid = shares.notna() & (shares > 0) & ttm_ebitda.notna() & (ttm_ebitda.abs() > 0) & close_for_mf.notna()
                    market_cap = close_for_mf[valid] * shares[valid]
                    ev = market_cap + total_liab[valid] - cash[valid]
                    hist_df.loc[valid, "ev_ebitda"] = ev / ttm_ebitda[valid]

                try:
                    from pathlib import Path as _Path
                    spy_cache_file = _Path("data/training_cache/SPY_daily.csv")
                    if spy_cache_file.exists():
                        spy_df = pd.read_csv(spy_cache_file, index_col=0, parse_dates=True)
                    else:
                        spy_df = get_historical_data("SPY", interval="daily", outputsize="full")
                    if not spy_df.empty:
                        spy_close = pd.to_numeric(spy_df["Close"], errors="coerce")
                        spy_returns = spy_close.pct_change()
                        stock_returns = close_series.pct_change()
                        aligned = pd.DataFrame({"stock": stock_returns, "spy": spy_returns}).dropna()
                        if len(aligned) > 60:
                            rolling_cov = aligned["stock"].rolling(252, min_periods=60).cov(aligned["spy"])
                            rolling_var = aligned["spy"].rolling(252, min_periods=60).var()
                            rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
                            hist_df["beta"] = rolling_beta.reindex(hist_df.index).ffill()
                except Exception as exc:
                    self._log(f"[backtest] beta computation failed: {exc}")

                if "pe_ratio" in hist_df.columns and "earnings_growth_yoy" in hist_df.columns:
                    pe = hist_df["pe_ratio"]
                    eg = hist_df["earnings_growth_yoy"]
                    eg_pct = eg * 100
                    valid = pe.notna() & eg_pct.notna() & (eg_pct.abs() > 1.0) & (pe > 0)
                    hist_df.loc[valid, "peg_ratio"] = pe[valid] / eg_pct[valid]

                # Drop intermediate columns
                for col in _INTERMEDIATE_COLS:
                    if col in hist_df.columns:
                        hist_df.drop(columns=[col], inplace=True)

                # Build per-date snapshots keyed by "YYYY-MM-DD"
                for date_idx, row in hist_df.iterrows():
                    date_key = str(date_idx)[:10]  # "YYYY-MM-DD"

                    macro_feats: Dict[str, Any] = {}
                    fund_feats: Dict[str, Any] = {}

                    for col in MACRO_FEATURE_COLUMNS:
                        val = row.get(col)
                        macro_feats[col] = float(val) if pd.notna(val) else None

                    for col in FUNDAMENTAL_FEATURE_COLUMNS:
                        val = row.get(col)
                        fund_feats[col] = float(val) if pd.notna(val) else None

                    snapshots[date_key] = {
                        "status": "success",
                        "macro_features": macro_feats,
                        "fundamental_features": fund_feats,
                    }

                self._macro_hist_available = True
                # Pre-sort date keys for efficient binary search lookup
                self._sorted_macro_dates = sorted(snapshots.keys())
                self._log(
                    f"[backtest] Historical macro data loaded: {len(snapshots)} daily snapshots "
                    f"({hist_df.index.min().strftime('%Y-%m-%d')} to {hist_df.index.max().strftime('%Y-%m-%d')})"
                    f" (with derived features)"
                )
            else:
                self._log("[backtest] Historical macro data returned empty, falling back to current snapshot")

        except Exception as exc:
            self._log(f"[backtest] Historical macro fetch failed: {exc}, falling back to current snapshot")
        finally:
            # Restore original cache TTL
            _mfp._CACHE_TTL_DAYS = original_ttl

        # Fallback: use current snapshot for all months if historical fetch failed
        if not snapshots:
            self._log("[backtest] Using current macro snapshot as fallback")
            try:
                current_macro = self.macro_fund_provider.extract(ticker)
            except Exception as exc:
                self._log(f"[backtest] Macro data extraction failed: {exc}")
                current_macro = {}

            current = start_dt
            while current <= end_dt:
                key = current.strftime("%Y-%m")
                snapshots[key] = current_macro
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

        return snapshots

    def _get_macro_snapshot(
        self,
        current_date: Any,
        macro_snapshots: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get macro snapshot for the given date.

        Tries exact date match first (historical mode), then falls back
        to the nearest earlier date, then to month-level key (legacy mode).
        """
        date_str = str(current_date)[:10]  # "YYYY-MM-DD"

        # Exact date match (historical mode)
        if date_str in macro_snapshots:
            return macro_snapshots[date_str]

        # Nearest earlier date via binary search (historical mode — handles weekends/holidays)
        if self._macro_hist_available and hasattr(self, '_sorted_macro_dates'):
            import bisect
            idx = bisect.bisect_right(self._sorted_macro_dates, date_str) - 1
            if idx >= 0:
                return macro_snapshots[self._sorted_macro_dates[idx]]

        # Month-level fallback (legacy mode)
        month_key = date_str[:7]  # "YYYY-MM"
        return macro_snapshots.get(month_key, {})

    # ── Feature history construction ─────────────────────────────────────

    def _build_feature_history(
        self,
        ticker: str,
        historical_data: pd.DataFrame,
        current_idx: int,
        lookback: int = 60,
        feature_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build feature_history for RegimeAgent smoothing.

        For each of the last `lookback` days, compute features from data[:day]
        and collect the features dict into a time-ascending list.

        Uses a cache to avoid recomputing features for the same day index.
        """
        start_idx = max(0, current_idx - lookback)
        feature_history: List[Dict[str, Any]] = []

        for day_idx in range(start_idx, current_idx + 1):
            # Check cache first
            if feature_cache is not None and day_idx in feature_cache:
                cached = feature_cache[day_idx]
                if cached:
                    feature_history.append(cached)
                continue

            # Slice data up to this day (no look-ahead)
            data_slice = historical_data.iloc[: day_idx + 1]
            if len(data_slice) < 60:
                if feature_cache is not None:
                    feature_cache[day_idx] = {}
                continue

            try:
                result = self.feature_agent.analyze(ticker, data_override=data_slice)
                features = result.get("features", {})
            except Exception:
                features = {}

            if feature_cache is not None:
                feature_cache[day_idx] = features

            if features:
                feature_history.append(features)

        return feature_history

    # ── Memory simulation ────────────────────────────────────────────────

    @staticmethod
    def _build_simulated_memory(
        ticker: str,
        trade_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a simulated MemoryAgent output from accumulated trade history.

        This allows RiskAgent's Kelly position sizing to adapt during backtest,
        just as it would in live trading with real MemoryAgent data.
        """
        if not trade_log:
            return {
                "agent": "memory",
                "status": "skipped",
                "memory": {},
                "track_record_factor": 1.0,
                "summary": "No trade history yet.",
            }

        # Filter trades that were actually executed
        ticker_trades = [t for t in trade_log if t.get("position_size", 0) != 0]

        if not ticker_trades:
            return {
                "agent": "memory",
                "status": "skipped",
                "memory": {},
                "track_record_factor": 1.0,
                "summary": "No executed trades yet.",
            }

        # Compute directional accuracy
        correct = sum(1 for t in ticker_trades if t["net_return"] > 0)
        total = len(ticker_trades)
        accuracy = correct / total if total > 0 else 0.5

        # Compute avg_win / avg_loss
        wins = [t["raw_return"] for t in ticker_trades if t["raw_return"] > 0]
        losses = [abs(t["raw_return"]) for t in ticker_trades if t["raw_return"] < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.09
        avg_loss = sum(losses) / len(losses) if losses else 0.02

        # Track record factor (same logic as real MemoryAgent)
        if total >= 3:
            if accuracy >= 0.6:
                track_record_factor = min(1.2, 0.8 + accuracy * 0.5)
            elif accuracy >= 0.4:
                track_record_factor = 1.0
            else:
                track_record_factor = max(0.5, accuracy * 1.5)
        else:
            track_record_factor = 1.0

        return {
            "agent": "memory",
            "status": "success",
            "memory": {
                "prediction_count": total,
                "directional_accuracy": round(accuracy, 4),
                "avg_win": round(avg_win, 6),
                "avg_loss": round(avg_loss, 6),
            },
            "track_record_factor": round(track_record_factor, 4),
            "summary": f"Simulated memory: {total} trades, accuracy={accuracy:.2%}",
        }

    # ── Holding period simulation ────────────────────────────────────────

    @staticmethod
    def _simulate_holding_period(
        entry_price: float,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        final_close: float,
        position_size: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Tuple[float, str]:
        """Simulate daily stop-loss / take-profit checks during holding period.

        Returns:
            (exit_price, exit_reason)
            exit_reason: "horizon" | "stop_loss" | "take_profit"
        """
        direction = 1 if position_size > 0 else -1

        for i in range(len(high_prices)):
            if direction == 1:  # Long position
                if low_prices[i] <= entry_price * (1 - stop_loss_pct):
                    return entry_price * (1 - stop_loss_pct), "stop_loss"
                if high_prices[i] >= entry_price * (1 + take_profit_pct):
                    return entry_price * (1 + take_profit_pct), "take_profit"
            else:  # Short position
                if high_prices[i] >= entry_price * (1 + stop_loss_pct):
                    return entry_price * (1 + stop_loss_pct), "stop_loss"
                if low_prices[i] <= entry_price * (1 - take_profit_pct):
                    return entry_price * (1 - take_profit_pct), "take_profit"

        return final_close, "horizon"

    # ── Equity curve construction ────────────────────────────────────────

    @staticmethod
    def _build_equity_curve(trade_log: List[Dict[str, Any]]) -> pd.Series:
        """Build cumulative equity curve from trade log.

        Initial equity = 1.0 (normalized).
        Each trade's net_return is compounded: equity *= (1 + net_return).
        """
        equity = 1.0
        curve: Dict[str, float] = {}

        for trade in trade_log:
            date = trade["date"]
            net_return = trade.get("net_return", 0.0)
            equity *= 1.0 + net_return
            curve[str(date)] = equity

        if not curve:
            return pd.Series(dtype=float, name="equity")

        series = pd.Series(curve, name="equity")
        series.index = pd.to_datetime(series.index)
        return series

    @staticmethod
    def _build_benchmark_curve(start_date: str, end_date: str) -> pd.Series:
        """Build SPY buy-and-hold benchmark curve, normalized to 1.0."""
        try:
            spy_data = get_historical_data("SPY", interval="daily", outputsize="full")
            if not isinstance(spy_data.index, pd.DatetimeIndex):
                spy_data.index = pd.to_datetime(spy_data.index)
            spy_data = spy_data.sort_index()
            spy_data = spy_data.loc[start_date:end_date]

            if spy_data.empty:
                return pd.Series(dtype=float, name="benchmark")

            close = pd.to_numeric(spy_data["Close"], errors="coerce")
            benchmark = close / close.iloc[0]
            benchmark.name = "benchmark"
            return benchmark
        except Exception:
            return pd.Series(dtype=float, name="benchmark")

    # ── Main run method ──────────────────────────────────────────────────

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        warmup_days: int = 60,
        entry_cutoff_date: Optional[str] = None,
    ) -> BacktestResult:
        """Execute walk-forward backtest.

        Args:
            ticker: Stock symbol.
            start_date: Backtest start date (YYYY-MM-DD).
            end_date: Backtest end date (YYYY-MM-DD). Historical data / exit
                prices are loaded up to this date (inclusive). Callers should
                pass `month_end + horizon_days + buffer` when running monthly
                slices so positions opened near month-end can still reach a
                natural horizon exit without being truncated.
            warmup_days: Number of days reserved for feature calculation warmup.
            entry_cutoff_date: Last date (YYYY-MM-DD) on which a *new* entry
                may be opened. If None, defaults to `end_date`. Used by the
                walk-forward orchestrator to restrict entries to the
                prediction month while still allowing positions to exit on
                days that fall in the following month's data buffer.

        Returns:
            BacktestResult with trade log, equity curve, and benchmark.
        """
        ticker = ticker.upper().strip()
        warnings: List[str] = []

        # Enforce max backtest span
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        span_years = (end_ts - start_ts).days / 365.25
        if span_years > MAX_BACKTEST_YEARS:
            original_start = start_date
            start_ts = end_ts - pd.Timedelta(days=int(MAX_BACKTEST_YEARS * 365.25))
            start_date = start_ts.strftime("%Y-%m-%d")
            warnings.append(
                f"Backtest span {span_years:.1f}y exceeds {MAX_BACKTEST_YEARS}y limit. "
                f"Trimmed start from {original_start} to {start_date}."
            )

        # Add in-sample bias warning
        warnings.append(
            "ForecastAgent LightGBM model may have in-sample bias for the backtest period."
        )

        # Load data
        historical_data = self._load_historical_data(ticker, start_date, end_date)
        dates = historical_data.index
        open_prices = pd.to_numeric(historical_data["Open"], errors="coerce").values
        high_prices = pd.to_numeric(historical_data["High"], errors="coerce").values
        low_prices = pd.to_numeric(historical_data["Low"], errors="coerce").values
        close_prices = pd.to_numeric(historical_data["Close"], errors="coerce").values

        # Find the index of start_date in the data
        start_idx = dates.searchsorted(pd.Timestamp(start_date))
        start_idx = max(start_idx, warmup_days)
        total_days = len(dates)

        # Resolve entry cutoff (new entries may only open on dates <= cutoff).
        # Defaults to end_date when the caller does not supply one.
        cutoff_ts = pd.Timestamp(entry_cutoff_date) if entry_cutoff_date else pd.Timestamp(end_date)

        self._log(
            f"[backtest] {ticker}: {total_days} total bars, "
            f"stepping from idx={start_idx} with horizon={self.horizon_days}, "
            f"entry_cutoff={cutoff_ts.strftime('%Y-%m-%d')}"
        )

        # Initialize macro snapshots (historical time-varying data)
        macro_snapshots = self._init_macro_snapshots(ticker, start_date, end_date)

        # Add warning if historical macro data was not available
        if not getattr(self, '_macro_hist_available', False):
            warnings.append(
                "Macro data uses current snapshot for all months (historical FRED data fetch failed)."
            )
        else:
            self._log("[backtest] Using time-varying historical macro data (no look-ahead bias)")

        # Feature cache to avoid recomputation
        feature_cache: Dict[int, Dict[str, Any]] = {}

        trade_log: List[Dict[str, Any]] = []
        step_count = 0

        # Walk-forward loop — step = horizon_days on executed/rejected bars,
        # step = 1 when an agent fails. This matches the Stage3 debug loop
        # (t += horizon after each evaluation) so the sampling cadence is
        # identical between debug and production runs.
        t = start_idx
        while t < total_days:
            current_date = dates[t]

            # Stop evaluating once we pass the entry cutoff. Any position
            # already opened before the cutoff is allowed to exit using the
            # full-range data window the caller provided.
            if current_date > cutoff_ts:
                break

            step_count += 1

            if self.verbose and step_count % 10 == 0:
                self._log(
                    f"[backtest] Step {step_count}: date={current_date.strftime('%Y-%m-%d')}, "
                    f"trades={len(trade_log)}"
                )

            # 1. Slice data up to day t (no look-ahead)
            data_slice = historical_data.iloc[: t + 1]

            # 2. Build feature_history for Regime smoothing (last 60 days)
            feature_history = self._build_feature_history(
                ticker, historical_data, t, lookback=60, feature_cache=feature_cache,
            )

            # 3. Run agent pipeline on data_slice
            try:
                feature_result = self.feature_agent.analyze(ticker, data_override=data_slice)
            except Exception as exc:
                self._log(f"[backtest] FeatureAgent failed at {current_date}: {exc}")
                t += self.horizon_days
                continue

            # Cache current day's features
            features = feature_result.get("features", {})
            if features:
                feature_cache[t] = features

            macro_features = self._get_macro_snapshot(current_date, macro_snapshots)

            try:
                regime_result = self.regime_agent.analyze(
                    ticker, feature_result, macro_features, feature_history=feature_history,
                )
            except Exception as exc:
                self._log(f"[backtest] RegimeAgent failed at {current_date}: {exc}")
                t += self.horizon_days
                continue

            try:
                forecast_result = self.forecast_agent.analyze(
                    ticker, feature_result, regime_result, macro_features,
                )
            except Exception as exc:
                self._log(f"[backtest] ForecastAgent failed at {current_date}: {exc}")
                t += self.horizon_days
                continue

            memory_result = self._build_simulated_memory(ticker, trade_log)

            try:
                risk_result = self.risk_agent.analyze(
                    ticker, forecast_result, regime_result,
                    feature_result, memory_result, macro_features,
                )
            except Exception as exc:
                self._log(f"[backtest] RiskAgent failed at {current_date}: {exc}")
                t += self.horizon_days
                continue

            # 4. Record signal
            forecast = forecast_result.get("forecast", {})
            risk_plan = risk_result.get("risk_plan", {})
            regime = regime_result.get("regime", {})

            signal = {
                "date": current_date.strftime("%Y-%m-%d"),
                "action": forecast.get("action", "hold"),
                "probability_up": forecast.get("probability_up", 0.5),
                "position_size": risk_plan.get("position_size_fraction", 0.0),
                "stop_loss_pct": risk_plan.get("stop_loss_pct", 0.05),
                "take_profit_pct": risk_plan.get("take_profit_pct", 0.10),
                "regime_state": regime.get("state", "unknown"),
                "signal_alignment": risk_plan.get("signal_alignment"),
                "kelly_fraction": risk_plan.get("kelly_fraction"),
                "reject_reason": risk_plan.get("reject_reason"),
                "risk_flags": risk_plan.get("risk_flags", []),
            }

            # 5. Simulate execution (Signal at Close → Execute at Next Open)
            if t + 1 < total_days and signal["position_size"] != 0:
                entry_price = open_prices[t + 1]  # Next day open

                if np.isnan(entry_price) or entry_price <= 0:
                    t += self.horizon_days
                    continue

                exit_date_idx = min(t + 1 + self.horizon_days, total_days - 1)

                # Simulate holding period with stop-loss / take-profit
                holding_highs = high_prices[t + 1 : exit_date_idx + 1]
                holding_lows = low_prices[t + 1 : exit_date_idx + 1]
                final_close = close_prices[exit_date_idx]

                actual_exit_price, exit_reason = self._simulate_holding_period(
                    entry_price,
                    holding_highs,
                    holding_lows,
                    final_close,
                    signal["position_size"],
                    signal["stop_loss_pct"],
                    signal["take_profit_pct"],
                )

                # Calculate return
                raw_return = (actual_exit_price / entry_price - 1.0) * np.sign(
                    signal["position_size"]
                )
                position_return = raw_return * abs(signal["position_size"])
                cost = (
                    (self.transaction_cost_bps + self.slippage_bps)
                    / 10000.0
                    * 2
                    * abs(signal["position_size"])
                )
                net_return = position_return - cost

                trade = {
                    **signal,
                    "entry_price": float(entry_price),
                    "exit_price": float(actual_exit_price),
                    "exit_reason": exit_reason,
                    "raw_return": float(raw_return),
                    "net_return": float(net_return),
                    "position_return": float(position_return),
                    "cost": float(cost),
                }
                trade_log.append(trade)
            elif signal["position_size"] == 0:
                # Record rejected signal (position_size = 0)
                trade = {
                    **signal,
                    "entry_price": 0.0,
                    "exit_price": 0.0,
                    "exit_reason": "rejected",
                    "raw_return": 0.0,
                    "net_return": 0.0,
                    "position_return": 0.0,
                    "cost": 0.0,
                }
                trade_log.append(trade)

            # Advance by horizon_days to mirror Stage3 debug sampling and
            # keep holding periods non-overlapping by construction.
            t += self.horizon_days

        self._log(
            f"[backtest] Completed: {step_count} steps, {len(trade_log)} trades "
            f"({sum(1 for t in trade_log if t['position_size'] != 0)} executed)"
        )

        # Build equity and benchmark curves
        equity_curve = self._build_equity_curve(trade_log)
        benchmark_curve = self._build_benchmark_curve(start_date, end_date)

        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            horizon_days=self.horizon_days,
            trade_log=trade_log,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            params={
                "transaction_cost_bps": self.transaction_cost_bps,
                "slippage_bps": self.slippage_bps,
                "warmup_days": warmup_days,
            },
            warnings=warnings,
        )
