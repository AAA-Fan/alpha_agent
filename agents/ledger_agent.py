"""
Pair Ledger Agent
Builds and maintains a ledger of momentum-similar stock pairs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple

import pandas as pd

from utils.yfinance_cache import get_historical_data


@dataclass(frozen=True)
class PairSpec:
    symbol_a: str
    symbol_b: str
    similarity: float
    method: str
    window: int


class PairLedgerAgent:
    """Builds and persists momentum-similar stock pairs."""

    def __init__(
        self,
        universe_path: str | None = None,
        ledger_path: str | None = None,
        lookback_days: int | None = None,
        top_k: int | None = None,
        min_correlation: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.universe_path = universe_path or os.getenv(
            "PAIR_UNIVERSE_PATH", "data/sp500_top100.json"
        )
        self.ledger_path = ledger_path or os.getenv(
            "PAIR_LEDGER_PATH", "data/pairs_ledger.json"
        )
        self.lookback_days = lookback_days or int(os.getenv("PAIR_LOOKBACK_DAYS", "90"))
        self.top_k = top_k or int(os.getenv("PAIR_TOP_K", "5"))
        self.min_correlation = (
            float(os.getenv("PAIR_MIN_CORR", "0.85"))
            if min_correlation is None
            else min_correlation
        )
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _load_universe(self) -> List[str]:
        if os.getenv("PAIR_UNIVERSE"):
            tickers = [t.strip().upper() for t in os.getenv("PAIR_UNIVERSE", "").split(",") if t.strip()]
            if tickers:
                return tickers

        if not os.path.exists(self.universe_path):
            raise FileNotFoundError(
                f"Universe file not found: {self.universe_path}. Provide PAIR_UNIVERSE or PAIR_UNIVERSE_PATH."
            )

        with open(self.universe_path, "r") as f:
            payload = json.load(f)

        tickers = payload.get("tickers")
        if not tickers:
            raise ValueError(f"Universe file missing tickers: {self.universe_path}")
        return [str(t).upper() for t in tickers]

    def _fetch_returns(self, symbol: str) -> pd.Series | None:
        data = get_historical_data(symbol, interval="daily", days=self.lookback_days + 1)
        if data.empty or "Close" not in data.columns:
            return None

        close = pd.to_numeric(data["Close"], errors="coerce")
        returns = close.pct_change().dropna()
        if returns.empty:
            return None
        returns.name = symbol
        return returns

    def _compute_pairs(self, returns_frame: pd.DataFrame) -> List[PairSpec]:
        if returns_frame.shape[1] < 2:
            return []

        corr = returns_frame.corr()
        symbols = list(corr.columns)
        pairs: List[Tuple[float, str, str]] = []
        for i, sym_a in enumerate(symbols):
            for j in range(i + 1, len(symbols)):
                sym_b = symbols[j]
                value = corr.iat[i, j]
                if pd.isna(value):
                    continue
                if self.min_correlation is not None and value < self.min_correlation:
                    continue
                pairs.append((float(value), sym_a, sym_b))

        pairs.sort(key=lambda item: item[0], reverse=True)
        selected = pairs[: self.top_k]
        return [
            PairSpec(symbol_a=a, symbol_b=b, similarity=sim, method="return_corr", window=self.lookback_days)
            for sim, a, b in selected
        ]

    def build_pairs(self) -> Dict[str, Any]:
        universe = self._load_universe()
        returns_map: Dict[str, pd.Series] = {}
        errors: Dict[str, str] = {}

        for symbol in universe:
            try:
                returns = self._fetch_returns(symbol)
                if returns is None or len(returns) < 5:
                    errors[symbol] = "insufficient data"
                    continue
                returns_map[symbol] = returns
            except Exception as exc:
                errors[symbol] = str(exc)

        if not returns_map:
            return {
                "status": "error",
                "error": "No valid return series available for pair construction.",
                "errors": errors,
            }

        returns_frame = pd.concat(returns_map.values(), axis=1, join="inner")
        returns_frame = returns_frame.dropna(how="any")
        if returns_frame.empty:
            return {
                "status": "error",
                "error": "Return series could not be aligned across symbols.",
                "errors": errors,
            }

        pairs = self._compute_pairs(returns_frame)
        if not pairs:
            return {
                "status": "error",
                "error": "No pairs met the similarity threshold.",
                "errors": errors,
            }

        ledger = {
            "status": "success",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lookback_days": self.lookback_days,
            "universe_count": len(universe),
            "pairs": [pair.__dict__ for pair in pairs],
            "errors": errors,
        }
        return ledger

    def save_ledger(self, ledger: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)

    def load_ledger(self) -> Dict[str, Any] | None:
        if not os.path.exists(self.ledger_path):
            return None
        with open(self.ledger_path, "r") as f:
            return json.load(f)

    def _normalize_pair(self, symbol_a: str, symbol_b: str) -> Tuple[str, str]:
        left = symbol_a.upper().strip()
        right = symbol_b.upper().strip()
        return (left, right) if left <= right else (right, left)

    def add_pair(self, symbol_a: str, symbol_b: str, note: str = "manual") -> Dict[str, Any]:
        ledger = self.load_ledger() or {
            "status": "success",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lookback_days": self.lookback_days,
            "universe_count": None,
            "pairs": [],
            "errors": {},
        }

        a, b = self._normalize_pair(symbol_a, symbol_b)
        existing = {
            self._normalize_pair(pair["symbol_a"], pair["symbol_b"])
            for pair in ledger.get("pairs", [])
        }
        if (a, b) in existing:
            return ledger

        ledger.setdefault("pairs", []).append(
            {
                "symbol_a": a,
                "symbol_b": b,
                "similarity": None,
                "method": note,
                "window": self.lookback_days,
            }
        )
        self.save_ledger(ledger)
        return ledger

    def remove_pair(self, symbol_a: str, symbol_b: str) -> Dict[str, Any]:
        ledger = self.load_ledger()
        if not ledger:
            return {
                "status": "error",
                "error": "Ledger not found.",
            }

        a, b = self._normalize_pair(symbol_a, symbol_b)
        pairs = []
        for pair in ledger.get("pairs", []):
            left, right = self._normalize_pair(pair["symbol_a"], pair["symbol_b"])
            if (left, right) != (a, b):
                pairs.append(pair)
        ledger["pairs"] = pairs
        self.save_ledger(ledger)
        return ledger

    def _apply_env_overrides(self, ledger: Dict[str, Any]) -> Dict[str, Any]:
        add_pairs = [
            item.strip()
            for item in os.getenv("PAIR_ADD", "").split(",")
            if item.strip()
        ]
        remove_pairs = [
            item.strip()
            for item in os.getenv("PAIR_REMOVE", "").split(",")
            if item.strip()
        ]

        for spec in add_pairs:
            if "/" in spec:
                left, right = spec.split("/", 1)
                ledger = self.add_pair(left, right)

        for spec in remove_pairs:
            if "/" in spec:
                left, right = spec.split("/", 1)
                ledger = self.remove_pair(left, right)
        return ledger

    def _parse_created_at(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).astimezone(timezone.utc)
        except ValueError:
            return None

    def _should_refresh(self, ledger: Dict[str, Any]) -> bool:
        created_at = self._parse_created_at(ledger.get("created_at"))
        if not created_at:
            return True

        refresh_interval = os.getenv("PAIR_REFRESH_INTERVAL_HOURS")
        if refresh_interval:
            try:
                hours = float(refresh_interval)
                if hours > 0:
                    return (datetime.now(timezone.utc) - created_at) >= timedelta(hours=hours)
            except ValueError:
                pass

        refresh_hour = int(os.getenv("PAIR_REFRESH_UTC_HOUR", "21"))
        refresh_minute = int(os.getenv("PAIR_REFRESH_UTC_MINUTE", "0"))
        now = datetime.now(timezone.utc)
        today_refresh = datetime(
            year=now.year,
            month=now.month,
            day=now.day,
            hour=refresh_hour,
            minute=refresh_minute,
            tzinfo=timezone.utc,
        )
        if now >= today_refresh:
            return created_at < today_refresh
        yesterday_refresh = today_refresh - timedelta(days=1)
        return created_at < yesterday_refresh

    def load_or_build_pairs(self, force_rebuild: bool | None = None) -> Dict[str, Any]:
        if force_rebuild is None:
            force_rebuild = os.getenv("PAIR_REBUILD", "false").lower() == "true"

        ledger = None if force_rebuild else self.load_ledger()
        if ledger and not self._should_refresh(ledger):
            ledger = self._apply_env_overrides(ledger)
            return ledger
        if ledger:
            self._log("Pair ledger refresh triggered.")

        ledger = self.build_pairs()
        if ledger.get("status") == "success":
            self.save_ledger(ledger)
            ledger = self._apply_env_overrides(ledger)
        return ledger

    def _mock_ledger(self) -> Dict[str, Any]:
        """Return a minimal mock ledger for debug/testing purposes."""
        self._log("[DEBUG] Returning mock pair ledger (PAIR_DEBUG_SKIP=true)")
        return {
            "status": "success",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lookback_days": self.lookback_days,
            "universe_count": 0,
            "pairs": [
                {
                    "symbol_a": "AAPL",
                    "symbol_b": "MSFT",
                    "similarity": 0.92,
                    "method": "mock",
                    "window": self.lookback_days,
                },
                {
                    "symbol_a": "GOOGL",
                    "symbol_b": "META",
                    "similarity": 0.89,
                    "method": "mock",
                    "window": self.lookback_days,
                },
            ],
            "errors": {},
            "_debug": True,
        }

    def analyze(self) -> Dict[str, Any]:
        # Fast path: skip entirely in debug mode
        if os.getenv("PAIR_DEBUG_SKIP", "false").lower() == "true":
            return self._mock_ledger()
        return self.load_or_build_pairs()
