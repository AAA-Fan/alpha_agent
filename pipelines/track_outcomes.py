#!/usr/bin/env python3
"""
Outcome Tracker
Checks for matured predictions in the database, fetches actual prices,
computes realized returns, and writes them to the realized_outcomes table.

Can be run standalone:  python -m pipelines.track_outcomes
Or invoked programmatically before each analysis run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv

from utils.storage import Storage
from utils.yfinance_cache import get_historical_data


def _parse_datetime(dt_str: str) -> datetime:
    """Parse a datetime string from the database."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {dt_str}")


def _get_close_price_on_date(data, target_date: datetime) -> float | None:
    """
    Find the closing price on or nearest to the target date.
    Looks within a 5-day window to handle weekends/holidays.
    """
    if data.empty:
        return None

    target = target_date.date() if hasattr(target_date, "date") else target_date
    # Search within a window of +/- 5 days
    for offset in range(0, 6):
        for direction in [0, -1, 1]:
            check_date = target + timedelta(days=offset * direction) if offset > 0 else target
            matches = data.index[data.index.date == check_date]
            if len(matches) > 0:
                return float(data.loc[matches[0], "Close"])
    return None


def track_outcomes(
    storage: Storage | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Check for matured predictions and record realized outcomes.

    Returns a summary dict with counts and details.
    """
    if storage is None:
        storage = Storage()

    pending = storage.get_pending_predictions()
    if not pending:
        if verbose:
            print("[track_outcomes] No matured predictions to track.")
        return {
            "status": "success",
            "tracked": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

    tracked = 0
    skipped = 0
    errors = 0
    details: List[Dict[str, Any]] = []

    # Group by symbol to minimize API calls
    symbol_groups: Dict[str, List[Dict[str, Any]]] = {}
    for pred in pending:
        sym = pred["stock_symbol"]
        symbol_groups.setdefault(sym, []).append(pred)

    for symbol, preds in symbol_groups.items():
        try:
            # Fetch enough historical data to cover all predictions
            data = get_historical_data(symbol, interval="daily", days=120)
            if data.empty:
                if verbose:
                    print(f"[track_outcomes] {symbol}: no price data available, skipping.")
                skipped += len(preds)
                continue
        except Exception as exc:
            if verbose:
                print(f"[track_outcomes] {symbol}: failed to fetch data: {exc}")
            errors += len(preds)
            continue

        # Also try to fetch SPY for benchmark
        spy_data = None
        try:
            spy_data = get_historical_data("SPY", interval="daily", days=120)
        except Exception:
            pass

        for pred in preds:
            try:
                predicted_at = _parse_datetime(pred["created_at"])
                horizon = pred["horizon_days"]
                maturity_date = predicted_at + timedelta(days=horizon)

                price_start = _get_close_price_on_date(data, predicted_at)
                price_end = _get_close_price_on_date(data, maturity_date)

                if price_start is None or price_end is None or price_start == 0:
                    if verbose:
                        print(
                            f"[track_outcomes] {symbol} pred#{pred['id']}: "
                            f"missing price data (start={price_start}, end={price_end}), skipping."
                        )
                    skipped += 1
                    continue

                realized_return = (price_end / price_start) - 1.0

                # Benchmark return (SPY)
                benchmark_return = None
                if spy_data is not None and not spy_data.empty:
                    spy_start = _get_close_price_on_date(spy_data, predicted_at)
                    spy_end = _get_close_price_on_date(spy_data, maturity_date)
                    if spy_start and spy_end and spy_start != 0:
                        benchmark_return = (spy_end / spy_start) - 1.0

                metadata = {
                    "prediction_id": pred["id"],
                    "action": pred.get("action"),
                    "probability_up": pred.get("probability_up"),
                    "price_start": price_start,
                    "price_end": price_end,
                }

                storage.save_realized_outcome(
                    stock_symbol=symbol,
                    predicted_at=pred["created_at"],
                    horizon_days=horizon,
                    realized_return=realized_return,
                    benchmark_return=benchmark_return,
                    metadata=metadata,
                )

                tracked += 1
                detail = {
                    "symbol": symbol,
                    "prediction_id": pred["id"],
                    "predicted_at": pred["created_at"],
                    "action": pred.get("action"),
                    "probability_up": pred.get("probability_up"),
                    "realized_return": round(realized_return, 6),
                    "benchmark_return": round(benchmark_return, 6) if benchmark_return is not None else None,
                }
                details.append(detail)

                if verbose:
                    action = pred.get("action", "hold").lower()
                    prob_up = pred.get("probability_up")
                    pred_dir = 1 if action == "buy" else (-1 if action == "sell" else (1 if (prob_up or 0.5) >= 0.5 else -1))
                    real_dir = 1 if realized_return >= 0 else -1
                    direction_correct = pred_dir == real_dir
                    mark = "✓" if direction_correct else "✗"
                    print(
                        f"[track_outcomes] {mark} {symbol} pred#{pred['id']}: "
                        f"action={action}, prob_up={prob_up}, "
                        f"realized={realized_return:+.4f}"
                    )

            except Exception as exc:
                if verbose:
                    print(f"[track_outcomes] {symbol} pred#{pred['id']}: error: {exc}")
                errors += 1

    return {
        "status": "success",
        "tracked": tracked,
        "skipped": skipped,
        "errors": errors,
        "details": details,
    }


def main() -> None:
    """Standalone entry point for tracking outcomes."""
    load_dotenv()
    verbose = os.getenv("TRACK_VERBOSE", "true").lower() == "true"
    print("=" * 60)
    print("Outcome Tracker — Checking matured predictions")
    print("=" * 60)

    result = track_outcomes(verbose=verbose)

    print(f"\nResults: tracked={result['tracked']}, skipped={result['skipped']}, errors={result['errors']}")
    if result["details"]:
        print("\nDetails:")
        for d in result["details"]:
            print(f"  {d['symbol']} ({d['predicted_at']}): action={d['action']}, prob_up={d.get('probability_up')}, realized={d['realized_return']}")


if __name__ == "__main__":
    main()
