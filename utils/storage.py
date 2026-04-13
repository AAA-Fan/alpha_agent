"""Lightweight persistence layer for recommendations, predictions, and pair signals."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


class Storage:
    """Persist recommendations, model predictions, and pair signals to SQLite or Postgres."""

    def __init__(self, url: str | None = None, user_id: str | None = None) -> None:
        self.url = url or os.getenv("STORAGE_URL", "sqlite:///data/agent_store.db")
        self.user_id = user_id or os.getenv("STORAGE_USER_ID") or os.getenv("USER_ID")
        self.engine = None
        self.conn = None
        self._connect()
        self._init_schema()

    def _connect(self) -> None:
        if self.url.startswith("sqlite:///"):
            path = self.url.replace("sqlite:///", "", 1)
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            self.engine = "sqlite"
            self.conn = sqlite3.connect(path)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            return

        if self.url.startswith("postgres://") or self.url.startswith("postgresql://"):
            try:
                import psycopg2
            except ImportError as exc:
                raise RuntimeError(
                    "Postgres support requires psycopg2-binary. Install it or use SQLite."
                ) from exc
            self.engine = "postgres"
            self.conn = psycopg2.connect(self.url)
            return

        raise ValueError(f"Unsupported STORAGE_URL: {self.url}")

    def _execute(self, statement: str, params: Iterable[Any] | None = None) -> None:
        if self.conn is None:
            return
        cursor = self.conn.cursor()
        cursor.execute(statement, params or [])
        self.conn.commit()
        cursor.close()

    def _init_schema(self) -> None:
        if self.engine == "sqlite":
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    stock_symbol TEXT NOT NULL,
                    created_at TEXT,
                    status TEXT,
                    recommendation TEXT,
                    report_text TEXT,
                    historical_status TEXT,
                    indicator_status TEXT,
                    news_status TEXT,
                    pair_monitor_status TEXT,
                    raw_json TEXT
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS pair_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    stock_symbol TEXT,
                    created_at TEXT,
                    symbol_a TEXT,
                    symbol_b TEXT,
                    leading TEXT,
                    lagging TEXT,
                    divergence REAL,
                    z_score REAL,
                    window INTEGER,
                    similarity REAL,
                    confidence REAL,
                    momentum_a REAL,
                    momentum_b REAL,
                    interval TEXT,
                    raw_json TEXT
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    stock_symbol TEXT NOT NULL,
                    created_at TEXT,
                    horizon_days INTEGER,
                    model_source TEXT,
                    action TEXT,
                    probability_up REAL,
                    predicted_return REAL,
                    ci_lower REAL,
                    ci_upper REAL,
                    confidence REAL,
                    regime_state TEXT,
                    raw_json TEXT
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS realized_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    stock_symbol TEXT NOT NULL,
                    created_at TEXT,
                    predicted_at TEXT,
                    horizon_days INTEGER,
                    realized_return REAL,
                    benchmark_return REAL,
                    raw_json TEXT
                )
                """
            )

        elif self.engine == "postgres":
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    stock_symbol TEXT NOT NULL,
                    created_at TEXT,
                    status TEXT,
                    recommendation TEXT,
                    report_text TEXT,
                    historical_status TEXT,
                    indicator_status TEXT,
                    news_status TEXT,
                    pair_monitor_status TEXT,
                    raw_json TEXT
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS pair_signals (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    stock_symbol TEXT,
                    created_at TEXT,
                    symbol_a TEXT,
                    symbol_b TEXT,
                    leading TEXT,
                    lagging TEXT,
                    divergence REAL,
                    z_score REAL,
                    window INTEGER,
                    similarity REAL,
                    confidence REAL,
                    momentum_a REAL,
                    momentum_b REAL,
                    interval TEXT,
                    raw_json TEXT
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    stock_symbol TEXT NOT NULL,
                    created_at TEXT,
                    horizon_days INTEGER,
                    model_source TEXT,
                    action TEXT,
                    probability_up REAL,
                    predicted_return REAL,
                    ci_lower REAL,
                    ci_upper REAL,
                    confidence REAL,
                    regime_state TEXT,
                    raw_json TEXT
                )
                """
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS realized_outcomes (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    stock_symbol TEXT NOT NULL,
                    created_at TEXT,
                    predicted_at TEXT,
                    horizon_days INTEGER,
                    realized_return REAL,
                    benchmark_return REAL,
                    raw_json TEXT
                )
                """
            )

        # ── Schema migration: add regime_state column to existing databases ──
        self._migrate_add_regime_state()

    def _migrate_add_regime_state(self) -> None:
        """Add regime_state column to predictions table if it does not exist yet."""
        if self.conn is None:
            return
        try:
            if self.engine == "sqlite":
                # PRAGMA table_info returns column metadata; check if regime_state exists
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA table_info(predictions)")
                columns = [row[1] for row in cursor.fetchall()]
                cursor.close()
                if "regime_state" not in columns:
                    self._execute("ALTER TABLE predictions ADD COLUMN regime_state TEXT")
            else:
                # Postgres: add column if not exists
                self._execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS regime_state TEXT
                    """
                )
        except Exception:
            pass  # Column already exists or table not yet created; safe to ignore

    def save_recommendation(
        self,
        recommendation: Dict[str, Any],
        stock_symbol: str,
        report_text: str | None = None,
        user_id: str | None = None,
    ) -> None:
        if self.conn is None:
            return
        created_at = recommendation.get("timestamp") or datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        payload = json.dumps(recommendation)
        values = (
            user_id or self.user_id,
            stock_symbol,
            created_at,
            recommendation.get("status"),
            recommendation.get("recommendation"),
            report_text,
            recommendation.get("historical_status"),
            recommendation.get("indicator_status"),
            recommendation.get("news_status"),
            recommendation.get("pair_monitor_status"),
            payload,
        )
        self._execute(
            """
            INSERT INTO recommendations (
                user_id, stock_symbol, created_at, status, recommendation, report_text,
                historical_status, indicator_status, news_status, pair_monitor_status, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            if self.engine == "sqlite"
            else """
            INSERT INTO recommendations (
                user_id, stock_symbol, created_at, status, recommendation, report_text,
                historical_status, indicator_status, news_status, pair_monitor_status, raw_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            values,
        )

    def save_pair_signals(
        self,
        pair_monitor_result: Dict[str, Any],
        stock_symbol: str,
        interval: str,
        user_id: str | None = None,
    ) -> None:
        if self.conn is None:
            return
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for signal in pair_monitor_result.get("signals", []):
            values = (
                user_id or self.user_id,
                stock_symbol,
                created_at,
                signal.get("symbol_a"),
                signal.get("symbol_b"),
                signal.get("leading"),
                signal.get("lagging"),
                signal.get("divergence"),
                signal.get("z_score"),
                signal.get("window"),
                signal.get("similarity"),
                signal.get("confidence"),
                signal.get("momentum_a"),
                signal.get("momentum_b"),
                interval,
                json.dumps(signal),
            )
            self._execute(
                """
                INSERT INTO pair_signals (
                    user_id, stock_symbol, created_at, symbol_a, symbol_b, leading, lagging,
                    divergence, z_score, window, similarity, confidence, momentum_a, momentum_b,
                    interval, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                if self.engine == "sqlite"
                else """
                INSERT INTO pair_signals (
                    user_id, stock_symbol, created_at, symbol_a, symbol_b, leading, lagging,
                    divergence, z_score, window, similarity, confidence, momentum_a, momentum_b,
                    interval, raw_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                values,
            )

    def _is_duplicate_prediction(
        self,
        stock_symbol: str,
        action: str | None,
        probability_up: float | None,
    ) -> bool:
        """Check if a similar prediction already exists for the same symbol on the same day (UTC).

        A prediction is considered duplicate if the same symbol + same action + same
        probability_up (rounded to 4 decimals) was already stored today.
        """
        if self.conn is None:
            return False

        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rounded_prob = round(probability_up, 4) if probability_up is not None else None

        if self.engine == "sqlite":
            sql = """
                SELECT COUNT(*) AS cnt
                FROM predictions
                WHERE stock_symbol = ?
                  AND action = ?
                  AND ROUND(probability_up, 4) = ?
                  AND date(created_at) = ?
            """
            params = [stock_symbol, action, rounded_prob, today_utc]
        else:
            sql = """
                SELECT COUNT(*) AS cnt
                FROM predictions
                WHERE stock_symbol = %s
                  AND action = %s
                  AND ROUND(probability_up::numeric, 4) = %s
                  AND created_at::date = %s::date
            """
            params = [stock_symbol, action, rounded_prob, today_utc]

        rows = self._fetchall(sql, params)
        if rows and rows[0].get("cnt", 0) > 0:
            return True
        return False

    def save_prediction(
        self,
        forecast_result: Dict[str, Any],
        stock_symbol: str,
        regime_result: Dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        if self.conn is None:
            return
        forecast = forecast_result.get("forecast", {})
        if not forecast:
            return

        # ── G1: Deduplication — skip if same symbol+action+probability already stored today ──
        action = forecast.get("action")
        probability_up = forecast.get("probability_up")
        if self._is_duplicate_prediction(stock_symbol, action, probability_up):
            return

        # ── G2: Extract regime state from regime_result ──
        regime_state = None
        if regime_result and regime_result.get("status") == "success":
            regime = regime_result.get("regime", {})
            regime_state = regime.get("state") or regime.get("regime_label")

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        values = (
            user_id or self.user_id,
            stock_symbol,
            created_at,
            forecast.get("horizon_days"),
            forecast.get("model_source"),
            action,
            forecast.get("probability_up"),
            None,  # predicted_return (removed)
            None,  # ci_lower (removed)
            None,  # ci_upper (removed)
            None,  # confidence (removed)
            regime_state,
            json.dumps(forecast_result),
        )
        self._execute(
            """
            INSERT INTO predictions (
                user_id, stock_symbol, created_at, horizon_days, model_source, action,
                probability_up, predicted_return, ci_lower, ci_upper, confidence,
                regime_state, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            if self.engine == "sqlite"
            else """
            INSERT INTO predictions (
                user_id, stock_symbol, created_at, horizon_days, model_source, action,
                probability_up, predicted_return, ci_lower, ci_upper, confidence,
                regime_state, raw_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            values,
        )

    def save_realized_outcome(
        self,
        stock_symbol: str,
        predicted_at: str,
        horizon_days: int,
        realized_return: float,
        benchmark_return: float | None = None,
        user_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if self.conn is None:
            return
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        payload = metadata or {}
        values = (
            user_id or self.user_id,
            stock_symbol,
            created_at,
            predicted_at,
            horizon_days,
            realized_return,
            benchmark_return,
            json.dumps(payload),
        )
        self._execute(
            """
            INSERT INTO realized_outcomes (
                user_id, stock_symbol, created_at, predicted_at, horizon_days,
                realized_return, benchmark_return, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            if self.engine == "sqlite"
            else """
            INSERT INTO realized_outcomes (
                user_id, stock_symbol, created_at, predicted_at, horizon_days,
                realized_return, benchmark_return, raw_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            values,
        )

    # ── Query methods for Memory / Outcome Tracking ──────────────────────

    def _fetchall(self, statement: str, params: Iterable[Any] | None = None) -> list:
        """Execute a SELECT and return all rows as a list of dicts."""
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        cursor.execute(statement, params or [])
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return rows

    def get_pending_predictions(self) -> list:
        """
        Return predictions whose horizon has matured but have no matching
        realized_outcome yet.
        """
        placeholder = "?" if self.engine == "sqlite" else "%s"
        # Use date arithmetic to find matured predictions
        if self.engine == "sqlite":
            sql = """
                SELECT p.*
                FROM predictions p
                LEFT JOIN realized_outcomes ro
                    ON p.stock_symbol = ro.stock_symbol
                    AND p.created_at = ro.predicted_at
                    AND p.horizon_days = ro.horizon_days
                WHERE ro.id IS NULL
                  AND date(p.created_at, '+' || p.horizon_days || ' days') <= date('now')
                ORDER BY p.created_at ASC
            """
        else:
            sql = """
                SELECT p.*
                FROM predictions p
                LEFT JOIN realized_outcomes ro
                    ON p.stock_symbol = ro.stock_symbol
                    AND p.created_at = ro.predicted_at
                    AND p.horizon_days = ro.horizon_days
                WHERE ro.id IS NULL
                  AND (p.created_at::date + (p.horizon_days || ' days')::interval) <= CURRENT_DATE
                ORDER BY p.created_at ASC
            """
        return self._fetchall(sql)

    def get_tracked_predictions(self, stock_symbol: str) -> list:
        """
        Return prediction-outcome pairs for a given stock, ordered by date.
        Each row contains both the original prediction fields and the realized return.
        """
        placeholder = "?" if self.engine == "sqlite" else "%s"
        sql = f"""
            SELECT
                p.id,
                p.stock_symbol,
                p.created_at AS predicted_at,
                p.horizon_days,
                p.model_source,
                p.action,
                p.probability_up,
                p.predicted_return,
                p.confidence,
                p.regime_state AS regime,
                ro.realized_return,
                ro.benchmark_return,
                ro.raw_json AS outcome_raw_json
            FROM predictions p
            INNER JOIN realized_outcomes ro
                ON p.stock_symbol = ro.stock_symbol
                AND p.created_at = ro.predicted_at
                AND p.horizon_days = ro.horizon_days
            WHERE p.stock_symbol = {placeholder}
            ORDER BY p.created_at ASC
        """
        return self._fetchall(sql, [stock_symbol.upper().strip()])

    def get_performance_stats(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Return aggregated performance statistics for a stock's predictions.
        Uses action/probability_up to determine predicted direction.
        """
        tracked = self.get_tracked_predictions(stock_symbol)
        if not tracked:
            return {"prediction_count": 0}

        total = 0
        correct = 0
        sum_realized = 0.0

        for row in tracked:
            real_ret = row.get("realized_return")
            if real_ret is None:
                continue
            # Determine predicted direction from action or probability_up
            action = (row.get("action") or "hold").lower()
            prob_up = row.get("probability_up")
            if action == "buy":
                pred_dir = 1
            elif action == "sell":
                pred_dir = -1
            elif prob_up is not None:
                pred_dir = 1 if prob_up >= 0.5 else -1
            else:
                continue  # Cannot determine direction
            total += 1
            sum_realized += real_ret
            real_dir = 1 if real_ret >= 0 else -1
            if pred_dir == real_dir:
                correct += 1

        if total == 0:
            return {"prediction_count": 0}

        return {
            "prediction_count": total,
            "directional_accuracy": round(correct / total, 4),
            "avg_realized_return": round(sum_realized / total, 6),
        }

    def get_recent_predictions(self, stock_symbol: str, limit: int = 10) -> list:
        """Return the most recent predictions for a stock (tracked or not)."""
        placeholder = "?" if self.engine == "sqlite" else "%s"
        sql = f"""
            SELECT *
            FROM predictions
            WHERE stock_symbol = {placeholder}
            ORDER BY created_at DESC
            LIMIT {int(limit)}
        """
        return self._fetchall(sql, [stock_symbol.upper().strip()])

    # ── Delete / Clear methods ───────────────────────────────────────────

    def clear_memory(self, stock_symbol: str | None = None) -> dict[str, int]:
        """
        Clear prediction and realized_outcome records.

        Args:
            stock_symbol: If provided, only clear records for this symbol.
                          If None, clear ALL records.

        Returns:
            A dict with counts of deleted rows per table.
        """
        placeholder = "?" if self.engine == "sqlite" else "%s"
        counts: dict[str, int] = {}

        for table_name in ("predictions", "realized_outcomes"):
            if stock_symbol:
                sql = f"DELETE FROM {table_name} WHERE stock_symbol = {placeholder}"
                params = [stock_symbol.upper().strip()]
            else:
                sql = f"DELETE FROM {table_name}"
                params = []

            if self.conn is None:
                counts[table_name] = 0
                continue

            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            counts[table_name] = cursor.rowcount
            self.conn.commit()
            cursor.close()

        return counts
