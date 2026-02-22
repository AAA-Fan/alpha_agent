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
            return

        if self.engine == "postgres":
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

    def save_prediction(
        self,
        forecast_result: Dict[str, Any],
        stock_symbol: str,
        user_id: str | None = None,
    ) -> None:
        if self.conn is None:
            return
        forecast = forecast_result.get("forecast", {})
        if not forecast:
            return

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ci = forecast.get("confidence_interval", {})
        values = (
            user_id or self.user_id,
            stock_symbol,
            created_at,
            forecast.get("horizon_days"),
            forecast.get("model_source"),
            forecast.get("action"),
            forecast.get("probability_up"),
            forecast.get("predicted_return"),
            ci.get("lower"),
            ci.get("upper"),
            forecast.get("confidence"),
            json.dumps(forecast_result),
        )
        self._execute(
            """
            INSERT INTO predictions (
                user_id, stock_symbol, created_at, horizon_days, model_source, action,
                probability_up, predicted_return, ci_lower, ci_upper, confidence, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            if self.engine == "sqlite"
            else """
            INSERT INTO predictions (
                user_id, stock_symbol, created_at, horizon_days, model_source, action,
                probability_up, predicted_return, ci_lower, ci_upper, confidence, raw_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
