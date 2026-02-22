"""
FastAPI backend for financial-agent analysis and monitoring.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import run_full_analysis


load_dotenv()

app = FastAPI(
    title="Financial Agent API",
    version="1.0.0",
    description="API for running multi-agent financial analysis and querying stored artifacts.",
)


def _allowed_origins() -> List[str]:
    raw = os.getenv("API_ALLOWED_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12, description="Ticker symbol, e.g. AAPL")
    persist: bool = Field(default=True, description="Persist outputs into configured storage")
    save_report: bool = Field(default=True, description="Write text report file to disk")
    verbose: bool = Field(default=False, description="Enable verbose agent logging")


class AnalyzeResponse(BaseModel):
    status: str
    stock_symbol: str | None = None
    timestamp: str | None = None
    output_file: str | None = None
    persisted: bool | None = None
    final_report: str | None = None
    results: Dict[str, Any] | None = None
    progress_logs: List[Dict[str, Any]] | None = None
    error: str | None = None


def _storage_url() -> str:
    return os.getenv("STORAGE_URL", "sqlite:///data/agent_store.db")


def _sqlite_path() -> Path:
    url = _storage_url()
    if not url.startswith("sqlite:///"):
        raise HTTPException(
            status_code=501,
            detail=(
                "Read endpoints currently support SQLite storage only. "
                f"Current STORAGE_URL: {url}"
            ),
        )
    return Path(url.replace("sqlite:///", "", 1))


def _query_table(table: str, symbol: str | None, limit: int) -> List[Dict[str, Any]]:
    path = _sqlite_path()
    if not path.exists():
        return []

    where_clause = ""
    params: list[Any] = []
    if symbol:
        where_clause = " WHERE stock_symbol = ? "
        params.append(symbol.upper().strip())
    params.append(limit)
    query = f"SELECT * FROM {table}{where_clause} ORDER BY id DESC LIMIT ?"

    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Storage query failed: {exc}") from exc


def _query_count(table: str, symbol: str | None) -> int:
    path = _sqlite_path()
    if not path.exists():
        return 0

    where_clause = ""
    params: list[Any] = []
    if symbol:
        where_clause = " WHERE stock_symbol = ? "
        params.append(symbol.upper().strip())
    query = f"SELECT COUNT(*) FROM {table}{where_clause}"
    try:
        with sqlite3.connect(path) as conn:
            count = conn.execute(query, params).fetchone()[0]
            return int(count)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Storage count query failed: {exc}") from exc


def _table_response(table: str, symbol: str | None, limit: int) -> Dict[str, Any]:
    rows = _query_table(table, symbol=symbol, limit=limit)
    return {
        "table": table,
        "symbol": symbol.upper().strip() if symbol else None,
        "count": len(rows),
        "rows": rows,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "storage_url": _storage_url(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    logs: List[Dict[str, Any]] = []

    def _callback(step: int, total: int, message: str) -> None:
        logs.append(
            {
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "step": step,
                "total": total,
                "message": message,
            }
        )

    result = run_full_analysis(
        payload.symbol,
        verbose=payload.verbose,
        persist=payload.persist,
        save_report=payload.save_report,
        progress_callback=_callback,
    )
    if result.get("status") != "success":
        return AnalyzeResponse(
            status="error",
            error=result.get("error", "Unknown analysis error"),
            progress_logs=logs,
        )

    return AnalyzeResponse(
        status="success",
        stock_symbol=result.get("stock_symbol"),
        timestamp=result.get("timestamp"),
        output_file=result.get("output_file"),
        persisted=result.get("persisted"),
        final_report=result.get("final_report"),
        results=result.get("results"),
        progress_logs=logs,
    )


@app.get("/storage/status")
def storage_status() -> Dict[str, Any]:
    url = _storage_url()
    exists = False
    path = None
    if url.startswith("sqlite:///"):
        path_obj = Path(url.replace("sqlite:///", "", 1))
        path = str(path_obj)
        exists = path_obj.exists()

    return {
        "storage_url": url,
        "sqlite_path": path,
        "sqlite_exists": exists,
    }


@app.get("/storage/summary")
def storage_summary(
    symbol: str | None = Query(default=None, description="Optional ticker filter"),
) -> Dict[str, Any]:
    return {
        "symbol": symbol.upper().strip() if symbol else None,
        "recommendations": _query_count("recommendations", symbol),
        "predictions": _query_count("predictions", symbol),
        "pair_signals": _query_count("pair_signals", symbol),
        "realized_outcomes": _query_count("realized_outcomes", symbol),
    }


@app.get("/storage/recommendations")
def list_recommendations(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> Dict[str, Any]:
    return _table_response("recommendations", symbol=symbol, limit=limit)


@app.get("/storage/predictions")
def list_predictions(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> Dict[str, Any]:
    return _table_response("predictions", symbol=symbol, limit=limit)


@app.get("/storage/pair-signals")
def list_pair_signals(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> Dict[str, Any]:
    return _table_response("pair_signals", symbol=symbol, limit=limit)


@app.get("/storage/realized-outcomes")
def list_realized_outcomes(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> Dict[str, Any]:
    return _table_response("realized_outcomes", symbol=symbol, limit=limit)

