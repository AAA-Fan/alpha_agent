#!/usr/bin/env python3
"""
Train a lightweight linear score model for ForecastAgent.

This script does not require external training datasets. It builds a training matrix
from historical Alpha Vantage OHLCV data and saves a coefficient config JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.yfinance_cache import get_historical_data


FEATURE_COLUMNS = [
    "momentum_5",
    "momentum_20",
    "sma_20_ratio",
    "macd_hist",
    "volatility_20",
    "rsi_14_centered",
    "volume_zscore_20",
]


def build_feature_frame(data: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    frame = data.sort_index().copy()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    returns = close.pct_change()

    frame["momentum_5"] = close.pct_change(5)
    frame["momentum_20"] = close.pct_change(20)
    frame["sma_20_ratio"] = (close / close.rolling(20).mean()) - 1
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = macd - macd_signal
    frame["volatility_20"] = returns.rolling(20).std() * np.sqrt(252)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    frame["rsi_14_centered"] = (rsi_14 - 50.0) / 50.0

    volume_mean_20 = volume.rolling(20).mean()
    volume_std_20 = volume.rolling(20).std().replace(0, np.nan)
    frame["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20

    frame["future_return"] = close.shift(-horizon_days) / close - 1
    frame["target"] = np.where(frame["future_return"] > 0, 1.0, -1.0)

    cols = FEATURE_COLUMNS + ["future_return", "target"]
    frame = frame.dropna(subset=cols)
    return frame


def fit_ridge_score_model(
    rows: pd.DataFrame,
    ridge_lambda: float = 1e-2,
) -> Dict[str, object]:
    X = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = rows["target"].to_numpy(dtype=float)

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds == 0.0, 1.0, stds)
    Xs = (X - means) / stds

    X_design = np.column_stack([np.ones(len(Xs)), Xs])
    ident = np.eye(X_design.shape[1], dtype=float)
    ident[0, 0] = 0.0  # do not regularize intercept
    w = np.linalg.solve(X_design.T @ X_design + ridge_lambda * ident, X_design.T @ y)

    intercept = float(w[0])
    coefs = w[1:]
    score = intercept + Xs @ coefs
    prob = 1.0 / (1.0 + np.exp(-score))
    pred = np.where(prob >= 0.5, 1.0, -1.0)
    acc = float((pred == y).mean())

    config = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "intercept": intercept,
        "coefficients": {name: float(value) for name, value in zip(FEATURE_COLUMNS, coefs)},
        "feature_means": {name: float(value) for name, value in zip(FEATURE_COLUMNS, means)},
        "feature_stds": {name: float(value) for name, value in zip(FEATURE_COLUMNS, stds)},
        "in_sample_accuracy": acc,
        "row_count": int(len(rows)),
    }
    return config


def main() -> None:
    load_dotenv()
    tickers_env = os.getenv("TRAIN_TICKERS", "")
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
    if not tickers:
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

    lookback_days = int(os.getenv("TRAIN_LOOKBACK_DAYS", "420"))
    horizon_days = int(os.getenv("TRAIN_HORIZON_DAYS", "5"))
    ridge_lambda = float(os.getenv("TRAIN_RIDGE_LAMBDA", "0.01"))
    output_path = os.getenv("FORECAST_MODEL_PATH", "data/forecast_model.json")

    all_rows: List[pd.DataFrame] = []
    for ticker in tickers:
        try:
            data = get_historical_data(ticker, interval="daily", days=lookback_days + horizon_days + 60)
            if data.empty:
                print(f"[warn] {ticker}: no data")
                continue
            frame = build_feature_frame(data, horizon_days=horizon_days)
            if frame.empty:
                print(f"[warn] {ticker}: insufficient feature rows")
                continue
            frame["ticker"] = ticker
            all_rows.append(frame)
            print(f"[ok] {ticker}: rows={len(frame)}")
        except Exception as exc:
            print(f"[warn] {ticker}: {exc}")

    if not all_rows:
        raise SystemExit("No training rows were generated. Check API key, rate limits, and tickers.")

    rows = pd.concat(all_rows, axis=0, ignore_index=True)
    config = fit_ridge_score_model(rows, ridge_lambda=ridge_lambda)
    config["target_horizon_days"] = horizon_days
    config["tickers"] = tickers
    config["trainer"] = "pipelines/train_forecast_model.py"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"Saved model config to {output_path} | rows={config['row_count']} "
        f"| in_sample_accuracy={config['in_sample_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
