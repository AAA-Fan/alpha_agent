"""
Dynamic frontend dashboard for running and monitoring the financial agent system.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from orchestrator import run_full_analysis


load_dotenv()

st.set_page_config(
    page_title="Market Intel Deck",
    page_icon="MI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #14181f;
  --muted: #6b7480;
  --card: rgba(255,255,255,0.88);
  --accent: #0f766e;
  --accent-2: #ea580c;
  --bg-grad-1: #e7f6ff;
  --bg-grad-2: #fef6e8;
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1000px 500px at 10% -10%, var(--bg-grad-1), transparent 70%),
    radial-gradient(1000px 600px at 95% -20%, var(--bg-grad-2), transparent 70%),
    linear-gradient(120deg, #fbfdff 0%, #fff9f1 100%);
}

h1, h2, h3, h4, h5 {
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  letter-spacing: 0.2px;
}

p, li, label, .stMarkdown, .stText {
  font-family: "Space Grotesk", sans-serif;
}

code, pre {
  font-family: "IBM Plex Mono", monospace !important;
}

.card {
  border: 1px solid rgba(20,24,31,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  background: var(--card);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.accent {
  color: var(--accent);
}
</style>
""",
    unsafe_allow_html=True,
)


def _auto_refresh(seconds: int) -> None:
    if seconds <= 0:
        return
    components.html(
        f"""
<script>
setTimeout(function() {{
  window.parent.location.reload();
}}, {seconds * 1000});
</script>
""",
        height=0,
        width=0,
    )


def _storage_url() -> str:
    return os.getenv("STORAGE_URL", "sqlite:///data/agent_store.db")


def _default_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000")


def _sqlite_path_from_url(url: str) -> Path | None:
    if not url.startswith("sqlite:///"):
        return None
    return Path(url.replace("sqlite:///", "", 1))


def _read_table(sqlite_path: Path, table: str, limit: int, symbol: str = "") -> pd.DataFrame:
    if not sqlite_path.exists():
        return pd.DataFrame()
    where_clause = ""
    params: list[Any] = []
    if symbol:
        where_clause = " WHERE stock_symbol = ? "
        params.append(symbol.upper())
    params.append(limit)
    query = f"SELECT * FROM {table}{where_clause} ORDER BY id DESC LIMIT ?"
    try:
        with sqlite3.connect(sqlite_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def _normalize_api_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _api_get(base_url: str, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = _normalize_api_base_url(base_url) + path
    response = requests.get(url, params=params or {}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("API response must be a JSON object")
    return payload


def _api_post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = _normalize_api_base_url(base_url) + path
    response = requests.post(url, json=payload, timeout=3600)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("API response must be a JSON object")
    return body


def _table_from_api(base_url: str, endpoint_path: str, symbol: str, limit: int) -> pd.DataFrame:
    params: Dict[str, Any] = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    payload = _api_get(base_url, endpoint_path, params=params)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _show_forecast_gauge(probability_up: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_up * 100.0,
            title={"text": "Probability Up (%)"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, 45], "color": "#fee2e2"},
                    {"range": [45, 55], "color": "#fff7ed"},
                    {"range": [55, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _analysis_panel(use_api_backend: bool, api_base_url: str) -> None:
    st.markdown("## Live Analysis")
    st.markdown(
        "<div class='card'>Run the full 10-step analysis pipeline and inspect each agent output.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([2, 1])
    with left:
        symbol = st.text_input("Ticker Symbol", value="AAPL", max_chars=12).strip().upper()
    with right:
        save_report = st.checkbox("Save text report file", value=True)

    persist = st.checkbox("Persist results to storage", value=True)
    run_clicked = st.button("Run Full Analysis", type="primary", use_container_width=True)

    if run_clicked:
        progress_placeholder = st.empty()
        logs_placeholder = st.empty()
        logs: list[str] = []

        with st.spinner(f"Running full analysis for {symbol}..."):
            if use_api_backend:
                try:
                    progress_placeholder.progress(0.05, text="Submitting request to API backend")
                    payload = _api_post(
                        api_base_url,
                        "/analyze",
                        {
                            "symbol": symbol,
                            "persist": persist,
                            "save_report": save_report,
                            "verbose": False,
                        },
                    )
                    progress_logs = payload.get("progress_logs", [])
                    if isinstance(progress_logs, list):
                        for item in progress_logs:
                            if isinstance(item, dict):
                                logs.append(
                                    f"{item.get('time', '--')} "
                                    f"[{item.get('step', '?')}/{item.get('total', '?')}] "
                                    f"{item.get('message', '')}"
                                )
                    progress_placeholder.progress(1.0, text="API run completed")
                    result = payload
                except Exception as exc:
                    result = {"status": "error", "error": f"API request failed: {exc}"}
            else:
                def callback(step: int, total: int, message: str) -> None:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    logs.append(f"{timestamp} [{step}/{total}] {message}")
                    progress_placeholder.progress(step / total, text=f"{step}/{total} - {message}")
                    logs_placeholder.code("\n".join(logs[-12:]), language="text")

                result = run_full_analysis(
                    symbol,
                    verbose=False,
                    persist=persist,
                    save_report=save_report,
                    progress_callback=callback,
                )
        st.session_state["latest_result"] = result
        st.session_state["latest_logs"] = logs

    result = st.session_state.get("latest_result")
    if not result:
        st.info("Run an analysis to populate live results.")
        return

    if result.get("status") != "success":
        st.error(result.get("error", "Analysis failed."))
        return

    results = result.get("results", {})
    forecast = results.get("forecast", {}).get("forecast", {})
    risk = results.get("risk", {}).get("risk_plan", {})
    rec_text = results.get("recommendation", {}).get("recommendation", "")

    c1, c2, c3 = st.columns(3)
    c1.metric("Action", str(forecast.get("action", "n/a")).upper())
    c2.metric("Prob. Up", f"{float(forecast.get('probability_up', 0.0)):.2%}")
    c3.metric("Position Size", f"{float(risk.get('position_size_fraction', 0.0)):.2f}")

    gauge_col, _ = st.columns([1, 2])
    with gauge_col:
        _show_forecast_gauge(float(forecast.get("probability_up", 0.0)))

    with st.expander("Final Recommendation", expanded=True):
        st.markdown(rec_text or "No recommendation text generated.")
        st.caption(f"Report file: {result.get('output_file') or 'not saved'}")

    with st.expander("Historical Analysis"):
        st.markdown(results.get("historical", {}).get("analysis", "No data"))
    with st.expander("Indicator Analysis"):
        st.markdown(results.get("indicator", {}).get("analysis", "No data"))
    with st.expander("News Sentiment Analysis"):
        st.markdown(results.get("news", {}).get("analysis", "No data"))
    with st.expander("Pair Monitor Analysis"):
        st.write(results.get("pair_monitor", {}))
    with st.expander("Feature Engineering"):
        st.json(results.get("feature", {}))
    with st.expander("Regime Analysis"):
        st.json(results.get("regime", {}))
    with st.expander("Forecast Analysis"):
        st.json(results.get("forecast", {}))
    with st.expander("Risk Plan"):
        st.json(results.get("risk", {}))

    with st.expander("Pipeline Logs"):
        st.code("\n".join(st.session_state.get("latest_logs", [])), language="text")


def _monitor_panel(
    symbol_filter: str,
    limit_rows: int,
    auto_refresh: bool,
    refresh_seconds: int,
    use_api_backend: bool,
    api_base_url: str,
) -> None:
    st.markdown("## Monitoring")
    st.markdown(
        "<div class='card'>Track stored recommendations, model forecasts, pair signals, and realized outcomes.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if auto_refresh:
        _auto_refresh(refresh_seconds)
        st.caption(f"Auto refresh enabled: every {refresh_seconds}s")

    if use_api_backend:
        try:
            recommendations_df = _table_from_api(
                api_base_url, "/storage/recommendations", symbol_filter, limit_rows
            )
            predictions_df = _table_from_api(
                api_base_url, "/storage/predictions", symbol_filter, limit_rows
            )
            pair_signals_df = _table_from_api(
                api_base_url, "/storage/pair-signals", symbol_filter, limit_rows
            )
            outcomes_df = _table_from_api(
                api_base_url, "/storage/realized-outcomes", symbol_filter, limit_rows
            )
        except Exception as exc:
            st.error(f"Failed to load monitor data from API: {exc}")
            return
    else:
        storage_url = _storage_url()
        sqlite_path = _sqlite_path_from_url(storage_url)
        if sqlite_path is None:
            st.warning(
                "Monitoring table viewer currently supports SQLite storage URLs only. "
                f"Current STORAGE_URL: {storage_url}"
            )
            return
        if not sqlite_path.exists():
            st.info(f"Storage file does not exist yet: `{sqlite_path}`. Run at least one analysis first.")
            return

        recommendations_df = _read_table(sqlite_path, "recommendations", limit_rows, symbol_filter)
        predictions_df = _read_table(sqlite_path, "predictions", limit_rows, symbol_filter)
        pair_signals_df = _read_table(sqlite_path, "pair_signals", limit_rows, symbol_filter)
        outcomes_df = _read_table(sqlite_path, "realized_outcomes", limit_rows, symbol_filter)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recommendations", f"{len(recommendations_df)}")
    k2.metric("Predictions", f"{len(predictions_df)}")
    k3.metric("Pair Signals", f"{len(pair_signals_df)}")
    k4.metric("Outcomes", f"{len(outcomes_df)}")

    if not predictions_df.empty:
        predictions_df["created_at"] = pd.to_datetime(predictions_df["created_at"], errors="coerce")
        chart_df = predictions_df.dropna(subset=["created_at"]).sort_values("created_at")
        if not chart_df.empty:
            p_col, r_col = st.columns(2)
            with p_col:
                prob_fig = px.line(
                    chart_df,
                    x="created_at",
                    y="probability_up",
                    color="stock_symbol" if "stock_symbol" in chart_df.columns else None,
                    markers=True,
                    title="Probability-Up Timeline",
                )
                prob_fig.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=10))
                st.plotly_chart(prob_fig, use_container_width=True)
            with r_col:
                ret_fig = px.bar(
                    chart_df.tail(80),
                    x="created_at",
                    y="probability_up",
                    color="action" if "action" in chart_df.columns else None,
                    title="Probability-Up by Action",
                )
                ret_fig.update_layout(height=280, margin=dict(l=20, r=20, t=45, b=10))
                st.plotly_chart(ret_fig, use_container_width=True)

    if not pair_signals_df.empty and {"z_score", "confidence"}.issubset(pair_signals_df.columns):
        pair_fig = px.scatter(
            pair_signals_df,
            x="z_score",
            y="confidence",
            color="stock_symbol" if "stock_symbol" in pair_signals_df.columns else None,
            hover_data=["symbol_a", "symbol_b", "leading", "lagging"],
            title="Pair Signal Quality (z-score vs confidence)",
        )
        pair_fig.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=10))
        st.plotly_chart(pair_fig, use_container_width=True)

    st.markdown("### Recommendations")
    st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
    st.markdown("### Predictions")
    st.dataframe(predictions_df, use_container_width=True, hide_index=True)
    st.markdown("### Pair Signals")
    st.dataframe(pair_signals_df, use_container_width=True, hide_index=True)
    st.markdown("### Realized Outcomes")
    st.dataframe(outcomes_df, use_container_width=True, hide_index=True)


def _config_panel(use_api_backend: bool, api_base_url: str) -> None:
    st.markdown("## Runtime Configuration")
    model_path = Path(os.getenv("FORECAST_MODEL_PATH", "data/forecast_model.json"))
    model_exists = model_path.exists()
    storage_url = _storage_url()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Model Status")
        st.write(f"Forecast model path: `{model_path}`")
        st.write(f"Model config present: `{model_exists}`")
        if model_exists:
            try:
                payload = json.loads(model_path.read_text(encoding="utf-8"))
                st.json(
                    {
                        "trained_at": payload.get("trained_at"),
                        "row_count": payload.get("row_count"),
                        "in_sample_accuracy": payload.get("in_sample_accuracy"),
                        "tickers": payload.get("tickers"),
                    }
                )
            except Exception as exc:
                st.warning(f"Model file exists but failed to parse: {exc}")
        else:
            st.info("No model config found. ForecastAgent will use heuristic fallback.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Environment Snapshot")
        st.write(f"Storage URL: `{storage_url}`")
        st.write(f"Use API backend: `{use_api_backend}`")
        st.write(f"API base URL: `{api_base_url}`")
        st.write(f"OpenAI model: `{os.getenv('OPENAI_MODEL', 'gpt-4o')}`")
        st.write(f"Verbose mode: `{os.getenv('VERBOSE', 'false')}`")
        st.write(f"Storage enabled: `{os.getenv('STORAGE_ENABLED', 'true')}`")
        st.write(f"Pair monitor interval: `{os.getenv('PAIR_MONITOR_INTERVAL', 'daily')}`")
        st.write(f"Forecast horizon: `{os.getenv('FORECAST_HORIZON_DAYS', '5')}`")
        if use_api_backend:
            try:
                health = _api_get(api_base_url, "/health")
                st.success(f"API health: {health.get('status')} ({health.get('time_utc')})")
            except Exception as exc:
                st.error(f"API health check failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Useful Commands")
    st.code(
        "python main.py\n"
        "uvicorn api:app --host 0.0.0.0 --port 8000\n"
        "streamlit run dashboard.py\n"
        "python pipelines/train_forecast_model.py\n",
        language="bash",
    )


def main() -> None:
    st.markdown("# Market Intel Deck")
    st.caption("Dynamic control center for multi-agent analysis, forecasting, and monitoring.")

    with st.sidebar:
        st.markdown("## Monitor Controls")
        symbol_filter = st.text_input("Filter by symbol", value="").strip().upper()
        limit_rows = st.slider("Rows per table", min_value=20, max_value=500, value=120, step=10)
        auto_refresh = st.toggle("Auto refresh monitor", value=False)
        refresh_seconds = st.slider("Refresh interval (s)", min_value=5, max_value=120, value=20, step=5)
        st.markdown("---")
        st.markdown("## Backend Mode")
        use_api_backend = st.toggle(
            "Use API backend",
            value=True,
        )
        api_base_url = st.text_input("API base URL", value=_default_api_base_url()).strip()
        st.markdown("---")
        st.markdown("## App")
        st.write("Run full pipeline and inspect all artifacts in one place.")

    tab_analyze, tab_monitor, tab_config = st.tabs(["Live Analysis", "Monitoring", "Configuration"])
    with tab_analyze:
        _analysis_panel(use_api_backend=use_api_backend, api_base_url=api_base_url)
    with tab_monitor:
        _monitor_panel(
            symbol_filter,
            limit_rows,
            auto_refresh,
            refresh_seconds,
            use_api_backend=use_api_backend,
            api_base_url=api_base_url,
        )
    with tab_config:
        _config_panel(use_api_backend=use_api_backend, api_base_url=api_base_url)


if __name__ == "__main__":
    main()
