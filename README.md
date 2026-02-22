# Multi-Agent Financial Advisory System

A sophisticated financial advisory system built with LangChain that uses multiple AI agents to analyze stocks and provide investment recommendations.

Full HTML documentation is available at `APP_DOCUMENTATION.html`.

## System Architecture

The system consists of specialized agents:

1. **Historical Data Analysis Agent**: Fetches and analyzes historical stock data (1 week) including price trends, volatility, and volume patterns.

2. **Indicator Analysis Agent**: Computes RSI signals across multiple timeframes and produces momentum-based buy/sell/hold signals.

3. **News Sentiment Analysis Agent**: Fetches the latest news from the internet and performs sentiment analysis to identify positive and negative factors affecting the stock.

4. **Pair Ledger Agent**: Builds and persists momentum-similar stock pairs from a defined universe (default: top 100 S&P 500 by weight).

5. **Pair Monitor Agent**: Monitors those pairs for short-term momentum divergence and flags leading/lagging moves.

6. **Feature Engineering Agent**: Builds quantitative features (momentum, volatility, RSI, MACD, volume pressure, drawdown).

7. **Regime Agent**: Classifies trend/volatility regime to contextualize signals (e.g., trending_up, choppy_high_vol).

8. **Forecast Agent**: Produces probabilistic forecasts with confidence intervals. Uses trained config if present, otherwise heuristic fallback.

9. **Risk Agent**: Converts forecast + regime into position sizing, stop/take-profit, and risk flags.

10. **Backtest Agent**: Runs a walk-forward strategy snapshot for sanity-checking recent behavior.

11. **Supervisor Agent**: Coordinates all agents, synthesizes qualitative + quantitative outputs, and provides final investment recommendations.

## Features

- **Multi-Agent Architecture**: Specialized agents working in parallel
- **Historical Analysis**: Technical analysis of stock price movements and trends
- **News Sentiment**: Real-time news fetching and sentiment analysis
- **Intelligent Recommendations**: AI-powered synthesis and investment advice
- **Pair Momentum Monitoring**: Detects divergence across momentum-similar stock pairs
- **Probabilistic Forecasting**: Model-ready forecast layer with confidence intervals
- **Risk-Aware Guidance**: Position sizing and guardrails based on volatility and confidence
- **Backtest Snapshot**: Lightweight strategy check on recent history
- **Comprehensive Reports**: Detailed reports saved to files

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Alpha Vantage API key (for historical price/indicator data)
- (Optional) Tavily API key for enhanced news search

### Setup

1. Clone or navigate to the project directory:
```bash
cd financailagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

For enhanced news search, you can optionally add:
```
TAVILY_API_KEY=your_tavily_api_key_here
```

For market data, add Alpha Vantage API key:
```
ALPHAVANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

## Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

Enter a stock symbol when prompted (e.g., AAPL, GOOGL, MSFT, TSLA).

### Dynamic Frontend Dashboard

Launch the real-time frontend:

```bash
streamlit run dashboard.py
```

The dashboard includes:
- **Live Analysis**: Run the full 10-step pipeline with live progress logs
- **Monitoring**: Track stored recommendations, predictions, pair signals, and realized outcomes
- **Configuration**: See current runtime/model/storage state

Use sidebar controls to filter by symbol and enable auto-refresh monitoring.

### FastAPI Backend

Run API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Core endpoints:
- `GET /health`
- `POST /analyze`
- `GET /storage/status`
- `GET /storage/summary`
- `GET /storage/recommendations`
- `GET /storage/predictions`
- `GET /storage/pair-signals`
- `GET /storage/realized-outcomes`

Example request:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","persist":true,"save_report":true,"verbose":false}'
```

Dashboard API mode:
- Toggle `Use API backend` in sidebar.
- Set API base URL (default: `http://localhost:8000`).
- In this mode, both live analysis and monitoring read/write via API.

### Programmatic Usage

You can also use the agents programmatically:

```python
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from agents import (
    HistoricalAnalysisAgent,
    IndicatorAnalysisAgent,
    FeatureEngineeringAgent,
    RegimeAgent,
    ForecastAgent,
    RiskAgent,
    BacktestAgent,
    PairLedgerAgent,
    PairMonitorAgent,
    NewsSentimentAgent,
    SupervisorAgent,
)

load_dotenv()
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)

# Initialize agents
historical_agent = HistoricalAnalysisAgent(llm)
indicator_agent = IndicatorAnalysisAgent(llm)
news_agent = NewsSentimentAgent(llm)
feature_agent = FeatureEngineeringAgent()
regime_agent = RegimeAgent()
forecast_agent = ForecastAgent()
risk_agent = RiskAgent()
backtest_agent = BacktestAgent()
ledger_agent = PairLedgerAgent()
pair_monitor_agent = PairMonitorAgent()
supervisor_agent = SupervisorAgent(llm)

# Analyze a stock
stock_symbol = "AAPL"

pair_ledger = ledger_agent.analyze()

historical_result = historical_agent.analyze(stock_symbol)
indicator_result = indicator_agent.analyze(stock_symbol)
news_result = news_agent.analyze(stock_symbol)
pair_monitor_result = pair_monitor_agent.analyze(
    pair_ledger.get("pairs", []),
    focus_symbol=stock_symbol,
)
feature_result = feature_agent.analyze(stock_symbol)
regime_result = regime_agent.analyze(stock_symbol, feature_result)
forecast_result = forecast_agent.analyze(stock_symbol, feature_result, regime_result)
risk_result = risk_agent.analyze(stock_symbol, forecast_result, regime_result, feature_result)
backtest_result = backtest_agent.analyze(stock_symbol)
recommendation = supervisor_agent.make_recommendation(
    historical_result,
    indicator_result,
    news_result,
    pair_monitor_result,
    stock_symbol,
    feature_result,
    regime_result,
    forecast_result,
    risk_result,
    backtest_result,
)

print(supervisor_agent.format_final_report(recommendation))
```

## Agent Details

### Historical Data Analysis Agent

- Fetches 7 days of historical stock data using Alpha Vantage
- Calculates key metrics:
  - Price changes and percentage changes
  - Volatility measures
  - Volume analysis
  - Trend identification
  - Support and resistance levels
- Identifies bullish or bearish patterns
### Indicator Analysis Agent

- Calculates RSI across daily/weekly/monthly timeframes
- Applies RSI logic to detect divergences, reversals, and channel regimes
- Outputs structured buy/sell/hold signals with reasoning

### Pair Ledger Agent

- Builds a persistent ledger of momentum-similar stock pairs
- Uses return correlation over a configurable lookback window
- Supports add/remove pairs via environment overrides on startup

### Pair Monitor Agent

- Monitors pair momentum over a short window (default: 5 trading days)
- Uses spread z-scores to detect divergence and identify leading/lagging stocks
- Feeds pair signals into the supervisor recommendation

### Feature Engineering Agent

- Computes machine-learning-ready features from OHLCV data
- Normalizes current market state into structured numeric signals

### Regime Agent

- Classifies trend and volatility regimes
- Adds market-context awareness to the forecast/risk pipeline

### Forecast Agent

- Produces probability-up forecast and confidence interval
- Supports trained model config at `FORECAST_MODEL_PATH`
- Falls back to deterministic heuristic if model config is missing

### Risk Agent

- Converts forecast confidence + volatility into position size guidance
- Produces stop-loss/take-profit levels and risk flags

### Backtest Agent

- Runs a lightweight walk-forward simulation
- Reports hit-rate, approximate Sharpe, max drawdown, and strategy return

### News Sentiment Analysis Agent

- Fetches latest news articles from multiple sources
- Uses DuckDuckGo and Tavily search APIs
- Performs detailed sentiment analysis:
  - Overall sentiment classification (Positive/Negative/Neutral)
  - Sentiment scoring
  - Key positive and negative factors
  - Impact assessment on stock price
  - Market and sector considerations

### Supervisor Agent

- Synthesizes information from all agents
- Identifies alignment or conflicts between technical, sentiment, and pair signals
- Provides clear recommendations:
  - BUY/SELL/HOLD decision
  - Risk assessment
  - Supporting factors
  - Potential price targets
  - Important caveats

## Output

The system generates:
1. Console output with real-time progress
2. Detailed final report displayed in terminal
3. Saved report file: `report_{SYMBOL}_{TIMESTAMP}.txt`

Report includes:
- Executive summary and recommendation
- Detailed historical analysis
- Comprehensive news sentiment analysis
- Quant feature/regime/forecast/risk outputs
- Backtest snapshot metrics
- Risk assessment and key considerations

## Dependencies

- **langchain**: Core framework for agent orchestration
- **langchain-openai**: OpenAI LLM integration
- **langchain-community**: Community tools and integrations
- **langgraph**: Advanced multi-agent workflows
- **Alpha Vantage (via requests)**: Stock market data
- **duckduckgo-search**: News search
- **tavily-python**: Enhanced news search (optional)
- **pandas, numpy**: Data analysis
- **beautifulsoup4, feedparser, newspaper3k**: Web scraping

## Configuration

You can customize the system by:
- Adjusting LLM temperature in `main.py` (default: 0.3 for consistency)
- Changing the number of days for historical analysis (default: 7)
- Modifying the number of news articles fetched (default: 10)
- Using different LLM models

## Pair Monitoring Configuration

Environment variables:

- Default universe file `data/sp500_top100.json` is sourced from us500.com as of 2026-01-30; update it as needed for freshness.
- `PAIR_UNIVERSE_PATH`: JSON file with tickers (default: `data/sp500_top100.json`)
- `PAIR_UNIVERSE`: Comma-separated override list of tickers
- `PAIR_LEDGER_PATH`: Where the ledger is stored (default: `data/pairs_ledger.json`)
- `PAIR_LOOKBACK_DAYS`: Lookback for correlation (default: 90)
- `PAIR_TOP_K`: Number of pairs to keep (default: 5)
- `PAIR_MIN_CORR`: Minimum correlation for pairs (default: 0.85)
- `PAIR_REBUILD`: Force rebuild on startup (true/false)
- `PAIR_REFRESH_INTERVAL_HOURS`: Rebuild ledger if older than N hours (overrides refresh time)
- `PAIR_REFRESH_UTC_HOUR`: Daily refresh hour in UTC (default: 21, ~4pm ET)
- `PAIR_REFRESH_UTC_MINUTE`: Daily refresh minute in UTC (default: 0)
- `PAIR_ADD`: Add pairs on startup, format `AAPL/MSFT,MSFT/NVDA`
- `PAIR_REMOVE`: Remove pairs on startup, format `AAPL/MSFT`
- `PAIR_MONITOR_WINDOW`: Momentum window for monitoring (default: 5)
- `PAIR_MONITOR_INTERVAL`: Data interval for monitoring (`daily`, `5min`, `15min`, `30min`, `60min`)
- `PAIR_ZSCORE_WINDOW`: Window for spread z-score (default: 20)
- `PAIR_ZSCORE_THRESHOLD`: Z-score threshold for divergence (default: 1.5)
- `PAIR_DIVERGENCE_THRESHOLD`: Backward-compatible alias for z-score threshold

## Storage Configuration

Environment variables:

- `STORAGE_ENABLED`: Enable persistence (default: true)
- `STORAGE_URL`: Storage connection string (default: `sqlite:///data/agent_store.db`)
- `STORAGE_USER_ID`: Optional user identifier to tag rows
If you want Postgres, install `psycopg2-binary` and set `STORAGE_URL` to your Postgres DSN.
For dashboard table viewing, SQLite storage is supported directly.

## API Configuration

Environment variables:

- `API_ALLOWED_ORIGINS`: CORS allowlist, comma-separated (default: `*`)
- `API_BASE_URL`: Dashboard default API URL (default: `http://localhost:8000`)
- `DASHBOARD_USE_API`: Start dashboard with API mode on/off (`true`/`false`, default: `false`)

## Alpha Vantage Rate Limits & Caching

Environment variables:

- `ALPHAVANTAGE_RATE_LIMIT_PER_MIN`: Throttle requests per minute (default: 5)
- `ALPHAVANTAGE_INTRADAY_TTL`: Intraday cache TTL in seconds (default: 300)

## Forecast/Model Configuration

Environment variables:

- `FORECAST_MODEL_PATH`: Path to trained model config JSON (default: `data/forecast_model.json`)
- `FORECAST_HORIZON_DAYS`: Forecast horizon for prediction layer (default: 5)
- `FORECAST_BUY_THRESHOLD`: Probability threshold for buy action (default: 0.55)
- `FORECAST_SELL_THRESHOLD`: Probability threshold for sell action (default: 0.45)

Training script (builds a model config from live market history):

```bash
python pipelines/train_forecast_model.py
```

## Limitations

- Requires internet connection for news fetching
- API rate limits may apply for OpenAI and search APIs
- Historical data depends on market hours and data availability
- News sentiment is based on available sources and may not capture all relevant news
- Recommendations are for informational purposes only and should not be considered as financial advice

## Troubleshooting

1. **API Key Errors**: Ensure your `.env` file contains a valid OpenAI API key
2. **No News Found**: Try adding a Tavily API key for better news search results
3. **Data Fetching Errors**: Check your internet connection and verify the stock symbol is correct
4. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This system provides AI-generated financial analysis and recommendations for informational purposes only. It should not be considered as professional financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results, and investments carry inherent risks.
