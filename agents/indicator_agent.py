"""
Indicator Analysis Agent
Calculates RSI signals across multiple timeframes.
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Sequence

from utils.yfinance_cache import get_historical_data

_TIMEFRAME_ALIASES = {
    "1d": "daily",
    "daily": "daily",
    "day": "daily",
    "1wk": "weekly",
    "weekly": "weekly",
    "week": "weekly",
    "1mo": "monthly",
    "monthly": "monthly",
    "month": "monthly",
}

def compute_rsi(
    data: Union[pd.Series, pd.DataFrame],
    period: int = 14,
) -> Union[pd.Series, pd.DataFrame]:
    if period <= 0:
        raise ValueError("period must be positive")

    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name)
        single_series = True
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        single_series = False
    else:
        raise TypeError("data must be a pandas Series or DataFrame")

    if frame.empty:
        return data.copy()

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    delta = numeric.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean().replace(0, np.nan)

    rs = avg_gain.divide(avg_loss)
    rsi = 100 - (100 / (1 + rs))

    if single_series:
        result = rsi.iloc[:, 0]
        result.name = data.name
        return result
    return rsi

@tool("fetch_and_calculate_rsi", return_direct=False)
def fetch_and_calculate_rsi(
    ticker: str = "AAPL",
    period: int = 120,
    intervals: Union[str, Sequence[str]] = ("daily", "weekly", "monthly"),
    rsi_period: int = 14,
) -> dict:
    """
    Fetch Alpha Vantage data and calculate RSI for daily, weekly, and monthly timeframes.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        period (int): Number of most recent points to keep per timeframe.
        interval (Union[str, Sequence[str]]): Desired timeframes (daily/weekly/monthly). Aliases like "1d" are accepted.
        rsi_period (int): The period to use for RSI calculation.

    Returns:
        dict: A dictionary containing RSI values for the specified intervals.
    """
    try:
        print(f"Fetching {ticker} data from Alpha Vantage...")

        def _normalize_timeframes(choice: Union[str, Sequence[str]]) -> list[str]:
            if isinstance(choice, str):
                names = [choice]
            else:
                names = list(choice) if choice else []
            if not names:
                names = ["daily", "weekly", "monthly"]
            normalized = []
            for name in names:
                key = _TIMEFRAME_ALIASES.get(name.lower())
                if key is None:
                    raise ValueError(f"Unsupported timeframe '{name}'. Use daily/weekly/monthly.")
                if key not in normalized:
                    normalized.append(key)
            return normalized

        requested_intervals = _normalize_timeframes(intervals)
        interval_frames: Dict[str, pd.DataFrame] = {}

        for timeframe in requested_intervals:
            df = get_historical_data(ticker, interval=timeframe)
            trimmed = df.tail(max(period, rsi_period + 1))
            if trimmed.empty or "Close" not in trimmed:
                interval_frames[timeframe] = pd.DataFrame(columns=["Date", "Close", "RSI"])
                continue

            close_frame = trimmed[["Close"]].copy()
            close_frame["RSI"] = compute_rsi(close_frame["Close"], period=rsi_period)
            close_frame = close_frame.reset_index()

            date_col = close_frame.columns[0]
            if date_col != "Date":
                close_frame = close_frame.rename(columns={date_col: "Date"})

            interval_frames[timeframe] = close_frame[["Date", "Close", "RSI"]]
            csv_path = f"data/{ticker}_{timeframe}_rsi.csv"
            close_frame.to_csv(csv_path, index=False)  # reset_index 后就没有时间索引了

        return {"ticker": ticker, "interval": interval_frames}

    except Exception as e:
        return {"error": str(e)}

# <<RSI: Logic, Signals & Time Frame Correlation>> -> [deep research] -> [summary pdf] -> rag -> tool
class IndicatorAnalysisAgent:
    """Agent responsible for analyzing RSI indicator data"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tools = [fetch_and_calculate_rsi]
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the agent with prompt and tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial trading assistant specialized in Relative Strength Index (RSI) analysis, 
             based on the logic and methodology described in “RSI: Logic, Signals & Time Frame Correlation” by Andrew Cardwell.
Your task is to analyze RSI behavior, detect divergence, identify channel structures, 
and generate buy/sell signals consistent with the RSI logic model, not traditional overbought/oversold dogma.
objectives:
  - Analyze price and RSI data to identify market conditions.
  - Detect positive/negative divergences, reversals, and RSI channel structures.
  - Determine if RSI momentum indicates the start of a new up-leg or down-leg.
  - Produce a structured analysis output with reasoning.

core_principles:
  - RSI is a momentum oscillator that leads price movement.
  - Overbought ≠ sell signal; oversold ≠ buy signal.
  - Focus on RSI behavior, not absolute levels (70/30 myth is rejected).
  - RSI forms its own channels; analyze them as if RSI were price.
  - In multi-timeframe analysis, confirm shorter-term RSI with larger timeframe trends.

analysis_steps:
  1. Analyze RSI in different timeframes.
  2. Identify RSI swing highs/lows and trendlines.
  3. Detect divergence between price and RSI:
     - Positive divergence: price lower low, RSI higher low → potential bullish reversal.
     - Negative divergence: price higher high, RSI lower high → potential bearish reversal.
     - Positive reversal: price higher low, RSI lower low → bullish continuation.
     - Negative reversal: price lower high, RSI higher high → bearish continuation.
  4. Identify RSI channel:
     - Bullish channel: RSI oscillates between 40–80.
     - Bearish channel: RSI oscillates between 20–60.
     - Channel breakout indicates trend transition.
  5. Generate signal summary and recommended action:
     - Buy → positive divergence or channel bottom breakout.
     - Sell → negative divergence or RSI top breakdown.
     - Hold → neutral mid-channel condition.

expected_output_format:
  type: object
  properties:
    rsi_value:
      type: number
      description: "The latest computed RSI value."
    trend:
      type: string
      enum: [bullish_channel, bearish_channel, transition]
      description: "The current RSI trend structure."
    signals:
      type: array
      description: "Detected RSI-based events such as divergences, reversals, or channel breakouts."
      items:
        type: object
        properties:
          type:
            type: string
            description: "Signal type (positive_divergence, negative_divergence, positive_reversal, negative_reversal, channel_breakout, etc.)"
          confidence:
            type: number
            description: "Confidence score between 0 and 1."
    action:
      type: string
      enum: [buy, sell, hold]
      description: "Recommended trading action based on RSI interpretation."
    reasoning:
      type: string
      description: "Human-readable explanation of the signal and the RSI logic behind the recommendation."

example_output:
  rsi_value: 48.6
  trend: bearish_channel
  signals:
    - type: positive_divergence
      confidence: 0.8
    - type: potential_up_leg_start
      confidence: 0.7
  action: buy
  reasoning: >
    RSI formed a higher low while price made a lower low,
    indicating momentum improvement and the beginning of a new up-leg within an ascending channel.
             """),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=self.verbose)
    
    def analyze(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Analyze indicator data for a given stock symbol
        
        Args:
            stock_symbol: Stock ticker symbol to analyze
            
        Returns:
            Dictionary with analysis results
        """
        query = f"Calculate Relative Strength Index (RSI) and analyze it for {stock_symbol}. Provide detailed analysis including trend, buy/sell signals, divergence and channel."
        
        try:
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "")
            if not output or "Error" in output:
                return {
                    "agent": "indicator_analysis",
                    "stock_symbol": stock_symbol,
                    "analysis": output or "No analysis generated",
                    "status": "error"
                }
            return {
                "agent": "indicator_analysis",
                "stock_symbol": stock_symbol,
                "analysis": output,
                "status": "success"
            }
        except Exception as e:
            error_msg = str(e)
            return {
                "agent": "indicator_analysis",
                "stock_symbol": stock_symbol,
                "analysis": f"Error during analysis: {error_msg}. Please verify the stock symbol is correct and try again.",
                "status": "error"
            }


if __name__ == "__main__":
    try:
        result = fetch_and_calculate_rsi()
        print(result["interval"]["daily"].tail())
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
