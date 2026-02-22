"""
Historical Data Analysis Agent
Fetches and analyzes historical stock data from the past week
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from typing import Dict, Any

from utils.yfinance_cache import get_historical_data


@tool
def fetch_historical_stock_data(symbol: str, days: int = 7) -> str:
    """
    Fetches historical stock data for the given symbol over the specified number of days.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days: Number of days of historical data to fetch (default: 7)
    
    Returns:
        String containing formatted historical data and key statistics
    """
    try:
        days = max(1, int(days))
        hist_data = get_historical_data(symbol, interval="daily", days=days)
        
        if hist_data.empty:
            print(f"No historical data found for {symbol}")
            return f"No historical data found for {symbol}"

        start_date = hist_data.index[0]
        end_date = hist_data.index[-1]
        
        # Calculate key metrics
        current_price = hist_data['Close'].iloc[-1]
        week_ago_price = hist_data['Close'].iloc[0]
        price_change = current_price - week_ago_price
        price_change_pct = (price_change / week_ago_price) * 100
        
        avg_volume = hist_data['Volume'].mean()
        max_price = hist_data['High'].max()
        min_price = hist_data['Low'].min()
        volatility = hist_data['Close'].pct_change().std() * 100
        company_name = symbol.upper()
        
        analysis = f"""
HISTORICAL DATA ANALYSIS FOR {symbol} ({company_name})
{'='*60}
Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (last {len(hist_data)} trading days)

PRICE ANALYSIS:
- Current Price: ${current_price:.2f}
- Price {days} days ago: ${week_ago_price:.2f}
- Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
- Week High: ${max_price:.2f}
- Week Low: ${min_price:.2f}
- Volatility: {volatility:.2f}%

VOLUME ANALYSIS:
- Average Daily Volume: {avg_volume:,.0f}

TREND INDICATORS:
"""
        # Simple trend analysis
        if price_change_pct > 2:
            analysis += "- Strong upward trend detected\n"
        elif price_change_pct > 0:
            analysis += "- Moderate upward trend detected\n"
        elif price_change_pct > -2:
            analysis += "- Stable/flat trend\n"
        else:
            analysis += "- Downward trend detected\n"
        
        # Recent momentum
        if len(hist_data) >= 3:
            recent_change = (hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-3]) / hist_data['Close'].iloc[-3] * 100
            if recent_change > 0:
                analysis += f"- Recent momentum: Positive ({recent_change:.2f}% in last 3 days)\n"
            else:
                analysis += f"- Recent momentum: Negative ({recent_change:.2f}% in last 3 days)\n"
        
        csv_path = f"data/{symbol}_history_{len(hist_data)}d.csv"
        hist_data.to_csv(csv_path, index=True)
        
        return analysis
        
    except Exception as e:
        return f"Error fetching historical data for {symbol}: {str(e)}"


class HistoricalAnalysisAgent:
    """Agent responsible for analyzing historical stock data"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tools = [fetch_historical_stock_data]
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the agent with prompt and tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data analyst specializing in historical stock data analysis.
Your role is to:
1. Fetch and analyze historical stock price data (typically 1 week)
2. Calculate key metrics: price changes, volatility, volume trends
3. Identify price trends and patterns
4. Provide clear, concise analysis of the historical performance

Be thorough in your analysis and highlight both positive and negative indicators.
Format your response in a clear, structured manner."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=self.verbose)
    
    def analyze(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Analyze historical data for a given stock symbol
        
        Args:
            stock_symbol: Stock ticker symbol to analyze
            
        Returns:
            Dictionary with analysis results
        """
        query = f"Fetch and analyze historical stock data for {stock_symbol} over the past 7 days. Provide detailed analysis including price trends, volatility, and volume patterns."
        
        try:
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "")
            if not output or "Error" in output:
                return {
                    "agent": "historical_analysis",
                    "stock_symbol": stock_symbol,
                    "analysis": output or "No analysis generated",
                    "status": "error"
                }
            return {
                "agent": "historical_analysis",
                "stock_symbol": stock_symbol,
                "analysis": output,
                "status": "success"
            }
        except Exception as e:
            error_msg = str(e)
            return {
                "agent": "historical_analysis",
                "stock_symbol": stock_symbol,
                "analysis": f"Error during analysis: {error_msg}. Please verify the stock symbol is correct and try again.",
                "status": "error"
            }
