"""
News Sentiment Analysis Agent
Fetches latest news feeds and performs sentiment analysis on stock-related news
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
import warnings
from typing import Dict, Any, List

# Suppress duckduckgo_search deprecation warning
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*", category=RuntimeWarning)


@tool
def fetch_stock_news(symbol: str, max_results: int = 10) -> str:
    """
    Fetches the latest news articles related to a stock symbol from the internet.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        max_results: Maximum number of news articles to fetch (default: 10)
    
    Returns:
        String containing news articles with titles, sources, and summaries
    """
    try:
        news_data = []
        
        # Try using DuckDuckGo search
        try:
            search = DuckDuckGoSearchAPIWrapper()
            query = f"{symbol} stock news latest"
            results = search.results(query, max_results=max_results)
            
            for result in results:
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No snippet')
                link = result.get('link', 'No link')
                news_data.append({
                    'title': title,
                    'snippet': snippet,
                    'link': link,
                    'source': 'DuckDuckGo'
                })
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
        
        # Also try Tavily if API key is available
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                tavily_search = TavilySearchResults(
                    max_results=max_results,
                    api_key=tavily_api_key
                )
                tavily_results = tavily_search.run(f"{symbol} stock news financial latest")
                if isinstance(tavily_results, list):
                    for result in tavily_results:
                        if isinstance(result, dict):
                            news_data.append({
                                'title': result.get('title', 'No title'),
                                'snippet': result.get('content', 'No content'),
                                'link': result.get('url', 'No link'),
                                'source': 'Tavily'
                            })
            except Exception as e:
                # Silently continue if Tavily fails
                pass
        
        if not news_data:
            return f"No recent news found for {symbol}. Try checking financial news websites directly."
        
        # Format the news
        formatted_news = f"\nLATEST NEWS FOR {symbol}\n{'='*60}\n\n"
        for i, article in enumerate(news_data[:max_results], 1):
            formatted_news += f"{i}. {article['title']}\n"
            formatted_news += f"   Source: {article['source']}\n"
            formatted_news += f"   Summary: {article['snippet'][:200]}...\n"
            formatted_news += f"   Link: {article['link']}\n\n"
        
        return formatted_news
        
    except Exception as e:
        return f"Error fetching news for {symbol}: {str(e)}"


@tool
def analyze_news_sentiment(news_text: str) -> str:
    """
    Analyzes the sentiment of news text to determine if it's positive, negative, or neutral.
    
    Args:
        news_text: The news text to analyze
    
    Returns:
        String containing sentiment analysis results
    """
    try:
        # This will be enhanced by the LLM for better sentiment analysis
        # For now, return a structured prompt for the LLM to analyze
        return f"""
Please analyze the sentiment of the following news articles:

{news_text}

Provide a detailed sentiment analysis including:
1. Overall sentiment (Positive/Negative/Neutral)
2. Sentiment score (scale of -1 to +1, where -1 is very negative, 0 is neutral, +1 is very positive)
3. Key positive points mentioned
4. Key negative points mentioned
5. Potential impact on stock price
"""
    except Exception as e:
        return f"Error in sentiment analysis: {str(e)}"


class NewsSentimentAgent:
    """Agent responsible for fetching news and performing sentiment analysis"""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tools = [fetch_stock_news, analyze_news_sentiment]
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the agent with prompt and tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial news analyst specializing in sentiment analysis of stock-related news.
Your role is to:
1. Fetch the latest news articles related to a stock from the internet
2. Perform detailed sentiment analysis on the news
3. Identify positive and negative news factors
4. Assess potential impact on stock price based on news sentiment
5. Provide clear, actionable insights

Be thorough in your sentiment analysis. Consider:
- Overall tone and language used
- Specific events, earnings, product launches, regulatory issues
- Market reactions and analyst opinions
- Sector and market-wide news that might affect the stock

Format your response with clear positive and negative indicators."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=self.verbose)
    
    def analyze(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Fetch news and perform sentiment analysis for a given stock symbol
        
        Args:
            stock_symbol: Stock ticker symbol to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        query = f"Fetch the latest news articles for {stock_symbol} and perform a detailed sentiment analysis. Identify positive and negative news factors and assess potential impact on stock price."
        
        try:
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "")
            if not output or "Error" in output:
                return {
                    "agent": "news_sentiment",
                    "stock_symbol": stock_symbol,
                    "analysis": output or "No analysis generated",
                    "status": "error"
                }
            return {
                "agent": "news_sentiment",
                "stock_symbol": stock_symbol,
                "analysis": output,
                "status": "success"
            }
        except Exception as e:
            error_msg = str(e)
            return {
                "agent": "news_sentiment",
                "stock_symbol": stock_symbol,
                "analysis": f"Error during analysis: {error_msg}. This may be due to network issues or API limitations. Please try again later.",
                "status": "error"
            }

