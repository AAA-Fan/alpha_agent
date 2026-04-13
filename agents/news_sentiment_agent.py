"""
News Sentiment Analysis Agent
Fetches latest news and sentiment data from Alpha Vantage NEWS_SENTIMENT API
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List


# ── Alpha Vantage helpers ───────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"
_CALL_DELAY_SECONDS = 12  # Respect 5 calls/min free-tier limit


def _av_news_request(api_key: str, **params) -> Dict[str, Any]:
    """Make a NEWS_SENTIMENT request to Alpha Vantage."""
    request_params = {"function": "NEWS_SENTIMENT", "apikey": api_key, **params}
    resp = requests.get(_AV_BASE, params=request_params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info: {data['Information']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    return data


def _format_datetime(date_str: str) -> str:
    """Convert yyyy-mm-dd to Alpha Vantage format yyyymmddTHHmm."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m%dT%H%M")


def _sentiment_label(score: float) -> str:
    """Convert numeric sentiment score to human-readable label."""
    if score >= 0.35:
        return "Bullish 🟢"
    elif score >= 0.15:
        return "Somewhat Bullish 🟡"
    elif score > -0.15:
        return "Neutral ⚪"
    elif score > -0.35:
        return "Somewhat Bearish 🟠"
    else:
        return "Bearish 🔴"


def _format_articles(articles: List[Dict], ticker: str = None, max_articles: int = 15) -> str:
    """Format Alpha Vantage news articles into a readable report."""
    if not articles:
        return "No articles found."

    formatted = ""
    for i, article in enumerate(articles[:max_articles], 1):
        title = article.get("title", "No title")
        source = article.get("source", "Unknown")
        url = article.get("url", "")
        time_published = article.get("time_published", "")
        summary = article.get("summary", "")
        overall_score = float(article.get("overall_sentiment_score", 0))
        overall_label = article.get("overall_sentiment_label", "Neutral")

        # Format publish time
        pub_time = ""
        if time_published:
            try:
                dt = datetime.strptime(time_published[:15], "%Y%m%dT%H%M%S")
                pub_time = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, IndexError):
                pub_time = time_published

        formatted += f"{i}. {title}\n"
        formatted += f"   Source: {source}  |  Published: {pub_time}\n"
        formatted += f"   Overall Sentiment: {overall_label} (score: {overall_score:+.4f})\n"

        # Show ticker-specific sentiment if available
        if ticker:
            ticker_sentiments = article.get("ticker_sentiment", [])
            for ts in ticker_sentiments:
                if ts.get("ticker", "").upper() == ticker.upper():
                    ts_score = float(ts.get("ticker_sentiment_score", 0))
                    ts_label = ts.get("ticker_sentiment_label", "Neutral")
                    relevance = float(ts.get("relevance_score", 0))
                    formatted += f"   {ticker} Sentiment: {ts_label} (score: {ts_score:+.4f}, relevance: {relevance:.4f})\n"
                    break

        if summary:
            # Truncate long summaries
            display_summary = summary[:300] + "..." if len(summary) > 300 else summary
            formatted += f"   Summary: {display_summary}\n"
        if url:
            formatted += f"   Link: {url}\n"
        formatted += "\n"

    return formatted


# ── Tools ───────────────────────────────────────────────────────────────

@tool
def fetch_stock_news(symbol: str, max_results: int = 15) -> str:
    """
    Fetches the latest news articles and sentiment data for a specific stock
    from Alpha Vantage NEWS_SENTIMENT API. Returns articles with AI-powered
    sentiment scores from 50+ premier financial news sources.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        max_results: Maximum number of news articles to fetch (default: 15)

    Returns:
        String containing news articles with titles, sources, sentiment scores, and summaries
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        return "Error: ALPHAVANTAGE_API_KEY not set in environment variables."

    try:
        # Look back 7 days for recent news
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = _av_news_request(
            api_key,
            tickers=symbol,
            time_from=_format_datetime(start_date.strftime("%Y-%m-%d")),
            time_to=_format_datetime(end_date.strftime("%Y-%m-%d")),
            limit=str(max_results),
            sort="LATEST",
        )

        articles = data.get("feed", [])
        total_results = data.get("items", "0")

        if not articles:
            return f"No recent news found for {symbol} in the past 7 days."

        # Compute aggregate sentiment stats
        ticker_scores = []
        for article in articles:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    score = float(ts.get("ticker_sentiment_score", 0))
                    ticker_scores.append(score)
                    break

        report = f"""
ALPHA VANTAGE NEWS SENTIMENT REPORT — {symbol}
{'=' * 60}
Data Source: Alpha Vantage NEWS_SENTIMENT API (50+ financial news sources)
Period: Past 7 days  |  Articles found: {total_results}
"""

        if ticker_scores:
            avg_score = sum(ticker_scores) / len(ticker_scores)
            max_score = max(ticker_scores)
            min_score = min(ticker_scores)
            bullish_count = sum(1 for s in ticker_scores if s >= 0.15)
            bearish_count = sum(1 for s in ticker_scores if s <= -0.15)
            neutral_count = len(ticker_scores) - bullish_count - bearish_count

            report += f"""
AGGREGATE SENTIMENT STATISTICS:
  Average Sentiment Score: {avg_score:+.4f} — {_sentiment_label(avg_score)}
  Highest Score: {max_score:+.4f}  |  Lowest Score: {min_score:+.4f}
  Bullish Articles: {bullish_count}  |  Neutral: {neutral_count}  |  Bearish: {bearish_count}
{'─' * 60}
"""

        report += f"\nARTICLES:\n{'─' * 60}\n"
        report += _format_articles(articles, ticker=symbol, max_articles=max_results)

        return report

    except Exception as e:
        return f"Error fetching news for {symbol}: {str(e)}"


@tool
def fetch_global_market_news(max_results: int = 15) -> str:
    """
    Fetches global market and macroeconomic news from Alpha Vantage NEWS_SENTIMENT API.
    Covers topics: financial markets, economy & macro, monetary policy, fiscal policy.
    Useful for understanding the broader market environment.

    Args:
        max_results: Maximum number of news articles to fetch (default: 15)

    Returns:
        String containing global market news with sentiment scores
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        return "Error: ALPHAVANTAGE_API_KEY not set in environment variables."

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = _av_news_request(
            api_key,
            topics="financial_markets,economy_macro,economy_monetary,economy_fiscal",
            time_from=_format_datetime(start_date.strftime("%Y-%m-%d")),
            time_to=_format_datetime(end_date.strftime("%Y-%m-%d")),
            limit=str(max_results),
            sort="LATEST",
        )

        time.sleep(_CALL_DELAY_SECONDS)

        articles = data.get("feed", [])
        total_results = data.get("items", "0")

        if not articles:
            return "No global market news found in the past 7 days."

        # Compute overall sentiment distribution
        scores = [float(a.get("overall_sentiment_score", 0)) for a in articles]
        avg_score = sum(scores) / len(scores) if scores else 0
        bullish_count = sum(1 for s in scores if s >= 0.15)
        bearish_count = sum(1 for s in scores if s <= -0.15)
        neutral_count = len(scores) - bullish_count - bearish_count

        report = f"""
GLOBAL MARKET NEWS SENTIMENT REPORT
{'=' * 60}
Data Source: Alpha Vantage NEWS_SENTIMENT API
Topics: Financial Markets, Economy & Macro, Monetary Policy, Fiscal Policy
Period: Past 7 days  |  Articles found: {total_results}

AGGREGATE SENTIMENT:
  Average Score: {avg_score:+.4f} — {_sentiment_label(avg_score)}
  Bullish Articles: {bullish_count}  |  Neutral: {neutral_count}  |  Bearish: {bearish_count}
{'─' * 60}

ARTICLES:
{'─' * 60}
"""
        report += _format_articles(articles, max_articles=max_results)

        return report

    except Exception as e:
        return f"Error fetching global market news: {str(e)}"


# ── Agent class ─────────────────────────────────────────────────────────

class NewsSentimentAgent:
    """Agent responsible for fetching news and performing sentiment analysis
    using Alpha Vantage NEWS_SENTIMENT API as the data source."""

    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose
        self.tools = [fetch_stock_news, fetch_global_market_news]
        self._setup_agent()

    def _setup_agent(self):
        """Setup the agent with prompt and tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial news analyst specializing in sentiment analysis of stock-related news.
You have access to Alpha Vantage's NEWS_SENTIMENT API, which provides news from 50+ premier
financial news sources with AI-powered sentiment scores.

Your role is to:
1. Fetch the latest news articles related to a stock using fetch_stock_news
2. Fetch broader market/macro news using fetch_global_market_news for context
3. Analyze both the AI sentiment scores AND the actual news content
4. Identify key positive and negative catalysts
5. Assess potential impact on stock price based on news sentiment

When analyzing sentiment, consider:
- The aggregate sentiment score and distribution (bullish vs bearish articles)
- Individual article sentiment scores and relevance scores
- Specific events: earnings, product launches, regulatory issues, M&A, lawsuits
- Market-wide factors from global news that could affect the stock
- Sector-specific news and trends

Provide your analysis in clear sections:
1. OVERALL NEWS SENTIMENT: Bullish / Neutral / Bearish (with score)
2. KEY POSITIVE CATALYSTS: List specific positive news items
3. KEY NEGATIVE CATALYSTS: List specific negative news items
4. MARKET CONTEXT: How broader market news affects this stock
5. SENTIMENT TREND: Is sentiment improving or deteriorating?
6. IMPACT ASSESSMENT: Expected impact on stock price (short-term and medium-term)"""),
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
        query = (
            f"Analyze the news sentiment for {stock_symbol}. "
            f"First, fetch the latest stock-specific news for {stock_symbol} using fetch_stock_news. "
            f"Then, fetch global market news using fetch_global_market_news for broader context. "
            f"Finally, provide a comprehensive sentiment analysis combining both data sources. "
            f"Pay attention to the AI sentiment scores provided by Alpha Vantage, "
            f"and also apply your own judgment on the news content."
        )

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
                "analysis": (
                    f"Error during news sentiment analysis: {error_msg}. "
                    "This may be due to Alpha Vantage API rate limits (5 calls/min on free tier). "
                    "Please try again shortly."
                ),
                "status": "error"
            }

