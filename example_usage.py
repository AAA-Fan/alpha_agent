"""
Example usage of the Multi-Agent Financial Advisory System
Demonstrates programmatic usage of the agents
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents import (
    HistoricalAnalysisAgent,
    IndicatorAnalysisAgent,
    FeatureEngineeringAgent,
    RegimeAgent,
    ForecastAgent,
    RiskAgent,
    NewsSentimentAgent,
    PairLedgerAgent,
    PairMonitorAgent,
    SupervisorAgent,
)
# proxy = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy

def example_single_stock():
    """Example: Analyze a single stock"""
    load_dotenv()
    
    # Initialize LLM
    llm = ChatOpenAI(
model="gpt-4o",
        temperature=0.3
    )
    
    # Initialize agents
    historical_agent = HistoricalAnalysisAgent(llm)
    indicator_agent = IndicatorAnalysisAgent(llm)
    news_agent = NewsSentimentAgent(llm)
    feature_agent = FeatureEngineeringAgent()
    regime_agent = RegimeAgent()
    forecast_agent = ForecastAgent()
    risk_agent = RiskAgent()
    ledger_agent = PairLedgerAgent()
    pair_monitor_agent = PairMonitorAgent()
    supervisor_agent = SupervisorAgent(llm)

    pair_ledger = ledger_agent.analyze()
    
    # Analyze a stock
    stock_symbol = "AAPL"
    print(f"Analyzing {stock_symbol}...\n")
    
    # Get historical analysis
    print("Fetching historical data...")
    historical_result = historical_agent.analyze(stock_symbol)
    print(historical_result)

    # Get indicator analysis
    print("Fetching indicator data...")
    indicator_result = indicator_agent.analyze(stock_symbol)
    print(indicator_result)
    
    # Get news sentiment
    print("Fetching news and analyzing sentiment...")
    news_result = news_agent.analyze(stock_symbol)
    print(news_result)
    
    # Get pair monitoring analysis
    print("Monitoring momentum pairs...")
    if pair_ledger.get("status") == "success":
        pair_monitor_result = pair_monitor_agent.analyze(
            pair_ledger.get("pairs", []),
            focus_symbol=stock_symbol,
        )
    else:
        pair_monitor_result = {
            "agent": "pair_monitor",
            "status": "skipped",
            "summary": "Pair ledger unavailable; monitoring skipped.",
            "signals": [],
        }

    # Quantitative pipeline
    print("Building features...")
    feature_result = feature_agent.analyze(stock_symbol)
    print(feature_result)

    print("Classifying regime...")
    regime_result = regime_agent.analyze(stock_symbol, feature_result)
    print(regime_result)

    print("Generating forecast...")
    forecast_result = forecast_agent.analyze(stock_symbol, feature_result, regime_result)
    print(forecast_result)

    print("Preparing risk plan...")
    risk_result = risk_agent.analyze(stock_symbol, forecast_result, regime_result, feature_result)
    print(risk_result)

    # Get recommendation
    print("Generating recommendation...")
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
    )
    
    # Display report
    print(supervisor_agent.format_final_report(recommendation))


def example_multiple_stocks():
    """Example: Analyze multiple stocks"""
    load_dotenv()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    historical_agent = HistoricalAnalysisAgent(llm)
    news_agent = NewsSentimentAgent(llm)
    indicator_agent = IndicatorAnalysisAgent(llm)
    feature_agent = FeatureEngineeringAgent()
    regime_agent = RegimeAgent()
    forecast_agent = ForecastAgent()
    risk_agent = RiskAgent()
    ledger_agent = PairLedgerAgent()
    pair_monitor_agent = PairMonitorAgent()
    supervisor_agent = SupervisorAgent(llm)
    
    stocks = ["AAPL", "GOOGL", "MSFT"]
    
    pair_ledger = ledger_agent.analyze()

    for stock in stocks:
        print(f"\n{'='*80}")
        print(f"Analyzing {stock}")
        print('='*80)
        
        historical_result = historical_agent.analyze(stock)
        news_result = news_agent.analyze(stock)
        indicator_result = indicator_agent.analyze(stock)
        if pair_ledger.get("status") == "success":
            pair_monitor_result = pair_monitor_agent.analyze(
                pair_ledger.get("pairs", []),
                focus_symbol=stock,
            )
        else:
            pair_monitor_result = {
                "agent": "pair_monitor",
                "status": "skipped",
                "summary": "Pair ledger unavailable; monitoring skipped.",
                "signals": [],
            }

        feature_result = feature_agent.analyze(stock)
        regime_result = regime_agent.analyze(stock, feature_result)
        forecast_result = forecast_agent.analyze(stock, feature_result, regime_result)
        risk_result = risk_agent.analyze(stock, forecast_result, regime_result, feature_result)

        recommendation = supervisor_agent.make_recommendation(
            historical_result,
            indicator_result,
            news_result,
            pair_monitor_result,
            stock,
            feature_result,
            regime_result,
            forecast_result,
            risk_result,
        )
        
        print(supervisor_agent.format_final_report(recommendation))
        print("\n")


if __name__ == "__main__":
    # Run single stock example
    example_single_stock()
    
    # Uncomment to run multiple stocks example
    # example_multiple_stocks()
