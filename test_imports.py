#!/usr/bin/env python3
"""
Quick test script to verify all imports and basic functionality
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from agents import (
            HistoricalAnalysisAgent,
            IndicatorAnalysisAgent,
            PairLedgerAgent,
            PairMonitorAgent,
            FeatureEngineeringAgent,
            RegimeAgent,
            ForecastAgent,
            RiskAgent,
            NewsSentimentAgent,
            SupervisorAgent,
        )
        print("✓ All agent imports successful")
        
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        print("✓ LangChain imports successful")
        
        import yfinance as yf
        print("✓ yfinance import successful")
        
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        print("✓ DuckDuckGo search import successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_env():
    """Test that environment variables are set"""
    print("\nTesting environment...")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OPENAI_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print("✗ OPENAI_API_KEY not found in .env file")
        return False

def test_agent_initialization():
    """Test that agents can be initialized"""
    print("\nTesting agent initialization...")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("✗ Cannot test agent initialization without API key")
        return False
    
    try:
        from langchain_openai import ChatOpenAI
        from agents import (
            HistoricalAnalysisAgent,
            IndicatorAnalysisAgent,
            FeatureEngineeringAgent,
            RegimeAgent,
            ForecastAgent,
            RiskAgent,
            NewsSentimentAgent,
            SupervisorAgent,
        )
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        
        historical_agent = HistoricalAnalysisAgent(llm, verbose=False)
        print("✓ HistoricalAnalysisAgent initialized")

        indicator_agent = IndicatorAnalysisAgent(llm, verbose=False)
        print("✓ IndicatorAnalysisAgent initialized")

        feature_agent = FeatureEngineeringAgent(verbose=False)
        print("✓ FeatureEngineeringAgent initialized")

        regime_agent = RegimeAgent(verbose=False)
        print("✓ RegimeAgent initialized")

        forecast_agent = ForecastAgent(verbose=False)
        print("✓ ForecastAgent initialized")

        risk_agent = RiskAgent(verbose=False)
        print("✓ RiskAgent initialized")

        news_agent = NewsSentimentAgent(llm, verbose=False)
        print("✓ NewsSentimentAgent initialized")
        
        supervisor_agent = SupervisorAgent(llm)
        print("✓ SupervisorAgent initialized")
        
        return True
    except Exception as e:
        print(f"✗ Agent initialization error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Financial Advisory System - Import Test")
    print("=" * 60)
    
    success = True
    success &= test_imports()
    success &= test_env()
    
    if success:
        success &= test_agent_initialization()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! System is ready to use.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
