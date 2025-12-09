#!/usr/bin/env python3
"""
SmartTrade AI - LLM Trading Demo

Demonstrates LLM-powered trading predictions using:
- Real stock data from Yahoo Finance (up to 50 years history)
- Technical indicators
- Multiple LLM providers (Ollama, Gemini)
- Multi-LLM consensus voting

Run with: python llm_trading_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import our modules
from src.indicators import TechnicalIndicators
from src.models.llm_predictor import LLMTradingPredictor, MultiLLMPredictor


def fetch_extended_data(symbol: str, years: int = 50) -> pd.DataFrame:
    """
    Fetch extended historical data (up to 50 years).
    
    Note: Yahoo Finance typically has ~20-30 years for most stocks,
    but we'll request the maximum available.
    """
    print(f"📊 Fetching {years} years of data for {symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            print(f"   ⚠️ No data found for {symbol}")
            return pd.DataFrame()
        
        # Rename columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'date'
        df = df.reset_index()
        
        years_available = (df['date'].max() - df['date'].min()).days / 365
        print(f"   ✅ Got {len(df)} rows ({years_available:.1f} years available)")
        
        return df
    except Exception as e:
        print(f"   ❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    indicators = TechnicalIndicators(df)
    df = indicators.add_all_indicators()
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """Extract latest indicator values as a dictionary."""
    latest = df.iloc[-1]
    
    indicator_cols = [
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_percent', 'bb_middle',
        'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
        'momentum_10', 'volatility_20', 'stoch_k', 'stoch_d',
        'atr', 'obv', 'daily_return', 'close', 'volume'
    ]
    
    indicators = {}
    for col in indicator_cols:
        if col in df.columns and pd.notna(latest[col]):
            indicators[col] = float(latest[col])
    
    return indicators


def print_prediction(prediction: dict):
    """Pretty print a prediction."""
    signal = prediction.get('signal', 'UNKNOWN')
    emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal, '⚪')
    
    print(f"\n{emoji} {signal} - {prediction.get('symbol', 'Unknown')}")
    print(f"   Confidence: {prediction.get('confidence', 0)*100:.0f}%")
    print(f"   Risk Level: {prediction.get('risk_level', 'N/A')}")
    print(f"   Provider: {prediction.get('provider', 'N/A')}")
    
    if prediction.get('reasoning'):
        print(f"   Reasoning: {prediction.get('reasoning', '')[:200]}...")
    
    if prediction.get('key_factors'):
        print("   Key Factors:")
        for factor in prediction.get('key_factors', [])[:3]:
            print(f"      • {factor}")
    
    if prediction.get('target_price'):
        print(f"   Target: ${prediction.get('target_price'):.2f}")
    if prediction.get('stop_loss'):
        print(f"   Stop Loss: ${prediction.get('stop_loss'):.2f}")


def demo_single_llm():
    """Demo with a single LLM (Ollama)."""
    print("\n" + "="*60)
    print("🤖 SINGLE LLM DEMO (Ollama)")
    print("="*60)
    
    # Fetch data
    df = fetch_extended_data('AAPL', years=50)
    if df.empty:
        print("Failed to fetch data")
        return
    
    # Add indicators
    df = add_indicators(df)
    indicators = get_latest_indicators(df)
    
    print(f"\n📈 Latest Indicators for AAPL:")
    for key in ['close', 'rsi_14', 'macd', 'bb_percent', 'momentum_10']:
        if key in indicators:
            print(f"   {key}: {indicators[key]:.4f}")
    
    # Create Ollama predictor
    print("\n🔄 Calling Ollama (local LLM)...")
    predictor = LLMTradingPredictor(provider='ollama')
    
    prediction = predictor.predict('AAPL', indicators, df[['close', 'volume']].tail(20))
    print_prediction(prediction)
    
    return prediction


def demo_multi_stock():
    """Demo with multiple stocks."""
    print("\n" + "="*60)
    print("📊 MULTI-STOCK ANALYSIS")
    print("="*60)
    
    stocks = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
    predictor = LLMTradingPredictor(provider='ollama')
    
    predictions = []
    
    for symbol in stocks:
        df = fetch_extended_data(symbol, years=50)
        if df.empty:
            continue
        
        df = add_indicators(df)
        indicators = get_latest_indicators(df)
        
        print(f"\n🔄 Analyzing {symbol}...")
        prediction = predictor.predict(symbol, indicators, df[['close', 'volume']].tail(20))
        predictions.append(prediction)
        print_prediction(prediction)
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    buys = [p for p in predictions if p.get('signal') == 'BUY']
    sells = [p for p in predictions if p.get('signal') == 'SELL']
    holds = [p for p in predictions if p.get('signal') == 'HOLD']
    
    print(f"\n🟢 BUY:  {len(buys)} stocks - {[p.get('symbol') for p in buys]}")
    print(f"🔴 SELL: {len(sells)} stocks - {[p.get('symbol') for p in sells]}")
    print(f"🟡 HOLD: {len(holds)} stocks - {[p.get('symbol') for p in holds]}")
    
    return predictions


def demo_multi_llm():
    """Demo with multiple LLMs for consensus."""
    print("\n" + "="*60)
    print("🗳️ MULTI-LLM CONSENSUS VOTING")
    print("="*60)
    
    # Fetch data
    df = fetch_extended_data('NVDA', years=50)
    if df.empty:
        print("Failed to fetch data")
        return
    
    df = add_indicators(df)
    indicators = get_latest_indicators(df)
    
    print("\n📈 Analyzing NVDA with multiple LLMs...")
    print("   (Requires API key for Gemini, or Ollama running locally)")
    
    # Try different providers
    providers = [
        {'provider': 'ollama', 'api_key': None},  # Free, local
        # Add your API keys here:
        # {'provider': 'gemini', 'api_key': 'YOUR_GEMINI_KEY'},
    ]
    
    multi_predictor = MultiLLMPredictor(providers)
    
    if len(multi_predictor.predictors) > 0:
        result = multi_predictor.predict('NVDA', indicators, df[['close', 'volume']].tail(20))
        
        print(f"\n🗳️ Consensus: {result.get('consensus_signal', 'N/A')}")
        print(f"   Agreement: {result.get('agreement_level', 0)*100:.0f}%")
        print(f"   Votes: {result.get('votes', {})}")
        print(f"   Average Confidence: {result.get('average_confidence', 0)*100:.0f}%")
        
        return result
    else:
        print("   No LLM providers available. Please configure API keys or run Ollama.")
        return None


def demo_historical_analysis():
    """Analyze how well LLM predictions would have performed historically."""
    print("\n" + "="*60)
    print("📜 HISTORICAL DATA OVERVIEW")
    print("="*60)
    
    stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corp.',
        'NVDA': 'NVIDIA Corp.',
        'GOOGL': 'Alphabet Inc.',
        'JPM': 'JPMorgan Chase'
    }
    
    for symbol, name in stocks.items():
        df = fetch_extended_data(symbol, years=50)
        if not df.empty:
            first_date = df['date'].min().strftime('%Y-%m-%d')
            last_date = df['date'].max().strftime('%Y-%m-%d')
            years = (df['date'].max() - df['date'].min()).days / 365
            
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            total_return = (last_price / first_price - 1) * 100
            
            print(f"\n{symbol} ({name}):")
            print(f"   Data Range: {first_date} to {last_date} ({years:.1f} years)")
            print(f"   First Price: ${first_price:.2f}")
            print(f"   Last Price: ${last_price:.2f}")
            print(f"   Total Return: {total_return:,.0f}%")


def main():
    """Main demo function."""
    print("="*60)
    print("🚀 SMARTTRADE AI - LLM TRADING PREDICTOR DEMO")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n⚠️ DISCLAIMER: This is for educational purposes only.")
    print("   Do not use for actual trading decisions.")
    
    # Check Ollama availability
    print("\n🔍 Checking Ollama availability...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama is running with {len(models)} models")
            if models:
                print(f"   Available: {[m.get('name', 'unknown') for m in models[:5]]}")
        else:
            print("   ⚠️ Ollama server responded but may have issues")
    except:
        print("   ⚠️ Ollama not running. Start with: ollama serve")
        print("   ⚠️ Install a model with: ollama pull mistral")
    
    # Run demos
    print("\n" + "-"*60)
    print("Select demo:")
    print("1. Single LLM prediction (AAPL)")
    print("2. Multi-stock analysis (5 stocks)")
    print("3. Multi-LLM consensus voting")
    print("4. Historical data overview (50 years)")
    print("5. Run all demos")
    print("-"*60)
    
    try:
        choice = input("Enter choice (1-5): ").strip()
    except:
        choice = "5"  # Default to all
    
    if choice == "1":
        demo_single_llm()
    elif choice == "2":
        demo_multi_stock()
    elif choice == "3":
        demo_multi_llm()
    elif choice == "4":
        demo_historical_analysis()
    else:
        demo_historical_analysis()
        demo_single_llm()
        demo_multi_stock()
        demo_multi_llm()
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
