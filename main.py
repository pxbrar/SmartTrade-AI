#!/usr/bin/env python3
"""
SmartTrade AI - Main Entry Point

Quick setup and data collection script.
Run this first to set up the project and collect data.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import StockDataCollector, ALL_SYMBOLS
from database import SmartTradeDB
from indicators import calculate_all_indicators


def main():
    print("=" * 60)
    print("🤖 SmartTrade AI - Setup & Data Collection")
    print("=" * 60)
    
    # Step 1: Initialize database
    print("\n📁 Step 1: Initializing database...")
    db = SmartTradeDB()
    
    # Step 2: Collect stock data
    print("\n📥 Step 2: Collecting stock data...")
    collector = StockDataCollector()
    stock_data = collector.fetch_all_stocks(period='5y')
    
    # Step 3: Save to CSV
    print("\n💾 Step 3: Saving raw data to CSV...")
    collector.save_to_csv(stock_data)
    
    # Step 3.5: Save to SQLite database
    print("\n🗄️ Step 3.5: Saving data to SQLite database...")
    total_saved = 0
    for symbol, df in stock_data.items():
        try:
            # Insert stock metadata
            db.insert_stock(symbol, name=symbol)
            
            # Insert price data
            df_to_save = df.copy()
            df_to_save['symbol'] = symbol
            rows = db.insert_prices(df_to_save)
            total_saved += rows
            print(f"  💾 {symbol}: {rows} records saved to database")
        except Exception as e:
            print(f"  ⚠️ {symbol}: {e}")
    print(f"  ✅ Total: {total_saved:,} records saved to database")
    
    # Step 4: Calculate indicators and save processed data
    print("\n📊 Step 4: Calculating technical indicators...")
    for symbol, df in stock_data.items():
        try:
            df_ind = calculate_all_indicators(df)
            
            # Save processed data
            filename = symbol.replace('^', 'INDEX_').replace('.', '_')
            filepath = os.path.join('data', 'processed', f"{filename}_processed.csv")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df_ind.to_csv(filepath, index=False)
            
            print(f"  ✅ {symbol}: {len(df_ind)} records, {len(df_ind.columns)} features")
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n📊 Data Summary:")
    print(f"  • Stocks collected: {len(stock_data)}")
    print(f"  • Total records: {sum(len(df) for df in stock_data.values()):,}")
    print(f"  • Database: database/smarttrade.db")
    print(f"  • Raw data: data/raw/")
    print(f"  • Processed data: data/processed/")
    
    print("\n🚀 Next Steps:")
    print("  1. Run notebooks: jupyter notebook notebooks/01_EDA.ipynb")
    print("  2. Launch dashboard: python dashboard/app.py")
    print("  3. Open http://localhost:8050 in your browser")
    
    return stock_data


if __name__ == "__main__":
    main()
