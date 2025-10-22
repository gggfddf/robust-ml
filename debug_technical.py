#!/usr/bin/env python3
"""
Debug script to test technical analysis and identify the 'list' object has no attribute 'shift' error
"""

import sys
import pandas as pd
from data.live_data_loader import DataLoader
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from config import get_config

def debug_technical_analysis():
    """Debug the technical analysis process."""
    print("🔍 Debugging Technical Analysis Process")
    print("=" * 50)
    
    try:
        # Initialize components
        config = get_config()
        loader = DataLoader(config)
        analyzer = TechnicalAnalyzer(config)
        
        print(f"📊 Symbol: {config.SYMBOL}")
        
        # Get data
        print("\n🔍 Loading data...")
        daily_data = loader.fetch_single_timeframe_data(timeframe="1d")
        
        if daily_data is not None and not daily_data.empty:
            print(f"✅ Data loaded successfully")
            print(f"   Type: {type(daily_data)}")
            print(f"   Shape: {daily_data.shape}")
            print(f"   Columns: {list(daily_data.columns)}")
            
            # Test technical analysis
            print("\n🔍 Testing technical analysis...")
            try:
                indicators = analyzer.calculate_all_indicators(daily_data)
                print(f"✅ Technical analysis completed successfully")
                print(f"   Number of indicators: {len(indicators)}")
                
                # Test a few specific indicators
                for indicator_name, indicator_data in list(indicators.items())[:5]:
                    print(f"   {indicator_name}: {type(indicator_data)}")
                    
            except Exception as e:
                print(f"❌ Technical analysis failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No data available")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_technical_analysis()