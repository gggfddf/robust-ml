#!/usr/bin/env python3
"""
Debug script to test data loading and identify the issue with 'list' object has no attribute 'shift'
"""

import sys
import pandas as pd
from data.live_data_loader import DataLoader
from config import get_config

def debug_data_loading():
    """Debug the data loading process."""
    print("🔍 Debugging Data Loading Process")
    print("=" * 50)
    
    try:
        # Initialize data loader
        config = get_config()
        loader = DataLoader(config)
        
        print(f"📊 Symbol: {config.SYMBOL}")
        print(f"⏰ Timeframes: {config.TIMEFRAMES}")
        
        # Test single timeframe data
        print("\n🔍 Testing single timeframe data (1d)...")
        daily_data = loader.fetch_single_timeframe_data(timeframe="1d")
        
        if daily_data is not None:
            print(f"✅ Data type: {type(daily_data)}")
            print(f"✅ Data shape: {daily_data.shape}")
            print(f"✅ Data columns: {list(daily_data.columns)}")
            print(f"✅ Data index type: {type(daily_data.index)}")
            print(f"✅ First few rows:")
            print(daily_data.head())
            
            # Test if it's a DataFrame
            if isinstance(daily_data, pd.DataFrame):
                print("✅ Data is a proper DataFrame")
                # Test shift operation
                try:
                    shifted = daily_data['Close'].shift(1)
                    print("✅ Shift operation works")
                except Exception as e:
                    print(f"❌ Shift operation failed: {e}")
            else:
                print(f"❌ Data is not a DataFrame, it's a {type(daily_data)}")
        else:
            print("❌ No data returned")
        
        # Test multi-timeframe data
        print("\n🔍 Testing multi-timeframe data...")
        multi_data = loader.get_latest_data()
        
        if multi_data:
            print(f"✅ Multi-data type: {type(multi_data)}")
            print(f"✅ Number of timeframes: {len(multi_data)}")
            
            for timeframe, data in multi_data.items():
                print(f"\n📊 {timeframe} timeframe:")
                print(f"   Type: {type(data)}")
                if isinstance(data, pd.DataFrame):
                    print(f"   Shape: {data.shape}")
                    print(f"   Columns: {list(data.columns)}")
                else:
                    print(f"   ❌ Not a DataFrame: {type(data)}")
        else:
            print("❌ No multi-timeframe data returned")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_loading()