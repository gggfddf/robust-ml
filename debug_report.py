#!/usr/bin/env python3
"""
Debug script to test report generation and identify the exact error
"""

import sys
import pandas as pd
from data.live_data_loader import DataLoader
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from reports.report_generator import ReportGenerator
from config import get_config

def debug_report_generation():
    """Debug the report generation process."""
    print("🔍 Debugging Report Generation Process")
    print("=" * 50)
    
    try:
        # Initialize components
        config = get_config()
        loader = DataLoader(config)
        analyzer = TechnicalAnalyzer(config)
        report_gen = ReportGenerator(config)
        
        print(f"📊 Symbol: {config.SYMBOL}")
        
        # Get data
        print("\n🔍 Loading data...")
        daily_data = loader.fetch_single_timeframe_data(timeframe="1d")
        
        if daily_data is not None and not daily_data.empty:
            print(f"✅ Data loaded successfully")
            print(f"   Type: {type(daily_data)}")
            print(f"   Shape: {daily_data.shape}")
            
            # Get technical indicators
            print("\n🔍 Getting technical indicators...")
            indicators = analyzer.calculate_all_indicators(daily_data)
            print(f"✅ Technical indicators calculated: {len(indicators)}")
            
            # Test report generation
            print("\n🔍 Testing report generation...")
            try:
                # Mock data for testing
                patterns = {}
                predictions = {
                    'movement_direction': 'BULLISH',
                    'confidence_percentage': 75.5,
                    'target_price': 2500.0,
                    'time_horizon': '1 week'
                }
                
                reports = report_gen.generate_technical_analysis_report(
                    indicators, patterns, predictions, daily_data
                )
                
                print(f"✅ Report generation completed successfully")
                print(f"   Excel report: {reports.get('excel', 'Not generated')}")
                print(f"   HTML report: {reports.get('html', 'Not generated')}")
                print(f"   JSON report: {reports.get('json', 'Not generated')}")
                
            except Exception as e:
                print(f"❌ Report generation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No data available")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_report_generation()