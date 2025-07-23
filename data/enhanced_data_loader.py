#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - ENHANCED DATA LOADER
Multi-timeframe Data Integration with Advanced Features
Production-Grade Implementation with Comprehensive Data Management
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataLoader:
    """Enhanced Multi-timeframe Data Loader with Advanced Features"""
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.cache_dir = "cache"
        self.data_cache = {}
        
        # Timeframe configurations
        self.timeframes = {
            "5m": {"period": "1y", "interval": "5m", "max_records": 52560},
            "15m": {"period": "3y", "interval": "15m", "max_records": 52560},
            "1d": {"period": "10y", "interval": "1d", "max_records": 2500},
            "1w": {"period": "10y", "interval": "1wk", "max_records": 520}
        }
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Enhanced DataLoader initialized for symbol: {symbol}")
    
    def load_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all timeframes"""
        logger.info("Loading multi-timeframe data...")
        
        multi_data = {}
        
        for timeframe, config in self.timeframes.items():
            logger.info(f"Loading {timeframe} data...")
            data = self.load_data(timeframe, self.symbol)
            if data is not None and len(data) > 0:
                multi_data[timeframe] = data
                logger.info(f"✅ {timeframe}: {len(data)} records loaded")
            else:
                logger.warning(f"⚠️ No data available for {timeframe}")
        
        return multi_data
    
    def load_data(self, timeframe: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for specific timeframe with caching"""
        try:
            # Check cache first
            cache_file = self._get_cache_file(symbol, timeframe)
            if os.path.exists(cache_file):
                logger.info(f"Loading {timeframe} data from cache...")
                data = pd.read_pickle(cache_file)
                if self._is_cache_valid(data, timeframe):
                    logger.info(f"✅ {timeframe} data loaded from cache: {len(data)} records")
                    return data
            
            # Fetch fresh data
            logger.info(f"Fetching fresh {timeframe} data for {symbol}...")
            data = self._fetch_data(symbol, timeframe)
            
            if data is not None and len(data) > 0:
                # Save to cache
                data.to_pickle(cache_file)
                logger.info(f"✅ {timeframe} data cached: {len(data)} records")
                return data
            else:
                logger.error(f"❌ Failed to fetch {timeframe} data")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading {timeframe} data: {str(e)}")
            return None
    
    def _fetch_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            config = self.timeframes[timeframe]
            
            # Calculate date range
            end_date = datetime.now()
            if config["period"] == "1y":
                start_date = end_date - timedelta(days=365)
            elif config["period"] == "3y":
                start_date = end_date - timedelta(days=3*365)
            elif config["period"] == "10y":
                start_date = end_date - timedelta(days=10*365)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=config["interval"],
                auto_adjust=True,
                prepost=True
            )
            
            if data is not None and len(data) > 0:
                # Clean and prepare data
                data = self._clean_data(data)
                return data
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching data: {str(e)}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        try:
            # Remove any rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill missing values for OHLC
            ohlc_columns = ['Open', 'High', 'Low', 'Close']
            data[ohlc_columns] = data[ohlc_columns].fillna(method='ffill')
            
            # Fill volume with 0 if missing
            if 'Volume' in data.columns:
                data['Volume'] = data['Volume'].fillna(0)
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 0
            
            # Sort by date
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error cleaning data: {str(e)}")
            return data
    
    def _get_cache_file(self, symbol: str, timeframe: str) -> str:
        """Get cache file path"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.cache_dir, f"{symbol}_{timeframe}_{end_date}.pkl")
    
    def _is_cache_valid(self, data: pd.DataFrame, timeframe: str) -> bool:
        """Check if cached data is valid"""
        try:
            if data is None or len(data) == 0:
                return False
            
            # Check if data is recent (within 1 day for daily, 1 hour for intraday)
            latest_date = data.index.max()
            current_date = datetime.now()
            
            if timeframe in ["1d", "1w"]:
                # Daily/weekly data should be within 1 day
                return (current_date - latest_date).days <= 1
            else:
                # Intraday data should be within 1 hour
                return (current_date - latest_date).total_seconds() <= 3600
                
        except Exception as e:
            logger.error(f"❌ Error validating cache: {str(e)}")
            return False
    
    def get_data_summary(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Get comprehensive data summary"""
        summary = {
            'symbol': self.symbol,
            'timeframes': {},
            'overall': {
                'total_records': 0,
                'date_range': {},
                'data_quality': {}
            }
        }
        
        for timeframe, data in multi_data.items():
            if data is not None and len(data) > 0:
                timeframe_summary = {
                    'records': len(data),
                    'date_range': f"{data.index.min()} to {data.index.max()}",
                    'price_range': f"₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}",
                    'avg_volume': f"{data['Volume'].mean():,.0f}",
                    'volatility': f"{data['Close'].pct_change().std()*100:.2f}%",
                    'missing_values': data.isnull().sum().sum(),
                    'duplicates': data.duplicated().sum()
                }
                
                summary['timeframes'][timeframe] = timeframe_summary
                summary['overall']['total_records'] += len(data)
        
        return summary
    
    def validate_data_quality(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Validate data quality across all timeframes"""
        quality_report = {
            'overall_score': 0,
            'timeframes': {},
            'issues': [],
            'recommendations': []
        }
        
        total_score = 0
        timeframe_count = 0
        
        for timeframe, data in multi_data.items():
            if data is not None and len(data) > 0:
                # Calculate quality metrics
                completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
                consistency = 1 - (data.duplicated().sum() / len(data))
                
                # Check for reasonable data ranges
                price_range = (data['Close'].max() - data['Close'].min()) / data['Close'].mean()
                volume_consistency = data['Volume'].std() / data['Volume'].mean() if data['Volume'].mean() > 0 else 0
                
                # Calculate timeframe score
                timeframe_score = (completeness * 0.4 + consistency * 0.3 + 
                                 min(1, price_range) * 0.2 + min(1, 1/volume_consistency) * 0.1)
                
                quality_report['timeframes'][timeframe] = {
                    'score': round(timeframe_score * 10, 2),
                    'completeness': round(completeness * 100, 2),
                    'consistency': round(consistency * 100, 2),
                    'records': len(data),
                    'issues': []
                }
                
                # Identify issues
                if completeness < 0.95:
                    quality_report['timeframes'][timeframe]['issues'].append("Missing data detected")
                if consistency < 0.99:
                    quality_report['timeframes'][timeframe]['issues'].append("Duplicate data detected")
                if len(data) < 100:
                    quality_report['timeframes'][timeframe]['issues'].append("Insufficient data points")
                
                total_score += timeframe_score
                timeframe_count += 1
        
        # Calculate overall score
        if timeframe_count > 0:
            quality_report['overall_score'] = round((total_score / timeframe_count) * 10, 2)
        
        # Generate recommendations
        if quality_report['overall_score'] < 8.0:
            quality_report['recommendations'].append("Consider refreshing data sources")
        if quality_report['overall_score'] < 6.0:
            quality_report['recommendations'].append("Data quality issues detected - manual review required")
        
        return quality_report

def main():
    """Test the enhanced data loader"""
    print("🚀 Testing Enhanced Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = EnhancedDataLoader("RELIANCE.NS")
    
    # Load multi-timeframe data
    multi_data = loader.load_multi_timeframe_data()
    
    # Get summary
    summary = loader.get_data_summary(multi_data)
    
    # Validate quality
    quality = loader.validate_data_quality(multi_data)
    
    # Print results
    print(f"\n📊 Data Summary for {summary['symbol']}:")
    print(f"Total Records: {summary['overall']['total_records']}")
    
    for timeframe, data in summary['timeframes'].items():
        print(f"\n{timeframe.upper()}:")
        print(f"  Records: {data['records']}")
        print(f"  Date Range: {data['date_range']}")
        print(f"  Price Range: {data['price_range']}")
        print(f"  Quality Score: {quality['timeframes'].get(timeframe, {}).get('score', 'N/A')}/10")
    
    print(f"\n🏆 Overall Quality Score: {quality['overall_score']}/10")
    
    if quality['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in quality['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()