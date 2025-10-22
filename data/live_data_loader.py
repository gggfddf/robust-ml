"""
Ultimate Market AI Engine - Live Data Loader
===========================================

This module provides live data fetching capabilities for Indian stocks
with multi-timeframe support, caching, and data quality validation.
"""

import asyncio
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pickle
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import subprocess
import sys

# Try to import additional data sources
try:
    from nsepython import equity_history
    NSEPYTHON_AVAILABLE = True
except ImportError:
    NSEPYTHON_AVAILABLE = False
    logging.warning("nsepython not available, will use fallback sources")

try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    logging.warning("nsepy not available, will use fallback sources")

try:
    from nsetools import nse
    NSETOOLS_AVAILABLE = True
except ImportError:
    NSETOOLS_AVAILABLE = False
    logging.warning("nsetools not available, will use fallback sources")

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Advanced data loader for Indian market data with multi-timeframe support.
    
    Features:
    - Multi-timeframe data fetching (5m, 15m, 1d, 1w)
    - Intelligent caching to avoid redundant API calls
    - Data quality validation and cleaning
    - Corporate action adjustments
    - Timezone-aware processing (IST)
    - Exponential backoff for API failures
    - Rate limiting to respect API constraints
    """
    
    def __init__(self, config=None):
        """Initialize the data loader with configuration."""
        self.config = config or get_config()
        self.cache_dir = Path(self.config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Data quality thresholds
        self.min_data_points = 100
        self.max_missing_ratio = 0.1
        
        # Data sources priority
        self.data_sources = ['yfinance', 'nsepython', 'nsepy', 'nsetools']
        
        # Install missing packages
        self._install_missing_packages()
        
        logger.info(f"DataLoader initialized for symbol: {self.config.SYMBOL}")
    
    def _install_missing_packages(self):
        """Install missing packages automatically."""
        required_packages = {
            'yfinance': 'yfinance',
            'nsepython': 'nsepython',
            'nsepy': 'nsepy',
            'nsetools': 'nsetools',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'scikit-learn',
            'tensorflow': 'tensorflow',
            'plotly': 'plotly',
            'ta': 'ta'
        }
        
        for package_name, pip_name in required_packages.items():
            try:
                __import__(package_name)
                logger.debug(f"Package {package_name} is available")
            except ImportError:
                logger.info(f"Installing {package_name}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                    logger.info(f"Successfully installed {package_name}")
                except Exception as e:
                    logger.warning(f"Failed to install {package_name}: {e}")

    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generate cache key for data."""
        return f"{symbol}_{timeframe}_{start_date}_{end_date}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        cache_file = self.cache_dir / cache_key
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > self.config.CACHE_DURATION:
            logger.debug(f"Cache expired for {cache_key}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded data from cache: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        try:
            cache_file = self.cache_dir / cache_key
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def _validate_data_quality(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Validate data quality and completeness."""
        if df is None or df.empty:
            logger.warning(f"Empty dataframe for {timeframe}")
            return False
        
        # Check minimum data points
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data points for {timeframe}: {len(df)}")
            return False
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > self.max_missing_ratio:
            logger.warning(f"Too many missing values for {timeframe}: {missing_ratio:.2%}")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns for {timeframe}")
            return False
        
        # Check for reasonable price values
        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            logger.warning(f"Invalid price values found for {timeframe}")
            return False
        
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        if df is None or df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by index
        df = df.sort_index()
        
        # Forward fill missing values (limited)
        df = df.fillna(method='ffill', limit=3)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Ensure volume is non-negative
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
            df['Volume'] = df['Volume'].clip(lower=0)
        
        # Add derived columns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Add OHLC relationships
        df['Body'] = df['Close'] - df['Open']
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Body_Ratio'] = abs(df['Body']) / (df['High'] - df['Low'])
        
        return df
    
    def _fetch_data_with_retry(self, symbol: str, timeframe: str, start_date: str, end_date: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data with exponential backoff retry logic using multiple sources."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Try multiple data sources in priority order
                df = self._fetch_from_multiple_sources(symbol, timeframe, start_date, end_date)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched {len(df)} records for {symbol} ({timeframe})")
                    return df
                else:
                    logger.warning(f"Empty data received for {symbol} ({timeframe}) on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol} ({timeframe}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retries failed for {symbol} ({timeframe})")
        
        return None

    def _fetch_from_multiple_sources(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from multiple sources with fallback."""
        # Remove .NS suffix for NSE sources
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        # Try yfinance first
        try:
            df = self._fetch_from_yfinance(symbol, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")
        
        # Try nsepython
        if NSEPYTHON_AVAILABLE:
            try:
                df = self._fetch_from_nsepython(clean_symbol, timeframe, start_date, end_date)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"nsepython failed: {e}")
        
        # Try nsepy
        if NSEPY_AVAILABLE:
            try:
                df = self._fetch_from_nsepy(clean_symbol, timeframe, start_date, end_date)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"nsepy failed: {e}")
        
        # Try nsetools
        if NSETOOLS_AVAILABLE:
            try:
                df = self._fetch_from_nsetools(clean_symbol, timeframe, start_date, end_date)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"nsetools failed: {e}")
        
        return None

    def _fetch_from_yfinance(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance."""
        yf_interval = self._convert_timeframe(timeframe)
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=yf_interval,
            auto_adjust=True,
            prepost=False
        )
        return df

    def _fetch_from_nsepython(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from nsepython."""
        # nsepython doesn't support intraday data, only daily
        if timeframe in ['1m', '5m', '15m']:
            return None
        
        data = equity_history(symbol, "NSE")
        if data is not None and not data.empty:
            # Convert to standard format
            df = pd.DataFrame(data)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            return df
        return None

    def _fetch_from_nsepy(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from nsepy."""
        # nsepy doesn't support intraday data, only daily
        if timeframe in ['1m', '5m', '15m']:
            return None
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        data = get_history(symbol=symbol, start=start_dt, end=end_dt)
        if data is not None and not data.empty:
            return data
        return None

    def _fetch_from_nsetools(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from nsetools."""
        # nsetools has limited functionality, mainly for current data
        try:
            nse_obj = nse()
            quote = nse_obj.get_quote(symbol)
            if quote:
                # Create a single row DataFrame with current data
                current_time = datetime.now()
                df = pd.DataFrame({
                    'Open': [quote['openPrice']],
                    'High': [quote['highPrice']],
                    'Low': [quote['lowPrice']],
                    'Close': [quote['lastPrice']],
                    'Volume': [quote['totalTradedVolume']]
                }, index=[current_time])
                return df
        except Exception as e:
            logger.warning(f"nsetools error: {e}")
        return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe to yfinance format."""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1wk',
            '1mo': '1mo'
        }
        return timeframe_map.get(timeframe, '1d')
    
    def _calculate_date_range(self, timeframe: str) -> Tuple[str, str]:
        """Calculate appropriate date range for the timeframe."""
        end_date = datetime.now()
        
        if timeframe == '1m':
            # 1-minute data - last 7 days (yfinance limit)
            start_date = end_date - timedelta(days=7)
        elif timeframe == '5m':
            # 5-minute data - last 30 days (yfinance limit)
            start_date = end_date - timedelta(days=30)
        elif timeframe == '15m':
            # 15-minute data - last 30 days (yfinance limit)
            start_date = end_date - timedelta(days=30)
        elif timeframe == '30m':
            # 30-minute data - last 30 days (yfinance limit)
            start_date = end_date - timedelta(days=30)
        elif timeframe == '1h':
            # Hourly data - last 730 days
            start_date = end_date - timedelta(days=730)
        elif timeframe == '1d':
            # Daily data - last 730 days
            start_date = end_date - timedelta(days=730)
        elif timeframe == '1w':
            # Weekly data - last 1825 days
            start_date = end_date - timedelta(days=1825)
        else:
            # Default to 730 days
            start_date = end_date - timedelta(days=730)
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    async def fetch_multi_timeframe_data(self, symbol: str = None, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes asynchronously.
        
        Args:
            symbol: Stock symbol (defaults to config symbol)
            timeframes: List of timeframes to fetch (defaults to config timeframes)
        
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        symbol = symbol or self.config.SYMBOL
        timeframes = timeframes or self.config.TIMEFRAMES
        
        logger.info(f"Fetching multi-timeframe data for {symbol}: {timeframes}")
        
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=len(timeframes)) as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for timeframe in timeframes:
                task = loop.run_in_executor(
                    executor,
                    self.fetch_single_timeframe_data,
                    symbol,
                    timeframe
                )
                tasks.append((timeframe, task))
            
            # Collect results
            results = {}
            for timeframe, task in tasks:
                try:
                    df = await task
                    if df is not None and not df.empty:
                        results[timeframe] = df
                    else:
                        logger.warning(f"No data received for {timeframe}")
                except Exception as e:
                    logger.error(f"Error fetching {timeframe}: {e}")
            
            logger.info(f"Successfully fetched data for {len(results)} timeframes")
            return results
    
    def fetch_single_timeframe_data(self, symbol: str = None, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch data for a single timeframe.
        
        Args:
            symbol: Stock symbol (defaults to config symbol)
            timeframe: Timeframe to fetch
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        symbol = symbol or self.config.SYMBOL
        
        # Calculate date range
        start_date, end_date = self._calculate_date_range(timeframe)
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        logger.info(f"Fetching {timeframe} data for {symbol} from {start_date} to {end_date}")
        
        df = self._fetch_data_with_retry(symbol, timeframe, start_date, end_date)
        
        if df is not None and not df.empty:
            # Validate and clean data
            if self._validate_data_quality(df, timeframe):
                df = self._clean_data(df)
                
                # Save to cache
                self._save_to_cache(cache_key, df)
                
                return df
            else:
                logger.error(f"Data quality validation failed for {timeframe}")
                return None
        else:
            logger.error(f"Failed to fetch data for {timeframe}")
            return None
    
    def get_latest_data(self, symbol: str = None, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get the latest data for all timeframes.
        
        Args:
            symbol: Stock symbol (defaults to config symbol)
            timeframes: List of timeframes (defaults to config timeframes)
        
        Returns:
            Dictionary with latest data for each timeframe
        """
        symbol = symbol or self.config.SYMBOL
        timeframes = timeframes or self.config.TIMEFRAMES
        
        logger.info(f"Getting latest data for {symbol}")
        
        results = {}
        for timeframe in timeframes:
            df = self.fetch_single_timeframe_data(symbol, timeframe)
            if df is not None and not df.empty:
                results[timeframe] = df
        
        return results
    
    def get_historical_data(self, symbol: str = None, start_date: str = None, end_date: str = None, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical data for a specific date range.
        
        Args:
            symbol: Stock symbol (defaults to config symbol)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe for data
        
        Returns:
            DataFrame with historical data or None if failed
        """
        symbol = symbol or self.config.SYMBOL
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        
        return self._fetch_data_with_retry(symbol, timeframe, start_date, end_date)
    
    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate summary statistics for multi-timeframe data.
        
        Args:
            data_dict: Dictionary with timeframe as key and DataFrame as value
        
        Returns:
            Dictionary with summary statistics for each timeframe
        """
        summary = {}
        
        for timeframe, df in data_dict.items():
            if df is not None and not df.empty:
                summary[timeframe] = {
                    'records': len(df),
                    'date_range': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                    'avg_volume': df['Volume'].mean() if 'Volume' in df.columns else 0,
                    'avg_price': df['Close'].mean(),
                    'volatility': df['Returns'].std() if 'Returns' in df.columns else 0,
                    'missing_values': df.isnull().sum().sum()
                }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    print("Testing DataLoader...")
    print(f"Symbol: {loader.config.SYMBOL}")
    print(f"Timeframes: {loader.config.TIMEFRAMES}")
    
    # Test single timeframe
    print("\nTesting single timeframe (1d)...")
    daily_data = loader.fetch_single_timeframe_data(timeframe="1d")
    if daily_data is not None:
        print(f"Daily data shape: {daily_data.shape}")
        print(f"Date range: {daily_data.index[0]} to {daily_data.index[-1]}")
        print(f"Columns: {list(daily_data.columns)}")
    
    # Test multi-timeframe
    print("\nTesting multi-timeframe...")
    multi_data = loader.get_latest_data()
    print(f"Fetched {len(multi_data)} timeframes")
    
    # Generate summary
    summary = loader.get_data_summary(multi_data)
    for timeframe, stats in summary.items():
        print(f"\n{timeframe}: {stats}")
