"""
Ultimate Market AI Engine - Time-Based Market Cycle Analysis
==========================================================

This module implements comprehensive time-based market cycle analysis
to uncover hidden temporal patterns in market behavior.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
import calendar

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Comprehensive time-based market cycle analysis system.
    
    Features:
    - Intraday patterns: Hourly movement analysis
    - Day-of-week effects: Monday-Friday patterns
    - Monthly patterns: Month-end/start effects
    - Seasonal patterns: Quarterly behaviors
    - Gap analysis: Gap up/down with fill probability
    - Expiry effects: F&O expiry patterns
    - Maximum movement analysis per session
    - Reversal timing predictions
    - Special events: Earnings, dividends, splits
    """
    
    def __init__(self, config=None):
        """Initialize the temporal analyzer."""
        self.config = config or get_config()
        self.scaler = StandardScaler()
        
        # Analysis results storage
        self.intraday_patterns = {}
        self.day_of_week_patterns = {}
        self.monthly_patterns = {}
        self.seasonal_patterns = {}
        self.gap_patterns = {}
        self.expiry_patterns = {}
        self.special_event_patterns = {}
        
        logger.info("Temporal Analyzer initialized")
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Perform comprehensive temporal pattern analysis.
        
        Args:
            df: DataFrame with OHLCV data and datetime index
        
        Returns:
            Dictionary with all temporal analysis results
        """
        if df is None or df.empty:
            return {}
        
        logger.info("Starting comprehensive temporal pattern analysis...")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Perform all analyses
        results = {}
        
        results['intraday'] = self._analyze_intraday_patterns(df)
        results['day_of_week'] = self._analyze_day_of_week_patterns(df)
        results['monthly'] = self._analyze_monthly_patterns(df)
        results['seasonal'] = self._analyze_seasonal_patterns(df)
        results['gaps'] = self._analyze_gap_patterns(df)
        results['expiry'] = self._analyze_expiry_patterns(df)
        results['special_events'] = self._analyze_special_event_patterns(df)
        results['cycle_detection'] = self._detect_market_cycles(df)
        
        # Store results
        self.intraday_patterns = results['intraday']
        self.day_of_week_patterns = results['day_of_week']
        self.monthly_patterns = results['monthly']
        self.seasonal_patterns = results['seasonal']
        self.gap_patterns = results['gaps']
        self.expiry_patterns = results['expiry']
        self.special_event_patterns = results['special_events']
        
        logger.info("Temporal pattern analysis completed")
        return results
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the dataframe."""
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['week_of_year'] = df.index.isocalendar().week
        
        # Market session features
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 15) & 
                               (df['day_of_week'] < 5)).astype(int)
        
        # Gap features
        df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
        df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
        df['gap_size'] = abs(df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Expiry features (simplified - last Thursday of month)
        df['is_expiry_day'] = self._is_expiry_day(df.index)
        
        # Month-end/start features
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        
        return df
    
    def _is_expiry_day(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Identify F&O expiry days (last Thursday of month)."""
        expiry_days = pd.Series(False, index=dates)
        
        for date in dates:
            # Get last Thursday of the month
            last_day = calendar.monthrange(date.year, date.month)[1]
            for day in range(last_day, 0, -1):
                if calendar.weekday(date.year, date.month, day) == calendar.THURSDAY:
                    if date.day == day:
                        expiry_days[date] = True
                    break
        
        return expiry_days
    
    def _analyze_intraday_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze intraday patterns and hourly movements."""
        logger.info("Analyzing intraday patterns...")
        
        patterns = {}
        
        # Hourly returns analysis
        hourly_returns = df.groupby('hour')['Close'].apply(
            lambda x: x.pct_change().mean()
        ).dropna()
        
        patterns['hourly_returns'] = hourly_returns.to_dict()
        
        # Hourly volatility analysis
        hourly_volatility = df.groupby('hour')['Close'].apply(
            lambda x: x.pct_change().std()
        ).dropna()
        
        patterns['hourly_volatility'] = hourly_volatility.to_dict()
        
        # Best/worst performing hours
        best_hour = hourly_returns.idxmax()
        worst_hour = hourly_returns.idxmin()
        
        patterns['best_performing_hour'] = {
            'hour': best_hour,
            'avg_return': hourly_returns[best_hour]
        }
        
        patterns['worst_performing_hour'] = {
            'hour': worst_hour,
            'avg_return': hourly_returns[worst_hour]
        }
        
        # Volume patterns by hour
        if 'Volume' in df.columns:
            hourly_volume = df.groupby('hour')['Volume'].mean()
            patterns['hourly_volume'] = hourly_volume.to_dict()
        
        # Opening hour analysis (9:15 AM)
        opening_data = df[df['hour'] == 9]
        if not opening_data.empty:
            patterns['opening_patterns'] = {
                'avg_return': opening_data['Close'].pct_change().mean(),
                'volatility': opening_data['Close'].pct_change().std(),
                'gap_fill_rate': (opening_data['Low'] <= opening_data['Open'].shift(1)).mean()
            }
        
        # Closing hour analysis (3:30 PM)
        closing_data = df[df['hour'] == 15]
        if not closing_data.empty:
            patterns['closing_patterns'] = {
                'avg_return': closing_data['Close'].pct_change().mean(),
                'volatility': closing_data['Close'].pct_change().std(),
                'end_of_day_effect': closing_data['Close'].pct_change().mean()
            }
        
        return patterns
    
    def _analyze_day_of_week_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze day-of-week patterns."""
        logger.info("Analyzing day-of-week patterns...")
        
        patterns = {}
        
        # Daily returns by day of week
        daily_returns = df.groupby('day_of_week')['Close'].apply(
            lambda x: x.pct_change().mean()
        ).dropna()
        
        patterns['daily_returns'] = daily_returns.to_dict()
        
        # Daily volatility by day of week
        daily_volatility = df.groupby('day_of_week')['Close'].apply(
            lambda x: x.pct_change().std()
        ).dropna()
        
        patterns['daily_volatility'] = daily_volatility.to_dict()
        
        # Monday effect
        monday_data = df[df['day_of_week'] == 0]
        if not monday_data.empty:
            patterns['monday_effect'] = {
                'avg_return': monday_data['Close'].pct_change().mean(),
                'volatility': monday_data['Close'].pct_change().std(),
                'gap_frequency': monday_data['gap_up'].mean()
            }
        
        # Friday effect
        friday_data = df[df['day_of_week'] == 4]
        if not friday_data.empty:
            patterns['friday_effect'] = {
                'avg_return': friday_data['Close'].pct_change().mean(),
                'volatility': friday_data['Close'].pct_change().std(),
                'weekend_effect': friday_data['Close'].pct_change().mean()
            }
        
        # Best/worst performing days
        best_day = daily_returns.idxmax()
        worst_day = daily_returns.idxmin()
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        patterns['best_performing_day'] = {
            'day': day_names[best_day],
            'avg_return': daily_returns[best_day]
        }
        
        patterns['worst_performing_day'] = {
            'day': day_names[worst_day],
            'avg_return': daily_returns[worst_day]
        }
        
        return patterns
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly patterns and month-end/start effects."""
        logger.info("Analyzing monthly patterns...")
        
        patterns = {}
        
        # Monthly returns
        monthly_returns = df.groupby('month')['Close'].apply(
            lambda x: x.pct_change().mean()
        ).dropna()
        
        patterns['monthly_returns'] = monthly_returns.to_dict()
        
        # Month-end effect
        month_end_data = df[df['is_month_end'] == 1]
        if not month_end_data.empty:
            patterns['month_end_effect'] = {
                'avg_return': month_end_data['Close'].pct_change().mean(),
                'volatility': month_end_data['Close'].pct_change().std(),
                'volume_effect': month_end_data['Volume'].mean() if 'Volume' in month_end_data.columns else 0
            }
        
        # Month-start effect
        month_start_data = df[df['is_month_start'] == 1]
        if not month_start_data.empty:
            patterns['month_start_effect'] = {
                'avg_return': month_start_data['Close'].pct_change().mean(),
                'volatility': month_start_data['Close'].pct_change().std(),
                'momentum_effect': month_start_data['Close'].pct_change().mean()
            }
        
        # Best/worst performing months
        best_month = monthly_returns.idxmax()
        worst_month = monthly_returns.idxmin()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        patterns['best_performing_month'] = {
            'month': month_names[best_month - 1],
            'avg_return': monthly_returns[best_month]
        }
        
        patterns['worst_performing_month'] = {
            'month': month_names[worst_month - 1],
            'avg_return': monthly_returns[worst_month]
        }
        
        return patterns
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns and quarterly behaviors."""
        logger.info("Analyzing seasonal patterns...")
        
        patterns = {}
        
        # Quarterly returns
        quarterly_returns = df.groupby('quarter')['Close'].apply(
            lambda x: x.pct_change().mean()
        ).dropna()
        
        patterns['quarterly_returns'] = quarterly_returns.to_dict()
        
        # Seasonal volatility
        quarterly_volatility = df.groupby('quarter')['Close'].apply(
            lambda x: x.pct_change().std()
        ).dropna()
        
        patterns['quarterly_volatility'] = quarterly_volatility.to_dict()
        
        # Year-end effect (Q4)
        q4_data = df[df['quarter'] == 4]
        if not q4_data.empty:
            patterns['year_end_effect'] = {
                'avg_return': q4_data['Close'].pct_change().mean(),
                'volatility': q4_data['Close'].pct_change().std(),
                'window_dressing_effect': q4_data['Close'].pct_change().mean()
            }
        
        # Year-start effect (Q1)
        q1_data = df[df['quarter'] == 1]
        if not q1_data.empty:
            patterns['year_start_effect'] = {
                'avg_return': q1_data['Close'].pct_change().mean(),
                'volatility': q1_data['Close'].pct_change().std(),
                'january_effect': q1_data['Close'].pct_change().mean()
            }
        
        return patterns
    
    def _analyze_gap_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gap patterns and fill probabilities."""
        logger.info("Analyzing gap patterns...")
        
        patterns = {}
        
        # Gap statistics
        gap_up_data = df[df['gap_up'] == 1]
        gap_down_data = df[df['gap_down'] == 1]
        
        patterns['gap_statistics'] = {
            'gap_up_frequency': df['gap_up'].mean(),
            'gap_down_frequency': df['gap_down'].mean(),
            'avg_gap_size': df['gap_size'].mean(),
            'max_gap_size': df['gap_size'].max()
        }
        
        # Gap fill analysis
        if not gap_up_data.empty:
            # Check if gap gets filled (price goes below previous close)
            gap_up_fill = (gap_up_data['Low'] <= gap_up_data['Close'].shift(1)).mean()
            patterns['gap_up_analysis'] = {
                'fill_probability': gap_up_fill,
                'avg_return': gap_up_data['Close'].pct_change().mean(),
                'avg_fill_time': self._calculate_gap_fill_time(gap_up_data, 'up')
            }
        
        if not gap_down_data.empty:
            # Check if gap gets filled (price goes above previous close)
            gap_down_fill = (gap_down_data['High'] >= gap_down_data['Close'].shift(1)).mean()
            patterns['gap_down_analysis'] = {
                'fill_probability': gap_down_fill,
                'avg_return': gap_down_data['Close'].pct_change().mean(),
                'avg_fill_time': self._calculate_gap_fill_time(gap_down_data, 'down')
            }
        
        # Large gap analysis
        large_gaps = df[df['gap_size'] > self.config.GAP_THRESHOLD]
        if not large_gaps.empty:
            patterns['large_gap_analysis'] = {
                'frequency': len(large_gaps) / len(df),
                'avg_return': large_gaps['Close'].pct_change().mean(),
                'fill_probability': self._calculate_large_gap_fill(large_gaps)
            }
        
        return patterns
    
    def _calculate_gap_fill_time(self, gap_data: pd.DataFrame, gap_type: str) -> float:
        """Calculate average time to fill gaps."""
        fill_times = []
        
        for i, (idx, row) in enumerate(gap_data.iterrows()):
            if i == 0:
                continue
            
            # Look ahead for gap fill
            future_data = gap_data.iloc[i:]
            if gap_type == 'up':
                fill_condition = future_data['Low'] <= row['Close']
            else:
                fill_condition = future_data['High'] >= row['Close']
            
            if fill_condition.any():
                fill_time = fill_condition.idxmax() - idx
                fill_times.append(fill_time.days if hasattr(fill_time, 'days') else 1)
        
        return np.mean(fill_times) if fill_times else 0
    
    def _calculate_large_gap_fill(self, large_gaps: pd.DataFrame) -> float:
        """Calculate fill probability for large gaps."""
        fills = 0
        total = len(large_gaps)
        
        for i, (idx, row) in enumerate(large_gaps.iterrows()):
            if i == 0:
                continue
            
            # Check if gap gets filled within 5 days
            future_data = large_gaps.iloc[i:i+5]
            if not future_data.empty:
                if (future_data['Low'].min() <= row['Close'] or 
                    future_data['High'].max() >= row['Close']):
                    fills += 1
        
        return fills / total if total > 0 else 0
    
    def _analyze_expiry_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze F&O expiry day patterns."""
        logger.info("Analyzing expiry patterns...")
        
        patterns = {}
        
        # Expiry day analysis
        expiry_data = df[df['is_expiry_day'] == 1]
        non_expiry_data = df[df['is_expiry_day'] == 0]
        
        if not expiry_data.empty:
            patterns['expiry_day_analysis'] = {
                'avg_return': expiry_data['Close'].pct_change().mean(),
                'volatility': expiry_data['Close'].pct_change().std(),
                'volume_effect': expiry_data['Volume'].mean() if 'Volume' in expiry_data.columns else 0,
                'gap_frequency': expiry_data['gap_up'].mean() + expiry_data['gap_down'].mean()
            }
        
        # Pre-expiry effect (day before expiry)
        pre_expiry_data = df[df['is_expiry_day'].shift(-1) == 1]
        if not pre_expiry_data.empty:
            patterns['pre_expiry_effect'] = {
                'avg_return': pre_expiry_data['Close'].pct_change().mean(),
                'volatility': pre_expiry_data['Close'].pct_change().std(),
                'rollover_effect': pre_expiry_data['Close'].pct_change().mean()
            }
        
        # Post-expiry effect (day after expiry)
        post_expiry_data = df[df['is_expiry_day'].shift(1) == 1]
        if not post_expiry_data.empty:
            patterns['post_expiry_effect'] = {
                'avg_return': post_expiry_data['Close'].pct_change().mean(),
                'volatility': post_expiry_data['Close'].pct_change().std(),
                'settlement_effect': post_expiry_data['Close'].pct_change().mean()
            }
        
        return patterns
    
    def _analyze_special_event_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze special event patterns (earnings, dividends, splits)."""
        logger.info("Analyzing special event patterns...")
        
        patterns = {}
        
        # This is a simplified analysis - in practice, you'd need event data
        # For now, we'll analyze patterns around month-end and quarter-end
        
        # Earnings season effect (simplified - around month-end)
        earnings_data = df[df['day_of_month'].isin([25, 26, 27, 28, 29, 30])]
        if not earnings_data.empty:
            patterns['earnings_season_effect'] = {
                'avg_return': earnings_data['Close'].pct_change().mean(),
                'volatility': earnings_data['Close'].pct_change().std(),
                'volume_effect': earnings_data['Volume'].mean() if 'Volume' in earnings_data.columns else 0
            }
        
        # Dividend effect (simplified - around month-start)
        dividend_data = df[df['day_of_month'].isin([1, 2, 3, 4, 5])]
        if not dividend_data.empty:
            patterns['dividend_effect'] = {
                'avg_return': dividend_data['Close'].pct_change().mean(),
                'volatility': dividend_data['Close'].pct_change().std()
            }
        
        return patterns
    
    def _detect_market_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect market cycles using FFT and statistical analysis."""
        logger.info("Detecting market cycles...")
        
        patterns = {}
        
        # Calculate returns
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < 100:
            logger.warning("Insufficient data for cycle detection")
            return patterns
        
        # FFT analysis
        fft_result = fft(returns.values)
        freqs = fftfreq(len(returns))
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft_result) ** 2
        dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
        
        cycles = []
        for idx in dominant_freq_idx:
            if freqs[idx] != 0:
                period = 1 / abs(freqs[idx])
                if 2 <= period <= len(returns) // 2:  # Reasonable cycle length
                    cycles.append({
                        'period': period,
                        'power': power_spectrum[idx],
                        'frequency': freqs[idx]
                    })
        
        patterns['detected_cycles'] = sorted(cycles, key=lambda x: x['power'], reverse=True)
        
        # Statistical cycle detection
        patterns['statistical_cycles'] = self._statistical_cycle_detection(returns)
        
        return patterns
    
    def _statistical_cycle_detection(self, returns: pd.Series) -> Dict[str, Any]:
        """Detect cycles using statistical methods."""
        cycles = {}
        
        # Autocorrelation analysis
        autocorr = pd.Series(returns).autocorr(lag=1)
        cycles['autocorrelation'] = autocorr
        
        # Rolling correlation
        rolling_corr = returns.rolling(20).corr(returns.shift(5))
        cycles['rolling_correlation'] = {
            'mean': rolling_corr.mean(),
            'std': rolling_corr.std(),
            'current': rolling_corr.iloc[-1] if not rolling_corr.empty else 0
        }
        
        # Volatility clustering
        volatility = returns.rolling(20).std()
        vol_autocorr = volatility.autocorr(lag=1)
        cycles['volatility_clustering'] = vol_autocorr
        
        return cycles
    
    def get_temporal_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate temporal-based predictions.
        
        Args:
            df: Recent market data
        
        Returns:
            Dictionary with temporal predictions
        """
        if df is None or df.empty:
            return {}
        
        # Ensure we have temporal features
        df = self._add_temporal_features(df)
        
        predictions = {
            'optimal_timing': {},
            'cycle_predictions': {},
            'risk_periods': {},
            'opportunity_periods': {}
        }
        
        # Optimal timing based on patterns
        if self.intraday_patterns:
            best_hour = self.intraday_patterns.get('best_performing_hour', {})
            predictions['optimal_timing']['best_hour'] = best_hour.get('hour', 10)
        
        if self.day_of_week_patterns:
            best_day = self.day_of_week_patterns.get('best_performing_day', {})
            predictions['optimal_timing']['best_day'] = best_day.get('day', 'Tuesday')
        
        if self.monthly_patterns:
            best_month = self.monthly_patterns.get('best_performing_month', {})
            predictions['optimal_timing']['best_month'] = best_month.get('month', 'Dec')
        
        # Cycle predictions
        if hasattr(self, 'seasonal_patterns') and self.seasonal_patterns:
            current_quarter = df.index[-1].quarter
            quarterly_returns = self.seasonal_patterns.get('quarterly_returns', {})
            if current_quarter in quarterly_returns:
                predictions['cycle_predictions']['quarterly_expectation'] = quarterly_returns[current_quarter]
        
        # Risk periods
        if self.gap_patterns:
            gap_stats = self.gap_patterns.get('gap_statistics', {})
            predictions['risk_periods']['gap_probability'] = gap_stats.get('gap_up_frequency', 0) + gap_stats.get('gap_down_frequency', 0)
        
        # Opportunity periods
        if self.expiry_patterns:
            expiry_analysis = self.expiry_patterns.get('expiry_day_analysis', {})
            predictions['opportunity_periods']['expiry_volatility'] = expiry_analysis.get('volatility', 0)
        
        return predictions
    
    def visualize_temporal_patterns(self, save_path: str = None):
        """Create visualizations of temporal patterns."""
        if not any([self.intraday_patterns, self.day_of_week_patterns, self.monthly_patterns]):
            logger.warning("No temporal patterns to visualize. Run analyze_temporal_patterns() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Intraday patterns
        if self.intraday_patterns and 'hourly_returns' in self.intraday_patterns:
            hourly_returns = self.intraday_patterns['hourly_returns']
            axes[0, 0].bar(hourly_returns.keys(), hourly_returns.values())
            axes[0, 0].set_title('Intraday Returns by Hour')
            axes[0, 0].set_xlabel('Hour')
            axes[0, 0].set_ylabel('Average Return')
        
        # Day of week patterns
        if self.day_of_week_patterns and 'daily_returns' in self.day_of_week_patterns:
            daily_returns = self.day_of_week_patterns['daily_returns']
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            axes[0, 1].bar(range(len(daily_returns)), daily_returns.values())
            axes[0, 1].set_title('Daily Returns by Day of Week')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Average Return')
            axes[0, 1].set_xticks(range(len(daily_returns)))
            axes[0, 1].set_xticklabels(day_names)
        
        # Monthly patterns
        if self.monthly_patterns and 'monthly_returns' in self.monthly_patterns:
            monthly_returns = self.monthly_patterns['monthly_returns']
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[1, 0].bar(range(len(monthly_returns)), monthly_returns.values())
            axes[1, 0].set_title('Monthly Returns')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Average Return')
            axes[1, 0].set_xticks(range(len(monthly_returns)))
            axes[1, 0].set_xticklabels(month_names)
        
        # Gap analysis
        if self.gap_patterns and 'gap_statistics' in self.gap_patterns:
            gap_stats = self.gap_patterns['gap_statistics']
            gap_types = ['Gap Up', 'Gap Down']
            gap_freqs = [gap_stats.get('gap_up_frequency', 0), gap_stats.get('gap_down_frequency', 0)]
            axes[1, 1].bar(gap_types, gap_freqs)
            axes[1, 1].set_title('Gap Frequency')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Temporal patterns visualization saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the temporal analyzer
    from data.live_data_loader import DataLoader
    
    print("Testing Temporal Analyzer...")
    
    # Load sample data
    loader = DataLoader()
    data = loader.fetch_single_timeframe_data(timeframe="1d")
    
    if data is not None and not data.empty:
        # Initialize temporal analyzer
        analyzer = TemporalAnalyzer()
        
        # Analyze temporal patterns
        patterns = analyzer.analyze_temporal_patterns(data)
        
        print(f"Temporal patterns analyzed: {len(patterns)} categories")
        
        # Get temporal predictions
        predictions = analyzer.get_temporal_predictions(data)
        print(f"Temporal predictions: {predictions}")
        
        # Visualize patterns
        analyzer.visualize_temporal_patterns()
        
    else:
        print("No data available for testing")
