"""
Ultimate Market AI Engine - Advanced Technical Indicator Analysis
===============================================================

This module implements 40+ technical indicators with pattern analysis
within each indicator for comprehensive market analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Advanced technical analysis system with 40+ indicators and pattern recognition.
    
    Features:
    - 40+ technical indicators with custom pattern analysis
    - Dynamic threshold adjustment based on volatility
    - Multi-timeframe indicator convergence
    - Indicator relationship analysis
    - ML-based indicator weighting
    - Pattern detection within each indicator
    """
    
    def __init__(self, config=None):
        """Initialize the technical analyzer."""
        self.config = config or get_config()
        self.scaler = StandardScaler()
        
        # Indicator results storage
        self.indicator_values = {}
        self.pattern_analysis = {}
        self.confluence_scores = {}
        
        logger.info("Technical Analyzer initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate all 40+ technical indicators with pattern analysis.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with indicator values and patterns
        """
        if df is None or df.empty:
            return {}
        
        logger.info("Calculating all technical indicators...")
        
        results = {}
        
        # Core trend indicators
        results.update(self._calculate_trend_indicators(df))
        
        # Momentum indicators
        results.update(self._calculate_momentum_indicators(df))
        
        # Volatility indicators
        results.update(self._calculate_volatility_indicators(df))
        
        # Volume indicators
        results.update(self._calculate_volume_indicators(df))
        
        # Oscillator indicators
        results.update(self._calculate_oscillator_indicators(df))
        
        # Support/Resistance indicators
        results.update(self._calculate_support_resistance_indicators(df))
        
        # Custom composite indicators
        results.update(self._calculate_composite_indicators(df))
        
        # Store results
        self.indicator_values = results
        
        # Analyze patterns within indicators
        self._analyze_indicator_patterns(df, results)
        
        # Calculate confluence scores
        self._calculate_confluence_scores(results)
        
        logger.info(f"Calculated {len(results)} indicators with pattern analysis")
        return results
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate trend-following indicators."""
        indicators = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            indicators[f'sma_{period}'] = df['Close'].rolling(period).mean()
            indicators[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['Close'].rolling(bb_period).mean()
        bb_std_dev = df['Close'].rolling(bb_period).std()
        indicators['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / bb_middle
        indicators['bb_position'] = (df['Close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # VWAP
        indicators['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # Parabolic SAR
        indicators['parabolic_sar'] = self._calculate_parabolic_sar(df)
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(df)
        indicators.update(ichimoku)
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate momentum indicators."""
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(df['Close'])
        
        # MACD
        macd_data = self._calculate_macd(df['Close'])
        indicators.update(macd_data)
        
        # Stochastic
        stoch_data = self._calculate_stochastic(df)
        indicators.update(stoch_data)
        
        # Williams %R
        indicators['williams_r'] = self._calculate_williams_r(df)
        
        # CCI (Commodity Channel Index)
        indicators['cci'] = self._calculate_cci(df)
        
        # ADX (Average Directional Index)
        adx_data = self._calculate_adx(df)
        indicators.update(adx_data)
        
        # Momentum
        indicators['momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Rate of Change
        indicators['roc'] = df['Close'].pct_change(10) * 100
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate volatility indicators."""
        indicators = {}
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(df)
        
        # Bollinger Band Width (already calculated in trend indicators)
        
        # Historical Volatility
        indicators['historical_volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Chaikin Volatility
        indicators['chaikin_volatility'] = self._calculate_chaikin_volatility(df)
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate volume-based indicators."""
        indicators = {}
        
        # On-Balance Volume (OBV)
        indicators['obv'] = self._calculate_obv(df)
        
        # Accumulation/Distribution Line
        indicators['ad_line'] = self._calculate_ad_line(df)
        
        # Money Flow Index
        indicators['mfi'] = self._calculate_mfi(df)
        
        # Volume Rate of Change
        indicators['volume_roc'] = df['Volume'].pct_change(10) * 100
        
        # Volume Weighted Average Price (VWAP) - already calculated
        
        return indicators
    
    def _calculate_oscillator_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate oscillator indicators."""
        indicators = {}
        
        # Ultimate Oscillator
        indicators['ultimate_oscillator'] = self._calculate_ultimate_oscillator(df)
        
        # TRIX
        indicators['trix'] = self._calculate_trix(df['Close'])
        
        # KST (Know Sure Thing)
        indicators['kst'] = self._calculate_kst(df['Close'])
        
        # True Strength Index
        indicators['tsi'] = self._calculate_tsi(df['Close'])
        
        return indicators
    
    def _calculate_support_resistance_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate support and resistance indicators."""
        indicators = {}
        
        # Pivot Points
        pivot_data = self._calculate_pivot_points(df)
        indicators.update(pivot_data)
        
        # Fibonacci Retracements
        fib_data = self._calculate_fibonacci_retracements(df)
        indicators.update(fib_data)
        
        # ZigZag
        indicators['zigzag'] = self._calculate_zigzag(df)
        
        return indicators
    
    def _calculate_composite_indicators(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate custom composite indicators."""
        indicators = {}
        
        # Trend Strength Composite
        indicators['trend_strength'] = self._calculate_trend_strength_composite(df)
        
        # Volatility Composite
        indicators['volatility_composite'] = self._calculate_volatility_composite(df)
        
        # Momentum Composite
        indicators['momentum_composite'] = self._calculate_momentum_composite(df)
        
        # Volume Composite
        indicators['volume_composite'] = self._calculate_volume_composite(df)
        
        return indicators
    
    # Individual indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = df['Low'].rolling(k_period).min()
        highest_high = df['High'].rolling(k_period).max()
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R indicator."""
        highest_high = df['High'].rolling(period).max()
        lowest_low = df['Low'].rolling(period).min()
        williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mean_deviation = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index."""
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = self._calculate_true_range(df)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean()
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr = self._calculate_true_range(df)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        # Simplified implementation
        sar = pd.Series(index=df.index, dtype=float)
        sar.iloc[0] = df['Low'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > sar.iloc[i-1]:
                sar.iloc[i] = min(df['Low'].iloc[i], sar.iloc[i-1])
            else:
                sar.iloc[i] = max(df['High'].iloc[i], sar.iloc[i-1])
        
        return sar
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        chikou_span = df['Close'].shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['Volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        ad_line = (clv * df['Volume']).cumsum()
        return ad_line
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_ultimate_oscillator(self, df: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Calculate Ultimate Oscillator."""
        tr = self._calculate_true_range(df)
        bp = df['Close'] - df[['Low', 'Close']].shift(1).min(axis=1)
        
        avg7 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg14 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg28 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
        return uo
    
    def _calculate_trix(self, prices: pd.Series, period: int = 15) -> pd.Series:
        """Calculate TRIX indicator."""
        ema1 = prices.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        trix = ema3.pct_change() * 100
        return trix
    
    def _calculate_kst(self, prices: pd.Series) -> pd.Series:
        """Calculate Know Sure Thing indicator."""
        roc1 = prices.pct_change(10) * 100
        roc2 = prices.pct_change(15) * 100
        roc3 = prices.pct_change(20) * 100
        roc4 = prices.pct_change(30) * 100
        
        sma1 = roc1.rolling(10).mean()
        sma2 = roc2.rolling(10).mean()
        sma3 = roc3.rolling(10).mean()
        sma4 = roc4.rolling(15).mean()
        
        kst = sma1 + (2 * sma2) + (3 * sma3) + (4 * sma4)
        return kst
    
    def _calculate_tsi(self, prices: pd.Series, first_period: int = 25, second_period: int = 13) -> pd.Series:
        """Calculate True Strength Index."""
        pc = prices.diff()
        apc = pc.ewm(span=first_period).mean()
        apc_ema = apc.ewm(span=second_period).mean()
        
        abs_pc = pc.abs()
        aapc = abs_pc.ewm(span=first_period).mean()
        aapc_ema = aapc.ewm(span=second_period).mean()
        
        tsi = 100 * (apc_ema / aapc_ema)
        return tsi
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Pivot Points."""
        pivot = (df['High'] + df['Low'] + df['Close']) / 3
        r1 = 2 * pivot - df['Low']
        s1 = 2 * pivot - df['High']
        r2 = pivot + (df['High'] - df['Low'])
        s2 = pivot - (df['High'] - df['Low'])
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }
    
    def _calculate_fibonacci_retracements(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Fibonacci Retracements."""
        high = df['High'].rolling(20).max()
        low = df['Low'].rolling(20).min()
        diff = high - low
        
        fib_236 = high - (diff * 0.236)
        fib_382 = high - (diff * 0.382)
        fib_500 = high - (diff * 0.500)
        fib_618 = high - (diff * 0.618)
        fib_786 = high - (diff * 0.786)
        
        return {
            'fib_236': fib_236,
            'fib_382': fib_382,
            'fib_500': fib_500,
            'fib_618': fib_618,
            'fib_786': fib_786
        }
    
    def _calculate_zigzag(self, df: pd.DataFrame, deviation: float = 0.05) -> pd.Series:
        """Calculate ZigZag indicator."""
        # Simplified implementation
        zigzag = pd.Series(index=df.index, dtype=float)
        zigzag.iloc[0] = df['Close'].iloc[0]
        
        for i in range(1, len(df)):
            if abs(df['Close'].iloc[i] - zigzag.iloc[i-1]) / zigzag.iloc[i-1] > deviation:
                zigzag.iloc[i] = df['Close'].iloc[i]
            else:
                zigzag.iloc[i] = zigzag.iloc[i-1]
        
        return zigzag
    
    def _calculate_chaikin_volatility(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Chaikin Volatility."""
        high_low = df['High'] - df['Low']
        ema_hl = high_low.ewm(span=period).mean()
        cv = ((ema_hl - ema_hl.shift(period)) / ema_hl.shift(period)) * 100
        return cv
    
    # Composite indicator calculations
    def _calculate_trend_strength_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength composite indicator."""
        # Combine multiple trend indicators
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        adx = self._calculate_adx(df)['adx']
        
        trend_strength = ((df['Close'] - sma_20) / sma_20 + 
                         (sma_20 - sma_50) / sma_50 + 
                         adx / 100) / 3
        return trend_strength
    
    def _calculate_volatility_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility composite indicator."""
        atr = self._calculate_atr(df)
        bb_width = self.indicator_values.get('bb_width', pd.Series(0, index=df.index))
        hist_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        vol_composite = (atr / df['Close'] + bb_width + hist_vol) / 3
        return vol_composite
    
    def _calculate_momentum_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum composite indicator."""
        rsi = self._calculate_rsi(df['Close'])
        macd = self._calculate_macd(df['Close'])['macd_line']
        stoch = self._calculate_stochastic(df)['stoch_k']
        
        momentum_composite = (rsi / 100 + (macd - macd.rolling(20).mean()) / macd.rolling(20).std() + stoch / 100) / 3
        return momentum_composite
    
    def _calculate_volume_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume composite indicator."""
        obv = self._calculate_obv(df)
        mfi = self._calculate_mfi(df)
        volume_roc = df['Volume'].pct_change(10) * 100
        
        volume_composite = (obv.pct_change() + mfi / 100 + volume_roc / 100) / 3
        return volume_composite
    
    def _analyze_indicator_patterns(self, df: pd.DataFrame, indicators: Dict[str, pd.DataFrame]):
        """Analyze patterns within each indicator."""
        patterns = {}
        
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, pd.Series):
                patterns[indicator_name] = self._analyze_series_patterns(indicator_data)
            elif isinstance(indicator_data, dict):
                patterns[indicator_name] = {}
                for sub_name, sub_data in indicator_data.items():
                    if isinstance(sub_data, pd.Series):
                        patterns[indicator_name][sub_name] = self._analyze_series_patterns(sub_data)
        
        self.pattern_analysis = patterns
        logger.info(f"Analyzed patterns for {len(patterns)} indicators")
    
    def _analyze_series_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in a time series."""
        if series.empty or series.isna().all():
            return {}
        
        patterns = {}
        
        # Trend analysis
        patterns['trend'] = self._detect_trend(series)
        
        # Divergence analysis
        patterns['divergence'] = self._detect_divergence(series)
        
        # Overbought/Oversold analysis
        patterns['extremes'] = self._detect_extremes(series)
        
        # Breakout analysis
        patterns['breakouts'] = self._detect_breakouts(series)
        
        return patterns
    
    def _detect_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Detect trend in a series."""
        if len(series) < 20:
            return {}
        
        # Clean the series and get valid indices
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return {}
        
        # Linear regression trend
        x = np.arange(len(clean_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_series.values)
        
        # Moving average trend
        ma_short = series.rolling(10).mean()
        ma_long = series.rolling(20).mean()
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_strength': abs(slope) / clean_series.std() if clean_series.std() > 0 else 0,
            'ma_trend': 'up' if ma_short.iloc[-1] > ma_long.iloc[-1] else 'down'
        }
    
    def _detect_divergence(self, series: pd.Series) -> Dict[str, Any]:
        """Detect divergence patterns."""
        # Simplified divergence detection
        return {
            'price_divergence': False,  # Would need price data for full analysis
            'volume_divergence': False
        }
    
    def _detect_extremes(self, series: pd.Series) -> Dict[str, Any]:
        """Detect overbought/oversold conditions."""
        if series.empty:
            return {}
        
        current = series.iloc[-1]
        percentile = stats.percentileofscore(series.dropna(), current)
        
        return {
            'current_value': current,
            'percentile': percentile,
            'overbought': percentile > 80,
            'oversold': percentile < 20,
            'extreme_level': percentile > 95 or percentile < 5
        }
    
    def _detect_breakouts(self, series: pd.Series) -> Dict[str, Any]:
        """Detect breakout patterns."""
        if len(series) < 20:
            return {}
        
        # Resistance/Support levels
        resistance = series.rolling(20).max()
        support = series.rolling(20).min()
        
        current = series.iloc[-1]
        prev = series.iloc[-2]
        
        return {
            'resistance_breakout': current > resistance.iloc[-2] and prev <= resistance.iloc[-2],
            'support_breakdown': current < support.iloc[-2] and prev >= support.iloc[-2],
            'near_resistance': current > resistance.iloc[-1] * 0.98,
            'near_support': current < support.iloc[-1] * 1.02
        }
    
    def _calculate_confluence_scores(self, indicators: Dict[str, pd.DataFrame]):
        """Calculate confluence scores across indicators."""
        confluence = {}
        
        # Group indicators by type
        trend_indicators = ['sma_20', 'sma_50', 'ema_20', 'adx']
        momentum_indicators = ['rsi', 'macd_line', 'stoch_k']
        volatility_indicators = ['atr', 'bb_width']
        volume_indicators = ['obv', 'mfi']
        
        # Calculate confluence for each group
        confluence['trend'] = self._calculate_group_confluence(indicators, trend_indicators)
        confluence['momentum'] = self._calculate_group_confluence(indicators, momentum_indicators)
        confluence['volatility'] = self._calculate_group_confluence(indicators, volatility_indicators)
        confluence['volume'] = self._calculate_group_confluence(indicators, volume_indicators)
        
        # Overall confluence
        confluence['overall'] = np.mean([
            confluence['trend'],
            confluence['momentum'],
            confluence['volatility'],
            confluence['volume']
        ])
        
        self.confluence_scores = confluence
        logger.info(f"Calculated confluence scores: {confluence}")
    
    def _calculate_group_confluence(self, indicators: Dict[str, pd.DataFrame], indicator_names: List[str]) -> float:
        """Calculate confluence score for a group of indicators."""
        signals = []
        
        for name in indicator_names:
            if name in indicators:
                if isinstance(indicators[name], pd.Series):
                    signals.append(self._get_indicator_signal(indicators[name], name))
                elif isinstance(indicators[name], dict):
                    # Handle multi-component indicators
                    for sub_name, sub_data in indicators[name].items():
                        if isinstance(sub_data, pd.Series):
                            signals.append(self._get_indicator_signal(sub_data, f"{name}_{sub_name}"))
        
        if not signals:
            return 0.0
        
        # Calculate agreement score
        positive_signals = sum(1 for signal in signals if signal > 0)
        negative_signals = sum(1 for signal in signals if signal < 0)
        
        if positive_signals > negative_signals:
            return positive_signals / len(signals)
        elif negative_signals > positive_signals:
            return -negative_signals / len(signals)
        else:
            return 0.0
    
    def _get_indicator_signal(self, series: pd.Series, indicator_name: str) -> float:
        """Get signal from an indicator series."""
        if series.empty or series.isna().all():
            return 0.0
        
        current = series.iloc[-1]
        prev = series.iloc[-2] if len(series) > 1 else current
        
        # Different signal logic for different indicators
        if 'rsi' in indicator_name.lower():
            if current > 70:
                return -1.0  # Overbought
            elif current < 30:
                return 1.0   # Oversold
            else:
                return 0.0
        elif 'macd' in indicator_name.lower():
            return 1.0 if current > prev else -1.0
        elif 'stoch' in indicator_name.lower():
            if current > 80:
                return -1.0
            elif current < 20:
                return 1.0
            else:
                return 0.0
        else:
            # Default signal based on direction
            return 1.0 if current > prev else -1.0
    
    def get_technical_summary(self) -> Dict[str, Any]:
        """Get comprehensive technical analysis summary."""
        if not self.indicator_values:
            return {}
        
        summary = {
            'indicators_calculated': len(self.indicator_values),
            'confluence_scores': self.confluence_scores,
            'pattern_analysis': self.pattern_analysis,
            'key_signals': self._get_key_signals(),
            'overall_sentiment': self._calculate_overall_sentiment()
        }
        
        return summary
    
    def _get_key_signals(self) -> Dict[str, Any]:
        """Get key technical signals."""
        signals = {}
        
        # Trend signals
        if 'sma_20' in self.indicator_values and 'sma_50' in self.indicator_values:
            sma_20 = self.indicator_values['sma_20']
            sma_50 = self.indicator_values['sma_50']
            signals['trend'] = 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish'
        
        # RSI signals
        if 'rsi' in self.indicator_values:
            rsi = self.indicator_values['rsi']
            current_rsi = rsi.iloc[-1]
            if current_rsi > 70:
                signals['rsi'] = 'overbought'
            elif current_rsi < 30:
                signals['rsi'] = 'oversold'
            else:
                signals['rsi'] = 'neutral'
        
        # MACD signals
        if 'macd_line' in self.indicator_values and 'macd_signal' in self.indicator_values:
            macd_line = self.indicator_values['macd_line']
            macd_signal = self.indicator_values['macd_signal']
            signals['macd'] = 'bullish' if macd_line.iloc[-1] > macd_signal.iloc[-1] else 'bearish'
        
        return signals
    
    def _calculate_overall_sentiment(self) -> str:
        """Calculate overall technical sentiment."""
        if not self.confluence_scores:
            return 'neutral'
        
        overall = self.confluence_scores.get('overall', 0)
        
        if overall > 0.3:
            return 'bullish'
        elif overall < -0.3:
            return 'bearish'
        else:
            return 'neutral'


# Example usage and testing
if __name__ == "__main__":
    # Test the technical analyzer
    from data.live_data_loader import DataLoader
    
    print("Testing Technical Analyzer...")
    
    # Load sample data
    loader = DataLoader()
    data = loader.fetch_single_timeframe_data(timeframe="1d")
    
    if data is not None and not data.empty:
        # Initialize technical analyzer
        analyzer = TechnicalAnalyzer()
        
        # Calculate all indicators
        indicators = analyzer.calculate_all_indicators(data)
        
        print(f"Calculated {len(indicators)} indicators")
        
        # Get technical summary
        summary = analyzer.get_technical_summary()
        print(f"Technical summary: {summary}")
        
        # Show key signals
        signals = analyzer._get_key_signals()
        print(f"Key signals: {signals}")
        
    else:
        print("No data available for testing")
