#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - BREAKOUT DETECTION SYSTEM
Real vs Fake Breakout Detection with Multi-timeframe Analysis
Production-Grade Implementation with Advanced Criteria
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreakoutDetection:
    """Comprehensive Breakout Detection System"""
    
    def __init__(self):
        self.support_resistance_levels = {}
        self.breakout_signals = {}
        self.ml_model = None
        self.scaler = StandardScaler()
        
        logger.info("Breakout Detection System initialized")
    
    def detect_breakouts(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Detect breakouts in the data"""
        logger.info(f"Detecting breakouts for {timeframe}")
        
        breakouts = {
            'timeframe': timeframe,
            'support_resistance': {},
            'breakouts': [],
            'fake_breakouts': [],
            'summary': {
                'total_breakouts': 0,
                'real_breakouts': 0,
                'fake_breakouts': 0,
                'confidence': 0.0
            }
        }
        
        if len(data) < 50:
            logger.warning(f"Insufficient data for breakout detection: {len(data)} records")
            return breakouts
        
        try:
            # Calculate support and resistance levels
            support_resistance = self._calculate_support_resistance(data)
            breakouts['support_resistance'] = support_resistance
            
            # Detect breakout candidates
            breakout_candidates = self._detect_breakout_candidates(data, support_resistance)
            
            # Analyze each candidate
            for candidate in breakout_candidates:
                analysis = self._analyze_breakout_candidate(data, candidate, support_resistance)
                
                if analysis['is_real_breakout']:
                    breakouts['breakouts'].append(analysis)
                    breakouts['summary']['real_breakouts'] += 1
                else:
                    breakouts['fake_breakouts'].append(analysis)
                    breakouts['summary']['fake_breakouts'] += 1
                
                breakouts['summary']['total_breakouts'] += 1
            
            # Calculate overall confidence
            if breakouts['summary']['total_breakouts'] > 0:
                breakouts['summary']['confidence'] = breakouts['summary']['real_breakouts'] / breakouts['summary']['total_breakouts']
            
            logger.info(f"✅ {timeframe}: {breakouts['summary']['real_breakouts']} real, {breakouts['summary']['fake_breakouts']} fake breakouts")
            
        except Exception as e:
            logger.error(f"❌ Error in breakout detection: {str(e)}")
        
        return breakouts
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            df = data.copy()
            
            # Calculate moving averages
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            
            # Calculate pivot points
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
            df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
            
            # Find recent highs and lows
            recent_highs = self._find_recent_highs(df, window=20)
            recent_lows = self._find_recent_lows(df, window=20)
            
            # Calculate volume-weighted average price (VWAP)
            df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            
            # Identify key levels
            key_levels = {
                'support': {
                    'recent_lows': recent_lows,
                    'sma_200': df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else None,
                    'sma_50': df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else None,
                    's1': df['S1'].iloc[-1] if not pd.isna(df['S1'].iloc[-1]) else None,
                    's2': df['S2'].iloc[-1] if not pd.isna(df['S2'].iloc[-1]) else None,
                    'vwap': df['VWAP'].iloc[-1] if not pd.isna(df['VWAP'].iloc[-1]) else None
                },
                'resistance': {
                    'recent_highs': recent_highs,
                    'sma_200': df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else None,
                    'sma_50': df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else None,
                    'r1': df['R1'].iloc[-1] if not pd.isna(df['R1'].iloc[-1]) else None,
                    'r2': df['R2'].iloc[-1] if not pd.isna(df['R2'].iloc[-1]) else None,
                    'vwap': df['VWAP'].iloc[-1] if not pd.isna(df['VWAP'].iloc[-1]) else None
                }
            }
            
            return key_levels
            
        except Exception as e:
            logger.error(f"❌ Error calculating support/resistance: {str(e)}")
            return {}
    
    def _find_recent_highs(self, data: pd.DataFrame, window: int = 20) -> List[float]:
        """Find recent highs"""
        try:
            highs = []
            for i in range(window, len(data)):
                window_data = data.iloc[i-window:i]
                if data.iloc[i]['High'] == window_data['High'].max():
                    highs.append(data.iloc[i]['High'])
            return highs[-5:] if len(highs) > 5 else highs  # Return last 5 highs
        except Exception as e:
            logger.error(f"❌ Error finding recent highs: {str(e)}")
            return []
    
    def _find_recent_lows(self, data: pd.DataFrame, window: int = 20) -> List[float]:
        """Find recent lows"""
        try:
            lows = []
            for i in range(window, len(data)):
                window_data = data.iloc[i-window:i]
                if data.iloc[i]['Low'] == window_data['Low'].min():
                    lows.append(data.iloc[i]['Low'])
            return lows[-5:] if len(lows) > 5 else lows  # Return last 5 lows
        except Exception as e:
            logger.error(f"❌ Error finding recent lows: {str(e)}")
            return []
    
    def _detect_breakout_candidates(self, data: pd.DataFrame, support_resistance: Dict) -> List[Dict]:
        """Detect potential breakout candidates"""
        candidates = []
        
        try:
            df = data.copy()
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            current_high = df['High'].iloc[-1]
            current_low = df['Low'].iloc[-1]
            
            # Check resistance breakouts
            resistance_levels = support_resistance.get('resistance', {})
            for level_name, level_value in resistance_levels.items():
                if level_value is not None and current_high > level_value:
                    candidates.append({
                        'type': 'resistance_breakout',
                        'level_name': level_name,
                        'level_value': level_value,
                        'breakout_price': current_high,
                        'date': df.index[-1],
                        'strength': (current_high - level_value) / level_value
                    })
            
            # Check support breakouts (downside)
            support_levels = support_resistance.get('support', {})
            for level_name, level_value in support_levels.items():
                if level_value is not None and current_low < level_value:
                    candidates.append({
                        'type': 'support_breakout',
                        'level_name': level_name,
                        'level_value': level_value,
                        'breakout_price': current_low,
                        'date': df.index[-1],
                        'strength': (level_value - current_low) / level_value
                    })
            
            # Check for consolidation breakouts
            consolidation_breakout = self._detect_consolidation_breakout(df)
            if consolidation_breakout:
                candidates.append(consolidation_breakout)
            
        except Exception as e:
            logger.error(f"❌ Error detecting breakout candidates: {str(e)}")
        
        return candidates
    
    def _detect_consolidation_breakout(self, data: pd.DataFrame) -> Optional[Dict]:
        """Detect consolidation breakouts"""
        try:
            df = data.copy()
            
            # Calculate recent volatility
            recent_volatility = df['Close'].pct_change().rolling(20).std()
            avg_volatility = recent_volatility.mean()
            
            # Check if recent volatility is low (consolidation)
            if recent_volatility.iloc[-1] < avg_volatility * 0.5:
                # Check for breakout from consolidation range
                recent_high = df['High'].rolling(20).max().iloc[-1]
                recent_low = df['Low'].rolling(20).min().iloc[-1]
                consolidation_range = recent_high - recent_low
                
                current_price = df['Close'].iloc[-1]
                
                if current_price > recent_high:
                    return {
                        'type': 'consolidation_breakout',
                        'level_name': 'consolidation_high',
                        'level_value': recent_high,
                        'breakout_price': current_price,
                        'date': df.index[-1],
                        'strength': (current_price - recent_high) / consolidation_range
                    }
                elif current_price < recent_low:
                    return {
                        'type': 'consolidation_breakout',
                        'level_name': 'consolidation_low',
                        'level_value': recent_low,
                        'breakout_price': current_price,
                        'date': df.index[-1],
                        'strength': (recent_low - current_price) / consolidation_range
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detecting consolidation breakout: {str(e)}")
            return None
    
    def _analyze_breakout_candidate(self, data: pd.DataFrame, candidate: Dict, support_resistance: Dict) -> Dict:
        """Analyze a breakout candidate to determine if it's real or fake"""
        try:
            df = data.copy()
            
            # Get recent data for analysis
            recent_data = df.tail(10)  # Last 10 periods
            
            # Calculate analysis metrics
            volume_analysis = self._analyze_volume_confirmation(recent_data)
            price_action_analysis = self._analyze_price_action(recent_data, candidate)
            momentum_analysis = self._analyze_momentum(recent_data)
            time_confirmation = self._analyze_time_confirmation(recent_data, candidate)
            
            # Calculate overall score
            score = self._calculate_breakout_score(volume_analysis, price_action_analysis, 
                                                 momentum_analysis, time_confirmation)
            
            # Determine if it's a real breakout
            is_real_breakout = score > 0.7  # Threshold for real breakout
            
            analysis = {
                'candidate': candidate,
                'volume_analysis': volume_analysis,
                'price_action_analysis': price_action_analysis,
                'momentum_analysis': momentum_analysis,
                'time_confirmation': time_confirmation,
                'overall_score': score,
                'is_real_breakout': is_real_breakout,
                'confidence': min(0.95, score),
                'recommendation': self._generate_breakout_recommendation(score, candidate)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing breakout candidate: {str(e)}")
            return {
                'candidate': candidate,
                'is_real_breakout': False,
                'confidence': 0.0,
                'recommendation': 'Analysis failed'
            }
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame) -> Dict:
        """Analyze volume confirmation for breakout"""
        try:
            # Calculate volume metrics
            avg_volume = data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Volume confirmation criteria
            volume_confirmed = volume_ratio > 1.5  # 150% above average
            volume_score = min(1.0, volume_ratio / 2.0)  # Normalize to 0-1
            
            return {
                'volume_ratio': volume_ratio,
                'volume_confirmed': volume_confirmed,
                'volume_score': volume_score,
                'avg_volume': avg_volume,
                'current_volume': current_volume
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing volume: {str(e)}")
            return {'volume_confirmed': False, 'volume_score': 0.0}
    
    def _analyze_price_action(self, data: pd.DataFrame, candidate: Dict) -> Dict:
        """Analyze price action for breakout confirmation"""
        try:
            # Get recent price action
            recent_closes = data['Close'].tail(3)
            recent_highs = data['High'].tail(3)
            recent_lows = data['Low'].tail(3)
            
            # Check for clean breakout (no long wicks)
            current_candle = data.iloc[-1]
            body_size = abs(current_candle['Close'] - current_candle['Open'])
            total_range = current_candle['High'] - current_candle['Low']
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            # Price action criteria
            clean_breakout = body_ratio > 0.6  # Strong body
            consecutive_closes = len([c for c in recent_closes if c > candidate['level_value']]) if candidate['type'] == 'resistance_breakout' else len([c for c in recent_closes if c < candidate['level_value']])
            price_confirmed = consecutive_closes >= 2  # At least 2 closes beyond level
            
            price_score = (body_ratio * 0.5 + (consecutive_closes / 3) * 0.5)
            
            return {
                'clean_breakout': clean_breakout,
                'price_confirmed': price_confirmed,
                'body_ratio': body_ratio,
                'consecutive_closes': consecutive_closes,
                'price_score': price_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing price action: {str(e)}")
            return {'price_confirmed': False, 'price_score': 0.0}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators for breakout confirmation"""
        try:
            df = data.copy()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            # Momentum criteria
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
            current_signal = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
            
            # RSI confirmation
            rsi_bullish = current_rsi > 50 and current_rsi < 70
            rsi_bearish = current_rsi < 50 and current_rsi > 30
            
            # MACD confirmation
            macd_bullish = current_macd > current_signal and current_macd > 0
            macd_bearish = current_macd < current_signal and current_macd < 0
            
            # Overall momentum score
            momentum_score = 0.0
            if rsi_bullish and macd_bullish:
                momentum_score = 0.8
            elif rsi_bearish and macd_bearish:
                momentum_score = 0.8
            elif rsi_bullish or macd_bullish:
                momentum_score = 0.5
            elif rsi_bearish or macd_bearish:
                momentum_score = 0.5
            
            return {
                'rsi': current_rsi,
                'macd': current_macd,
                'rsi_bullish': rsi_bullish,
                'rsi_bearish': rsi_bearish,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing momentum: {str(e)}")
            return {'momentum_score': 0.0}
    
    def _analyze_time_confirmation(self, data: pd.DataFrame, candidate: Dict) -> Dict:
        """Analyze time confirmation for breakout"""
        try:
            # Check if breakout has held for multiple periods
            breakout_level = candidate['level_value']
            breakout_type = candidate['type']
            
            # Count periods since breakout
            periods_held = 0
            for i in range(len(data) - 1, -1, -1):
                if breakout_type == 'resistance_breakout':
                    if data.iloc[i]['Close'] > breakout_level:
                        periods_held += 1
                    else:
                        break
                else:  # support_breakout
                    if data.iloc[i]['Close'] < breakout_level:
                        periods_held += 1
                    else:
                        break
            
            # Time confirmation criteria
            time_confirmed = periods_held >= 2  # At least 2 periods
            time_score = min(1.0, periods_held / 5.0)  # Normalize to 0-1
            
            return {
                'periods_held': periods_held,
                'time_confirmed': time_confirmed,
                'time_score': time_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing time confirmation: {str(e)}")
            return {'time_confirmed': False, 'time_score': 0.0}
    
    def _calculate_breakout_score(self, volume_analysis: Dict, price_action_analysis: Dict, 
                                 momentum_analysis: Dict, time_confirmation: Dict) -> float:
        """Calculate overall breakout score"""
        try:
            # Weighted scoring
            volume_weight = 0.3
            price_weight = 0.3
            momentum_weight = 0.2
            time_weight = 0.2
            
            volume_score = volume_analysis.get('volume_score', 0.0)
            price_score = price_action_analysis.get('price_score', 0.0)
            momentum_score = momentum_analysis.get('momentum_score', 0.0)
            time_score = time_confirmation.get('time_score', 0.0)
            
            # Calculate weighted score
            overall_score = (volume_score * volume_weight + 
                           price_score * price_weight + 
                           momentum_score * momentum_weight + 
                           time_score * time_weight)
            
            return round(overall_score, 3)
            
        except Exception as e:
            logger.error(f"❌ Error calculating breakout score: {str(e)}")
            return 0.0
    
    def _generate_breakout_recommendation(self, score: float, candidate: Dict) -> str:
        """Generate breakout recommendation"""
        try:
            if score > 0.8:
                return f"STRONG {candidate['type'].upper()} - High confidence breakout detected"
            elif score > 0.6:
                return f"MODERATE {candidate['type'].upper()} - Medium confidence breakout"
            elif score > 0.4:
                return f"WEAK {candidate['type'].upper()} - Low confidence, monitor closely"
            else:
                return f"FAKE {candidate['type'].upper()} - Likely false breakout"
                
        except Exception as e:
            logger.error(f"❌ Error generating recommendation: {str(e)}")
            return "Analysis failed"
    
    def analyze_multi_timeframe_breakouts(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze breakouts across multiple timeframes"""
        logger.info("Analyzing breakouts across multiple timeframes")
        
        multi_breakouts = {
            'timeframe_breakouts': {},
            'confluence_analysis': {},
            'summary': {
                'total_breakouts': 0,
                'real_breakouts': 0,
                'fake_breakouts': 0,
                'confluence_signals': 0
            }
        }
        
        # Analyze each timeframe
        for timeframe, data in multi_data.items():
            if data is not None and len(data) > 0:
                breakouts = self.detect_breakouts(data, timeframe)
                multi_breakouts['timeframe_breakouts'][timeframe] = breakouts
                multi_breakouts['summary']['total_breakouts'] += breakouts['summary']['total_breakouts']
                multi_breakouts['summary']['real_breakouts'] += breakouts['summary']['real_breakouts']
                multi_breakouts['summary']['fake_breakouts'] += breakouts['summary']['fake_breakouts']
        
        # Analyze confluence across timeframes
        confluence = self._analyze_breakout_confluence(multi_breakouts)
        multi_breakouts['confluence_analysis'] = confluence
        multi_breakouts['summary']['confluence_signals'] = len(confluence['signals'])
        
        logger.info(f"✅ Multi-timeframe breakout analysis complete: {multi_breakouts['summary']['real_breakouts']} real, {multi_breakouts['summary']['fake_breakouts']} fake breakouts")
        
        return multi_breakouts
    
    def _analyze_breakout_confluence(self, multi_breakouts: Dict) -> Dict:
        """Analyze breakout confluence across timeframes"""
        confluence = {
            'signals': [],
            'strength': 0,
            'recommendations': []
        }
        
        try:
            # Collect all breakouts by date
            all_breakouts = []
            
            for timeframe, breakouts in multi_breakouts['timeframe_breakouts'].items():
                for breakout in breakouts['breakouts']:
                    breakout['timeframe'] = timeframe
                    all_breakouts.append(breakout)
            
            # Group breakouts by date
            from collections import defaultdict
            breakouts_by_date = defaultdict(list)
            
            for breakout in all_breakouts:
                date_key = breakout['candidate']['date'].strftime('%Y-%m-%d')
                breakouts_by_date[date_key].append(breakout)
            
            # Analyze confluence
            for date, breakouts in breakouts_by_date.items():
                if len(breakouts) >= 2:  # At least 2 breakouts for confluence
                    signal = self._calculate_confluence_signal(breakouts)
                    if signal:
                        confluence['signals'].append(signal)
            
            # Calculate overall strength
            if confluence['signals']:
                confluence['strength'] = sum(s['strength'] for s in confluence['signals']) / len(confluence['signals'])
            
            # Generate recommendations
            if confluence['strength'] > 0.7:
                confluence['recommendations'].append("Strong breakout confluence detected - high confidence signals")
            elif confluence['strength'] > 0.5:
                confluence['recommendations'].append("Moderate breakout confluence - medium confidence signals")
            else:
                confluence['recommendations'].append("Weak breakout confluence - low confidence signals")
            
        except Exception as e:
            logger.error(f"❌ Error in breakout confluence analysis: {str(e)}")
        
        return confluence
    
    def _calculate_confluence_signal(self, breakouts: List[Dict]) -> Optional[Dict]:
        """Calculate confluence signal from multiple breakouts"""
        try:
            # Count real vs fake breakouts
            real_count = sum(1 for b in breakouts if b['is_real_breakout'])
            fake_count = len(breakouts) - real_count
            
            # Determine dominant signal
            total_breakouts = len(breakouts)
            if real_count > fake_count:
                dominant_signal = 'REAL_BREAKOUT'
                strength = real_count / total_breakouts
            else:
                dominant_signal = 'FAKE_BREAKOUT'
                strength = fake_count / total_breakouts
            
            # Calculate average confidence
            avg_confidence = sum(b['confidence'] for b in breakouts) / len(breakouts)
            
            # Get timeframes involved
            timeframes = list(set(b['timeframe'] for b in breakouts))
            
            return {
                'date': breakouts[0]['candidate']['date'],
                'signal': dominant_signal,
                'strength': round(strength, 2),
                'confidence': round(avg_confidence, 2),
                'breakout_count': len(breakouts),
                'timeframes': timeframes,
                'breakout_types': [b['candidate']['type'] for b in breakouts]
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating confluence signal: {str(e)}")
            return None

def main():
    """Test the breakout detection system"""
    print("🚀 Testing Breakout Detection System")
    print("=" * 60)
    
    # Initialize breakout detection
    breakout_detection = BreakoutDetection()
    
    # Load sample data
    import sys
    sys.path.append('/workspace')
    from data.nse_enhanced_data_loader import NSEEnhancedDataLoader
    
    loader = NSEEnhancedDataLoader("RELIANCE")
    multi_data = loader.load_multi_timeframe_data()
    
    if multi_data:
        # Analyze breakouts
        breakouts = breakout_detection.analyze_multi_timeframe_breakouts(multi_data)
        
        # Print results
        print(f"\n📊 Breakout Analysis Results:")
        print(f"Total Breakouts: {breakouts['summary']['total_breakouts']}")
        print(f"Real Breakouts: {breakouts['summary']['real_breakouts']}")
        print(f"Fake Breakouts: {breakouts['summary']['fake_breakouts']}")
        print(f"Confluence Signals: {breakouts['summary']['confluence_signals']}")
        
        # Print detailed results for each timeframe
        for timeframe in multi_data.keys():
            if timeframe in breakouts['timeframe_breakouts']:
                tf_breakouts = breakouts['timeframe_breakouts'][timeframe]
                print(f"\n{timeframe.upper()} Breakouts:")
                print(f"  Real: {tf_breakouts['summary']['real_breakouts']}")
                print(f"  Fake: {tf_breakouts['summary']['fake_breakouts']}")
                print(f"  Confidence: {tf_breakouts['summary']['confidence']:.2f}")
        
        # Print confluence analysis
        if breakouts['confluence_analysis']['signals']:
            print(f"\n🤝 Confluence Analysis:")
            print(f"  Strength: {breakouts['confluence_analysis']['strength']:.2f}")
            print(f"  Signals: {len(breakouts['confluence_analysis']['signals'])}")
            
            for rec in breakouts['confluence_analysis']['recommendations']:
                print(f"  💡 {rec}")
    else:
        print("❌ No data available for breakout analysis")

if __name__ == "__main__":
    main()