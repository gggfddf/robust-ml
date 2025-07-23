#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - COMPREHENSIVE PATTERN RECOGNITION
Traditional + ML-Based Candlestick Pattern Recognition
Production-Grade Implementation with Multi-timeframe Analysis
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensivePatternRecognition:
    """Comprehensive Pattern Recognition System"""
    
    def __init__(self):
        self.traditional_patterns = self._define_traditional_patterns()
        self.ml_patterns = {}
        self.scalers = {}
        self.models = {}
        
        logger.info("Comprehensive Pattern Recognition initialized")
    
    def _define_traditional_patterns(self) -> Dict:
        """Define traditional candlestick patterns"""
        return {
            # Bullish Patterns
            'hammer': {
                'description': 'Bullish reversal pattern',
                'conditions': ['small_body', 'long_lower_shadow', 'short_upper_shadow'],
                'signal': 'BULLISH'
            },
            'inverted_hammer': {
                'description': 'Bullish reversal pattern',
                'conditions': ['small_body', 'long_upper_shadow', 'short_lower_shadow'],
                'signal': 'BULLISH'
            },
            'bullish_engulfing': {
                'description': 'Bullish reversal pattern',
                'conditions': ['prev_bearish', 'current_bullish', 'current_engulfs_prev'],
                'signal': 'BULLISH'
            },
            'morning_star': {
                'description': 'Bullish reversal pattern',
                'conditions': ['three_candles', 'first_bearish', 'second_small', 'third_bullish'],
                'signal': 'BULLISH'
            },
            'three_white_soldiers': {
                'description': 'Bullish continuation pattern',
                'conditions': ['three_bullish', 'increasing_closes', 'small_shadows'],
                'signal': 'BULLISH'
            },
            'piercing_line': {
                'description': 'Bullish reversal pattern',
                'conditions': ['prev_bearish', 'current_bullish', 'current_opens_below_prev_low'],
                'signal': 'BULLISH'
            },
            
            # Bearish Patterns
            'shooting_star': {
                'description': 'Bearish reversal pattern',
                'conditions': ['small_body', 'long_upper_shadow', 'short_lower_shadow'],
                'signal': 'BEARISH'
            },
            'bearish_engulfing': {
                'description': 'Bearish reversal pattern',
                'conditions': ['prev_bullish', 'current_bearish', 'current_engulfs_prev'],
                'signal': 'BEARISH'
            },
            'evening_star': {
                'description': 'Bearish reversal pattern',
                'conditions': ['three_candles', 'first_bullish', 'second_small', 'third_bearish'],
                'signal': 'BEARISH'
            },
            'three_black_crows': {
                'description': 'Bearish continuation pattern',
                'conditions': ['three_bearish', 'decreasing_closes', 'small_shadows'],
                'signal': 'BEARISH'
            },
            'dark_cloud_cover': {
                'description': 'Bearish reversal pattern',
                'conditions': ['prev_bullish', 'current_bearish', 'current_opens_above_prev_high'],
                'signal': 'BEARISH'
            },
            
            # Neutral Patterns
            'doji': {
                'description': 'Indecision pattern',
                'conditions': ['very_small_body', 'similar_open_close'],
                'signal': 'NEUTRAL'
            },
            'spinning_top': {
                'description': 'Indecision pattern',
                'conditions': ['small_body', 'long_shadows'],
                'signal': 'NEUTRAL'
            },
            'harami': {
                'description': 'Potential reversal pattern',
                'conditions': ['prev_large_body', 'current_small_body', 'current_inside_prev'],
                'signal': 'NEUTRAL'
            }
        }
    
    def detect_traditional_patterns(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Detect traditional candlestick patterns"""
        logger.info(f"Detecting traditional patterns for {timeframe}")
        
        patterns_found = {
            'timeframe': timeframe,
            'patterns': [],
            'summary': {
                'total_patterns': 0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'neutral_patterns': 0
            }
        }
        
        if len(data) < 3:
            logger.warning(f"Insufficient data for pattern detection: {len(data)} records")
            return patterns_found
        
        # Calculate candlestick features
        data = self._calculate_candlestick_features(data)
        
        # Detect patterns
        for pattern_name, pattern_config in self.traditional_patterns.items():
            detected = self._detect_single_pattern(data, pattern_name, pattern_config)
            if detected:
                patterns_found['patterns'].extend(detected)
        
        # Update summary
        patterns_found['summary']['total_patterns'] = len(patterns_found['patterns'])
        patterns_found['summary']['bullish_patterns'] = len([p for p in patterns_found['patterns'] if p['signal'] == 'BULLISH'])
        patterns_found['summary']['bearish_patterns'] = len([p for p in patterns_found['patterns'] if p['signal'] == 'BEARISH'])
        patterns_found['summary']['neutral_patterns'] = len([p for p in patterns_found['patterns'] if p['signal'] == 'NEUTRAL'])
        
        logger.info(f"✅ {timeframe}: {patterns_found['summary']['total_patterns']} patterns detected")
        
        return patterns_found
    
    def _calculate_candlestick_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick features for pattern detection"""
        df = data.copy()
        
        # Basic candlestick features
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['total_range'] = df['High'] - df['Low']
        
        # Body characteristics
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
        
        # Candlestick type
        df['is_bullish'] = df['Close'] > df['Open']
        df['is_bearish'] = df['Close'] < df['Open']
        df['is_doji'] = df['body_ratio'] < 0.1
        
        # Relative sizes
        df['body_avg'] = df['body_size'].rolling(20).mean()
        df['body_relative'] = df['body_size'] / df['body_avg']
        
        return df
    
    def _detect_single_pattern(self, data: pd.DataFrame, pattern_name: str, pattern_config: Dict) -> List[Dict]:
        """Detect a single pattern type"""
        detected_patterns = []
        
        try:
            if pattern_name == 'hammer':
                detected_patterns = self._detect_hammer(data)
            elif pattern_name == 'inverted_hammer':
                detected_patterns = self._detect_inverted_hammer(data)
            elif pattern_name == 'bullish_engulfing':
                detected_patterns = self._detect_bullish_engulfing(data)
            elif pattern_name == 'bearish_engulfing':
                detected_patterns = self._detect_bearish_engulfing(data)
            elif pattern_name == 'doji':
                detected_patterns = self._detect_doji(data)
            elif pattern_name == 'spinning_top':
                detected_patterns = self._detect_spinning_top(data)
            elif pattern_name == 'morning_star':
                detected_patterns = self._detect_morning_star(data)
            elif pattern_name == 'evening_star':
                detected_patterns = self._detect_evening_star(data)
            elif pattern_name == 'three_white_soldiers':
                detected_patterns = self._detect_three_white_soldiers(data)
            elif pattern_name == 'three_black_crows':
                detected_patterns = self._detect_three_black_crows(data)
            
        except Exception as e:
            logger.error(f"Error detecting {pattern_name}: {str(e)}")
        
        return detected_patterns
    
    def _detect_hammer(self, data: pd.DataFrame) -> List[Dict]:
        """Detect hammer pattern"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            # Hammer conditions
            is_small_body = current['body_ratio'] < 0.3
            is_long_lower_shadow = current['lower_shadow_ratio'] > 0.6
            is_short_upper_shadow = current['upper_shadow_ratio'] < 0.1
            is_bullish = current['is_bullish']
            
            if is_small_body and is_long_lower_shadow and is_short_upper_shadow and is_bullish:
                patterns.append({
                    'pattern': 'hammer',
                    'date': current.name,
                    'signal': 'BULLISH',
                    'confidence': 0.8,
                    'price': current['Close'],
                    'description': 'Bullish reversal pattern'
                })
        
        return patterns
    
    def _detect_inverted_hammer(self, data: pd.DataFrame) -> List[Dict]:
        """Detect inverted hammer pattern"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            # Inverted hammer conditions
            is_small_body = current['body_ratio'] < 0.3
            is_long_upper_shadow = current['upper_shadow_ratio'] > 0.6
            is_short_lower_shadow = current['lower_shadow_ratio'] < 0.1
            is_bullish = current['is_bullish']
            
            if is_small_body and is_long_upper_shadow and is_short_lower_shadow and is_bullish:
                patterns.append({
                    'pattern': 'inverted_hammer',
                    'date': current.name,
                    'signal': 'BULLISH',
                    'confidence': 0.75,
                    'price': current['Close'],
                    'description': 'Bullish reversal pattern'
                })
        
        return patterns
    
    def _detect_bullish_engulfing(self, data: pd.DataFrame) -> List[Dict]:
        """Detect bullish engulfing pattern"""
        patterns = []
        
        for i in range(1, len(data)):
            prev = data.iloc[i-1]
            current = data.iloc[i]
            
            # Bullish engulfing conditions
            prev_bearish = prev['is_bearish']
            current_bullish = current['is_bullish']
            current_engulfs_prev = (current['Open'] < prev['Close'] and 
                                  current['Close'] > prev['Open'])
            
            if prev_bearish and current_bullish and current_engulfs_prev:
                patterns.append({
                    'pattern': 'bullish_engulfing',
                    'date': current.name,
                    'signal': 'BULLISH',
                    'confidence': 0.85,
                    'price': current['Close'],
                    'description': 'Bullish reversal pattern'
                })
        
        return patterns
    
    def _detect_bearish_engulfing(self, data: pd.DataFrame) -> List[Dict]:
        """Detect bearish engulfing pattern"""
        patterns = []
        
        for i in range(1, len(data)):
            prev = data.iloc[i-1]
            current = data.iloc[i]
            
            # Bearish engulfing conditions
            prev_bullish = prev['is_bullish']
            current_bearish = current['is_bearish']
            current_engulfs_prev = (current['Open'] > prev['Close'] and 
                                  current['Close'] < prev['Open'])
            
            if prev_bullish and current_bearish and current_engulfs_prev:
                patterns.append({
                    'pattern': 'bearish_engulfing',
                    'date': current.name,
                    'signal': 'BEARISH',
                    'confidence': 0.85,
                    'price': current['Close'],
                    'description': 'Bearish reversal pattern'
                })
        
        return patterns
    
    def _detect_doji(self, data: pd.DataFrame) -> List[Dict]:
        """Detect doji pattern"""
        patterns = []
        
        for i in range(len(data)):
            current = data.iloc[i]
            
            # Doji conditions
            is_very_small_body = current['body_ratio'] < 0.1
            similar_open_close = abs(current['Open'] - current['Close']) < (current['total_range'] * 0.1)
            
            if is_very_small_body and similar_open_close:
                patterns.append({
                    'pattern': 'doji',
                    'date': current.name,
                    'signal': 'NEUTRAL',
                    'confidence': 0.9,
                    'price': current['Close'],
                    'description': 'Indecision pattern'
                })
        
        return patterns
    
    def _detect_spinning_top(self, data: pd.DataFrame) -> List[Dict]:
        """Detect spinning top pattern"""
        patterns = []
        
        for i in range(len(data)):
            current = data.iloc[i]
            
            # Spinning top conditions
            is_small_body = current['body_ratio'] < 0.3
            is_long_shadows = (current['upper_shadow_ratio'] > 0.3 and 
                             current['lower_shadow_ratio'] > 0.3)
            
            if is_small_body and is_long_shadows:
                patterns.append({
                    'pattern': 'spinning_top',
                    'date': current.name,
                    'signal': 'NEUTRAL',
                    'confidence': 0.7,
                    'price': current['Close'],
                    'description': 'Indecision pattern'
                })
        
        return patterns
    
    def _detect_morning_star(self, data: pd.DataFrame) -> List[Dict]:
        """Detect morning star pattern"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # Morning star conditions
            first_bearish = first['is_bearish']
            second_small = second['body_ratio'] < 0.3
            third_bullish = third['is_bullish']
            
            if first_bearish and second_small and third_bullish:
                patterns.append({
                    'pattern': 'morning_star',
                    'date': third.name,
                    'signal': 'BULLISH',
                    'confidence': 0.9,
                    'price': third['Close'],
                    'description': 'Bullish reversal pattern'
                })
        
        return patterns
    
    def _detect_evening_star(self, data: pd.DataFrame) -> List[Dict]:
        """Detect evening star pattern"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # Evening star conditions
            first_bullish = first['is_bullish']
            second_small = second['body_ratio'] < 0.3
            third_bearish = third['is_bearish']
            
            if first_bullish and second_small and third_bearish:
                patterns.append({
                    'pattern': 'evening_star',
                    'date': third.name,
                    'signal': 'BEARISH',
                    'confidence': 0.9,
                    'price': third['Close'],
                    'description': 'Bearish reversal pattern'
                })
        
        return patterns
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame) -> List[Dict]:
        """Detect three white soldiers pattern"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # Three white soldiers conditions
            all_bullish = first['is_bullish'] and second['is_bullish'] and third['is_bullish']
            increasing_closes = first['Close'] < second['Close'] < third['Close']
            small_shadows = (first['upper_shadow_ratio'] < 0.2 and 
                           second['upper_shadow_ratio'] < 0.2 and 
                           third['upper_shadow_ratio'] < 0.2)
            
            if all_bullish and increasing_closes and small_shadows:
                patterns.append({
                    'pattern': 'three_white_soldiers',
                    'date': third.name,
                    'signal': 'BULLISH',
                    'confidence': 0.85,
                    'price': third['Close'],
                    'description': 'Bullish continuation pattern'
                })
        
        return patterns
    
    def _detect_three_black_crows(self, data: pd.DataFrame) -> List[Dict]:
        """Detect three black crows pattern"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # Three black crows conditions
            all_bearish = first['is_bearish'] and second['is_bearish'] and third['is_bearish']
            decreasing_closes = first['Close'] > second['Close'] > third['Close']
            small_shadows = (first['lower_shadow_ratio'] < 0.2 and 
                           second['lower_shadow_ratio'] < 0.2 and 
                           third['lower_shadow_ratio'] < 0.2)
            
            if all_bearish and decreasing_closes and small_shadows:
                patterns.append({
                    'pattern': 'three_black_crows',
                    'date': third.name,
                    'signal': 'BEARISH',
                    'confidence': 0.85,
                    'price': third['Close'],
                    'description': 'Bearish continuation pattern'
                })
        
        return patterns
    
    def discover_ml_patterns(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Discover ML-based patterns using clustering"""
        logger.info(f"Discovering ML patterns for {timeframe}")
        
        ml_patterns = {
            'timeframe': timeframe,
            'clusters': {},
            'patterns': [],
            'summary': {
                'total_clusters': 0,
                'total_patterns': 0
            }
        }
        
        if len(data) < 50:
            logger.warning(f"Insufficient data for ML pattern discovery: {len(data)} records")
            return ml_patterns
        
        try:
            # Prepare features for ML
            features = self._prepare_ml_features(data)
            
            if features is None or len(features) < 10:
                logger.warning("Insufficient features for ML pattern discovery")
                return ml_patterns
            
            # Apply clustering algorithms
            clusters = self._apply_clustering(features, timeframe)
            
            # Analyze clusters and identify patterns
            patterns = self._analyze_clusters(data, clusters, timeframe)
            
            ml_patterns['clusters'] = clusters
            ml_patterns['patterns'] = patterns
            ml_patterns['summary']['total_clusters'] = len(clusters)
            ml_patterns['summary']['total_patterns'] = len(patterns)
            
            logger.info(f"✅ {timeframe}: {ml_patterns['summary']['total_patterns']} ML patterns discovered")
            
        except Exception as e:
            logger.error(f"❌ Error in ML pattern discovery: {str(e)}")
        
        return ml_patterns
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML pattern discovery"""
        try:
            df = data.copy()
            
            # Calculate technical features
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_ma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # Price-based features
            df['price_ma_5'] = df['Close'].rolling(5).mean()
            df['price_ma_20'] = df['Close'].rolling(20).mean()
            df['price_ma_ratio'] = df['price_ma_5'] / df['price_ma_20']
            
            # Candlestick features
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['body_ratio'] = df['body_size'] / (df['High'] - df['Low'])
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 10:
                return None
            
            # Select features for clustering
            feature_columns = [
                'returns', 'volatility', 'volume_ratio', 'price_ma_ratio',
                'body_ratio', 'upper_shadow', 'lower_shadow'
            ]
            
            features = df[feature_columns].values
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Store scaler for later use
            self.scalers[df.index[0].strftime('%Y%m%d')] = scaler
            
            return features_scaled
            
        except Exception as e:
            logger.error(f"❌ Error preparing ML features: {str(e)}")
            return None
    
    def _apply_clustering(self, features: np.ndarray, timeframe: str) -> Dict:
        """Apply clustering algorithms to discover patterns"""
        clusters = {}
        
        try:
            # K-means clustering
            n_clusters = min(10, len(features) // 10)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features)
            
            clusters['kmeans'] = {
                'labels': kmeans_labels,
                'centers': kmeans.cluster_centers_,
                'n_clusters': n_clusters
            }
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(features)
            
            clusters['dbscan'] = {
                'labels': dbscan_labels,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            }
            
            # Store models
            self.models[f'{timeframe}_kmeans'] = kmeans
            self.models[f'{timeframe}_dbscan'] = dbscan
            
        except Exception as e:
            logger.error(f"❌ Error in clustering: {str(e)}")
        
        return clusters
    
    def _analyze_clusters(self, data: pd.DataFrame, clusters: Dict, timeframe: str) -> List[Dict]:
        """Analyze clusters to identify patterns"""
        patterns = []
        
        try:
            # Analyze K-means clusters
            if 'kmeans' in clusters:
                kmeans_data = clusters['kmeans']
                labels = kmeans_data['labels']
                
                for cluster_id in range(kmeans_data['n_clusters']):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    
                    if len(cluster_indices) > 5:  # Minimum cluster size
                        cluster_data = data.iloc[cluster_indices]
                        
                        # Analyze cluster characteristics
                        pattern = self._analyze_cluster_characteristics(
                            cluster_data, cluster_id, 'kmeans', timeframe
                        )
                        
                        if pattern:
                            patterns.append(pattern)
            
            # Analyze DBSCAN clusters
            if 'dbscan' in clusters:
                dbscan_data = clusters['dbscan']
                labels = dbscan_data['labels']
                
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)  # Remove noise
                
                for cluster_id in unique_labels:
                    cluster_indices = np.where(labels == cluster_id)[0]
                    
                    if len(cluster_indices) > 5:  # Minimum cluster size
                        cluster_data = data.iloc[cluster_indices]
                        
                        # Analyze cluster characteristics
                        pattern = self._analyze_cluster_characteristics(
                            cluster_data, cluster_id, 'dbscan', timeframe
                        )
                        
                        if pattern:
                            patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing clusters: {str(e)}")
        
        return patterns
    
    def _analyze_cluster_characteristics(self, cluster_data: pd.DataFrame, 
                                       cluster_id: int, method: str, timeframe: str) -> Optional[Dict]:
        """Analyze characteristics of a cluster to identify pattern"""
        try:
            # Calculate cluster statistics
            avg_body_ratio = cluster_data['body_ratio'].mean()
            avg_volume_ratio = cluster_data['Volume'].mean() / cluster_data['Volume'].rolling(20).mean().mean()
            avg_returns = cluster_data['Close'].pct_change().mean()
            
            # Determine pattern type based on characteristics
            if avg_returns > 0.01:  # Positive returns
                signal = 'BULLISH'
                confidence = min(0.9, abs(avg_returns) * 10)
            elif avg_returns < -0.01:  # Negative returns
                signal = 'BEARISH'
                confidence = min(0.9, abs(avg_returns) * 10)
            else:
                signal = 'NEUTRAL'
                confidence = 0.5
            
            # Pattern description
            if avg_body_ratio > 0.7:
                body_desc = "strong momentum"
            elif avg_body_ratio < 0.3:
                body_desc = "indecision"
            else:
                body_desc = "moderate momentum"
            
            if avg_volume_ratio > 1.5:
                volume_desc = "high volume"
            elif avg_volume_ratio < 0.5:
                volume_desc = "low volume"
            else:
                volume_desc = "normal volume"
            
            description = f"ML-{method.upper()} cluster: {body_desc}, {volume_desc}"
            
            return {
                'pattern': f'ml_{method}_{cluster_id}',
                'method': method,
                'cluster_id': cluster_id,
                'signal': signal,
                'confidence': round(confidence, 2),
                'date': cluster_data.index[-1],
                'price': cluster_data['Close'].iloc[-1],
                'description': description,
                'cluster_size': len(cluster_data),
                'avg_returns': round(avg_returns, 4),
                'avg_body_ratio': round(avg_body_ratio, 3),
                'avg_volume_ratio': round(avg_volume_ratio, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing cluster characteristics: {str(e)}")
            return None
    
    def analyze_multi_timeframe_patterns(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze patterns across multiple timeframes"""
        logger.info("Analyzing patterns across multiple timeframes")
        
        multi_patterns = {
            'traditional_patterns': {},
            'ml_patterns': {},
            'confluence_analysis': {},
            'summary': {
                'total_traditional': 0,
                'total_ml': 0,
                'confluence_signals': 0
            }
        }
        
        # Analyze each timeframe
        for timeframe, data in multi_data.items():
            if data is not None and len(data) > 0:
                # Traditional patterns
                traditional = self.detect_traditional_patterns(data, timeframe)
                multi_patterns['traditional_patterns'][timeframe] = traditional
                multi_patterns['summary']['total_traditional'] += traditional['summary']['total_patterns']
                
                # ML patterns
                ml = self.discover_ml_patterns(data, timeframe)
                multi_patterns['ml_patterns'][timeframe] = ml
                multi_patterns['summary']['total_ml'] += ml['summary']['total_patterns']
        
        # Analyze confluence across timeframes
        confluence = self._analyze_pattern_confluence(multi_patterns)
        multi_patterns['confluence_analysis'] = confluence
        multi_patterns['summary']['confluence_signals'] = len(confluence['signals'])
        
        logger.info(f"✅ Multi-timeframe analysis complete: {multi_patterns['summary']['total_traditional']} traditional, {multi_patterns['summary']['total_ml']} ML patterns")
        
        return multi_patterns
    
    def _analyze_pattern_confluence(self, multi_patterns: Dict) -> Dict:
        """Analyze pattern confluence across timeframes"""
        confluence = {
            'signals': [],
            'strength': 0,
            'recommendations': []
        }
        
        try:
            # Collect all patterns by date
            all_patterns = []
            
            for timeframe, traditional in multi_patterns['traditional_patterns'].items():
                for pattern in traditional['patterns']:
                    pattern['timeframe'] = timeframe
                    pattern['type'] = 'traditional'
                    all_patterns.append(pattern)
            
            for timeframe, ml in multi_patterns['ml_patterns'].items():
                for pattern in ml['patterns']:
                    pattern['timeframe'] = timeframe
                    pattern['type'] = 'ml'
                    all_patterns.append(pattern)
            
            # Group patterns by date
            from collections import defaultdict
            patterns_by_date = defaultdict(list)
            
            for pattern in all_patterns:
                date_key = pattern['date'].strftime('%Y-%m-%d')
                patterns_by_date[date_key].append(pattern)
            
            # Analyze confluence
            for date, patterns in patterns_by_date.items():
                if len(patterns) >= 2:  # At least 2 patterns for confluence
                    signal = self._calculate_confluence_signal(patterns)
                    if signal:
                        confluence['signals'].append(signal)
            
            # Calculate overall strength
            if confluence['signals']:
                confluence['strength'] = sum(s['strength'] for s in confluence['signals']) / len(confluence['signals'])
            
            # Generate recommendations
            if confluence['strength'] > 0.7:
                confluence['recommendations'].append("Strong pattern confluence detected - high confidence signals")
            elif confluence['strength'] > 0.5:
                confluence['recommendations'].append("Moderate pattern confluence - medium confidence signals")
            else:
                confluence['recommendations'].append("Weak pattern confluence - low confidence signals")
            
        except Exception as e:
            logger.error(f"❌ Error in confluence analysis: {str(e)}")
        
        return confluence
    
    def _calculate_confluence_signal(self, patterns: List[Dict]) -> Optional[Dict]:
        """Calculate confluence signal from multiple patterns"""
        try:
            # Count signals
            bullish_count = sum(1 for p in patterns if p['signal'] == 'BULLISH')
            bearish_count = sum(1 for p in patterns if p['signal'] == 'BEARISH')
            neutral_count = sum(1 for p in patterns if p['signal'] == 'NEUTRAL')
            
            # Determine dominant signal
            total_patterns = len(patterns)
            if bullish_count > bearish_count and bullish_count > neutral_count:
                dominant_signal = 'BULLISH'
                strength = bullish_count / total_patterns
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                dominant_signal = 'BEARISH'
                strength = bearish_count / total_patterns
            else:
                dominant_signal = 'NEUTRAL'
                strength = neutral_count / total_patterns
            
            # Calculate average confidence
            avg_confidence = sum(p['confidence'] for p in patterns) / len(patterns)
            
            # Get timeframes involved
            timeframes = list(set(p['timeframe'] for p in patterns))
            
            return {
                'date': patterns[0]['date'],
                'signal': dominant_signal,
                'strength': round(strength, 2),
                'confidence': round(avg_confidence, 2),
                'pattern_count': len(patterns),
                'timeframes': timeframes,
                'patterns': [p['pattern'] for p in patterns]
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating confluence signal: {str(e)}")
            return None

def main():
    """Test the comprehensive pattern recognition system"""
    print("🚀 Testing Comprehensive Pattern Recognition")
    print("=" * 60)
    
    # Initialize pattern recognition
    pattern_recognition = ComprehensivePatternRecognition()
    
    # Load sample data (you would normally load from your data loader)
    import sys
    sys.path.append('/workspace')
    from data.enhanced_data_loader import EnhancedDataLoader
    
    loader = EnhancedDataLoader("RELIANCE.NS")
    multi_data = loader.load_multi_timeframe_data()
    
    if multi_data:
        # Analyze patterns
        patterns = pattern_recognition.analyze_multi_timeframe_patterns(multi_data)
        
        # Print results
        print(f"\n📊 Pattern Analysis Results:")
        print(f"Traditional Patterns: {patterns['summary']['total_traditional']}")
        print(f"ML Patterns: {patterns['summary']['total_ml']}")
        print(f"Confluence Signals: {patterns['summary']['confluence_signals']}")
        
        # Print detailed results for each timeframe
        for timeframe in multi_data.keys():
            if timeframe in patterns['traditional_patterns']:
                trad = patterns['traditional_patterns'][timeframe]
                print(f"\n{timeframe.upper()} Traditional Patterns:")
                print(f"  Bullish: {trad['summary']['bullish_patterns']}")
                print(f"  Bearish: {trad['summary']['bearish_patterns']}")
                print(f"  Neutral: {trad['summary']['neutral_patterns']}")
            
            if timeframe in patterns['ml_patterns']:
                ml = patterns['ml_patterns'][timeframe]
                print(f"\n{timeframe.upper()} ML Patterns:")
                print(f"  Clusters: {ml['summary']['total_clusters']}")
                print(f"  Patterns: {ml['summary']['total_patterns']}")
        
        # Print confluence analysis
        if patterns['confluence_analysis']['signals']:
            print(f"\n🤝 Confluence Analysis:")
            print(f"  Strength: {patterns['confluence_analysis']['strength']:.2f}")
            print(f"  Signals: {len(patterns['confluence_analysis']['signals'])}")
            
            for rec in patterns['confluence_analysis']['recommendations']:
                print(f"  💡 {rec}")
    else:
        print("❌ No data available for pattern analysis")

if __name__ == "__main__":
    main()