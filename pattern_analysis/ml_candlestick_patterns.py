"""
Ultimate Market AI Engine - ML Candlestick Pattern Discovery
==========================================================

This module implements proprietary ML-based candlestick pattern discovery
using unsupervised learning algorithms to find NEW patterns beyond traditional ones.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
import pickle
import os
from pathlib import Path

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLCandlestickDiscovery:
    """
    Proprietary ML-based candlestick pattern discovery system.
    
    Features:
    - Discovers NEW patterns using unsupervised ML (K-means, DBSCAN, GMM)
    - Analyzes 1-20 candle sequences
    - Tracks pattern evolution across timeframes
    - Calculates pattern success rates and predictive power
    - Creates pattern clustering to group similar formations
    - Builds breakout prediction based on discovered patterns
    - NO copying of traditional patterns - discovers NEW ones
    """
    
    def __init__(self, config=None):
        """Initialize the ML candlestick pattern discovery system."""
        self.config = config or get_config()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.tsne = TSNE(n_components=2, random_state=42)
        
        # Pattern storage
        self.discovered_patterns = {}
        self.pattern_clusters = {}
        self.pattern_performance = {}
        
        # Feature engineering parameters
        self.feature_columns = [
            'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'volume_ratio', 'price_change', 'volatility_ratio',
            'trend_strength', 'momentum', 'relative_position'
        ]
        
        logger.info("ML Candlestick Discovery initialized")
    
    def _extract_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features from candlestick data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with extracted features
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        features = df.copy()
        
        # Basic candlestick features
        features['body'] = features['Close'] - features['Open']
        features['body_abs'] = abs(features['body'])
        features['upper_shadow'] = features['High'] - features[['Open', 'Close']].max(axis=1)
        features['lower_shadow'] = features[['Open', 'Close']].min(axis=1) - features['Low']
        features['total_range'] = features['High'] - features['Low']
        
        # Ratios
        features['body_ratio'] = features['body_abs'] / features['total_range']
        features['upper_shadow_ratio'] = features['upper_shadow'] / features['total_range']
        features['lower_shadow_ratio'] = features['lower_shadow'] / features['total_range']
        
        # Volume features
        if 'Volume' in features.columns:
            features['volume_ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()
            features['volume_trend'] = features['Volume'].pct_change()
        else:
            features['volume_ratio'] = 1.0
            features['volume_trend'] = 0.0
        
        # Price movement features
        features['price_change'] = features['Close'].pct_change()
        features['price_change_abs'] = abs(features['price_change'])
        
        # Volatility features
        features['volatility_ratio'] = features['total_range'] / features['Close'].rolling(20).std()
        
        # Trend features
        features['trend_strength'] = features['Close'].rolling(10).apply(
            lambda x: np.corrcoef(x, range(len(x)))[0, 1] if len(x) > 1 else 0
        )
        
        # Momentum features
        features['momentum'] = features['Close'] / features['Close'].shift(5) - 1
        
        # Relative position features
        features['relative_position'] = (features['Close'] - features['Low']) / features['total_range']
        
        # Additional derived features
        features['gap_up'] = (features['Open'] > features['Close'].shift(1)).astype(int)
        features['gap_down'] = (features['Open'] < features['Close'].shift(1)).astype(int)
        features['doji'] = (features['body_abs'] < features['total_range'] * 0.1).astype(int)
        features['hammer'] = ((features['lower_shadow'] > features['body_abs'] * 2) & 
                             (features['upper_shadow'] < features['body_abs'] * 0.5)).astype(int)
        features['shooting_star'] = ((features['upper_shadow'] > features['body_abs'] * 2) & 
                                   (features['lower_shadow'] < features['body_abs'] * 0.5)).astype(int)
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _create_pattern_sequences(self, df: pd.DataFrame, min_length: int = 1, max_length: int = 20) -> List[np.ndarray]:
        """
        Create pattern sequences from candlestick data.
        
        Args:
            df: DataFrame with features
            min_length: Minimum pattern length
            max_length: Maximum pattern length
        
        Returns:
            List of pattern sequences
        """
        sequences = []
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        if not feature_cols:
            logger.warning("No valid feature columns found")
            return sequences
        
        for length in range(min_length, max_length + 1):
            for i in range(len(df) - length + 1):
                sequence = df[feature_cols].iloc[i:i+length].values
                if not np.isnan(sequence).any():
                    sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} pattern sequences (length {min_length}-{max_length})")
        return sequences
    
    def _extract_sequence_features(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from pattern sequences for clustering.
        
        Args:
            sequences: List of pattern sequences
        
        Returns:
            Feature matrix for clustering
        """
        if not sequences:
            return np.array([])
        
        features = []
        
        for seq in sequences:
            try:
                # Ensure sequence is 2D
                if seq.ndim == 1:
                    seq = seq.reshape(-1, 1)
                
                # Get sequence dimensions
                seq_length, n_features = seq.shape
                
                # Statistical features (fixed size regardless of sequence length)
                seq_mean = np.mean(seq, axis=0)
                seq_std = np.std(seq, axis=0)
                seq_min = np.min(seq, axis=0)
                seq_max = np.max(seq, axis=0)
                
                # Trend features (using first feature for trend calculation)
                if seq_length > 1:
                    seq_trend = np.polyfit(range(seq_length), seq[:, 0], 1)[0]
                    seq_volatility = np.std(seq[:, 0])
                else:
                    seq_trend = 0.0
                    seq_volatility = 0.0
                
                # Pattern shape features (fixed size)
                seq_start = seq[0] if seq_length > 0 else np.zeros(n_features)
                seq_end = seq[-1] if seq_length > 0 else np.zeros(n_features)
                seq_mid = seq[seq_length//2] if seq_length > 2 else seq[0] if seq_length > 0 else np.zeros(n_features)
                
                # Additional fixed-size features
                seq_range = seq_max - seq_min
                seq_median = np.median(seq, axis=0)
                seq_skew = stats.skew(seq, axis=0) if seq_length > 2 else np.zeros(n_features)
                seq_kurtosis = stats.kurtosis(seq, axis=0) if seq_length > 3 else np.zeros(n_features)
                
                # Combine all features into a fixed-size vector
                seq_features = np.concatenate([
                    seq_mean,           # n_features
                    seq_std,            # n_features
                    seq_min,            # n_features
                    seq_max,            # n_features
                    seq_range,          # n_features
                    seq_median,         # n_features
                    seq_skew,           # n_features
                    seq_kurtosis,       # n_features
                    seq_start,          # n_features
                    seq_end,            # n_features
                    seq_mid,            # n_features
                    [seq_trend, seq_volatility, seq_length]  # 3 scalar features
                ])
                
                # Ensure no NaN or infinite values
                seq_features = np.nan_to_num(seq_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                features.append(seq_features)
                
            except Exception as e:
                logger.warning(f"Error processing sequence: {e}")
                continue
        
        if not features:
            return np.array([])
        
        # Convert to numpy array and ensure all features have the same length
        features_array = np.array(features)
        
        # Validate feature matrix
        if features_array.size == 0:
            return np.array([])
        
        # Ensure all rows have the same number of features
        feature_lengths = [len(f) for f in features]
        if len(set(feature_lengths)) > 1:
            logger.warning(f"Inconsistent feature lengths: {set(feature_lengths)}")
            # Pad shorter features with zeros
            max_length = max(feature_lengths)
            padded_features = []
            for f in features:
                if len(f) < max_length:
                    padded = np.pad(f, (0, max_length - len(f)), 'constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            features_array = np.array(padded_features)
        
        logger.info(f"Extracted {len(features_array)} feature vectors with {features_array.shape[1]} features each")
        return features_array
    
    def _apply_clustering(self, features: np.ndarray, algorithm: str = 'kmeans') -> Tuple[np.ndarray, Any]:
        """
        Apply clustering algorithm to discover patterns.
        
        Args:
            features: Feature matrix
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'gmm')
        
        Returns:
            Tuple of (labels, model)
        """
        if features.size == 0:
            return np.array([]), None
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        if algorithm == 'kmeans':
            # Determine optimal number of clusters
            n_clusters = min(20, max(3, len(features) // 50))
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        
        elif algorithm == 'gmm':
            n_components = min(10, max(2, len(features) // 100))
            model = GaussianMixture(n_components=n_components, random_state=42)
        
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Fit and predict
        labels = model.fit_predict(features_scaled)
        
        logger.info(f"Applied {algorithm} clustering: {len(np.unique(labels))} clusters")
        return labels, model
    
    def _analyze_pattern_clusters(self, sequences: List[np.ndarray], labels: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze discovered pattern clusters.
        
        Args:
            sequences: List of pattern sequences
            labels: Cluster labels
        
        Returns:
            Dictionary with cluster analysis
        """
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
            
            cluster_sequences = [seq for i, seq in enumerate(sequences) if labels[i] == cluster_id]
            
            if len(cluster_sequences) < self.config.MIN_PATTERN_FREQUENCY:
                continue
            
            # Calculate cluster statistics
            cluster_lengths = [len(seq) for seq in cluster_sequences]
            cluster_features = np.array([seq.flatten() for seq in cluster_sequences])
            
            analysis = {
                'count': len(cluster_sequences),
                'avg_length': np.mean(cluster_lengths),
                'length_std': np.std(cluster_lengths),
                'min_length': min(cluster_lengths),
                'max_length': max(cluster_lengths),
                'avg_features': np.mean(cluster_features, axis=0),
                'feature_std': np.std(cluster_features, axis=0),
                'sequences': cluster_sequences
            }
            
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
    
    def _calculate_pattern_performance(self, df: pd.DataFrame, patterns: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Calculate performance metrics for discovered patterns.
        
        Args:
            df: Original price data
            patterns: Discovered patterns
        
        Returns:
            Dictionary with performance metrics for each pattern
        """
        performance = {}
        
        for pattern_id, pattern_info in patterns.items():
            sequences = pattern_info['sequences']
            
            # Calculate forward returns for pattern occurrences
            forward_returns = []
            pattern_occurrences = []
            
            for seq in sequences:
                # Find where this pattern occurs in the data
                seq_length = len(seq)
                seq_features = seq.flatten()
                
                # Look for similar patterns in the data
                for i in range(len(df) - seq_length - 5):  # Leave 5 periods for forward analysis
                    window_features = df.iloc[i:i+seq_length][self.feature_columns].values.flatten()
                    
                    # Calculate similarity
                    similarity = 1 / (1 + np.linalg.norm(seq_features - window_features))
                    
                    if similarity > 0.8:  # High similarity threshold
                        # Calculate forward returns
                        current_price = df.iloc[i + seq_length - 1]['Close']
                        future_prices = df.iloc[i + seq_length:i + seq_length + 5]['Close']
                        
                        if len(future_prices) > 0:
                            forward_return = (future_prices.iloc[-1] - current_price) / current_price
                            forward_returns.append(forward_return)
                            pattern_occurrences.append(i)
            
            if forward_returns:
                performance[pattern_id] = {
                    'occurrences': len(forward_returns),
                    'avg_return': np.mean(forward_returns),
                    'return_std': np.std(forward_returns),
                    'win_rate': np.mean(np.array(forward_returns) > 0),
                    'max_gain': np.max(forward_returns),
                    'max_loss': np.min(forward_returns),
                    'sharpe_ratio': np.mean(forward_returns) / np.std(forward_returns) if np.std(forward_returns) > 0 else 0,
                    'confidence': len(forward_returns) / len(sequences)  # How often pattern leads to price movement
                }
        
        return performance
    
    def _test_statistical_significance(self, performance: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Test statistical significance of pattern performance.
        
        Args:
            performance: Pattern performance dictionary
        
        Returns:
            Dictionary with significance test results
        """
        significant_patterns = {}
        
        for pattern_id, perf in performance.items():
            if perf['occurrences'] < 10:  # Need minimum occurrences for significance test
                continue
            
            # Bootstrap test for significance
            returns = np.random.normal(0, perf['return_std'], 10000)
            p_value = np.mean(returns >= perf['avg_return'])
            
            if p_value < self.config.STATISTICAL_SIGNIFICANCE:
                significant_patterns[pattern_id] = {
                    **perf,
                    'p_value': p_value,
                    'significant': True
                }
        
        return significant_patterns
    
    def discover_patterns(self, df: pd.DataFrame, timeframes: List[str] = None) -> Dict[str, Dict]:
        """
        Discover candlestick patterns using ML algorithms.
        
        Args:
            df: DataFrame with OHLCV data
            timeframes: List of timeframes to analyze
        
        Returns:
            Dictionary with discovered patterns for each timeframe
        """
        timeframes = timeframes or self.config.TIMEFRAMES
        all_patterns = {}
        
        logger.info(f"Starting pattern discovery for {len(timeframes)} timeframes")
        
        for timeframe in timeframes:
            logger.info(f"Discovering patterns for {timeframe} timeframe")
            
            # Extract features
            features_df = self._extract_candlestick_features(df)
            
            if features_df.empty:
                logger.warning(f"No features extracted for {timeframe}")
                continue
            
            # Create pattern sequences
            sequences = self._create_pattern_sequences(
                features_df,
                self.config.MIN_PATTERN_LENGTH,
                self.config.MAX_PATTERN_LENGTH
            )
            
            if not sequences:
                logger.warning(f"No sequences created for {timeframe}")
                continue
            
            # Extract sequence features
            sequence_features = self._extract_sequence_features(sequences)
            
            if sequence_features.size == 0:
                logger.warning(f"No sequence features for {timeframe}")
                continue
            
            timeframe_patterns = {}
            
            # Apply multiple clustering algorithms
            for algorithm in self.config.CLUSTERING_ALGORITHMS:
                try:
                    labels, model = self._apply_clustering(sequence_features, algorithm)
                    
                    if len(labels) > 0:
                        # Analyze clusters
                        cluster_analysis = self._analyze_pattern_clusters(sequences, labels)
                        
                        # Calculate performance
                        performance = self._calculate_pattern_performance(df, cluster_analysis)
                        
                        # Test significance
                        significant_patterns = self._test_statistical_significance(performance)
                        
                        timeframe_patterns[algorithm] = {
                            'clusters': cluster_analysis,
                            'performance': performance,
                            'significant_patterns': significant_patterns,
                            'model': model
                        }
                        
                        logger.info(f"{algorithm}: {len(significant_patterns)} significant patterns found")
                
                except Exception as e:
                    logger.error(f"Error in {algorithm} clustering for {timeframe}: {e}")
            
            all_patterns[timeframe] = timeframe_patterns
        
        # Store discovered patterns
        self.discovered_patterns = all_patterns
        
        logger.info(f"Pattern discovery completed. Total significant patterns: {self._count_total_patterns()}")
        return all_patterns
    
    def _count_total_patterns(self) -> int:
        """Count total significant patterns across all timeframes."""
        total = 0
        for timeframe_data in self.discovered_patterns.values():
            for algorithm_data in timeframe_data.values():
                total += len(algorithm_data.get('significant_patterns', {}))
        return total
    
    def get_pattern_predictions(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """
        Generate predictions based on discovered patterns.
        
        Args:
            df: Current market data
            lookback: Number of periods to look back for pattern matching
        
        Returns:
            Dictionary with pattern-based predictions
        """
        predictions = {
            'pattern_signals': [],
            'confidence_scores': [],
            'predicted_direction': None,
            'predicted_magnitude': None,
            'matching_patterns': []
        }
        
        if not self.discovered_patterns:
            logger.warning("No patterns discovered yet. Run discover_patterns() first.")
            return predictions
        
        # Extract features from recent data
        features_df = self._extract_candlestick_features(df.tail(lookback))
        
        if features_df.empty:
            return predictions
        
        # Look for pattern matches
        for timeframe, timeframe_data in self.discovered_patterns.items():
            for algorithm, algorithm_data in timeframe_data.items():
                significant_patterns = algorithm_data.get('significant_patterns', {})
                
                for pattern_id, pattern_info in significant_patterns.items():
                    # Check if current market conditions match this pattern
                    match_score = self._calculate_pattern_match(features_df, pattern_info)
                    
                    if match_score > 0.7:  # High match threshold
                        predictions['matching_patterns'].append({
                            'pattern_id': pattern_id,
                            'timeframe': timeframe,
                            'algorithm': algorithm,
                            'match_score': match_score,
                            'expected_return': pattern_info['avg_return'],
                            'confidence': pattern_info['confidence']
                        })
        
        # Aggregate predictions
        if predictions['matching_patterns']:
            # Calculate weighted prediction
            total_weight = sum(p['match_score'] * p['confidence'] for p in predictions['matching_patterns'])
            
            if total_weight > 0:
                weighted_return = sum(
                    p['match_score'] * p['confidence'] * p['expected_return'] 
                    for p in predictions['matching_patterns']
                ) / total_weight
                
                predictions['predicted_direction'] = 'up' if weighted_return > 0 else 'down'
                predictions['predicted_magnitude'] = abs(weighted_return)
                predictions['confidence_scores'] = [p['confidence'] for p in predictions['matching_patterns']]
        
        return predictions
    
    def _calculate_pattern_match(self, features_df: pd.DataFrame, pattern_info: Dict) -> float:
        """
        Calculate how well current features match a discovered pattern.
        
        Args:
            features_df: Current market features
            pattern_info: Pattern information
        
        Returns:
            Match score between 0 and 1
        """
        if features_df.empty:
            return 0.0
        
        # Use the most recent data points
        recent_features = features_df[self.feature_columns].tail(5).values.flatten()
        pattern_features = pattern_info['avg_features']
        
        # Calculate similarity
        similarity = 1 / (1 + np.linalg.norm(recent_features - pattern_features))
        
        return similarity
    
    def visualize_patterns(self, save_path: str = None):
        """
        Create visualizations of discovered patterns.
        
        Args:
            save_path: Path to save visualizations
        """
        if not self.discovered_patterns:
            logger.warning("No patterns to visualize. Run discover_patterns() first.")
            return
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pattern count by timeframe
        timeframe_counts = {}
        for timeframe, data in self.discovered_patterns.items():
            total_patterns = sum(len(alg_data.get('significant_patterns', {})) 
                               for alg_data in data.values())
            timeframe_counts[timeframe] = total_patterns
        
        axes[0, 0].bar(timeframe_counts.keys(), timeframe_counts.values())
        axes[0, 0].set_title('Significant Patterns by Timeframe')
        axes[0, 0].set_ylabel('Number of Patterns')
        
        # Performance distribution
        all_returns = []
        for timeframe_data in self.discovered_patterns.values():
            for algorithm_data in timeframe_data.values():
                for pattern_info in algorithm_data.get('significant_patterns', {}).values():
                    all_returns.append(pattern_info['avg_return'])
        
        if all_returns:
            axes[0, 1].hist(all_returns, bins=20, alpha=0.7)
            axes[0, 1].set_title('Distribution of Pattern Returns')
            axes[0, 1].set_xlabel('Average Return')
            axes[0, 1].set_ylabel('Frequency')
        
        # Win rate vs confidence
        win_rates = []
        confidences = []
        for timeframe_data in self.discovered_patterns.values():
            for algorithm_data in timeframe_data.values():
                for pattern_info in algorithm_data.get('significant_patterns', {}).values():
                    win_rates.append(pattern_info['win_rate'])
                    confidences.append(pattern_info['confidence'])
        
        if win_rates and confidences:
            axes[1, 0].scatter(confidences, win_rates, alpha=0.6)
            axes[1, 0].set_title('Win Rate vs Confidence')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Win Rate')
        
        # Pattern length distribution
        lengths = []
        for timeframe_data in self.discovered_patterns.values():
            for algorithm_data in timeframe_data.values():
                for pattern_info in algorithm_data.get('clusters', {}).values():
                    lengths.append(pattern_info['avg_length'])
        
        if lengths:
            axes[1, 1].hist(lengths, bins=15, alpha=0.7)
            axes[1, 1].set_title('Pattern Length Distribution')
            axes[1, 1].set_xlabel('Average Length (candles)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pattern visualization saved to {save_path}")
        
        plt.show()
    
    def save_patterns(self, filepath: str):
        """Save discovered patterns to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.discovered_patterns, f)
            logger.info(f"Patterns saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def load_patterns(self, filepath: str):
        """Load discovered patterns from file."""
        try:
            with open(filepath, 'rb') as f:
                self.discovered_patterns = pickle.load(f)
            logger.info(f"Patterns loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the ML candlestick pattern discovery
    from data.live_data_loader import DataLoader
    
    print("Testing ML Candlestick Pattern Discovery...")
    
    # Load sample data
    loader = DataLoader()
    data = loader.fetch_single_timeframe_data(timeframe="1d")
    
    if data is not None and not data.empty:
        # Initialize pattern discovery
        pattern_discovery = MLCandlestickDiscovery()
        
        # Discover patterns
        patterns = pattern_discovery.discover_patterns(data)
        
        print(f"Discovered patterns: {pattern_discovery._count_total_patterns()}")
        
        # Generate predictions
        predictions = pattern_discovery.get_pattern_predictions(data)
        print(f"Pattern predictions: {predictions}")
        
        # Visualize patterns
        pattern_discovery.visualize_patterns()
        
    else:
        print("No data available for testing")
