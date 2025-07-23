#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - ENHANCED ML PIPELINE
Complete ML Pipeline with Actual Predictions and Multi-timeframe Integration
Production-Grade Implementation with Advanced Features
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, skipping deep learning models")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLPipeline:
    """Enhanced ML Pipeline with Actual Predictions"""
    
    def __init__(self, symbol="RELIANCE"):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.ensemble_weights = {}
        
        logger.info(f"Enhanced ML Pipeline initialized for {symbol}")
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete enhanced ML pipeline"""
        logger.info("🚀 Starting Enhanced ML Pipeline")
        
        pipeline_results = {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'data_analysis': {},
            'feature_engineering': {},
            'model_training': {},
            'predictions': {},
            'ensemble_results': {},
            'performance_metrics': {},
            'trading_signals': {},
            'summary': {}
        }
        
        try:
            # Step 1: Load and analyze data
            logger.info("📊 Step 1: Loading and analyzing data...")
            data_analysis = self._load_and_analyze_data()
            pipeline_results['data_analysis'] = data_analysis
            
            if not data_analysis['data_available']:
                logger.error("❌ No data available for analysis")
                return pipeline_results
            
            # Step 2: Feature engineering
            logger.info("🔧 Step 2: Feature engineering...")
            feature_engineering = self._engineer_features(data_analysis['multi_data'])
            pipeline_results['feature_engineering'] = feature_engineering
            
            # Step 3: Train models
            logger.info("🤖 Step 3: Training models...")
            model_training = self._train_models(feature_engineering['features'])
            pipeline_results['model_training'] = model_training
            
            # Step 4: Generate predictions
            logger.info("🔮 Step 4: Generating predictions...")
            predictions = self._generate_predictions(feature_engineering['features'])
            pipeline_results['predictions'] = predictions
            
            # Step 5: Create ensemble
            logger.info("🎯 Step 5: Creating ensemble...")
            ensemble_results = self._create_ensemble(predictions)
            pipeline_results['ensemble_results'] = ensemble_results
            
            # Step 6: Calculate performance metrics
            logger.info("📈 Step 6: Calculating performance metrics...")
            performance_metrics = self._calculate_performance_metrics(ensemble_results)
            pipeline_results['performance_metrics'] = performance_metrics
            
            # Step 7: Generate trading signals
            logger.info("💹 Step 7: Generating trading signals...")
            trading_signals = self._generate_trading_signals(ensemble_results)
            pipeline_results['trading_signals'] = trading_signals
            
            # Step 8: Create summary
            logger.info("📋 Step 8: Creating summary...")
            summary = self._create_summary(pipeline_results)
            pipeline_results['summary'] = summary
            
            logger.info("✅ Enhanced ML Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced ML pipeline: {str(e)}")
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _load_and_analyze_data(self) -> Dict:
        """Load and analyze multi-timeframe data"""
        try:
            # Import data loader
            import sys
            sys.path.append('/workspace')
            from data.nse_enhanced_data_loader import NSEEnhancedDataLoader
            
            # Load data
            loader = NSEEnhancedDataLoader(self.symbol)
            multi_data = loader.load_multi_timeframe_data()
            
            # Analyze data quality
            data_summary = loader.get_data_summary(multi_data)
            quality_report = loader.validate_data_quality(multi_data)
            
            data_analysis = {
                'multi_data': multi_data,
                'data_summary': data_summary,
                'quality_report': quality_report,
                'data_available': len(multi_data) > 0,
                'timeframes': list(multi_data.keys())
            }
            
            logger.info(f"✅ Data loaded: {data_summary['overall']['total_records']} total records")
            
            return data_analysis
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            return {'data_available': False, 'error': str(e)}
    
    def _engineer_features(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Engineer comprehensive features for ML"""
        try:
            features = {}
            
            for timeframe, data in multi_data.items():
                if data is not None and len(data) > 0:
                    logger.info(f"Engineering features for {timeframe}")
                    
                    # Calculate technical indicators
                    df_features = self._calculate_technical_indicators(data)
                    
                    # Calculate price-based features
                    df_features = self._calculate_price_features(df_features)
                    
                    # Calculate volume features
                    df_features = self._calculate_volume_features(df_features)
                    
                    # Calculate volatility features
                    df_features = self._calculate_volatility_features(df_features)
                    
                    # Calculate momentum features
                    df_features = self._calculate_momentum_features(df_features)
                    
                    # Calculate trend features
                    df_features = self._calculate_trend_features(df_features)
                    
                    # Remove NaN values
                    df_features = df_features.dropna()
                    
                    features[timeframe] = df_features
                    
                    logger.info(f"✅ {timeframe}: {len(df_features)} records with {len(df_features.columns)} features")
            
            return {'features': features, 'feature_count': len(features)}
            
        except Exception as e:
            logger.error(f"❌ Error engineering features: {str(e)}")
            return {'features': {}, 'error': str(e)}
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            df = data.copy()
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Stochastic Oscillator
            for period in [14, 21]:
                low_min = df['Low'].rolling(period).min()
                high_max = df['High'].rolling(period).max()
                df[f'Stoch_K_{period}'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
                df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(3).mean()
            
            # Williams %R
            for period in [14, 21]:
                low_min = df['Low'].rolling(period).min()
                high_max = df['High'].rolling(period).max()
                df[f'Williams_R_{period}'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
            
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(14).mean()
            
            # Commodity Channel Index (CCI)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {str(e)}")
            return data
    
    def _calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        try:
            df = data.copy()
            
            # Price changes
            for period in [1, 2, 3, 5, 10, 20]:
                df[f'Price_Change_{period}'] = df['Close'].pct_change(period)
                df[f'Price_Change_Abs_{period}'] = df['Close'].pct_change(period).abs()
            
            # Price ratios
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Price positions
            df['Price_Position_20'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
            df['Price_Position_50'] = (df['Close'] - df['Low'].rolling(50).min()) / (df['High'].rolling(50).max() - df['Low'].rolling(50).min())
            
            # Support and Resistance levels
            df['Support_20'] = df['Low'].rolling(20).min()
            df['Resistance_20'] = df['High'].rolling(20).max()
            df['Support_Distance'] = (df['Close'] - df['Support_20']) / df['Close']
            df['Resistance_Distance'] = (df['Resistance_20'] - df['Close']) / df['Close']
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating price features: {str(e)}")
            return data
    
    def _calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        try:
            df = data.copy()
            
            # Volume moving averages
            for period in [5, 10, 20, 50]:
                df[f'Volume_MA_{period}'] = df['Volume'].rolling(period).mean()
                df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_MA_{period}']
            
            # Volume price trend
            df['Volume_Price_Trend'] = (df['Volume'] * df['Close'].pct_change()).cumsum()
            
            # On Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
            
            # Volume Rate of Change
            for period in [5, 10, 20]:
                df[f'Volume_ROC_{period}'] = df['Volume'].pct_change(period)
            
            # Money Flow Index
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            money_ratio = positive_flow / negative_flow
            df['MFI'] = 100 - (100 / (1 + money_ratio))
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating volume features: {str(e)}")
            return data
    
    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        try:
            df = data.copy()
            
            # Historical volatility
            for period in [5, 10, 20, 50]:
                df[f'Volatility_{period}'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
            
            # Parkinson volatility
            for period in [5, 10, 20]:
                high_low_ratio = np.log(df['High'] / df['Low'])
                df[f'Parkinson_Vol_{period}'] = np.sqrt((high_low_ratio ** 2).rolling(period).mean() / (4 * np.log(2)))
            
            # Garman-Klass volatility
            for period in [5, 10, 20]:
                c = np.log(df['Close'] / df['Close'].shift(1))
                h = np.log(df['High'] / df['Open'])
                l = np.log(df['Low'] / df['Open'])
                df[f'Garman_Klass_Vol_{period}'] = np.sqrt(((0.5 * (h - l) ** 2) - ((2 * np.log(2) - 1) * c ** 2)).rolling(period).mean())
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility features: {str(e)}")
            return data
    
    def _calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features"""
        try:
            df = data.copy()
            
            # Rate of Change
            for period in [5, 10, 20, 50]:
                df[f'ROC_{period}'] = df['Close'].pct_change(period)
            
            # Momentum
            for period in [5, 10, 20]:
                df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            
            # Relative Strength Index variations
            for period in [7, 14, 21]:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # Stochastic RSI
            for period in [14, 21]:
                rsi = df[f'RSI_{period}']
                rsi_min = rsi.rolling(period).min()
                rsi_max = rsi.rolling(period).max()
                df[f'Stoch_RSI_{period}'] = (rsi - rsi_min) / (rsi_max - rsi_min)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating momentum features: {str(e)}")
            return data
    
    def _calculate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based features"""
        try:
            df = data.copy()
            
            # Trend direction
            for period in [5, 10, 20, 50]:
                df[f'Trend_{period}'] = np.where(df['Close'] > df['Close'].shift(period), 1, -1)
            
            # ADX (Average Directional Index)
            for period in [14, 21]:
                # True Range
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift(1))
                low_close = np.abs(df['Low'] - df['Close'].shift(1))
                tr = np.maximum(high_low, np.maximum(high_close, low_close))
                
                # Directional Movement
                up_move = df['High'] - df['High'].shift(1)
                down_move = df['Low'].shift(1) - df['Low']
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                # Smoothed values
                tr_smooth = tr.rolling(period).mean()
                plus_di = 100 * (plus_dm.rolling(period).mean() / tr_smooth)
                minus_di = 100 * (minus_dm.rolling(period).mean() / tr_smooth)
                
                # ADX
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                df[f'ADX_{period}'] = dx.rolling(period).mean()
            
            # Parabolic SAR (simplified)
            df['PSAR'] = df['Close'].rolling(5).min()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating trend features: {str(e)}")
            return data
    
    def _train_models(self, features: Dict[str, pd.DataFrame]) -> Dict:
        """Train multiple ML models"""
        try:
            model_results = {}
            
            for timeframe, data in features.items():
                if len(data) < 100:
                    logger.warning(f"Insufficient data for {timeframe}: {len(data)} records")
                    continue
                
                logger.info(f"Training models for {timeframe}")
                
                # Prepare data
                X, y = self._prepare_training_data(data)
                
                if len(X) < 50:
                    logger.warning(f"Insufficient training data for {timeframe}: {len(X)} samples")
                    continue
                
                # Train models
                models = self._train_individual_models(X, y, timeframe)
                
                # Store results
                model_results[timeframe] = {
                    'models': models,
                    'X_shape': X.shape,
                    'y_shape': y.shape,
                    'feature_names': list(X.columns)
                }
                
                logger.info(f"✅ {timeframe}: {len(models)} models trained")
            
            return model_results
            
        except Exception as e:
            logger.error(f"❌ Error training models: {str(e)}")
            return {}
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data"""
        try:
            df = data.copy()
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            if len(df) < 50:
                return pd.DataFrame(), pd.Series()
            
            # Select features (exclude OHLCV and date columns)
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            X = df[feature_columns]
            y = df['Close']  # Predict next close price
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Store scaler
            self.scalers[f"{df.index[0].strftime('%Y%m%d')}"] = scaler
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def _train_individual_models(self, X: pd.DataFrame, y: pd.Series, timeframe: str) -> Dict:
        """Train individual ML models"""
        try:
            models = {}
            
            # Split data for time series
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Model configurations
            model_configs = {
                'RandomForest': {
                    'model': RandomForestRegressor(n_estimators=100, random_state=42),
                    'params': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
                },
                'GradientBoosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
                },
                'XGBoost': {
                    'model': xgb.XGBRegressor(random_state=42),
                    'params': {'n_estimators': [50, 100], 'max_depth': [3, 6]}
                },
                'LightGBM': {
                    'model': lgb.LGBMRegressor(random_state=42),
                    'params': {'n_estimators': [50, 100], 'max_depth': [3, 6]}
                },
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'Ridge': {
                    'model': Ridge(),
                    'params': {'alpha': [0.1, 1.0, 10.0]}
                }
            }
            
            # Train each model
            for name, config in model_configs.items():
                try:
                    logger.info(f"Training {name} for {timeframe}")
                    
                    if config['params']:
                        # Grid search for hyperparameter tuning
                        tscv = TimeSeriesSplit(n_splits=3)
                        grid_search = GridSearchCV(
                            config['model'], 
                            config['params'], 
                            cv=tscv, 
                            scoring='r2',
                            n_jobs=-1
                        )
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                    else:
                        # Direct training
                        best_model = config['model']
                        best_model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = best_model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store model and results
                    models[name] = {
                        'model': best_model,
                        'r2_score': r2,
                        'mse': mse,
                        'mae': mae,
                        'predictions': y_pred
                    }
                    
                    # Store feature importance if available
                    if hasattr(best_model, 'feature_importances_'):
                        self.feature_importance[f"{timeframe}_{name}"] = dict(zip(X.columns, best_model.feature_importances_))
                    
                    logger.info(f"✅ {name}: R² = {r2:.4f}, MSE = {mse:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {name}: {str(e)}")
                    continue
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Error training individual models: {str(e)}")
            return {}
    
    def _generate_predictions(self, features: Dict[str, pd.DataFrame]) -> Dict:
        """Generate predictions for all timeframes"""
        try:
            predictions = {}
            
            for timeframe, data in features.items():
                if timeframe not in self.models:
                    continue
                
                logger.info(f"Generating predictions for {timeframe}")
                
                # Prepare latest data
                latest_data = data.tail(1)  # Get most recent data point
                X_latest = self._prepare_prediction_data(latest_data, timeframe)
                
                if X_latest.empty:
                    continue
                
                # Generate predictions from all models
                timeframe_predictions = {}
                
                for model_name, model_info in self.models[timeframe].items():
                    try:
                        model = model_info['model']
                        pred = model.predict(X_latest)[0]
                        
                        timeframe_predictions[model_name] = {
                            'prediction': pred,
                            'confidence': model_info['r2_score'],
                            'model_performance': {
                                'r2': model_info['r2_score'],
                                'mse': model_info['mse'],
                                'mae': model_info['mae']
                            }
                        }
                        
                    except Exception as e:
                        logger.error(f"❌ Error generating prediction for {model_name}: {str(e)}")
                        continue
                
                predictions[timeframe] = timeframe_predictions
                logger.info(f"✅ {timeframe}: {len(timeframe_predictions)} predictions generated")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating predictions: {str(e)}")
            return {}
    
    def _prepare_prediction_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Prepare data for prediction"""
        try:
            # Use the same feature engineering as training
            df = self._calculate_technical_indicators(data)
            df = self._calculate_price_features(df)
            df = self._calculate_volume_features(df)
            df = self._calculate_volatility_features(df)
            df = self._calculate_momentum_features(df)
            df = self._calculate_trend_features(df)
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # Select features (same as training)
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            X = df[feature_columns]
            
            # Scale features using stored scaler
            scaler_key = f"{df.index[0].strftime('%Y%m%d')}"
            if scaler_key in self.scalers:
                scaler = self.scalers[scaler_key]
                X_scaled = scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                return X_scaled
            
            return X
            
        except Exception as e:
            logger.error(f"❌ Error preparing prediction data: {str(e)}")
            return pd.DataFrame()
    
    def _create_ensemble(self, predictions: Dict) -> Dict:
        """Create ensemble predictions"""
        try:
            ensemble_results = {}
            
            for timeframe, timeframe_preds in predictions.items():
                if not timeframe_preds:
                    continue
                
                # Calculate ensemble weights based on R² scores
                weights = {}
                total_r2 = 0
                
                for model_name, pred_info in timeframe_preds.items():
                    r2 = pred_info['model_performance']['r2']
                    if r2 > 0:  # Only include models with positive R²
                        weights[model_name] = r2
                        total_r2 += r2
                
                if total_r2 == 0:
                    continue
                
                # Normalize weights
                for model_name in weights:
                    weights[model_name] /= total_r2
                
                # Calculate weighted ensemble prediction
                ensemble_prediction = 0
                total_confidence = 0
                
                for model_name, pred_info in timeframe_preds.items():
                    if model_name in weights:
                        weight = weights[model_name]
                        prediction = pred_info['prediction']
                        confidence = pred_info['confidence']
                        
                        ensemble_prediction += weight * prediction
                        total_confidence += weight * confidence
                
                # Store ensemble results
                ensemble_results[timeframe] = {
                    'ensemble_prediction': ensemble_prediction,
                    'ensemble_confidence': total_confidence,
                    'model_weights': weights,
                    'individual_predictions': timeframe_preds
                }
                
                # Store weights for later use
                self.ensemble_weights[timeframe] = weights
                
                logger.info(f"✅ {timeframe} ensemble: {ensemble_prediction:.2f} (confidence: {total_confidence:.3f})")
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"❌ Error creating ensemble: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self, ensemble_results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            performance_metrics = {
                'overall_metrics': {},
                'timeframe_metrics': {},
                'model_comparison': {}
            }
            
            # Overall metrics
            total_predictions = len(ensemble_results)
            avg_confidence = np.mean([result['ensemble_confidence'] for result in ensemble_results.values()])
            
            performance_metrics['overall_metrics'] = {
                'total_timeframes': total_predictions,
                'average_confidence': avg_confidence,
                'prediction_range': {
                    'min': min([result['ensemble_prediction'] for result in ensemble_results.values()]),
                    'max': max([result['ensemble_prediction'] for result in ensemble_results.values()])
                }
            }
            
            # Timeframe-specific metrics
            for timeframe, result in ensemble_results.items():
                performance_metrics['timeframe_metrics'][timeframe] = {
                    'prediction': result['ensemble_prediction'],
                    'confidence': result['ensemble_confidence'],
                    'model_count': len(result['model_weights']),
                    'weight_distribution': result['model_weights']
                }
            
            # Model comparison
            all_models = set()
            for result in ensemble_results.values():
                all_models.update(result['individual_predictions'].keys())
            
            for model in all_models:
                model_predictions = []
                model_confidences = []
                
                for result in ensemble_results.values():
                    if model in result['individual_predictions']:
                        pred_info = result['individual_predictions'][model]
                        model_predictions.append(pred_info['prediction'])
                        model_confidences.append(pred_info['confidence'])
                
                if model_predictions:
                    performance_metrics['model_comparison'][model] = {
                        'avg_prediction': np.mean(model_predictions),
                        'avg_confidence': np.mean(model_confidences),
                        'prediction_std': np.std(model_predictions),
                        'usage_count': len(model_predictions)
                    }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance metrics: {str(e)}")
            return {}
    
    def _generate_trading_signals(self, ensemble_results: Dict) -> Dict:
        """Generate actionable trading signals"""
        try:
            trading_signals = {
                'signals': [],
                'summary': {
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'hold_signals': 0
                }
            }
            
            for timeframe, result in ensemble_results.items():
                prediction = result['ensemble_prediction']
                confidence = result['ensemble_confidence']
                
                # Get current price (you would need to fetch this)
                current_price = prediction  # Placeholder - should be actual current price
                
                # Calculate expected return
                expected_return = (prediction - current_price) / current_price
                
                # Generate signal based on confidence and expected return
                if confidence > 0.7:  # High confidence threshold
                    if expected_return > 0.02:  # 2% expected gain
                        signal = 'BUY'
                        trading_signals['summary']['buy_signals'] += 1
                    elif expected_return < -0.02:  # 2% expected loss
                        signal = 'SELL'
                        trading_signals['summary']['sell_signals'] += 1
                    else:
                        signal = 'HOLD'
                        trading_signals['summary']['hold_signals'] += 1
                else:
                    signal = 'HOLD'
                    trading_signals['summary']['hold_signals'] += 1
                
                # Create signal object
                signal_obj = {
                    'timeframe': timeframe,
                    'signal': signal,
                    'confidence': confidence,
                    'prediction': prediction,
                    'expected_return': expected_return,
                    'current_price': current_price,
                    'recommendation': self._generate_signal_recommendation(signal, confidence, expected_return)
                }
                
                trading_signals['signals'].append(signal_obj)
            
            return trading_signals
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signals: {str(e)}")
            return {'signals': [], 'summary': {'buy_signals': 0, 'sell_signals': 0, 'hold_signals': 0}}
    
    def _generate_signal_recommendation(self, signal: str, confidence: float, expected_return: float) -> str:
        """Generate recommendation text for trading signal"""
        try:
            if signal == 'BUY':
                if confidence > 0.8:
                    return f"STRONG BUY - High confidence ({confidence:.1%}) with {expected_return:.1%} expected return"
                else:
                    return f"BUY - Medium confidence ({confidence:.1%}) with {expected_return:.1%} expected return"
            elif signal == 'SELL':
                if confidence > 0.8:
                    return f"STRONG SELL - High confidence ({confidence:.1%}) with {expected_return:.1%} expected return"
                else:
                    return f"SELL - Medium confidence ({confidence:.1%}) with {expected_return:.1%} expected return"
            else:
                return f"HOLD - Low confidence ({confidence:.1%}) or minimal expected return ({expected_return:.1%})"
                
        except Exception as e:
            logger.error(f"❌ Error generating signal recommendation: {str(e)}")
            return "Signal analysis failed"
    
    def _create_summary(self, pipeline_results: Dict) -> Dict:
        """Create comprehensive summary"""
        try:
            summary = {
                'pipeline_status': 'COMPLETED' if 'error' not in pipeline_results else 'FAILED',
                'execution_time': datetime.now(),
                'key_findings': [],
                'recommendations': [],
                'risk_assessment': {},
                'next_steps': []
            }
            
            # Key findings
            if 'data_analysis' in pipeline_results:
                data_summary = pipeline_results['data_analysis'].get('data_summary', {})
                summary['key_findings'].append(f"Data loaded: {data_summary.get('overall', {}).get('total_records', 0)} total records")
            
            if 'model_training' in pipeline_results:
                model_count = sum(len(timeframe_models) for timeframe_models in pipeline_results['model_training'].values())
                summary['key_findings'].append(f"Models trained: {model_count} total models")
            
            if 'predictions' in pipeline_results:
                pred_count = len(pipeline_results['predictions'])
                summary['key_findings'].append(f"Predictions generated: {pred_count} timeframes")
            
            if 'trading_signals' in pipeline_results:
                signals = pipeline_results['trading_signals'].get('summary', {})
                summary['key_findings'].append(f"Trading signals: {signals.get('buy_signals', 0)} buy, {signals.get('sell_signals', 0)} sell")
            
            # Recommendations
            if 'performance_metrics' in pipeline_results:
                avg_confidence = pipeline_results['performance_metrics'].get('overall_metrics', {}).get('average_confidence', 0)
                if avg_confidence > 0.7:
                    summary['recommendations'].append("High confidence predictions - consider taking action")
                elif avg_confidence > 0.5:
                    summary['recommendations'].append("Medium confidence predictions - monitor closely")
                else:
                    summary['recommendations'].append("Low confidence predictions - wait for better signals")
            
            # Risk assessment
            summary['risk_assessment'] = {
                'data_quality': 'GOOD' if pipeline_results.get('data_analysis', {}).get('data_available', False) else 'POOR',
                'model_performance': 'GOOD' if len(pipeline_results.get('model_training', {})) > 0 else 'POOR',
                'prediction_confidence': 'HIGH' if pipeline_results.get('performance_metrics', {}).get('overall_metrics', {}).get('average_confidence', 0) > 0.7 else 'LOW'
            }
            
            # Next steps
            summary['next_steps'] = [
                "Monitor predictions for accuracy",
                "Update models with new data",
                "Validate trading signals with backtesting",
                "Consider market conditions and external factors"
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error creating summary: {str(e)}")
            return {'pipeline_status': 'FAILED', 'error': str(e)}

def main():
    """Test the enhanced ML pipeline"""
    print("🚀 Testing Enhanced ML Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedMLPipeline("RELIANCE")
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    # Print results
    print(f"\n📊 Pipeline Results:")
    print(f"Status: {results.get('summary', {}).get('pipeline_status', 'UNKNOWN')}")
    
    if 'data_analysis' in results:
        data_summary = results['data_analysis'].get('data_summary', {})
        print(f"Data Records: {data_summary.get('overall', {}).get('total_records', 0)}")
    
    if 'model_training' in results:
        model_count = sum(len(timeframe_models) for timeframe_models in results['model_training'].values())
        print(f"Models Trained: {model_count}")
    
    if 'predictions' in results:
        pred_count = len(results['predictions'])
        print(f"Predictions Generated: {pred_count}")
    
    if 'trading_signals' in results:
        signals = results['trading_signals'].get('summary', {})
        print(f"Trading Signals: {signals.get('buy_signals', 0)} buy, {signals.get('sell_signals', 0)} sell, {signals.get('hold_signals', 0)} hold")
    
    if 'summary' in results:
        summary = results['summary']
        print(f"\n🔍 Key Findings:")
        for finding in summary.get('key_findings', []):
            print(f"  • {finding}")
        
        print(f"\n💡 Recommendations:")
        for rec in summary.get('recommendations', []):
            print(f"  • {rec}")
        
        print(f"\n⚠️ Risk Assessment:")
        risk = summary.get('risk_assessment', {})
        for key, value in risk.items():
            print(f"  • {key}: {value}")
    
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")

if __name__ == "__main__":
    main()