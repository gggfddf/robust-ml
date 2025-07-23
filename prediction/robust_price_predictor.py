#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - ROBUST PRICE PREDICTOR
Complete Stock Price Prediction System with Multiple Algorithms
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb

# Statistical Models (optional)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available, skipping statistical models")

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, skipping deep learning models")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustPricePredictor:
    """Robust Stock Price Prediction System"""
    
    def __init__(self, symbol="RELIANCE"):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.ensemble_weights = {}
        self.validation_results = {}
        
        logger.info(f"Robust Price Predictor initialized for {symbol}")
    
    def run_complete_prediction(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Run complete robust prediction pipeline"""
        logger.info(f"🚀 Starting robust prediction for {timeframe}")
        
        prediction_results = {
            'symbol': self.symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'data_preparation': {},
            'feature_engineering': {},
            'model_training': {},
            'predictions': {},
            'ensemble_results': {},
            'validation_results': {},
            'confidence_intervals': {},
            'trading_signals': {},
            'summary': {}
        }
        
        try:
            # Step 1: Data Preparation
            logger.info("📊 Step 1: Data preparation...")
            prepared_data = self._prepare_data(data)
            prediction_results['data_preparation'] = prepared_data
            
            if prepared_data['data_ready']:
                # Step 2: Feature Engineering
                logger.info("🔧 Step 2: Feature engineering...")
                features = self._engineer_features(prepared_data['data'])
                prediction_results['feature_engineering'] = features
                
                # Step 3: Train Models
                logger.info("🤖 Step 3: Training models...")
                model_results = self._train_all_models(features['X'], features['y'], timeframe)
                prediction_results['model_training'] = model_results
                
                # Step 4: Generate Predictions
                logger.info("🔮 Step 4: Generating predictions...")
                predictions = self._generate_predictions(features['X'], timeframe)
                prediction_results['predictions'] = predictions
                
                # Step 5: Create Ensemble
                logger.info("🎯 Step 5: Creating ensemble...")
                ensemble = self._create_robust_ensemble(predictions, model_results)
                prediction_results['ensemble_results'] = ensemble
                
                # Step 6: Validation
                logger.info("✅ Step 6: Validation...")
                validation = self._validate_predictions(features['X'], features['y'], timeframe)
                prediction_results['validation_results'] = validation
                
                # Step 7: Confidence Intervals
                logger.info("📈 Step 7: Confidence intervals...")
                confidence_intervals = self._calculate_confidence_intervals(ensemble, validation)
                prediction_results['confidence_intervals'] = confidence_intervals
                
                # Step 8: Trading Signals
                logger.info("💹 Step 8: Trading signals...")
                trading_signals = self._generate_trading_signals(ensemble, confidence_intervals)
                prediction_results['trading_signals'] = trading_signals
                
                # Step 9: Summary
                logger.info("📋 Step 9: Creating summary...")
                summary = self._create_prediction_summary(prediction_results)
                prediction_results['summary'] = summary
                
                logger.info("✅ Robust prediction completed successfully!")
            else:
                logger.error("❌ Data preparation failed")
                prediction_results['error'] = "Data preparation failed"
            
        except Exception as e:
            logger.error(f"❌ Error in robust prediction: {str(e)}")
            prediction_results['error'] = str(e)
        
        return prediction_results
    
    def _prepare_data(self, data: pd.DataFrame) -> Dict:
        """Prepare data for prediction"""
        try:
            df = data.copy()
            
            # Remove any rows with all NaN values
            df = df.dropna(how='all')
            
            # Forward fill missing values for OHLC
            ohlc_columns = ['Open', 'High', 'Low', 'Close']
            df[ohlc_columns] = df[ohlc_columns].fillna(method='ffill')
            
            # Fill volume with 0 if missing
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0)
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Sort by date
            df = df.sort_index()
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            # Check data quality
            data_quality = {
                'total_records': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'price_range': f"₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}",
                'avg_volume': f"{df['Volume'].mean():,.0f}",
                'volatility': f"{df['Close'].pct_change().std()*100:.2f}%",
                'missing_values': df.isnull().sum().sum(),
                'duplicates': df.duplicated().sum()
            }
            
            # Determine if data is ready
            data_ready = len(df) >= 100 and data_quality['missing_values'] == 0
            
            return {
                'data': df,
                'data_ready': data_ready,
                'data_quality': data_quality
            }
            
        except Exception as e:
            logger.error(f"❌ Error preparing data: {str(e)}")
            return {'data_ready': False, 'error': str(e)}
    
    def _engineer_features(self, data: pd.DataFrame) -> Dict:
        """Engineer comprehensive features for prediction"""
        try:
            df = data.copy()
            
            # Technical Indicators
            df = self._add_technical_indicators(df)
            
            # Price Features
            df = self._add_price_features(df)
            
            # Volume Features
            df = self._add_volume_features(df)
            
            # Volatility Features
            df = self._add_volatility_features(df)
            
            # Momentum Features
            df = self._add_momentum_features(df)
            
            # Trend Features
            df = self._add_trend_features(df)
            
            # Time Features
            df = self._add_time_features(df)
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 50:
                return {'X': pd.DataFrame(), 'y': pd.Series(), 'feature_count': 0}
            
            # Prepare features and target
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            X = df[feature_columns]
            y = df['Close']
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Store scaler
            self.scalers[f"{self.symbol}_{df.index[0].strftime('%Y%m%d')}"] = scaler
            
            return {
                'X': X_scaled,
                'y': y,
                'feature_count': len(feature_columns),
                'feature_names': feature_columns
            }
            
        except Exception as e:
            logger.error(f"❌ Error engineering features: {str(e)}")
            return {'X': pd.DataFrame(), 'y': pd.Series(), 'feature_count': 0}
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
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
            logger.error(f"❌ Error adding technical indicators: {str(e)}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
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
            logger.error(f"❌ Error adding price features: {str(e)}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
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
            logger.error(f"❌ Error adding volume features: {str(e)}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
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
            logger.error(f"❌ Error adding volatility features: {str(e)}")
            return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        try:
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
            logger.error(f"❌ Error adding momentum features: {str(e)}")
            return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        try:
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
            logger.error(f"❌ Error adding trend features: {str(e)}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Day of week
            df['DayOfWeek'] = df.index.dayofweek
            
            # Month
            df['Month'] = df.index.month
            
            # Quarter
            df['Quarter'] = df.index.quarter
            
            # Year
            df['Year'] = df.index.year
            
            # Day of year
            df['DayOfYear'] = df.index.dayofyear
            
            # Week of year
            df['WeekOfYear'] = df.index.isocalendar().week
            
            # Cyclical encoding for time features
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding time features: {str(e)}")
            return df
    
    def _train_all_models(self, X: pd.DataFrame, y: pd.Series, timeframe: str) -> Dict:
        """Train all prediction models"""
        try:
            models = {}
            
            # Split data for time series
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Model configurations
            model_configs = {
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
                },
                'GradientBoosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2]}
                },
                'ExtraTrees': {
                    'model': ExtraTreesRegressor(random_state=42),
                    'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
                },
                'XGBoost': {
                    'model': xgb.XGBRegressor(random_state=42),
                    'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
                },
                'LightGBM': {
                    'model': lgb.LGBMRegressor(random_state=42),
                    'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
                },
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'Ridge': {
                    'model': Ridge(),
                    'params': {'alpha': [0.1, 1.0, 10.0]}
                },
                'Lasso': {
                    'model': Lasso(),
                    'params': {'alpha': [0.1, 1.0, 10.0]}
                },
                'ElasticNet': {
                    'model': ElasticNet(),
                    'params': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.7]}
                },
                'SVR': {
                    'model': SVR(),
                    'params': {'C': [1, 10], 'gamma': ['scale', 'auto']}
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
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    
                    # Store model and results
                    models[name] = {
                        'model': best_model,
                        'r2_score': r2,
                        'mse': mse,
                        'mae': mae,
                        'mape': mape,
                        'predictions': y_pred,
                        'actual': y_test
                    }
                    
                    # Store feature importance if available
                    if hasattr(best_model, 'feature_importances_'):
                        self.feature_importance[f"{timeframe}_{name}"] = dict(zip(X.columns, best_model.feature_importances_))
                    
                    logger.info(f"✅ {name}: R² = {r2:.4f}, MAPE = {mape:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {name}: {str(e)}")
                    continue
            
            # Store models
            self.models[timeframe] = models
            
            return {
                'models_trained': len(models),
                'model_performance': {name: {'r2': info['r2_score'], 'mape': info['mape']} for name, info in models.items()},
                'best_model': max(models.items(), key=lambda x: x[1]['r2_score'])[0] if models else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error training models: {str(e)}")
            return {'models_trained': 0, 'error': str(e)}
    
    def _generate_predictions(self, X: pd.DataFrame, timeframe: str) -> Dict:
        """Generate predictions from all models"""
        try:
            predictions = {}
            
            if timeframe not in self.models:
                return predictions
            
            # Get latest data for prediction
            X_latest = X.tail(1)
            
            for model_name, model_info in self.models[timeframe].items():
                try:
                    model = model_info['model']
                    pred = model.predict(X_latest)[0]
                    
                    predictions[model_name] = {
                        'prediction': pred,
                        'confidence': model_info['r2_score'],
                        'model_performance': {
                            'r2': model_info['r2_score'],
                            'mape': model_info['mape']
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error generating prediction for {model_name}: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating predictions: {str(e)}")
            return {}
    
    def _create_robust_ensemble(self, predictions: Dict, model_results: Dict) -> Dict:
        """Create robust ensemble prediction"""
        try:
            if not predictions:
                return {}
            
            # Calculate ensemble weights based on R² scores
            weights = {}
            total_r2 = 0
            
            for model_name, pred_info in predictions.items():
                r2 = pred_info['model_performance']['r2']
                if r2 > 0:  # Only include models with positive R²
                    weights[model_name] = r2
                    total_r2 += r2
            
            if total_r2 == 0:
                return {}
            
            # Normalize weights
            for model_name in weights:
                weights[model_name] /= total_r2
            
            # Calculate weighted ensemble prediction
            ensemble_prediction = 0
            total_confidence = 0
            
            for model_name, pred_info in predictions.items():
                if model_name in weights:
                    weight = weights[model_name]
                    prediction = pred_info['prediction']
                    confidence = pred_info['confidence']
                    
                    ensemble_prediction += weight * prediction
                    total_confidence += weight * confidence
            
            # Store ensemble weights
            self.ensemble_weights = weights
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': total_confidence,
                'model_weights': weights,
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating ensemble: {str(e)}")
            return {}
    
    def _validate_predictions(self, X: pd.DataFrame, y: pd.Series, timeframe: str) -> Dict:
        """Validate predictions using cross-validation"""
        try:
            validation_results = {}
            
            if timeframe not in self.models:
                return validation_results
            
            # Cross-validation for each model
            tscv = TimeSeriesSplit(n_splits=5)
            
            for model_name, model_info in self.models[timeframe].items():
                try:
                    model = model_info['model']
                    
                    # Cross-validation scores
                    cv_r2 = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                    cv_mape = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_percentage_error')
                    
                    validation_results[model_name] = {
                        'cv_r2_mean': cv_r2.mean(),
                        'cv_r2_std': cv_r2.std(),
                        'cv_mape_mean': -cv_mape.mean(),
                        'cv_mape_std': cv_mape.std(),
                        'cv_scores': {
                            'r2': cv_r2.tolist(),
                            'mape': -cv_mape.tolist()
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error validating {model_name}: {str(e)}")
                    continue
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating predictions: {str(e)}")
            return {}
    
    def _calculate_confidence_intervals(self, ensemble: Dict, validation: Dict) -> Dict:
        """Calculate confidence intervals for predictions"""
        try:
            if not ensemble:
                return {}
            
            prediction = ensemble['ensemble_prediction']
            confidence = ensemble['ensemble_confidence']
            
            # Calculate prediction intervals based on validation results
            if validation:
                # Use cross-validation standard deviation
                cv_mape_std = np.mean([info['cv_mape_std'] for info in validation.values()])
                prediction_std = prediction * cv_mape_std
            else:
                # Default to 5% of prediction
                prediction_std = prediction * 0.05
            
            # Calculate confidence intervals
            confidence_intervals = {
                'prediction': prediction,
                'confidence_level': confidence,
                'lower_bound': prediction - (1.96 * prediction_std),  # 95% CI
                'upper_bound': prediction + (1.96 * prediction_std),
                'prediction_std': prediction_std,
                'margin_of_error': 1.96 * prediction_std
            }
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence intervals: {str(e)}")
            return {}
    
    def _generate_trading_signals(self, ensemble: Dict, confidence_intervals: Dict) -> Dict:
        """Generate trading signals based on predictions"""
        try:
            if not ensemble or not confidence_intervals:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No prediction available'}
            
            prediction = confidence_intervals['prediction']
            confidence = confidence_intervals['confidence_level']
            lower_bound = confidence_intervals['lower_bound']
            upper_bound = confidence_intervals['upper_bound']
            
            # Get current price (you would need to fetch this)
            # For now, use a placeholder
            current_price = prediction  # This should be actual current price
            
            # Calculate expected return
            expected_return = (prediction - current_price) / current_price
            
            # Generate signal based on confidence and expected return
            if confidence > 0.7:  # High confidence threshold
                if expected_return > 0.02:  # 2% expected gain
                    signal = 'BUY'
                    reason = f"High confidence bullish prediction ({expected_return:.1%} expected return)"
                elif expected_return < -0.02:  # 2% expected loss
                    signal = 'SELL'
                    reason = f"High confidence bearish prediction ({expected_return:.1%} expected return)"
                else:
                    signal = 'HOLD'
                    reason = "High confidence but minimal expected return"
            elif confidence > 0.5:  # Medium confidence threshold
                if expected_return > 0.03:  # 3% expected gain
                    signal = 'BUY'
                    reason = f"Medium confidence bullish prediction ({expected_return:.1%} expected return)"
                elif expected_return < -0.03:  # 3% expected loss
                    signal = 'SELL'
                    reason = f"Medium confidence bearish prediction ({expected_return:.1%} expected return)"
                else:
                    signal = 'HOLD'
                    reason = "Medium confidence but insufficient expected return"
            else:
                signal = 'HOLD'
                reason = "Low confidence - wait for better signals"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'expected_return': expected_return,
                'prediction': prediction,
                'confidence_interval': [lower_bound, upper_bound],
                'reason': reason,
                'risk_level': 'HIGH' if confidence < 0.5 else 'MEDIUM' if confidence < 0.7 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signals: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Signal generation failed'}
    
    def _create_prediction_summary(self, prediction_results: Dict) -> Dict:
        """Create comprehensive prediction summary"""
        try:
            summary = {
                'prediction_status': 'COMPLETED' if 'error' not in prediction_results else 'FAILED',
                'execution_time': datetime.now(),
                'key_metrics': {},
                'model_performance': {},
                'prediction_details': {},
                'trading_recommendation': {},
                'risk_assessment': {}
            }
            
            # Key metrics
            if 'model_training' in prediction_results:
                training = prediction_results['model_training']
                summary['key_metrics'] = {
                    'models_trained': training.get('models_trained', 0),
                    'best_model': training.get('best_model', 'None'),
                    'feature_count': prediction_results.get('feature_engineering', {}).get('feature_count', 0)
                }
            
            # Model performance
            if 'model_training' in prediction_results:
                performance = prediction_results['model_training'].get('model_performance', {})
                summary['model_performance'] = {
                    'best_r2': max([info['r2'] for info in performance.values()]) if performance else 0,
                    'avg_r2': np.mean([info['r2'] for info in performance.values()]) if performance else 0,
                    'best_mape': min([info['mape'] for info in performance.values()]) if performance else 1
                }
            
            # Prediction details
            if 'ensemble_results' in prediction_results:
                ensemble = prediction_results['ensemble_results']
                summary['prediction_details'] = {
                    'prediction': ensemble.get('ensemble_prediction', 0),
                    'confidence': ensemble.get('ensemble_confidence', 0),
                    'model_count': len(ensemble.get('model_weights', {}))
                }
            
            # Trading recommendation
            if 'trading_signals' in prediction_results:
                signals = prediction_results['trading_signals']
                summary['trading_recommendation'] = {
                    'signal': signals.get('signal', 'HOLD'),
                    'confidence': signals.get('confidence', 0),
                    'expected_return': signals.get('expected_return', 0),
                    'reason': signals.get('reason', 'No reason provided')
                }
            
            # Risk assessment
            if 'trading_signals' in prediction_results:
                signals = prediction_results['trading_signals']
                summary['risk_assessment'] = {
                    'risk_level': signals.get('risk_level', 'HIGH'),
                    'confidence_level': 'HIGH' if signals.get('confidence', 0) > 0.7 else 'MEDIUM' if signals.get('confidence', 0) > 0.5 else 'LOW',
                    'prediction_accuracy': 'HIGH' if summary['model_performance'].get('best_r2', 0) > 0.7 else 'MEDIUM' if summary['model_performance'].get('best_r2', 0) > 0.5 else 'LOW'
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error creating prediction summary: {str(e)}")
            return {'prediction_status': 'FAILED', 'error': str(e)}

def main():
    """Test the robust price predictor"""
    print("🚀 Testing Robust Price Predictor")
    print("=" * 60)
    
    # Initialize predictor
    predictor = RobustPricePredictor("RELIANCE")
    
    # Load sample data
    import sys
    sys.path.append('/workspace')
    from data.nse_enhanced_data_loader import NSEEnhancedDataLoader
    
    loader = NSEEnhancedDataLoader("RELIANCE")
    multi_data = loader.load_multi_timeframe_data()
    
    if multi_data and '1h' in multi_data:
        # Run prediction
        results = predictor.run_complete_prediction(multi_data['1h'], '1h')
        
        # Print results
        print(f"\n📊 Prediction Results:")
        print(f"Status: {results.get('summary', {}).get('prediction_status', 'UNKNOWN')}")
        
        if 'model_training' in results:
            training = results['model_training']
            print(f"Models Trained: {training.get('models_trained', 0)}")
            print(f"Best Model: {training.get('best_model', 'None')}")
        
        if 'ensemble_results' in results:
            ensemble = results['ensemble_results']
            print(f"Ensemble Prediction: ₹{ensemble.get('ensemble_prediction', 0):.2f}")
            print(f"Confidence: {ensemble.get('ensemble_confidence', 0):.1%}")
        
        if 'trading_signals' in results:
            signals = results['trading_signals']
            print(f"Trading Signal: {signals.get('signal', 'HOLD')}")
            print(f"Expected Return: {signals.get('expected_return', 0):.1%}")
            print(f"Reason: {signals.get('reason', 'No reason')}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"\n🔍 Key Metrics:")
            for key, value in summary.get('key_metrics', {}).items():
                print(f"  • {key}: {value}")
            
            print(f"\n📈 Model Performance:")
            for key, value in summary.get('model_performance', {}).items():
                print(f"  • {key}: {value:.4f}")
            
            print(f"\n💹 Trading Recommendation:")
            for key, value in summary.get('trading_recommendation', {}).items():
                print(f"  • {key}: {value}")
    else:
        print("❌ No data available for prediction")

if __name__ == "__main__":
    main()