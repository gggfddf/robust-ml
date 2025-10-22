#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - COMPLETE ML PIPELINE
Comprehensive Machine Learning Pipeline for Reliance Industries Analysis
Production-Grade Implementation with Full Feature Set
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Advanced ML
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Custom imports
from data.live_data_loader import DataLoader
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from config import get_config

class CompleteMLPipeline:
    """Complete Machine Learning Pipeline for Market Analysis"""
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.config = get_config()
        self.config.SYMBOL = symbol
        self.symbol = symbol
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        
        # ML Components
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.predictions = {}
        self.performance_metrics = {}
        
        # Results storage
        self.results = {
            'data_quality': {},
            'feature_engineering': {},
            'model_performance': {},
            'predictions': {},
            'insights': {}
        }
        
        print(f"🚀 Initializing Complete ML Pipeline for {symbol}")
        print("=" * 60)
    
    def execute_complete_pipeline(self):
        """Execute the complete ML pipeline end-to-end"""
        print("🎯 EXECUTING COMPLETE ML PIPELINE")
        print("=" * 60)
        
        try:
            # Stage 1: Data Collection & Quality Assessment
            print("\n📊 STAGE 1: DATA COLLECTION & QUALITY ASSESSMENT")
            print("-" * 50)
            data = self._collect_and_assess_data()
            
            # Stage 2: Feature Engineering & Selection
            print("\n🔧 STAGE 2: FEATURE ENGINEERING & SELECTION")
            print("-" * 50)
            features = self._engineer_features(data)
            
            # Stage 3: Data Preprocessing
            print("\n⚙️ STAGE 3: DATA PREPROCESSING")
            print("-" * 50)
            processed_data = self._preprocess_data(features)
            
            # Stage 4: Model Training & Optimization
            print("\n🤖 STAGE 4: MODEL TRAINING & OPTIMIZATION")
            print("-" * 50)
            trained_models = self._train_models(processed_data)
            
            # Stage 5: Model Evaluation & Ensemble
            print("\n📈 STAGE 5: MODEL EVALUATION & ENSEMBLE")
            print("-" * 50)
            ensemble_results = self._evaluate_and_ensemble(trained_models, processed_data)
            
            # Stage 6: Advanced Analytics & Insights
            print("\n🧠 STAGE 6: ADVANCED ANALYTICS & INSIGHTS")
            print("-" * 50)
            insights = self._generate_advanced_insights(ensemble_results, processed_data)
            
            # Stage 7: Model Deployment & Persistence
            print("\n🚀 STAGE 7: MODEL DEPLOYMENT & PERSISTENCE")
            print("-" * 50)
            deployment_results = self._deploy_models(trained_models)
            
            # Stage 8: Performance Reporting
            print("\n📋 STAGE 8: PERFORMANCE REPORTING")
            print("-" * 50)
            self._generate_comprehensive_report()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Pipeline Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _collect_and_assess_data(self):
        """Stage 1: Collect and assess data quality"""
        print("📥 Collecting market data...")
        
        # Collect data for all timeframes
        timeframes = ['1d', '1w']  # Focus on daily and weekly for ML
        data_dict = {}
        
        for tf in timeframes:
            print(f"  📊 Loading {tf} data...")
            try:
                data = self.data_loader.fetch_single_timeframe_data(self.symbol, tf)
                if data is not None and len(data) > 0:
                    data_dict[tf] = data
                    print(f"    ✅ {tf}: {len(data)} records loaded")
                else:
                    print(f"    ❌ {tf}: No data available")
            except Exception as e:
                print(f"    ❌ {tf}: Error - {str(e)}")
        
        # Use daily data as primary dataset
        if '1d' in data_dict:
            data = data_dict['1d'].copy()
            
            # Data quality assessment
            quality_metrics = {
                'total_records': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'missing_values': data.isnull().sum().sum(),
                'duplicates': data.duplicated().sum(),
                'data_types': data.dtypes.value_counts().to_dict(),
                'outliers_detected': self._detect_outliers(data).sum()
            }
            
            self.results['data_quality'] = quality_metrics
            
            print(f"  📊 Data Quality Metrics:")
            for metric, value in quality_metrics.items():
                print(f"    {metric}: {value}")
            
            return data
        else:
            raise ValueError("No daily data available for ML pipeline")
    
    def _detect_outliers(self, data, method='iqr'):
        """Detect outliers in price data"""
        if method == 'iqr':
            Q1 = data['Close'].quantile(0.25)
            Q3 = data['Close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data['Close'] < lower_bound) | (data['Close'] > upper_bound)
        return pd.Series([False] * len(data))
    
    def _engineer_features(self, data):
        """Stage 2: Engineer advanced features"""
        print("🔧 Engineering features...")
        
        # Create copy for feature engineering
        df = data.copy()
        
        # Technical indicators (already calculated)
        print("  📊 Adding technical indicators...")
        try:
            indicators = self.technical_analyzer.calculate_all_indicators(df)
            df = pd.concat([df, indicators], axis=1)
            print(f"    ✅ Added {len(indicators.columns)} technical indicators")
        except Exception as e:
            print(f"    ⚠️ Technical indicators error: {str(e)}")
        
        # Price-based features
        print("  💰 Creating price-based features...")
        df['price_change'] = df['Close'].pct_change()
        df['price_change_2d'] = df['Close'].pct_change(2)
        df['price_change_5d'] = df['Close'].pct_change(5)
        df['price_change_10d'] = df['Close'].pct_change(10)
        
        # Volatility features
        df['volatility_5d'] = df['price_change'].rolling(5).std()
        df['volatility_10d'] = df['price_change'].rolling(10).std()
        df['volatility_20d'] = df['price_change'].rolling(20).std()
        
        # Volume features
        df['volume_ma_5'] = df['Volume'].rolling(5).mean()
        df['volume_ma_10'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_5']
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_ma_{window}'] = df['Close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'volume_ma_{window}'] = df['Volume'].rolling(window).mean()
        
        # Target variable (next day's price change)
        df['target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        feature_metrics = {
            'total_features': len(df_clean.columns),
            'price_features': len([col for col in df_clean.columns if 'price' in col.lower()]),
            'volume_features': len([col for col in df_clean.columns if 'volume' in col.lower()]),
            'technical_features': len([col for col in df_clean.columns if any(ind in col.lower() for ind in ['rsi', 'macd', 'bollinger', 'stoch'])]),
            'time_features': len([col for col in df_clean.columns if any(time in col.lower() for time in ['day', 'month', 'quarter', 'year'])]),
            'clean_records': len(df_clean)
        }
        
        self.results['feature_engineering'] = feature_metrics
        
        print(f"  📊 Feature Engineering Complete:")
        for metric, value in feature_metrics.items():
            print(f"    {metric}: {value}")
        
        return df_clean
    
    def _preprocess_data(self, data):
        """Stage 3: Preprocess data for ML"""
        print("⚙️ Preprocessing data...")
        
        # Separate features and target
        target_col = 'target'
        feature_cols = [col for col in data.columns if col != target_col and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = data[feature_cols]
        y = data[target_col]
        
        print(f"  📊 Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Feature selection
        print("  🔍 Performing feature selection...")
        selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"    ✅ Selected {len(selected_features)} best features")
        
        # Split data (time series split)
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X_selected))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        self.scalers['standard'] = scaler
        self.scalers['feature_selector'] = selector
        
        preprocessing_results = {
            'original_features': X.shape[1],
            'selected_features': len(selected_features),
            'selected_feature_names': selected_features,
            'train_test_splits': len(splits),
            'scaling_method': 'StandardScaler'
        }
        
        self.results['preprocessing'] = preprocessing_results
        
        return {
            'X': X_scaled,
            'y': y.values,
            'feature_names': selected_features,
            'splits': splits,
            'original_data': data
        }
    
    def _train_models(self, processed_data):
        """Stage 4: Train multiple ML models"""
        print("🤖 Training ML models...")
        
        X = processed_data['X']
        y = processed_data['y']
        splits = processed_data['splits']
        feature_names = processed_data['feature_names']
        
        # Model configurations
        models_config = {
            'RandomForest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'LightGBM': {
                'model': LGBMRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }
        
        trained_models = {}
        
        for model_name, config in models_config.items():
            print(f"  🚀 Training {model_name}...")
            
            try:
                # Cross-validation with hyperparameter tuning
                model = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=splits,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                model.fit(X, y)
                
                # Store model and results
                trained_models[model_name] = {
                    'model': model.best_estimator_,
                    'best_params': model.best_params_,
                    'best_score': -model.best_score_,
                    'cv_scores': -model.cv_results_['mean_test_score']
                }
                
                # Feature importance
                if hasattr(model.best_estimator_, 'feature_importances_'):
                    importance = model.best_estimator_.feature_importances_
                    self.feature_importance[model_name] = dict(zip(feature_names, importance))
                
                print(f"    ✅ {model_name}: Best Score = {model.best_score_:.6f}")
                
            except Exception as e:
                print(f"    ❌ {model_name}: Error - {str(e)}")
        
        self.models = trained_models
        
        return trained_models
    
    def _evaluate_and_ensemble(self, trained_models, processed_data):
        """Stage 5: Evaluate models and create ensemble"""
        print("📈 Evaluating models and creating ensemble...")
        
        X = processed_data['X']
        y = processed_data['y']
        splits = processed_data['splits']
        
        # Evaluate each model
        evaluation_results = {}
        
        for model_name, model_info in trained_models.items():
            print(f"  📊 Evaluating {model_name}...")
            
            model = model_info['model']
            
            # Cross-validation predictions
            cv_predictions = []
            cv_actuals = []
            
            for train_idx, test_idx in splits:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                cv_predictions.extend(pred)
                cv_actuals.extend(y_test)
            
            # Calculate metrics
            mse = mean_squared_error(cv_actuals, cv_predictions)
            mae = mean_absolute_error(cv_actuals, cv_predictions)
            r2 = r2_score(cv_actuals, cv_predictions)
            
            evaluation_results[model_name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'predictions': cv_predictions,
                'actuals': cv_actuals
            }
            
            print(f"    📊 {model_name} Metrics:")
            print(f"      MSE: {mse:.6f}")
            print(f"      MAE: {mae:.6f}")
            print(f"      R²: {r2:.6f}")
            print(f"      RMSE: {np.sqrt(mse):.6f}")
        
        # Create ensemble
        print("  🤝 Creating ensemble model...")
        
        # Simple weighted average ensemble
        weights = {}
        total_score = 0
        
        for model_name, results in evaluation_results.items():
            # Use R² score as weight (higher is better)
            weight = max(0, results['r2'])  # Ensure non-negative
            weights[model_name] = weight
            total_score += weight
        
        # Normalize weights
        if total_score > 0:
            weights = {k: v/total_score for k, v in weights.items()}
        else:
            # Equal weights if all models perform poorly
            weights = {k: 1.0/len(weights) for k in weights.keys()}
        
        # Ensemble predictions
        ensemble_predictions = np.zeros(len(y))
        
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            pred = model.predict(X)
            ensemble_predictions += weights[model_name] * pred
        
        # Ensemble metrics
        ensemble_mse = mean_squared_error(y, ensemble_predictions)
        ensemble_mae = mean_absolute_error(y, ensemble_predictions)
        ensemble_r2 = r2_score(y, ensemble_predictions)
        
        evaluation_results['Ensemble'] = {
            'mse': ensemble_mse,
            'mae': ensemble_mae,
            'r2': ensemble_r2,
            'rmse': np.sqrt(ensemble_mse),
            'predictions': ensemble_predictions,
            'actuals': y,
            'weights': weights
        }
        
        self.performance_metrics = evaluation_results
        self.predictions = ensemble_predictions
        
        print(f"  🏆 Ensemble Performance:")
        print(f"    MSE: {ensemble_mse:.6f}")
        print(f"    MAE: {ensemble_mae:.6f}")
        print(f"    R²: {ensemble_r2:.6f}")
        print(f"    RMSE: {np.sqrt(ensemble_mse):.6f}")
        
        return evaluation_results
    
    def _generate_advanced_insights(self, evaluation_results, processed_data):
        """Stage 6: Generate advanced analytics and insights"""
        print("🧠 Generating advanced insights...")
        
        insights = {}
        
        # Model performance comparison
        print("  📊 Analyzing model performance...")
        performance_comparison = {}
        
        for model_name, results in evaluation_results.items():
            performance_comparison[model_name] = {
                'R²': results['r2'],
                'RMSE': results['rmse'],
                'MAE': results['mae']
            }
        
        insights['model_performance'] = performance_comparison
        
        # Feature importance analysis
        print("  🔍 Analyzing feature importance...")
        if self.feature_importance:
            # Aggregate feature importance across models
            all_features = set()
            for model_importance in self.feature_importance.values():
                all_features.update(model_importance.keys())
            
            aggregated_importance = {}
            for feature in all_features:
                importance_scores = []
                for model_importance in self.feature_importance.values():
                    if feature in model_importance:
                        importance_scores.append(model_importance[feature])
                
                if importance_scores:
                    aggregated_importance[feature] = np.mean(importance_scores)
            
            # Top features
            top_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            insights['top_features'] = dict(top_features)
        
        # Prediction analysis
        print("  📈 Analyzing predictions...")
        ensemble_results = evaluation_results.get('Ensemble', {})
        if ensemble_results:
            predictions = ensemble_results['predictions']
            actuals = ensemble_results['actuals']
            
            # Prediction accuracy by magnitude
            prediction_accuracy = {
                'high_volatility': {'correct': 0, 'total': 0},
                'medium_volatility': {'correct': 0, 'total': 0},
                'low_volatility': {'correct': 0, 'total': 0}
            }
            
            for pred, actual in zip(predictions, actuals):
                error = abs(pred - actual)
                actual_magnitude = abs(actual)
                
                if actual_magnitude > 0.02:  # High volatility
                    prediction_accuracy['high_volatility']['total'] += 1
                    if error < 0.01:  # Good prediction
                        prediction_accuracy['high_volatility']['correct'] += 1
                elif actual_magnitude > 0.01:  # Medium volatility
                    prediction_accuracy['medium_volatility']['total'] += 1
                    if error < 0.005:  # Good prediction
                        prediction_accuracy['medium_volatility']['correct'] += 1
                else:  # Low volatility
                    prediction_accuracy['low_volatility']['total'] += 1
                    if error < 0.002:  # Good prediction
                        prediction_accuracy['low_volatility']['correct'] += 1
            
            insights['prediction_accuracy'] = prediction_accuracy
        
        # Market regime analysis
        print("  🌊 Analyzing market regimes...")
        original_data = processed_data['original_data']
        
        # Calculate market volatility regimes
        volatility = original_data['price_change'].rolling(20).std()
        high_vol_periods = volatility > volatility.quantile(0.75)
        low_vol_periods = volatility < volatility.quantile(0.25)
        
        insights['market_regimes'] = {
            'high_volatility_periods': high_vol_periods.sum(),
            'low_volatility_periods': low_vol_periods.sum(),
            'total_periods': len(volatility.dropna()),
            'avg_volatility': volatility.mean()
        }
        
        self.results['insights'] = insights
        
        return insights
    
    def _deploy_models(self, trained_models):
        """Stage 7: Deploy and persist models"""
        print("🚀 Deploying models...")
        
        # Create models directory
        os.makedirs('models/ml_models', exist_ok=True)
        
        deployment_results = {}
        
        # Save individual models
        for model_name, model_info in trained_models.items():
            try:
                model_path = f'models/ml_models/{model_name.lower()}_model.pkl'
                joblib.dump(model_info['model'], model_path)
                deployment_results[model_name] = {
                    'path': model_path,
                    'status': 'saved',
                    'best_params': model_info['best_params']
                }
                print(f"  ✅ {model_name}: Saved to {model_path}")
            except Exception as e:
                print(f"  ❌ {model_name}: Save failed - {str(e)}")
                deployment_results[model_name] = {'status': 'failed', 'error': str(e)}
        
        # Save scalers and feature selector
        try:
            scaler_path = 'models/ml_models/scalers.pkl'
            joblib.dump(self.scalers, scaler_path)
            deployment_results['scalers'] = {'path': scaler_path, 'status': 'saved'}
            print(f"  ✅ Scalers: Saved to {scaler_path}")
        except Exception as e:
            print(f"  ❌ Scalers: Save failed - {str(e)}")
        
        # Save ensemble weights
        if 'Ensemble' in self.performance_metrics:
            weights = self.performance_metrics['Ensemble']['weights']
            weights_path = 'models/ml_models/ensemble_weights.json'
            import json
            with open(weights_path, 'w') as f:
                json.dump(weights, f, indent=2)
            deployment_results['ensemble_weights'] = {'path': weights_path, 'status': 'saved'}
            print(f"  ✅ Ensemble weights: Saved to {weights_path}")
        
        self.results['deployment'] = deployment_results
        
        return deployment_results
    
    def _generate_comprehensive_report(self):
        """Stage 8: Generate comprehensive performance report"""
        print("📋 Generating comprehensive report...")
        
        # Calculate overall performance score
        if self.performance_metrics and 'Ensemble' in self.performance_metrics:
            ensemble_r2 = self.performance_metrics['Ensemble']['r2']
            ensemble_rmse = self.performance_metrics['Ensemble']['rmse']
            
            # Convert to 0-10 scale
            r2_score_10 = max(0, min(10, ensemble_r2 * 10))  # R² to 0-10 scale
            rmse_score_10 = max(0, min(10, (1 - ensemble_rmse) * 10))  # RMSE to 0-10 scale
            
            overall_score = (r2_score_10 + rmse_score_10) / 2
        else:
            overall_score = 0
        
        # Generate summary
        summary = {
            'symbol': self.symbol,
            'pipeline_completion': 'SUCCESS',
            'overall_score': overall_score,
            'total_models_trained': len(self.models),
            'best_model': max(self.performance_metrics.items(), key=lambda x: x[1]['r2'])[0] if self.performance_metrics else 'None',
            'ensemble_accuracy': self.performance_metrics.get('Ensemble', {}).get('r2', 0),
            'total_features_used': len(self.results.get('preprocessing', {}).get('selected_features', [])),
            'data_quality_score': self._calculate_data_quality_score(),
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.results['summary'] = summary
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 COMPLETE ML PIPELINE - FINAL RESULTS")
        print("=" * 60)
        print(f"📊 Symbol: {summary['symbol']}")
        print(f"🏆 Overall Score: {summary['overall_score']:.2f}/10")
        print(f"🤖 Models Trained: {summary['total_models_trained']}")
        print(f"🎯 Best Model: {summary['best_model']}")
        print(f"📈 Ensemble R²: {summary['ensemble_accuracy']:.4f}")
        print(f"🔧 Features Used: {summary['total_features_used']}")
        print(f"📊 Data Quality: {summary['data_quality_score']:.2f}/10")
        print(f"⏰ Completion Time: {summary['execution_time']}")
        print("=" * 60)
        
        return summary
    
    def _calculate_data_quality_score(self):
        """Calculate data quality score"""
        quality = self.results.get('data_quality', {})
        
        if not quality:
            return 0
        
        # Calculate score based on quality metrics
        total_records = quality.get('total_records', 0)
        missing_values = quality.get('missing_values', 0)
        duplicates = quality.get('duplicates', 0)
        
        if total_records == 0:
            return 0
        
        # Score components
        completeness = 1 - (missing_values / (total_records * len(quality.get('data_types', {}))))
        uniqueness = 1 - (duplicates / total_records)
        
        # Overall quality score
        quality_score = (completeness + uniqueness) / 2 * 10
        
        return min(10, max(0, quality_score))

def main():
    """Main execution function"""
    print("🚀 ULTIMATE MARKET AI ENGINE - COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Initialize and execute pipeline
    pipeline = CompleteMLPipeline("RELIANCE.NS")
    results = pipeline.execute_complete_pipeline()
    
    if results:
        print("\n✅ ML Pipeline completed successfully!")
        print(f"📊 Final Score: {results['summary']['overall_score']:.2f}/10")
    else:
        print("\n❌ ML Pipeline failed!")
    
    return results

if __name__ == "__main__":
    main()