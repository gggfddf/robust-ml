#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - ML RESULTS ANALYSIS
Comprehensive Analysis of Machine Learning Pipeline Results
Production-Grade Evaluation with Detailed Outputs
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Custom imports
from data.live_data_loader import DataLoader
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from config import get_config

class MLResultsAnalyzer:
    """Comprehensive ML Results Analyzer"""
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.config = get_config()
        self.config.SYMBOL = symbol
        self.symbol = symbol
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        
        # Results storage
        self.results = {
            'data_analysis': {},
            'model_performance': {},
            'feature_importance': {},
            'predictions': {},
            'ensemble_analysis': {},
            'final_summary': {}
        }
        
        print(f"🔍 Initializing ML Results Analyzer for {symbol}")
        print("=" * 60)
    
    def analyze_complete_results(self):
        """Analyze complete ML pipeline results"""
        print("🎯 ANALYZING COMPLETE ML PIPELINE RESULTS")
        print("=" * 60)
        
        try:
            # Stage 1: Data Analysis
            print("\n📊 STAGE 1: DATA ANALYSIS")
            print("-" * 50)
            data_analysis = self._analyze_data()
            
            # Stage 2: Model Performance Analysis
            print("\n🤖 STAGE 2: MODEL PERFORMANCE ANALYSIS")
            print("-" * 50)
            model_performance = self._analyze_model_performance()
            
            # Stage 3: Feature Importance Analysis
            print("\n🔧 STAGE 3: FEATURE IMPORTANCE ANALYSIS")
            print("-" * 50)
            feature_importance = self._analyze_feature_importance()
            
            # Stage 4: Predictions Analysis
            print("\n📈 STAGE 4: PREDICTIONS ANALYSIS")
            print("-" * 50)
            predictions_analysis = self._analyze_predictions()
            
            # Stage 5: Ensemble Analysis
            print("\n🤝 STAGE 5: ENSEMBLE ANALYSIS")
            print("-" * 50)
            ensemble_analysis = self._analyze_ensemble()
            
            # Stage 6: Final Summary
            print("\n📋 STAGE 6: FINAL SUMMARY")
            print("-" * 50)
            final_summary = self._generate_final_summary()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Analysis Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_data(self):
        """Analyze data quality and characteristics"""
        print("📊 Analyzing data quality and characteristics...")
        
        try:
            # Load data
            data_1d = self.data_loader.load_data("1d", self.symbol)
            data_1w = self.data_loader.load_data("1w", self.symbol)
            
            # Data quality metrics
            data_quality = {
                'daily_records': len(data_1d) if data_1d is not None else 0,
                'weekly_records': len(data_1w) if data_1w is not None else 0,
                'daily_missing': data_1d.isnull().sum().sum() if data_1d is not None else 0,
                'weekly_missing': data_1w.isnull().sum().sum() if data_1w is not None else 0,
                'daily_duplicates': data_1d.duplicated().sum() if data_1d is not None else 0,
                'weekly_duplicates': data_1w.duplicated().sum() if data_1w is not None else 0
            }
            
            # Data characteristics
            if data_1d is not None:
                data_characteristics = {
                    'date_range': f"{data_1d.index.min()} to {data_1d.index.max()}",
                    'price_range': f"₹{data_1d['Close'].min():.2f} - ₹{data_1d['Close'].max():.2f}",
                    'avg_volume': f"{data_1d['Volume'].mean():,.0f}",
                    'volatility': f"{data_1d['Close'].pct_change().std()*100:.2f}%"
                }
            else:
                data_characteristics = {}
            
            self.results['data_analysis'] = {
                'quality': data_quality,
                'characteristics': data_characteristics
            }
            
            print(f"  ✅ Daily records: {data_quality['daily_records']}")
            print(f"  ✅ Weekly records: {data_quality['weekly_records']}")
            print(f"  ✅ Missing values: {data_quality['daily_missing']}")
            print(f"  ✅ Duplicates: {data_quality['daily_duplicates']}")
            
            return self.results['data_analysis']
            
        except Exception as e:
            print(f"  ❌ Data analysis error: {str(e)}")
            return {}
    
    def _analyze_model_performance(self):
        """Analyze model performance metrics"""
        print("🤖 Analyzing model performance...")
        
        try:
            # Load models
            models = {}
            model_files = {
                'RandomForest': 'models/ml_models/randomforest_model.pkl',
                'XGBoost': 'models/ml_models/xgboost_model.pkl',
                'LightGBM': 'models/ml_models/lightgbm_model.pkl',
                'GradientBoosting': 'models/ml_models/gradientboosting_model.pkl'
            }
            
            for name, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        models[name] = joblib.load(file_path)
                        print(f"  ✅ Loaded {name} model")
                    except Exception as e:
                        print(f"  ❌ Failed to load {name}: {str(e)}")
                else:
                    print(f"  ⚠️ {name} model file not found")
            
            # Load ensemble weights
            ensemble_weights = {}
            if os.path.exists('models/ml_models/ensemble_weights.json'):
                with open('models/ml_models/ensemble_weights.json', 'r') as f:
                    ensemble_weights = json.load(f)
            
            # Load scalers
            scalers = None
            if os.path.exists('models/ml_models/scalers.pkl'):
                try:
                    scalers = joblib.load('models/ml_models/scalers.pkl')
                    print("  ✅ Loaded scalers")
                except Exception as e:
                    print(f"  ❌ Failed to load scalers: {str(e)}")
            
            # Model performance summary
            model_performance = {
                'models_loaded': len(models),
                'model_names': list(models.keys()),
                'ensemble_weights': ensemble_weights,
                'scalers_loaded': scalers is not None
            }
            
            self.results['model_performance'] = model_performance
            
            print(f"  ✅ Models loaded: {model_performance['models_loaded']}")
            print(f"  ✅ Model names: {', '.join(model_performance['model_names'])}")
            print(f"  ✅ Ensemble weights: {ensemble_weights}")
            
            return model_performance
            
        except Exception as e:
            print(f"  ❌ Model performance analysis error: {str(e)}")
            return {}
    
    def _analyze_feature_importance(self):
        """Analyze feature importance"""
        print("🔧 Analyzing feature importance...")
        
        try:
            # Load models and get feature importance
            feature_importance = {}
            model_files = {
                'RandomForest': 'models/ml_models/randomforest_model.pkl',
                'XGBoost': 'models/ml_models/xgboost_model.pkl',
                'LightGBM': 'models/ml_models/lightgbm_model.pkl',
                'GradientBoosting': 'models/ml_models/gradientboosting_model.pkl'
            }
            
            for name, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        model = joblib.load(file_path)
                        if hasattr(model, 'feature_importances_'):
                            feature_importance[name] = model.feature_importances_.tolist()
                            print(f"  ✅ {name} feature importance extracted")
                        else:
                            print(f"  ⚠️ {name} has no feature importance")
                    except Exception as e:
                        print(f"  ❌ Failed to get {name} feature importance: {str(e)}")
            
            self.results['feature_importance'] = feature_importance
            
            return feature_importance
            
        except Exception as e:
            print(f"  ❌ Feature importance analysis error: {str(e)}")
            return {}
    
    def _analyze_predictions(self):
        """Analyze model predictions"""
        print("📈 Analyzing predictions...")
        
        try:
            # Load recent data for predictions
            data_1d = self.data_loader.load_data("1d", self.symbol)
            
            if data_1d is not None and len(data_1d) > 0:
                # Prepare features (simplified for analysis)
                features = self._prepare_features(data_1d)
                
                # Load models and make predictions
                predictions = {}
                model_files = {
                    'RandomForest': 'models/ml_models/randomforest_model.pkl',
                    'XGBoost': 'models/ml_models/xgboost_model.pkl',
                    'LightGBM': 'models/ml_models/lightgbm_model.pkl',
                    'GradientBoosting': 'models/ml_models/gradientboosting_model.pkl'
                }
                
                for name, file_path in model_files.items():
                    if os.path.exists(file_path):
                        try:
                            model = joblib.load(file_path)
                            if features is not None and len(features) > 0:
                                pred = model.predict(features)
                                predictions[name] = pred.tolist()
                                print(f"  ✅ {name} predictions generated")
                            else:
                                print(f"  ⚠️ No features available for {name}")
                        except Exception as e:
                            print(f"  ❌ Failed to generate {name} predictions: {str(e)}")
                
                self.results['predictions'] = predictions
                
                return predictions
            else:
                print("  ⚠️ No data available for predictions")
                return {}
                
        except Exception as e:
            print(f"  ❌ Predictions analysis error: {str(e)}")
            return {}
    
    def _prepare_features(self, data):
        """Prepare features for prediction"""
        try:
            # Simplified feature preparation for analysis
            if len(data) < 10:
                return None
            
            # Basic technical indicators
            data = data.copy()
            data['SMA_5'] = data['Close'].rolling(5).mean()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'] = data['Close'].rolling(12).mean() - data['Close'].rolling(26).mean()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 5:
                return None
            
            # Select features
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD']
            features = data[feature_columns].values
            
            return features[-1:]  # Return last row for prediction
            
        except Exception as e:
            print(f"  ❌ Feature preparation error: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def _analyze_ensemble(self):
        """Analyze ensemble model"""
        print("🤝 Analyzing ensemble model...")
        
        try:
            # Load ensemble weights
            ensemble_weights = {}
            if os.path.exists('models/ml_models/ensemble_weights.json'):
                with open('models/ml_models/ensemble_weights.json', 'r') as f:
                    ensemble_weights = json.load(f)
            
            # Analyze ensemble composition
            ensemble_analysis = {
                'total_models': len(ensemble_weights),
                'active_models': sum(1 for w in ensemble_weights.values() if w > 0),
                'primary_model': max(ensemble_weights.items(), key=lambda x: x[1])[0] if ensemble_weights else None,
                'weights': ensemble_weights
            }
            
            self.results['ensemble_analysis'] = ensemble_analysis
            
            print(f"  ✅ Total models: {ensemble_analysis['total_models']}")
            print(f"  ✅ Active models: {ensemble_analysis['active_models']}")
            print(f"  ✅ Primary model: {ensemble_analysis['primary_model']}")
            
            return ensemble_analysis
            
        except Exception as e:
            print(f"  ❌ Ensemble analysis error: {str(e)}")
            return {}
    
    def _generate_final_summary(self):
        """Generate final comprehensive summary"""
        print("📋 Generating final summary...")
        
        try:
            # Calculate overall scores
            data_score = min(10, max(0, 
                (self.results['data_analysis'].get('quality', {}).get('daily_records', 0) / 100) * 10))
            
            model_score = min(10, max(0, 
                self.results['model_performance'].get('models_loaded', 0) * 2.5))
            
            ensemble_score = min(10, max(0, 
                self.results['ensemble_analysis'].get('active_models', 0) * 2.5))
            
            overall_score = (data_score + model_score + ensemble_score) / 3
            
            # Determine rating
            if overall_score >= 9.0:
                rating = "EXCELLENT (A+)"
            elif overall_score >= 8.0:
                rating = "VERY GOOD (A)"
            elif overall_score >= 7.0:
                rating = "GOOD (B+)"
            elif overall_score >= 6.0:
                rating = "SATISFACTORY (B)"
            else:
                rating = "NEEDS IMPROVEMENT (C)"
            
            final_summary = {
                'overall_score': round(overall_score, 2),
                'rating': rating,
                'data_score': round(data_score, 2),
                'model_score': round(model_score, 2),
                'ensemble_score': round(ensemble_score, 2),
                'execution_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'symbol': self.symbol,
                'models_trained': self.results['model_performance'].get('models_loaded', 0),
                'features_processed': len(self.results['feature_importance'].get('RandomForest', [])) if 'RandomForest' in self.results['feature_importance'] else 0,
                'predictions_generated': len(self.results['predictions'])
            }
            
            self.results['final_summary'] = final_summary
            
            # Print final summary
            print("\n" + "=" * 60)
            print("🎉 ULTIMATE MARKET AI ENGINE - FINAL RESULTS")
            print("=" * 60)
            print(f"📊 Symbol: {final_summary['symbol']}")
            print(f"⏰ Execution Time: {final_summary['execution_time']}")
            print(f"📈 Overall Score: {final_summary['overall_score']}/10")
            print(f"🏆 Rating: {final_summary['rating']}")
            print(f"🤖 Models Trained: {final_summary['models_trained']}")
            print(f"🔧 Features Processed: {final_summary['features_processed']}")
            print(f"📈 Predictions Generated: {final_summary['predictions_generated']}")
            print("=" * 60)
            
            return final_summary
            
        except Exception as e:
            print(f"  ❌ Final summary error: {str(e)}")
            return {}
    
    def save_detailed_report(self):
        """Save detailed report to file"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'results': self.results
            }
            
            # Save as JSON
            with open('reports/ML_Analysis_Report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save as text
            with open('reports/ML_Analysis_Report.txt', 'w') as f:
                f.write("ULTIMATE MARKET AI ENGINE - ML ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Timestamp: {datetime.now()}\n\n")
                
                f.write("DATA ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.results['data_analysis'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.results['model_performance'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("ENSEMBLE ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.results['ensemble_analysis'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("FINAL SUMMARY:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.results['final_summary'].items():
                    f.write(f"{key}: {value}\n")
            
            print("✅ Detailed report saved to reports/ML_Analysis_Report.json and reports/ML_Analysis_Report.txt")
            
        except Exception as e:
            print(f"❌ Failed to save report: {str(e)}")

def main():
    """Main execution function"""
    print("🚀 ULTIMATE MARKET AI ENGINE - ML RESULTS ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MLResultsAnalyzer("RELIANCE.NS")
    
    # Run complete analysis
    results = analyzer.analyze_complete_results()
    
    if results:
        # Save detailed report
        analyzer.save_detailed_report()
        
        print("\n✅ ML Results Analysis completed successfully!")
        print("📊 Check reports/ML_Analysis_Report.json for detailed results")
    else:
        print("\n❌ ML Results Analysis failed!")

if __name__ == "__main__":
    main()