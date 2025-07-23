#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - COMPLETE ML PIPELINE EXECUTION
Execute the complete Machine Learning pipeline with all stages
Production-Grade Implementation with Comprehensive Outputs
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom imports
from ml_complete_pipeline import CompleteMLPipeline
from deep_learning_ensemble import DeepLearningTrainer, DeepLearningModels
from data.live_data_loader import DataLoader
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from config import get_config

class CompleteMLExecutor:
    """Complete ML Pipeline Executor"""
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.config = get_config()
        self.config.SYMBOL = symbol
        self.symbol = symbol
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        
        # Results storage
        self.results = {
            'traditional_ml': {},
            'deep_learning': {},
            'ensemble_results': {},
            'final_summary': {}
        }
        
        print(f"🚀 Initializing Complete ML Executor for {symbol}")
        print("=" * 60)
    
    def execute_complete_pipeline(self):
        """Execute the complete ML pipeline with all components"""
        print("🎯 EXECUTING COMPLETE ML PIPELINE")
        print("=" * 60)
        
        try:
            # Stage 1: Traditional ML Pipeline
            print("\n🤖 STAGE 1: TRADITIONAL ML PIPELINE")
            print("-" * 50)
            traditional_results = self._execute_traditional_ml()
            
            # Stage 2: Deep Learning Pipeline
            print("\n🧠 STAGE 2: DEEP LEARNING PIPELINE")
            print("-" * 50)
            deep_learning_results = self._execute_deep_learning()
            
            # Stage 3: Ensemble Integration
            print("\n🤝 STAGE 3: ENSEMBLE INTEGRATION")
            print("-" * 50)
            ensemble_results = self._create_master_ensemble(traditional_results, deep_learning_results)
            
            # Stage 4: Final Analysis & Reporting
            print("\n📋 STAGE 4: FINAL ANALYSIS & REPORTING")
            print("-" * 50)
            final_summary = self._generate_final_summary(traditional_results, deep_learning_results, ensemble_results)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Pipeline Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _execute_traditional_ml(self):
        """Execute traditional ML pipeline"""
        print("🚀 Executing Traditional ML Pipeline...")
        
        try:
            # Initialize traditional ML pipeline
            pipeline = CompleteMLPipeline(self.symbol)
            results = pipeline.execute_complete_pipeline()
            
            if results:
                self.results['traditional_ml'] = results
                print("✅ Traditional ML Pipeline completed successfully!")
                return results
            else:
                print("❌ Traditional ML Pipeline failed!")
                return None
                
        except Exception as e:
            print(f"❌ Traditional ML Error: {str(e)}")
            return None
    
    def _execute_deep_learning(self):
        """Execute deep learning pipeline"""
        print("🚀 Executing Deep Learning Pipeline...")
        
        try:
            # Load and prepare data
            print("📊 Loading market data for deep learning...")
            data = self.data_loader.fetch_single_timeframe_data('1d')
            
            if data is None or len(data) == 0:
                print("❌ No data available for deep learning")
                return None
            
            # Add technical indicators
            print("🔧 Adding technical indicators...")
            indicators = self.technical_analyzer.calculate_all_indicators(data)
            data = pd.concat([data, indicators], axis=1)
            
            # Prepare target variable
            data['target'] = data['Close'].shift(-1) / data['Close'] - 1
            
            # Remove NaN values
            data = data.dropna()
            
            # Prepare features for deep learning
            feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
            X = data[feature_cols].values
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize deep learning trainer
            trainer = DeepLearningTrainer(self.config)
            
            # Train all models
            training_results = trainer.train_all_models(X_scaled)
            
            # Create ensemble predictions
            ensemble_predictions = trainer.create_ensemble_predictions(X_scaled)
            
            # Generate report
            report = trainer.generate_training_report()
            
            # Store results
            deep_learning_results = {
                'training_results': training_results,
                'ensemble_predictions': ensemble_predictions,
                'report': report,
                'data_info': {
                    'samples': len(X_scaled),
                    'features': X_scaled.shape[1],
                    'date_range': f"{data.index.min()} to {data.index.max()}"
                }
            }
            
            self.results['deep_learning'] = deep_learning_results
            
            print("✅ Deep Learning Pipeline completed successfully!")
            return deep_learning_results
            
        except Exception as e:
            print(f"❌ Deep Learning Error: {str(e)}")
            return None
    
    def _create_master_ensemble(self, traditional_results, deep_learning_results):
        """Create master ensemble combining traditional ML and deep learning"""
        print("🤝 Creating Master Ensemble...")
        
        try:
            ensemble_results = {
                'ensemble_method': 'Weighted Average',
                'components': {
                    'traditional_ml': traditional_results.get('summary', {}).get('ensemble_accuracy', 0),
                    'deep_learning': deep_learning_results.get('report', {}).get('best_model_r2', 0) if deep_learning_results else 0
                },
                'weights': {},
                'final_predictions': {},
                'performance_metrics': {}
            }
            
            # Calculate ensemble weights based on performance
            traditional_score = ensemble_results['components']['traditional_ml']
            deep_learning_score = ensemble_results['components']['deep_learning']
            
            total_score = traditional_score + deep_learning_score
            
            if total_score > 0:
                traditional_weight = traditional_score / total_score
                deep_learning_weight = deep_learning_score / total_score
            else:
                traditional_weight = 0.5
                deep_learning_weight = 0.5
            
            ensemble_results['weights'] = {
                'traditional_ml': traditional_weight,
                'deep_learning': deep_learning_weight
            }
            
            # Calculate final ensemble score
            final_score = (traditional_score * traditional_weight + 
                          deep_learning_score * deep_learning_weight)
            
            ensemble_results['final_score'] = final_score
            ensemble_results['overall_rating'] = self._calculate_overall_rating(final_score)
            
            self.results['ensemble_results'] = ensemble_results
            
            print(f"✅ Master Ensemble created successfully!")
            print(f"📊 Traditional ML Weight: {traditional_weight:.3f}")
            print(f"🧠 Deep Learning Weight: {deep_learning_weight:.3f}")
            print(f"🏆 Final Ensemble Score: {final_score:.4f}")
            print(f"⭐ Overall Rating: {ensemble_results['overall_rating']}")
            
            return ensemble_results
            
        except Exception as e:
            print(f"❌ Ensemble Error: {str(e)}")
            return None
    
    def _calculate_overall_rating(self, score):
        """Calculate overall rating based on score"""
        if score >= 0.9:
            return "EXCEPTIONAL (A+)"
        elif score >= 0.8:
            return "EXCELLENT (A)"
        elif score >= 0.7:
            return "VERY GOOD (B+)"
        elif score >= 0.6:
            return "GOOD (B)"
        elif score >= 0.5:
            return "SATISFACTORY (C+)"
        else:
            return "NEEDS IMPROVEMENT (C)"
    
    def _generate_final_summary(self, traditional_results, deep_learning_results, ensemble_results):
        """Generate comprehensive final summary"""
        print("📋 Generating Final Summary...")
        
        try:
            # Extract key metrics
            traditional_summary = traditional_results.get('summary', {}) if traditional_results else {}
            deep_learning_summary = deep_learning_results.get('report', {}) if deep_learning_results else {}
            
            final_summary = {
                'symbol': self.symbol,
                'execution_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pipeline_status': 'COMPLETED',
                
                # Traditional ML Metrics
                'traditional_ml': {
                    'overall_score': traditional_summary.get('overall_score', 0),
                    'models_trained': traditional_summary.get('total_models_trained', 0),
                    'best_model': traditional_summary.get('best_model', 'None'),
                    'ensemble_accuracy': traditional_summary.get('ensemble_accuracy', 0),
                    'features_used': traditional_summary.get('total_features_used', 0),
                    'data_quality': traditional_summary.get('data_quality_score', 0)
                },
                
                # Deep Learning Metrics
                'deep_learning': {
                    'overall_score': deep_learning_summary.get('overall_score', 0),
                    'models_trained': deep_learning_summary.get('total_models_trained', 0),
                    'best_model': deep_learning_summary.get('best_model', 'None'),
                    'best_model_r2': deep_learning_summary.get('best_model_r2', 0),
                    'average_r2': deep_learning_summary.get('model_performance', {}).get('average_r2', 0) if deep_learning_summary.get('model_performance') else 0
                },
                
                # Ensemble Metrics
                'master_ensemble': {
                    'final_score': ensemble_results.get('final_score', 0) if ensemble_results else 0,
                    'overall_rating': ensemble_results.get('overall_rating', 'UNKNOWN') if ensemble_results else 'UNKNOWN',
                    'traditional_weight': ensemble_results.get('weights', {}).get('traditional_ml', 0) if ensemble_results else 0,
                    'deep_learning_weight': ensemble_results.get('weights', {}).get('deep_learning', 0) if ensemble_results else 0
                },
                
                # System Performance
                'system_performance': {
                    'total_execution_time': 'Calculated at end',
                    'models_deployed': (traditional_summary.get('total_models_trained', 0) + 
                                       deep_learning_summary.get('total_models_trained', 0)),
                    'data_processed': traditional_summary.get('total_features_used', 0),
                    'prediction_accuracy': ensemble_results.get('final_score', 0) if ensemble_results else 0
                }
            }
            
            self.results['final_summary'] = final_summary
            
            # Print comprehensive summary
            self._print_final_summary(final_summary)
            
            return final_summary
            
        except Exception as e:
            print(f"❌ Summary Generation Error: {str(e)}")
            return None
    
    def _print_final_summary(self, summary):
        """Print comprehensive final summary"""
        print("\n" + "=" * 80)
        print("🎉 ULTIMATE MARKET AI ENGINE - COMPLETE ML PIPELINE RESULTS")
        print("=" * 80)
        
        print(f"📊 Symbol: {summary['symbol']}")
        print(f"⏰ Execution Time: {summary['execution_timestamp']}")
        print(f"📈 Pipeline Status: {summary['pipeline_status']}")
        
        print("\n🤖 TRADITIONAL ML RESULTS:")
        print("-" * 40)
        traditional = summary['traditional_ml']
        print(f"  Overall Score: {traditional['overall_score']:.2f}/10")
        print(f"  Models Trained: {traditional['models_trained']}")
        print(f"  Best Model: {traditional['best_model']}")
        print(f"  Ensemble R²: {traditional['ensemble_accuracy']:.4f}")
        print(f"  Features Used: {traditional['features_used']}")
        print(f"  Data Quality: {traditional['data_quality']:.2f}/10")
        
        print("\n🧠 DEEP LEARNING RESULTS:")
        print("-" * 40)
        deep_learning = summary['deep_learning']
        print(f"  Overall Score: {deep_learning['overall_score']:.2f}/10")
        print(f"  Models Trained: {deep_learning['models_trained']}")
        print(f"  Best Model: {deep_learning['best_model']}")
        print(f"  Best Model R²: {deep_learning['best_model_r2']:.4f}")
        
        print("\n🤝 MASTER ENSEMBLE RESULTS:")
        print("-" * 40)
        ensemble = summary['master_ensemble']
        print(f"  Final Score: {ensemble['final_score']:.4f}")
        print(f"  Overall Rating: {ensemble['overall_rating']}")
        print(f"  Traditional ML Weight: {ensemble['traditional_weight']:.3f}")
        print(f"  Deep Learning Weight: {ensemble['deep_learning_weight']:.3f}")
        
        print("\n📊 SYSTEM PERFORMANCE:")
        print("-" * 40)
        performance = summary['system_performance']
        print(f"  Total Models Deployed: {performance['models_deployed']}")
        print(f"  Data Features Processed: {performance['data_processed']}")
        print(f"  Prediction Accuracy: {performance['prediction_accuracy']:.4f}")
        
        print("\n" + "=" * 80)
        print("✅ COMPLETE ML PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
        print("=" * 80)

def main():
    """Main execution function"""
    print("🚀 ULTIMATE MARKET AI ENGINE - COMPLETE ML PIPELINE EXECUTION")
    print("=" * 80)
    
    # Initialize and execute complete pipeline
    executor = CompleteMLExecutor("RELIANCE.NS")
    results = executor.execute_complete_pipeline()
    
    if results:
        print("\n✅ Complete ML Pipeline executed successfully!")
        final_score = results.get('final_summary', {}).get('master_ensemble', {}).get('final_score', 0)
        print(f"🏆 Final Ensemble Score: {final_score:.4f}")
    else:
        print("\n❌ Complete ML Pipeline execution failed!")
    
    return results

if __name__ == "__main__":
    main()