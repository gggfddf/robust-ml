#!/usr/bin/env python3
"""
ULTIMATE MARKET AI ENGINE - COMPLETE INTEGRATION SYSTEM
Complete Market Analysis with Multi-timeframe Integration, Pattern Recognition, 
Breakout Detection, ML Predictions, and Actionable Trading Insights
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateMarketAISystem:
    """Ultimate Market AI System - Complete Integration"""
    
    def __init__(self, symbol="RELIANCE"):
        self.symbol = symbol
        self.results = {}
        self.system_rating = {}
        
        logger.info(f"Ultimate Market AI System initialized for {symbol}")
    
    def run_complete_analysis(self) -> Dict:
        """Run the complete market analysis system"""
        logger.info("🚀 Starting Ultimate Market AI System")
        
        analysis_results = {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'data_analysis': {},
            'pattern_analysis': {},
            'breakout_analysis': {},
            'ml_analysis': {},
            'trading_insights': {},
            'system_rating': {},
            'summary': {}
        }
        
        try:
            # Phase 1: Data Loading and Analysis
            logger.info("📊 Phase 1: Data Loading and Analysis...")
            data_analysis = self._run_data_analysis()
            analysis_results['data_analysis'] = data_analysis
            
            if not data_analysis['data_available']:
                logger.error("❌ No data available for analysis")
                return analysis_results
            
            # Phase 2: Pattern Recognition
            logger.info("🔍 Phase 2: Pattern Recognition...")
            pattern_analysis = self._run_pattern_analysis(data_analysis['multi_data'])
            analysis_results['pattern_analysis'] = pattern_analysis
            
            # Phase 3: Breakout Detection
            logger.info("💥 Phase 3: Breakout Detection...")
            breakout_analysis = self._run_breakout_analysis(data_analysis['multi_data'])
            analysis_results['breakout_analysis'] = breakout_analysis
            
            # Phase 4: ML Analysis and Predictions
            logger.info("🤖 Phase 4: ML Analysis and Predictions...")
            ml_analysis = self._run_ml_analysis(data_analysis['multi_data'])
            analysis_results['ml_analysis'] = ml_analysis
            
            # Phase 5: Trading Insights
            logger.info("💹 Phase 5: Trading Insights...")
            trading_insights = self._generate_trading_insights(analysis_results)
            analysis_results['trading_insights'] = trading_insights
            
            # Phase 6: System Rating
            logger.info("⭐ Phase 6: System Rating...")
            system_rating = self._calculate_system_rating(analysis_results)
            analysis_results['system_rating'] = system_rating
            
            # Phase 7: Final Summary
            logger.info("📋 Phase 7: Final Summary...")
            summary = self._create_final_summary(analysis_results)
            analysis_results['summary'] = summary
            
            logger.info("✅ Ultimate Market AI System completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error in ultimate market AI system: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _run_data_analysis(self) -> Dict:
        """Run comprehensive data analysis"""
        try:
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
                'timeframes': list(multi_data.keys()),
                'total_records': data_summary['overall']['total_records'],
                'quality_score': quality_report['overall_score']
            }
            
            logger.info(f"✅ Data Analysis: {data_analysis['total_records']} records, Quality: {data_analysis['quality_score']}/10")
            
            return data_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in data analysis: {str(e)}")
            return {'data_available': False, 'error': str(e)}
    
    def _run_pattern_analysis(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run comprehensive pattern analysis"""
        try:
            import sys
            sys.path.append('/workspace')
            from pattern_analysis.comprehensive_pattern_recognition import ComprehensivePatternRecognition
            
            # Initialize pattern recognition
            pattern_recognition = ComprehensivePatternRecognition()
            
            # Analyze patterns
            patterns = pattern_recognition.analyze_multi_timeframe_patterns(multi_data)
            
            pattern_analysis = {
                'traditional_patterns': patterns['summary']['total_traditional'],
                'ml_patterns': patterns['summary']['total_ml'],
                'confluence_signals': patterns['summary']['confluence_signals'],
                'pattern_details': patterns,
                'pattern_score': self._calculate_pattern_score(patterns)
            }
            
            logger.info(f"✅ Pattern Analysis: {pattern_analysis['traditional_patterns']} traditional, {pattern_analysis['ml_patterns']} ML patterns")
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in pattern analysis: {str(e)}")
            return {'traditional_patterns': 0, 'ml_patterns': 0, 'pattern_score': 0}
    
    def _run_breakout_analysis(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run comprehensive breakout analysis"""
        try:
            import sys
            sys.path.append('/workspace')
            from analysis.breakout_detection import BreakoutDetection
            
            # Initialize breakout detection
            breakout_detection = BreakoutDetection()
            
            # Analyze breakouts
            breakouts = breakout_detection.analyze_multi_timeframe_breakouts(multi_data)
            
            breakout_analysis = {
                'real_breakouts': breakouts['summary']['real_breakouts'],
                'fake_breakouts': breakouts['summary']['fake_breakouts'],
                'confluence_signals': breakouts['summary']['confluence_signals'],
                'breakout_details': breakouts,
                'breakout_score': self._calculate_breakout_score(breakouts)
            }
            
            logger.info(f"✅ Breakout Analysis: {breakout_analysis['real_breakouts']} real, {breakout_analysis['fake_breakouts']} fake breakouts")
            
            return breakout_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in breakout analysis: {str(e)}")
            return {'real_breakouts': 0, 'fake_breakouts': 0, 'breakout_score': 0}
    
    def _run_ml_analysis(self, multi_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run comprehensive ML analysis"""
        try:
            import sys
            sys.path.append('/workspace')
            from ml_enhanced_pipeline import EnhancedMLPipeline
            
            # Initialize ML pipeline
            ml_pipeline = EnhancedMLPipeline(self.symbol)
            
            # Run ML analysis
            ml_results = ml_pipeline.run_complete_pipeline()
            
            ml_analysis = {
                'models_trained': len(ml_results.get('model_training', {})),
                'predictions_generated': len(ml_results.get('predictions', {})),
                'trading_signals': ml_results.get('trading_signals', {}).get('summary', {}),
                'ml_details': ml_results,
                'ml_score': self._calculate_ml_score(ml_results)
            }
            
            logger.info(f"✅ ML Analysis: {ml_analysis['models_trained']} models, {ml_analysis['predictions_generated']} predictions")
            
            return ml_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in ML analysis: {str(e)}")
            return {'models_trained': 0, 'predictions_generated': 0, 'ml_score': 0}
    
    def _generate_trading_insights(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive trading insights"""
        try:
            trading_insights = {
                'overall_sentiment': 'NEUTRAL',
                'confidence_level': 0.0,
                'risk_level': 'MEDIUM',
                'recommended_actions': [],
                'key_signals': [],
                'risk_warnings': [],
                'opportunities': []
            }
            
            # Analyze sentiment from different components
            sentiment_scores = []
            
            # Pattern sentiment
            if 'pattern_analysis' in analysis_results:
                pattern_score = analysis_results['pattern_analysis']['pattern_score']
                sentiment_scores.append(pattern_score)
                
                if pattern_score > 0.7:
                    trading_insights['key_signals'].append("Strong pattern recognition detected")
                elif pattern_score > 0.5:
                    trading_insights['key_signals'].append("Moderate pattern activity")
            
            # Breakout sentiment
            if 'breakout_analysis' in analysis_results:
                breakout_score = analysis_results['breakout_analysis']['breakout_score']
                sentiment_scores.append(breakout_score)
                
                real_breakouts = analysis_results['breakout_analysis']['real_breakouts']
                fake_breakouts = analysis_results['breakout_analysis']['fake_breakouts']
                
                if real_breakouts > fake_breakouts:
                    trading_insights['opportunities'].append(f"Strong breakout signals: {real_breakouts} real vs {fake_breakouts} fake")
                elif fake_breakouts > real_breakouts:
                    trading_insights['risk_warnings'].append(f"Caution: {fake_breakouts} fake breakouts detected")
            
            # ML sentiment
            if 'ml_analysis' in analysis_results:
                ml_score = analysis_results['ml_analysis']['ml_score']
                sentiment_scores.append(ml_score)
                
                trading_signals = analysis_results['ml_analysis']['trading_signals']
                buy_signals = trading_signals.get('buy_signals', 0)
                sell_signals = trading_signals.get('sell_signals', 0)
                
                if buy_signals > sell_signals:
                    trading_insights['recommended_actions'].append(f"ML models suggest BUY signals ({buy_signals} vs {sell_signals} sell)")
                elif sell_signals > buy_signals:
                    trading_insights['recommended_actions'].append(f"ML models suggest SELL signals ({sell_signals} vs {buy_signals} buy)")
            
            # Calculate overall sentiment
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                trading_insights['confidence_level'] = avg_sentiment
                
                if avg_sentiment > 0.7:
                    trading_insights['overall_sentiment'] = 'BULLISH'
                    trading_insights['risk_level'] = 'LOW'
                elif avg_sentiment > 0.5:
                    trading_insights['overall_sentiment'] = 'SLIGHTLY_BULLISH'
                    trading_insights['risk_level'] = 'MEDIUM'
                elif avg_sentiment > 0.3:
                    trading_insights['overall_sentiment'] = 'NEUTRAL'
                    trading_insights['risk_level'] = 'MEDIUM'
                elif avg_sentiment > 0.1:
                    trading_insights['overall_sentiment'] = 'SLIGHTLY_BEARISH'
                    trading_insights['risk_level'] = 'HIGH'
                else:
                    trading_insights['overall_sentiment'] = 'BEARISH'
                    trading_insights['risk_level'] = 'HIGH'
            
            # Add general recommendations
            if trading_insights['confidence_level'] > 0.7:
                trading_insights['recommended_actions'].append("High confidence signals - consider taking action")
            elif trading_insights['confidence_level'] > 0.5:
                trading_insights['recommended_actions'].append("Medium confidence - monitor closely")
            else:
                trading_insights['recommended_actions'].append("Low confidence - wait for better signals")
            
            # Add risk management
            trading_insights['recommended_actions'].append("Always use proper risk management and stop-loss orders")
            trading_insights['recommended_actions'].append("Consider market conditions and external factors")
            
            return trading_insights
            
        except Exception as e:
            logger.error(f"❌ Error generating trading insights: {str(e)}")
            return {'overall_sentiment': 'NEUTRAL', 'confidence_level': 0.0, 'recommended_actions': ['Analysis failed']}
    
    def _calculate_system_rating(self, analysis_results: Dict) -> Dict:
        """Calculate comprehensive system rating"""
        try:
            system_rating = {
                'overall_score': 0.0,
                'component_scores': {},
                'grade': 'F',
                'performance_level': 'POOR',
                'recommendations': []
            }
            
            # Calculate component scores
            component_scores = {}
            
            # Data Quality Score (0-10)
            if 'data_analysis' in analysis_results:
                data_score = analysis_results['data_analysis'].get('quality_score', 0)
                component_scores['data_quality'] = data_score
            else:
                component_scores['data_quality'] = 0
            
            # Pattern Recognition Score (0-10)
            if 'pattern_analysis' in analysis_results:
                pattern_score = analysis_results['pattern_analysis'].get('pattern_score', 0) * 10
                component_scores['pattern_recognition'] = pattern_score
            else:
                component_scores['pattern_recognition'] = 0
            
            # Breakout Detection Score (0-10)
            if 'breakout_analysis' in analysis_results:
                breakout_score = analysis_results['breakout_analysis'].get('breakout_score', 0) * 10
                component_scores['breakout_detection'] = breakout_score
            else:
                component_scores['breakout_detection'] = 0
            
            # ML Analysis Score (0-10)
            if 'ml_analysis' in analysis_results:
                ml_score = analysis_results['ml_analysis'].get('ml_score', 0) * 10
                component_scores['ml_analysis'] = ml_score
            else:
                component_scores['ml_analysis'] = 0
            
            # Trading Insights Score (0-10)
            if 'trading_insights' in analysis_results:
                confidence = analysis_results['trading_insights'].get('confidence_level', 0)
                insights_score = confidence * 10
                component_scores['trading_insights'] = insights_score
            else:
                component_scores['trading_insights'] = 0
            
            # Calculate overall score (weighted average)
            weights = {
                'data_quality': 0.2,
                'pattern_recognition': 0.2,
                'breakout_detection': 0.2,
                'ml_analysis': 0.25,
                'trading_insights': 0.15
            }
            
            overall_score = 0
            for component, score in component_scores.items():
                weight = weights.get(component, 0.2)
                overall_score += score * weight
            
            # Determine grade and performance level
            if overall_score >= 9.0:
                grade = 'A+'
                performance_level = 'EXCELLENT'
            elif overall_score >= 8.0:
                grade = 'A'
                performance_level = 'EXCELLENT'
            elif overall_score >= 7.0:
                grade = 'B+'
                performance_level = 'GOOD'
            elif overall_score >= 6.0:
                grade = 'B'
                performance_level = 'GOOD'
            elif overall_score >= 5.0:
                grade = 'C+'
                performance_level = 'AVERAGE'
            elif overall_score >= 4.0:
                grade = 'C'
                performance_level = 'AVERAGE'
            elif overall_score >= 3.0:
                grade = 'D+'
                performance_level = 'POOR'
            elif overall_score >= 2.0:
                grade = 'D'
                performance_level = 'POOR'
            else:
                grade = 'F'
                performance_level = 'FAILED'
            
            # Generate recommendations
            recommendations = []
            
            if component_scores['data_quality'] < 7.0:
                recommendations.append("Improve data quality and sources")
            
            if component_scores['pattern_recognition'] < 7.0:
                recommendations.append("Enhance pattern recognition algorithms")
            
            if component_scores['breakout_detection'] < 7.0:
                recommendations.append("Refine breakout detection criteria")
            
            if component_scores['ml_analysis'] < 7.0:
                recommendations.append("Optimize ML models and feature engineering")
            
            if component_scores['trading_insights'] < 7.0:
                recommendations.append("Improve trading signal generation")
            
            if overall_score < 7.0:
                recommendations.append("Overall system needs improvement - consider comprehensive optimization")
            else:
                recommendations.append("System performing well - continue monitoring and optimization")
            
            system_rating.update({
                'overall_score': round(overall_score, 2),
                'component_scores': {k: round(v, 2) for k, v in component_scores.items()},
                'grade': grade,
                'performance_level': performance_level,
                'recommendations': recommendations
            })
            
            logger.info(f"✅ System Rating: {grade} ({overall_score:.2f}/10) - {performance_level}")
            
            return system_rating
            
        except Exception as e:
            logger.error(f"❌ Error calculating system rating: {str(e)}")
            return {'overall_score': 0.0, 'grade': 'F', 'performance_level': 'FAILED', 'recommendations': ['Rating calculation failed']}
    
    def _calculate_pattern_score(self, patterns: Dict) -> float:
        """Calculate pattern recognition score"""
        try:
            total_patterns = patterns['summary']['total_traditional'] + patterns['summary']['total_ml']
            confluence_signals = patterns['summary']['confluence_signals']
            
            if total_patterns == 0:
                return 0.0
            
            # Score based on pattern density and confluence
            pattern_density = min(1.0, total_patterns / 100)  # Normalize to 0-1
            confluence_ratio = min(1.0, confluence_signals / max(1, total_patterns))
            
            score = (pattern_density * 0.6 + confluence_ratio * 0.4)
            return round(score, 3)
            
        except Exception as e:
            logger.error(f"❌ Error calculating pattern score: {str(e)}")
            return 0.0
    
    def _calculate_breakout_score(self, breakouts: Dict) -> float:
        """Calculate breakout detection score"""
        try:
            total_breakouts = breakouts['summary']['total_breakouts']
            real_breakouts = breakouts['summary']['real_breakouts']
            confluence_signals = breakouts['summary']['confluence_signals']
            
            if total_breakouts == 0:
                return 0.0
            
            # Score based on real vs fake ratio and confluence
            real_ratio = real_breakouts / total_breakouts
            confluence_ratio = min(1.0, confluence_signals / max(1, total_breakouts))
            
            score = (real_ratio * 0.7 + confluence_ratio * 0.3)
            return round(score, 3)
            
        except Exception as e:
            logger.error(f"❌ Error calculating breakout score: {str(e)}")
            return 0.0
    
    def _calculate_ml_score(self, ml_results: Dict) -> float:
        """Calculate ML analysis score"""
        try:
            models_trained = len(ml_results.get('model_training', {}))
            predictions_generated = len(ml_results.get('predictions', {}))
            
            # Score based on model training and predictions
            model_score = min(1.0, models_trained / 5)  # Normalize to 0-1
            prediction_score = min(1.0, predictions_generated / 3)
            
            # Consider performance metrics if available
            performance_score = 0.5  # Default
            if 'performance_metrics' in ml_results:
                avg_confidence = ml_results['performance_metrics'].get('overall_metrics', {}).get('average_confidence', 0)
                performance_score = avg_confidence
            
            score = (model_score * 0.4 + prediction_score * 0.3 + performance_score * 0.3)
            return round(score, 3)
            
        except Exception as e:
            logger.error(f"❌ Error calculating ML score: {str(e)}")
            return 0.0
    
    def _create_final_summary(self, analysis_results: Dict) -> Dict:
        """Create comprehensive final summary"""
        try:
            summary = {
                'analysis_status': 'COMPLETED' if 'error' not in analysis_results else 'FAILED',
                'execution_time': datetime.now(),
                'key_achievements': [],
                'critical_findings': [],
                'action_items': [],
                'system_performance': {},
                'market_outlook': {}
            }
            
            # Key achievements
            if 'data_analysis' in analysis_results:
                data_records = analysis_results['data_analysis'].get('total_records', 0)
                summary['key_achievements'].append(f"Successfully analyzed {data_records} data records")
            
            if 'pattern_analysis' in analysis_results:
                patterns = analysis_results['pattern_analysis']['traditional_patterns']
                summary['key_achievements'].append(f"Detected {patterns} candlestick patterns")
            
            if 'breakout_analysis' in analysis_results:
                real_breakouts = analysis_results['breakout_analysis']['real_breakouts']
                summary['key_achievements'].append(f"Identified {real_breakouts} real breakout signals")
            
            if 'ml_analysis' in analysis_results:
                models = analysis_results['ml_analysis']['models_trained']
                summary['key_achievements'].append(f"Trained {models} ML models")
            
            # Critical findings
            if 'trading_insights' in analysis_results:
                sentiment = analysis_results['trading_insights']['overall_sentiment']
                confidence = analysis_results['trading_insights']['confidence_level']
                summary['critical_findings'].append(f"Market sentiment: {sentiment} (confidence: {confidence:.1%})")
            
            if 'system_rating' in analysis_results:
                rating = analysis_results['system_rating']
                summary['critical_findings'].append(f"System performance: {rating['grade']} ({rating['overall_score']}/10)")
            
            # Action items
            if 'trading_insights' in analysis_results:
                insights = analysis_results['trading_insights']
                summary['action_items'].extend(insights.get('recommended_actions', []))
            
            if 'system_rating' in analysis_results:
                rating = analysis_results['system_rating']
                summary['action_items'].extend(rating.get('recommendations', []))
            
            # System performance
            if 'system_rating' in analysis_results:
                rating = analysis_results['system_rating']
                summary['system_performance'] = {
                    'overall_score': rating['overall_score'],
                    'grade': rating['grade'],
                    'performance_level': rating['performance_level'],
                    'component_scores': rating['component_scores']
                }
            
            # Market outlook
            if 'trading_insights' in analysis_results:
                insights = analysis_results['trading_insights']
                summary['market_outlook'] = {
                    'sentiment': insights['overall_sentiment'],
                    'confidence': insights['confidence_level'],
                    'risk_level': insights['risk_level'],
                    'key_signals': insights['key_signals'],
                    'opportunities': insights['opportunities'],
                    'risk_warnings': insights['risk_warnings']
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error creating final summary: {str(e)}")
            return {'analysis_status': 'FAILED', 'error': str(e)}

def main():
    """Test the ultimate market AI system"""
    print("🚀 Testing Ultimate Market AI System")
    print("=" * 70)
    
    # Initialize system
    system = UltimateMarketAISystem("RELIANCE")
    
    # Run complete analysis
    results = system.run_complete_analysis()
    
    # Print comprehensive results
    print(f"\n📊 ULTIMATE MARKET AI SYSTEM RESULTS")
    print(f"Symbol: {results['symbol']}")
    print(f"Status: {results.get('summary', {}).get('analysis_status', 'UNKNOWN')}")
    print(f"Timestamp: {results['timestamp']}")
    
    # Data Analysis
    if 'data_analysis' in results:
        data = results['data_analysis']
        print(f"\n📈 DATA ANALYSIS:")
        print(f"  Records: {data.get('total_records', 0)}")
        print(f"  Timeframes: {', '.join(data.get('timeframes', []))}")
        print(f"  Quality Score: {data.get('quality_score', 0)}/10")
    
    # Pattern Analysis
    if 'pattern_analysis' in results:
        patterns = results['pattern_analysis']
        print(f"\n🔍 PATTERN ANALYSIS:")
        print(f"  Traditional Patterns: {patterns.get('traditional_patterns', 0)}")
        print(f"  ML Patterns: {patterns.get('ml_patterns', 0)}")
        print(f"  Confluence Signals: {patterns.get('confluence_signals', 0)}")
        print(f"  Pattern Score: {patterns.get('pattern_score', 0):.3f}")
    
    # Breakout Analysis
    if 'breakout_analysis' in results:
        breakouts = results['breakout_analysis']
        print(f"\n💥 BREAKOUT ANALYSIS:")
        print(f"  Real Breakouts: {breakouts.get('real_breakouts', 0)}")
        print(f"  Fake Breakouts: {breakouts.get('fake_breakouts', 0)}")
        print(f"  Confluence Signals: {breakouts.get('confluence_signals', 0)}")
        print(f"  Breakout Score: {breakouts.get('breakout_score', 0):.3f}")
    
    # ML Analysis
    if 'ml_analysis' in results:
        ml = results['ml_analysis']
        print(f"\n🤖 ML ANALYSIS:")
        print(f"  Models Trained: {ml.get('models_trained', 0)}")
        print(f"  Predictions Generated: {ml.get('predictions_generated', 0)}")
        print(f"  ML Score: {ml.get('ml_score', 0):.3f}")
        
        trading_signals = ml.get('trading_signals', {})
        print(f"  Trading Signals: {trading_signals.get('buy_signals', 0)} buy, {trading_signals.get('sell_signals', 0)} sell, {trading_signals.get('hold_signals', 0)} hold")
    
    # Trading Insights
    if 'trading_insights' in results:
        insights = results['trading_insights']
        print(f"\n💹 TRADING INSIGHTS:")
        print(f"  Overall Sentiment: {insights.get('overall_sentiment', 'NEUTRAL')}")
        print(f"  Confidence Level: {insights.get('confidence_level', 0):.1%}")
        print(f"  Risk Level: {insights.get('risk_level', 'MEDIUM')}")
        
        print(f"  Key Signals:")
        for signal in insights.get('key_signals', []):
            print(f"    • {signal}")
        
        print(f"  Opportunities:")
        for opp in insights.get('opportunities', []):
            print(f"    • {opp}")
        
        print(f"  Risk Warnings:")
        for warning in insights.get('risk_warnings', []):
            print(f"    • {warning}")
    
    # System Rating
    if 'system_rating' in results:
        rating = results['system_rating']
        print(f"\n⭐ SYSTEM RATING:")
        print(f"  Overall Score: {rating.get('overall_score', 0)}/10")
        print(f"  Grade: {rating.get('grade', 'F')}")
        print(f"  Performance Level: {rating.get('performance_level', 'FAILED')}")
        
        print(f"  Component Scores:")
        for component, score in rating.get('component_scores', {}).items():
            print(f"    • {component}: {score}/10")
        
        print(f"  Recommendations:")
        for rec in rating.get('recommendations', []):
            print(f"    • {rec}")
    
    # Final Summary
    if 'summary' in results:
        summary = results['summary']
        print(f"\n📋 FINAL SUMMARY:")
        
        print(f"  Key Achievements:")
        for achievement in summary.get('key_achievements', []):
            print(f"    • {achievement}")
        
        print(f"  Critical Findings:")
        for finding in summary.get('critical_findings', []):
            print(f"    • {finding}")
        
        print(f"  Action Items:")
        for action in summary.get('action_items', []):
            print(f"    • {action}")
    
    if 'error' in results:
        print(f"\n❌ Error: {results['error']}")
    
    print(f"\n🎯 ULTIMATE MARKET AI SYSTEM ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()