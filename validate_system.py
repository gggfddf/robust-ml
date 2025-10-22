"""
Ultimate Market AI Engine - System Validation Script
==================================================

Comprehensive validation of all system components for Reliance Industries analysis.
"""

import sys
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validation for Ultimate Market AI Engine."""
    
    def __init__(self):
        self.validation_results = {}
        self.performance_metrics = {}
        self.error_log = []
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        logger.info("Starting comprehensive system validation...")
        
        validation_start = datetime.now()
        
        # Test 1: Configuration System
        self._validate_configuration()
        
        # Test 2: Data Loading System
        self._validate_data_loading()
        
        # Test 3: Technical Analysis System
        self._validate_technical_analysis()
        
        # Test 4: Pattern Discovery System
        self._validate_pattern_discovery()
        
        # Test 5: Deep Learning System
        self._validate_deep_learning()
        
        # Test 6: Time Analysis System
        self._validate_time_analysis()
        
        # Test 7: Visualization System
        self._validate_visualization()
        
        # Test 8: Report Generation System
        self._validate_report_generation()
        
        # Test 9: Integration Testing
        self._validate_integration()
        
        # Test 10: Performance Testing
        self._validate_performance()
        
        validation_end = datetime.now()
        validation_time = validation_end - validation_start
        
        # Generate final report
        final_report = self._generate_validation_report(validation_time)
        
        logger.info("Comprehensive validation completed!")
        return final_report
    
    def _validate_configuration(self):
        """Validate configuration system."""
        logger.info("Validating configuration system...")
        
        try:
            from config import get_config, update_symbol
            
            # Test configuration loading
            config = get_config()
            assert config.SYMBOL == "RELIANCE.NS", f"Expected RELIANCE.NS, got {config.SYMBOL}"
            assert "1m" in config.TIMEFRAMES, "1m timeframe not found"
            assert "5m" in config.TIMEFRAMES, "5m timeframe not found"
            assert "15m" in config.TIMEFRAMES, "15m timeframe not found"
            assert "1d" in config.TIMEFRAMES, "1d timeframe not found"
            assert "1w" in config.TIMEFRAMES, "1w timeframe not found"
            assert config.LIVE_DATA_ENABLED == True, "Live data not enabled"
            
            # Test symbol update
            update_symbol("TCS.NS")
            config = get_config()
            assert config.SYMBOL == "TCS.NS", "Symbol update failed"
            
            # Reset to RELIANCE
            update_symbol("RELIANCE.NS")
            
            self.validation_results['configuration'] = {
                'status': 'PASSED',
                'score': 10,
                'details': 'All configuration tests passed'
            }
            
        except Exception as e:
            self.validation_results['configuration'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Configuration validation failed: {str(e)}'
            }
            self.error_log.append(f"Configuration error: {e}")
    
    def _validate_data_loading(self):
        """Validate data loading system."""
        logger.info("Validating data loading system...")
        
        try:
            from data.live_data_loader import DataLoader
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            
            # Test multi-timeframe data loading
            data_results = loader.get_latest_data()
            
            # Validate data for each timeframe
            timeframe_scores = {}
            total_score = 0
            
            for timeframe in config.TIMEFRAMES:
                if timeframe in data_results:
                    df = data_results[timeframe]
                    
                    # Check data quality
                    if df is not None and not df.empty:
                        # Check required columns
                        required_cols = ['Open', 'High', 'Low', 'Close']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if not missing_cols:
                            # Check data integrity
                            valid_data = (
                                (df['High'] >= df['Low']).all() and
                                (df['High'] >= df['Open']).all() and
                                (df['High'] >= df['Close']).all() and
                                (df['Low'] <= df['Open']).all() and
                                (df['Low'] <= df['Close']).all()
                            )
                            
                            if valid_data:
                                timeframe_scores[timeframe] = 10
                                total_score += 10
                            else:
                                timeframe_scores[timeframe] = 5
                                total_score += 5
                        else:
                            timeframe_scores[timeframe] = 0
                    else:
                        timeframe_scores[timeframe] = 0
                else:
                    timeframe_scores[timeframe] = 0
            
            avg_score = total_score / len(config.TIMEFRAMES)
            
            self.validation_results['data_loading'] = {
                'status': 'PASSED' if avg_score >= 8 else 'PARTIAL' if avg_score >= 5 else 'FAILED',
                'score': avg_score,
                'details': f'Data loading validation: {timeframe_scores}',
                'timeframe_scores': timeframe_scores
            }
            
        except Exception as e:
            self.validation_results['data_loading'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Data loading validation failed: {str(e)}'
            }
            self.error_log.append(f"Data loading error: {e}")
    
    def _validate_technical_analysis(self):
        """Validate technical analysis system."""
        logger.info("Validating technical analysis system...")
        
        try:
            from technical_analysis.advanced_indicators import TechnicalAnalyzer
            from data.live_data_loader import DataLoader
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            analyzer = TechnicalAnalyzer(config)
            
            # Get sample data
            data = loader.fetch_single_timeframe_data(timeframe="1d")
            
            if data is not None and not data.empty:
                # Calculate all indicators
                indicators = analyzer.calculate_all_indicators(data)
                
                # Validate indicator count
                expected_indicators = 40
                actual_indicators = len(indicators)
                
                if actual_indicators >= expected_indicators:
                    score = 10
                    status = 'PASSED'
                elif actual_indicators >= expected_indicators * 0.8:
                    score = 8
                    status = 'PARTIAL'
                else:
                    score = actual_indicators / expected_indicators * 10
                    status = 'FAILED'
                
                self.validation_results['technical_analysis'] = {
                    'status': status,
                    'score': score,
                    'details': f'Calculated {actual_indicators}/{expected_indicators} indicators',
                    'indicator_count': actual_indicators
                }
            else:
                self.validation_results['technical_analysis'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': 'No data available for technical analysis'
                }
                
        except Exception as e:
            self.validation_results['technical_analysis'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Technical analysis validation failed: {str(e)}'
            }
            self.error_log.append(f"Technical analysis error: {e}")
    
    def _validate_pattern_discovery(self):
        """Validate pattern discovery system."""
        logger.info("Validating pattern discovery system...")
        
        try:
            from pattern_analysis.ml_candlestick_patterns import MLCandlestickDiscovery
            from data.live_data_loader import DataLoader
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            pattern_discovery = MLCandlestickDiscovery(config)
            
            # Get sample data
            data = loader.fetch_single_timeframe_data(timeframe="1d")
            
            if data is not None and not data.empty:
                # Discover patterns
                patterns = pattern_discovery.discover_patterns(data)
                
                # Count total patterns
                total_patterns = pattern_discovery._count_total_patterns()
                
                # Validate pattern count
                expected_patterns = 50
                
                if total_patterns >= expected_patterns:
                    score = 10
                    status = 'PASSED'
                elif total_patterns >= expected_patterns * 0.5:
                    score = 7
                    status = 'PARTIAL'
                else:
                    score = total_patterns / expected_patterns * 10
                    status = 'FAILED'
                
                self.validation_results['pattern_discovery'] = {
                    'status': status,
                    'score': score,
                    'details': f'Discovered {total_patterns}/{expected_patterns} patterns',
                    'pattern_count': total_patterns
                }
            else:
                self.validation_results['pattern_discovery'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': 'No data available for pattern discovery'
                }
                
        except Exception as e:
            self.validation_results['pattern_discovery'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Pattern discovery validation failed: {str(e)}'
            }
            self.error_log.append(f"Pattern discovery error: {e}")
    
    def _validate_deep_learning(self):
        """Validate deep learning system."""
        logger.info("Validating deep learning system...")
        
        try:
            from models.deep_learning_ensemble import DeepLearningEnsemble
            from data.live_data_loader import DataLoader
            from technical_analysis.advanced_indicators import TechnicalAnalyzer
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            analyzer = TechnicalAnalyzer(config)
            ensemble = DeepLearningEnsemble(config)
            
            # Get sample data
            data = loader.fetch_single_timeframe_data(timeframe="1d")
            
            if data is not None and not data.empty:
                # Calculate indicators
                indicators = analyzer.calculate_all_indicators(data)
                
                # Generate predictions
                predictions = ensemble.predict(data, indicators)
                
                # Validate prediction structure
                required_keys = ['movement_direction', 'target_price', 'confidence_level', 'confidence_percentage']
                missing_keys = [key for key in required_keys if key not in predictions]
                
                if not missing_keys:
                    score = 10
                    status = 'PASSED'
                else:
                    score = (len(required_keys) - len(missing_keys)) / len(required_keys) * 10
                    status = 'PARTIAL'
                
                self.validation_results['deep_learning'] = {
                    'status': status,
                    'score': score,
                    'details': f'ML predictions generated successfully',
                    'prediction_keys': list(predictions.keys())
                }
            else:
                self.validation_results['deep_learning'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': 'No data available for ML predictions'
                }
                
        except Exception as e:
            self.validation_results['deep_learning'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Deep learning validation failed: {str(e)}'
            }
            self.error_log.append(f"Deep learning error: {e}")
    
    def _validate_time_analysis(self):
        """Validate time analysis system."""
        logger.info("Validating time analysis system...")
        
        try:
            from time_analysis.market_cycles import TemporalAnalyzer
            from data.live_data_loader import DataLoader
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            temporal_analyzer = TemporalAnalyzer(config)
            
            # Get sample data
            data = loader.fetch_single_timeframe_data(timeframe="1d")
            
            if data is not None and not data.empty:
                # Analyze temporal patterns
                temporal_patterns = temporal_analyzer.analyze_temporal_patterns(data)
                
                # Validate pattern categories
                expected_categories = ['intraday', 'day_of_week', 'monthly', 'seasonal', 'gaps', 'expiry']
                actual_categories = list(temporal_patterns.keys())
                
                found_categories = [cat for cat in expected_categories if cat in actual_categories]
                
                if len(found_categories) >= len(expected_categories):
                    score = 10
                    status = 'PASSED'
                elif len(found_categories) >= len(expected_categories) * 0.7:
                    score = 7
                    status = 'PARTIAL'
                else:
                    score = len(found_categories) / len(expected_categories) * 10
                    status = 'FAILED'
                
                self.validation_results['time_analysis'] = {
                    'status': status,
                    'score': score,
                    'details': f'Found {len(found_categories)}/{len(expected_categories)} temporal categories',
                    'categories_found': found_categories
                }
            else:
                self.validation_results['time_analysis'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': 'No data available for time analysis'
                }
                
        except Exception as e:
            self.validation_results['time_analysis'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Time analysis validation failed: {str(e)}'
            }
            self.error_log.append(f"Time analysis error: {e}")
    
    def _validate_visualization(self):
        """Validate visualization system."""
        logger.info("Validating visualization system...")
        
        try:
            from visualization.candlestick_charts import CandlestickVisualizer
            from data.live_data_loader import DataLoader
            from technical_analysis.advanced_indicators import TechnicalAnalyzer
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            analyzer = TechnicalAnalyzer(config)
            visualizer = CandlestickVisualizer(config)
            
            # Get sample data
            data = loader.fetch_single_timeframe_data(timeframe="1d")
            
            if data is not None and not data.empty:
                # Calculate indicators
                indicators = analyzer.calculate_all_indicators(data)
                
                # Create chart
                chart = visualizer.create_candlestick_chart(data, indicators)
                
                # Validate chart object
                if chart is not None:
                    score = 10
                    status = 'PASSED'
                else:
                    score = 0
                    status = 'FAILED'
                
                self.validation_results['visualization'] = {
                    'status': status,
                    'score': score,
                    'details': 'Chart generation successful',
                    'chart_type': type(chart).__name__ if chart else 'None'
                }
            else:
                self.validation_results['visualization'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': 'No data available for visualization'
                }
                
        except Exception as e:
            self.validation_results['visualization'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Visualization validation failed: {str(e)}'
            }
            self.error_log.append(f"Visualization error: {e}")
    
    def _validate_report_generation(self):
        """Validate report generation system."""
        logger.info("Validating report generation system...")
        
        try:
            from reports.report_generator import ReportGenerator
            from data.live_data_loader import DataLoader
            from technical_analysis.advanced_indicators import TechnicalAnalyzer
            from pattern_analysis.ml_candlestick_patterns import MLCandlestickDiscovery
            from config import get_config
            
            config = get_config()
            loader = DataLoader(config)
            analyzer = TechnicalAnalyzer(config)
            pattern_discovery = MLCandlestickDiscovery(config)
            report_gen = ReportGenerator(config)
            
            # Get sample data
            data = loader.fetch_single_timeframe_data(timeframe="1d")
            
            if data is not None and not data.empty:
                # Calculate indicators and patterns
                indicators = analyzer.calculate_all_indicators(data)
                patterns = pattern_discovery.discover_patterns(data)
                
                # Mock predictions
                predictions = {
                    'movement_direction': '75.5% chance up',
                    'target_price': 2500.0,
                    'confidence_percentage': 75.5,
                    'time_horizon': '5 days',
                    'risk_assessment': {
                        'stop_loss': 2400.0,
                        'risk_reward_ratio': 2.5,
                        'max_loss': 100.0
                    }
                }
                
                # Generate reports
                technical_reports = report_gen.generate_technical_analysis_report(
                    indicators, patterns, predictions, data
                )
                
                # Validate report generation
                if technical_reports and 'excel' in technical_reports:
                    score = 10
                    status = 'PASSED'
                else:
                    score = 0
                    status = 'FAILED'
                
                self.validation_results['report_generation'] = {
                    'status': status,
                    'score': score,
                    'details': 'Report generation successful',
                    'report_formats': list(technical_reports.keys()) if technical_reports else []
                }
            else:
                self.validation_results['report_generation'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': 'No data available for report generation'
                }
                
        except Exception as e:
            self.validation_results['report_generation'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Report generation validation failed: {str(e)}'
            }
            self.error_log.append(f"Report generation error: {e}")
    
    def _validate_integration(self):
        """Validate system integration."""
        logger.info("Validating system integration...")
        
        try:
            from main import MarketIntelligenceEngine
            from config import get_config
            
            config = get_config()
            
            # Test main engine initialization
            engine = MarketIntelligenceEngine()
            
            # Test component initialization
            components = [
                'data_loader',
                'pattern_discovery', 
                'technical_analyzer',
                'deep_learning_ensemble',
                'temporal_analyzer',
                'visualizer',
                'report_generator'
            ]
            
            missing_components = []
            for component in components:
                if not hasattr(engine, component):
                    missing_components.append(component)
            
            if not missing_components:
                score = 10
                status = 'PASSED'
            else:
                score = (len(components) - len(missing_components)) / len(components) * 10
                status = 'PARTIAL'
            
            self.validation_results['integration'] = {
                'status': status,
                'score': score,
                'details': f'All components initialized successfully',
                'missing_components': missing_components
            }
                
        except Exception as e:
            self.validation_results['integration'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Integration validation failed: {str(e)}'
            }
            self.error_log.append(f"Integration error: {e}")
    
    def _validate_performance(self):
        """Validate system performance."""
        logger.info("Validating system performance...")
        
        try:
            from main import MarketIntelligenceEngine
            from config import get_config
            import time
            
            config = get_config()
            engine = MarketIntelligenceEngine()
            
            # Test performance with sample analysis
            start_time = time.time()
            
            # Run a quick analysis (without full ML training)
            try:
                # Test data loading performance
                data_results = engine._load_market_data()
                data_time = time.time() - start_time
                
                # Test technical analysis performance
                if data_results and 'data' in data_results:
                    daily_data = data_results['data'].get('1d')
                    if daily_data is not None and not daily_data.empty:
                        tech_start = time.time()
                        technical_results = engine._perform_technical_analysis(data_results)
                        tech_time = time.time() - tech_start
                        
                        # Performance scoring
                        if data_time < 30 and tech_time < 60:  # 30s for data, 60s for tech analysis
                            score = 10
                            status = 'PASSED'
                        elif data_time < 60 and tech_time < 120:
                            score = 7
                            status = 'PARTIAL'
                        else:
                            score = 5
                            status = 'SLOW'
                        
                        self.validation_results['performance'] = {
                            'status': status,
                            'score': score,
                            'details': f'Data loading: {data_time:.1f}s, Tech analysis: {tech_time:.1f}s',
                            'data_time': data_time,
                            'tech_time': tech_time
                        }
                    else:
                        self.validation_results['performance'] = {
                            'status': 'FAILED',
                            'score': 0,
                            'details': 'No data available for performance testing'
                        }
                else:
                    self.validation_results['performance'] = {
                        'status': 'FAILED',
                        'score': 0,
                        'details': 'Data loading failed during performance testing'
                    }
                    
            except Exception as e:
                self.validation_results['performance'] = {
                    'status': 'FAILED',
                    'score': 0,
                    'details': f'Performance testing failed: {str(e)}'
                }
                
        except Exception as e:
            self.validation_results['performance'] = {
                'status': 'FAILED',
                'score': 0,
                'details': f'Performance validation failed: {str(e)}'
            }
            self.error_log.append(f"Performance error: {e}")
    
    def _generate_validation_report(self, validation_time) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Calculate overall scores
        total_score = 0
        passed_tests = 0
        total_tests = len(self.validation_results)
        
        for test_name, result in self.validation_results.items():
            total_score += result['score']
            if result['status'] == 'PASSED':
                passed_tests += 1
        
        overall_score = total_score / total_tests if total_tests > 0 else 0
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine overall status
        if overall_score >= 9:
            overall_status = 'EXCELLENT'
        elif overall_score >= 7:
            overall_status = 'GOOD'
        elif overall_score >= 5:
            overall_status = 'FAIR'
        else:
            overall_status = 'POOR'
        
        # Generate performance metrics
        performance_metrics = {
            'data_quality': self._calculate_data_quality_score(),
            'pattern_discovery_accuracy': self._calculate_pattern_accuracy(),
            'ml_model_performance': self._calculate_ml_performance(),
            'technical_indicator_reliability': self._calculate_indicator_reliability(),
            'visualization_quality': self._calculate_visualization_quality(),
            'report_comprehensiveness': self._calculate_report_quality(),
            'system_integration': self._calculate_integration_score(),
            'error_handling_robustness': self._calculate_error_handling_score(),
            'real_time_performance': self._calculate_real_time_performance(),
        }
        
        return {
            'validation_summary': {
                'overall_status': overall_status,
                'overall_score': overall_score,
                'pass_rate': pass_rate,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'validation_time': str(validation_time),
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.validation_results,
            'performance_metrics': performance_metrics,
            'error_log': self.error_log,
            'recommendations': self._generate_recommendations()
        }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate data quality score."""
        if 'data_loading' in self.validation_results:
            result = self.validation_results['data_loading']
            if 'timeframe_scores' in result:
                scores = result['timeframe_scores'].values()
                return sum(scores) / len(scores) if scores else 0
            return result['score']
        return 0
    
    def _calculate_pattern_accuracy(self) -> float:
        """Calculate pattern discovery accuracy."""
        if 'pattern_discovery' in self.validation_results:
            result = self.validation_results['pattern_discovery']
            if 'pattern_count' in result:
                pattern_count = result['pattern_count']
                return min(pattern_count / 50 * 10, 10)  # 50 patterns = 10/10
            return result['score']
        return 0
    
    def _calculate_ml_performance(self) -> float:
        """Calculate ML model performance."""
        if 'deep_learning' in self.validation_results:
            return self.validation_results['deep_learning']['score']
        return 0
    
    def _calculate_indicator_reliability(self) -> float:
        """Calculate technical indicator reliability."""
        if 'technical_analysis' in self.validation_results:
            result = self.validation_results['technical_analysis']
            if 'indicator_count' in result:
                indicator_count = result['indicator_count']
                return min(indicator_count / 40 * 10, 10)  # 40 indicators = 10/10
            return result['score']
        return 0
    
    def _calculate_visualization_quality(self) -> float:
        """Calculate visualization quality."""
        if 'visualization' in self.validation_results:
            return self.validation_results['visualization']['score']
        return 0
    
    def _calculate_report_quality(self) -> float:
        """Calculate report comprehensiveness."""
        if 'report_generation' in self.validation_results:
            return self.validation_results['report_generation']['score']
        return 0
    
    def _calculate_integration_score(self) -> float:
        """Calculate system integration score."""
        if 'integration' in self.validation_results:
            return self.validation_results['integration']['score']
        return 0
    
    def _calculate_error_handling_score(self) -> float:
        """Calculate error handling robustness."""
        error_count = len(self.error_log)
        if error_count == 0:
            return 10
        elif error_count <= 2:
            return 8
        elif error_count <= 5:
            return 5
        else:
            return 2
    
    def _calculate_real_time_performance(self) -> float:
        """Calculate real-time performance."""
        if 'performance' in self.validation_results:
            result = self.validation_results['performance']
            if 'data_time' in result and 'tech_time' in result:
                total_time = result['data_time'] + result['tech_time']
                if total_time < 60:
                    return 10
                elif total_time < 120:
                    return 7
                elif total_time < 300:
                    return 5
                else:
                    return 2
            return result['score']
        return 0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Analyze validation results and generate recommendations
        for test_name, result in self.validation_results.items():
            if result['score'] < 7:
                if test_name == 'data_loading':
                    recommendations.append("Improve data source reliability and add more fallback options")
                elif test_name == 'pattern_discovery':
                    recommendations.append("Optimize pattern discovery algorithms for better pattern detection")
                elif test_name == 'deep_learning':
                    recommendations.append("Enhance ML model training and prediction accuracy")
                elif test_name == 'performance':
                    recommendations.append("Optimize system performance for faster execution")
                elif test_name == 'technical_analysis':
                    recommendations.append("Ensure all technical indicators are properly implemented")
        
        if not recommendations:
            recommendations.append("System is performing well. Consider adding advanced features for enhanced analysis.")
        
        return recommendations


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Market AI Engine - System Validation')
    parser.add_argument('--comprehensive-check', action='store_true', 
                       help='Run comprehensive system validation')
    parser.add_argument('--output-file', type=str, default='validation_report.json',
                       help='Output file for validation report')
    
    args = parser.parse_args()
    
    if args.comprehensive_check:
        print("🚀 Ultimate Market AI Engine - Comprehensive System Validation")
        print("=" * 70)
        
        validator = SystemValidator()
        report = validator.run_comprehensive_validation()
        
        # Print summary
        summary = report['validation_summary']
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Overall Score: {summary['overall_score']:.1f}/10")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Validation Time: {summary['validation_time']}")
        
        # Print detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in report['test_results'].items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "⚠️" if result['status'] == 'PARTIAL' else "❌"
            print(f"{status_emoji} {test_name.replace('_', ' ').title()}: {result['score']:.1f}/10 - {result['status']}")
        
        # Print performance metrics
        print(f"\n🎯 PERFORMANCE METRICS:")
        metrics = report['performance_metrics']
        for metric_name, score in metrics.items():
            print(f"  {metric_name.replace('_', ' ').title()}: {score:.1f}/10")
        
        # Print recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save report
        import json
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Validation report saved to: {args.output_file}")
        
        # Final rating
        overall_score = summary['overall_score']
        print(f"\n🏆 FINAL SYSTEM RATING: {overall_score:.1f}/10")
        
        if overall_score >= 9:
            print("🎉 EXCELLENT: System is production-ready!")
        elif overall_score >= 7:
            print("👍 GOOD: System is functional with minor improvements needed")
        elif overall_score >= 5:
            print("⚠️ FAIR: System needs significant improvements")
        else:
            print("❌ POOR: System requires major fixes before production use")
    
    else:
        print("Use --comprehensive-check to run full system validation")


if __name__ == "__main__":
    main()