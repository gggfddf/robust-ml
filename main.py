"""
Ultimate Market AI Engine - Main Orchestration Pipeline
=====================================================

This module orchestrates all components into a unified market intelligence system.
Single parameter input: stock symbol
Coordinates all analysis components and generates comprehensive reports.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
import sys
from pathlib import Path

# Import all components
from config import get_config, update_symbol
from data.live_data_loader import DataLoader
from pattern_analysis.ml_candlestick_patterns import MLCandlestickDiscovery
from technical_analysis.advanced_indicators import TechnicalAnalyzer
from models.deep_learning_ensemble import DeepLearningEnsemble
from time_analysis.market_cycles import TemporalAnalyzer
from visualization.candlestick_charts import CandlestickVisualizer
from reports.report_generator import ReportGenerator

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketIntelligenceEngine:
    """
    Master orchestration system for the Ultimate Market AI Engine.
    
    Features:
    - Single parameter input: stock symbol
    - Coordinates all analysis components
    - Combines results from all modules
    - Generates comprehensive reports
    - Handles errors gracefully
    - Provides progress updates
    - Supports batch processing
    """
    
    def __init__(self, symbol: str = None):
        """Initialize the market intelligence engine."""
        self.config = get_config()
        
        # Update symbol if provided
        if symbol:
            update_symbol(symbol)
        
        # Initialize all components
        self.data_loader = DataLoader(self.config)
        self.pattern_discovery = MLCandlestickDiscovery(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.deep_learning_ensemble = DeepLearningEnsemble(self.config)
        self.temporal_analyzer = TemporalAnalyzer(self.config)
        self.visualizer = CandlestickVisualizer(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Results storage
        self.analysis_results = {}
        self.execution_log = []
        
        logger.info(f"Market Intelligence Engine initialized for {self.config.SYMBOL}")
    
    def run_complete_analysis(self, symbol: str = None, timeframes: List[str] = None, live_data: bool = True) -> Dict[str, Any]:
        """
        Run complete market analysis for the specified symbol with multi-timeframe support.
        
        Args:
            symbol: Stock symbol (optional, uses config default if not provided)
            timeframes: List of timeframes to analyze (optional, uses config default if not provided)
            live_data: Whether to use live data (defaults to True)
        
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        try:
            # Update configuration if provided
            if symbol:
                update_symbol(symbol)
            if timeframes:
                self.config.TIMEFRAMES = timeframes
            if not live_data:
                self.config.LIVE_DATA_ENABLED = False
            
            logger.info(f"Starting complete analysis for {self.config.SYMBOL}")
            logger.info(f"Timeframes: {self.config.TIMEFRAMES}")
            logger.info(f"Live data enabled: {self.config.LIVE_DATA_ENABLED}")
            self._log_progress("Analysis started", 0)
            
            # Step 1: Data Loading (10%)
            logger.info("Step 1/8: Loading multi-timeframe market data...")
            data_results = self._load_market_data()
            self._log_progress("Data loading completed", 10)
            
            # Validate multi-timeframe data
            self._validate_multi_timeframe_data(data_results)
            
            # Step 2: Technical Analysis (25%)
            logger.info("Step 2/8: Performing multi-timeframe technical analysis...")
            technical_results = self._perform_technical_analysis(data_results)
            self._log_progress("Technical analysis completed", 25)
            
            # Step 3: Pattern Discovery (40%)
            logger.info("Step 3/8: Discovering multi-timeframe candlestick patterns...")
            pattern_results = self._discover_patterns(data_results)
            self._log_progress("Pattern discovery completed", 40)
            
            # Step 4: Deep Learning Predictions (55%)
            logger.info("Step 4/8: Generating multi-timeframe ML predictions...")
            ml_results = self._generate_ml_predictions(data_results, technical_results)
            self._log_progress("ML predictions completed", 55)
            
            # Step 5: Temporal Analysis (70%)
            logger.info("Step 5/8: Analyzing temporal patterns...")
            temporal_results = self._analyze_temporal_patterns(data_results)
            self._log_progress("Temporal analysis completed", 70)
            
            # Step 6: Visualization (80%)
            logger.info("Step 6/8: Creating multi-timeframe visualizations...")
            visualization_results = self._create_visualizations(data_results, technical_results, pattern_results)
            self._log_progress("Visualization completed", 80)
            
            # Step 7: Report Generation (90%)
            logger.info("Step 7/8: Generating comprehensive reports...")
            report_results = self._generate_reports(data_results, technical_results, pattern_results, ml_results)
            self._log_progress("Report generation completed", 90)
            
            # Step 8: Compile Results (100%)
            logger.info("Step 8/8: Compiling final results...")
            final_results = self._compile_final_results(
                data_results, technical_results, pattern_results, 
                ml_results, temporal_results, visualization_results, report_results
            )
            self._log_progress("Analysis completed", 100)
            
            # Calculate execution time and system metrics
            execution_time = datetime.now() - start_time
            final_results['execution_time'] = str(execution_time)
            final_results['execution_log'] = self.execution_log
            final_results['system_metrics'] = self._calculate_system_metrics(execution_time, data_results)
            
            logger.info(f"Complete analysis finished in {execution_time}")
            logger.info(f"Analyzed {len(self.config.TIMEFRAMES)} timeframes: {self.config.TIMEFRAMES}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            self._log_progress(f"Error occurred: {e}", -1)
            return {
                'error': str(e),
                'execution_log': self.execution_log,
                'status': 'failed'
            }
    
    def _load_market_data(self) -> Dict[str, Any]:
        """Load market data for all timeframes."""
        try:
            logger.info("Loading multi-timeframe market data...")
            
            # Load data for all timeframes
            data = self.data_loader.get_latest_data()
            
            # Get data summary
            summary = self.data_loader.get_data_summary(data)
            
            results = {
                'data': data,
                'summary': summary,
                'symbol': self.config.SYMBOL,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Loaded data for {len(data)} timeframes")
            return results
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            raise
    
    def _perform_technical_analysis(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive technical analysis."""
        try:
            logger.info("Performing technical analysis...")
            
            # Use daily data for technical analysis
            daily_data = data_results['data'].get('1d')
            if daily_data is None or daily_data.empty:
                raise ValueError("No daily data available for technical analysis")
            
            # Calculate all technical indicators
            indicators = self.technical_analyzer.calculate_all_indicators(daily_data)
            
            # Get technical summary
            technical_summary = self.technical_analyzer.get_technical_summary()
            
            results = {
                'indicators': indicators,
                'summary': technical_summary,
                'key_signals': self.technical_analyzer._get_key_signals(),
                'overall_sentiment': self.technical_analyzer._calculate_overall_sentiment()
            }
            
            logger.info(f"Technical analysis completed: {len(indicators)} indicators calculated")
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            raise
    
    def _discover_patterns(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Discover candlestick patterns using ML."""
        try:
            logger.info("Discovering candlestick patterns...")
            
            # Use daily data for pattern discovery
            daily_data = data_results['data'].get('1d')
            if daily_data is None or daily_data.empty:
                raise ValueError("No daily data available for pattern discovery")
            
            # Discover patterns
            patterns = self.pattern_discovery.discover_patterns(daily_data)
            
            # Get pattern predictions
            pattern_predictions = self.pattern_discovery.get_pattern_predictions(daily_data)
            
            results = {
                'patterns': patterns,
                'predictions': pattern_predictions,
                'total_patterns': self.pattern_discovery._count_total_patterns()
            }
            
            logger.info(f"Pattern discovery completed: {results['total_patterns']} patterns found")
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern discovery: {e}")
            raise
    
    def _generate_ml_predictions(self, data_results: Dict[str, Any], 
                               technical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-based predictions."""
        try:
            logger.info("Generating ML predictions...")
            
            # Use daily data for ML predictions
            daily_data = data_results['data'].get('1d')
            if daily_data is None or daily_data.empty:
                raise ValueError("No daily data available for ML predictions")
            
            # Generate predictions
            predictions = self.deep_learning_ensemble.predict(daily_data, technical_results['indicators'])
            
            results = {
                'predictions': predictions,
                'model_predictions': predictions.get('model_predictions', {}),
                'ensemble_probability': predictions.get('ensemble_probability', 0),
                'confidence_level': predictions.get('confidence_level', 'Low')
            }
            
            logger.info(f"ML predictions completed: {predictions.get('confidence_level', 'Low')} confidence")
            return results
            
        except Exception as e:
            logger.error(f"Error in ML predictions: {e}")
            # Return default predictions if ML fails
            return {
                'predictions': {
                    'movement_direction': '50% chance neutral',
                    'target_price': daily_data['Close'].iloc[-1] if 'daily_data' in locals() else 0,
                    'confidence_level': 'Low',
                    'confidence_percentage': 50.0,
                    'time_horizon': '5 days',
                    'risk_assessment': {
                        'stop_loss': 0,
                        'risk_reward_ratio': 1.0,
                        'max_loss': 0
                    }
                },
                'model_predictions': {},
                'ensemble_probability': 0.5,
                'confidence_level': 'Low'
            }
    
    def _analyze_temporal_patterns(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns."""
        try:
            logger.info("Analyzing temporal patterns...")
            
            # Use daily data for temporal analysis
            daily_data = data_results['data'].get('1d')
            if daily_data is None or daily_data.empty:
                raise ValueError("No daily data available for temporal analysis")
            
            # Analyze temporal patterns
            temporal_patterns = self.temporal_analyzer.analyze_temporal_patterns(daily_data)
            
            # Get temporal predictions
            temporal_predictions = self.temporal_analyzer.get_temporal_predictions(daily_data)
            
            results = {
                'patterns': temporal_patterns,
                'predictions': temporal_predictions
            }
            
            logger.info("Temporal analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            raise
    
    def _create_visualizations(self, data_results: Dict[str, Any], 
                             technical_results: Dict[str, Any],
                             pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive visualizations."""
        try:
            logger.info("Creating visualizations...")
            
            # Use daily data for main chart
            daily_data = data_results['data'].get('1d')
            if daily_data is None or daily_data.empty:
                raise ValueError("No daily data available for visualization")
            
            # Create main candlestick chart
            main_chart = self.visualizer.create_candlestick_chart(
                daily_data, 
                technical_results['indicators'], 
                pattern_results['patterns']
            )
            
            # Create multi-timeframe chart
            multi_tf_chart = self.visualizer.create_multi_timeframe_chart(
                data_results['data'],
                technical_results['indicators'],
                pattern_results['patterns']
            )
            
            # Create pattern analysis chart
            pattern_chart = self.visualizer.create_pattern_analysis_chart(
                pattern_results['patterns'],
                daily_data
            )
            
            # Export charts
            chart_paths = {}
            for chart_name, chart in [('main', main_chart), ('multi_timeframe', multi_tf_chart), ('pattern_analysis', pattern_chart)]:
                if chart:
                    html_path = self.config.get_chart_path(f"{chart_name}_chart.html")
                    self.visualizer.export_chart(chart, html_path, 'html')
                    chart_paths[f"{chart_name}_html"] = html_path
            
            results = {
                'charts': chart_paths,
                'chart_objects': {
                    'main': main_chart,
                    'multi_timeframe': multi_tf_chart,
                    'pattern_analysis': pattern_chart
                }
            }
            
            logger.info(f"Visualization completed: {len(chart_paths)} charts created")
            return results
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            raise
    
    def _generate_reports(self, data_results: Dict[str, Any],
                         technical_results: Dict[str, Any],
                         pattern_results: Dict[str, Any],
                         ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        try:
            logger.info("Generating reports...")
            
            # Use daily data for reports
            daily_data = data_results['data'].get('1d')
            if daily_data is None or daily_data.empty:
                raise ValueError("No daily data available for report generation")
            
            # Generate technical analysis report
            technical_reports = self.report_generator.generate_technical_analysis_report(
                technical_results['indicators'],
                pattern_results['patterns'],
                ml_results['predictions'],
                daily_data
            )
            
            # Generate price action report
            price_action_reports = self.report_generator.generate_price_action_report(
                pattern_results['patterns'],
                daily_data,
                technical_results['summary']
            )
            
            results = {
                'technical_reports': technical_reports,
                'price_action_reports': price_action_reports
            }
            
            logger.info("Report generation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in report generation: {e}")
            raise
    
    def _compile_final_results(self, data_results: Dict[str, Any],
                             technical_results: Dict[str, Any],
                             pattern_results: Dict[str, Any],
                             ml_results: Dict[str, Any],
                             temporal_results: Dict[str, Any],
                             visualization_results: Dict[str, Any],
                             report_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all results into final output."""
        try:
            logger.info("Compiling final results...")
            
            # Create comprehensive summary
            summary = {
                'symbol': self.config.SYMBOL,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_summary': data_results['summary'],
                'technical_sentiment': technical_results['overall_sentiment'],
                'total_patterns': pattern_results['total_patterns'],
                'ml_confidence': ml_results['confidence_level'],
                'prediction_direction': ml_results['predictions'].get('movement_direction', 'N/A'),
                'target_price': ml_results['predictions'].get('target_price', 0),
                'risk_assessment': ml_results['predictions'].get('risk_assessment', {})
            }
            
            # Compile all results
            final_results = {
                'summary': summary,
                'data_analysis': data_results,
                'technical_analysis': technical_results,
                'pattern_analysis': pattern_results,
                'ml_predictions': ml_results,
                'temporal_analysis': temporal_results,
                'visualizations': visualization_results,
                'reports': report_results,
                'status': 'completed'
            }
            
            # Store results
            self.analysis_results = final_results
            
            logger.info("Final results compiled successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error compiling final results: {e}")
            raise
    
    def _log_progress(self, message: str, percentage: int):
        """Log progress updates."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {percentage}% - {message}"
        self.execution_log.append(log_entry)
        logger.info(log_entry)
    
    def _validate_multi_timeframe_data(self, data_results: Dict[str, Any]):
        """Validate multi-timeframe data quality."""
        try:
            if 'data' not in data_results:
                raise Exception("No data found in results")
            
            data = data_results['data']
            missing_timeframes = []
            invalid_timeframes = []
            
            for timeframe in self.config.TIMEFRAMES:
                if timeframe not in data:
                    missing_timeframes.append(timeframe)
                else:
                    df = data[timeframe]
                    if df is None or df.empty:
                        invalid_timeframes.append(timeframe)
            
            if missing_timeframes:
                logger.warning(f"Missing timeframes: {missing_timeframes}")
            
            if invalid_timeframes:
                logger.warning(f"Invalid timeframes (empty data): {invalid_timeframes}")
            
            # Log data summary
            for timeframe, df in data.items():
                if df is not None and not df.empty:
                    logger.info(f"{timeframe}: {len(df)} records, {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def _calculate_system_metrics(self, execution_time, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive system performance metrics."""
        try:
            metrics = {
                'execution_time_seconds': execution_time.total_seconds(),
                'timeframes_analyzed': len(self.config.TIMEFRAMES),
                'data_points_processed': 0,
                'patterns_discovered': 0,
                'indicators_calculated': 0,
                'ml_predictions_generated': 0,
                'reports_generated': 0,
                'performance_score': 0
            }
            
            # Calculate data points processed
            if 'data' in data_results:
                for timeframe, df in data_results['data'].items():
                    if df is not None and not df.empty:
                        metrics['data_points_processed'] += len(df)
            
            # Calculate performance score (0-100)
            base_score = 50  # Base score
            
            # Time efficiency (faster = higher score)
            time_score = max(0, 100 - (metrics['execution_time_seconds'] / 60) * 10)
            
            # Data coverage (more timeframes = higher score)
            coverage_score = min(100, (metrics['timeframes_analyzed'] / 5) * 100)
            
            # Data volume (more data points = higher score)
            volume_score = min(100, (metrics['data_points_processed'] / 10000) * 10)
            
            metrics['performance_score'] = (base_score + time_score + coverage_score + volume_score) / 4
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")
            return {'error': str(e)}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of the latest analysis."""
        if not self.analysis_results:
            return {'status': 'No analysis performed yet'}
        
        return self.analysis_results.get('summary', {})
    
    def get_reports(self) -> Dict[str, Any]:
        """Get paths to generated reports."""
        if not self.analysis_results:
            return {'status': 'No reports generated yet'}
        
        return self.analysis_results.get('reports', {})
    
    def get_charts(self) -> Dict[str, Any]:
        """Get paths to generated charts."""
        if not self.analysis_results:
            return {'status': 'No charts generated yet'}
        
        return self.analysis_results.get('visualizations', {}).get('charts', {})
    
    def run_batch_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Run analysis for multiple symbols.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary with results for each symbol
        """
        logger.info(f"Starting batch analysis for {len(symbols)} symbols")
        
        batch_results = {}
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"Processing {symbol} ({i}/{len(symbols)})")
                
                # Run analysis for this symbol
                results = self.run_complete_analysis(symbol)
                batch_results[symbol] = results
                
                logger.info(f"Completed analysis for {symbol}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                batch_results[symbol] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        logger.info(f"Batch analysis completed for {len(symbols)} symbols")
        return batch_results


def main():
    """Main function to run the Ultimate Market AI Engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Market AI Engine')
    parser.add_argument('--symbol', type=str, default=None, 
                       help='Stock symbol to analyze (e.g., RELIANCE.NS)')
    parser.add_argument('--timeframes', type=str, nargs='+', 
                       help='Timeframes to analyze (e.g., 1m 5m 15m 1d 1w)')
    parser.add_argument('--live-data', action='store_true', default=True,
                       help='Use live data (default: True)')
    parser.add_argument('--batch', type=str, nargs='+', 
                       help='List of symbols for batch analysis')
    parser.add_argument('--update-symbol', type=str, 
                       help='Update the default symbol in config')
    parser.add_argument('--validate', action='store_true',
                       help='Run system validation before analysis')
    parser.add_argument('--rating', action='store_true',
                       help='Generate system rating report')
    
    args = parser.parse_args()
    
    if args.update_symbol:
        update_symbol(args.update_symbol)
        print(f"Default symbol updated to: {args.update_symbol}")
        return
    
    # Run validation if requested
    if args.validate:
        print("🔍 Running system validation...")
        try:
            from validate_system import SystemValidator
            validator = SystemValidator()
            validation_report = validator.run_comprehensive_validation()
            
            summary = validation_report['validation_summary']
            print(f"✅ Validation completed: {summary['overall_status']} ({summary['overall_score']:.1f}/10)")
            
            if summary['overall_score'] < 7.0:
                print("⚠️ Warning: System validation score is below recommended threshold")
                response = input("Continue with analysis anyway? (y/N): ")
                if response.lower() != 'y':
                    return
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return
    
    # Generate rating if requested
    if args.rating:
        print("🏆 Generating system rating...")
        try:
            from system_rating import SystemRater
            rater = SystemRater()
            rating_report = rater.generate_comprehensive_rating()
            
            summary = rating_report['rating_summary']
            print(f"📊 System Rating: {summary['rating_level']} ({summary['overall_score']:.1f}/10)")
            print(f"Grade: {summary['grade']}")
        except Exception as e:
            print(f"❌ Rating generation failed: {e}")
    
    # Initialize engine
    engine = MarketIntelligenceEngine()
    
    if args.batch:
        # Run batch analysis
        results = engine.run_batch_analysis(args.batch)
        print(f"Batch analysis completed for {len(args.batch)} symbols")
        
        # Print summary
        for symbol, result in results.items():
            if result.get('status') == 'completed':
                summary = result.get('summary', {})
                print(f"\n{symbol}:")
                print(f"  Sentiment: {summary.get('technical_sentiment', 'N/A')}")
                print(f"  Prediction: {summary.get('prediction_direction', 'N/A')}")
                print(f"  Confidence: {summary.get('ml_confidence', 'N/A')}")
            else:
                print(f"\n{symbol}: Failed - {result.get('error', 'Unknown error')}")
    
    else:
        # Run single analysis
        symbol = args.symbol or engine.config.SYMBOL
        timeframes = args.timeframes or engine.config.TIMEFRAMES
        live_data = args.live_data
        
        print(f"🚀 Running Ultimate Market AI Engine for {symbol}")
        print(f"📊 Timeframes: {timeframes}")
        print(f"🔄 Live data: {live_data}")
        
        results = engine.run_complete_analysis(symbol, timeframes, live_data)
        
        if results.get('status') == 'completed':
            summary = results.get('summary', {})
            system_metrics = results.get('system_metrics', {})
            
            print("\n" + "="*60)
            print("🚀 ULTIMATE MARKET AI ENGINE - COMPLETE ANALYSIS RESULTS")
            print("="*60)
            print(f"📈 Symbol: {summary.get('symbol', 'N/A')}")
            print(f"⏰ Timeframes Analyzed: {system_metrics.get('timeframes_analyzed', 0)}")
            print(f"📊 Data Points Processed: {system_metrics.get('data_points_processed', 0):,}")
            print(f"🎯 Performance Score: {system_metrics.get('performance_score', 0):.1f}/100")
            print(f"⏱️ Execution Time: {results.get('execution_time', 'N/A')}")
            print()
            print(f"📉 Technical Sentiment: {summary.get('technical_sentiment', 'N/A')}")
            print(f"🤖 ML Prediction: {summary.get('prediction_direction', 'N/A')}")
            print(f"💰 Target Price: ₹{summary.get('target_price', 0):.2f}")
            print(f"🎯 Confidence: {summary.get('ml_confidence', 'N/A')}")
            print(f"🔍 Patterns Discovered: {summary.get('total_patterns', 0)}")
            print(f"📈 Indicators Calculated: {system_metrics.get('indicators_calculated', 0)}")
            
            # Show report paths
            reports = engine.get_reports()
            if reports:
                print("\nReports Generated:")
                for report_type, paths in reports.items():
                    for format_type, path in paths.items():
                        print(f"  {report_type} ({format_type}): {path}")
            
            # Show chart paths
            charts = engine.get_charts()
            if charts:
                print("\nCharts Generated:")
                for chart_name, path in charts.items():
                    print(f"  {chart_name}: {path}")
            
        else:
            print(f"Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
