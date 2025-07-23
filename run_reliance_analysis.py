#!/usr/bin/env python3
"""
Ultimate Market AI Engine - Reliance Industries Complete Analysis
================================================================

Comprehensive execution script for Reliance Industries Limited (RELIANCE.NS)
with full multi-timeframe analysis, live data integration, and complete feature preservation.

This script demonstrates the complete Ultimate Market AI Engine capabilities.
"""

import sys
import os
import logging
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reliance_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print the Ultimate Market AI Engine banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    🚀 ULTIMATE MARKET AI ENGINE 🚀                          ║
    ║                                                                              ║
    ║                    Reliance Industries Limited Analysis                     ║
    ║                                                                              ║
    ║  • Multi-Timeframe Analysis (1m, 5m, 15m, 1d, 1w)                         ║
    ║  • Live Data Integration with Multiple Sources                             ║
    ║  • Proprietary ML Pattern Discovery                                        ║
    ║  • 40+ Advanced Technical Indicators                                       ║
    ║  • Multi-Architecture Deep Learning Ensemble                               ║
    ║  • Time-Based Market Cycle Analysis                                        ║
    ║  • Professional Interactive Visualizations                                 ║
    ║  • Comprehensive Report Generation                                         ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check and install required dependencies."""
    print("🔧 Checking and installing dependencies...")
    
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'scikit-learn', 'plotly',
        'ta', 'torch', 'tensorflow', 'matplotlib', 'seaborn',
        'openpyxl', 'reportlab', 'jinja2', 'nsepython', 'nsepy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            try:
                os.system(f"pip install {package}")
                print(f"  ✅ Installed {package}")
            except Exception as e:
                print(f"  ❌ Failed to install {package}: {e}")
    else:
        print("  ✅ All dependencies are available!")

def run_system_validation():
    """Run comprehensive system validation."""
    print("\n🔍 Running comprehensive system validation...")
    
    try:
        from validate_system import SystemValidator
        
        validator = SystemValidator()
        validation_report = validator.run_comprehensive_validation()
        
        summary = validation_report['validation_summary']
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Overall Score: {summary['overall_score']:.1f}/10")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        # Print detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, result in validation_report['test_results'].items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "⚠️" if result['status'] == 'PARTIAL' else "❌"
            print(f"{status_emoji} {test_name.replace('_', ' ').title()}: {result['score']:.1f}/10")
        
        return validation_report
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

def run_system_rating():
    """Generate comprehensive system rating."""
    print("\n🏆 Generating comprehensive system rating...")
    
    try:
        from system_rating import SystemRater
        
        rater = SystemRater()
        rating_report = rater.generate_comprehensive_rating()
        
        summary = rating_report['rating_summary']
        print(f"\n📊 SYSTEM RATING:")
        print(f"Overall Score: {summary['overall_score']:.1f}/10")
        print(f"Rating Level: {summary['rating_level']}")
        print(f"Grade: {summary['grade']}")
        print(f"Description: {summary['description']}")
        
        # Print detailed scores
        print(f"\n📋 DETAILED SCORES:")
        for category, scores in rating_report['detailed_scores'].items():
            score = scores['overall_score']
            description = scores['description']
            weight = scores['weight']
            status_emoji = "🟢" if score >= 8.5 else "🟡" if score >= 7.0 else "🔴"
            print(f"{status_emoji} {description}: {score:.1f}/10 (Weight: {weight:.2f})")
        
        return rating_report
        
    except Exception as e:
        print(f"❌ Rating generation failed: {e}")
        return None

def run_complete_analysis():
    """Run complete Reliance Industries analysis."""
    print("\n🚀 Starting complete Reliance Industries analysis...")
    
    try:
        from main import MarketIntelligenceEngine
        
        # Initialize engine with Reliance Industries
        engine = MarketIntelligenceEngine("RELIANCE.NS")
        
        # Configure for complete analysis
        timeframes = ["1m", "5m", "15m", "1d", "1w"]
        live_data = True
        
        print(f"📈 Symbol: RELIANCE.NS")
        print(f"⏰ Timeframes: {timeframes}")
        print(f"🔄 Live Data: {live_data}")
        print(f"📅 Analysis Period: 2+ years historical data")
        
        # Run complete analysis
        start_time = time.time()
        results = engine.run_complete_analysis("RELIANCE.NS", timeframes, live_data)
        analysis_time = time.time() - start_time
        
        if results.get('status') == 'completed':
            summary = results.get('summary', {})
            system_metrics = results.get('system_metrics', {})
            
            print("\n" + "="*70)
            print("🎉 RELIANCE INDUSTRIES - COMPLETE ANALYSIS RESULTS")
            print("="*70)
            print(f"📈 Symbol: {summary.get('symbol', 'RELIANCE.NS')}")
            print(f"⏰ Timeframes Analyzed: {system_metrics.get('timeframes_analyzed', 0)}")
            print(f"📊 Data Points Processed: {system_metrics.get('data_points_processed', 0):,}")
            print(f"🎯 Performance Score: {system_metrics.get('performance_score', 0):.1f}/100")
            print(f"⏱️ Analysis Time: {analysis_time:.2f} seconds")
            print()
            print(f"📉 Technical Sentiment: {summary.get('technical_sentiment', 'N/A')}")
            print(f"🤖 ML Prediction: {summary.get('prediction_direction', 'N/A')}")
            print(f"💰 Target Price: ₹{summary.get('target_price', 0):.2f}")
            print(f"🎯 Confidence: {summary.get('ml_confidence', 'N/A')}")
            print(f"🔍 Patterns Discovered: {summary.get('total_patterns', 0)}")
            print(f"📈 Indicators Calculated: {system_metrics.get('indicators_calculated', 0)}")
            
            # Show generated files
            print(f"\n📄 GENERATED FILES:")
            reports = engine.get_reports()
            if reports:
                for report_type, paths in reports.items():
                    for format_type, path in paths.items():
                        if os.path.exists(path):
                            print(f"  📋 {report_type} ({format_type}): {path}")
            
            charts = engine.get_charts()
            if charts:
                for chart_type, path in charts.items():
                    if os.path.exists(path):
                        print(f"  📊 {chart_type}: {path}")
            
            return results
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis error: {e}")
        return None

def generate_execution_summary(validation_report, rating_report, analysis_results):
    """Generate comprehensive execution summary."""
    print("\n" + "="*70)
    print("📋 EXECUTION SUMMARY")
    print("="*70)
    
    # System Status
    if validation_report:
        val_summary = validation_report['validation_summary']
        print(f"🔍 System Validation: {val_summary['overall_status']} ({val_summary['overall_score']:.1f}/10)")
    
    if rating_report:
        rating_summary = rating_report['rating_summary']
        print(f"🏆 System Rating: {rating_summary['rating_level']} ({rating_summary['overall_score']:.1f}/10)")
    
    # Analysis Results
    if analysis_results and analysis_results.get('status') == 'completed':
        summary = analysis_results.get('summary', {})
        system_metrics = analysis_results.get('system_metrics', {})
        
        print(f"📈 Analysis Status: ✅ COMPLETED")
        print(f"🎯 Key Findings:")
        print(f"   • Technical Sentiment: {summary.get('technical_sentiment', 'N/A')}")
        print(f"   • ML Prediction: {summary.get('prediction_direction', 'N/A')}")
        print(f"   • Target Price: ₹{summary.get('target_price', 0):.2f}")
        print(f"   • Confidence Level: {summary.get('ml_confidence', 'N/A')}")
        print(f"   • Patterns Discovered: {summary.get('total_patterns', 0)}")
        print(f"   • Data Points Processed: {system_metrics.get('data_points_processed', 0):,}")
    else:
        print(f"📈 Analysis Status: ❌ FAILED")
    
    # Performance Metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    if analysis_results and 'system_metrics' in analysis_results:
        metrics = analysis_results['system_metrics']
        print(f"   • Execution Time: {metrics.get('execution_time_seconds', 0):.2f} seconds")
        print(f"   • Timeframes Analyzed: {metrics.get('timeframes_analyzed', 0)}")
        print(f"   • Performance Score: {metrics.get('performance_score', 0):.1f}/100")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if rating_report and 'recommendations' in rating_report:
        for i, rec in enumerate(rating_report['recommendations'][:3], 1):
            print(f"   {i}. {rec['recommendation']}")
    else:
        print("   • System is performing optimally")
        print("   • Consider running regular analysis for market monitoring")
        print("   • Review generated reports for detailed insights")

def main():
    """Main execution function."""
    print_banner()
    
    print("🎯 RELIANCE INDUSTRIES LIMITED - COMPLETE MARKET ANALYSIS")
    print("=" * 60)
    print(f"📅 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️ System: {sys.platform}")
    print(f"🐍 Python: {sys.version}")
    
    # Step 1: Check dependencies
    check_dependencies()
    
    # Step 2: Run system validation
    validation_report = run_system_validation()
    
    # Step 3: Generate system rating
    rating_report = run_system_rating()
    
    # Step 4: Run complete analysis
    analysis_results = run_complete_analysis()
    
    # Step 5: Generate execution summary
    generate_execution_summary(validation_report, rating_report, analysis_results)
    
    # Final status
    print(f"\n🎉 EXECUTION COMPLETED!")
    print(f"📁 Check the generated files in the current directory")
    print(f"📋 Review the analysis log: reliance_analysis.log")
    
    if analysis_results and analysis_results.get('status') == 'completed':
        print(f"✅ Reliance Industries analysis completed successfully!")
        return 0
    else:
        print(f"❌ Analysis encountered issues. Check logs for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)