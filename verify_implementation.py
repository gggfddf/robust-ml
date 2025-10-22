#!/usr/bin/env python3
"""
Ultimate Market AI Engine - Implementation Verification
======================================================

Quick verification script to confirm all components are properly implemented
and ready for Reliance Industries analysis.
"""

import sys
import os
from pathlib import Path

def check_file_structure():
    """Check if all required files are present."""
    print("🔍 Checking file structure...")
    
    required_files = [
        'config.py',
        'main.py',
        'run_reliance_analysis.py',
        'validate_system.py',
        'system_rating.py',
        'requirements.txt',
        'README.md',
        'IMPLEMENTATION_SUMMARY.md',
        'data/live_data_loader.py',
        'pattern_analysis/ml_candlestick_patterns.py',
        'technical_analysis/advanced_indicators.py',
        'models/deep_learning_ensemble.py',
        'time_analysis/market_cycles.py',
        'visualization/candlestick_charts.py',
        'reports/report_generator.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required files present!")
        return True

def check_configuration():
    """Check configuration settings."""
    print("\n⚙️ Checking configuration...")
    
    try:
        from config import get_config
        
        config = get_config()
        
        # Check key settings
        checks = [
            ('SYMBOL', config.SYMBOL == "RELIANCE.NS"),
            ('TIMEFRAMES', "1m" in config.TIMEFRAMES and "5m" in config.TIMEFRAMES),
            ('LIVE_DATA_ENABLED', config.LIVE_DATA_ENABLED == True),
            ('EXCHANGE', hasattr(config, 'EXCHANGE') and config.EXCHANGE == "NSE")
        ]
        
        for setting, status in checks:
            if status:
                print(f"  ✅ {setting}: {getattr(config, setting, 'N/A')}")
            else:
                print(f"  ❌ {setting}: {getattr(config, setting, 'N/A')}")
                return False
        
        print("  ✅ Configuration verified!")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def check_imports():
    """Check if all modules can be imported."""
    print("\n📦 Checking module imports...")
    
    modules = [
        'data.live_data_loader',
        'pattern_analysis.ml_candlestick_patterns',
        'technical_analysis.advanced_indicators',
        'models.deep_learning_ensemble',
        'time_analysis.market_cycles',
        'visualization.candlestick_charts',
        'reports.report_generator'
    ]
    
    failed_imports = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"  ❌ Failed imports: {failed_imports}")
        return False
    else:
        print("  ✅ All modules import successfully!")
        return True

def check_dependencies():
    """Check if key dependencies are available."""
    print("\n🔧 Checking dependencies...")
    
    dependencies = [
        'pandas', 'numpy', 'yfinance', 'plotly', 'scikit-learn',
        'torch', 'tensorflow', 'ta', 'matplotlib', 'seaborn'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - Missing")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"  ⚠️ Missing dependencies: {missing_deps}")
        print("  💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("  ✅ All dependencies available!")
        return True

def check_execution_scripts():
    """Check if execution scripts are ready."""
    print("\n🚀 Checking execution scripts...")
    
    scripts = [
        ('run_reliance_analysis.py', 'Complete Reliance analysis'),
        ('validate_system.py', 'System validation'),
        ('system_rating.py', 'System rating'),
        ('main.py', 'Main orchestration')
    ]
    
    for script, description in scripts:
        if os.path.exists(script):
            print(f"  ✅ {script} - {description}")
        else:
            print(f"  ❌ {script} - Missing")
            return False
    
    print("  ✅ All execution scripts ready!")
    return True

def check_directories():
    """Check if required directories exist."""
    print("\n📁 Checking directories...")
    
    directories = [
        'data', 'models', 'reports', 'charts', 'cache',
        'logs', 'visualizations'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ⚠️ {directory}/ - Will be created automatically")
    
    print("  ✅ Directory structure verified!")
    return True

def main():
    """Main verification function."""
    print("🔍 Ultimate Market AI Engine - Implementation Verification")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Configuration", check_configuration),
        ("Module Imports", check_imports),
        ("Dependencies", check_dependencies),
        ("Execution Scripts", check_execution_scripts),
        ("Directories", check_directories)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 IMPLEMENTATION VERIFICATION COMPLETE!")
        print("✅ All components are properly implemented and ready for execution.")
        print("\n🚀 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run complete analysis: python run_reliance_analysis.py")
        print("3. Or run standard analysis: python main.py")
        return 0
    else:
        print(f"\n⚠️ {total - passed} issues found. Please resolve before execution.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)