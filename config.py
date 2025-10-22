"""
Ultimate Market AI Engine - Configuration System
===============================================

This module contains the configuration for the Ultimate Market AI Engine.
ONLY the SYMBOL parameter should be edited by users.
All other parameters are optimized defaults for Indian market analysis.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging
from pathlib import Path


@dataclass
class MarketConfig:
    """Configuration for market analysis parameters."""
    
    # ===== USER EDITABLE PARAMETER =====
    SYMBOL: str = "RELIANCE.NS"  # ONLY EDIT THIS - Use .NS for NSE, .BO for BSE
    EXCHANGE: str = "NSE"
    
    # ===== OPTIMIZED DEFAULTS (DO NOT EDIT) =====
    
    # Timeframes for analysis
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1d", "1w"])
    
    # Data parameters
    HISTORICAL_YEARS: int = 2
    DATA_SOURCE: str = "yfinance"
    CACHE_DURATION: int = 300  # 5 minutes
    LIVE_DATA_ENABLED: bool = True
    ANALYSIS_PERIOD: str = "2y"
    
    # Market hours (IST)
    MARKET_OPEN: str = "09:15"
    MARKET_CLOSE: str = "15:30"
    TIMEZONE: str = "Asia/Kolkata"
    
    # Technical Analysis
    TECHNICAL_INDICATORS: List[str] = field(default_factory=lambda: [
        "sma", "ema", "bollinger_bands", "rsi", "macd", "stochastic", "atr",
        "williams_r", "cci", "adx", "parabolic_sar", "ichimoku", "fibonacci",
        "pivot_points", "vwap", "obv", "ad_line", "mfi", "roc", "momentum",
        "ultimate_oscillator", "trix", "kst", "money_flow_index", "chaikin_money_flow",
        "force_index", "ease_of_movement", "commodity_channel_index", "detrended_price_oscillator",
        "klinger_oscillator", "know_sure_thing", "mass_index", "median_price",
        "price_oscillator", "price_rate_of_change", "trix", "true_strength_index",
        "vortex_indicator", "williams_alligator", "zigzag"
    ])
    
    # ML Model Parameters
    LSTM_UNITS: int = 128
    LSTM_LAYERS: int = 3
    CNN_FILTERS: List[int] = field(default_factory=lambda: [32, 64, 128])
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 6
    DROPOUT_RATE: float = 0.2
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    VALIDATION_SPLIT: float = 0.2
    
    # Pattern Discovery
    MIN_PATTERN_LENGTH: int = 1
    MAX_PATTERN_LENGTH: int = 20
    CLUSTERING_ALGORITHMS: List[str] = field(default_factory=lambda: ["kmeans", "dbscan", "gmm"])
    MIN_PATTERN_FREQUENCY: int = 10
    STATISTICAL_SIGNIFICANCE: float = 0.05
    
    # Time Analysis
    INTRADAY_HOURS: List[int] = field(default_factory=lambda: list(range(9, 16)))
    GAP_THRESHOLD: float = 0.02  # 2% gap threshold
    EXPIRY_DAYS: List[int] = field(default_factory=lambda: [25, 26, 27, 28, 29, 30])
    
    # Visualization
    CHART_THEME: str = "plotly_dark"
    CHART_HEIGHT: int = 600
    CHART_WIDTH: int = 1000
    EXPORT_FORMATS: List[str] = field(default_factory=lambda: ["html", "png", "pdf"])
    
    # Report Generation
    REPORT_TEMPLATES: Dict[str, str] = field(default_factory=lambda: {
        "technical": "templates/technical_report.html",
        "price_action": "templates/price_action_report.html",
        "summary": "templates/summary_report.html"
    })
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "market_ai_engine.log"
    
    # Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    REPORTS_DIR: str = "reports"
    CHARTS_DIR: str = "charts"
    CACHE_DIR: str = "cache"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_symbol()
        self._create_directories()
        self._setup_logging()
    
    def _validate_symbol(self):
        """Validate the stock symbol format."""
        if not self.SYMBOL:
            raise ValueError("SYMBOL cannot be empty")
        
        # Check for valid Indian market suffixes
        valid_suffixes = [".NS", ".BO", ".NSE", ".BSE"]
        if not any(self.SYMBOL.endswith(suffix) for suffix in valid_suffixes):
            raise ValueError(f"SYMBOL must end with one of: {valid_suffixes}")
        
        # Remove any spaces and convert to uppercase
        self.SYMBOL = self.SYMBOL.strip().upper()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.REPORTS_DIR,
            self.CHARTS_DIR,
            self.CACHE_DIR,
            "templates",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format=self.LOG_FORMAT,
            handlers=[
                logging.FileHandler(f"logs/{self.LOG_FILE}"),
                logging.StreamHandler()
            ]
        )
    
    def get_data_path(self, filename: str) -> str:
        """Get full path for data files."""
        return os.path.join(self.DATA_DIR, filename)
    
    def get_model_path(self, filename: str) -> str:
        """Get full path for model files."""
        return os.path.join(self.MODELS_DIR, filename)
    
    def get_report_path(self, filename: str) -> str:
        """Get full path for report files."""
        return os.path.join(self.REPORTS_DIR, filename)
    
    def get_chart_path(self, filename: str) -> str:
        """Get full path for chart files."""
        return os.path.join(self.CHARTS_DIR, filename)
    
    def get_cache_path(self, filename: str) -> str:
        """Get full path for cache files."""
        return os.path.join(self.CACHE_DIR, filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "symbol": self.SYMBOL,
            "timeframes": self.TIMEFRAMES,
            "historical_years": self.HISTORICAL_YEARS,
            "data_source": self.DATA_SOURCE,
            "market_hours": {
                "open": self.MARKET_OPEN,
                "close": self.MARKET_CLOSE,
                "timezone": self.TIMEZONE
            },
            "technical_indicators_count": len(self.TECHNICAL_INDICATORS),
            "ml_parameters": {
                "lstm_units": self.LSTM_UNITS,
                "lstm_layers": self.LSTM_LAYERS,
                "dropout_rate": self.DROPOUT_RATE,
                "learning_rate": self.LEARNING_RATE
            }
        }


# Global configuration instance
config = MarketConfig()


def get_config() -> MarketConfig:
    """Get the global configuration instance."""
    return config


def update_symbol(new_symbol: str):
    """Update the stock symbol in configuration."""
    config.SYMBOL = new_symbol
    config._validate_symbol()
    logging.info(f"Symbol updated to: {config.SYMBOL}")


if __name__ == "__main__":
    # Test configuration
    print("Ultimate Market AI Engine Configuration")
    print("=" * 50)
    print(f"Symbol: {config.SYMBOL}")
    print(f"Timeframes: {config.TIMEFRAMES}")
    print(f"Technical Indicators: {len(config.TECHNICAL_INDICATORS)}")
    print(f"Data Source: {config.DATA_SOURCE}")
    print(f"Market Hours: {config.MARKET_OPEN} - {config.MARKET_CLOSE} IST")
    print(f"ML Parameters: LSTM {config.LSTM_LAYERS} layers, {config.LSTM_UNITS} units")
    print("\nConfiguration validation passed!")
