"""
Ultimate Market AI Engine - Professional Candlestick Chart Visualization
======================================================================

This module creates professional-grade interactive candlestick charts
that rival institutional trading platforms.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CandlestickVisualizer:
    """
    Professional candlestick chart visualization system.
    
    Features:
    - Interactive candlestick charts (NO line charts)
    - Multi-timeframe views (5m, 15m, 1d, 1w)
    - Pattern highlighting with annotations
    - Support/resistance level overlays
    - Volume bars with color coding
    - Technical indicator overlays
    - Zoom, pan, and hover functionality
    - Pattern detection markers
    - Professional trading interface appearance
    """
    
    def __init__(self, config=None):
        """Initialize the candlestick visualizer."""
        self.config = config or get_config()
        
        # Chart configuration
        self.chart_height = self.config.CHART_HEIGHT
        self.chart_width = self.config.CHART_WIDTH
        self.chart_theme = self.config.CHART_THEME
        
        # Color schemes
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'volume_up': '#00ff88',
            'volume_down': '#ff4444',
            'support': '#00aaff',
            'resistance': '#ffaa00',
            'pattern': '#ff00ff',
            'background': '#1e1e1e',
            'grid': '#333333'
        }
        
        logger.info("Candlestick Visualizer initialized")
    
    def create_candlestick_chart(self, df: pd.DataFrame, indicators: Dict = None, 
                                patterns: Dict = None, title: str = "Market Analysis") -> go.Figure:
        """
        Create a comprehensive candlestick chart with all overlays.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Technical indicator values
            patterns: Pattern analysis results
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        if df is None or df.empty:
            logger.warning("No data provided for chart creation")
            return go.Figure()
        
        logger.info("Creating comprehensive candlestick chart...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(title, "Volume", "Indicators")
        )
        
        # Main candlestick chart
        self._add_candlestick_trace(fig, df, row=1)
        
        # Add volume
        if 'Volume' in df.columns:
            self._add_volume_trace(fig, df, row=2)
        
        # Add technical indicators
        if indicators:
            self._add_indicator_overlays(fig, df, indicators, row=3)
        
        # Add pattern markers
        if patterns:
            self._add_pattern_markers(fig, df, patterns, row=1)
        
        # Add support/resistance levels
        self._add_support_resistance(fig, df, row=1)
        
        # Update layout
        self._update_chart_layout(fig, title)
        
        logger.info("Candlestick chart created successfully")
        return fig
    
    def _add_candlestick_trace(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add candlestick trace to the chart."""
        # Create candlestick trace
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color=self.colors['bullish'],
            decreasing_line_color=self.colors['bearish'],
            increasing_fillcolor=self.colors['bullish'],
            decreasing_fillcolor=self.colors['bearish'],
            line=dict(width=1)
        )
        
        fig.add_trace(candlestick, row=row, col=1)
    
    def _add_volume_trace(self, fig: go.Figure, df: pd.DataFrame, row: int = 2):
        """Add volume bars to the chart."""
        # Color volume bars based on price direction
        colors = []
        for i in range(len(df)):
            if i == 0:
                colors.append(self.colors['neutral'])
            else:
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    colors.append(self.colors['volume_up'])
                else:
                    colors.append(self.colors['volume_down'])
        
        volume_trace = go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        )
        
        fig.add_trace(volume_trace, row=row, col=1)
    
    def _add_indicator_overlays(self, fig: go.Figure, df: pd.DataFrame, indicators: Dict, row: int = 3):
        """Add technical indicator overlays."""
        # Moving averages
        if 'sma_20' in indicators:
            sma_trace = go.Scatter(
                x=df.index,
                y=indicators['sma_20'],
                name="SMA 20",
                line=dict(color='#00aaff', width=1),
                opacity=0.8
            )
            fig.add_trace(sma_trace, row=1, col=1)  # Add to main chart
        
        if 'sma_50' in indicators:
            sma50_trace = go.Scatter(
                x=df.index,
                y=indicators['sma_50'],
                name="SMA 50",
                line=dict(color='#ffaa00', width=1),
                opacity=0.8
            )
            fig.add_trace(sma50_trace, row=1, col=1)
        
        # Bollinger Bands
        if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_upper = go.Scatter(
                x=df.index,
                y=indicators['bb_upper'],
                name="BB Upper",
                line=dict(color='#888888', width=1, dash='dash'),
                opacity=0.6
            )
            fig.add_trace(bb_upper, row=1, col=1)
            
            bb_lower = go.Scatter(
                x=df.index,
                y=indicators['bb_lower'],
                name="BB Lower",
                line=dict(color='#888888', width=1, dash='dash'),
                opacity=0.6,
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            )
            fig.add_trace(bb_lower, row=1, col=1)
        
        # RSI
        if 'rsi' in indicators:
            rsi_trace = go.Scatter(
                x=df.index,
                y=indicators['rsi'],
                name="RSI",
                line=dict(color='#ff00ff', width=2),
                yaxis='y3'
            )
            fig.add_trace(rsi_trace, row=3, col=1)
            
            # Add RSI overbought/oversold lines
            overbought_line = go.Scatter(
                x=df.index,
                y=[70] * len(df),
                name="RSI Overbought",
                line=dict(color='#ff4444', width=1, dash='dash'),
                yaxis='y3'
            )
            fig.add_trace(overbought_line, row=3, col=1)
            
            oversold_line = go.Scatter(
                x=df.index,
                y=[30] * len(df),
                name="RSI Oversold",
                line=dict(color='#00ff88', width=1, dash='dash'),
                yaxis='y3'
            )
            fig.add_trace(oversold_line, row=3, col=1)
        
        # MACD
        if all(key in indicators for key in ['macd_line', 'macd_signal', 'macd_histogram']):
            macd_line = go.Scatter(
                x=df.index,
                y=indicators['macd_line'],
                name="MACD",
                line=dict(color='#00aaff', width=2),
                yaxis='y4'
            )
            fig.add_trace(macd_line, row=3, col=1)
            
            macd_signal = go.Scatter(
                x=df.index,
                y=indicators['macd_signal'],
                name="MACD Signal",
                line=dict(color='#ffaa00', width=2),
                yaxis='y4'
            )
            fig.add_trace(macd_signal, row=3, col=1)
            
            # MACD Histogram
            colors = ['#00ff88' if val >= 0 else '#ff4444' for val in indicators['macd_histogram']]
            macd_hist = go.Bar(
                x=df.index,
                y=indicators['macd_histogram'],
                name="MACD Histogram",
                marker_color=colors,
                opacity=0.7,
                yaxis='y4'
            )
            fig.add_trace(macd_hist, row=3, col=1)
    
    def _add_pattern_markers(self, fig: go.Figure, df: pd.DataFrame, patterns: Dict, row: int = 1):
        """Add pattern detection markers to the chart."""
        if not patterns:
            return
        
        # Add markers for discovered patterns
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'sequences' in pattern_data:
                sequences = pattern_data['sequences']
                
                for seq in sequences[:5]:  # Show first 5 patterns
                    if len(seq) > 0:
                        # Mark the end of the pattern
                        pattern_end = len(df) - len(seq)
                        if pattern_end >= 0:
                            marker = go.Scatter(
                                x=[df.index[pattern_end]],
                                y=[df['High'].iloc[pattern_end]],
                                mode='markers',
                                marker=dict(
                                    symbol='diamond',
                                    size=12,
                                    color=self.colors['pattern'],
                                    line=dict(width=2, color='white')
                                ),
                                name=f"{pattern_type} Pattern",
                                showlegend=False
                            )
                            fig.add_trace(marker, row=row, col=1)
    
    def _add_support_resistance(self, fig: go.Figure, df: pd.DataFrame, row: int = 1):
        """Add support and resistance level overlays."""
        # Calculate support and resistance levels
        high_levels = self._find_support_resistance_levels(df['High'])
        low_levels = self._find_support_resistance_levels(df['Low'])
        
        # Add resistance levels
        for level in high_levels:
            resistance_line = go.Scatter(
                x=df.index,
                y=[level] * len(df),
                name=f"Resistance {level:.2f}",
                line=dict(color=self.colors['resistance'], width=2, dash='dash'),
                opacity=0.7
            )
            fig.add_trace(resistance_line, row=row, col=1)
        
        # Add support levels
        for level in low_levels:
            support_line = go.Scatter(
                x=df.index,
                y=[level] * len(df),
                name=f"Support {level:.2f}",
                line=dict(color=self.colors['support'], width=2, dash='dash'),
                opacity=0.7
            )
            fig.add_trace(support_line, row=row, col=1)
    
    def _find_support_resistance_levels(self, series: pd.Series, window: int = 20) -> List[float]:
        """Find support and resistance levels using local extrema."""
        levels = []
        
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                levels.append(series.iloc[i])
            elif series.iloc[i] == series.iloc[i-window:i+window+1].min():
                levels.append(series.iloc[i])
        
        # Group nearby levels
        if levels:
            levels = sorted(levels)
            grouped_levels = []
            current_group = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] < 0.02:  # 2% threshold
                    current_group.append(level)
                else:
                    grouped_levels.append(np.mean(current_group))
                    current_group = [level]
            
            grouped_levels.append(np.mean(current_group))
            return grouped_levels[-3:]  # Return top 3 levels
        
        return []
    
    def _update_chart_layout(self, fig: go.Figure, title: str):
        """Update chart layout with professional styling."""
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20, color='white')
            ),
            template=self.chart_theme,
            height=self.chart_height,
            width=self.chart_width,
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            xaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                title="Date"
            ),
            yaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                title="Price",
                side="right"
            ),
            yaxis2=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                title="Volume",
                side="right"
            ),
            yaxis3=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                title="RSI",
                range=[0, 100],
                side="right"
            ),
            yaxis4=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                title="MACD",
                side="right"
            ),
            legend=dict(
                bgcolor=self.colors['background'],
                bordercolor=self.colors['grid'],
                borderwidth=1
            ),
            hovermode='x unified',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            zeroline=False
        )
        
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            zeroline=False
        )
    
    def create_multi_timeframe_chart(self, data_dict: Dict[str, pd.DataFrame], 
                                   indicators: Dict = None, patterns: Dict = None) -> go.Figure:
        """
        Create multi-timeframe candlestick chart.
        
        Args:
            data_dict: Dictionary with timeframe as key and DataFrame as value
            indicators: Technical indicator values
            patterns: Pattern analysis results
        
        Returns:
            Plotly figure object
        """
        if not data_dict:
            logger.warning("No data provided for multi-timeframe chart")
            return go.Figure()
        
        logger.info("Creating multi-timeframe candlestick chart...")
        
        # Create subplots for each timeframe
        timeframes = list(data_dict.keys())
        n_timeframes = len(timeframes)
        
        fig = make_subplots(
            rows=n_timeframes, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f"{tf.upper()} Timeframe" for tf in timeframes]
        )
        
        # Add candlestick charts for each timeframe
        for i, (timeframe, df) in enumerate(data_dict.items()):
            row = i + 1
            
            # Add candlestick trace
            self._add_candlestick_trace(fig, df, row=row)
            
            # Add indicators for daily timeframe
            if timeframe == '1d' and indicators:
                self._add_indicator_overlays(fig, df, indicators, row=row)
            
            # Add patterns
            if patterns and timeframe in patterns:
                self._add_pattern_markers(fig, df, patterns[timeframe], row=row)
        
        # Update layout
        self._update_chart_layout(fig, "Multi-Timeframe Analysis")
        
        logger.info("Multi-timeframe chart created successfully")
        return fig
    
    def create_pattern_analysis_chart(self, patterns: Dict, df: pd.DataFrame) -> go.Figure:
        """
        Create specialized chart for pattern analysis.
        
        Args:
            patterns: Pattern analysis results
            df: Original price data
        
        Returns:
            Plotly figure object
        """
        if not patterns:
            logger.warning("No patterns provided for pattern analysis chart")
            return go.Figure()
        
        logger.info("Creating pattern analysis chart...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Price with Pattern Markers", "Pattern Performance")
        )
        
        # Main price chart with pattern markers
        self._add_candlestick_trace(fig, df, row=1)
        
        # Add pattern markers with performance information
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'performance' in pattern_data:
                perf = pattern_data['performance']
                
                # Color code based on performance
                if perf.get('avg_return', 0) > 0:
                    color = self.colors['bullish']
                else:
                    color = self.colors['bearish']
                
                # Add performance annotation
                if 'sequences' in pattern_data and pattern_data['sequences']:
                    seq = pattern_data['sequences'][0]
                    if len(seq) > 0:
                        pattern_end = len(df) - len(seq)
                        if pattern_end >= 0:
                            fig.add_annotation(
                                x=df.index[pattern_end],
                                y=df['High'].iloc[pattern_end],
                                text=f"{pattern_type}<br>Return: {perf.get('avg_return', 0):.2%}<br>Win Rate: {perf.get('win_rate', 0):.1%}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor=color,
                                bgcolor=self.colors['background'],
                                bordercolor=color,
                                borderwidth=1,
                                font=dict(color='white', size=10)
                            )
        
        # Pattern performance summary
        pattern_names = []
        pattern_returns = []
        pattern_win_rates = []
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'performance' in pattern_data:
                perf = pattern_data['performance']
                pattern_names.append(pattern_type)
                pattern_returns.append(perf.get('avg_return', 0))
                pattern_win_rates.append(perf.get('win_rate', 0))
        
        if pattern_names:
            # Performance bar chart
            colors = [self.colors['bullish'] if ret > 0 else self.colors['bearish'] for ret in pattern_returns]
            
            perf_trace = go.Bar(
                x=pattern_names,
                y=pattern_returns,
                name="Average Return",
                marker_color=colors,
                opacity=0.7
            )
            fig.add_trace(perf_trace, row=2, col=1)
        
        # Update layout
        self._update_chart_layout(fig, "Pattern Analysis")
        
        logger.info("Pattern analysis chart created successfully")
        return fig
    
    def export_chart(self, fig: go.Figure, filepath: str, format: str = 'html'):
        """
        Export chart to file.
        
        Args:
            fig: Plotly figure object
            filepath: Output file path
            format: Export format ('html', 'png', 'pdf')
        """
        try:
            if format == 'html':
                fig.write_html(filepath)
            elif format == 'png':
                fig.write_image(filepath)
            elif format == 'pdf':
                fig.write_image(filepath)
            else:
                logger.warning(f"Unsupported format: {format}")
                return
            
            logger.info(f"Chart exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
    
    def create_dashboard(self, df: pd.DataFrame, indicators: Dict, patterns: Dict, 
                        predictions: Dict) -> go.Figure:
        """
        Create comprehensive trading dashboard.
        
        Args:
            df: Price data
            indicators: Technical indicators
            patterns: Pattern analysis
            predictions: ML predictions
        
        Returns:
            Dashboard figure
        """
        logger.info("Creating comprehensive trading dashboard...")
        
        # Create subplots for dashboard
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.05,
            subplot_titles=(
                "Price Chart", "Volume Profile",
                "Technical Indicators", "Pattern Analysis",
                "Prediction Confidence", "Risk Assessment"
            ),
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"secondary_y": True}, {"type": "scatter"}],
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}]
            ]
        )
        
        # Main price chart
        self._add_candlestick_trace(fig, df, row=1, col=1)
        if indicators:
            self._add_indicator_overlays(fig, df, indicators, row=1, col=1)
        
        # Volume profile
        if 'Volume' in df.columns:
            volume_profile = df.groupby(df.index.hour)['Volume'].mean()
            vol_trace = go.Bar(
                x=volume_profile.index,
                y=volume_profile.values,
                name="Volume Profile",
                marker_color=self.colors['volume_up']
            )
            fig.add_trace(vol_trace, row=1, col=2)
        
        # Technical indicators summary
        if indicators:
            # RSI gauge
            current_rsi = indicators.get('rsi', pd.Series([50])).iloc[-1]
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_rsi,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "RSI"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 30], 'color': "lightgray"},
                                    {'range': [70, 100], 'color': "lightgray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}
                ),
                row=3, col=1
            )
        
        # Prediction confidence
        if predictions:
            confidence = predictions.get('confidence_percentage', 50)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Prediction Confidence"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "green" if confidence > 70 else "orange" if confidence > 40 else "red"},
                           'steps': [{'range': [0, 40], 'color': "lightgray"},
                                    {'range': [40, 70], 'color': "lightgray"},
                                    {'range': [70, 100], 'color': "lightgray"}]}
                ),
                row=3, col=2
            )
        
        # Risk assessment
        if predictions and 'risk_assessment' in predictions:
            risk = predictions['risk_assessment']
            risk_reward = risk.get('risk_reward_ratio', 1)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_reward,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk/Reward Ratio"},
                    gauge={'axis': {'range': [0, 3]},
                           'bar': {'color': "green" if risk_reward > 2 else "orange" if risk_reward > 1 else "red"},
                           'steps': [{'range': [0, 1], 'color': "lightgray"},
                                    {'range': [1, 2], 'color': "lightgray"},
                                    {'range': [2, 3], 'color': "lightgray"}]}
                ),
                row=4, col=1
            )
        
        # Update layout
        self._update_chart_layout(fig, "Ultimate Market AI Engine Dashboard")
        
        logger.info("Trading dashboard created successfully")
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Test the candlestick visualizer
    from data.live_data_loader import DataLoader
    from technical_analysis.advanced_indicators import TechnicalAnalyzer
    
    print("Testing Candlestick Visualizer...")
    
    # Load sample data
    loader = DataLoader()
    data = loader.fetch_single_timeframe_data(timeframe="1d")
    
    if data is not None and not data.empty:
        # Calculate technical indicators
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_all_indicators(data)
        
        # Initialize visualizer
        visualizer = CandlestickVisualizer()
        
        # Create candlestick chart
        fig = visualizer.create_candlestick_chart(data, indicators)
        
        # Export chart
        visualizer.export_chart(fig, "charts/candlestick_chart.html", "html")
        
        print("Candlestick chart created and exported successfully")
        
    else:
        print("No data available for testing")
