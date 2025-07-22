"""
Ultimate Market AI Engine - Report Generation System
==================================================

This module generates comprehensive technical and price action analysis reports
in multiple formats (Excel, PDF, HTML, JSON).
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings

# Import configuration
import sys
sys.path.append('..')
from config import get_config

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive report generation system for market analysis.
    
    Features:
    - Technical Analysis Report: All 40+ indicators and patterns
    - Price Action Analysis Report: Candlestick patterns and chart analysis
    - Multiple output formats: Excel, PDF, HTML, JSON
    - Professional formatting and branding
    - Automated chart embedding
    - Performance optimization
    """
    
    def __init__(self, config=None):
        """Initialize the report generator."""
        self.config = config or get_config()
        
        # Report templates
        self.templates = {
            'technical': self._get_technical_template(),
            'price_action': self._get_price_action_template(),
            'summary': self._get_summary_template()
        }
        
        # Styling
        self.styles = {
            'header': Font(bold=True, size=14, color="FFFFFF"),
            'subheader': Font(bold=True, size=12, color="FFFFFF"),
            'normal': Font(size=10),
            'highlight': Font(bold=True, color="00FF00"),
            'warning': Font(bold=True, color="FF0000"),
            'header_fill': PatternFill(start_color="366092", end_color="366092", fill_type="solid"),
            'subheader_fill': PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid"),
            'highlight_fill': PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"),
            'warning_fill': PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        }
        
        logger.info("Report Generator initialized")
    
    def _get_technical_template(self) -> str:
        """Get technical analysis report template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Technical Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #366092; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .indicator { margin: 10px 0; padding: 10px; background-color: #f9f9f9; }
                .signal { font-weight: bold; }
                .bullish { color: green; }
                .bearish { color: red; }
                .neutral { color: orange; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4472C4; color: white; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Technical Analysis Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            {content}
        </body>
        </html>
        """
    
    def _get_price_action_template(self) -> str:
        """Get price action analysis report template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Price Action Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #70AD47; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .pattern { margin: 10px 0; padding: 10px; background-color: #f9f9f9; }
                .performance { font-weight: bold; }
                .positive { color: green; }
                .negative { color: red; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #70AD47; color: white; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Price Action Analysis Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            {content}
        </body>
        </html>
        """
    
    def _get_summary_template(self) -> str:
        """Get summary report template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Analysis Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #C5504B; color: white; padding: 20px; text-align: center; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .prediction { font-size: 18px; font-weight: bold; margin: 10px 0; }
                .bullish { color: green; }
                .bearish { color: red; }
                .neutral { color: orange; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #C5504B; color: white; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Market Analysis Summary</h1>
                <p>Generated on {timestamp}</p>
            </div>
            {content}
        </body>
        </html>
        """
    
    def generate_technical_analysis_report(self, indicators: Dict, patterns: Dict, 
                                         predictions: Dict, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate comprehensive technical analysis report.
        
        Args:
            indicators: Technical indicator values
            patterns: Pattern analysis results
            predictions: ML predictions
            df: Price data
        
        Returns:
            Dictionary with report file paths
        """
        logger.info("Generating technical analysis report...")
        
        # Create Excel workbook
        wb = Workbook()
        
        # Remove default sheet
        wb.remove(wb.active)
        
        # Create sheets
        self._create_technical_summary_sheet(wb, indicators, predictions)
        self._create_indicator_values_sheet(wb, indicators)
        self._create_pattern_analysis_sheet(wb, patterns)
        self._create_confluence_analysis_sheet(wb, indicators)
        self._create_risk_assessment_sheet(wb, predictions, df)
        
        # Save Excel file
        excel_path = self.config.get_report_path("Technical_Analysis_Report.xlsx")
        wb.save(excel_path)
        
        # Generate HTML report
        html_content = self._generate_technical_html(indicators, patterns, predictions, df)
        html_path = self.config.get_report_path("Technical_Analysis_Report.html")
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Generate JSON export
        json_data = self._generate_technical_json(indicators, patterns, predictions, df)
        json_path = self.config.get_report_path("Technical_Analysis_Report.json")
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info("Technical analysis report generated successfully")
        
        return {
            'excel': excel_path,
            'html': html_path,
            'json': json_path
        }
    
    def _create_technical_summary_sheet(self, wb: Workbook, indicators: Dict, predictions: Dict):
        """Create technical analysis summary sheet."""
        ws = wb.create_sheet("Technical Summary")
        
        # Title
        ws['A1'] = "TECHNICAL ANALYSIS SUMMARY"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:E1')
        
        # Current market status
        ws['A3'] = "Current Market Status"
        ws['A3'].font = self.styles['subheader']
        ws['A3'].fill = self.styles['subheader_fill']
        
        # Key indicators
        row = 4
        key_indicators = ['rsi', 'macd_line', 'sma_20', 'bb_position', 'atr']
        
        for indicator in key_indicators:
            if indicator in indicators:
                if isinstance(indicators[indicator], pd.Series):
                    value = indicators[indicator].iloc[-1]
                else:
                    value = indicators[indicator]
                
                ws[f'A{row}'] = indicator.upper()
                ws[f'B{row}'] = f"{value:.4f}"
                
                # Color coding
                if indicator == 'rsi':
                    if value > 70:
                        ws[f'B{row}'].font = self.styles['warning']
                        ws[f'B{row}'].fill = self.styles['warning_fill']
                    elif value < 30:
                        ws[f'B{row}'].font = self.styles['highlight']
                        ws[f'B{row}'].fill = self.styles['highlight_fill']
                
                row += 1
        
        # Prediction summary
        row += 2
        ws[f'A{row}'] = "ML Prediction Summary"
        ws[f'A{row}'].font = self.styles['subheader']
        ws[f'A{row}'].fill = self.styles['subheader_fill']
        
        if predictions:
            row += 1
            ws[f'A{row}'] = "Direction"
            ws[f'B{row}'] = predictions.get('movement_direction', 'N/A')
            
            row += 1
            ws[f'A{row}'] = "Confidence"
            ws[f'B{row}'] = f"{predictions.get('confidence_percentage', 0):.1f}%"
            
            row += 1
            ws[f'A{row}'] = "Target Price"
            ws[f'B{row}'] = f"{predictions.get('target_price', 0):.2f}"
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_indicator_values_sheet(self, wb: Workbook, indicators: Dict):
        """Create sheet with all indicator values."""
        ws = wb.create_sheet("Indicator Values")
        
        # Title
        ws['A1'] = "TECHNICAL INDICATOR VALUES"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:C1')
        
        # Create DataFrame for indicators
        indicator_data = {}
        for name, data in indicators.items():
            if isinstance(data, pd.Series):
                indicator_data[name] = data.iloc[-1] if not data.empty else 0
            elif isinstance(data, dict):
                for sub_name, sub_data in data.items():
                    if isinstance(sub_data, pd.Series):
                        indicator_data[f"{name}_{sub_name}"] = sub_data.iloc[-1] if not sub_data.empty else 0
            else:
                indicator_data[name] = data
        
        # Convert to DataFrame and write to sheet
        if indicator_data:
            df = pd.DataFrame(list(indicator_data.items()), columns=['Indicator', 'Value'])
            
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
            
            # Style the header
            for cell in ws[2]:
                cell.font = self.styles['subheader']
                cell.fill = self.styles['subheader_fill']
    
    def _create_pattern_analysis_sheet(self, wb: Workbook, patterns: Dict):
        """Create sheet with pattern analysis."""
        ws = wb.create_sheet("Pattern Analysis")
        
        # Title
        ws['A1'] = "PATTERN ANALYSIS"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:F1')
        
        # Headers
        headers = ['Pattern Type', 'Algorithm', 'Count', 'Avg Return', 'Win Rate', 'Confidence']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = self.styles['subheader']
            cell.fill = self.styles['subheader_fill']
        
        # Pattern data
        row = 4
        for timeframe, timeframe_data in patterns.items():
            for algorithm, algorithm_data in timeframe_data.items():
                significant_patterns = algorithm_data.get('significant_patterns', {})
                
                for pattern_id, pattern_info in significant_patterns.items():
                    ws.cell(row=row, column=1, value=f"Pattern_{pattern_id}")
                    ws.cell(row=row, column=2, value=algorithm)
                    ws.cell(row=row, column=3, value=pattern_info.get('occurrences', 0))
                    ws.cell(row=row, column=4, value=f"{pattern_info.get('avg_return', 0):.4f}")
                    ws.cell(row=row, column=5, value=f"{pattern_info.get('win_rate', 0):.2%}")
                    ws.cell(row=row, column=6, value=f"{pattern_info.get('confidence', 0):.2%}")
                    
                    # Color coding for returns
                    if pattern_info.get('avg_return', 0) > 0:
                        ws.cell(row=row, column=4).font = self.styles['highlight']
                    else:
                        ws.cell(row=row, column=4).font = self.styles['warning']
                    
                    row += 1
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_confluence_analysis_sheet(self, wb: Workbook, indicators: Dict):
        """Create sheet with indicator confluence analysis."""
        ws = wb.create_sheet("Confluence Analysis")
        
        # Title
        ws['A1'] = "INDICATOR CONFLUENCE ANALYSIS"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:C1')
        
        # This would be populated with confluence analysis from TechnicalAnalyzer
        ws['A3'] = "Confluence analysis will be populated from technical analyzer results"
    
    def _create_risk_assessment_sheet(self, wb: Workbook, predictions: Dict, df: pd.DataFrame):
        """Create sheet with risk assessment."""
        ws = wb.create_sheet("Risk Assessment")
        
        # Title
        ws['A1'] = "RISK ASSESSMENT"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:C1')
        
        if predictions and 'risk_assessment' in predictions:
            risk = predictions['risk_assessment']
            
            row = 3
            ws.cell(row=row, column=1, value="Stop Loss")
            ws.cell(row=row, column=2, value=f"{risk.get('stop_loss', 0):.2f}")
            row += 1
            
            ws.cell(row=row, column=1, value="Risk/Reward Ratio")
            ws.cell(row=row, column=2, value=f"{risk.get('risk_reward_ratio', 0):.2f}")
            row += 1
            
            ws.cell(row=row, column=1, value="Maximum Loss")
            ws.cell(row=row, column=2, value=f"{risk.get('max_loss', 0):.2f}")
    
    def _generate_technical_html(self, indicators: Dict, patterns: Dict, 
                               predictions: Dict, df: pd.DataFrame) -> str:
        """Generate HTML content for technical analysis report."""
        content = ""
        
        # Market Overview
        content += "<div class='section'>"
        content += "<h2>Market Overview</h2>"
        content += f"<p><strong>Symbol:</strong> {self.config.SYMBOL}</p>"
        content += f"<p><strong>Current Price:</strong> {df['Close'].iloc[-1]:.2f}</p>"
        content += f"<p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        content += "</div>"
        
        # Key Indicators
        content += "<div class='section'>"
        content += "<h2>Key Technical Indicators</h2>"
        content += "<table>"
        content += "<tr><th>Indicator</th><th>Value</th><th>Signal</th></tr>"
        
        key_indicators = ['rsi', 'macd_line', 'sma_20', 'bb_position', 'atr']
        for indicator in key_indicators:
            if indicator in indicators:
                if isinstance(indicators[indicator], pd.Series):
                    value = indicators[indicator].iloc[-1]
                else:
                    value = indicators[indicator]
                
                signal = self._get_indicator_signal(indicator, value)
                signal_class = self._get_signal_class(signal)
                
                content += f"<tr>"
                content += f"<td>{indicator.upper()}</td>"
                content += f"<td>{value:.4f}</td>"
                content += f"<td class='signal {signal_class}'>{signal}</td>"
                content += f"</tr>"
        
        content += "</table>"
        content += "</div>"
        
        # ML Predictions
        if predictions:
            content += "<div class='section'>"
            content += "<h2>Machine Learning Predictions</h2>"
            content += f"<p><strong>Direction:</strong> {predictions.get('movement_direction', 'N/A')}</p>"
            content += f"<p><strong>Confidence:</strong> {predictions.get('confidence_percentage', 0):.1f}%</p>"
            content += f"<p><strong>Target Price:</strong> {predictions.get('target_price', 0):.2f}</p>"
            content += f"<p><strong>Time Horizon:</strong> {predictions.get('time_horizon', 'N/A')}</p>"
            content += "</div>"
        
        # Pattern Analysis
        if patterns:
            content += "<div class='section'>"
            content += "<h2>Pattern Analysis</h2>"
            content += "<table>"
            content += "<tr><th>Pattern Type</th><th>Algorithm</th><th>Performance</th><th>Confidence</th></tr>"
            
            for timeframe, timeframe_data in patterns.items():
                for algorithm, algorithm_data in timeframe_data.items():
                    significant_patterns = algorithm_data.get('significant_patterns', {})
                    
                    for pattern_id, pattern_info in significant_patterns.items():
                        avg_return = pattern_info.get('avg_return', 0)
                        win_rate = pattern_info.get('win_rate', 0)
                        confidence = pattern_info.get('confidence', 0)
                        
                        performance_class = 'positive' if avg_return > 0 else 'negative'
                        
                        content += f"<tr>"
                        content += f"<td>Pattern_{pattern_id}</td>"
                        content += f"<td>{algorithm}</td>"
                        content += f"<td class='performance {performance_class}'>{avg_return:.2%} ({win_rate:.1%} win rate)</td>"
                        content += f"<td>{confidence:.1%}</td>"
                        content += f"</tr>"
            
            content += "</table>"
            content += "</div>"
        
        return self.templates['technical'].format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            content=content
        )
    
    def _generate_technical_json(self, indicators: Dict, patterns: Dict, 
                               predictions: Dict, df: pd.DataFrame) -> Dict:
        """Generate JSON data for technical analysis report."""
        return {
            'report_type': 'technical_analysis',
            'symbol': self.config.SYMBOL,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(df['Close'].iloc[-1]),
            'indicators': {
                name: float(value.iloc[-1]) if isinstance(value, pd.Series) else float(value)
                for name, value in indicators.items()
            },
            'patterns': patterns,
            'predictions': predictions,
            'summary': {
                'total_indicators': len(indicators),
                'significant_patterns': sum(
                    len(alg_data.get('significant_patterns', {}))
                    for timeframe_data in patterns.values()
                    for alg_data in timeframe_data.values()
                ),
                'prediction_confidence': predictions.get('confidence_percentage', 0)
            }
        }
    
    def generate_price_action_report(self, patterns: Dict, df: pd.DataFrame, 
                                   technical_summary: Dict) -> Dict[str, str]:
        """
        Generate price action analysis report.
        
        Args:
            patterns: Pattern analysis results
            df: Price data
            technical_summary: Technical analysis summary
        
        Returns:
            Dictionary with report file paths
        """
        logger.info("Generating price action analysis report...")
        
        # Create Excel workbook
        wb = Workbook()
        wb.remove(wb.active)
        
        # Create sheets
        self._create_price_action_summary_sheet(wb, patterns, df)
        self._create_pattern_details_sheet(wb, patterns)
        self._create_chart_analysis_sheet(wb, df)
        self._create_volume_analysis_sheet(wb, df)
        
        # Save Excel file
        excel_path = self.config.get_report_path("Price_Action_Report.xlsx")
        wb.save(excel_path)
        
        # Generate HTML report
        html_content = self._generate_price_action_html(patterns, df, technical_summary)
        html_path = self.config.get_report_path("Price_Action_Report.html")
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Generate JSON export
        json_data = self._generate_price_action_json(patterns, df, technical_summary)
        json_path = self.config.get_report_path("Price_Action_Report.json")
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info("Price action analysis report generated successfully")
        
        return {
            'excel': excel_path,
            'html': html_path,
            'json': json_path
        }
    
    def _create_price_action_summary_sheet(self, wb: Workbook, patterns: Dict, df: pd.DataFrame):
        """Create price action summary sheet."""
        ws = wb.create_sheet("Price Action Summary")
        
        # Title
        ws['A1'] = "PRICE ACTION ANALYSIS SUMMARY"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:E1')
        
        # Price action statistics
        ws['A3'] = "Price Action Statistics"
        ws['A3'].font = self.styles['subheader']
        ws['A3'].fill = self.styles['subheader_fill']
        
        row = 4
        stats = [
            ("Current Price", f"{df['Close'].iloc[-1]:.2f}"),
            ("Price Change", f"{df['Close'].pct_change().iloc[-1]:.2%}"),
            ("High", f"{df['High'].iloc[-1]:.2f}"),
            ("Low", f"{df['Low'].iloc[-1]:.2f}"),
            ("Range", f"{df['High'].iloc[-1] - df['Low'].iloc[-1]:.2f}"),
            ("Volume", f"{df['Volume'].iloc[-1]:,.0f}" if 'Volume' in df.columns else "N/A")
        ]
        
        for stat_name, stat_value in stats:
            ws[f'A{row}'] = stat_name
            ws[f'B{row}'] = stat_value
            row += 1
        
        # Pattern summary
        row += 2
        ws[f'A{row}'] = "Pattern Summary"
        ws[f'A{row}'].font = self.styles['subheader']
        ws[f'A{row}'].fill = self.styles['subheader_fill']
        
        total_patterns = sum(
            len(alg_data.get('significant_patterns', {}))
            for timeframe_data in patterns.values()
            for alg_data in timeframe_data.values()
        )
        
        row += 1
        ws[f'A{row}'] = "Total Significant Patterns"
        ws[f'B{row}'] = total_patterns
    
    def _create_pattern_details_sheet(self, wb: Workbook, patterns: Dict):
        """Create sheet with detailed pattern information."""
        ws = wb.create_sheet("Pattern Details")
        
        # Title
        ws['A1'] = "PATTERN DETAILS"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:G1')
        
        # Headers
        headers = ['Timeframe', 'Algorithm', 'Pattern ID', 'Count', 'Avg Return', 'Win Rate', 'Confidence']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = self.styles['subheader']
            cell.fill = self.styles['subheader_fill']
        
        # Pattern data
        row = 4
        for timeframe, timeframe_data in patterns.items():
            for algorithm, algorithm_data in timeframe_data.items():
                significant_patterns = algorithm_data.get('significant_patterns', {})
                
                for pattern_id, pattern_info in significant_patterns.items():
                    ws.cell(row=row, column=1, value=timeframe)
                    ws.cell(row=row, column=2, value=algorithm)
                    ws.cell(row=row, column=3, value=f"Pattern_{pattern_id}")
                    ws.cell(row=row, column=4, value=pattern_info.get('occurrences', 0))
                    ws.cell(row=row, column=5, value=f"{pattern_info.get('avg_return', 0):.4f}")
                    ws.cell(row=row, column=6, value=f"{pattern_info.get('win_rate', 0):.2%}")
                    ws.cell(row=row, column=7, value=f"{pattern_info.get('confidence', 0):.2%}")
                    
                    row += 1
    
    def _create_chart_analysis_sheet(self, wb: Workbook, df: pd.DataFrame):
        """Create sheet with chart analysis."""
        ws = wb.create_sheet("Chart Analysis")
        
        # Title
        ws['A1'] = "CHART ANALYSIS"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:C1')
        
        # Support and resistance levels
        ws['A3'] = "Support and Resistance Levels"
        ws['A3'].font = self.styles['subheader']
        ws['A3'].fill = self.styles['subheader_fill']
        
        # Calculate support and resistance
        high_levels = self._find_support_resistance_levels(df['High'])
        low_levels = self._find_support_resistance_levels(df['Low'])
        
        row = 4
        ws.cell(row=row, column=1, value="Resistance Levels")
        for i, level in enumerate(high_levels, 1):
            ws.cell(row=row, column=2, value=f"R{i}")
            ws.cell(row=row, column=3, value=f"{level:.2f}")
            row += 1
        
        row += 1
        ws.cell(row=row, column=1, value="Support Levels")
        for i, level in enumerate(low_levels, 1):
            ws.cell(row=row, column=2, value=f"S{i}")
            ws.cell(row=row, column=3, value=f"{level:.2f}")
            row += 1
    
    def _create_volume_analysis_sheet(self, wb: Workbook, df: pd.DataFrame):
        """Create sheet with volume analysis."""
        ws = wb.create_sheet("Volume Analysis")
        
        # Title
        ws['A1'] = "VOLUME ANALYSIS"
        ws['A1'].font = self.styles['header']
        ws['A1'].fill = self.styles['header_fill']
        ws.merge_cells('A1:C1')
        
        if 'Volume' in df.columns:
            # Volume statistics
            ws['A3'] = "Volume Statistics"
            ws['A3'].font = self.styles['subheader']
            ws['A3'].fill = self.styles['subheader_fill']
            
            row = 4
            stats = [
                ("Current Volume", f"{df['Volume'].iloc[-1]:,.0f}"),
                ("Average Volume", f"{df['Volume'].mean():,.0f}"),
                ("Volume Change", f"{df['Volume'].pct_change().iloc[-1]:.2%}"),
                ("Volume Trend", "Increasing" if df['Volume'].iloc[-1] > df['Volume'].iloc[-2] else "Decreasing")
            ]
            
            for stat_name, stat_value in stats:
                ws[f'A{row}'] = stat_name
                ws[f'B{row}'] = stat_value
                row += 1
    
    def _find_support_resistance_levels(self, series: pd.Series, window: int = 20) -> List[float]:
        """Find support and resistance levels."""
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
                if abs(level - current_group[-1]) / current_group[-1] < 0.02:
                    current_group.append(level)
                else:
                    grouped_levels.append(np.mean(current_group))
                    current_group = [level]
            
            grouped_levels.append(np.mean(current_group))
            return grouped_levels[-3:]  # Return top 3 levels
        
        return []
    
    def _generate_price_action_html(self, patterns: Dict, df: pd.DataFrame, 
                                  technical_summary: Dict) -> str:
        """Generate HTML content for price action report."""
        content = ""
        
        # Price Action Overview
        content += "<div class='section'>"
        content += "<h2>Price Action Overview</h2>"
        content += f"<p><strong>Current Price:</strong> {df['Close'].iloc[-1]:.2f}</p>"
        content += f"<p><strong>Price Change:</strong> {df['Close'].pct_change().iloc[-1]:.2%}</p>"
        content += f"<p><strong>High:</strong> {df['High'].iloc[-1]:.2f}</p>"
        content += f"<p><strong>Low:</strong> {df['Low'].iloc[-1]:.2f}</p>"
        content += f"<p><strong>Range:</strong> {df['High'].iloc[-1] - df['Low'].iloc[-1]:.2f}</p>"
        if 'Volume' in df.columns:
            content += f"<p><strong>Volume:</strong> {df['Volume'].iloc[-1]:,.0f}</p>"
        content += "</div>"
        
        # Pattern Analysis
        if patterns:
            content += "<div class='section'>"
            content += "<h2>Pattern Analysis</h2>"
            content += "<table>"
            content += "<tr><th>Timeframe</th><th>Algorithm</th><th>Patterns Found</th><th>Best Performance</th></tr>"
            
            for timeframe, timeframe_data in patterns.items():
                for algorithm, algorithm_data in timeframe_data.items():
                    significant_patterns = algorithm_data.get('significant_patterns', {})
                    
                    if significant_patterns:
                        best_pattern = max(significant_patterns.values(), 
                                         key=lambda x: x.get('avg_return', 0))
                        
                        content += f"<tr>"
                        content += f"<td>{timeframe}</td>"
                        content += f"<td>{algorithm}</td>"
                        content += f"<td>{len(significant_patterns)}</td>"
                        content += f"<td class='performance {'positive' if best_pattern.get('avg_return', 0) > 0 else 'negative'}'>{best_pattern.get('avg_return', 0):.2%}</td>"
                        content += f"</tr>"
            
            content += "</table>"
            content += "</div>"
        
        return self.templates['price_action'].format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            content=content
        )
    
    def _generate_price_action_json(self, patterns: Dict, df: pd.DataFrame, 
                                  technical_summary: Dict) -> Dict:
        """Generate JSON data for price action report."""
        return {
            'report_type': 'price_action_analysis',
            'symbol': self.config.SYMBOL,
            'timestamp': datetime.now().isoformat(),
            'price_data': {
                'current_price': float(df['Close'].iloc[-1]),
                'price_change': float(df['Close'].pct_change().iloc[-1]),
                'high': float(df['High'].iloc[-1]),
                'low': float(df['Low'].iloc[-1]),
                'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0
            },
            'patterns': patterns,
            'summary': {
                'total_patterns': sum(
                    len(alg_data.get('significant_patterns', {}))
                    for timeframe_data in patterns.values()
                    for alg_data in timeframe_data.values()
                ),
                'timeframes_analyzed': list(patterns.keys())
            }
        }
    
    def _get_indicator_signal(self, indicator: str, value: float) -> str:
        """Get signal for an indicator."""
        if indicator == 'rsi':
            if value > 70:
                return "Overbought"
            elif value < 30:
                return "Oversold"
            else:
                return "Neutral"
        elif indicator == 'macd_line':
            return "Bullish" if value > 0 else "Bearish"
        elif indicator == 'bb_position':
            if value > 0.8:
                return "Overbought"
            elif value < 0.2:
                return "Oversold"
            else:
                return "Neutral"
        else:
            return "Neutral"
    
    def _get_signal_class(self, signal: str) -> str:
        """Get CSS class for signal."""
        if signal in ["Bullish", "Oversold", "Positive"]:
            return "bullish"
        elif signal in ["Bearish", "Overbought", "Negative"]:
            return "bearish"
        else:
            return "neutral"


# Example usage and testing
if __name__ == "__main__":
    # Test the report generator
    from data.live_data_loader import DataLoader
    from technical_analysis.advanced_indicators import TechnicalAnalyzer
    from pattern_analysis.ml_candlestick_patterns import MLCandlestickDiscovery
    
    print("Testing Report Generator...")
    
    # Load sample data
    loader = DataLoader()
    data = loader.fetch_single_timeframe_data(timeframe="1d")
    
    if data is not None and not data.empty:
        # Calculate technical indicators
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.calculate_all_indicators(data)
        
        # Discover patterns
        pattern_discovery = MLCandlestickDiscovery()
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
        
        # Initialize report generator
        report_gen = ReportGenerator()
        
        # Generate reports
        technical_reports = report_gen.generate_technical_analysis_report(
            indicators, patterns, predictions, data
        )
        
        price_action_reports = report_gen.generate_price_action_report(
            patterns, data, {}
        )
        
        print(f"Technical reports generated: {technical_reports}")
        print(f"Price action reports generated: {price_action_reports}")
        
    else:
        print("No data available for testing")
