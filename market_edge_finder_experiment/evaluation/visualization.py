"""
Visualization utilities for evaluation results and model performance analysis.

This module provides comprehensive plotting and visualization functions for:
- Performance metrics visualization
- Regime analysis plots
- Correlation heatmaps
- Time series performance charts
- Drawdown analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)


class EvaluationVisualizer:
    """
    Comprehensive visualization class for evaluation results.
    
    Provides methods to create various plots and charts for analyzing
    model performance, regime behavior, and trading metrics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
            'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
            'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD',
            'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD'
        ]
    
    def plot_sharpe_ratios(self, 
                          sharpe_data: Dict[str, float], 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar plot of Sharpe ratios by instrument.
        
        Args:
            sharpe_data: Dictionary with instrument names and Sharpe ratios
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Filter out portfolio for separate handling
        instruments = [k for k in sharpe_data.keys() if k != 'portfolio']
        sharpe_values = [sharpe_data[k] for k in instruments]
        
        # Create bar plot
        bars = ax.bar(range(len(instruments)), sharpe_values, 
                     color='steelblue', alpha=0.7, edgecolor='navy')
        
        # Highlight portfolio if present
        if 'portfolio' in sharpe_data:
            portfolio_bar = ax.bar(len(instruments), sharpe_data['portfolio'], 
                                 color='red', alpha=0.8, edgecolor='darkred', 
                                 label='Portfolio')
            instruments.append('Portfolio')
        
        # Formatting
        ax.set_xlabel('Instruments', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Sharpe Ratios by Instrument (Net of Transaction Costs)', 
                    fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.set_xticks(range(len(instruments)))
        ax.set_xticklabels(instruments, rotation=45, ha='right')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sharpe_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Add portfolio value if present
        if 'portfolio' in sharpe_data:
            ax.text(len(instruments)-1 + 0.5, sharpe_data['portfolio'] + 0.01,
                   f'{sharpe_data["portfolio"]:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_precision_at_k(self, 
                           precision_data: Dict[str, Dict[int, float]], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create line plot of Precision@k curves.
        
        Args:
            precision_data: Dictionary with instruments and their precision@k scores
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot precision curves for each instrument
        for instrument, precision_scores in precision_data.items():
            if instrument == 'average':
                continue  # Handle average separately
                
            k_values = sorted(precision_scores.keys())
            precision_values = [precision_scores[k] for k in k_values]
            
            ax.plot(k_values, precision_values, marker='o', markersize=4, 
                   alpha=0.6, linewidth=1, label=instrument)
        
        # Highlight average if present
        if 'average' in precision_data:
            k_values = sorted(precision_data['average'].keys())
            avg_values = [precision_data['average'][k] for k in k_values]
            ax.plot(k_values, avg_values, marker='s', markersize=6, 
                   linewidth=3, color='red', label='Average', alpha=0.9)
        
        # Formatting
        ax.set_xlabel('k (Top-k Predictions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision@k', fontsize=12, fontweight='bold')
        ax.set_title('Precision@k Performance Across Instruments', 
                    fontsize=14, fontweight='bold')
        
        # Add horizontal line at 0.5 (random performance)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, 
                  label='Random Performance')
        
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown_analysis(self, 
                              drawdown_data: Dict[str, float], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of maximum drawdown by instrument.
        
        Args:
            drawdown_data: Dictionary with instrument names and max drawdown values
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Filter out portfolio for separate handling
        instruments = [k for k in drawdown_data.keys() if k != 'portfolio']
        drawdown_values = [abs(drawdown_data[k]) * 100 for k in instruments]  # Convert to positive percentages
        
        # Create bar plot
        bars = ax.bar(range(len(instruments)), drawdown_values, 
                     color='red', alpha=0.7, edgecolor='darkred')
        
        # Highlight portfolio if present
        if 'portfolio' in drawdown_data:
            portfolio_bar = ax.bar(len(instruments), abs(drawdown_data['portfolio']) * 100, 
                                 color='darkred', alpha=0.9, edgecolor='black', 
                                 label='Portfolio')
            instruments.append('Portfolio')
        
        # Formatting
        ax.set_xlabel('Instruments', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Maximum Drawdown by Instrument', 
                    fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        ax.set_xticks(range(len(instruments)))
        ax.set_xticklabels(instruments, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, drawdown_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Add portfolio value if present
        if 'portfolio' in drawdown_data:
            portfolio_value = abs(drawdown_data['portfolio']) * 100
            ax.text(len(instruments)-1 + 0.5, portfolio_value + 0.5,
                   f'{portfolio_value:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_regime_analysis(self, 
                           regime_data: Dict[str, Any], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of regime-specific performance.
        
        Args:
            regime_data: Dictionary with regime analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        
        instruments = list(regime_data.keys())
        
        # 1. Accuracy by regime
        high_vol_acc = []
        low_vol_acc = []
        
        for inst in instruments:
            if 'high_volatility' in regime_data[inst]:
                high_vol_acc.append(regime_data[inst]['high_volatility']['accuracy'])
            else:
                high_vol_acc.append(0)
                
            if 'low_volatility' in regime_data[inst]:
                low_vol_acc.append(regime_data[inst]['low_volatility']['accuracy'])
            else:
                low_vol_acc.append(0)
        
        x = np.arange(len(instruments))
        width = 0.35
        
        ax1.bar(x - width/2, high_vol_acc, width, label='High Volatility', 
               color='red', alpha=0.7)
        ax1.bar(x + width/2, low_vol_acc, width, label='Low Volatility', 
               color='blue', alpha=0.7)
        
        ax1.set_xlabel('Instruments')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Volatility Regime')
        ax1.set_xticks(x)
        ax1.set_xticklabels(instruments, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Correlation by regime
        high_vol_corr = []
        low_vol_corr = []
        
        for inst in instruments:
            if 'high_volatility' in regime_data[inst]:
                high_vol_corr.append(regime_data[inst]['high_volatility']['correlation'])
            else:
                high_vol_corr.append(0)
                
            if 'low_volatility' in regime_data[inst]:
                low_vol_corr.append(regime_data[inst]['low_volatility']['correlation'])
            else:
                low_vol_corr.append(0)
        
        ax2.bar(x - width/2, high_vol_corr, width, label='High Volatility', 
               color='red', alpha=0.7)
        ax2.bar(x + width/2, low_vol_corr, width, label='Low Volatility', 
               color='blue', alpha=0.7)
        
        ax2.set_xlabel('Instruments')
        ax2.set_ylabel('Correlation')
        ax2.set_title('Correlation by Volatility Regime')
        ax2.set_xticks(x)
        ax2.set_xticklabels(instruments, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime persistence
        persistence_ratios = []
        avg_durations = []
        
        for inst in instruments:
            if 'persistence' in regime_data[inst]:
                persistence_ratios.append(regime_data[inst]['persistence']['persistence_ratio'])
                avg_durations.append(regime_data[inst]['persistence']['average_duration_hours'])
            else:
                persistence_ratios.append(0)
                avg_durations.append(0)
        
        ax3.bar(instruments, persistence_ratios, color='green', alpha=0.7)
        ax3.set_xlabel('Instruments')
        ax3.set_ylabel('Persistence Ratio')
        ax3.set_title('Regime Persistence Ratio')
        ax3.set_xticklabels(instruments, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Average regime duration
        ax4.bar(instruments, avg_durations, color='orange', alpha=0.7)
        ax4.set_xlabel('Instruments')
        ax4.set_ylabel('Average Duration (hours)')
        ax4.set_title('Average Regime Duration')
        ax4.set_xticklabels(instruments, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_decay(self, 
                              correlation_data: Dict[str, List[float]], 
                              max_pairs: int = 10,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of cross-instrument correlation decay.
        
        Args:
            correlation_data: Dictionary with instrument pairs and correlation values
            max_pairs: Maximum number of pairs to plot
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Select top pairs by absolute correlation at lag 0
        pairs_by_correlation = sorted(correlation_data.items(), 
                                    key=lambda x: abs(x[1][0]), reverse=True)
        
        selected_pairs = pairs_by_correlation[:max_pairs]
        
        # Plot correlation decay for selected pairs
        max_lag = len(list(correlation_data.values())[0]) - 1
        lags = range(max_lag + 1)
        
        for pair_name, correlations in selected_pairs:
            ax.plot(lags, correlations, marker='o', markersize=3, 
                   alpha=0.7, linewidth=2, label=pair_name)
        
        # Formatting
        ax.set_xlabel('Lag (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross-Instrument Correlation', fontsize=12, fontweight='bold')
        ax.set_title(f'Cross-Instrument Correlation Decay (Top {max_pairs} Pairs)', 
                    fontsize=14, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_over_time(self, 
                                  predictions: np.ndarray, 
                                  returns: np.ndarray,
                                  timestamps: Optional[np.ndarray] = None,
                                  window_hours: int = 720,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create time series plot of rolling performance metrics.
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            timestamps: Optional timestamps for x-axis
            window_hours: Rolling window size in hours
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figsize[0], 12), dpi=self.dpi)
        
        # Calculate rolling metrics
        rolling_accuracy = []
        rolling_correlation = []
        rolling_sharpe = []
        window_centers = []
        
        for i in range(window_hours, len(predictions), window_hours // 4):
            window_pred = predictions[i-window_hours:i]
            window_ret = returns[i-window_hours:i]
            
            # Accuracy
            accuracy = np.mean(np.sign(window_pred) == np.sign(window_ret))
            rolling_accuracy.append(accuracy)
            
            # Correlation
            if window_pred.ndim > 1:
                flat_pred = window_pred.flatten()
                flat_ret = window_ret.flatten()
            else:
                flat_pred = window_pred
                flat_ret = window_ret
                
            correlation = np.corrcoef(flat_pred, flat_ret)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            rolling_correlation.append(correlation)
            
            # Sharpe ratio (simplified)
            trading_returns = window_pred * window_ret
            if np.std(trading_returns) > 1e-8:
                sharpe = np.mean(trading_returns) / np.std(trading_returns) * np.sqrt(8760)
            else:
                sharpe = 0.0
            rolling_sharpe.append(sharpe)
            
            window_centers.append(i - window_hours // 2)
        
        # Create time axis
        if timestamps is not None:
            time_axis = pd.to_datetime(timestamps)[window_centers]
        else:
            time_axis = window_centers
        
        # Plot rolling accuracy
        ax1.plot(time_axis, rolling_accuracy, linewidth=2, color='blue')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax1.set_ylabel('Rolling Accuracy')
        ax1.set_title('Rolling Performance Metrics Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot rolling correlation
        ax2.plot(time_axis, rolling_correlation, linewidth=2, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Rolling Correlation')
        ax2.grid(True, alpha=0.3)
        
        # Plot rolling Sharpe ratio
        ax3.plot(time_axis, rolling_sharpe, linewidth=2, color='orange')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Rolling Sharpe Ratio')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis if using timestamps
        if timestamps is not None:
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_dashboard(self, 
                                     evaluation_results: Dict[str, Any],
                                     predictions: Optional[np.ndarray] = None,
                                     returns: Optional[np.ndarray] = None,
                                     timestamps: Optional[np.ndarray] = None,
                                     save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive evaluation dashboard with all visualizations.
        
        Args:
            evaluation_results: Complete evaluation results dictionary
            predictions: Optional predictions for time series plots
            returns: Optional returns for time series plots
            timestamps: Optional timestamps for time series plots
            save_dir: Optional directory to save all plots
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Sharpe Ratios
        if 'sharpe_ratios' in evaluation_results:
            fig = self.plot_sharpe_ratios(
                evaluation_results['sharpe_ratios'],
                save_path=save_dir / 'sharpe_ratios.png' if save_dir else None
            )
            figures['sharpe_ratios'] = fig
        
        # 2. Precision@k
        if 'precision_at_k' in evaluation_results:
            fig = self.plot_precision_at_k(
                evaluation_results['precision_at_k'],
                save_path=save_dir / 'precision_at_k.png' if save_dir else None
            )
            figures['precision_at_k'] = fig
        
        # 3. Maximum Drawdown
        if 'maximum_drawdown' in evaluation_results:
            fig = self.plot_drawdown_analysis(
                evaluation_results['maximum_drawdown'],
                save_path=save_dir / 'maximum_drawdown.png' if save_dir else None
            )
            figures['maximum_drawdown'] = fig
        
        # 4. Regime Analysis
        if 'regime_analysis' in evaluation_results:
            fig = self.plot_regime_analysis(
                evaluation_results['regime_analysis'],
                save_path=save_dir / 'regime_analysis.png' if save_dir else None
            )
            figures['regime_analysis'] = fig
        
        # 5. Correlation Decay
        if 'correlation_decay' in evaluation_results:
            fig = self.plot_correlation_decay(
                evaluation_results['correlation_decay'],
                save_path=save_dir / 'correlation_decay.png' if save_dir else None
            )
            figures['correlation_decay'] = fig
        
        # 6. Performance Over Time
        if predictions is not None and returns is not None:
            fig = self.plot_performance_over_time(
                predictions, returns, timestamps,
                save_path=save_dir / 'performance_over_time.png' if save_dir else None
            )
            figures['performance_over_time'] = fig
        
        return figures


def create_summary_table(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table of key metrics.
    
    Args:
        evaluation_results: Complete evaluation results dictionary
        
    Returns:
        Pandas DataFrame with summary metrics
    """
    summary_data = []
    
    # Extract key metrics for each instrument
    instruments = []
    if 'sharpe_ratios' in evaluation_results:
        instruments = [k for k in evaluation_results['sharpe_ratios'].keys() if k != 'portfolio']
    
    for instrument in instruments:
        row = {'Instrument': instrument}
        
        # Sharpe ratio
        if 'sharpe_ratios' in evaluation_results:
            row['Sharpe_Ratio'] = evaluation_results['sharpe_ratios'].get(instrument, 0.0)
        
        # Best precision@k
        if 'precision_at_k' in evaluation_results and instrument in evaluation_results['precision_at_k']:
            precision_scores = evaluation_results['precision_at_k'][instrument]
            row['Best_Precision_K'] = max(precision_scores.values()) if precision_scores else 0.0
        
        # Maximum drawdown
        if 'maximum_drawdown' in evaluation_results:
            row['Max_Drawdown_Pct'] = abs(evaluation_results['maximum_drawdown'].get(instrument, 0.0)) * 100
        
        summary_data.append(row)
    
    # Add portfolio summary if available
    if 'portfolio' in evaluation_results.get('sharpe_ratios', {}):
        portfolio_row = {'Instrument': 'PORTFOLIO'}
        portfolio_row['Sharpe_Ratio'] = evaluation_results['sharpe_ratios']['portfolio']
        
        if 'maximum_drawdown' in evaluation_results and 'portfolio' in evaluation_results['maximum_drawdown']:
            portfolio_row['Max_Drawdown_Pct'] = abs(evaluation_results['maximum_drawdown']['portfolio']) * 100
        
        summary_data.append(portfolio_row)
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    print("Market Edge Finder Evaluation Visualization Module")
    print("This module provides comprehensive visualization capabilities for evaluation results.")