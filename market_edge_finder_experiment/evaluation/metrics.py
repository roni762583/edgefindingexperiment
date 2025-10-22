"""
Comprehensive evaluation metrics for the Market Edge Finder Experiment.

This module implements all evaluation metrics specified in the original requirements:
- Sharpe ratio (net of transaction costs)
- Precision@k 
- Maximum drawdown
- Regime persistence analysis
- Cross-instrument correlation decay

All metrics are designed for multi-instrument FX trading systems with walk-forward validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
import warnings
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics calculation."""
    
    # Transaction cost parameters
    spread_bps: float = 0.5  # Average spread in basis points
    commission_bps: float = 0.1  # Commission in basis points
    
    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Precision@k parameters
    k_values: List[int] = None  # Top-k predictions to evaluate
    
    # Regime analysis parameters
    regime_window: int = 168  # 1 week in hours for regime persistence
    volatility_threshold: float = 1.5  # Threshold for high volatility regimes
    
    # Correlation decay parameters
    max_lag_hours: int = 24  # Maximum lag for correlation analysis
    correlation_window: int = 720  # 30 days rolling window
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 5, 10, 20]  # Default k values


class TradingMetricsCalculator:
    """
    Comprehensive metrics calculator for multi-instrument FX trading systems.
    
    This class provides methods to calculate all key performance metrics including
    risk-adjusted returns, precision metrics, drawdown analysis, and regime-specific
    performance evaluation.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            config: Configuration for metrics calculation. If None, uses defaults.
        """
        self.config = config or MetricsConfig()
        self.instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
            'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
            'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD',
            'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD'
        ]
        
    def calculate_transaction_costs(self, 
                                  predictions: np.ndarray, 
                                  prices: np.ndarray,
                                  instrument: Optional[str] = None) -> np.ndarray:
        """
        Calculate transaction costs for trading signals.
        
        Args:
            predictions: Model predictions (N, instruments) or (N,) for single instrument
            prices: Corresponding prices (N, instruments) or (N,) for single instrument
            instrument: Specific instrument name for custom spread/commission
            
        Returns:
            Transaction costs per trade
        """
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            
        # Calculate position changes (trades occur when prediction sign changes)
        position_changes = np.diff(np.sign(predictions), axis=0, prepend=0)
        trades = np.abs(position_changes) > 0
        
        # Calculate costs: spread + commission as percentage of notional
        total_cost_bps = self.config.spread_bps + self.config.commission_bps
        transaction_costs = trades * (total_cost_bps / 10000.0) * np.abs(prices)
        
        return transaction_costs
    
    def calculate_sharpe_ratio(self, 
                              returns: np.ndarray, 
                              predictions: np.ndarray,
                              prices: np.ndarray,
                              annualization_factor: float = 8760) -> Dict[str, float]:
        """
        Calculate Sharpe ratio net of transaction costs.
        
        Args:
            returns: Actual returns (N, instruments)
            predictions: Model predictions (N, instruments)
            prices: Corresponding prices (N, instruments)
            annualization_factor: Factor to annualize returns (8760 for hourly data)
            
        Returns:
            Dictionary with Sharpe ratios per instrument and portfolio
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            
        # Calculate trading returns (predictions * actual returns)
        trading_returns = predictions * returns
        
        # Subtract transaction costs
        transaction_costs = self.calculate_transaction_costs(predictions, prices)
        net_returns = trading_returns - transaction_costs
        
        # Calculate Sharpe ratios
        sharpe_ratios = {}
        
        # Per instrument Sharpe ratios
        for i in range(net_returns.shape[1]):
            instrument_returns = net_returns[:, i]
            if np.std(instrument_returns) > 1e-8:  # Avoid division by zero
                excess_returns = instrument_returns - (self.config.risk_free_rate / annualization_factor)
                sharpe = np.mean(excess_returns) / np.std(instrument_returns) * np.sqrt(annualization_factor)
                instrument_name = self.instruments[i] if i < len(self.instruments) else f"instrument_{i}"
                sharpe_ratios[instrument_name] = sharpe
            else:
                instrument_name = self.instruments[i] if i < len(self.instruments) else f"instrument_{i}"
                sharpe_ratios[instrument_name] = 0.0
        
        # Portfolio Sharpe ratio (equal-weighted)
        portfolio_returns = np.mean(net_returns, axis=1)
        if np.std(portfolio_returns) > 1e-8:
            portfolio_excess = portfolio_returns - (self.config.risk_free_rate / annualization_factor)
            portfolio_sharpe = np.mean(portfolio_excess) / np.std(portfolio_returns) * np.sqrt(annualization_factor)
            sharpe_ratios['portfolio'] = portfolio_sharpe
        else:
            sharpe_ratios['portfolio'] = 0.0
            
        return sharpe_ratios
    
    def calculate_precision_at_k(self, 
                                predictions: np.ndarray, 
                                returns: np.ndarray,
                                k_values: Optional[List[int]] = None) -> Dict[str, Dict[int, float]]:
        """
        Calculate Precision@k for top-k predictions.
        
        Args:
            predictions: Model predictions (N, instruments)
            returns: Actual returns (N, instruments)
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with precision@k scores per instrument and k value
        """
        if k_values is None:
            k_values = self.config.k_values
            
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
            
        precision_scores = {}
        
        for i in range(predictions.shape[1]):
            instrument_name = self.instruments[i] if i < len(self.instruments) else f"instrument_{i}"
            precision_scores[instrument_name] = {}
            
            pred_col = predictions[:, i]
            ret_col = returns[:, i]
            
            for k in k_values:
                if len(pred_col) < k:
                    precision_scores[instrument_name][k] = 0.0
                    continue
                    
                # Get top-k predictions by absolute value
                top_k_indices = np.argsort(np.abs(pred_col))[-k:]
                
                # Check if predictions and returns have same sign (correct direction)
                correct_predictions = np.sum(np.sign(pred_col[top_k_indices]) == np.sign(ret_col[top_k_indices]))
                precision_scores[instrument_name][k] = correct_predictions / k
        
        # Calculate average precision across instruments
        avg_precision = {}
        for k in k_values:
            scores = [precision_scores[inst][k] for inst in precision_scores.keys()]
            avg_precision[k] = np.mean(scores)
        precision_scores['average'] = avg_precision
        
        return precision_scores
    
    def calculate_maximum_drawdown(self, 
                                  returns: np.ndarray, 
                                  predictions: np.ndarray,
                                  prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown for trading strategy.
        
        Args:
            returns: Actual returns (N, instruments)
            predictions: Model predictions (N, instruments)
            prices: Corresponding prices (N, instruments)
            
        Returns:
            Dictionary with maximum drawdown per instrument and portfolio
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
            
        # Calculate net trading returns
        trading_returns = predictions * returns
        transaction_costs = self.calculate_transaction_costs(predictions, prices)
        net_returns = trading_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + net_returns, axis=0)
        
        drawdowns = {}
        
        # Per instrument drawdowns
        for i in range(cumulative_returns.shape[1]):
            cum_ret = cumulative_returns[:, i]
            running_max = np.maximum.accumulate(cum_ret)
            drawdown = (cum_ret - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            instrument_name = self.instruments[i] if i < len(self.instruments) else f"instrument_{i}"
            drawdowns[instrument_name] = max_drawdown
        
        # Portfolio drawdown (equal-weighted)
        portfolio_returns = np.mean(net_returns, axis=1)
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        portfolio_running_max = np.maximum.accumulate(portfolio_cumulative)
        portfolio_drawdown = (portfolio_cumulative - portfolio_running_max) / portfolio_running_max
        drawdowns['portfolio'] = np.min(portfolio_drawdown)
        
        return drawdowns
    
    def analyze_regime_persistence(self, 
                                  predictions: np.ndarray, 
                                  returns: np.ndarray,
                                  volatilities: np.ndarray,
                                  window: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze model performance across different market regimes.
        
        Args:
            predictions: Model predictions (N, instruments)
            returns: Actual returns (N, instruments)
            volatilities: Market volatilities (N, instruments)
            window: Rolling window for regime analysis
            
        Returns:
            Dictionary with regime persistence analysis results
        """
        if window is None:
            window = self.config.regime_window
            
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        if volatilities.ndim == 1:
            volatilities = volatilities.reshape(-1, 1)
            
        regime_analysis = {}
        
        for i in range(predictions.shape[1]):
            instrument_name = self.instruments[i] if i < len(self.instruments) else f"instrument_{i}"
            
            pred_col = predictions[:, i]
            ret_col = returns[:, i]
            vol_col = volatilities[:, i]
            
            # Define regimes based on volatility
            vol_threshold = np.percentile(vol_col, 75)  # High vol = top 25%
            high_vol_mask = vol_col > vol_threshold
            low_vol_mask = ~high_vol_mask
            
            # Calculate performance in each regime
            regime_performance = {}
            
            # High volatility regime
            if np.sum(high_vol_mask) > 0:
                high_vol_accuracy = np.mean(np.sign(pred_col[high_vol_mask]) == np.sign(ret_col[high_vol_mask]))
                high_vol_correlation = np.corrcoef(pred_col[high_vol_mask], ret_col[high_vol_mask])[0, 1]
                regime_performance['high_volatility'] = {
                    'accuracy': high_vol_accuracy,
                    'correlation': high_vol_correlation if not np.isnan(high_vol_correlation) else 0.0,
                    'samples': np.sum(high_vol_mask)
                }
            
            # Low volatility regime
            if np.sum(low_vol_mask) > 0:
                low_vol_accuracy = np.mean(np.sign(pred_col[low_vol_mask]) == np.sign(ret_col[low_vol_mask]))
                low_vol_correlation = np.corrcoef(pred_col[low_vol_mask], ret_col[low_vol_mask])[0, 1]
                regime_performance['low_volatility'] = {
                    'accuracy': low_vol_accuracy,
                    'correlation': low_vol_correlation if not np.isnan(low_vol_correlation) else 0.0,
                    'samples': np.sum(low_vol_mask)
                }
            
            # Regime persistence (how long each regime lasts)
            regime_switches = np.diff(high_vol_mask.astype(int))
            switch_points = np.where(regime_switches != 0)[0]
            
            if len(switch_points) > 1:
                regime_durations = np.diff(switch_points)
                avg_persistence = np.mean(regime_durations)
                regime_performance['persistence'] = {
                    'average_duration_hours': avg_persistence,
                    'total_switches': len(switch_points),
                    'persistence_ratio': avg_persistence / window
                }
            else:
                regime_performance['persistence'] = {
                    'average_duration_hours': len(pred_col),
                    'total_switches': 0,
                    'persistence_ratio': 1.0
                }
            
            regime_analysis[instrument_name] = regime_performance
        
        return regime_analysis
    
    def calculate_correlation_decay(self, 
                                  predictions: np.ndarray, 
                                  returns: np.ndarray,
                                  max_lag: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Calculate cross-instrument correlation decay over time lags.
        
        Args:
            predictions: Model predictions (N, instruments)
            returns: Actual returns (N, instruments)
            max_lag: Maximum lag in hours to analyze
            
        Returns:
            Dictionary with correlation decay patterns
        """
        if max_lag is None:
            max_lag = self.config.max_lag_hours
            
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
            
        correlation_decay = {}
        
        # Calculate correlation decay for each instrument pair
        n_instruments = predictions.shape[1]
        
        for i in range(n_instruments):
            for j in range(i + 1, n_instruments):
                inst_i = self.instruments[i] if i < len(self.instruments) else f"instrument_{i}"
                inst_j = self.instruments[j] if j < len(self.instruments) else f"instrument_{j}"
                pair_name = f"{inst_i}_{inst_j}"
                
                pred_i = predictions[:, i]
                pred_j = predictions[:, j]
                ret_i = returns[:, i]
                ret_j = returns[:, j]
                
                # Calculate correlation at different lags
                correlations = []
                
                for lag in range(max_lag + 1):
                    if lag == 0:
                        # Contemporaneous correlation
                        pred_corr = np.corrcoef(pred_i, pred_j)[0, 1]
                        ret_corr = np.corrcoef(ret_i, ret_j)[0, 1]
                    else:
                        # Lagged correlation
                        if len(pred_i) > lag:
                            pred_corr = np.corrcoef(pred_i[:-lag], pred_j[lag:])[0, 1]
                            ret_corr = np.corrcoef(ret_i[:-lag], ret_j[lag:])[0, 1]
                        else:
                            pred_corr = 0.0
                            ret_corr = 0.0
                    
                    # Use actual return correlation as baseline, measure prediction correlation
                    if not np.isnan(pred_corr):
                        correlations.append(pred_corr)
                    else:
                        correlations.append(0.0)
                
                correlation_decay[pair_name] = correlations
        
        return correlation_decay
    
    def generate_comprehensive_report(self, 
                                    predictions: np.ndarray, 
                                    returns: np.ndarray,
                                    prices: np.ndarray,
                                    volatilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report with all metrics.
        
        Args:
            predictions: Model predictions (N, instruments)
            returns: Actual returns (N, instruments)
            prices: Corresponding prices (N, instruments)
            volatilities: Market volatilities (N, instruments), optional
            
        Returns:
            Complete evaluation report dictionary
        """
        logger.info("Generating comprehensive evaluation report...")
        
        report = {}
        
        # 1. Sharpe Ratios
        logger.info("Calculating Sharpe ratios...")
        report['sharpe_ratios'] = self.calculate_sharpe_ratio(returns, predictions, prices)
        
        # 2. Precision@k
        logger.info("Calculating Precision@k metrics...")
        report['precision_at_k'] = self.calculate_precision_at_k(predictions, returns)
        
        # 3. Maximum Drawdown
        logger.info("Calculating maximum drawdown...")
        report['maximum_drawdown'] = self.calculate_maximum_drawdown(returns, predictions, prices)
        
        # 4. Regime Analysis (if volatilities provided)
        if volatilities is not None:
            logger.info("Analyzing regime persistence...")
            report['regime_analysis'] = self.analyze_regime_persistence(predictions, returns, volatilities)
        
        # 5. Correlation Decay
        logger.info("Calculating correlation decay...")
        report['correlation_decay'] = self.calculate_correlation_decay(predictions, returns)
        
        # 6. Summary Statistics
        logger.info("Calculating summary statistics...")
        report['summary'] = self._calculate_summary_stats(predictions, returns, prices)
        
        logger.info("Evaluation report generation complete.")
        return report
    
    def _calculate_summary_stats(self, predictions: np.ndarray, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Calculate summary statistics for the evaluation report."""
        
        # Overall accuracy (direction prediction)
        accuracy = np.mean(np.sign(predictions) == np.sign(returns))
        
        # Overall correlation
        flat_pred = predictions.flatten()
        flat_ret = returns.flatten()
        correlation = np.corrcoef(flat_pred, flat_ret)[0, 1] if not np.isnan(np.corrcoef(flat_pred, flat_ret)[0, 1]) else 0.0
        
        # Hit rate (profitable trades)
        trading_returns = predictions * returns
        hit_rate = np.mean(trading_returns > 0)
        
        # Average return per trade
        transaction_costs = self.calculate_transaction_costs(predictions, prices)
        net_trading_returns = trading_returns - transaction_costs
        avg_return_per_trade = np.mean(net_trading_returns)
        
        return {
            'overall_accuracy': accuracy,
            'overall_correlation': correlation,
            'hit_rate': hit_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'total_trades': np.sum(np.abs(np.diff(np.sign(predictions), axis=0)) > 0),
            'evaluation_period_hours': len(predictions)
        }


class BacktestEvaluator:
    """
    Specialized evaluator for walk-forward backtesting results.
    
    This class handles evaluation of models trained and tested using walk-forward
    validation to prevent data leakage and provide realistic performance estimates.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize the backtest evaluator.
        
        Args:
            config: Configuration for metrics calculation
        """
        self.config = config or MetricsConfig()
        self.metrics_calculator = TradingMetricsCalculator(config)
        
    def evaluate_walk_forward_results(self, 
                                     backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate walk-forward backtest results.
        
        Args:
            backtest_results: Dictionary containing walk-forward test results
                Should have keys: 'predictions', 'returns', 'prices', 'timestamps'
                
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Evaluating walk-forward backtest results...")
        
        predictions = backtest_results['predictions']
        returns = backtest_results['returns']
        prices = backtest_results['prices']
        timestamps = backtest_results.get('timestamps', None)
        volatilities = backtest_results.get('volatilities', None)
        
        # Generate comprehensive evaluation
        evaluation_results = self.metrics_calculator.generate_comprehensive_report(
            predictions, returns, prices, volatilities
        )
        
        # Add walk-forward specific metrics
        evaluation_results['walk_forward_stats'] = self._calculate_walk_forward_stats(
            predictions, returns, timestamps
        )
        
        return evaluation_results
    
    def _calculate_walk_forward_stats(self, 
                                    predictions: np.ndarray, 
                                    returns: np.ndarray,
                                    timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate statistics specific to walk-forward validation."""
        
        stats = {}
        
        # Calculate performance consistency across time periods
        if timestamps is not None:
            # Split into monthly periods
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps),
                'predictions': predictions.flatten() if predictions.ndim > 1 else predictions,
                'returns': returns.flatten() if returns.ndim > 1 else returns
            })
            
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_performance = []
            
            for month, group in df.groupby('month'):
                monthly_accuracy = np.mean(np.sign(group['predictions']) == np.sign(group['returns']))
                monthly_performance.append(monthly_accuracy)
            
            stats['monthly_performance_consistency'] = {
                'mean_monthly_accuracy': np.mean(monthly_performance),
                'std_monthly_accuracy': np.std(monthly_performance),
                'consistency_ratio': np.mean(monthly_performance) / (np.std(monthly_performance) + 1e-8)
            }
        
        # Performance stability over time
        window_size = min(720, len(predictions) // 4)  # 30 days or quarter of data
        rolling_accuracies = []
        
        for i in range(window_size, len(predictions), window_size // 2):
            window_pred = predictions[i-window_size:i]
            window_ret = returns[i-window_size:i]
            window_accuracy = np.mean(np.sign(window_pred) == np.sign(window_ret))
            rolling_accuracies.append(window_accuracy)
        
        if rolling_accuracies:
            stats['performance_stability'] = {
                'rolling_accuracy_mean': np.mean(rolling_accuracies),
                'rolling_accuracy_std': np.std(rolling_accuracies),
                'stability_score': np.mean(rolling_accuracies) / (np.std(rolling_accuracies) + 1e-8)
            }
        
        return stats


def save_evaluation_results(results: Dict[str, Any], 
                          output_path: Union[str, Path],
                          format: str = 'json') -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
        format: Output format ('json' or 'pickle')
    """
    import json
    import pickle
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Market Edge Finder Evaluation Metrics Module")
    logger.info("This module provides comprehensive evaluation capabilities for multi-instrument FX trading systems.")