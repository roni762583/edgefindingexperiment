"""
Basic Evaluator for Market Edge Finder Experiment

Integrates Monte Carlo validation with edge discovery evaluation.
Determines statistical significance of discovered edges using robust validation methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from validation.edge_discovery_monte_carlo import EdgeMonteCarloValidator

logger = logging.getLogger(__name__)


class EdgeDiscoveryEvaluator:
    """
    Evaluator for edge discovery experiments.
    
    Combines traditional ML metrics with Monte Carlo statistical validation
    to determine if discovered patterns represent statistically significant edges.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.validation_config = config.get('validation', {})
        
        # Thresholds for edge discovery
        self.sharpe_threshold = self.evaluation_config.get('sharpe_threshold', 0.5)
        self.max_drawdown_threshold = self.evaluation_config.get('max_drawdown_threshold', 0.15)
        self.min_sample_size = self.evaluation_config.get('min_sample_size', 2000)
        self.confidence_level = self.evaluation_config.get('confidence_level', 0.95)
        
        # Initialize Monte Carlo validator
        self.monte_carlo_validator = EdgeMonteCarloValidator(config)
        
        # Results storage
        self.evaluation_results_: Dict[str, Any] = {}
        
        logger.info(f"EdgeDiscoveryEvaluator initialized with thresholds: "
                   f"Sharpe≥{self.sharpe_threshold}, MaxDD≤{self.max_drawdown_threshold}")
    
    def calculate_performance_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                    instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each instrument.
        
        Args:
            predictions: Prediction matrix [n_samples, n_instruments]
            targets: Target matrix [n_samples, n_instruments]
            instruments: List of instrument names
            
        Returns:
            Dictionary of metrics per instrument
        """
        metrics = {}
        
        for i, instrument in enumerate(instruments):
            y_true = targets[:, i]
            y_pred = predictions[:, i]
            
            # Filter valid predictions
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            
            if valid_mask.sum() < self.min_sample_size:
                metrics[instrument] = {
                    'status': 'INSUFFICIENT_DATA',
                    'n_samples': valid_mask.sum()
                }
                continue
            
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            # Calculate traditional ML metrics
            instrument_metrics = self._calculate_ml_metrics(y_true_valid, y_pred_valid)
            
            # Calculate trading-oriented metrics
            trading_metrics = self._calculate_trading_metrics(y_true_valid, y_pred_valid)
            
            # Combine metrics
            instrument_metrics.update(trading_metrics)
            instrument_metrics['n_samples'] = len(y_true_valid)
            instrument_metrics['status'] = 'VALID'
            
            metrics[instrument] = instrument_metrics
        
        return metrics
    
    def _calculate_ml_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate traditional ML metrics."""
        # Basic regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        # Information coefficient (rank correlation)
        from scipy.stats import spearmanr
        ic, ic_pvalue = spearmanr(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'correlation': correlation,
            'information_coefficient': ic,
            'ic_pvalue': ic_pvalue
        }
    
    def _calculate_trading_metrics(self, returns: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate trading-oriented metrics.
        
        Args:
            returns: Actual returns (targets)
            predictions: Predicted returns
            
        Returns:
            Dictionary of trading metrics
        """
        # Create simple strategy: long if prediction > 0, short if < 0
        positions = np.sign(predictions)
        strategy_returns = positions * returns
        
        # Remove zero position periods
        active_returns = strategy_returns[positions != 0]
        
        if len(active_returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'hit_rate': 0.0,
                'profit_factor': 0.0,
                'avg_return': 0.0,
                'volatility': 0.0
            }
        
        # Sharpe ratio (annualized, assuming hourly returns)
        avg_return = np.mean(active_returns)
        volatility = np.std(active_returns)
        sharpe_ratio = (avg_return / volatility * np.sqrt(24 * 365)) if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + active_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdown))
        
        # Hit rate (percentage of profitable trades)
        hit_rate = np.mean(active_returns > 0)
        
        # Profit factor (total profit / total loss)
        profits = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        
        if len(profits) > 0 and len(losses) > 0:
            profit_factor = np.sum(profits) / np.abs(np.sum(losses))
        else:
            profit_factor = np.inf if len(losses) == 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'avg_return': avg_return,
            'volatility': volatility,
            'n_active_trades': len(active_returns)
        }
    
    def identify_edge_candidates(self, performance_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Identify instruments that show potential edge based on performance thresholds.
        
        Args:
            performance_metrics: Performance metrics per instrument
            
        Returns:
            List of instrument names showing potential edge
        """
        edge_candidates = []
        
        for instrument, metrics in performance_metrics.items():
            if metrics.get('status') != 'VALID':
                continue
            
            # Check edge criteria
            sharpe_ok = metrics.get('sharpe_ratio', 0) >= self.sharpe_threshold
            drawdown_ok = metrics.get('max_drawdown', 1) <= self.max_drawdown_threshold
            sample_size_ok = metrics.get('n_samples', 0) >= self.min_sample_size
            
            # Additional filters
            hit_rate_ok = metrics.get('hit_rate', 0) > 0.5  # Better than random
            ic_significant = metrics.get('ic_pvalue', 1) < 0.05  # Significant information coefficient
            
            if sharpe_ok and drawdown_ok and sample_size_ok and hit_rate_ok and ic_significant:
                edge_candidates.append(instrument)
                logger.info(f"✅ Edge candidate: {instrument} "
                           f"(Sharpe: {metrics['sharpe_ratio']:.3f}, "
                           f"MaxDD: {metrics['max_drawdown']:.3f}, "
                           f"Hit Rate: {metrics['hit_rate']:.3f})")
        
        logger.info(f"Identified {len(edge_candidates)} edge candidates out of {len(performance_metrics)} instruments")
        return edge_candidates
    
    def validate_edges_monte_carlo(self, predictions: np.ndarray, targets: np.ndarray,
                                  instruments: List[str], edge_candidates: List[str]) -> Dict[str, Any]:
        """
        Validate edge candidates using Monte Carlo methods.
        
        Args:
            predictions: Prediction matrix
            targets: Target matrix  
            instruments: List of instrument names
            edge_candidates: List of candidate instruments
            
        Returns:
            Monte Carlo validation results
        """
        if not edge_candidates:
            logger.info("No edge candidates to validate")
            return {'validated_edges': [], 'monte_carlo_results': {}}
        
        logger.info(f"Running Monte Carlo validation for {len(edge_candidates)} edge candidates")
        
        # Prepare data for Monte Carlo validation
        edge_indices = [instruments.index(inst) for inst in edge_candidates]
        
        mc_predictions = predictions[:, edge_indices]
        mc_targets = targets[:, edge_indices]
        
        # Run Monte Carlo validation
        mc_results = self.monte_carlo_validator.run_bootstrap_scenarios(
            predictions=mc_predictions,
            targets=mc_targets,
            instruments=edge_candidates
        )
        
        # Analyze results to determine validated edges
        validated_edges = self._analyze_monte_carlo_results(mc_results, edge_candidates)
        
        return {
            'validated_edges': validated_edges,
            'monte_carlo_results': mc_results
        }
    
    def _analyze_monte_carlo_results(self, mc_results: Dict[str, Any], 
                                   edge_candidates: List[str]) -> List[str]:
        """
        Analyze Monte Carlo results to determine validated edges.
        
        Args:
            mc_results: Monte Carlo validation results
            edge_candidates: Edge candidate instruments
            
        Returns:
            List of validated edge instruments
        """
        validated_edges = []
        
        confidence_threshold = self.confidence_level
        
        for instrument in edge_candidates:
            # Check if instrument passed validation in multiple scenarios
            scenario_passes = 0
            total_scenarios = 0
            
            for scenario_name, scenario_results in mc_results.items():
                if 'instrument_results' in scenario_results:
                    inst_results = scenario_results['instrument_results']
                    if instrument in inst_results:
                        total_scenarios += 1
                        
                        # Check if this instrument passed validation criteria
                        inst_metrics = inst_results[instrument]
                        
                        # Criteria for validation
                        sharpe_stable = inst_metrics.get('confidence_interval_sharpe', {}).get('lower', 0) > 0.2
                        consistent_performance = inst_metrics.get('stability_score', 0) > 0.7
                        
                        if sharpe_stable and consistent_performance:
                            scenario_passes += 1
            
            # Validate if passed majority of scenarios
            if total_scenarios > 0 and (scenario_passes / total_scenarios) >= confidence_threshold:
                validated_edges.append(instrument)
                logger.info(f"✅ VALIDATED EDGE: {instrument} "
                           f"({scenario_passes}/{total_scenarios} scenarios passed)")
        
        return validated_edges
    
    def generate_edge_discovery_report(self, performance_metrics: Dict[str, Dict[str, float]],
                                     edge_candidates: List[str],
                                     validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive edge discovery report.
        
        Args:
            performance_metrics: Performance metrics per instrument
            edge_candidates: Edge candidate instruments
            validation_results: Monte Carlo validation results
            
        Returns:
            Comprehensive edge discovery report
        """
        validated_edges = validation_results.get('validated_edges', [])
        
        # Summary statistics
        total_instruments = len(performance_metrics)
        valid_instruments = sum(1 for m in performance_metrics.values() if m.get('status') == 'VALID')
        
        # Edge discovery summary
        edge_discovery_summary = {
            'total_instruments_tested': total_instruments,
            'valid_instruments': valid_instruments,
            'edge_candidates_identified': len(edge_candidates),
            'validated_edges': len(validated_edges),
            'edge_discovery_rate': len(validated_edges) / valid_instruments if valid_instruments > 0 else 0
        }
        
        # Performance summary for validated edges
        validated_edge_performance = {}
        for instrument in validated_edges:
            if instrument in performance_metrics:
                validated_edge_performance[instrument] = performance_metrics[instrument]
        
        # Statistical significance assessment
        statistical_assessment = {
            'confidence_level': self.confidence_level,
            'monte_carlo_scenarios_run': len(validation_results.get('monte_carlo_results', {})),
            'validation_passed': len(validated_edges) > 0
        }
        
        # Conclusion
        if validated_edges:
            conclusion = f"EDGE DISCOVERED: {len(validated_edges)} statistically significant edge(s) found"
            recommendation = "Proceed with deeper analysis and potential strategy development"
        else:
            conclusion = "NO EDGE DISCOVERED: No statistically significant edges found"
            recommendation = "Consider alternative approaches, different timeframes, or additional features"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': edge_discovery_summary,
            'conclusion': conclusion,
            'recommendation': recommendation,
            'validated_edges': validated_edges,
            'edge_candidates': edge_candidates,
            'performance_metrics': performance_metrics,
            'validated_edge_performance': validated_edge_performance,
            'statistical_assessment': statistical_assessment,
            'monte_carlo_results': validation_results.get('monte_carlo_results', {}),
            'configuration': {
                'sharpe_threshold': self.sharpe_threshold,
                'max_drawdown_threshold': self.max_drawdown_threshold,
                'min_sample_size': self.min_sample_size,
                'confidence_level': self.confidence_level
            }
        }
        
        return report
    
    def evaluate_experiment(self, predictions: np.ndarray, targets: np.ndarray,
                          instruments: List[str]) -> Dict[str, Any]:
        """
        Run complete edge discovery evaluation.
        
        Args:
            predictions: Model predictions
            targets: Target values
            instruments: Instrument names
            
        Returns:
            Complete evaluation results
        """
        logger.info("🔍 Starting edge discovery evaluation")
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        performance_metrics = self.calculate_performance_metrics(predictions, targets, instruments)
        
        # Identify edge candidates
        logger.info("Identifying edge candidates...")
        edge_candidates = self.identify_edge_candidates(performance_metrics)
        
        # Validate edges with Monte Carlo
        logger.info("Running Monte Carlo validation...")
        validation_results = self.validate_edges_monte_carlo(
            predictions, targets, instruments, edge_candidates
        )
        
        # Generate comprehensive report
        logger.info("Generating edge discovery report...")
        report = self.generate_edge_discovery_report(
            performance_metrics, edge_candidates, validation_results
        )
        
        # Store results
        self.evaluation_results_ = report
        
        logger.info("✅ Edge discovery evaluation completed")
        return report
    
    def save_results(self, output_path: Union[str, Path]):
        """
        Save evaluation results to file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results_, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def print_summary(self):
        """Print evaluation summary to console."""
        if not self.evaluation_results_:
            logger.warning("No evaluation results to display")
            return
        
        print("\n" + "="*80)
        print("EDGE DISCOVERY EVALUATION SUMMARY")
        print("="*80)
        
        summary = self.evaluation_results_['summary']
        print(f"Total instruments tested: {summary['total_instruments_tested']}")
        print(f"Valid instruments: {summary['valid_instruments']}")
        print(f"Edge candidates identified: {summary['edge_candidates_identified']}")
        print(f"Validated edges: {summary['validated_edges']}")
        print(f"Edge discovery rate: {summary['edge_discovery_rate']:.1%}")
        
        print(f"\nConclusion: {self.evaluation_results_['conclusion']}")
        print(f"Recommendation: {self.evaluation_results_['recommendation']}")
        
        if self.evaluation_results_['validated_edges']:
            print(f"\nValidated Edges:")
            for edge in self.evaluation_results_['validated_edges']:
                metrics = self.evaluation_results_['validated_edge_performance'].get(edge, {})
                print(f"  • {edge}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                      f"Hit Rate={metrics.get('hit_rate', 0):.1%}")
        
        print("="*80)


def load_config_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = load_config_from_yaml("configs/production_config.yaml")
    
    # Initialize evaluator
    evaluator = EdgeDiscoveryEvaluator(config)
    
    # Generate test data
    np.random.seed(42)
    n_samples, n_instruments = 2000, 5
    
    # Create realistic test data with some edge-like behavior
    instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CHF']
    
    # Generate targets (true returns)
    targets = np.random.randn(n_samples, n_instruments) * 0.01
    
    # Generate predictions with varying quality
    predictions = np.zeros_like(targets)
    for i in range(n_instruments):
        # Different quality for each instrument
        signal_strength = [0.8, 0.3, 0.1, 0.6, 0.05][i]  # Varying edge strength
        noise_level = 1 - signal_strength
        
        predictions[:, i] = (signal_strength * targets[:, i] + 
                           noise_level * np.random.randn(n_samples) * 0.01)
    
    print("Running edge discovery evaluation test...")
    
    # Run evaluation
    results = evaluator.evaluate_experiment(predictions, targets, instruments)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results("test_edge_discovery_results.json")
    
    print("\n✅ EdgeDiscoveryEvaluator test completed successfully")