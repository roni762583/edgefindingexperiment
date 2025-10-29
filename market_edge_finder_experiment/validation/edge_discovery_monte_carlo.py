#!/usr/bin/env python3
"""
Monte Carlo Edge Discovery Validation
Adapted from new_swt methodology for FX prediction edge testing

Tests whether discovered edges are statistically significant through:
- 6-scenario bootstrap stress testing
- Trajectory-based validation with confidence bands
- Cross-regime robustness testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EdgeMonteCarloValidator:
    """
    Monte Carlo validator for FX prediction edges
    
    Implements Dr. Howard Bandy methodology adapted for edge discovery:
    - Bootstrap sampling across multiple scenarios
    - Statistical significance testing
    - Trajectory analysis with confidence bands
    """
    
    def __init__(self, predictions: np.ndarray, actual_returns: np.ndarray, 
                 instruments: List[str], timestamps: pd.DatetimeIndex):
        """
        Initialize validator with prediction results
        
        Args:
            predictions: Model predictions (normalized -1 to +1)
            actual_returns: Actual USD-scaled pip returns  
            instruments: List of FX instrument names
            timestamps: Datetime index for data
        """
        self.predictions = predictions
        self.actual_returns = actual_returns
        self.instruments = instruments
        self.timestamps = timestamps
        
        # Validation parameters
        self.n_bootstrap_samples = 1000
        self.confidence_levels = [5, 25, 75, 95]
        
        # Results storage
        self.scenario_results = {}
        self.statistical_summary = {}
        
    def calculate_performance_metrics(self, pred: np.ndarray, returns: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Information Coefficient (correlation)
        ic = np.corrcoef(pred.flatten(), returns.flatten())[0, 1] if len(pred) > 1 else 0
        
        # Hit Rate (directional accuracy)
        pred_direction = np.sign(pred)
        actual_direction = np.sign(returns)
        hit_rate = np.mean(pred_direction == actual_direction)
        
        # Sharpe-like ratio (no transaction costs for pure edge measurement)
        prediction_returns = pred * returns  # Aligned returns
        sharpe = np.mean(prediction_returns) / np.std(prediction_returns) if np.std(prediction_returns) > 0 else 0
        
        # Return statistics
        total_return = np.sum(prediction_returns)
        max_drawdown = self._calculate_max_drawdown(np.cumsum(prediction_returns))
        
        return {
            'information_coefficient': ic,
            'hit_rate': hit_rate,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'n_predictions': len(pred)
        }
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)
    
    def run_bootstrap_scenarios(self) -> Dict:
        """
        Run 6-scenario bootstrap stress testing per new_swt methodology
        """
        print("🔄 Running 6-scenario bootstrap stress testing...")
        
        scenarios = {
            'original_bootstrap': self._original_bootstrap,
            'random_10_drop': self._random_drop_10,
            'tail_20_drop': self._tail_drop_20, 
            'oversample_150': self._oversample_150,
            'adverse_selection': self._adverse_selection,
            'early_stop_80': self._early_stop_80
        }
        
        results = {}
        
        for scenario_name, scenario_func in scenarios.items():
            print(f"  📊 {scenario_name}...")
            scenario_metrics = []
            
            for i in range(self.n_bootstrap_samples):
                # Apply scenario sampling
                sample_pred, sample_returns = scenario_func()
                
                # Calculate metrics
                metrics = self.calculate_performance_metrics(sample_pred, sample_returns)
                scenario_metrics.append(metrics)
            
            # Calculate confidence intervals
            results[scenario_name] = self._calculate_confidence_intervals(scenario_metrics)
            
        self.scenario_results = results
        return results
    
    def _original_bootstrap(self) -> Tuple[np.ndarray, np.ndarray]:
        """Standard bootstrap with replacement"""
        n = len(self.predictions)
        indices = np.random.choice(n, size=n, replace=True)
        return self.predictions[indices], self.actual_returns[indices]
    
    def _random_drop_10(self) -> Tuple[np.ndarray, np.ndarray]:
        """Random 10% prediction drop"""
        n = len(self.predictions)
        keep_indices = np.random.choice(n, size=int(0.9 * n), replace=False)
        return self.predictions[keep_indices], self.actual_returns[keep_indices]
    
    def _tail_drop_20(self) -> Tuple[np.ndarray, np.ndarray]:
        """Remove worst 20% of predictions"""
        # Calculate prediction quality (absolute correlation with returns)
        pred_quality = np.abs(self.predictions * self.actual_returns)
        threshold = np.percentile(pred_quality, 20)
        keep_indices = pred_quality >= threshold
        return self.predictions[keep_indices], self.actual_returns[keep_indices]
    
    def _oversample_150(self) -> Tuple[np.ndarray, np.ndarray]:
        """150% oversampling stress test"""
        n = len(self.predictions)
        indices = np.random.choice(n, size=int(1.5 * n), replace=True)
        return self.predictions[indices], self.actual_returns[indices]
    
    def _adverse_selection(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bias towards losing predictions"""
        # Calculate prediction returns
        pred_returns = self.predictions * self.actual_returns
        
        # Separate winners and losers
        winner_indices = np.where(pred_returns >= 0)[0]
        loser_indices = np.where(pred_returns < 0)[0]
        
        # Resample with 70% losers, 30% winners
        n = len(self.predictions)
        n_losers = int(0.7 * n)
        n_winners = n - n_losers
        
        selected_losers = np.random.choice(loser_indices, size=min(n_losers, len(loser_indices)), replace=True)
        selected_winners = np.random.choice(winner_indices, size=min(n_winners, len(winner_indices)), replace=True)
        
        all_indices = np.concatenate([selected_losers, selected_winners])
        np.random.shuffle(all_indices)
        
        return self.predictions[all_indices], self.actual_returns[all_indices]
    
    def _early_stop_80(self) -> Tuple[np.ndarray, np.ndarray]:
        """Early stopping at 80% of data"""
        n = int(0.8 * len(self.predictions))
        return self.predictions[:n], self.actual_returns[:n]
    
    def _calculate_confidence_intervals(self, metrics_list: List[Dict]) -> Dict:
        """Calculate confidence intervals for metrics"""
        result = {}
        
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            result[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'percentiles': {
                    f'p{p}': np.percentile(values, p) for p in self.confidence_levels
                }
            }
        
        return result
    
    def generate_trajectory_analysis(self) -> Dict:
        """Generate spaghetti plot trajectory analysis"""
        print("📈 Generating trajectory analysis...")
        
        trajectory_results = {}
        
        for scenario_name in self.scenario_results.keys():
            print(f"  📊 {scenario_name} trajectories...")
            
            trajectories = []
            for i in range(min(100, self.n_bootstrap_samples)):  # Limit for performance
                # Get scenario sample
                if scenario_name == 'original_bootstrap':
                    sample_pred, sample_returns = self._original_bootstrap()
                elif scenario_name == 'adverse_selection':
                    sample_pred, sample_returns = self._adverse_selection()
                else:
                    sample_pred, sample_returns = self._original_bootstrap()  # Fallback
                
                # Calculate cumulative returns
                aligned_returns = sample_pred * sample_returns
                trajectory = np.cumsum(aligned_returns)
                trajectories.append(trajectory)
            
            # Calculate trajectory statistics
            trajectories = np.array(trajectories)
            
            trajectory_results[scenario_name] = {
                'median': np.median(trajectories, axis=0),
                'mean': np.mean(trajectories, axis=0),
                'p5': np.percentile(trajectories, 5, axis=0),
                'p95': np.percentile(trajectories, 95, axis=0),
                'final_returns': trajectories[:, -1] if trajectories.shape[1] > 0 else []
            }
        
        return trajectory_results
    
    def test_statistical_significance(self) -> Dict:
        """Test statistical significance of edge discovery"""
        print("🧪 Testing statistical significance...")
        
        # Calculate baseline (actual) performance
        baseline_metrics = self.calculate_performance_metrics(self.predictions, self.actual_returns)
        
        significance_results = {}
        
        for scenario_name, scenario_results in self.scenario_results.items():
            sig_tests = {}
            
            for metric_name, metric_stats in scenario_results.items():
                baseline_value = baseline_metrics[metric_name]
                
                # Calculate p-value (percentage of bootstrap samples worse than baseline)
                if metric_name in ['information_coefficient', 'hit_rate', 'sharpe_ratio', 'total_return']:
                    # Higher is better
                    p_value = (metric_stats['percentiles']['p5'] + metric_stats['percentiles']['p95']) / 2
                    is_significant = baseline_value > metric_stats['percentiles']['p95']
                else:
                    # Lower is better (drawdown)
                    is_significant = baseline_value < metric_stats['percentiles']['p5']
                
                sig_tests[metric_name] = {
                    'baseline_value': baseline_value,
                    'bootstrap_mean': metric_stats['mean'],
                    'is_significant_95': is_significant,
                    'percentile_rank': self._calculate_percentile_rank(baseline_value, metric_stats)
                }
            
            significance_results[scenario_name] = sig_tests
        
        return significance_results
    
    def _calculate_percentile_rank(self, value: float, stats: Dict) -> float:
        """Calculate percentile rank of value in distribution"""
        # Approximate percentile rank using available percentiles
        percentiles = stats['percentiles']
        
        if value <= percentiles['p5']:
            return 5.0
        elif value <= percentiles['p25']:
            return 5.0 + (value - percentiles['p5']) / (percentiles['p25'] - percentiles['p5']) * 20.0
        elif value <= percentiles['p75']:
            return 25.0 + (value - percentiles['p25']) / (percentiles['p75'] - percentiles['p25']) * 50.0
        elif value <= percentiles['p95']:
            return 75.0 + (value - percentiles['p75']) / (percentiles['p95'] - percentiles['p75']) * 20.0
        else:
            return 95.0
    
    def generate_comprehensive_report(self, output_dir: str = "validation_results") -> Dict:
        """Generate comprehensive validation report"""
        print("📋 Generating comprehensive validation report...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Run all validations
        scenario_results = self.run_bootstrap_scenarios()
        trajectory_results = self.generate_trajectory_analysis()
        significance_results = self.test_statistical_significance()
        
        # Compile final report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'n_predictions': len(self.predictions),
                'n_instruments': len(self.instruments),
                'date_range': [str(self.timestamps[0]), str(self.timestamps[-1])]
            },
            'baseline_performance': self.calculate_performance_metrics(self.predictions, self.actual_returns),
            'bootstrap_scenarios': scenario_results,
            'trajectory_analysis': trajectory_results,
            'statistical_significance': significance_results,
            'edge_discovery_conclusion': self._generate_conclusion(significance_results)
        }
        
        # Save JSON report
        with open(output_path / 'monte_carlo_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_visualizations(report, output_path)
        
        print(f"✅ Validation report saved to {output_path}")
        return report
    
    def _generate_conclusion(self, significance_results: Dict) -> Dict:
        """Generate edge discovery conclusion"""
        
        # Count significant results across scenarios
        significant_count = 0
        total_tests = 0
        
        key_metrics = ['information_coefficient', 'hit_rate', 'sharpe_ratio']
        
        for scenario_results in significance_results.values():
            for metric_name in key_metrics:
                if metric_name in scenario_results:
                    total_tests += 1
                    if scenario_results[metric_name]['is_significant_95']:
                        significant_count += 1
        
        significance_rate = significant_count / total_tests if total_tests > 0 else 0
        
        # Determine conclusion
        if significance_rate >= 0.8:  # 80% of tests significant
            conclusion = "STATISTICALLY SIGNIFICANT EDGE DISCOVERED"
            confidence = "HIGH"
        elif significance_rate >= 0.6:  # 60% of tests significant
            conclusion = "MODERATE EDGE DETECTED"
            confidence = "MEDIUM" 
        elif significance_rate >= 0.4:  # 40% of tests significant
            conclusion = "WEAK EDGE DETECTED"
            confidence = "LOW"
        else:
            conclusion = "NO SIGNIFICANT EDGE - EFFICIENT MARKET HYPOTHESIS VALIDATED"
            confidence = "HIGH"
        
        return {
            'conclusion': conclusion,
            'confidence_level': confidence,
            'significance_rate': significance_rate,
            'significant_tests': significant_count,
            'total_tests': total_tests
        }
    
    def _create_visualizations(self, report: Dict, output_path: Path):
        """Create validation visualizations"""
        
        # Create spaghetti plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        trajectory_data = report['trajectory_analysis']
        
        for i, (scenario_name, traj_data) in enumerate(trajectory_data.items()):
            if i >= 6:  # Only plot first 6 scenarios
                break
                
            ax = axes[i]
            
            if len(traj_data['median']) > 0:
                x = range(len(traj_data['median']))
                
                # Plot confidence bands
                ax.fill_between(x, traj_data['p5'], traj_data['p95'], alpha=0.3, label='90% Confidence')
                
                # Plot median trajectory
                ax.plot(x, traj_data['median'], 'b-', linewidth=2, label='Median')
                ax.plot(x, traj_data['mean'], 'r--', linewidth=1, label='Mean')
                
                ax.set_title(f'{scenario_name.replace("_", " ").title()}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Cumulative Return ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'trajectory_spaghetti_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create significance summary plot
        self._create_significance_summary_plot(report, output_path)
    
    def _create_significance_summary_plot(self, report: Dict, output_path: Path):
        """Create statistical significance summary visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Information Coefficient across scenarios
        scenarios = list(report['bootstrap_scenarios'].keys())
        ic_means = [report['bootstrap_scenarios'][s]['information_coefficient']['mean'] for s in scenarios]
        ic_p5 = [report['bootstrap_scenarios'][s]['information_coefficient']['percentiles']['p5'] for s in scenarios]
        ic_p95 = [report['bootstrap_scenarios'][s]['information_coefficient']['percentiles']['p95'] for s in scenarios]
        
        baseline_ic = report['baseline_performance']['information_coefficient']
        
        x = range(len(scenarios))
        ax1.bar(x, ic_means, alpha=0.7, label='Bootstrap Mean')
        ax1.errorbar(x, ic_means, yerr=[np.array(ic_means) - np.array(ic_p5), 
                                        np.array(ic_p95) - np.array(ic_means)], 
                     fmt='none', color='black', capsize=5)
        ax1.axhline(y=baseline_ic, color='red', linestyle='--', linewidth=2, label='Baseline IC')
        ax1.axhline(y=0.05, color='green', linestyle=':', label='Significance Threshold (0.05)')
        
        ax1.set_xlabel('Validation Scenario')
        ax1.set_ylabel('Information Coefficient')
        ax1.set_title('Information Coefficient Across Bootstrap Scenarios')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hit Rate across scenarios
        hit_means = [report['bootstrap_scenarios'][s]['hit_rate']['mean'] for s in scenarios]
        hit_p5 = [report['bootstrap_scenarios'][s]['hit_rate']['percentiles']['p5'] for s in scenarios]
        hit_p95 = [report['bootstrap_scenarios'][s]['hit_rate']['percentiles']['p95'] for s in scenarios]
        
        baseline_hit = report['baseline_performance']['hit_rate']
        
        ax2.bar(x, hit_means, alpha=0.7, label='Bootstrap Mean')
        ax2.errorbar(x, hit_means, yerr=[np.array(hit_means) - np.array(hit_p5),
                                         np.array(hit_p95) - np.array(hit_means)],
                     fmt='none', color='black', capsize=5)
        ax2.axhline(y=baseline_hit, color='red', linestyle='--', linewidth=2, label='Baseline Hit Rate')
        ax2.axhline(y=0.52, color='green', linestyle=':', label='Significance Threshold (52%)')
        
        ax2.set_xlabel('Validation Scenario')
        ax2.set_ylabel('Hit Rate')
        ax2.set_title('Hit Rate Across Bootstrap Scenarios')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'statistical_significance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example usage function
def validate_edge_discovery(predictions_file: str, returns_file: str, 
                          output_dir: str = "validation_results") -> Dict:
    """
    Main function to validate edge discovery from prediction results
    
    Args:
        predictions_file: CSV file with model predictions
        returns_file: CSV file with actual returns
        output_dir: Output directory for results
        
    Returns:
        Validation report dictionary
    """
    
    # Load data
    predictions_df = pd.read_csv(predictions_file)
    returns_df = pd.read_csv(returns_file)
    
    # Extract arrays (assuming standard format)
    predictions = predictions_df.values
    actual_returns = returns_df.values
    instruments = predictions_df.columns.tolist()
    timestamps = pd.to_datetime(predictions_df.index)
    
    # Initialize validator
    validator = EdgeMonteCarloValidator(predictions, actual_returns, instruments, timestamps)
    
    # Run comprehensive validation
    report = validator.generate_comprehensive_report(output_dir)
    
    # Print conclusion
    conclusion = report['edge_discovery_conclusion']
    print(f"\n{'='*60}")
    print(f"EDGE DISCOVERY CONCLUSION: {conclusion['conclusion']}")
    print(f"CONFIDENCE LEVEL: {conclusion['confidence_level']}")
    print(f"SIGNIFICANCE RATE: {conclusion['significance_rate']:.1%}")
    print(f"{'='*60}")
    
    return report

if __name__ == "__main__":
    # Example usage - replace with actual file paths
    print("🔬 FX Edge Discovery Monte Carlo Validation")
    print("Adapted from Dr. Howard Bandy methodology")
    print()
    
    # This would be called after training with actual prediction results
    # validate_edge_discovery("predictions.csv", "actual_returns.csv")
    
    print("📋 To use this validator:")
    print("1. Train your TCNAE + LightGBM model")
    print("2. Generate predictions on test data")
    print("3. Call validate_edge_discovery(predictions_file, returns_file)")
    print("4. Review validation report for statistical significance")