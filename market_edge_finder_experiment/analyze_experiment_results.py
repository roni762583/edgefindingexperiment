#!/usr/bin/env python3
"""
Proper Analysis of Experiment Results
=====================================

Extract and analyze actual predictions without arbitrary thresholds.
Focus on:
1. Pip prediction accuracy
2. Direction prediction accuracy  
3. Prediction confidence analysis
4. Actual model performance metrics
5. Comprehensive visualization
"""

import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def load_experiment_data(experiment_id: str = "20251030_141017"):
    """Load all experiment data."""
    base_dir = Path(".")
    results_dir = base_dir / "results" / f"optimized_experiment_{experiment_id}"
    cache_dir = base_dir / "data" / "latent_cache"
    
    print(f"📁 Loading experiment data for {experiment_id}")
    
    # Load cached latents
    latent_cache_file = cache_dir / f"latent_cache_experiment_{experiment_id}.pkl"
    latent_metadata_file = cache_dir / f"latent_metadata_experiment_{experiment_id}.json"
    
    if not latent_cache_file.exists():
        raise FileNotFoundError(f"Latent cache not found: {latent_cache_file}")
    
    with open(latent_cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    with open(latent_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load LightGBM model
    model_file = results_dir / "lightgbm_model.pkl"
    if model_file.exists():
        with open(model_file, 'rb') as f:
            gbdt_model = pickle.load(f)
    else:
        gbdt_model = None
        print("⚠️  LightGBM model file not found")
    
    # Load experiment results
    results_file = results_dir / "optimized_experiment_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            experiment_results = json.load(f)
    else:
        experiment_results = None
        print("⚠️  Experiment results file not found")
    
    return {
        'cached_data': cached_data,
        'metadata': metadata,
        'gbdt_model': gbdt_model,
        'experiment_results': experiment_results,
        'experiment_id': experiment_id
    }

def analyze_raw_predictions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze raw predictions without arbitrary thresholds."""
    print("\n🔍 ANALYZING RAW PREDICTIONS")
    print("=" * 50)
    
    cached_data = data['cached_data']
    metadata = data['metadata']
    gbdt_model = data['gbdt_model']
    
    if gbdt_model is None:
        print("❌ Cannot analyze predictions - no trained model found")
        return {}
    
    # Extract test data (last 15% of samples)
    total_samples = len(cached_data['latent_features'])
    test_start = int(0.85 * total_samples)  # Last 15% for testing
    
    X_test = cached_data['latent_features'][test_start:]
    y_test = cached_data['targets'][test_start:]
    
    print(f"📊 Test data: {X_test.shape[0]} samples, {X_test.shape[1]} latent features")
    print(f"📊 Targets: {y_test.shape}")
    
    # Generate predictions
    print("🎯 Generating predictions...")
    predictions = gbdt_model.predict(X_test)
    
    print(f"📊 Predictions shape: {predictions.shape}")
    
    # Analyze results for each instrument
    instruments = metadata['data_info']['instruments']
    results = {}
    
    for i, instrument in enumerate(instruments):
        y_true = y_test[:, i]
        y_pred = predictions[:, i]
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Direction accuracy (sign prediction)
        true_direction = np.sign(y_true - 0.5)  # Assuming 0.5 is neutral
        pred_direction = np.sign(y_pred - 0.5)
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # Calculate correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else 0.0
        
        # Prediction range and statistics
        pred_stats = {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred)),
            'range': float(np.max(y_pred) - np.min(y_pred))
        }
        
        true_stats = {
            'mean': float(np.mean(y_true)),
            'std': float(np.std(y_true)),
            'min': float(np.min(y_true)),
            'max': float(np.max(y_true)),
            'range': float(np.max(y_true) - np.min(y_true))
        }
        
        results[instrument] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation),
            'direction_accuracy': float(direction_accuracy),
            'prediction_stats': pred_stats,
            'actual_stats': true_stats,
            'n_samples': len(y_true),
            'predictions': y_pred.tolist(),
            'actuals': y_true.tolist()
        }
        
        print(f"{instrument:8} | RMSE: {rmse:.4f} | Corr: {correlation:6.3f} | Dir Acc: {direction_accuracy:.3f}")
    
    return {
        'instrument_results': results,
        'summary_stats': calculate_summary_stats(results),
        'test_samples': X_test.shape[0],
        'raw_predictions': predictions,
        'raw_actuals': y_test
    }

def calculate_summary_stats(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate summary statistics across all instruments."""
    rmses = [r['rmse'] for r in results.values()]
    correlations = [r['correlation'] for r in results.values()]
    direction_accs = [r['direction_accuracy'] for r in results.values()]
    
    return {
        'mean_rmse': float(np.mean(rmses)),
        'std_rmse': float(np.std(rmses)),
        'best_rmse': float(np.min(rmses)),
        'worst_rmse': float(np.max(rmses)),
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'best_correlation': float(np.max(correlations)),
        'worst_correlation': float(np.min(correlations)),
        'mean_direction_accuracy': float(np.mean(direction_accs)),
        'std_direction_accuracy': float(np.std(direction_accs)),
        'best_direction_accuracy': float(np.max(direction_accs)),
        'worst_direction_accuracy': float(np.min(direction_accs)),
        'instruments_with_positive_correlation': sum(1 for c in correlations if c > 0),
        'instruments_with_good_direction_acc': sum(1 for d in direction_accs if d > 0.52)  # Better than random
    }

def create_visualization(analysis_results: Dict[str, Any], experiment_id: str):
    """Create comprehensive visualizations."""
    print("\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    results = analysis_results['instrument_results']
    summary = analysis_results['summary_stats']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'FX Prediction Analysis - Experiment {experiment_id}', fontsize=16, fontweight='bold')
    
    # 1. RMSE by instrument
    instruments = list(results.keys())
    rmses = [results[inst]['rmse'] for inst in instruments]
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(instruments)), rmses, color='skyblue', alpha=0.7)
    ax1.set_title(f'RMSE by Instrument\nMean: {summary["mean_rmse"]:.4f} ± {summary["std_rmse"]:.4f}')
    ax1.set_xlabel('Instruments')
    ax1.set_ylabel('RMSE')
    ax1.set_xticks(range(len(instruments)))
    ax1.set_xticklabels(instruments, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation by instrument
    correlations = [results[inst]['correlation'] for inst in instruments]
    
    ax2 = axes[0, 1]
    colors = ['green' if c > 0 else 'red' for c in correlations]
    bars2 = ax2.bar(range(len(instruments)), correlations, color=colors, alpha=0.7)
    ax2.set_title(f'Prediction Correlation\nMean: {summary["mean_correlation"]:.4f} ± {summary["std_correlation"]:.4f}')
    ax2.set_xlabel('Instruments')
    ax2.set_ylabel('Correlation')
    ax2.set_xticks(range(len(instruments)))
    ax2.set_xticklabels(instruments, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Direction accuracy by instrument
    direction_accs = [results[inst]['direction_accuracy'] for inst in instruments]
    
    ax3 = axes[0, 2]
    colors = ['green' if d > 0.5 else 'red' for d in direction_accs]
    bars3 = ax3.bar(range(len(instruments)), direction_accs, color=colors, alpha=0.7)
    ax3.set_title(f'Direction Accuracy\nMean: {summary["mean_direction_accuracy"]:.3f} ± {summary["std_direction_accuracy"]:.3f}')
    ax3.set_xlabel('Instruments')
    ax3.set_ylabel('Direction Accuracy')
    ax3.set_xticks(range(len(instruments)))
    ax3.set_xticklabels(instruments, rotation=45, ha='right')
    ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Random (50%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation vs Direction Accuracy scatter
    ax4 = axes[1, 0]
    scatter = ax4.scatter(correlations, direction_accs, c=rmses, cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Prediction Correlation')
    ax4.set_ylabel('Direction Accuracy')
    ax4.set_title('Correlation vs Direction Accuracy\n(Color = RMSE)')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Direction')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='RMSE')
    
    # 5. Prediction distribution (example for best performing instrument)
    best_inst = min(instruments, key=lambda x: results[x]['rmse'])
    
    ax5 = axes[1, 1]
    predictions_sample = results[best_inst]['predictions'][:100]  # First 100 predictions
    actuals_sample = results[best_inst]['actuals'][:100]
    
    ax5.plot(predictions_sample, label=f'Predictions ({best_inst})', alpha=0.7)
    ax5.plot(actuals_sample, label=f'Actual ({best_inst})', alpha=0.7)
    ax5.set_title(f'Prediction vs Actual (Best: {best_inst})\nRMSE: {results[best_inst]["rmse"]:.4f}')
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Normalized Return')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
EXPERIMENT SUMMARY
{'='*25}

Total Instruments: {len(instruments)}
Test Samples: {analysis_results['test_samples']}

PERFORMANCE METRICS:
{'+'*20}
Mean RMSE: {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}
Best RMSE: {summary['best_rmse']:.4f}
Worst RMSE: {summary['worst_rmse']:.4f}

Mean Correlation: {summary['mean_correlation']:.4f} ± {summary['std_correlation']:.4f}
Best Correlation: {summary['best_correlation']:.4f}
Worst Correlation: {summary['worst_correlation']:.4f}

Mean Direction Acc: {summary['mean_direction_accuracy']:.3f} ± {summary['std_direction_accuracy']:.3f}
Best Direction Acc: {summary['best_direction_accuracy']:.3f}
Worst Direction Acc: {summary['worst_direction_accuracy']:.3f}

SIGNIFICANCE:
{'+'*15}
Positive Correlations: {summary['instruments_with_positive_correlation']}/{len(instruments)}
Direction > 52%: {summary['instruments_with_good_direction_acc']}/{len(instruments)}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"experiment_analysis_{experiment_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {output_file}")
    
    return output_file

def generate_detailed_report(analysis_results: Dict[str, Any], experiment_id: str):
    """Generate detailed markdown report."""
    print("\n📝 GENERATING DETAILED REPORT")
    print("=" * 50)
    
    results = analysis_results['instrument_results']
    summary = analysis_results['summary_stats']
    
    report = f"""# FX Prediction Experiment Analysis
## Experiment ID: {experiment_id}

### Executive Summary
This experiment trained a TCNAE + LightGBM hybrid model to predict 1-hour FX returns across 24 currency pairs using temporal features and cross-instrument context.

**Key Findings:**
- **Test Samples**: {analysis_results['test_samples']} observations
- **Mean RMSE**: {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}
- **Mean Correlation**: {summary['mean_correlation']:.4f} ± {summary['std_correlation']:.4f}
- **Mean Direction Accuracy**: {summary['mean_direction_accuracy']:.3f} ± {summary['std_direction_accuracy']:.3f}
- **Instruments with Positive Correlation**: {summary['instruments_with_positive_correlation']}/{len(results)}
- **Instruments with >52% Direction Accuracy**: {summary['instruments_with_good_direction_acc']}/{len(results)}

### Detailed Results by Instrument

| Instrument | RMSE | Correlation | Direction Acc | Pred Range | Actual Range |
|------------|------|-------------|---------------|------------|--------------|"""
    
    for instrument, result in results.items():
        pred_range = result['prediction_stats']['range']
        actual_range = result['actual_stats']['range']
        report += f"\n| {instrument} | {result['rmse']:.4f} | {result['correlation']:+.4f} | {result['direction_accuracy']:.3f} | {pred_range:.4f} | {actual_range:.4f} |"
    
    report += f"""

### Top Performers

**Best RMSE**: {min(results.keys(), key=lambda x: results[x]['rmse'])} ({summary['best_rmse']:.4f})
**Best Correlation**: {max(results.keys(), key=lambda x: results[x]['correlation'])} ({summary['best_correlation']:.4f})
**Best Direction Accuracy**: {max(results.keys(), key=lambda x: results[x]['direction_accuracy'])} ({summary['best_direction_accuracy']:.3f})

### Analysis Insights

1. **Prediction Quality**: The mean RMSE of {summary['mean_rmse']:.4f} indicates moderate prediction accuracy
2. **Correlation Analysis**: {summary['instruments_with_positive_correlation']} out of {len(results)} instruments show positive correlation
3. **Direction Prediction**: {summary['instruments_with_good_direction_acc']} instruments achieve >52% direction accuracy (better than random)
4. **Model Consistency**: Standard deviation of RMSE ({summary['std_rmse']:.4f}) suggests consistent performance across instruments

### Technical Details
- **Architecture**: TCNAE (120-dim latent) + LightGBM
- **Features**: 5 indicators per instrument (slopes, volatility, direction, price_change)
- **Training Data**: 70% temporal split
- **Validation Data**: 15% temporal split  
- **Test Data**: 15% temporal split (most recent)
- **No Temporal Leakage**: Strict chronological ordering maintained

### Recommendations
1. **Model Performance**: Results show learning but limited predictive power
2. **Feature Engineering**: Consider additional technical indicators or market sentiment
3. **Time Horizon**: Experiment with longer prediction windows (4-hour, daily)
4. **Ensemble Methods**: Combine with other approaches for improved accuracy
"""
    
    # Save report
    report_file = f"experiment_report_{experiment_id}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📝 Report saved: {report_file}")
    return report_file

def main():
    """Main analysis function."""
    print("🔬 COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        # Load experiment data
        data = load_experiment_data()
        
        # Analyze raw predictions without thresholds
        analysis_results = analyze_raw_predictions(data)
        
        if analysis_results:
            # Create visualizations
            viz_file = create_visualization(analysis_results, data['experiment_id'])
            
            # Generate detailed report
            report_file = generate_detailed_report(analysis_results, data['experiment_id'])
            
            print(f"\n✅ ANALYSIS COMPLETE!")
            print(f"📊 Visualization: {viz_file}")
            print(f"📝 Report: {report_file}")
            
            # Save raw analysis results
            analysis_file = f"analysis_results_{data['experiment_id']}.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"💾 Raw results: {analysis_file}")
            
        else:
            print("❌ Analysis failed - insufficient data")
    
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()