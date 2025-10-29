# Monte Carlo Edge Discovery Validation

This module implements rigorous statistical validation for FX prediction edges using Monte Carlo methodology adapted from Dr. Howard Bandy's approach.

## Purpose

Determine whether discovered edges are **statistically significant** or merely random patterns that would fail in live trading.

## Files

### Core Validation Scripts
- `edge_discovery_monte_carlo.py` - **Main validation framework** (adapted for edge discovery)
- `bootstrap_monte_carlo.py` - Original new_swt bootstrap validation
- `real_bootstrap_validation.py` - Real trading session validation
- `bootstrap_spaghetti.py` - Trajectory visualization tools

### Key Features

#### 6-Scenario Stress Testing
1. **Original Bootstrap**: Standard bootstrap with replacement
2. **Random 10% Drop**: Random 10% prediction removal
3. **Tail 20% Drop**: Remove worst 20% of predictions  
4. **150% Oversample**: Oversampling stress test
5. **Adverse Selection**: Bias towards losing predictions
6. **Early Stop 80%**: Stop at 80% of data

#### Statistical Metrics
- **Information Coefficient**: Correlation between predictions and returns
- **Hit Rate**: Directional accuracy (>52% is significant)
- **Sharpe Ratio**: Risk-adjusted returns (pure edge measurement)
- **Maximum Drawdown**: Worst-case scenario analysis

#### Visualization
- **Spaghetti Plots**: Equity trajectory visualization with confidence bands
- **Significance Summary**: Statistical significance across all scenarios
- **Confidence Intervals**: 5th, 25th, 75th, 95th percentile analysis

## Usage

### After Model Training

```python
from validation import validate_edge_discovery

# Run comprehensive Monte Carlo validation
report = validate_edge_discovery(
    predictions_file="model_predictions.csv",
    returns_file="actual_returns.csv", 
    output_dir="validation_results"
)

print(f"Edge Discovery: {report['edge_discovery_conclusion']['conclusion']}")
```

### Direct API Usage

```python
from validation import EdgeMonteCarloValidator
import pandas as pd

# Load your prediction results
predictions = model.predict(test_data)  # Shape: (n_samples, n_instruments)
actual_returns = test_returns.values    # USD-scaled pip returns

# Initialize validator
validator = EdgeMonteCarloValidator(
    predictions=predictions,
    actual_returns=actual_returns,
    instruments=['EUR_USD', 'GBP_USD', ...],
    timestamps=test_data.index
)

# Run validation
report = validator.generate_comprehensive_report()
```

## Success Criteria

An edge is considered **statistically significant** if:

1. **Information Coefficient** >0.05 across >80% of bootstrap scenarios
2. **Hit Rate** >52% with 95% confidence intervals excluding 50%
3. **Statistical Significance** p-value <0.05 in key metrics
4. **Cross-Scenario Robustness** edge persists across adverse conditions

## Interpretation

### Edge Discovered ✅
- **Significance Rate** >80%: Strong evidence of exploitable edge
- **Significance Rate** 60-80%: Moderate edge, requires careful implementation  
- **Significance Rate** 40-60%: Weak edge, marginally exploitable

### No Edge Found ❌
- **Significance Rate** <40%: No significant edge detected
- **Conclusion**: Efficient Market Hypothesis validated for this timeframe
- **Implication**: Sophisticated algorithms cannot overcome market efficiency

## Output Files

### JSON Report
- `monte_carlo_validation_report.json` - Comprehensive results
- All metrics, confidence intervals, and statistical tests

### Visualizations  
- `trajectory_spaghetti_plots.png` - Equity curve trajectories
- `statistical_significance_summary.png` - Significance across scenarios

## Integration with Training Pipeline

The validation framework integrates with the main training pipeline:

```python
# After training TCNAE + LightGBM
predictions = hybrid_model.predict(test_data)

# Convert to format for validation
predictions_normalized = scaler.transform(predictions)  # [-1, +1] range
actual_returns_usd = calculate_usd_pip_returns(test_data)

# Validate edge discovery
edge_report = validate_edge_discovery(predictions_normalized, actual_returns_usd)

if edge_report['edge_discovery_conclusion']['conclusion'].startswith('STATISTICALLY SIGNIFICANT'):
    print("🎉 EDGE DISCOVERED - Proceed with implementation")
else:
    print("❌ NO EDGE FOUND - Market appears efficient at this timeframe")
```

## Reference Implementation

Based on methodology from:
- **Repository**: https://github.com/roni762583/new_swt.git
- **Approach**: Dr. Howard Bandy statistical validation techniques
- **Adaptation**: Modified for FX prediction edge discovery vs trading system validation

This framework provides the statistical rigor needed to distinguish genuine predictive edges from statistical noise and overfitting artifacts.