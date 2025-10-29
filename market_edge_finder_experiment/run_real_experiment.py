#!/usr/bin/env python3
"""
Real Data Experiment Runner

Run the complete Market Edge Finder experiment using existing downloaded data.
Bypasses Docker to test the full pipeline with real OANDA data.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_data(data_dir: Path):
    """Load real processed feature data from CSV files."""
    
    processed_dir = data_dir / "processed"
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    # Get all processed feature files
    feature_files = list(processed_dir.glob("*_H1_precomputed_features.csv"))
    
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in {processed_dir}")
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Load data from CSV files
    instruments = []
    feature_data = []
    
    for file_path in sorted(feature_files):  # Use all 24 instruments as specified
        # Extract instrument name from filename
        instrument = file_path.stem.replace("_H1_precomputed_features", "")
        instruments.append(instrument)
        
        logger.info(f"Loading {instrument} from {file_path.name}")
        
        # Load CSV data
        df = pd.read_csv(file_path)
        
        # Expected columns: slope_high, slope_low, volatility, direction, price_change
        expected_cols = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change']
        
        # Select feature columns (skip timestamp columns)
        feature_cols = [col for col in expected_cols if col in df.columns]
        
        if not feature_cols:
            logger.warning(f"No expected feature columns found in {instrument}, using all numeric columns")
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Take first 5
        
        # Extract features for this instrument
        instrument_features = df[feature_cols].values  # [n_samples, n_features]
        
        # Handle NaN values
        instrument_features = np.nan_to_num(instrument_features, nan=0.0)
        
        feature_data.append(instrument_features)
        
        logger.info(f"Loaded {instrument}: {instrument_features.shape} features: {feature_cols}")
    
    # Find minimum length across all instruments (for alignment)
    min_length = min(len(data) for data in feature_data)
    logger.info(f"Aligning data to minimum length: {min_length}")
    
    # Use a reasonable subset for testing (last 2000 points)
    test_length = min(2000, min_length)
    
    # Truncate all instruments to same length and stack
    aligned_features = []
    for data in feature_data:
        aligned_features.append(data[-test_length:])  # Take last test_length points
    
    # Stack into [n_samples, n_instruments, n_features]
    features = np.stack(aligned_features, axis=1)
    
    logger.info(f"Final features shape: {features.shape}")
    logger.info(f"Instruments: {instruments}")
    
    # Generate targets based on price changes (future returns)
    targets = np.zeros((test_length, len(instruments)))
    
    for i, instrument in enumerate(instruments):
        # Load raw price data to calculate returns
        raw_file = data_dir / "raw" / f"{instrument}_3years_H1.csv"
        if raw_file.exists():
            raw_df = pd.read_csv(raw_file)
            if 'close' in raw_df.columns:
                prices = raw_df['close'].values[-test_length-1:]  # Need one extra for returns
                returns = np.diff(np.log(prices))  # Log returns
                targets[:, i] = returns
                logger.info(f"{instrument}: Generated {len(returns)} return targets")
            else:
                # Fallback: use price_change feature if available
                if len(feature_cols) >= 5:
                    targets[:, i] = features[:, i, -1]  # Last feature
                else:
                    targets[:, i] = np.random.normal(0, 0.001, test_length)
        else:
            logger.warning(f"Raw data file not found for {instrument}, using synthetic targets")
            targets[:, i] = np.random.normal(0, 0.001, test_length)
    
    logger.info(f"Targets shape: {targets.shape}")
    logger.info(f"Target stats - Mean: {np.mean(targets):.6f}, Std: {np.std(targets):.6f}")
    
    return features, targets, instruments


def run_basic_experiment():
    """Run basic experiment with real data."""
    
    logger.info("🚀 Starting Real Data Market Edge Finder Experiment")
    start_time = datetime.now()
    
    try:
        # Load real data
        data_dir = Path("data")
        features, targets, instruments = load_real_data(data_dir)
        
        logger.info(f"Data loaded: {features.shape} features, {targets.shape} targets")
        
        # Simple validation: Calculate correlations between features and targets
        correlations = {}
        for i, instrument in enumerate(instruments):
            feature_mean = np.mean(features[:, i, :], axis=1)  # Average features per timestep
            target_values = targets[:, i]
            
            # Remove NaN/inf values
            valid_mask = np.isfinite(feature_mean) & np.isfinite(target_values)
            
            if valid_mask.sum() > 10:
                corr = np.corrcoef(feature_mean[valid_mask], target_values[valid_mask])[0, 1]
                correlations[instrument] = corr
                logger.info(f"{instrument}: Feature-Target correlation = {corr:.4f}")
            else:
                correlations[instrument] = 0.0
                logger.warning(f"{instrument}: Insufficient valid data for correlation")
        
        # Identify potential edges (high correlations)
        edge_threshold = 0.03  # 3% correlation threshold (more sensitive)
        potential_edges = {k: v for k, v in correlations.items() if abs(v) >= edge_threshold}
        
        # Generate results
        results = {
            'experiment_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'instruments': instruments,
                'n_samples': int(features.shape[0]),
                'n_features': int(features.shape[2]),
                'data_period': '2000 most recent bars'
            },
            'correlations': correlations,
            'potential_edges': potential_edges,
            'summary': {
                'total_instruments': len(instruments),
                'potential_edges_found': len(potential_edges),
                'max_correlation': float(max(correlations.values())) if correlations else 0.0,
                'edge_discovery_rate': len(potential_edges) / len(instruments) if instruments else 0.0
            }
        }
        
        # Conclusion
        if potential_edges:
            conclusion = f"POTENTIAL EDGES DETECTED: {len(potential_edges)} instruments show correlation ≥{edge_threshold:.1%}"
            recommendation = "Proceed with full TCNAE + LightGBM training and Monte Carlo validation"
        else:
            conclusion = f"NO SIGNIFICANT EDGES: No correlations exceed {edge_threshold:.1%} threshold"
            recommendation = "Consider feature engineering improvements or alternative approaches"
        
        results['conclusion'] = conclusion
        results['recommendation'] = recommendation
        
        # Save results
        results_file = Path("results") / f"real_data_experiment_{results['experiment_id']}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        duration = datetime.now() - start_time
        
        print("\n" + "="*80)
        print("REAL DATA MARKET EDGE FINDER EXPERIMENT RESULTS")
        print("="*80)
        print(f"Duration: {duration}")
        print(f"Data: {features.shape[0]} samples × {len(instruments)} instruments × {features.shape[2]} features")
        print(f"Period: Last 2000 hourly bars (~3 months)")
        print()
        print("Feature-Target Correlations:")
        for instrument, corr in correlations.items():
            status = "🟢 EDGE" if abs(corr) >= edge_threshold else "⚪ NOISE"
            print(f"  {instrument}: {corr:+.4f} {status}")
        print()
        print(f"Conclusion: {conclusion}")
        print(f"Recommendation: {recommendation}")
        print(f"Results saved: {results_file}")
        print("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        # Check if we have the required data
        data_dir = Path("data")
        if not data_dir.exists():
            print("❌ Data directory not found. Please ensure you're in the project root.")
            sys.exit(1)
        
        processed_dir = data_dir / "processed"
        if not processed_dir.exists() or not list(processed_dir.glob("*.csv")):
            print("❌ No processed data found. Please run feature generation first.")
            sys.exit(1)
        
        # Run experiment
        results = run_basic_experiment()
        
        if results:
            print("\n✅ Real data experiment completed successfully!")
        else:
            print("\n❌ Experiment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)