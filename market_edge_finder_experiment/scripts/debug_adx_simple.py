#!/usr/bin/env python3
"""
Simple ADX debug - why is batch returning all NaN?
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def debug_adx_simple():
    """Simple check of batch ADX calculation"""
    
    # Load data - use 2000-bar slice like the correlation test
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same slice as correlation test
    test_slice = df.iloc[1666:3666].copy()
    print(f"🔍 Batch ADX Debug: {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Run batch processing
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    print(f"\n📊 Batch Results Analysis:")
    print(f"Columns: {list(batch_results.columns)}")
    
    # Check each column for NaN counts
    for col in batch_results.columns:
        nan_count = batch_results[col].isna().sum()
        total_count = len(batch_results[col])
        valid_count = total_count - nan_count
        print(f"{col:15s}: {valid_count:4d} valid / {total_count:4d} total ({valid_count/total_count*100:5.1f}%)")
        
        if col == 'direction' and valid_count > 0:
            valid_values = batch_results[col].dropna()
            print(f"                Direction range: {valid_values.min():.6f} to {valid_values.max():.6f}")
            print(f"                First 10 valid: {valid_values.head(10).values}")
    
    # Check if ADX column exists
    if 'direction' in batch_results.columns:
        direction_values = batch_results['direction']
        print(f"\nDirection column stats:")
        print(f"  NaN count: {direction_values.isna().sum()}")
        print(f"  Valid count: {(~direction_values.isna()).sum()}")
        
        if (~direction_values.isna()).sum() > 0:
            valid_dir = direction_values.dropna()
            print(f"  Min: {valid_dir.min():.6f}")
            print(f"  Max: {valid_dir.max():.6f}")
            print(f"  Mean: {valid_dir.mean():.6f}")
        else:
            print(f"  ❌ ALL VALUES ARE NaN!")
    
    # Look for raw ADX before scaling
    print(f"\n🔍 Looking for raw ADX calculation...")
    
    # Check if there are any non-NaN numeric columns
    numeric_cols = batch_results.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        non_nan = (~batch_results[col].isna()).sum()
        if non_nan > 0:
            print(f"  {col}: {non_nan} valid values")
    
    return batch_results

if __name__ == "__main__":
    debug_adx_simple()