#!/usr/bin/env python3
"""
Test correlation between practical batch and practical incremental methods
This should achieve high correlations for all indicators including slopes
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.implement_practical_method import create_practical_batch_processor
from features.practical_incremental import update_practical_indicators, PracticalMultiInstrumentState

def test_practical_correlation():
    """Test practical method correlation between batch and incremental"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use smaller slice for testing - 500 bars for faster iteration
    test_slice = df.iloc[1666:2166].copy()
    print(f"🎯 Testing Practical Method Correlation: {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Batch processing with practical method
    practical_processor = create_practical_batch_processor()
    batch_results = practical_processor(test_slice, "EUR_USD")
    
    print(f"\n📊 Batch Results:")
    batch_hsp = batch_results['sig_hsp'].sum()
    batch_lsp = batch_results['sig_lsp'].sum()
    print(f"HSP: {batch_hsp}, LSP: {batch_lsp}, Total: {batch_hsp + batch_lsp}")
    
    # Incremental processing with practical method
    multi_state = PracticalMultiInstrumentState()
    incremental_results = []
    
    print(f"\n🔄 Running incremental processing...")
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        indicators, multi_state = update_practical_indicators(new_ohlc, multi_state, 'EUR_USD')
        incremental_results.append(indicators)
        
        if i % 100 == 0:
            print(f"  Processed {i+1}/{len(test_slice)} bars...")
    
    # Convert incremental results to DataFrame
    incremental_df = pd.DataFrame(incremental_results)
    
    print(f"\n📊 Incremental Results:")
    state = multi_state.get_instrument_state('EUR_USD')
    incremental_hsp = len(state.hsp_indices)
    incremental_lsp = len(state.lsp_indices)
    print(f"HSP: {incremental_hsp}, LSP: {incremental_lsp}, Total: {incremental_hsp + incremental_lsp}")
    
    # Compare swing counts
    print(f"\n🔍 Swing Count Comparison:")
    print(f"Batch:       HSP={batch_hsp:3d}, LSP={batch_lsp:3d}, Total={batch_hsp + batch_lsp:3d}")
    print(f"Incremental: HSP={incremental_hsp:3d}, LSP={incremental_lsp:3d}, Total={incremental_hsp + incremental_lsp:3d}")
    
    swing_diff = abs((batch_hsp + batch_lsp) - (incremental_hsp + incremental_lsp))
    print(f"Difference: {swing_diff} swings")
    
    # Calculate correlations
    print(f"\n📈 Correlation Analysis:")
    
    common_columns = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change']
    correlations = {}
    
    for col in common_columns:
        batch_values = batch_results[col].values
        incremental_values = incremental_df[col].values
        
        # Find valid data points
        valid_mask = ~(np.isnan(batch_values) | np.isnan(incremental_values))
        valid_batch = batch_values[valid_mask]
        valid_incremental = incremental_values[valid_mask]
        
        if len(valid_batch) > 1:
            correlation = np.corrcoef(valid_batch, valid_incremental)[0, 1]
            mean_diff = np.mean(np.abs(valid_batch - valid_incremental))
            max_diff = np.max(np.abs(valid_batch - valid_incremental))
            
            correlations[col] = correlation
            
            status = "✅" if correlation > 0.95 else "⚠️" if correlation > 0.80 else "❌"
            print(f"{col:12s}: corr={correlation:6.3f} {status}, mean_diff={mean_diff:.6f}, valid_points={len(valid_batch)}")
        else:
            print(f"{col:12s}: No valid data points for comparison")
            correlations[col] = np.nan
    
    # Coverage analysis
    print(f"\n📊 Coverage Analysis:")
    for col in common_columns:
        batch_coverage = (~batch_results[col].isna()).sum() / len(batch_results) * 100
        incremental_coverage = (~incremental_df[col].isna()).sum() / len(incremental_df) * 100
        print(f"{col:12s}: Batch={batch_coverage:5.1f}%, Incremental={incremental_coverage:5.1f}%")
    
    # Show some swing points for verification
    print(f"\n🎯 Swing Point Verification:")
    print(f"First 10 incremental swings:")
    
    all_incremental_swings = []
    for idx in state.hsp_indices:
        if idx < len(state.asi_history):
            all_incremental_swings.append((idx, 'HSP', state.asi_history[idx]))
    for idx in state.lsp_indices:
        if idx < len(state.asi_history):
            all_incremental_swings.append((idx, 'LSP', state.asi_history[idx]))
    
    all_incremental_swings.sort()
    
    for i, (bar, type_, asi) in enumerate(all_incremental_swings[:10]):
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Summary
    print(f"\n📋 Summary:")
    avg_correlation = np.nanmean(list(correlations.values()))
    print(f"Average correlation: {avg_correlation:.3f}")
    
    high_corr_count = sum(1 for c in correlations.values() if c > 0.95)
    print(f"High correlations (>95%): {high_corr_count}/{len(correlations)}")
    
    return correlations

if __name__ == "__main__":
    test_practical_correlation()