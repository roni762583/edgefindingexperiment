#!/usr/bin/env python3
"""
Deep dive into ADX correlation issue - identify specific algorithmic differences
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import MultiInstrumentState, update_indicators

def debug_adx_correlation():
    """Deep analysis of ADX calculation differences between batch and incremental"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use smaller slice for detailed analysis
    test_slice = df.iloc[1666:1766].copy()  # 100 bars for detailed analysis
    print(f"🔍 Deep ADX Analysis: 100 bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Batch processing
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # Incremental processing
    multi_state = MultiInstrumentState()
    incremental_results = []
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        indicators, multi_state = update_indicators(new_ohlc, multi_state, 'EUR_USD')
        incremental_results.append(indicators['direction'])
    
    # Compare ADX values step by step
    batch_direction = batch_results['direction'].values
    incremental_direction = np.array(incremental_results)
    
    print(f"\n📊 ADX Comparison Analysis:")
    print(f"Batch direction range: {np.nanmin(batch_direction):.3f} to {np.nanmax(batch_direction):.3f}")
    print(f"Incremental direction range: {np.nanmin(incremental_direction):.3f} to {np.nanmax(incremental_direction):.3f}")
    
    # Find where values are available in both
    valid_mask = ~(np.isnan(batch_direction) | np.isnan(incremental_direction))
    valid_batch = batch_direction[valid_mask]
    valid_incremental = incremental_direction[valid_mask]
    
    if len(valid_batch) > 0:
        correlation = np.corrcoef(valid_batch, valid_incremental)[0, 1]
        mean_diff = np.mean(np.abs(valid_batch - valid_incremental))
        max_diff = np.max(np.abs(valid_batch - valid_incremental))
        
        print(f"Valid samples: {len(valid_batch)}")
        print(f"Correlation: {correlation:.3f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
    
    # Detailed bar-by-bar analysis for first differences
    print(f"\n🔍 Bar-by-bar differences (first 20 valid):")
    print("Bar | Batch    | Incremental | Difference")
    print("----|----------|-------------|------------")
    
    count = 0
    for i in range(len(batch_direction)):
        if not (np.isnan(batch_direction[i]) or np.isnan(incremental_direction[i])):
            diff = abs(batch_direction[i] - incremental_direction[i])
            print(f"{i:3d} | {batch_direction[i]:8.5f} | {incremental_direction[i]:11.5f} | {diff:10.6f}")
            count += 1
            if count >= 20:
                break
    
    # Check for systematic patterns
    if len(valid_batch) > 0:
        print(f"\n📈 Pattern Analysis:")
        
        # Check if incremental is consistently higher/lower
        higher_count = np.sum(valid_incremental > valid_batch)
        lower_count = np.sum(valid_incremental < valid_batch)
        equal_count = np.sum(valid_incremental == valid_batch)
        
        print(f"Incremental > Batch: {higher_count} times ({higher_count/len(valid_batch)*100:.1f}%)")
        print(f"Incremental < Batch: {lower_count} times ({lower_count/len(valid_batch)*100:.1f}%)")
        print(f"Incremental = Batch: {equal_count} times ({equal_count/len(valid_batch)*100:.1f}%)")
        
        # Check for offset patterns
        batch_mean = np.mean(valid_batch)
        incremental_mean = np.mean(valid_incremental)
        print(f"Batch mean: {batch_mean:.6f}")
        print(f"Incremental mean: {incremental_mean:.6f}")
        print(f"Mean offset: {incremental_mean - batch_mean:.6f}")
    
    # Look at raw ADX values before percentile scaling
    print(f"\n🔧 Raw ADX Investigation:")
    
    # Access the raw ADX values from incremental state
    state = multi_state.get_instrument_state('EUR_USD')
    if len(state.adx_history) > 0:
        print(f"Incremental ADX history length: {len(state.adx_history)}")
        print(f"Last 10 raw ADX values: {state.adx_history[-10:]}")
    
    # Check if the issue is in ADX calculation or percentile scaling
    print(f"\n🎯 Potential Issues to Investigate:")
    print(f"1. Different ADX calculation windows or initialization")
    print(f"2. Different percentile scaling windows (200-bar rolling)")
    print(f"3. Different NaN handling in ADX calculation")
    print(f"4. Different DI+ / DI- calculation methods")
    print(f"5. Different True Range calculation approaches")
    
    # Check percentile scaling differences
    print(f"\n📊 Percentile Scaling Analysis:")
    # This would require access to raw ADX values from batch method
    # For now, flag this as the likely source of differences
    
    print(f"\n💡 Likely Root Cause:")
    print(f"The 78.2% correlation suggests the ADX trend is correct but magnitudes differ.")
    print(f"Most likely causes:")
    print(f"  1. Different percentile window handling (200-bar rolling)")
    print(f"  2. Different initialization periods for ADX calculation")
    print(f"  3. Slightly different True Range or DI calculation methods")
    
    return correlation if len(valid_batch) > 0 else None

if __name__ == "__main__":
    debug_adx_correlation()