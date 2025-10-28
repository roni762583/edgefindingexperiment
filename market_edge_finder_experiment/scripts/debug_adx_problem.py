#!/usr/bin/env python3
"""
Debug ADX calculation problems
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.practical_incremental import update_practical_indicators, PracticalMultiInstrumentState

def debug_adx_calculation():
    """Debug what's happening with ADX calculation"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use small slice for debugging
    test_slice = df.iloc[2000:2050].copy()  # Only 50 bars
    print(f"🔍 Debugging ADX calculation: 50 bars")
    
    # Incremental processing
    multi_state = PracticalMultiInstrumentState()
    results = []
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        indicators, multi_state = update_practical_indicators(new_ohlc, multi_state, 'EUR_USD')
        
        # Get state for debugging
        state = multi_state.get_instrument_state('EUR_USD')
        
        debug_info = {
            'bar': i,
            'adx_raw': state.adx_history[-1] if state.adx_history else np.nan,
            'di_plus': state.di_plus_history[-1] if state.di_plus_history else np.nan,
            'di_minus': state.di_minus_history[-1] if state.di_minus_history else np.nan,
            'dx': state.dx_history[-1] if state.dx_history else np.nan,
            'atr_usd': state.atr_history[-1] if state.atr_history else np.nan,
            'direction_scaled': indicators['direction']
        }
        
        results.append(debug_info)
        
        # Print first 10 bars for debugging
        if i < 10:
            print(f"Bar {i:2d}: ADX={debug_info['adx_raw']:8.2f}, DI+={debug_info['di_plus']:8.2f}, "
                  f"DI-={debug_info['di_minus']:8.2f}, ATR=${debug_info['atr_usd']:6.2f}, "
                  f"Scaled={debug_info['direction_scaled']:.3f}")
    
    # Convert to DataFrame for analysis
    debug_df = pd.DataFrame(results)
    
    print(f"\n📊 ADX Statistics:")
    print(f"Raw ADX - Min: {debug_df['adx_raw'].min():.2f}, Max: {debug_df['adx_raw'].max():.2f}, Mean: {debug_df['adx_raw'].mean():.2f}")
    print(f"DI+ - Min: {debug_df['di_plus'].min():.2f}, Max: {debug_df['di_plus'].max():.2f}, Mean: {debug_df['di_plus'].mean():.2f}")
    print(f"DI- - Min: {debug_df['di_minus'].min():.2f}, Max: {debug_df['di_minus'].max():.2f}, Mean: {debug_df['di_minus'].mean():.2f}")
    print(f"ATR USD - Min: ${debug_df['atr_usd'].min():.2f}, Max: ${debug_df['atr_usd'].max():.2f}, Mean: ${debug_df['atr_usd'].mean():.2f}")
    print(f"Scaled Direction - Min: {debug_df['direction_scaled'].min():.3f}, Max: {debug_df['direction_scaled'].max():.3f}")
    
    # Check for overflow/inf values
    print(f"\n⚠️  Problem Detection:")
    overflow_adx = np.isinf(debug_df['adx_raw']).sum()
    overflow_di_plus = np.isinf(debug_df['di_plus']).sum()
    overflow_di_minus = np.isinf(debug_df['di_minus']).sum()
    
    print(f"Infinite ADX values: {overflow_adx}")
    print(f"Infinite DI+ values: {overflow_di_plus}")
    print(f"Infinite DI- values: {overflow_di_minus}")
    
    # Check if all scaled values are 0.99 (capped)
    capped_values = (debug_df['direction_scaled'] == 0.99).sum()
    total_valid = (~debug_df['direction_scaled'].isna()).sum()
    print(f"Values capped at 0.99: {capped_values}/{total_valid} ({capped_values/total_valid*100:.1f}%)")
    
    return debug_df

if __name__ == "__main__":
    debug_adx_calculation()