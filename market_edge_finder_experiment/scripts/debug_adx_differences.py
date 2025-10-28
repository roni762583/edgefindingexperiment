#!/usr/bin/env python3
"""
Debug ADX calculation differences between batch and incremental
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator, TechnicalIndicators
from features.incremental_indicators import IncrementalIndicatorCalculator, InstrumentState
from configs.instruments import get_pip_value

def debug_adx_calculation(num_bars=100):
    """Debug ADX calculation step by step"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take a small slice for debugging
    df_slice = df.tail(num_bars).copy()
    
    print(f"🔍 Debugging ADX calculation with {len(df_slice)} bars...")
    
    # Batch ADX calculation
    high = df_slice['high'].values
    low = df_slice['low'].values  
    close = df_slice['close'].values
    
    print("\n📊 Batch ADX Calculation:")
    batch_adx = TechnicalIndicators.calculate_adx(high, low, close, period=14)
    print(f"Raw batch ADX (last 10): {batch_adx[-10:]}")
    print(f"Batch ADX range: [{np.nanmin(batch_adx):.3f}, {np.nanmax(batch_adx):.3f}]")
    
    # Incremental ADX calculation
    print("\n📊 Incremental ADX Calculation:")
    calculator = IncrementalIndicatorCalculator()
    pip_size, pip_value = get_pip_value("EUR_USD")
    state = InstrumentState(pip_size=pip_size, pip_value=pip_value)
    
    incremental_adx = []
    
    for idx, row in df_slice.iterrows():
        ohlc = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        state.bar_count += 1
        adx_val = calculator.calculate_adx_incremental(ohlc, state)
        incremental_adx.append(adx_val)
        
        # Debug specific values
        if state.bar_count > num_bars - 5:
            print(f"Bar {state.bar_count}: ADX={adx_val:.3f}, TR_EMA={state.tr_ema:.6f}, DM+={state.dm_plus_ema:.6f}, DM-={state.dm_minus_ema:.6f}")
    
    incremental_adx = np.array(incremental_adx)
    print(f"Raw incremental ADX (last 10): {incremental_adx[-10:]}")
    print(f"Incremental ADX range: [{np.nanmin(incremental_adx):.3f}, {np.nanmax(incremental_adx):.3f}]")
    
    # Compare raw values
    print(f"\n📈 Raw ADX Comparison:")
    valid_mask = ~np.isnan(batch_adx) & ~np.isnan(incremental_adx)
    if valid_mask.sum() > 0:
        correlation = np.corrcoef(batch_adx[valid_mask], incremental_adx[valid_mask])[0,1]
        print(f"Raw ADX correlation: {correlation:.6f}")
        
        diff = np.abs(batch_adx[valid_mask] - incremental_adx[valid_mask])
        print(f"Mean absolute difference: {diff.mean():.6f}")
        print(f"Max absolute difference: {diff.max():.6f}")
        
        # Show side-by-side comparison
        print(f"\nSide-by-side comparison (last 10):")
        batch_valid = batch_adx[valid_mask][-10:]
        incr_valid = incremental_adx[valid_mask][-10:]
        for i, (b, inc) in enumerate(zip(batch_valid, incr_valid)):
            print(f"  {i}: Batch={b:.3f}, Incremental={inc:.3f}, Diff={abs(b-inc):.3f}")
    
    # Check the methodology differences
    print(f"\n🔍 Methodology Analysis:")
    print(f"Key difference: EMA calculation method")
    print(f"Batch (TechnicalIndicators): Uses Wilder's smoothing")
    print(f"Incremental: Uses standard EMA (alpha=2/(period+1))")
    print(f"Wilder's smoothing: alpha=1/period vs Standard EMA: alpha=2/(period+1)")
    print(f"For period=14: Wilder's alpha={1/14:.6f} vs Standard alpha={2/15:.6f}")
    
    return batch_adx, incremental_adx

def investigate_wilder_smoothing():
    """Compare Wilder's vs Standard EMA"""
    period = 14
    wilder_alpha = 1.0 / period
    standard_alpha = 2.0 / (period + 1)
    
    print(f"\n🔬 EMA Method Comparison:")
    print(f"Period: {period}")
    print(f"Wilder's alpha: {wilder_alpha:.6f}")
    print(f"Standard alpha: {standard_alpha:.6f}")
    print(f"Difference: {abs(wilder_alpha - standard_alpha):.6f}")
    print(f"This explains the 72.4% correlation vs 100%!")

if __name__ == "__main__":
    debug_adx_calculation()
    investigate_wilder_smoothing()