#!/usr/bin/env python3
"""
Debug the timing offset between batch and incremental swing detection
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import (
    IncrementalIndicatorCalculator,
    MultiInstrumentState
)

def debug_timing_offset():
    """Debug the 1-bar timing offset between batch and incremental"""
    
    # Load small slice for detailed analysis
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use the same slice that shows the timing issue
    test_slice = df.iloc[1500:1520].copy()  # Just 20 bars for detailed analysis
    
    print(f"🔍 Debugging timing offset on {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # --- Run batch processing ---
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    print(f"\n📊 BATCH RESULTS:")
    print(f"ASI values: {batch_results['asi'].values[:10]}")  # First 10 values
    
    # Show batch swing points
    if 'sig_hsp' in batch_results.columns:
        for i, (hsp, lsp, asi) in enumerate(zip(batch_results['sig_hsp'], batch_results['sig_lsp'], batch_results['asi'])):
            if hsp or lsp:
                timestamp = batch_results.index[i]
                print(f"🎯 Batch swing at bar {i} ({timestamp}): {'HSP' if hsp else 'LSP'} at ASI={asi:.1f}")
    
    # --- Run incremental processing with detailed logging ---
    print(f"\n📊 INCREMENTAL PROCESSING (Step by step):")
    
    multi_state = MultiInstrumentState()
    calculator = IncrementalIndicatorCalculator()
    
    incremental_asi_values = []
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        ohlc = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        state = multi_state.get_instrument_state("EUR_USD")
        state.bar_count += 1
        
        # Calculate ASI
        asi_value = calculator.calculate_asi_incremental(ohlc, state, "EUR_USD")
        incremental_asi_values.append(asi_value)
        
        # Detect swings
        is_hsp, is_lsp = calculator.detect_hsp_lsp_wilder_proper(
            asi_value, ohlc['high'], ohlc['low'], state
        )
        
        print(f"Bar {i:2d} ({timestamp.strftime('%m-%d %H:%M')}): ASI={asi_value:6.1f}, HSP={is_hsp}, LSP={is_lsp}")
        
        if is_hsp or is_lsp:
            print(f"  🎯 INCREMENTAL swing detected: {'HSP' if is_hsp else 'LSP'} at ASI={asi_value:.1f}")
            
            # Check ASI history for debugging
            if len(state.asi_history) >= 3:
                print(f"  📊 ASI history (last 3): {list(state.asi_history)[-3:]}")
                print(f"  📊 Pending HSP: idx={state.pending_hsp_index}, asi={state.pending_hsp_asi}")
                print(f"  📊 Pending LSP: idx={state.pending_lsp_index}, asi={state.pending_lsp_asi}")
                print(f"  📊 Last sig HSP: idx={state.last_sig_hsp_index}, asi={state.last_sig_hsp_asi}")
                print(f"  📊 Last sig LSP: idx={state.last_sig_lsp_index}, asi={state.last_sig_lsp_asi}")
        
        # Update state for next iteration
        state.prev_open = ohlc['open']
        state.prev_high = ohlc['high']
        state.prev_low = ohlc['low']
        state.prev_close = ohlc['close']
    
    # --- Compare ASI values ---
    print(f"\n📊 ASI COMPARISON:")
    print("Bar | Batch ASI | Incr ASI | Diff")
    print("----|-----------|----------|------")
    
    for i in range(min(len(batch_results), len(incremental_asi_values))):
        batch_asi = batch_results['asi'].iloc[i]
        incr_asi = incremental_asi_values[i]
        diff = abs(batch_asi - incr_asi)
        print(f"{i:3d} | {batch_asi:8.1f} | {incr_asi:7.1f} | {diff:5.1f}")
    
    # --- Analyze the timing offset ---
    print(f"\n🔍 TIMING ANALYSIS:")
    
    # Find batch swing points
    batch_swings = []
    if 'sig_hsp' in batch_results.columns:
        for i, (hsp, lsp) in enumerate(zip(batch_results['sig_hsp'], batch_results['sig_lsp'])):
            if hsp:
                batch_swings.append((i, 'HSP', batch_results['asi'].iloc[i]))
            if lsp:
                batch_swings.append((i, 'LSP', batch_results['asi'].iloc[i]))
    
    # Find incremental swing points
    incr_swings = []
    state = multi_state.get_instrument_state("EUR_USD")
    # Re-run to collect swing points
    multi_state = MultiInstrumentState()
    calculator = IncrementalIndicatorCalculator()
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        ohlc = {'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']}
        state = multi_state.get_instrument_state("EUR_USD")
        state.bar_count += 1
        
        asi_value = calculator.calculate_asi_incremental(ohlc, state, "EUR_USD")
        is_hsp, is_lsp = calculator.detect_hsp_lsp_wilder_proper(asi_value, ohlc['high'], ohlc['low'], state)
        
        if is_hsp:
            incr_swings.append((i, 'HSP', asi_value))
        if is_lsp:
            incr_swings.append((i, 'LSP', asi_value))
        
        state.prev_open = ohlc['open']
        state.prev_high = ohlc['high']
        state.prev_low = ohlc['low']
        state.prev_close = ohlc['close']
    
    print(f"Batch swings: {batch_swings}")
    print(f"Incremental swings: {incr_swings}")
    
    # Check for timing offset pattern
    if len(batch_swings) > 0 and len(incr_swings) > 0:
        batch_bar = batch_swings[0][0]
        incr_bar = incr_swings[0][0]
        offset = incr_bar - batch_bar
        print(f"\n⚠️  TIMING OFFSET DETECTED: {offset} bar(s)")
        print(f"   Batch swing at bar {batch_bar}")
        print(f"   Incremental swing at bar {incr_bar}")
        
        if offset == 1:
            print(f"🔧 DIAGNOSIS: Incremental method is 1 bar late")
            print(f"   This suggests an indexing issue in the swing detection logic")
            print(f"   The 3-bar pattern detection may be using wrong bar indices")
        
    return batch_results, incremental_asi_values, batch_swings, incr_swings

if __name__ == "__main__":
    debug_timing_offset()