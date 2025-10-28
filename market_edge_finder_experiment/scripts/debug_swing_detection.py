#!/usr/bin/env python3
"""
Debug incremental swing detection to understand why no swing points are detected
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.incremental_indicators import (
    IncrementalIndicatorCalculator,
    InstrumentState,
    MultiInstrumentState
)

def debug_asi_and_swings():
    """Debug ASI calculation and swing detection step by step"""
    
    # Load some test data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take small slice for debugging
    test_slice = df.iloc[1000:1020].copy()  # 20 bars
    
    print(f"🔍 Debugging ASI and swing detection on {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Initialize state
    multi_state = MultiInstrumentState()
    calculator = IncrementalIndicatorCalculator()
    instrument = "EUR_USD"
    
    asi_values = []
    swing_events = []
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        # Get state
        state = multi_state.get_instrument_state(instrument)
        state.bar_count += 1
        
        # Calculate ASI
        asi_value = calculator.calculate_asi_incremental(ohlc, state, instrument)
        asi_values.append(asi_value)
        
        # Detect swings
        is_hsp, is_lsp = calculator.detect_hsp_lsp_wilder_proper(
            asi_value, ohlc['high'], ohlc['low'], state
        )
        
        # Log details for debugging
        print(f"Bar {i+1:2d} ({timestamp.strftime('%m-%d %H:%M')}): ASI={asi_value:8.1f}, HSP={is_hsp}, LSP={is_lsp}")
        
        if is_hsp or is_lsp:
            swing_events.append((i+1, 'HSP' if is_hsp else 'LSP', asi_value))
            print(f"  🎯 SWING DETECTED: {'HSP' if is_hsp else 'LSP'} at bar {i+1}")
        
        # Print state info every few bars
        if (i+1) % 5 == 0:
            print(f"  📊 State: asi_history_len={len(state.asi_history)}, "
                  f"pending_hsp={state.pending_hsp_index}, "
                  f"pending_lsp={state.pending_lsp_index}")
            print(f"  📊 Significant: last_hsp_asi={state.last_sig_hsp_asi}, "
                  f"last_lsp_asi={state.last_sig_lsp_asi}")
            
        # Update previous OHLC for next iteration
        state.prev_open = ohlc['open']
        state.prev_high = ohlc['high']
        state.prev_low = ohlc['low']
        state.prev_close = ohlc['close']
    
    print(f"\n📈 Results:")
    print(f"ASI range: [{min(asi_values):.1f}, {max(asi_values):.1f}]")
    print(f"ASI final: {asi_values[-1]:.1f}")
    print(f"Swing events detected: {len(swing_events)}")
    
    for event in swing_events:
        bar, type, asi = event
        print(f"  Bar {bar}: {type} at ASI={asi:.1f}")
    
    # Check if ASI is actually changing
    asi_diffs = [abs(asi_values[i] - asi_values[i-1]) for i in range(1, len(asi_values))]
    print(f"\nASI differences: min={min(asi_diffs):.1f}, max={max(asi_diffs):.1f}, mean={np.mean(asi_diffs):.1f}")
    
    if max(asi_diffs) < 1.0:
        print("⚠️  ASI values are barely changing - this suggests calculation issues")
    else:
        print("✅ ASI values are changing normally")

if __name__ == "__main__":
    debug_asi_and_swings()