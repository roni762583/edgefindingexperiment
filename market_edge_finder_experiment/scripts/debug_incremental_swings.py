#!/usr/bin/env python3
"""
Debug why incremental swing detection is not finding any swings
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.incremental_indicators import MultiInstrumentState, update_indicators

def debug_incremental_swings():
    """Debug incremental swing detection"""
    
    # Load same data as comparison test
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 2000-bar slice
    test_slice = df.iloc[1666:3666].copy()
    print(f"🔍 Debugging incremental swing detection on {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Initialize state
    multi_state = MultiInstrumentState()
    
    # Process first 100 bars and track swing detection
    debug_bars = 100
    swing_events = []
    
    for i, (timestamp, row) in enumerate(test_slice.head(debug_bars).iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        # Update indicators
        indicators, multi_state = update_indicators(new_ohlc, multi_state, 'EUR_USD')
        
        # Get instrument state for debugging
        state = multi_state.get_instrument_state('EUR_USD')
        
        # Check for new swing points
        if len(state.hsp_indices) > len([s for s in swing_events if s[1] == 'HSP']):
            swing_events.append((i, 'HSP', state.hsp_values[-1]))
            print(f"  Bar {i:3d}: NEW HSP detected at ASI={state.hsp_values[-1]:.1f}")
            
        if len(state.lsp_indices) > len([s for s in swing_events if s[1] == 'LSP']):
            swing_events.append((i, 'LSP', state.lsp_values[-1]))
            print(f"  Bar {i:3d}: NEW LSP detected at ASI={state.lsp_values[-1]:.1f}")
        
        # Every 10 bars, show state summary
        if i % 20 == 0 and i > 0:
            print(f"Bar {i:3d}: ASI={len(state.asi_history):3d} bars, "
                  f"HSP={len(state.hsp_indices)}, LSP={len(state.lsp_indices)}, "
                  f"Pending HSP: {'Yes' if state.pending_hsp_index is not None else 'No'}, "
                  f"Pending LSP: {'Yes' if state.pending_lsp_index is not None else 'No'}")
            
            if hasattr(state, 'last_sig_hsp_asi') and state.last_sig_hsp_asi is not None:
                print(f"         Last HSP ASI: {state.last_sig_hsp_asi:.1f}")
            if hasattr(state, 'last_sig_lsp_asi') and state.last_sig_lsp_asi is not None:
                print(f"         Last LSP ASI: {state.last_sig_lsp_asi:.1f}")
    
    print(f"\n📊 Final swing detection results after {debug_bars} bars:")
    print(f"Total HSP detected: {len([s for s in swing_events if s[1] == 'HSP'])}")
    print(f"Total LSP detected: {len([s for s in swing_events if s[1] == 'LSP'])}")
    
    if swing_events:
        print(f"\nSwing sequence:")
        for bar, type_, asi in swing_events:
            print(f"  Bar {bar:3d}: {type_} at ASI={asi:.1f}")
    else:
        print("\n❌ No swing points detected! This indicates an issue with the incremental implementation.")
        
        # Show final state details
        state = multi_state.get_instrument_state('EUR_USD')
        print(f"\nFinal state details:")
        print(f"  ASI history length: {len(state.asi_history)}")
        print(f"  Last 5 ASI values: {state.asi_history[-5:] if len(state.asi_history) >= 5 else state.asi_history}")
        print(f"  Pending HSP: {state.pending_hsp_index is not None}")
        print(f"  Pending LSP: {state.pending_lsp_index is not None}")
        
        if hasattr(state, 'last_sig_hsp_asi'):
            print(f"  Last significant HSP ASI: {state.last_sig_hsp_asi}")
        if hasattr(state, 'last_sig_lsp_asi'):
            print(f"  Last significant LSP ASI: {state.last_sig_lsp_asi}")

if __name__ == "__main__":
    debug_incremental_swings()