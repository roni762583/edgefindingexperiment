#!/usr/bin/env python3
"""
Debug why no swing points are detected after bar 75 in incremental processing
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.incremental_indicators import MultiInstrumentState, update_indicators

def debug_swing_detection_after_75():
    """Debug swing detection in bars 75-200"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    print(f"🔍 Debugging swing detection after bar 75")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Initialize state and process up to bar 75
    multi_state = MultiInstrumentState()
    
    # Process all 200 bars but focus debug on 75+
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        # Update indicators
        indicators, multi_state = update_indicators(new_ohlc, multi_state, 'EUR_USD')
        
        # Get current state
        state = multi_state.get_instrument_state('EUR_USD')
        
        # Debug detailed state for bars 70-90
        if 70 <= i <= 90:
            current_asi = state.asi_history[-1] if state.asi_history else np.nan
            
            print(f"Bar {i:3d}: ASI={current_asi:6.1f} | "
                  f"HSP={len(state.hsp_indices)} LSP={len(state.lsp_indices)} | "
                  f"Pending HSP: {'Yes' if state.pending_hsp_index is not None else 'No':3s} "
                  f"Pending LSP: {'Yes' if state.pending_lsp_index is not None else 'No':3s}")
            
            # Show pending details
            if state.pending_hsp_index is not None:
                print(f"         Pending HSP: Bar {state.pending_hsp_index}, ASI={state.pending_hsp_asi:.1f}")
            if state.pending_lsp_index is not None:
                print(f"         Pending LSP: Bar {state.pending_lsp_index}, ASI={state.pending_lsp_asi:.1f}")
            
            # Show last significant points
            if hasattr(state, 'last_sig_hsp_asi') and state.last_sig_hsp_asi is not None:
                print(f"         Last HSP: Bar {state.last_sig_hsp_index}, ASI={state.last_sig_hsp_asi:.1f}")
            if hasattr(state, 'last_sig_lsp_asi') and state.last_sig_lsp_asi is not None:
                print(f"         Last LSP: Bar {state.last_sig_lsp_index}, ASI={state.last_sig_lsp_asi:.1f}")
        
        # Check for 3-bar patterns manually in the focus range
        if 75 <= i <= 95 and len(state.asi_history) >= 3:
            left = state.asi_history[-3]
            middle = state.asi_history[-2] 
            right = state.asi_history[-1]
            
            # Check if middle bar (i-1) forms a pattern
            if not (np.isnan(left) or np.isnan(middle) or np.isnan(right)):
                should_be_hsp = middle > left and middle > right
                should_be_lsp = middle < left and middle < right
                
                if should_be_hsp or should_be_lsp:
                    pattern_type = "HSP" if should_be_hsp else "LSP"
                    middle_bar = i - 1
                    print(f"  🔍 Bar {middle_bar} should be {pattern_type}: {left:.1f} {'<' if should_be_hsp else '>'} {middle:.1f} {'>' if should_be_hsp else '<'} {right:.1f}")
                    
                    # Check why it's not being confirmed
                    if should_be_hsp and state.pending_hsp_index != middle_bar:
                        print(f"      ❌ HSP candidate not stored - possible overwrite issue")
                    elif should_be_lsp and state.pending_lsp_index != middle_bar:
                        print(f"      ❌ LSP candidate not stored - possible overwrite issue")
                    elif should_be_hsp and state.pending_hsp_index == middle_bar:
                        # Check confirmation conditions
                        if state.last_sig_lsp_asi is not None:
                            needs_drop_below = state.last_sig_lsp_asi
                            print(f"      ⏳ HSP waiting for ASI to drop below {needs_drop_below:.1f} (current: {right:.1f})")
                        else:
                            print(f"      ❌ HSP has no prior LSP to confirm against")
                    elif should_be_lsp and state.pending_lsp_index == middle_bar:
                        # Check confirmation conditions
                        if state.last_sig_hsp_asi is not None:
                            needs_rise_above = state.last_sig_hsp_asi
                            print(f"      ⏳ LSP waiting for ASI to rise above {needs_rise_above:.1f} (current: {right:.1f})")
                        else:
                            print(f"      ❌ LSP has no prior HSP to confirm against")
    
    # Final state summary
    state = multi_state.get_instrument_state('EUR_USD')
    print(f"\n📊 Final State Summary:")
    print(f"Total HSP detected: {len(state.hsp_indices)}")
    for i, (bar, asi) in enumerate(zip(state.hsp_indices, state.hsp_values)):
        print(f"  HSP {i+1}: Bar {bar}, ASI={asi:.1f}")
    
    print(f"Total LSP detected: {len(state.lsp_indices)}")
    for i, (bar, asi) in enumerate(zip(state.lsp_indices, state.lsp_values)):
        print(f"  LSP {i+1}: Bar {bar}, ASI={asi:.1f}")
    
    print(f"\nPending candidates:")
    if state.pending_hsp_index is not None:
        print(f"  Pending HSP: Bar {state.pending_hsp_index}, ASI={state.pending_hsp_asi:.1f}")
        if state.last_sig_lsp_asi is not None:
            print(f"    Needs ASI < {state.last_sig_lsp_asi:.1f} to confirm")
    else:
        print(f"  No pending HSP")
        
    if state.pending_lsp_index is not None:
        print(f"  Pending LSP: Bar {state.pending_lsp_index}, ASI={state.pending_lsp_asi:.1f}")
        if state.last_sig_hsp_asi is not None:
            print(f"    Needs ASI > {state.last_sig_hsp_asi:.1f} to confirm")
    else:
        print(f"  No pending LSP")

if __name__ == "__main__":
    debug_swing_detection_after_75()