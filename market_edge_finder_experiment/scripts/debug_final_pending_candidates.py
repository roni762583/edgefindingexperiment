#!/usr/bin/env python3
"""
Debug why swing detection doesn't reach bar 200 - check pending candidates
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.incremental_indicators import MultiInstrumentState, update_indicators

def debug_final_pending_candidates():
    """Debug pending candidates that haven't been confirmed by bar 200"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    print(f"🔍 Debugging final pending candidates in 200-bar range")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Initialize state
    multi_state = MultiInstrumentState()
    
    # Process all 200 bars
    confirmed_swings = []
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        # Update indicators
        indicators, multi_state = update_indicators(new_ohlc, multi_state, 'EUR_USD')
        
        # Track confirmed swings
        state = multi_state.get_instrument_state('EUR_USD')
        
        # Check for new confirmations
        if len(state.hsp_indices) > len([s for s in confirmed_swings if s[1] == 'HSP']):
            confirmed_swings.append((state.hsp_indices[-1], 'HSP', state.hsp_values[-1]))
            print(f"✅ Bar {i:3d}: HSP CONFIRMED at bar {state.hsp_indices[-1]}, ASI={state.hsp_values[-1]:.1f}")
            
        if len(state.lsp_indices) > len([s for s in confirmed_swings if s[1] == 'LSP']):
            confirmed_swings.append((state.lsp_indices[-1], 'LSP', state.lsp_values[-1]))
            print(f"✅ Bar {i:3d}: LSP CONFIRMED at bar {state.lsp_indices[-1]}, ASI={state.lsp_values[-1]:.1f}")
    
    # Final state analysis
    state = multi_state.get_instrument_state('EUR_USD')
    current_asi = state.asi_history[-1] if state.asi_history else np.nan
    
    print(f"\n📊 Final Analysis at Bar 199 (ASI={current_asi:.1f}):")
    
    # Show all confirmed swings
    print(f"\n✅ Confirmed Swings:")
    for bar, type_, asi in sorted(confirmed_swings):
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Show pending candidates
    print(f"\n⏳ Pending Candidates (waiting for confirmation):")
    
    if state.pending_hsp_index is not None:
        print(f"  HSP Candidate: Bar {state.pending_hsp_index}, ASI={state.pending_hsp_asi:.1f}")
        if state.last_sig_lsp_asi is not None:
            print(f"    ⏳ Needs ASI < {state.last_sig_lsp_asi:.1f} to confirm (current: {current_asi:.1f})")
            if current_asi < state.last_sig_lsp_asi:
                print(f"    ✅ SHOULD BE CONFIRMED! Condition met.")
            else:
                print(f"    ❌ Not confirmed - ASI hasn't dropped enough")
        else:
            print(f"    ❌ No prior LSP to confirm against")
            
        # Check alternation requirement
        last_confirmed_was_hsp = (state.last_sig_hsp_index is not None and 
                                 (state.last_sig_lsp_index is None or 
                                  state.last_sig_hsp_index > state.last_sig_lsp_index))
        if last_confirmed_was_hsp:
            print(f"    ❌ ALTERNATION BLOCKED: Last confirmed was HSP, need LSP first")
        else:
            print(f"    ✅ ALTERNATION OK: Can confirm HSP")
    else:
        print(f"  No pending HSP")
    
    if state.pending_lsp_index is not None:
        print(f"  LSP Candidate: Bar {state.pending_lsp_index}, ASI={state.pending_lsp_asi:.1f}")
        if state.last_sig_hsp_asi is not None:
            print(f"    ⏳ Needs ASI > {state.last_sig_hsp_asi:.1f} to confirm (current: {current_asi:.1f})")
            if current_asi > state.last_sig_hsp_asi:
                print(f"    ✅ SHOULD BE CONFIRMED! Condition met.")
            else:
                print(f"    ❌ Not confirmed - ASI hasn't risen enough")
        else:
            print(f"    ❌ No prior HSP to confirm against")
            
        # Check alternation requirement
        last_confirmed_was_hsp = (state.last_sig_hsp_index is not None and 
                                 (state.last_sig_lsp_index is None or 
                                  state.last_sig_hsp_index > state.last_sig_lsp_index))
        if not last_confirmed_was_hsp:
            print(f"    ❌ ALTERNATION BLOCKED: Last confirmed was LSP, need HSP first")
        else:
            print(f"    ✅ ALTERNATION OK: Can confirm LSP")
    else:
        print(f"  No pending LSP")
    
    # Check last 20 bars for potential 3-bar patterns
    print(f"\n🔍 Potential patterns in last 20 bars (180-199):")
    
    focus_start = max(0, 180)
    for i in range(focus_start, min(200, len(state.asi_history))):
        if i >= 2:  # Need 3 bars for pattern
            left = state.asi_history[i-2]
            middle = state.asi_history[i-1] 
            right = state.asi_history[i]
            
            if not (np.isnan(left) or np.isnan(middle) or np.isnan(right)):
                should_be_hsp = middle > left and middle > right
                should_be_lsp = middle < left and middle < right
                
                if should_be_hsp:
                    middle_bar = i - 1
                    print(f"  Bar {middle_bar:3d}: HSP pattern {left:.1f} < {middle:.1f} > {right:.1f}")
                    
                if should_be_lsp:
                    middle_bar = i - 1
                    print(f"  Bar {middle_bar:3d}: LSP pattern {left:.1f} > {middle:.1f} < {right:.1f}")
    
    # Show final significant points
    print(f"\n📈 Last Confirmed Significant Points:")
    if state.last_sig_hsp_index is not None:
        print(f"  Last HSP: Bar {state.last_sig_hsp_index}, ASI={state.last_sig_hsp_asi:.1f}")
    if state.last_sig_lsp_index is not None:
        print(f"  Last LSP: Bar {state.last_sig_lsp_index}, ASI={state.last_sig_lsp_asi:.1f}")
    
    # Suggest what would happen with more data
    print(f"\n💡 What would happen with more data:")
    print(f"If we had more bars beyond 199, pending candidates might get confirmed")
    print(f"when ASI moves in the required direction for breakout confirmation.")

if __name__ == "__main__":
    debug_final_pending_candidates()