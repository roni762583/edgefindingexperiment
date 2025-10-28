#!/usr/bin/env python3
"""
Debug ASI values and swing detection in detail
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.incremental_indicators import (
    IncrementalIndicatorCalculator,
    MultiInstrumentState
)

def debug_incremental_asi_step_by_step():
    """Debug incremental ASI calculation step by step"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take slice for detailed analysis
    test_slice = df.iloc[1500:1550].copy()  # 50 bars
    
    print(f"🔍 Debugging incremental ASI on {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Initialize
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
        
        if is_hsp or is_lsp:
            swing_events.append({
                'bar': i+1,
                'timestamp': timestamp,
                'type': 'HSP' if is_hsp else 'LSP',
                'asi_value': asi_value,
                'price_high': ohlc['high'],
                'price_low': ohlc['low']
            })
            print(f"🎯 SWING DETECTED at bar {i+1}: {'HSP' if is_hsp else 'LSP'} at ASI={asi_value:.1f}")
        
        # Update previous OHLC
        state.prev_open = ohlc['open']
        state.prev_high = ohlc['high']
        state.prev_low = ohlc['low']
        state.prev_close = ohlc['close']
    
    print(f"\n📊 Final Results:")
    print(f"ASI range: [{min(asi_values):.1f}, {max(asi_values):.1f}]")
    print(f"Swing events: {len(swing_events)}")
    
    for event in swing_events:
        print(f"  Bar {event['bar']:2d}: {event['type']} at ASI={event['asi_value']:6.1f} "
              f"(H={event['price_high']:.5f}, L={event['price_low']:.5f})")
    
    # Create detailed DataFrame for analysis
    results_df = pd.DataFrame({
        'timestamp': test_slice.index,
        'open': test_slice['open'],
        'high': test_slice['high'], 
        'low': test_slice['low'],
        'close': test_slice['close'],
        'asi': asi_values
    })
    
    # Add swing markers
    results_df['swing_hsp'] = False
    results_df['swing_lsp'] = False
    
    for event in swing_events:
        if event['type'] == 'HSP':
            results_df.iloc[event['bar']-1, results_df.columns.get_loc('swing_hsp')] = True
        else:
            results_df.iloc[event['bar']-1, results_df.columns.get_loc('swing_lsp')] = True
    
    # Save for inspection
    save_path = project_root / "data/test/incremental_asi_debug_50bars.csv"
    results_df.to_csv(save_path)
    print(f"\n💾 Detailed results saved to: {save_path}")
    
    return results_df

if __name__ == "__main__":
    debug_incremental_asi_step_by_step()