#!/usr/bin/env python3
"""
Debug ASI calculation to understand why no HSP/LSP are detected
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import IncrementalIndicatorCalculator, InstrumentState
from configs.instruments import get_pip_value

def debug_asi_calculation():
    """Debug ASI calculation step by step"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take first 50 bars for detailed debugging
    df_slice = df.tail(50).copy()
    
    print(f"🔍 Debugging ASI calculation with {len(df_slice)} bars...")
    
    # Batch ASI calculation
    generator = FXFeatureGenerator()
    batch_result = generator.generate_features_single_instrument(df_slice, "EUR_USD")
    
    print(f"\n📊 Batch ASI values (last 10):")
    print(batch_result['asi'].tail(10).values)
    print(f"Batch ASI range: [{batch_result['asi'].min():.1f}, {batch_result['asi'].max():.1f}]")
    
    print(f"\n📊 Batch HSP/LSP detection:")
    print(f"HSP detected: {batch_result['sig_hsp'].sum()} times")
    print(f"LSP detected: {batch_result['sig_lsp'].sum()} times")
    
    if batch_result['sig_hsp'].sum() > 0:
        hsp_indices = batch_result.index[batch_result['sig_hsp']]
        print(f"HSP at indices: {hsp_indices[:5].tolist()}")
    
    if batch_result['sig_lsp'].sum() > 0:
        lsp_indices = batch_result.index[batch_result['sig_lsp']]
        print(f"LSP at indices: {lsp_indices[:5].tolist()}")
    
    # Incremental ASI calculation
    print(f"\n📊 Incremental ASI calculation:")
    calculator = IncrementalIndicatorCalculator()
    pip_size, pip_value = get_pip_value("EUR_USD")
    state = InstrumentState(pip_size=pip_size, pip_value=pip_value)
    
    incremental_asi = []
    hsp_detections = []
    lsp_detections = []
    
    # Store previous OHLC for proper ASI calculation
    prev_ohlc = None
    
    for idx, row in df_slice.iterrows():
        ohlc = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        # Store previous OHLC before updating bar count
        if prev_ohlc is not None:
            state.prev_open = prev_ohlc['open']
            state.prev_high = prev_ohlc['high']
            state.prev_low = prev_ohlc['low']
            state.prev_close = prev_ohlc['close']
        
        state.bar_count += 1
        
        # Debug ASI calculation
        asi_val = calculator.calculate_asi_incremental(ohlc, state, "EUR_USD")
        
        # Store current OHLC as previous for next iteration
        prev_ohlc = ohlc.copy()
        incremental_asi.append(asi_val)
        
        # Debug HSP/LSP detection
        is_hsp, is_lsp = calculator.detect_hsp_lsp_incremental(asi_val, state)
        hsp_detections.append(is_hsp)
        lsp_detections.append(is_lsp)
        
        # Print debug info for last few bars
        if state.bar_count > len(df_slice) - 5:
            print(f"Bar {state.bar_count}: ASI={asi_val:.1f}, HSP={is_hsp}, LSP={is_lsp}, ASI_hist_len={len(state.asi_history)}")
    
    incremental_asi = np.array(incremental_asi)
    print(f"\nIncremental ASI (last 10): {incremental_asi[-10:]}")
    print(f"Incremental ASI range: [{incremental_asi.min():.1f}, {incremental_asi.max():.1f}]")
    
    print(f"\nIncremental HSP detected: {sum(hsp_detections)} times")
    print(f"Incremental LSP detected: {sum(lsp_detections)} times")
    
    # Compare ASI values
    if len(incremental_asi) > 0:
        asi_batch = batch_result['asi'].values
        if len(asi_batch) == len(incremental_asi):
            correlation = np.corrcoef(asi_batch, incremental_asi)[0,1]
            print(f"\nASI correlation: {correlation:.6f}")
            
            diff = np.abs(asi_batch - incremental_asi)
            print(f"ASI mean absolute difference: {diff.mean():.6f}")
            print(f"ASI max absolute difference: {diff.max():.6f}")
            
            # Check if ASI is actually changing
            asi_changes = np.diff(incremental_asi)
            print(f"ASI changes range: [{asi_changes.min():.1f}, {asi_changes.max():.1f}]")
            print(f"ASI non-zero changes: {np.sum(asi_changes != 0)}/{len(asi_changes)}")
    
    # Check state information
    print(f"\n🔍 Final state information:")
    print(f"HSP indices in state: {len(state.hsp_indices)}")
    print(f"LSP indices in state: {len(state.lsp_indices)}")
    print(f"ASI history length: {len(state.asi_history)}")
    
    if len(state.hsp_indices) > 0:
        print(f"HSP indices: {list(state.hsp_indices)}")
        print(f"HSP values: {list(state.hsp_values)}")
    
    if len(state.lsp_indices) > 0:
        print(f"LSP indices: {list(state.lsp_indices)}")
        print(f"LSP values: {list(state.lsp_values)}")

if __name__ == "__main__":
    debug_asi_calculation()