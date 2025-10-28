#!/usr/bin/env python3
"""
Test AUD_CHF processing with dynamic pip values that update every bar
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.practical_incremental import update_practical_indicators, PracticalMultiInstrumentState
from configs.instruments import extract_instrument_from_filename, calculate_pip_value_usd, get_pip_size

def test_aud_chf_dynamic_pips():
    """Test AUD_CHF processing with dynamic pip values"""
    
    csv_path = project_root / "data/raw/AUD_CHF_3years_H1.csv"
    
    print("🧪 TESTING: AUD_CHF with Dynamic Pip Values")
    print("=" * 60)
    
    # Auto-extract instrument name from filename
    instrument = extract_instrument_from_filename(str(csv_path))
    print(f"🔍 Auto-detected instrument: {instrument}")
    print(f"📏 Pip size: {get_pip_size(instrument)}")
    
    # Load sample data
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Test first 100 bars
    test_data = df.head(100)
    print(f"📊 Testing with {len(test_data)} bars")
    print(f"📅 Date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Rate range analysis
    rate_min, rate_max = test_data['close'].min(), test_data['close'].max()
    print(f"💱 Rate range: {rate_min:.5f} to {rate_max:.5f}")
    
    # Initialize state
    multi_state = PracticalMultiInstrumentState()
    results = []
    pip_values = []
    
    print(f"\n🔄 Processing row by row with dynamic pip values...")
    
    for i, (timestamp, row) in enumerate(test_data.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        # Calculate dynamic pip value for current rate
        current_rate = row['close']
        pip_value_usd = calculate_pip_value_usd(instrument, current_rate)
        pip_values.append(pip_value_usd)
        
        # Generate indicators
        indicators, multi_state = update_practical_indicators(new_ohlc, multi_state, instrument)
        
        result = {
            'bar': i,
            'timestamp': timestamp,
            'rate': current_rate,
            'pip_value_usd': pip_value_usd,
            **indicators
        }
        results.append(result)
        
        # Show first 10 bars in detail
        if i < 10:
            print(f"Bar {i:2d}: Rate={current_rate:.5f}, PipValue=${pip_value_usd:.2f}, "
                  f"Vol={indicators['volatility']:.3f}, Dir={indicators['direction']:.3f}")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    print(f"\n📈 Dynamic Pip Value Analysis:")
    pip_min, pip_max = np.min(pip_values), np.max(pip_values)
    pip_mean, pip_std = np.mean(pip_values), np.std(pip_values)
    print(f"  Min pip value: ${pip_min:.2f}")
    print(f"  Max pip value: ${pip_max:.2f}")
    print(f"  Mean pip value: ${pip_mean:.2f}")
    print(f"  Std dev: ${pip_std:.2f}")
    print(f"  Variation: {(pip_max-pip_min)/pip_mean*100:.1f}%")
    
    # Show indicator statistics
    print(f"\n📊 Indicator Statistics:")
    for col in ['volatility', 'direction', 'price_change']:
        valid_data = results_df[col].dropna()
        if len(valid_data) > 0:
            print(f"  {col}: mean={valid_data.mean():.3f}, std={valid_data.std():.3f}, "
                  f"range=[{valid_data.min():.3f}, {valid_data.max():.3f}]")
        else:
            print(f"  {col}: No valid data")
    
    # Show swing detection results
    state = multi_state.get_instrument_state(instrument)
    print(f"\n🎯 Swing Detection Results:")
    print(f"  HSP detected: {len(state.hsp_indices)}")
    print(f"  LSP detected: {len(state.lsp_indices)}")
    print(f"  Total swings: {len(state.hsp_indices) + len(state.lsp_indices)}")
    
    # Show some swing points
    if state.hsp_indices or state.lsp_indices:
        print(f"\nFirst 5 swings:")
        all_swings = []
        for i, idx in enumerate(state.hsp_indices):
            if idx < len(state.asi_history):
                all_swings.append((idx, 'HSP', state.asi_history[idx]))
        for i, idx in enumerate(state.lsp_indices):
            if idx < len(state.asi_history):
                all_swings.append((idx, 'LSP', state.asi_history[idx]))
        
        all_swings.sort()
        for i, (bar, type_, asi) in enumerate(all_swings[:5]):
            rate = results_df.iloc[bar]['rate'] if bar < len(results_df) else 'N/A'
            pip_val = results_df.iloc[bar]['pip_value_usd'] if bar < len(results_df) else 'N/A'
            print(f"  Bar {bar:2d}: {type_} at ASI={asi:6.1f}, Rate={rate}, PipVal=${pip_val}")
    
    print(f"\n✅ Test complete: AUD_CHF processing with dynamic pip values working!")
    return results_df

if __name__ == "__main__":
    test_aud_chf_dynamic_pips()