#!/usr/bin/env python3
"""
Debug why simple method is not finding enough swing points
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def debug_simple_detection():
    """Debug step by step what simple detection should find"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use small slice first
    test_slice = df.iloc[2000:2050].copy()  # Just 50 bars to debug
    
    # Get ASI
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    asi_values = batch_results['asi'].values
    
    print(f"🔍 Debugging simple swing detection on first 50 bars")
    print(f"ASI range: {np.nanmin(asi_values):.1f} to {np.nanmax(asi_values):.1f}")
    
    # Manual 3-bar pattern detection (what should be found)
    print(f"\n📊 Manual 3-bar pattern detection:")
    hsp_manual = []
    lsp_manual = []
    
    for i in range(1, len(asi_values)-1):
        if np.isnan(asi_values[i-1]) or np.isnan(asi_values[i]) or np.isnan(asi_values[i+1]):
            continue
            
        left = asi_values[i-1]
        middle = asi_values[i]
        right = asi_values[i+1]
        
        # HSP: local maximum
        if middle > left and middle > right:
            hsp_manual.append(i)
            print(f"  Bar {i:2d}: HSP {left:6.1f} < {middle:6.1f} > {right:6.1f}")
        
        # LSP: local minimum  
        if middle < left and middle < right:
            lsp_manual.append(i)
            print(f"  Bar {i:2d}: LSP {left:6.1f} > {middle:6.1f} < {right:6.1f}")
    
    print(f"\nManual detection found:")
    print(f"  HSP: {len(hsp_manual)} points")
    print(f"  LSP: {len(lsp_manual)} points")
    print(f"  Total: {len(hsp_manual) + len(lsp_manual)} swings")
    
    # Test our simple method
    from features.simple_swing_detection import detect_simple_swings_batch
    
    print(f"\n🔧 Testing our simple method:")
    
    # Test without filter
    hsp_flags_raw, lsp_flags_raw = detect_simple_swings_batch(
        asi_values, min_distance=1, use_exceeding_filter=False
    )
    hsp_raw = np.where(hsp_flags_raw)[0]
    lsp_raw = np.where(lsp_flags_raw)[0]
    
    print(f"Raw simple method:")
    print(f"  HSP: {len(hsp_raw)} points")
    print(f"  LSP: {len(lsp_raw)} points")
    print(f"  Total: {len(hsp_raw) + len(lsp_raw)} swings")
    
    # Test with filter
    hsp_flags_filt, lsp_flags_filt = detect_simple_swings_batch(
        asi_values, min_distance=1, use_exceeding_filter=True
    )
    hsp_filt = np.where(hsp_flags_filt)[0]
    lsp_filt = np.where(lsp_flags_filt)[0]
    
    print(f"Filtered simple method:")
    print(f"  HSP: {len(hsp_filt)} points")
    print(f"  LSP: {len(lsp_filt)} points")
    print(f"  Total: {len(hsp_filt) + len(lsp_filt)} swings")
    
    # Compare what we're missing
    print(f"\n❌ Missing points analysis:")
    
    missing_hsp = set(hsp_manual) - set(hsp_raw)
    missing_lsp = set(lsp_manual) - set(lsp_raw)
    
    if missing_hsp:
        print(f"Missing HSP: {list(missing_hsp)}")
        for idx in missing_hsp:
            left = asi_values[idx-1]
            middle = asi_values[idx]
            right = asi_values[idx+1]
            print(f"  Bar {idx}: {left:.1f} < {middle:.1f} > {right:.1f} - should be HSP!")
    
    if missing_lsp:
        print(f"Missing LSP: {list(missing_lsp)}")
        for idx in missing_lsp:
            left = asi_values[idx-1]
            middle = asi_values[idx]
            right = asi_values[idx+1]
            print(f"  Bar {idx}: {left:.1f} > {middle:.1f} < {right:.1f} - should be LSP!")
    
    # Show what raw method actually found
    print(f"\n✅ Raw method found:")
    all_raw = []
    for idx in hsp_raw:
        all_raw.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_raw:
        all_raw.append((idx, 'LSP', asi_values[idx]))
    all_raw.sort()
    
    for bar, type_, asi in all_raw:
        print(f"  Bar {bar:2d}: {type_} at ASI={asi:6.1f}")
    
    if len(hsp_manual) + len(lsp_manual) != len(hsp_raw) + len(lsp_raw):
        print(f"\n🚨 BUG FOUND: Manual detection found {len(hsp_manual) + len(lsp_manual)} but simple method found {len(hsp_raw) + len(lsp_raw)}")
        print(f"The simple method implementation has a bug!")
    else:
        print(f"\n✅ Simple method raw detection is correct!")

if __name__ == "__main__":
    debug_simple_detection()