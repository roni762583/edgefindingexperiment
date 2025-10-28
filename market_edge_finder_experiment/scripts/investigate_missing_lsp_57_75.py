#!/usr/bin/env python3
"""
Investigate specifically why LSP markers are missing in bars 57-75 range
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def investigate_missing_lsp_57_75():
    """Investigate the specific missing LSP in bars 57-75"""
    
    # Load the same data used in 200-bar graph
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use the same 200-bar range
    test_slice = df.iloc[2000:2200].copy()
    
    print(f"🔍 Investigating missing LSP in bars 57-75")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Run batch processing
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # Focus on the problematic range
    focus_start = 55
    focus_end = 80
    
    print(f"\n📊 Focusing on bars {focus_start}-{focus_end}:")
    
    # Extract ASI values and swing flags for this range
    asi_focus = batch_results['asi'].iloc[focus_start:focus_end+1]
    hsp_focus = batch_results['sig_hsp'].iloc[focus_start:focus_end+1]
    lsp_focus = batch_results['sig_lsp'].iloc[focus_start:focus_end+1]
    
    print(f"\nASI values and swing detections:")
    print("Bar |  ASI  | HSP | LSP | Pattern")
    print("----|-------|-----|-----|--------")
    
    for i, (asi, hsp, lsp) in enumerate(zip(asi_focus, hsp_focus, lsp_focus)):
        bar_num = focus_start + i
        
        # Check if this should be an LSP (3-bar pattern)
        should_be_lsp = ""
        if i > 0 and i < len(asi_focus) - 1:
            left = asi_focus.iloc[i-1]
            middle = asi_focus.iloc[i]
            right = asi_focus.iloc[i+1]
            
            if middle < left and middle < right:
                should_be_lsp = "← SHOULD BE LSP"
        
        swing_marker = "HSP" if hsp else "LSP" if lsp else "---"
        print(f"{bar_num:3d} | {asi:5.1f} | {'✓' if hsp else ' '} | {'✓' if lsp else ' '} | {should_be_lsp}")
    
    # Find the deepest valleys that are missing
    missing_lsp_candidates = []
    
    for i in range(1, len(asi_focus) - 1):
        left = asi_focus.iloc[i-1]
        middle = asi_focus.iloc[i]
        right = asi_focus.iloc[i+1]
        
        bar_num = focus_start + i
        is_detected_lsp = lsp_focus.iloc[i]
        
        # Should be LSP but not detected
        if middle < left and middle < right and not is_detected_lsp:
            missing_lsp_candidates.append((bar_num, middle, left, right))
    
    print(f"\n❌ Missing LSP candidates in this range:")
    for bar, middle, left, right in missing_lsp_candidates:
        print(f"  Bar {bar}: ASI={middle:.1f} (left={left:.1f}, right={right:.1f})")
    
    # Now let's check WHY these aren't being detected
    # Look at the Wilder alternating rule
    
    print(f"\n🔍 Analyzing Wilder alternating rule violations:")
    
    # Get all detected swing points up to this range
    all_swings = []
    for i, (asi, hsp, lsp) in enumerate(zip(batch_results['asi'], batch_results['sig_hsp'], batch_results['sig_lsp'])):
        if hsp:
            all_swings.append((i, 'HSP', asi))
        if lsp:
            all_swings.append((i, 'LSP', asi))
    
    print(f"\nSwing sequence leading up to bars 57-75:")
    relevant_swings = [sw for sw in all_swings if sw[0] <= focus_end]
    
    for bar, type_, asi in relevant_swings[-10:]:  # Last 10 swings
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Check for alternating pattern violations
    print(f"\n🔧 Alternating pattern analysis:")
    last_hsp_bar = None
    last_lsp_bar = None
    last_hsp_asi = None
    last_lsp_asi = None
    
    for bar, type_, asi in relevant_swings:
        if type_ == 'HSP':
            last_hsp_bar = bar
            last_hsp_asi = asi
        else:
            last_lsp_bar = bar  
            last_lsp_asi = asi
    
    print(f"Last HSP: Bar {last_hsp_bar}, ASI={last_hsp_asi:.1f}")
    print(f"Last LSP: Bar {last_lsp_bar}, ASI={last_lsp_asi:.1f}")
    
    # For each missing LSP, check if it violates Wilder's rule
    for bar, middle, left, right in missing_lsp_candidates:
        print(f"\nBar {bar} analysis:")
        print(f"  ASI value: {middle:.1f}")
        print(f"  3-bar pattern: {left:.1f} > {middle:.1f} < {right:.1f} ✓")
        
        # Check Wilder's alternating rule
        # LSP can only be confirmed if ASI later rises ABOVE last significant HSP
        if last_hsp_asi is not None:
            # Simulate: would this LSP be confirmed by checking if any later ASI goes above last HSP?
            later_asi_values = batch_results['asi'].iloc[bar+1:focus_end+1]
            max_later_asi = later_asi_values.max() if len(later_asi_values) > 0 else middle
            
            if max_later_asi > last_hsp_asi:
                print(f"  Wilder confirmation: YES (later ASI {max_later_asi:.1f} > last HSP {last_hsp_asi:.1f})")
                print(f"  → This SHOULD be detected! Possible bug in implementation.")
            else:
                print(f"  Wilder confirmation: NO (later ASI {max_later_asi:.1f} ≤ last HSP {last_hsp_asi:.1f})")
                print(f"  → Correctly NOT detected per Wilder's rule.")
        else:
            print(f"  Wilder confirmation: N/A (no prior HSP)")
    
    # Create detailed visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot ASI for the focus range
    x_range = range(focus_start, focus_end+1)
    ax.plot(x_range, asi_focus.values, 'b-', linewidth=2, label='ASI', alpha=0.8)
    
    # Mark detected swing points
    for i, (hsp, lsp) in enumerate(zip(hsp_focus, lsp_focus)):
        bar_num = focus_start + i
        asi_val = asi_focus.iloc[i]
        
        if hsp:
            ax.scatter(bar_num, asi_val, marker='^', s=300, c='red', edgecolors='white', 
                      linewidth=2, label='Detected HSP' if i == 0 else "", zorder=10)
        if lsp:
            ax.scatter(bar_num, asi_val, marker='v', s=300, c='blue', edgecolors='white', 
                      linewidth=2, label='Detected LSP' if i == 0 else "", zorder=10)
    
    # Mark missing LSP candidates
    for bar, middle, left, right in missing_lsp_candidates:
        ax.scatter(bar, middle, marker='v', s=250, c='orange', edgecolors='black', 
                  linewidth=2, label='Missing LSP' if bar == missing_lsp_candidates[0][0] else "", zorder=9)
        ax.annotate(f'MISSING\nLSP {bar}', xy=(bar, middle), xytext=(bar, middle-15),
                   fontsize=10, ha='center', color='orange', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    ax.set_title(f'Missing LSP Investigation: Bars {focus_start}-{focus_end}', fontsize=14)
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('ASI Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = project_root / "data/test/missing_lsp_57_75_investigation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Investigation saved to: {save_path}")
    print(f"🎯 Found {len(missing_lsp_candidates)} missing LSP candidates in range {focus_start}-{focus_end}")
    
    return missing_lsp_candidates, relevant_swings

if __name__ == "__main__":
    investigate_missing_lsp_57_75()