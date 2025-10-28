#!/usr/bin/env python3
"""
Investigate missing HSP markers around bars 95-100
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def investigate_missing_hsp_95_100():
    """Investigate missing HSP around bars 95-100"""
    
    # Load the same 200-bar data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    test_slice = df.iloc[2000:2200].copy()
    
    print(f"🔍 Investigating missing HSP around bars 95-100")
    
    # Run batch processing
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # Focus on the problematic range
    focus_start = 90
    focus_end = 110
    
    print(f"\n📊 Focusing on bars {focus_start}-{focus_end}:")
    
    # Extract data for this range
    asi_focus = batch_results['asi'].iloc[focus_start:focus_end+1]
    hsp_focus = batch_results['sig_hsp'].iloc[focus_start:focus_end+1]
    lsp_focus = batch_results['sig_lsp'].iloc[focus_start:focus_end+1]
    
    print(f"\nASI values and swing detections:")
    print("Bar |  ASI  | HSP | LSP | Pattern")
    print("----|-------|-----|-----|--------")
    
    missing_hsp_candidates = []
    
    for i, (asi, hsp, lsp) in enumerate(zip(asi_focus, hsp_focus, lsp_focus)):
        bar_num = focus_start + i
        
        # Check if this should be an HSP (3-bar pattern)
        should_be_hsp = ""
        if i > 0 and i < len(asi_focus) - 1:
            left = asi_focus.iloc[i-1]
            middle = asi_focus.iloc[i]
            right = asi_focus.iloc[i+1]
            
            if middle > left and middle > right:
                should_be_hsp = "← SHOULD BE HSP"
                if not hsp:  # Missing HSP
                    missing_hsp_candidates.append((bar_num, middle, left, right))
        
        swing_marker = "HSP" if hsp else "LSP" if lsp else "---"
        print(f"{bar_num:3d} | {asi:5.1f} | {'✓' if hsp else ' '} | {'✓' if lsp else ' '} | {should_be_hsp}")
    
    print(f"\n❌ Missing HSP candidates in this range:")
    for bar, middle, left, right in missing_hsp_candidates:
        print(f"  Bar {bar}: ASI={middle:.1f} (left={left:.1f}, right={right:.1f})")
    
    # Get swing sequence for analysis
    all_swings = []
    for i, (asi, hsp, lsp) in enumerate(zip(batch_results['asi'], batch_results['sig_hsp'], batch_results['sig_lsp'])):
        if hsp:
            all_swings.append((i, 'HSP', asi))
        if lsp:
            all_swings.append((i, 'LSP', asi))
    
    print(f"\nSwing sequence around bars 95-100:")
    relevant_swings = [sw for sw in all_swings if abs(sw[0] - 97) <= 15]  # ±15 bars around 97
    
    for bar, type_, asi in relevant_swings:
        marker = "🎯" if 95 <= bar <= 100 else "  "
        print(f"  {marker} Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Analyze why HSP candidates are missing
    print(f"\n🔧 Wilder alternating rule analysis for missing HSP:")
    
    # Find last significant LSP before each missing HSP
    for bar, middle, left, right in missing_hsp_candidates:
        print(f"\nBar {bar} analysis:")
        print(f"  ASI value: {middle:.1f}")
        print(f"  3-bar pattern: {left:.1f} < {middle:.1f} > {right:.1f} ✓")
        
        # Find last LSP before this bar
        last_lsp_bar = None
        last_lsp_asi = None
        
        for swing_bar, swing_type, swing_asi in all_swings:
            if swing_bar < bar and swing_type == 'LSP':
                last_lsp_bar = swing_bar
                last_lsp_asi = swing_asi
        
        if last_lsp_asi is not None:
            print(f"  Last LSP: Bar {last_lsp_bar}, ASI={last_lsp_asi:.1f}")
            
            # Check if this HSP would be confirmed by ASI later dropping below last LSP
            later_asi_values = batch_results['asi'].iloc[bar+1:focus_end+10]  # Check next 10 bars
            min_later_asi = later_asi_values.min() if len(later_asi_values) > 0 else middle
            
            if min_later_asi < last_lsp_asi:
                print(f"  Wilder confirmation: YES (later ASI {min_later_asi:.1f} < last LSP {last_lsp_asi:.1f})")
                print(f"  → This SHOULD be detected! Possible bug in implementation.")
            else:
                print(f"  Wilder confirmation: NO (later ASI {min_later_asi:.1f} ≥ last LSP {last_lsp_asi:.1f})")
                print(f"  → Correctly NOT detected per Wilder's rule.")
        else:
            print(f"  Last LSP: None found")
            print(f"  → First HSP case - should be detected immediately")
    
    # Create visualization
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
    
    # Mark missing HSP candidates
    for bar, middle, left, right in missing_hsp_candidates:
        ax.scatter(bar, middle, marker='^', s=250, c='orange', edgecolors='black', 
                  linewidth=2, label='Missing HSP' if bar == missing_hsp_candidates[0][0] else "", zorder=9)
        ax.annotate(f'MISSING\nHSP {bar}', xy=(bar, middle), xytext=(bar, middle+15),
                   fontsize=10, ha='center', color='orange', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    # Highlight the 95-100 range
    ax.axvspan(95, 100, alpha=0.2, color='yellow', label='Target Range 95-100')
    
    ax.set_title(f'Missing HSP Investigation: Bars {focus_start}-{focus_end}', fontsize=14)
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('ASI Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = project_root / "data/test/missing_hsp_95_100_investigation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Investigation saved to: {save_path}")
    print(f"🎯 Found {len(missing_hsp_candidates)} missing HSP candidates in range {focus_start}-{focus_end}")
    
    return missing_hsp_candidates

if __name__ == "__main__":
    investigate_missing_hsp_95_100()