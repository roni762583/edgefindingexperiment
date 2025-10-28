#!/usr/bin/env python3
"""
Debug why we're missing LSP markers in the swing detection
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def debug_missing_lsp():
    """Debug the missing LSP detection issue"""
    
    # Load the same data range that showed missing LSP
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Focus on a smaller section where we can see missing LSP
    test_slice = df.iloc[2050:2120].copy()  # 70 bars around bar 50-120 area
    
    print(f"🔍 Debugging LSP detection on {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Run batch processing
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # Extract ASI values and analyze patterns
    asi_values = batch_results['asi'].values
    
    print(f"\n📊 ASI Analysis:")
    print(f"ASI range: [{asi_values.min():.1f}, {asi_values.max():.1f}]")
    
    # Check for potential LSP candidates manually (3-bar pattern)
    potential_lsp = []
    detected_lsp = []
    detected_hsp = []
    
    # Find actual detected swing points
    for i, (hsp, lsp, asi) in enumerate(zip(batch_results['sig_hsp'], batch_results['sig_lsp'], batch_results['asi'])):
        if hsp:
            detected_hsp.append((i, asi))
        if lsp:
            detected_lsp.append((i, asi))
    
    # Find potential LSP using simple 3-bar pattern
    for i in range(1, len(asi_values) - 1):
        left = asi_values[i-1]
        middle = asi_values[i]
        right = asi_values[i+1]
        
        # LSP: middle lower than both neighbors
        if middle < left and middle < right:
            potential_lsp.append((i, middle))
    
    print(f"\n🎯 Swing Point Analysis:")
    print(f"Detected HSP: {len(detected_hsp)}")
    print(f"Detected LSP: {len(detected_lsp)}")
    print(f"Potential LSP (3-bar): {len(potential_lsp)}")
    
    print(f"\nDetected LSP:")
    for bar, asi in detected_lsp:
        print(f"  Bar {bar:2d}: ASI = {asi:6.1f}")
    
    print(f"\nPotential LSP (should be detected):")
    for bar, asi in potential_lsp[:10]:  # Show first 10
        print(f"  Bar {bar:2d}: ASI = {asi:6.1f}")
    
    print(f"\nMissing LSP (potential but not detected):")
    missing_lsp = []
    for pot_bar, pot_asi in potential_lsp:
        found = False
        for det_bar, det_asi in detected_lsp:
            if abs(pot_bar - det_bar) <= 1:  # Allow 1 bar tolerance
                found = True
                break
        if not found:
            missing_lsp.append((pot_bar, pot_asi))
    
    for bar, asi in missing_lsp[:10]:  # Show first 10 missing
        print(f"  Bar {bar:2d}: ASI = {asi:6.1f} ❌ MISSING")
    
    # Create detailed visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # Plot ASI with all markers
    x_axis = range(len(test_slice))
    ax1.plot(x_axis, asi_values, 'b-', linewidth=2, label='ASI', alpha=0.8)
    
    # Mark detected swing points
    for bar, asi in detected_hsp:
        ax1.scatter(bar, asi, marker='^', s=200, c='red', edgecolors='white', 
                   linewidth=2, label='Detected HSP' if bar == detected_hsp[0][0] else "", zorder=10)
    
    for bar, asi in detected_lsp:
        ax1.scatter(bar, asi, marker='v', s=200, c='blue', edgecolors='white', 
                   linewidth=2, label='Detected LSP' if bar == detected_lsp[0][0] else "", zorder=10)
    
    # Mark missing LSP with different color
    for bar, asi in missing_lsp:
        ax1.scatter(bar, asi, marker='v', s=150, c='orange', edgecolors='black', 
                   linewidth=2, label='Missing LSP' if bar == missing_lsp[0][0] else "", zorder=9)
        ax1.annotate(f'MISSING\n{bar}', xy=(bar, asi), xytext=(bar, asi-20),
                    fontsize=8, ha='center', color='orange', fontweight='bold')
    
    ax1.set_title('LSP Detection Analysis - Missing Points Highlighted', fontsize=14)
    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('ASI Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create analysis table
    ax2.axis('off')
    
    analysis_text = f"""🔍 LSP DETECTION DEBUGGING RESULTS

📊 Detection Summary:
  • Total potential LSP (3-bar pattern): {len(potential_lsp)}
  • Actually detected LSP: {len(detected_lsp)}
  • Missing LSP: {len(missing_lsp)}
  • Detection rate: {len(detected_lsp)/len(potential_lsp)*100:.1f}%

❌ Missing LSP Examples:
"""
    
    # Add details of first few missing LSP
    for i, (bar, asi) in enumerate(missing_lsp[:5]):
        if i < len(asi_values) - 1 and bar > 0:
            left_asi = asi_values[bar-1]
            right_asi = asi_values[bar+1]
            analysis_text += f"  Bar {bar}: ASI={asi:.1f} (left={left_asi:.1f}, right={right_asi:.1f})\n"
    
    analysis_text += f"""
🔧 Possible Causes:
  1. Wilder's alternating rule too strict
  2. Pending/confirmation logic blocking valid LSP
  3. State tracking issues between HSP/LSP
  4. Index calculation problems
  
💡 Next Steps:
  1. Check alternating HSP/LSP enforcement
  2. Review pending candidate logic  
  3. Validate state management
  4. Test with simpler detection rules
"""
    
    ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    save_path = project_root / "data/test/debug_missing_lsp.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Debug analysis saved to: {save_path}")
    print(f"🔍 Missing LSP analysis complete!")
    print(f"⚠️  Detection rate: {len(detected_lsp)/len(potential_lsp)*100:.1f}% - needs improvement")
    
    return batch_results, missing_lsp, potential_lsp

if __name__ == "__main__":
    debug_missing_lsp()