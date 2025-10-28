#!/usr/bin/env python3
"""
Analyze the difference between reference simple swing detection vs our complex Wilder method
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def detect_swing_points_simple(asi_values, min_distance=1):
    """
    Simple swing point detection like the reference code
    Find local maxima and minima with minimum distance constraint
    """
    hsp_indices = []
    lsp_indices = []
    
    n = len(asi_values)
    
    for i in range(1, n-1):
        # Skip NaN values
        if np.isnan(asi_values[i-1]) or np.isnan(asi_values[i]) or np.isnan(asi_values[i+1]):
            continue
            
        # HSP: local maximum
        if asi_values[i] > asi_values[i-1] and asi_values[i] > asi_values[i+1]:
            # Check minimum distance from last HSP
            if not hsp_indices or (i - hsp_indices[-1]) >= min_distance:
                hsp_indices.append(i)
        
        # LSP: local minimum  
        if asi_values[i] < asi_values[i-1] and asi_values[i] < asi_values[i+1]:
            # Check minimum distance from last LSP
            if not lsp_indices or (i - lsp_indices[-1]) >= min_distance:
                lsp_indices.append(i)
    
    return hsp_indices, lsp_indices

def filter_by_exceeding_extremes(hsp_indices, lsp_indices, asi_values):
    """
    Reference-style filter: Only keep swing points that exceed previous extremes
    """
    hsp_filtered = []
    lsp_filtered = []
    
    # Combine and sort all swings by index
    all_swings = []
    for idx in hsp_indices:
        all_swings.append((idx, 'H', asi_values[idx]))
    for idx in lsp_indices:
        all_swings.append((idx, 'L', asi_values[idx]))
    
    all_swings.sort(key=lambda x: x[0])
    
    if not all_swings:
        return hsp_filtered, lsp_filtered
    
    # Keep first swing point
    first_idx, first_type, first_asi = all_swings[0]
    if first_type == 'H':
        hsp_filtered.append(first_idx)
        last_hsp_asi = first_asi
        last_lsp_asi = None
    else:
        lsp_filtered.append(first_idx)
        last_lsp_asi = first_asi
        last_hsp_asi = None
    
    # Filter subsequent swings
    for idx, swing_type, asi_val in all_swings[1:]:
        
        if swing_type == 'H':
            # Keep HSP if it's higher than last HSP or if we had an LSP since last HSP
            if last_hsp_asi is None or asi_val > last_hsp_asi:
                hsp_filtered.append(idx)
                last_hsp_asi = asi_val
                
        else:  # swing_type == 'L'
            # Keep LSP if it's lower than last LSP or if we had an HSP since last LSP  
            if last_lsp_asi is None or asi_val < last_lsp_asi:
                lsp_filtered.append(idx)
                last_lsp_asi = asi_val
    
    return hsp_filtered, lsp_filtered

def compare_methods():
    """Compare reference simple method vs our Wilder method"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    
    # Get ASI from batch processing
    sys.path.append(str(project_root))
    from features.feature_engineering import FXFeatureGenerator
    
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    asi_values = batch_results['asi'].values
    
    print(f"🔍 Comparing Swing Detection Methods on 200 bars")
    print(f"ASI range: {np.nanmin(asi_values):.1f} to {np.nanmax(asi_values):.1f}")
    
    # Method 1: Simple detection (like reference)
    hsp_simple, lsp_simple = detect_swing_points_simple(asi_values, min_distance=1)
    
    # Method 2: Simple + exceeding extremes filter
    hsp_filtered, lsp_filtered = filter_by_exceeding_extremes(hsp_simple, lsp_simple, asi_values)
    
    # Method 3: Our Wilder method (from batch processing)
    hsp_wilder = np.where(batch_results['sig_hsp'])[0]
    lsp_wilder = np.where(batch_results['sig_lsp'])[0]
    
    # Compare results
    print(f"\n📊 Results Comparison:")
    print(f"Simple Detection:     HSP={len(hsp_simple):2d}, LSP={len(lsp_simple):2d}")
    print(f"+ Exceeding Filter:   HSP={len(hsp_filtered):2d}, LSP={len(lsp_filtered):2d}")
    print(f"Wilder Method:        HSP={len(hsp_wilder):2d}, LSP={len(lsp_wilder):2d}")
    
    # Check alternation rates
    def check_alternation(hsp_list, lsp_list):
        all_swings = []
        for idx in hsp_list:
            all_swings.append((idx, 'H'))
        for idx in lsp_list:
            all_swings.append((idx, 'L'))
        all_swings.sort()
        
        if len(all_swings) <= 1:
            return 100.0
            
        violations = 0
        for i in range(1, len(all_swings)):
            if all_swings[i][1] == all_swings[i-1][1]:
                violations += 1
        
        return (len(all_swings) - violations) / len(all_swings) * 100
    
    print(f"\n🔄 Alternation Rates:")
    print(f"Simple Detection:     {check_alternation(hsp_simple, lsp_simple):.1f}%")
    print(f"+ Exceeding Filter:   {check_alternation(hsp_filtered, lsp_filtered):.1f}%")
    print(f"Wilder Method:        {check_alternation(hsp_wilder, lsp_wilder):.1f}%")
    
    # Show specific swing points
    print(f"\n📈 Simple Detection Swings (first 20):")
    all_simple = []
    for idx in hsp_simple:
        all_simple.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_simple:
        all_simple.append((idx, 'LSP', asi_values[idx]))
    all_simple.sort()
    
    for i, (bar, type_, asi) in enumerate(all_simple[:20]):
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\n🎯 Filtered Swings (all):")
    all_filtered = []
    for idx in hsp_filtered:
        all_filtered.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_filtered:
        all_filtered.append((idx, 'LSP', asi_values[idx]))
    all_filtered.sort()
    
    for bar, type_, asi in all_filtered:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\n⚡ Wilder Swings (all):")
    all_wilder = []
    for idx in hsp_wilder:
        all_wilder.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_wilder:
        all_wilder.append((idx, 'LSP', asi_values[idx]))
    all_wilder.sort()
    
    for bar, type_, asi in all_wilder:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    x = np.arange(len(asi_values))
    ax.plot(x, asi_values, 'b-', linewidth=1, alpha=0.7, label='ASI')
    
    # Plot different methods
    ax.scatter(hsp_simple, asi_values[hsp_simple], marker='^', s=50, c='orange', 
              alpha=0.7, label=f'Simple HSP ({len(hsp_simple)})')
    ax.scatter(lsp_simple, asi_values[lsp_simple], marker='v', s=50, c='orange', 
              alpha=0.7, label=f'Simple LSP ({len(lsp_simple)})')
    
    ax.scatter(hsp_filtered, asi_values[hsp_filtered], marker='^', s=100, c='green', 
              edgecolors='white', linewidth=2, label=f'Filtered HSP ({len(hsp_filtered)})')
    ax.scatter(lsp_filtered, asi_values[lsp_filtered], marker='v', s=100, c='green', 
              edgecolors='white', linewidth=2, label=f'Filtered LSP ({len(lsp_filtered)})')
    
    ax.scatter(hsp_wilder, asi_values[hsp_wilder], marker='^', s=150, c='red', 
              edgecolors='black', linewidth=2, label=f'Wilder HSP ({len(hsp_wilder)})')
    ax.scatter(lsp_wilder, asi_values[lsp_wilder], marker='v', s=150, c='red', 
              edgecolors='black', linewidth=2, label=f'Wilder LSP ({len(lsp_wilder)})')
    
    ax.set_title('Swing Detection Method Comparison - 200 Bars', fontsize=14)
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('ASI Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = project_root / "data/test/swing_detection_method_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Comparison chart saved to: {save_path}")
    
    return {
        'simple': (hsp_simple, lsp_simple),
        'filtered': (hsp_filtered, lsp_filtered), 
        'wilder': (hsp_wilder, lsp_wilder)
    }

if __name__ == "__main__":
    compare_methods()