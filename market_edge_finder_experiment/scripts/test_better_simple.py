#!/usr/bin/env python3
"""
Test a better simple swing detection approach
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def detect_practical_swings(asi_values, min_distance=3):
    """
    Practical swing detection - like reference code
    Find all 3-bar patterns with minimum distance between same-type swings
    """
    hsp_indices = []
    lsp_indices = []
    
    # Step 1: Find all 3-bar patterns
    for i in range(1, len(asi_values)-1):
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

def test_practical_approach():
    """Test the practical approach on 200 bars"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    
    # Get ASI
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    asi_values = batch_results['asi'].values
    
    print(f"🔍 Testing Practical Swing Detection on 200 bars")
    print(f"ASI range: {np.nanmin(asi_values):.1f} to {np.nanmax(asi_values):.1f}")
    
    # Test different minimum distances
    for min_dist in [1, 3, 5]:
        hsp_indices, lsp_indices = detect_practical_swings(asi_values, min_distance=min_dist)
        
        total_swings = len(hsp_indices) + len(lsp_indices)
        last_swing = max(max(hsp_indices, default=0), max(lsp_indices, default=0))
        coverage = last_swing / 199 * 100
        
        print(f"\nMin Distance {min_dist}:")
        print(f"  HSP: {len(hsp_indices):2d}, LSP: {len(lsp_indices):2d}, Total: {total_swings:2d}")
        print(f"  Coverage: {coverage:5.1f}% (last swing at bar {last_swing})")
    
    # Use min_distance=3 for detailed analysis
    hsp_indices, lsp_indices = detect_practical_swings(asi_values, min_distance=3)
    
    # Show swing distribution
    print(f"\n📈 Swing Distribution (min_distance=3):")
    all_swings = []
    for idx in hsp_indices:
        all_swings.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_indices:
        all_swings.append((idx, 'LSP', asi_values[idx]))
    all_swings.sort()
    
    print(f"First 10 swings:")
    for bar, type_, asi in all_swings[:10]:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"Last 10 swings:")
    for bar, type_, asi in all_swings[-10:]:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    x = np.arange(len(asi_values))
    ax.plot(x, asi_values, 'b-', linewidth=1.5, alpha=0.8, label='ASI')
    
    # Plot swing points
    ax.scatter(hsp_indices, asi_values[hsp_indices], marker='^', s=100, c='red', 
              edgecolors='white', linewidth=2, label=f'HSP ({len(hsp_indices)})', zorder=10)
    ax.scatter(lsp_indices, asi_values[lsp_indices], marker='v', s=100, c='blue', 
              edgecolors='white', linewidth=2, label=f'LSP ({len(lsp_indices)})', zorder=10)
    
    # Connect swing points
    if len(hsp_indices) >= 2:
        ax.plot(hsp_indices, asi_values[hsp_indices], 'r--', linewidth=2, alpha=0.8, zorder=5)
    if len(lsp_indices) >= 2:
        ax.plot(lsp_indices, asi_values[lsp_indices], 'b--', linewidth=2, alpha=0.8, zorder=5)
    
    ax.set_title(f'Practical Swing Detection: {len(all_swings)} Swings (min_distance=3)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('ASI Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = project_root / "data/test/practical_swing_detection_200bars.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Practical swing detection saved to: {save_path}")
    
    last_swing = max(max(hsp_indices, default=0), max(lsp_indices, default=0))
    print(f"🎯 Result: {len(all_swings)} swings with {last_swing/199*100:.1f}% coverage")
    print(f"This is much better than Wilder's 8 swings with 54% coverage!")
    
    return hsp_indices, lsp_indices

if __name__ == "__main__":
    test_practical_approach()