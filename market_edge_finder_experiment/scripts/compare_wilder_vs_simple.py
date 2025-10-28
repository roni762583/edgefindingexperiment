#!/usr/bin/env python3
"""
Compare Wilder complex method vs Simple practical method
Both methods preserved - user can choose which to use
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.simple_swing_detection import detect_simple_swings_batch

def compare_wilder_vs_simple():
    """Compare both swing detection methods on same data"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    print(f"🔍 Comparing Wilder vs Simple Swing Detection on 200 bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Get ASI from batch processing (with Wilder swings)
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    asi_values = batch_results['asi'].values
    
    print(f"ASI range: {np.nanmin(asi_values):.1f} to {np.nanmax(asi_values):.1f}")
    
    # Method 1: Wilder (complex, strict alternation)
    hsp_wilder = np.where(batch_results['sig_hsp'])[0]
    lsp_wilder = np.where(batch_results['sig_lsp'])[0]
    
    # Method 2: Simple (practical, more swings)
    hsp_simple_flags, lsp_simple_flags = detect_simple_swings_batch(
        asi_values, min_distance=1, use_exceeding_filter=True
    )
    hsp_simple = np.where(hsp_simple_flags)[0]
    lsp_simple = np.where(lsp_simple_flags)[0]
    
    # Method 3: Simple without filter (maximum swings)
    hsp_raw_flags, lsp_raw_flags = detect_simple_swings_batch(
        asi_values, min_distance=1, use_exceeding_filter=False
    )
    hsp_raw = np.where(hsp_raw_flags)[0]
    lsp_raw = np.where(lsp_raw_flags)[0]
    
    # Compare results
    print(f"\n📊 Results Comparison:")
    print(f"Wilder (complex):     HSP={len(hsp_wilder):2d}, LSP={len(lsp_wilder):2d}, Total={len(hsp_wilder)+len(lsp_wilder):2d}")
    print(f"Simple (filtered):    HSP={len(hsp_simple):2d}, LSP={len(lsp_simple):2d}, Total={len(hsp_simple)+len(lsp_simple):2d}")
    print(f"Simple (raw):         HSP={len(hsp_raw):2d}, LSP={len(lsp_raw):2d}, Total={len(hsp_raw)+len(lsp_raw):2d}")
    
    # Check alternation rates
    def check_alternation(hsp_list, lsp_list):
        all_swings = []
        for idx in hsp_list:
            all_swings.append((idx, 'H'))
        for idx in lsp_list:
            all_swings.append((idx, 'L'))
        all_swings.sort()
        
        if len(all_swings) <= 1:
            return 100.0, 0
            
        violations = 0
        for i in range(1, len(all_swings)):
            if all_swings[i][1] == all_swings[i-1][1]:
                violations += 1
        
        alternation_rate = (len(all_swings) - violations) / len(all_swings) * 100
        return alternation_rate, violations
    
    wilder_alt, wilder_viol = check_alternation(hsp_wilder, lsp_wilder)
    simple_alt, simple_viol = check_alternation(hsp_simple, lsp_simple)
    raw_alt, raw_viol = check_alternation(hsp_raw, lsp_raw)
    
    print(f"\n🔄 Alternation Analysis:")
    print(f"Wilder (complex):     {wilder_alt:5.1f}% ({wilder_viol} violations)")
    print(f"Simple (filtered):    {simple_alt:5.1f}% ({simple_viol} violations)")
    print(f"Simple (raw):         {raw_alt:5.1f}% ({raw_viol} violations)")
    
    # Coverage analysis - how far do swings extend?
    print(f"\n📈 Coverage Analysis:")
    wilder_last = max(max(hsp_wilder, default=0), max(lsp_wilder, default=0))
    simple_last = max(max(hsp_simple, default=0), max(lsp_simple, default=0))
    raw_last = max(max(hsp_raw, default=0), max(lsp_raw, default=0))
    
    print(f"Wilder last swing:    Bar {wilder_last:3d} ({wilder_last/199*100:.1f}% coverage)")
    print(f"Simple last swing:    Bar {simple_last:3d} ({simple_last/199*100:.1f}% coverage)")
    print(f"Raw last swing:       Bar {raw_last:3d} ({raw_last/199*100:.1f}% coverage)")
    
    # Show swing points in different ranges
    def format_swings(hsp_list, lsp_list, asi_values):
        all_swings = []
        for idx in hsp_list:
            all_swings.append((idx, 'HSP', asi_values[idx]))
        for idx in lsp_list:
            all_swings.append((idx, 'LSP', asi_values[idx]))
        all_swings.sort()
        return all_swings
    
    wilder_swings = format_swings(hsp_wilder, lsp_wilder, asi_values)
    simple_swings = format_swings(hsp_simple, lsp_simple, asi_values)
    
    print(f"\n⚡ Wilder Swings (all {len(wilder_swings)}):")
    for bar, type_, asi in wilder_swings:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\n🎯 Simple Swings (last 15 of {len(simple_swings)}):")
    for bar, type_, asi in simple_swings[-15:]:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    x = np.arange(len(asi_values))
    
    # Top plot: ASI with Wilder swings
    ax1.plot(x, asi_values, 'b-', linewidth=1, alpha=0.7, label='ASI')
    ax1.scatter(hsp_wilder, asi_values[hsp_wilder], marker='^', s=150, c='red', 
               edgecolors='white', linewidth=2, label=f'Wilder HSP ({len(hsp_wilder)})')
    ax1.scatter(lsp_wilder, asi_values[lsp_wilder], marker='v', s=150, c='red', 
               edgecolors='white', linewidth=2, label=f'Wilder LSP ({len(lsp_wilder)})')
    
    # Connect Wilder swings
    if len(hsp_wilder) >= 2:
        ax1.plot(hsp_wilder, asi_values[hsp_wilder], 'r--', alpha=0.8, linewidth=2)
    if len(lsp_wilder) >= 2:
        ax1.plot(lsp_wilder, asi_values[lsp_wilder], 'r--', alpha=0.8, linewidth=2)
    
    ax1.set_title(f'Wilder Method: {len(wilder_swings)} Swings, {wilder_alt:.1f}% Alternation', fontsize=12)
    ax1.set_ylabel('ASI Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: ASI with Simple swings
    ax2.plot(x, asi_values, 'b-', linewidth=1, alpha=0.7, label='ASI')
    ax2.scatter(hsp_simple, asi_values[hsp_simple], marker='^', s=100, c='green', 
               edgecolors='white', linewidth=2, label=f'Simple HSP ({len(hsp_simple)})')
    ax2.scatter(lsp_simple, asi_values[lsp_simple], marker='v', s=100, c='green', 
               edgecolors='white', linewidth=2, label=f'Simple LSP ({len(lsp_simple)})')
    
    # Connect Simple swings
    if len(hsp_simple) >= 2:
        ax2.plot(hsp_simple, asi_values[hsp_simple], 'g--', alpha=0.8, linewidth=2)
    if len(lsp_simple) >= 2:
        ax2.plot(lsp_simple, asi_values[lsp_simple], 'g--', alpha=0.8, linewidth=2)
    
    ax2.set_title(f'Simple Method: {len(simple_swings)} Swings, {simple_alt:.1f}% Alternation', fontsize=12)
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('ASI Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = project_root / "data/test/wilder_vs_simple_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Comparison chart saved to: {save_path}")
    
    # Recommendations
    print(f"\n💡 Method Recommendations:")
    print(f"📈 For trend analysis & ML features:")
    print(f"   → Use Simple method ({len(simple_swings)} swings, better coverage)")
    print(f"📊 For strict Wilder 1978 compliance:")
    print(f"   → Use Wilder method ({len(wilder_swings)} swings, perfect alternation)")
    print(f"🎯 For maximum pattern detection:")
    print(f"   → Use Raw simple method ({len(hsp_raw)+len(lsp_raw)} swings)")
    
    return {
        'wilder': (hsp_wilder, lsp_wilder),
        'simple': (hsp_simple, lsp_simple),
        'raw': (hsp_raw, lsp_raw)
    }

if __name__ == "__main__":
    compare_wilder_vs_simple()