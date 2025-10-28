#!/usr/bin/env python3
"""
Quick comparison of simple vs Wilder swing detection
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def detect_simple_swings(asi_values):
    """Simple 3-bar pattern detection without complex confirmation"""
    hsp_indices = []
    lsp_indices = []
    
    for i in range(1, len(asi_values)-1):
        if np.isnan(asi_values[i-1]) or np.isnan(asi_values[i]) or np.isnan(asi_values[i+1]):
            continue
            
        # HSP: local maximum
        if asi_values[i] > asi_values[i-1] and asi_values[i] > asi_values[i+1]:
            hsp_indices.append(i)
        
        # LSP: local minimum  
        if asi_values[i] < asi_values[i-1] and asi_values[i] < asi_values[i+1]:
            lsp_indices.append(i)
    
    return hsp_indices, lsp_indices

def main():
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    
    # Get ASI from batch processing
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    asi_values = batch_results['asi'].values
    
    print(f"🔍 Comparing Simple vs Wilder Swing Detection")
    
    # Simple method
    hsp_simple, lsp_simple = detect_simple_swings(asi_values)
    
    # Wilder method
    hsp_wilder = np.where(batch_results['sig_hsp'])[0]
    lsp_wilder = np.where(batch_results['sig_lsp'])[0]
    
    print(f"\n📊 Results:")
    print(f"Simple:  HSP={len(hsp_simple):2d}, LSP={len(lsp_simple):2d}, Total={len(hsp_simple)+len(lsp_simple):2d}")
    print(f"Wilder:  HSP={len(hsp_wilder):2d}, LSP={len(lsp_wilder):2d}, Total={len(hsp_wilder)+len(lsp_wilder):2d}")
    
    # Show first 20 of each
    print(f"\n📈 Simple Swings (first 20):")
    all_simple = []
    for idx in hsp_simple:
        all_simple.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_simple:
        all_simple.append((idx, 'LSP', asi_values[idx]))
    all_simple.sort()
    
    for i, (bar, type_, asi) in enumerate(all_simple[:20]):
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\n⚡ Wilder Swings (all {len(hsp_wilder)+len(lsp_wilder)}):")
    all_wilder = []
    for idx in hsp_wilder:
        all_wilder.append((idx, 'HSP', asi_values[idx]))
    for idx in lsp_wilder:
        all_wilder.append((idx, 'LSP', asi_values[idx]))
    all_wilder.sort()
    
    for bar, type_, asi in all_wilder:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # Check where simple finds swings but Wilder doesn't in later bars
    print(f"\n🔍 Simple swings after bar 125 (where Wilder stops):")
    late_simple = [(bar, type_, asi) for bar, type_, asi in all_simple if bar > 125]
    
    for bar, type_, asi in late_simple:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    if len(late_simple) > 0:
        print(f"\n💡 Simple method finds {len(late_simple)} more swings after bar 125!")
        print(f"This suggests our Wilder implementation is too restrictive.")

if __name__ == "__main__":
    main()