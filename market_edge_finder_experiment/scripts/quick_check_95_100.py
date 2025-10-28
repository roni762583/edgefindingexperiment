#!/usr/bin/env python3
"""
Quick check for missing swing points around bars 95-100
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def quick_check_95_100():
    """Quick analysis of bars 95-100"""
    
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    test_slice = df.iloc[2000:2200].copy()
    
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # Focus on bars 90-110
    start_bar = 90
    end_bar = 110
    
    print(f"🔍 Quick analysis of bars {start_bar}-{end_bar}")
    print("\nBar | ASI   | HSP | LSP | Should be HSP? | Should be LSP?")
    print("----|-------|-----|-----|----------------|----------------")
    
    for i in range(start_bar, end_bar+1):
        if i < len(batch_results):
            asi = batch_results['asi'].iloc[i]
            hsp = batch_results['sig_hsp'].iloc[i]
            lsp = batch_results['sig_lsp'].iloc[i]
            
            # Check 3-bar patterns
            should_hsp = ""
            should_lsp = ""
            
            if i > 0 and i < len(batch_results) - 1:
                left = batch_results['asi'].iloc[i-1]
                middle = batch_results['asi'].iloc[i]
                right = batch_results['asi'].iloc[i+1]
                
                if middle > left and middle > right:
                    should_hsp = "YES"
                if middle < left and middle < right:
                    should_lsp = "YES"
            
            hsp_mark = "✓" if hsp else " "
            lsp_mark = "✓" if lsp else " "
            
            print(f"{i:3d} | {asi:5.1f} |  {hsp_mark}  |  {lsp_mark}  | {should_hsp:14s} | {should_lsp}")

if __name__ == "__main__":
    quick_check_95_100()