#!/usr/bin/env python3
"""
Quick SI Analysis - Check raw SI values before capping
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent

def main():
    print("📊 Quick SI Values Analysis...")
    
    # Load processed data to extract ASI 
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    # Calculate SI values from ASI differences
    asi = df['asi'].values
    si_values = np.diff(asi)  # SI[i] = ASI[i] - ASI[i-1]
    
    # Filter non-zero SI values
    si_nonzero = si_values[si_values != 0]
    
    print(f"\n📊 SI Values Analysis:")
    print(f"Total SI values: {len(si_values)}")
    print(f"Non-zero SI values: {len(si_nonzero)}")
    print(f"SI range: [{np.min(si_nonzero):.2f}, {np.max(si_nonzero):.2f}]")
    print(f"SI mean: {np.mean(si_nonzero):.2f}")
    print(f"SI std: {np.std(si_nonzero):.2f}")
    
    # Check how many exceed ±100
    exceeds_100 = np.sum(np.abs(si_nonzero) > 100)
    print(f"Values exceeding ±100: {exceeds_100}/{len(si_nonzero)} ({exceeds_100/len(si_nonzero)*100:.1f}%)")
    
    # Show extremes
    print(f"\nTop 10 largest SI values:")
    top_si = np.sort(si_nonzero)[-10:]
    print(top_si)
    
    print(f"\nTop 10 smallest SI values:")
    bottom_si = np.sort(si_nonzero)[:10]
    print(bottom_si)
    
    # Percentiles
    print(f"\nSI Percentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(si_nonzero, p)
        print(f"P{p:2d}: {val:6.2f}")

if __name__ == "__main__":
    main()