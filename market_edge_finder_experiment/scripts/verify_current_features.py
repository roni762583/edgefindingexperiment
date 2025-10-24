#!/usr/bin/env python3
"""
Verify Current Features - Quick analysis of processed sample data
"""

import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent

def main():
    print("🔍 Verifying Current Feature Implementation...")
    
    # Load processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    print(f"\n📊 Processed Data Overview:")
    print(f"Total bars: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # ASI Analysis
    print(f"\n🎯 ASI (Accumulative Swing Index):")
    asi_min, asi_max = df['asi'].min(), df['asi'].max()
    print(f"Range: [{asi_min:.0f}, {asi_max:.0f}] USD per 100k lot")
    print(f"Mean: {df['asi'].mean():.1f}")
    print(f"Non-zero values: {(df['asi'] != 0).sum()}")
    
    # SI Analysis (from ASI differences)
    si_values = np.diff(df['asi'].values)
    si_nonzero = si_values[si_values != 0]
    print(f"\n🎯 SI (Swing Index - 50x multiplier):")
    print(f"Range: [{si_nonzero.min():.0f}, {si_nonzero.max():.0f}]")
    print(f"Mean: {si_nonzero.mean():.1f}")
    print(f"Non-zero values: {len(si_nonzero)}")
    print(f"Values exceeding ±100: {np.sum(np.abs(si_nonzero) > 100)}")
    
    # Swing Points Analysis
    hsp_count = df['sig_hsp'].sum()
    lsp_count = df['sig_lsp'].sum()
    print(f"\n🎯 Significant Swing Points:")
    print(f"HSP (High Swing Points): {hsp_count}")
    print(f"LSP (Low Swing Points): {lsp_count}")
    print(f"Local HSP: {df['local_hsp'].sum()}")
    print(f"Local LSP: {df['local_lsp'].sum()}")
    
    # Angles Analysis
    hsp_angles = pd.to_numeric(df['hsp_angles'], errors='coerce')
    lsp_angles = pd.to_numeric(df['lsp_angles'], errors='coerce')
    
    hsp_valid = ~np.isnan(hsp_angles)
    lsp_valid = ~np.isnan(lsp_angles)
    
    print(f"\n🎯 Linear Mapped Angles:")
    print(f"HSP angles: {hsp_valid.sum()}/{len(df)} valid values")
    if hsp_valid.any():
        print(f"  Range: [{hsp_angles.min():.3f}, {hsp_angles.max():.3f}]")
        print(f"  Mean: {hsp_angles.mean():.3f}")
        
    print(f"LSP angles: {lsp_valid.sum()}/{len(df)} valid values")
    if lsp_valid.any():
        print(f"  Range: [{lsp_angles.min():.3f}, {lsp_angles.max():.3f}]")
        print(f"  Mean: {lsp_angles.mean():.3f}")
    
    # Convert back to degrees for verification
    hsp_degrees = np.degrees(hsp_angles * (np.pi / 2))
    lsp_degrees = np.degrees(lsp_angles * (np.pi / 2))
    
    print(f"\n🎯 Angles in Degrees (converted back):")
    if hsp_valid.any():
        print(f"HSP: [{np.nanmin(hsp_degrees):.1f}°, {np.nanmax(hsp_degrees):.1f}°]")
    if lsp_valid.any():
        print(f"LSP: [{np.nanmin(lsp_degrees):.1f}°, {np.nanmax(lsp_degrees):.1f}°]")
    
    # Show sample rows with features
    print(f"\n📋 Sample Feature Rows (with swing points):")
    swing_rows = df[(df['sig_hsp'] == True) | (df['sig_lsp'] == True)].head(5)
    for idx, row in swing_rows.iterrows():
        swing_type = "HSP" if row['sig_hsp'] else "LSP"
        print(f"Bar {idx}: {swing_type}, ASI={row['asi']:.0f}, HSP_angle={row['hsp_angles']:.3f}, LSP_angle={row['lsp_angles']:.3f}")
    
    print(f"\n✅ Feature verification completed!")
    print(f"\n📊 Chart files available in data/test/:")
    print(f"• comprehensive_indicators_chart.png - All indicators overview")
    print(f"• raw_angles_chart.png - Detailed angle analysis")
    print(f"• linear_mapping_comparison.png - Linear mapping verification")

if __name__ == "__main__":
    main()