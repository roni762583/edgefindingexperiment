#!/usr/bin/env python3
"""
Quick Angle Statistics - Show raw angle statistics
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("📊 Analyzing raw angle statistics...")
    
    # Load processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    # Convert normalized angles back to radians and degrees
    hsp_angles_norm = pd.to_numeric(df['hsp_angles'], errors='coerce')
    lsp_angles_norm = pd.to_numeric(df['lsp_angles'], errors='coerce')
    
    # Convert from (-1, +1) back to (-π/2, +π/2) radians
    hsp_angles_rad = hsp_angles_norm * (np.pi / 2)
    lsp_angles_rad = lsp_angles_norm * (np.pi / 2)
    
    # Convert radians to degrees
    hsp_angles_deg = np.degrees(hsp_angles_rad)
    lsp_angles_deg = np.degrees(lsp_angles_rad)
    
    # Check valid values
    valid_hsp = ~np.isnan(hsp_angles_deg)
    valid_lsp = ~np.isnan(lsp_angles_deg)
    
    print(f"\n=== Raw Angle Analysis ===")
    print(f"Total bars: {len(df)}")
    print(f"HSP swing points: {df['sig_hsp'].sum()}")
    print(f"LSP swing points: {df['sig_lsp'].sum()}")
    
    print(f"\n=== HSP Angles (High Swing Points) ===")
    print(f"Valid angle values: {np.sum(valid_hsp)}")
    if valid_hsp.any():
        print(f"Range: [{np.nanmin(hsp_angles_deg):.1f}°, {np.nanmax(hsp_angles_deg):.1f}°]")
        print(f"Mean: {np.nanmean(hsp_angles_deg):.1f}°")
        print(f"Std: {np.nanstd(hsp_angles_deg):.1f}°")
        print(f"Median: {np.nanmedian(hsp_angles_deg):.1f}°")
        
        # Show some sample values
        non_nan_hsp = hsp_angles_deg[valid_hsp]
        print(f"Sample values: {non_nan_hsp[:10]}")
    
    print(f"\n=== LSP Angles (Low Swing Points) ===")
    print(f"Valid angle values: {np.sum(valid_lsp)}")
    if valid_lsp.any():
        print(f"Range: [{np.nanmin(lsp_angles_deg):.1f}°, {np.nanmax(lsp_angles_deg):.1f}°]")
        print(f"Mean: {np.nanmean(lsp_angles_deg):.1f}°")
        print(f"Std: {np.nanstd(lsp_angles_deg):.1f}°")
        print(f"Median: {np.nanmedian(lsp_angles_deg):.1f}°")
        
        # Show some sample values
        non_nan_lsp = lsp_angles_deg[valid_lsp]
        print(f"Sample values: {non_nan_lsp[:10]}")
    
    print(f"\n=== Intermediate Columns Available ===")
    print(f"Columns in processed data: {list(df.columns)}")
    
    # Check if we have the raw calculation data
    print(f"\n=== Data Availability ===")
    print(f"asi: {df['asi'].count()} non-null values")
    print(f"hsp_angles: {df['hsp_angles'].count()} non-null values") 
    print(f"lsp_angles: {df['lsp_angles'].count()} non-null values")
    
    print(f"\nChart saved to: {project_root}/data/test/raw_angles_chart.png")

if __name__ == "__main__":
    main()