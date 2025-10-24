#!/usr/bin/env python3
"""
Test Angle Calculation Direct - Test the angle calculation method directly
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def main():
    print("🔧 Testing angle calculation directly...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    # Get ASI values first
    open_prices = df['open'].values
    high = df['high'].values  
    low = df['low'].values
    close = df['close'].values
    
    print("🚀 Calling ASI calculation...")
    asi, atr_usd, si_values = TechnicalIndicators.calculate_asi_grok_spec(
        open_prices, high, low, close, 'EUR_USD'
    )
    
    print(f"✅ ASI calculation completed!")
    print(f"ASI range: [{np.nanmin(asi):.2f}, {np.nanmax(asi):.2f}]")
    
    # Now test the angle calculation specifically
    print(f"\n🔄 Testing angle calculation...")
    
    # Create test data with simple pattern
    test_asi = np.array([0, 10, 20, 15, 25, 35, 30, 40, 50])  # Simple trend
    test_sig_hsp = np.array([False, False, True, False, False, True, False, False, True])  # HSPs at indices 2, 5, 8
    test_sig_lsp = np.array([False, False, False, True, False, False, True, False, False])  # LSPs at indices 3, 6
    
    print(f"Test ASI: {test_asi}")
    print(f"Test HSPs at indices: {np.where(test_sig_hsp)[0]}")
    print(f"Test LSPs at indices: {np.where(test_sig_lsp)[0]}")
    
    # Test the angle calculation method directly
    tech_indicators = TechnicalIndicators()
    hsp_angles, lsp_angles = tech_indicators._calculate_angle_slopes_between_last_two_hsp_lsp(
        test_asi, test_sig_hsp, test_sig_lsp
    )
    
    print(f"\n📊 Angle calculation results:")
    print(f"HSP angles: {hsp_angles}")
    print(f"LSP angles: {lsp_angles}")
    
    # Check if values are in expected range
    valid_hsp = ~np.isnan(hsp_angles)
    valid_lsp = ~np.isnan(lsp_angles)
    
    if valid_hsp.any():
        print(f"HSP angles range: [{np.nanmin(hsp_angles):.3f}, {np.nanmax(hsp_angles):.3f}]")
        print(f"All HSP in [-1,1]: {np.all((hsp_angles[valid_hsp] >= -1.0) & (hsp_angles[valid_hsp] <= 1.0))}")
    
    if valid_lsp.any():
        print(f"LSP angles range: [{np.nanmin(lsp_angles):.3f}, {np.nanmax(lsp_angles):.3f}]")
        print(f"All LSP in [-1,1]: {np.all((lsp_angles[valid_lsp] >= -1.0) & (lsp_angles[valid_lsp] <= 1.0))}")
    
    # Test the linear mapping formula directly
    print(f"\n🧮 Testing linear mapping formula directly:")
    test_slopes = [0.0, 0.5, 1.0, 2.0, 10.0]  # Different slope values
    
    for slope in test_slopes:
        angle_rad = np.arctan(slope)
        angle_deg = np.degrees(angle_rad)
        
        # Old method
        old_norm = angle_rad / (np.pi / 2)
        
        # New linear mapping method
        new_linear = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
        
        print(f"Slope: {slope:4.1f} -> Rad: {angle_rad:6.3f} -> Deg: {angle_deg:6.1f}° -> Old: {old_norm:6.3f} -> New: {new_linear:6.3f}")

if __name__ == "__main__":
    main()