#!/usr/bin/env python3
"""
Debug Angle Mapping - Verify linear mapping vs degrees conversion
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_linear_mapping_consistency():
    """Test if linear mapping produces the same profile as degrees conversion"""
    
    print("🔧 Testing Linear Mapping Consistency...")
    
    # Load processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    # Get stored linear mapped angles
    hsp_angles_stored = pd.to_numeric(df['hsp_angles'], errors='coerce')
    lsp_angles_stored = pd.to_numeric(df['lsp_angles'], errors='coerce')
    
    print(f"Stored HSP angles range: [{np.nanmin(hsp_angles_stored):.6f}, {np.nanmax(hsp_angles_stored):.6f}]")
    print(f"Stored LSP angles range: [{np.nanmin(lsp_angles_stored):.6f}, {np.nanmax(lsp_angles_stored):.6f}]")
    
    # Convert stored angles back to degrees using the conversion from raw_angles_chart.py
    hsp_degrees_converted = np.degrees(hsp_angles_stored * (np.pi / 2))
    lsp_degrees_converted = np.degrees(lsp_angles_stored * (np.pi / 2))
    
    print(f"Converted to degrees - HSP: [{np.nanmin(hsp_degrees_converted):.1f}°, {np.nanmax(hsp_degrees_converted):.1f}°]")
    print(f"Converted to degrees - LSP: [{np.nanmin(lsp_degrees_converted):.1f}°, {np.nanmax(lsp_degrees_converted):.1f}°]")
    
    # Test the forward and reverse mapping with sample slopes
    print(f"\n🧮 Testing Forward/Reverse Mapping:")
    test_slopes = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
    
    print(f"{'Slope':<8} {'Radians':<10} {'Degrees':<10} {'Linear Map':<12} {'Back to Deg':<12} {'Difference':<10}")
    print("-" * 70)
    
    for slope in test_slopes:
        # Forward calculation
        angle_rad = np.arctan(slope)
        angle_deg = np.degrees(angle_rad)
        
        # Linear mapping (as implemented)
        linear_mapped = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
        
        # Reverse mapping back to degrees
        back_to_deg = np.degrees(linear_mapped * (np.pi / 2))
        
        # Difference
        diff = abs(angle_deg - back_to_deg)
        
        print(f"{slope:<8.1f} {angle_rad:<10.4f} {angle_deg:<10.1f} {linear_mapped:<12.6f} {back_to_deg:<12.1f} {diff:<10.6f}")
    
    # Check if there are any inconsistencies in stored data
    print(f"\n🔍 Checking Stored Data Consistency:")
    
    # Find some actual calculated angles from the data
    valid_hsp_mask = ~np.isnan(hsp_angles_stored)
    valid_lsp_mask = ~np.isnan(lsp_angles_stored)
    
    if valid_hsp_mask.any():
        sample_hsp = hsp_angles_stored[valid_hsp_mask][:10]
        print(f"Sample HSP stored values: {sample_hsp.values}")
        
        # Check if these follow the linear mapping pattern
        sample_hsp_degrees = np.degrees(sample_hsp * (np.pi / 2))
        print(f"Converted to degrees: {sample_hsp_degrees.values}")
    
    # Create a visual comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Angle Mapping Consistency Check', fontsize=14, fontweight='bold')
    
    # Plot 1: Stored linear mapped values
    ax1 = axes[0]
    x_range = range(len(hsp_angles_stored))
    ax1.plot(x_range, hsp_angles_stored, 'r-', label='HSP Linear Mapped', alpha=0.7)
    ax1.plot(x_range, lsp_angles_stored, 'b-', label='LSP Linear Mapped', alpha=0.7)
    ax1.set_ylabel('Linear Mapped Value')
    ax1.set_title('Stored Linear Mapped Angles (-1 to +1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # Plot 2: Converted to degrees
    ax2 = axes[1]
    ax2.plot(x_range, hsp_degrees_converted, 'r-', label='HSP Degrees', alpha=0.7)
    ax2.plot(x_range, lsp_degrees_converted, 'b-', label='LSP Degrees', alpha=0.7)
    ax2.set_ylabel('Angle (Degrees)')
    ax2.set_title('Converted Back to Degrees (-90° to +90°)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-95, 95)
    
    # Plot 3: Normalized comparison (both scaled to 0-1 for visual comparison)
    ax3 = axes[2]
    # Normalize both to 0-1 for profile comparison
    hsp_norm = (hsp_angles_stored + 1) / 2  # -1,1 -> 0,1
    lsp_norm = (lsp_angles_stored + 1) / 2
    hsp_deg_norm = (hsp_degrees_converted + 90) / 180  # -90,90 -> 0,1
    lsp_deg_norm = (lsp_degrees_converted + 90) / 180
    
    ax3.plot(x_range, hsp_norm, 'r-', label='HSP Linear (normalized)', alpha=0.8, linewidth=2)
    ax3.plot(x_range, hsp_deg_norm, 'r--', label='HSP Degrees (normalized)', alpha=0.8, linewidth=1)
    ax3.plot(x_range, lsp_norm, 'b-', label='LSP Linear (normalized)', alpha=0.8, linewidth=2)
    ax3.plot(x_range, lsp_deg_norm, 'b--', label='LSP Degrees (normalized)', alpha=0.8, linewidth=1)
    ax3.set_ylabel('Normalized Value (0-1)')
    ax3.set_xlabel('Bar Index')
    ax3.set_title('Profile Comparison: Should Be Identical')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/angle_mapping_debug.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Debug chart saved to: {save_path}")
    
    plt.show()
    
    # Check for perfect correlation
    valid_both_hsp = ~(np.isnan(hsp_norm) | np.isnan(hsp_deg_norm))
    valid_both_lsp = ~(np.isnan(lsp_norm) | np.isnan(lsp_deg_norm))
    
    if valid_both_hsp.any():
        hsp_corr = np.corrcoef(hsp_norm[valid_both_hsp], hsp_deg_norm[valid_both_hsp])[0,1]
        print(f"HSP correlation between mappings: {hsp_corr:.10f}")
        
    if valid_both_lsp.any():
        lsp_corr = np.corrcoef(lsp_norm[valid_both_lsp], lsp_deg_norm[valid_both_lsp])[0,1]
        print(f"LSP correlation between mappings: {lsp_corr:.10f}")
    
    # Check max differences
    if valid_both_hsp.any():
        hsp_max_diff = np.max(np.abs(hsp_norm[valid_both_hsp] - hsp_deg_norm[valid_both_hsp]))
        print(f"HSP max difference (normalized): {hsp_max_diff:.10f}")
        
    if valid_both_lsp.any():
        lsp_max_diff = np.max(np.abs(lsp_norm[valid_both_lsp] - lsp_deg_norm[valid_both_lsp]))
        print(f"LSP max difference (normalized): {lsp_max_diff:.10f}")

if __name__ == "__main__":
    test_linear_mapping_consistency()