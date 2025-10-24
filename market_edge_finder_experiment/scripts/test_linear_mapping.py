#!/usr/bin/env python3
"""
Test Linear Mapping - Compare old vs new angle normalization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def old_normalization(angle_rad):
    """Old normalization: angle_rad / (π/2)"""
    return angle_rad / (np.pi / 2)

def new_linear_mapping(angle_rad):
    """New linear mapping: map degrees (-90, +90) to (-1, +1)"""
    angle_deg = np.degrees(angle_rad)
    # mappedValue = outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin))
    # Map (-90, +90) to (-1, +1)
    return -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))

def main():
    print("🔧 Testing Linear Mapping vs Old Normalization...")
    
    # Test with a range of angle values
    test_angles_deg = np.array([-90, -45, -30, -10, 0, 10, 30, 45, 90])
    test_angles_rad = np.radians(test_angles_deg)
    
    print(f"\n=== Comparison Table ===")
    print(f"{'Degrees':<8} {'Radians':<10} {'Old Norm':<10} {'New Linear':<12} {'Difference':<10}")
    print("-" * 60)
    
    for deg, rad in zip(test_angles_deg, test_angles_rad):
        old_norm = old_normalization(rad)
        new_linear = new_linear_mapping(rad)
        diff = abs(new_linear - old_norm)
        
        print(f"{deg:<8.0f} {rad:<10.3f} {old_norm:<10.3f} {new_linear:<12.3f} {diff:<10.6f}")
    
    # Test with actual processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    # Get current angle values (should be using new linear mapping)
    hsp_angles_new = pd.to_numeric(df['hsp_angles'], errors='coerce')
    lsp_angles_new = pd.to_numeric(df['lsp_angles'], errors='coerce')
    
    print(f"\n=== Actual Data Analysis ===")
    print(f"HSP angles range: [{np.nanmin(hsp_angles_new):.3f}, {np.nanmax(hsp_angles_new):.3f}]")
    print(f"LSP angles range: [{np.nanmin(lsp_angles_new):.3f}, {np.nanmax(lsp_angles_new):.3f}]")
    
    # Check if values are exactly in [-1, 1] range
    hsp_valid = ~np.isnan(hsp_angles_new)
    lsp_valid = ~np.isnan(lsp_angles_new)
    
    hsp_in_range = np.all((hsp_angles_new[hsp_valid] >= -1.0) & (hsp_angles_new[hsp_valid] <= 1.0))
    lsp_in_range = np.all((lsp_angles_new[lsp_valid] >= -1.0) & (lsp_angles_new[lsp_valid] <= 1.0))
    
    print(f"HSP angles all in [-1, 1]: {hsp_in_range}")
    print(f"LSP angles all in [-1, 1]: {lsp_in_range}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Linear Mapping vs Old Normalization Comparison', fontsize=14, fontweight='bold')
    
    # 1. Comparison function plot
    ax1 = axes[0, 0]
    angles_deg_range = np.linspace(-90, 90, 181)
    angles_rad_range = np.radians(angles_deg_range)
    
    old_norms = [old_normalization(rad) for rad in angles_rad_range]
    new_linears = [new_linear_mapping(rad) for rad in angles_rad_range]
    
    ax1.plot(angles_deg_range, old_norms, 'b-', label='Old: angle_rad / (π/2)', linewidth=2)
    ax1.plot(angles_deg_range, new_linears, 'r-', label='New: Linear mapping', linewidth=2)
    ax1.set_xlabel('Input Angle (Degrees)')
    ax1.set_ylabel('Normalized Output')
    ax1.set_title('Normalization Function Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    
    # 2. Difference plot
    ax2 = axes[0, 1]
    differences = np.array(new_linears) - np.array(old_norms)
    ax2.plot(angles_deg_range, differences, 'g-', linewidth=2)
    ax2.set_xlabel('Input Angle (Degrees)')
    ax2.set_ylabel('Difference (New - Old)')
    ax2.set_title('Difference Between Methods')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    
    max_diff = np.max(np.abs(differences))
    ax2.text(0.05, 0.95, f'Max difference: {max_diff:.6f}', transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. HSP angles histogram
    ax3 = axes[1, 0]
    ax3.hist(hsp_angles_new[hsp_valid], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('HSP Angle Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('HSP Angles Distribution (New Linear Mapping)')
    ax3.axvline(-1, color='black', linestyle='--', alpha=0.5, label='±1 bounds')
    ax3.axvline(1, color='black', linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LSP angles histogram
    ax4 = axes[1, 1]
    ax4.hist(lsp_angles_new[lsp_valid], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('LSP Angle Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('LSP Angles Distribution (New Linear Mapping)')
    ax4.axvline(-1, color='black', linestyle='--', alpha=0.5, label='±1 bounds')
    ax4.axvline(1, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/linear_mapping_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved to: {save_path}")
    
    plt.show()
    
    print(f"\n✅ Linear mapping test completed!")
    print(f"The two methods are {'identical' if max_diff < 1e-10 else 'different'}")

if __name__ == "__main__":
    main()