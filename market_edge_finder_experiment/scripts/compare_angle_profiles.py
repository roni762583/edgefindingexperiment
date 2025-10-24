#!/usr/bin/env python3
"""
Compare Angle Profiles - Side-by-side comparison of linear mapped vs degrees
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("📊 Creating angle profile comparison chart...")
    
    # Load processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    # Load raw data for price context
    raw_path = project_root / "data/test/sample_data.csv"
    raw_df = pd.read_csv(raw_path)
    
    print(f"✅ Loaded {len(df)} processed bars")
    
    # Get stored linear mapped angles (-1 to +1)
    hsp_angles_linear = pd.to_numeric(df['hsp_angles'], errors='coerce')
    lsp_angles_linear = pd.to_numeric(df['lsp_angles'], errors='coerce')
    
    # Convert to degrees for comparison
    hsp_angles_degrees = np.degrees(hsp_angles_linear * (np.pi / 2))
    lsp_angles_degrees = np.degrees(lsp_angles_linear * (np.pi / 2))
    
    # Create time index
    time_index = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else range(len(df))
    
    # Create figure with side-by-side comparison
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Angle Profile Comparison: Linear Mapped vs Degrees\n(Profiles Should Be Identical)', fontsize=16, fontweight='bold')
    
    # Left column: Linear mapped (-1 to +1)
    # Right column: Degrees (-90° to +90°)
    
    # 1. HSP Angles - Linear Mapped
    ax1 = axes[0, 0]
    valid_hsp_linear = ~np.isnan(hsp_angles_linear)
    if valid_hsp_linear.any():
        ax1.plot(time_index, hsp_angles_linear, 'red', linewidth=2, label='HSP Linear Mapped')
        ax1.fill_between(time_index, hsp_angles_linear, alpha=0.3, color='red')
        
        # Add horizontal reference lines
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='±0.5')
        ax1.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Statistics
        mean_hsp = np.nanmean(hsp_angles_linear)
        std_hsp = np.nanstd(hsp_angles_linear)
        ax1.text(0.02, 0.95, f'Linear Mapped HSP:\\nMean: {mean_hsp:.3f}\\nStd: {std_hsp:.3f}\\nRange: [{np.nanmin(hsp_angles_linear):.3f}, {np.nanmax(hsp_angles_linear):.3f}]', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title('HSP Angles: Linear Mapped (-1 to +1)')
    ax1.set_ylabel('Linear Mapped Value')
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. HSP Angles - Degrees  
    ax2 = axes[0, 1]
    valid_hsp_degrees = ~np.isnan(hsp_angles_degrees)
    if valid_hsp_degrees.any():
        ax2.plot(time_index, hsp_angles_degrees, 'red', linewidth=2, label='HSP Degrees')
        ax2.fill_between(time_index, hsp_angles_degrees, alpha=0.3, color='red')
        
        # Add horizontal reference lines
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax2.axhline(45, color='gray', linestyle='--', alpha=0.5, label='±45°')
        ax2.axhline(-45, color='gray', linestyle='--', alpha=0.5)
        
        # Statistics
        mean_hsp_deg = np.nanmean(hsp_angles_degrees)
        std_hsp_deg = np.nanstd(hsp_angles_degrees)
        ax2.text(0.02, 0.95, f'Degrees HSP:\\nMean: {mean_hsp_deg:.1f}°\\nStd: {std_hsp_deg:.1f}°\\nRange: [{np.nanmin(hsp_angles_degrees):.1f}°, {np.nanmax(hsp_angles_degrees):.1f}°]', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('HSP Angles: Degrees (-90° to +90°)')
    ax2.set_ylabel('Angle (Degrees)')
    ax2.set_ylim(-95, 95)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. LSP Angles - Linear Mapped
    ax3 = axes[1, 0]
    valid_lsp_linear = ~np.isnan(lsp_angles_linear)
    if valid_lsp_linear.any():
        ax3.plot(time_index, lsp_angles_linear, 'blue', linewidth=2, label='LSP Linear Mapped')
        ax3.fill_between(time_index, lsp_angles_linear, alpha=0.3, color='blue')
        
        # Add horizontal reference lines
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='±0.5')
        ax3.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Statistics
        mean_lsp = np.nanmean(lsp_angles_linear)
        std_lsp = np.nanstd(lsp_angles_linear)
        ax3.text(0.02, 0.95, f'Linear Mapped LSP:\\nMean: {mean_lsp:.3f}\\nStd: {std_lsp:.3f}\\nRange: [{np.nanmin(lsp_angles_linear):.3f}, {np.nanmax(lsp_angles_linear):.3f}]', 
                transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_title('LSP Angles: Linear Mapped (-1 to +1)')
    ax3.set_ylabel('Linear Mapped Value')
    ax3.set_xlabel('Time')
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LSP Angles - Degrees
    ax4 = axes[1, 1]
    valid_lsp_degrees = ~np.isnan(lsp_angles_degrees)
    if valid_lsp_degrees.any():
        ax4.plot(time_index, lsp_angles_degrees, 'blue', linewidth=2, label='LSP Degrees')
        ax4.fill_between(time_index, lsp_angles_degrees, alpha=0.3, color='blue')
        
        # Add horizontal reference lines
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax4.axhline(45, color='gray', linestyle='--', alpha=0.5, label='±45°')
        ax4.axhline(-45, color='gray', linestyle='--', alpha=0.5)
        
        # Statistics
        mean_lsp_deg = np.nanmean(lsp_angles_degrees)
        std_lsp_deg = np.nanstd(lsp_angles_degrees)
        ax4.text(0.02, 0.95, f'Degrees LSP:\\nMean: {mean_lsp_deg:.1f}°\\nStd: {std_lsp_deg:.1f}°\\nRange: [{np.nanmin(lsp_angles_degrees):.1f}°, {np.nanmax(lsp_angles_degrees):.1f}°]', 
                transform=ax4.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_title('LSP Angles: Degrees (-90° to +90°)')
    ax4.set_ylabel('Angle (Degrees)')
    ax4.set_xlabel('Time')
    ax4.set_ylim(-95, 95)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis for time
    if isinstance(time_index[0], pd.Timestamp):
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/angle_profile_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Verify profiles are identical by correlation
    if valid_hsp_linear.any() and valid_hsp_degrees.any():
        # Normalize both to 0-1 for comparison
        hsp_linear_norm = (hsp_angles_linear + 1) / 2
        hsp_degrees_norm = (hsp_angles_degrees + 90) / 180
        
        valid_both = ~(np.isnan(hsp_linear_norm) | np.isnan(hsp_degrees_norm))
        if valid_both.any():
            corr_hsp = np.corrcoef(hsp_linear_norm[valid_both], hsp_degrees_norm[valid_both])[0,1]
            print(f"HSP profile correlation: {corr_hsp:.10f}")
    
    if valid_lsp_linear.any() and valid_lsp_degrees.any():
        # Normalize both to 0-1 for comparison
        lsp_linear_norm = (lsp_angles_linear + 1) / 2
        lsp_degrees_norm = (lsp_angles_degrees + 90) / 180
        
        valid_both = ~(np.isnan(lsp_linear_norm) | np.isnan(lsp_degrees_norm))
        if valid_both.any():
            corr_lsp = np.corrcoef(lsp_linear_norm[valid_both], lsp_degrees_norm[valid_both])[0,1]
            print(f"LSP profile correlation: {corr_lsp:.10f}")
    
    plt.show()
    print("🎉 Angle profile comparison completed!")
    print("\n📝 Note: Both profiles should appear identical in shape,")
    print("   only the Y-axis scales should differ:")
    print("   • Linear: -1 to +1")
    print("   • Degrees: -90° to +90°")

if __name__ == "__main__":
    main()