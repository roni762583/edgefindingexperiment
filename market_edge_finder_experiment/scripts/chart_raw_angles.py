#!/usr/bin/env python3
"""
Chart Raw Angles - Show angle indicators without normalization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("📊 Creating raw angle indicators chart...")
    
    # Load processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    df = pd.read_csv(processed_path)
    
    # Load raw data for price context
    raw_path = project_root / "data/test/sample_data.csv"
    raw_df = pd.read_csv(raw_path)
    
    print(f"✅ Loaded {len(df)} processed bars")
    print(f"Columns: {list(df.columns)}")
    
    # Convert normalized angles back to radians
    hsp_angles_norm = pd.to_numeric(df['hsp_angles'], errors='coerce')
    lsp_angles_norm = pd.to_numeric(df['lsp_angles'], errors='coerce')
    
    # Convert from (-1, +1) back to (-π/2, +π/2) radians
    hsp_angles_rad = hsp_angles_norm * (np.pi / 2)
    lsp_angles_rad = lsp_angles_norm * (np.pi / 2)
    
    # Convert radians to degrees for easier interpretation
    hsp_angles_deg = np.degrees(hsp_angles_rad)
    lsp_angles_deg = np.degrees(lsp_angles_rad)
    
    # Create time index
    time_index = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else range(len(df))
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle('Raw Angle Analysis - ASI Regression Lines', fontsize=16, fontweight='bold')
    
    # 1. Price Chart with Swing Points
    ax1 = axes[0]
    ax1.plot(time_index, raw_df['close'], 'k-', linewidth=1, label='Close Price')
    
    # Mark swing points
    hsp_mask = df['sig_hsp'].astype(bool)
    lsp_mask = df['sig_lsp'].astype(bool)
    
    if hsp_mask.any():
        hsp_times = np.array(time_index)[hsp_mask]
        hsp_prices = raw_df['close'].values[hsp_mask]
        ax1.scatter(hsp_times, hsp_prices, color='red', s=50, marker='^', label=f'HSP ({hsp_mask.sum()})', zorder=5)
    
    if lsp_mask.any():
        lsp_times = np.array(time_index)[lsp_mask]
        lsp_prices = raw_df['close'].values[lsp_mask]
        ax1.scatter(lsp_times, lsp_prices, color='blue', s=50, marker='v', label=f'LSP ({lsp_mask.sum()})', zorder=5)
    
    ax1.set_title('EUR/USD Price with Significant Swing Points')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ASI Values
    ax2 = axes[1]
    ax2.plot(time_index, df['asi'], 'purple', linewidth=1.5, label='ASI')
    ax2.set_title('Accumulative Swing Index (ASI)')
    ax2.set_ylabel('ASI (USD per 100k lot)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. HSP Angles (Raw Degrees)
    ax3 = axes[2]
    valid_hsp = ~np.isnan(hsp_angles_deg)
    if valid_hsp.any():
        ax3.plot(time_index, hsp_angles_deg, 'red', linewidth=2, label='HSP Regression Angle')
        ax3.fill_between(time_index, hsp_angles_deg, alpha=0.3, color='red')
        
        # Add horizontal reference lines
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax3.axhline(45, color='gray', linestyle='--', alpha=0.5, label='±45°')
        ax3.axhline(-45, color='gray', linestyle='--', alpha=0.5)
        
        # Statistics
        mean_hsp = np.nanmean(hsp_angles_deg)
        std_hsp = np.nanstd(hsp_angles_deg)
        ax3.text(0.02, 0.95, f'HSP Angle Stats:\\nMean: {mean_hsp:.1f}°\\nStd: {std_hsp:.1f}°', 
                transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_title('High Swing Point (HSP) Regression Angles')
    ax3.set_ylabel('Angle (Degrees)')
    ax3.set_ylim(-90, 90)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LSP Angles (Raw Degrees)
    ax4 = axes[3]
    valid_lsp = ~np.isnan(lsp_angles_deg)
    if valid_lsp.any():
        ax4.plot(time_index, lsp_angles_deg, 'blue', linewidth=2, label='LSP Regression Angle')
        ax4.fill_between(time_index, lsp_angles_deg, alpha=0.3, color='blue')
        
        # Add horizontal reference lines
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax4.axhline(45, color='gray', linestyle='--', alpha=0.5, label='±45°')
        ax4.axhline(-45, color='gray', linestyle='--', alpha=0.5)
        
        # Statistics
        mean_lsp = np.nanmean(lsp_angles_deg)
        std_lsp = np.nanstd(lsp_angles_deg)
        ax4.text(0.02, 0.95, f'LSP Angle Stats:\\nMean: {mean_lsp:.1f}°\\nStd: {std_lsp:.1f}°', 
                transform=ax4.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_title('Low Swing Point (LSP) Regression Angles')
    ax4.set_ylabel('Angle (Degrees)')
    ax4.set_ylim(-90, 90)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Angle Summary and Formulas
    ax5 = axes[4]
    ax5.axis('off')
    
    summary_text = f"""
Raw Angle Calculation Analysis:

📊 Angle Statistics:
• HSP angles: {np.sum(valid_hsp)} valid values
• LSP angles: {np.sum(valid_lsp)} valid values
• HSP range: [{np.nanmin(hsp_angles_deg):.1f}°, {np.nanmax(hsp_angles_deg):.1f}°]
• LSP range: [{np.nanmin(lsp_angles_deg):.1f}°, {np.nanmax(lsp_angles_deg):.1f}°]

🔧 Calculation Formulas:
1. Slope = (ASI₂ - ASI₁) / (bar₂ - bar₁)  [USD per bar]
2. Raw Angle = arctan(slope)  [radians: -π/2 to +π/2]
3. Angle in Degrees = Raw Angle × (180/π)  [degrees: -90° to +90°]
4. Normalized = Raw Angle / (π/2)  [normalized: -1 to +1]

📈 Interpretation:
• Positive angles: Upward trend between swing points
• Negative angles: Downward trend between swing points  
• ±45°: Moderate trend strength
• ±90°: Extreme vertical trend (theoretical maximum)

💡 Implementation:
• Forward-filled: Previous angle maintained until new regression available
• Based on last 2 significant swing points of same type (HSP or LSP)
• Cross-instrument comparable via USD normalization
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    # Format x-axis for time
    if isinstance(time_index[0], pd.Timestamp):
        for ax in axes[:-1]:  # Skip the summary chart
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[-2].set_xlabel('Time')  # Only the second-to-last chart needs xlabel
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/raw_angles_chart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print detailed statistics
    print(f"\n📊 Raw Angle Analysis Summary:")
    print(f"HSP valid angles: {np.sum(valid_hsp)}")
    print(f"LSP valid angles: {np.sum(valid_lsp)}")
    if valid_hsp.any():
        print(f"HSP angle range: [{np.nanmin(hsp_angles_deg):.1f}°, {np.nanmax(hsp_angles_deg):.1f}°]")
        print(f"HSP angle mean: {np.nanmean(hsp_angles_deg):.1f}°")
    if valid_lsp.any():
        print(f"LSP angle range: [{np.nanmin(lsp_angles_deg):.1f}°, {np.nanmax(lsp_angles_deg):.1f}°]")
        print(f"LSP angle mean: {np.nanmean(lsp_angles_deg):.1f}°")
    
    plt.show()
    print("🎉 Raw angle analysis completed!")

if __name__ == "__main__":
    main()