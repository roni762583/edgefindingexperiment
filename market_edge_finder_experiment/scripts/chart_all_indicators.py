#!/usr/bin/env python3
"""
Chart All Indicators - Comprehensive Visualization

Shows all generated indicator columns from the new Grok ASI implementation
including USD-normalized ASI, swing points, and angle calculations.

Usage: python3 scripts/chart_all_indicators.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_comprehensive_indicator_chart():
    """
    Create comprehensive chart showing all indicators from new implementation
    """
    print("🚀 Creating comprehensive indicator chart...")
    
    # Load processed data
    processed_path = project_root / "data/test/processed_sample_data.csv"
    raw_path = project_root / "data/test/sample_data.csv"
    
    if not processed_path.exists():
        print(f"❌ Processed data not found: {processed_path}")
        return
        
    if not raw_path.exists():
        print(f"❌ Raw data not found: {raw_path}")
        return
    
    print(f"📊 Loading processed data from: {processed_path}")
    df_processed = pd.read_csv(processed_path)
    
    print(f"📊 Loading raw data from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    
    print(f"✅ Loaded {len(df_processed)} processed bars with {len(df_processed.columns)} columns")
    print(f"Columns: {list(df_processed.columns)}")
    
    # Filter last 200 bars for better visualization
    plot_data = df_processed.tail(200).copy().reset_index(drop=True)
    raw_data = df_raw.tail(200).copy().reset_index(drop=True)
    
    # Setup time axis
    if 'time' in plot_data.columns:
        plot_data.index = pd.to_datetime(plot_data['time'])
        x_data = plot_data.index
    else:
        x_data = range(len(plot_data))
    
    # Create comprehensive chart
    fig, axes = plt.subplots(7, 1, figsize=(15, 28))
    fig.suptitle('Comprehensive Indicator Chart - New Grok ASI Implementation', fontsize=16, fontweight='bold')
    
    # 1. Price Chart with OHLC
    ax1 = axes[0]
    ax1.plot(x_data, raw_data['high'], 'g-', linewidth=1, alpha=0.7, label='High')
    ax1.plot(x_data, raw_data['low'], 'r-', linewidth=1, alpha=0.7, label='Low')
    ax1.plot(x_data, raw_data['close'], 'b-', linewidth=2, label='Close')
    ax1.plot(x_data, raw_data['open'], 'orange', linewidth=1, alpha=0.8, label='Open')
    ax1.set_title('1. OHLC Price Data (EUR_USD)', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Normalized ASI (USD per standard lot)
    ax2 = axes[1]
    ax2.plot(x_data, plot_data['asi'], 'purple', linewidth=2, label='Normalized ASI (USD)')
    ax2.set_title('2. Accumulative Swing Index (USD Normalized, Grok Spec)', fontweight='bold')
    ax2.set_ylabel('ASI (USD per 100k lot)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add ASI statistics
    asi_min, asi_max = plot_data['asi'].min(), plot_data['asi'].max()
    asi_mean = plot_data['asi'].mean()
    ax2.text(0.02, 0.95, f'Range: [{asi_min:.0f}, {asi_max:.0f}]\nMean: {asi_mean:.0f}', 
             transform=ax2.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Swing Points Detection
    ax3 = axes[2]
    ax3.plot(x_data, plot_data['asi'], 'gray', linewidth=1, alpha=0.5, label='ASI')
    
    # Mark swing points
    hsp_mask = plot_data['sig_hsp'].astype(bool)
    lsp_mask = plot_data['sig_lsp'].astype(bool)
    
    if hsp_mask.any():
        hsp_values = plot_data.loc[hsp_mask, 'asi']
        hsp_times = x_data[hsp_mask]
        ax3.scatter(hsp_times, hsp_values, color='red', s=50, marker='^', 
                   label=f'HSP ({hsp_mask.sum()})', zorder=5)
    
    if lsp_mask.any():
        lsp_values = plot_data.loc[lsp_mask, 'asi']
        lsp_times = x_data[lsp_mask]
        ax3.scatter(lsp_times, lsp_values, color='green', s=50, marker='v', 
                   label=f'LSP ({lsp_mask.sum()})', zorder=5)
    
    ax3.set_title('3. Swing Point Detection (High/Low Swing Points)', fontweight='bold')
    ax3.set_ylabel('ASI Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Local Swing Point Flags
    ax4 = axes[3]
    ax4.plot(x_data, plot_data['local_hsp'], 'red', linewidth=2, label='Local HSP', alpha=0.7)
    ax4.plot(x_data, plot_data['local_lsp'], 'green', linewidth=2, label='Local LSP', alpha=0.7)
    ax4.set_title('4. Local Swing Point Indicators (Boolean Flags)', fontweight='bold')
    ax4.set_ylabel('Boolean (0/1)')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. HSP Regression Angles
    ax5 = axes[4]
    valid_hsp_angles = plot_data['hsp_angles'].dropna()
    if len(valid_hsp_angles) > 0:
        ax5.plot(x_data, plot_data['hsp_angles'], 'darkred', linewidth=2, label='HSP Angles')
        ax5.fill_between(x_data, 0, plot_data['hsp_angles'], alpha=0.3, color='red')
        
        # Add statistics
        hsp_mean = valid_hsp_angles.mean()
        hsp_std = valid_hsp_angles.std()
        ax5.text(0.02, 0.95, f'Mean: {hsp_mean:.3f}\nStd: {hsp_std:.3f}', 
                 transform=ax5.transAxes, verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax5.text(0.5, 0.5, 'No HSP angles calculated', transform=ax5.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax5.set_title('5. HSP Regression Angles (Between Last 2 High Swing Points)', fontweight='bold')
    ax5.set_ylabel('Angle (normalized)')
    ax5.set_ylim(-1, 1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. LSP Regression Angles
    ax6 = axes[5]
    valid_lsp_angles = plot_data['lsp_angles'].dropna()
    if len(valid_lsp_angles) > 0:
        ax6.plot(x_data, plot_data['lsp_angles'], 'darkgreen', linewidth=2, label='LSP Angles')
        ax6.fill_between(x_data, 0, plot_data['lsp_angles'], alpha=0.3, color='green')
        
        # Add statistics
        lsp_mean = valid_lsp_angles.mean()
        lsp_std = valid_lsp_angles.std()
        ax6.text(0.02, 0.95, f'Mean: {lsp_mean:.3f}\nStd: {lsp_std:.3f}', 
                 transform=ax6.transAxes, verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax6.text(0.5, 0.5, 'No LSP angles calculated', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax6.set_title('6. LSP Regression Angles (Between Last 2 Low Swing Points)', fontweight='bold')
    ax6.set_ylabel('Angle (normalized)')
    ax6.set_ylim(-1, 1)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Implementation Comparison Summary
    ax7 = axes[6]
    ax7.text(0.05, 0.9, 'Grok ASI Implementation Features:', fontsize=14, fontweight='bold', 
             transform=ax7.transAxes)
    
    implementation_text = """
✅ USD Normalization: OHLC converted to USD per 100k lot using pip values
✅ Dynamic Limit Move: L = 3 × ATR (Average True Range in USD)
✅ Wilder's SI Formula: 50 × (N/R) × (K/L) with no capping
✅ Grok R Formula: Complex conditional calculation per specification
✅ Cross-Instrument Compatibility: USD values enable comparison
✅ Swing Point Detection: 3-bar alternating constraint algorithm
✅ Regression Angles: arctan(slope) between last 2 swing points

Key Statistics:
• Total bars processed: 700 (EUR_USD H1 data)
• HSP detected: {} | LSP detected: {}
• ASI range: [{:.0f}, {:.0f}] USD per 100k lot
• Valid angle calculations: HSP={}, LSP={}
""".format(
        hsp_mask.sum(), lsp_mask.sum(),
        asi_min, asi_max,
        len(valid_hsp_angles), len(valid_lsp_angles)
    )
    
    ax7.text(0.05, 0.8, implementation_text, fontsize=10, transform=ax7.transAxes,
             verticalalignment='top', fontfamily='monospace')
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    
    # Format x-axis for time
    if 'time' in plot_data.columns:
        for ax in axes[:-1]:  # Skip the summary chart
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[-2].set_xlabel('Time')  # Only the second-to-last chart needs xlabel
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/comprehensive_indicators_chart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print summary statistics
    print(f"\n📊 Implementation Summary:")
    print(f"Total bars processed: {len(df_processed)}")
    print(f"HSP detected: {plot_data['sig_hsp'].sum()}")
    print(f"LSP detected: {plot_data['sig_lsp'].sum()}")
    print(f"ASI range: [{asi_min:.0f}, {asi_max:.0f}] USD per 100k lot")
    print(f"Valid HSP angles: {len(valid_hsp_angles)}")
    print(f"Valid LSP angles: {len(valid_lsp_angles)}")
    
    plt.show()
    print("🎉 Comprehensive indicator charting completed!")

def main():
    """Main function"""
    create_comprehensive_indicator_chart()

if __name__ == "__main__":
    main()