#!/usr/bin/env python3
"""
Chart All Indicators - Comprehensive 4-Indicator System Visualization

Shows complete technical analysis system: ASI, HSP/LSP angles, direction, volatility
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
    
    # Create comprehensive chart with 4-indicator system
    fig, axes = plt.subplots(9, 1, figsize=(18, 32))
    fig.suptitle('Complete 4-Indicator Technical Analysis System', fontsize=16, fontweight='bold')
    
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
    
    # 7. Direction Indicator (ADX-based)
    ax7 = axes[6]
    if 'direction' in plot_data.columns:
        valid_direction = plot_data['direction'].dropna()
        if len(valid_direction) > 0:
            ax7.plot(x_data, plot_data['direction'], 'darkgreen', linewidth=2, label='Direction (ADX-based)')
            ax7.fill_between(x_data, 0, plot_data['direction'], alpha=0.3, color='green')
            
            # Add reference lines for ADX/100 scaling
            ax7.axhline(0.25, color='orange', linestyle='--', alpha=0.7, label='0.25 (ADX=25)')
            ax7.axhline(0.50, color='red', linestyle=':', alpha=0.7, label='0.50 (ADX=50)')
            
            # Add statistics
            dir_mean = valid_direction.mean()
            dir_std = valid_direction.std()
            dir_min, dir_max = valid_direction.min(), valid_direction.max()
            ax7.text(0.02, 0.95, f'Range: [{dir_min:.3f}, {dir_max:.3f}]\\nMean: {dir_mean:.3f}\\nStd: {dir_std:.3f}', 
                     transform=ax7.transAxes, verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax7.text(0.5, 0.5, 'No direction values calculated', transform=ax7.transAxes, 
                    ha='center', va='center', fontsize=12)
    else:
        ax7.text(0.5, 0.5, 'Direction column not found', transform=ax7.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax7.set_title('7. Direction Indicator - ADX/100 Linear Scaling', fontweight='bold')
    ax7.set_ylabel('Direction Strength (ADX/100)')
    ax7.set_ylim(-0.05, 0.7)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Volatility Indicator (ATR z-score)
    ax8 = axes[7]
    if 'volatility' in plot_data.columns:
        valid_volatility = plot_data['volatility'].dropna()
        if len(valid_volatility) > 0:
            ax8.plot(x_data, plot_data['volatility'], 'darkorange', linewidth=2, label='Volatility (Raw z-score)')
            ax8.fill_between(x_data, plot_data['volatility'], alpha=0.3, color='orange')
            
            # Add reference lines for z-score
            ax8.axhline(0, color='black', linestyle='-', alpha=0.7, label='0 (mean)')
            ax8.axhline(1, color='red', linestyle='--', alpha=0.7, label='+1σ (high vol)')
            ax8.axhline(-1, color='blue', linestyle='--', alpha=0.7, label='-1σ (low vol)')
            ax8.axhline(2, color='red', linestyle=':', alpha=0.5, label='+2σ (extreme vol)')
            ax8.axhline(-2, color='blue', linestyle=':', alpha=0.5, label='-2σ (very low vol)')
            
            # Add statistics
            vol_mean = valid_volatility.mean()
            vol_std = valid_volatility.std()
            vol_min, vol_max = valid_volatility.min(), valid_volatility.max()
            ax8.text(0.02, 0.95, f'Range: [{vol_min:.3f}, {vol_max:.3f}]\\nMean: {vol_mean:.3f}\\nStd: {vol_std:.3f}', 
                     transform=ax8.transAxes, verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax8.text(0.5, 0.5, 'No volatility values calculated', transform=ax8.transAxes, 
                    ha='center', va='center', fontsize=12)
    else:
        ax8.text(0.5, 0.5, 'Volatility column not found', transform=ax8.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax8.set_title('8. Volatility Indicator - Raw ATR Z-Score', fontweight='bold')
    ax8.set_ylabel('ATR Z-Score (unbounded)')
    ax8.set_ylim(-2.5, 2.5)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Implementation Summary
    ax9 = axes[8]
    ax9.text(0.05, 0.9, '4-Indicator System Features:', fontsize=14, fontweight='bold', 
             transform=ax9.transAxes)
    
    # Get direction and volatility stats
    dir_stats = f"N/A"
    vol_stats = f"N/A"
    if 'direction' in plot_data.columns and not plot_data['direction'].dropna().empty:
        valid_dir = plot_data['direction'].dropna()
        dir_stats = f"[{valid_dir.min():.3f}, {valid_dir.max():.3f}], Valid={len(valid_dir)}"
    if 'volatility' in plot_data.columns and not plot_data['volatility'].dropna().empty:
        valid_vol = plot_data['volatility'].dropna()
        vol_stats = f"[{valid_vol.min():.3f}, {valid_vol.max():.3f}], Valid={len(valid_vol)}"
    
    implementation_text = f"""
✅ 1. ASI (Accumulative Swing Index): USD normalized, Wilder's formula (50x multiplier)
✅ 2a. HSP Angles: Linear regression slopes between high swing points (-1,+1)
✅ 2b. LSP Angles: Linear regression slopes between low swing points (-1,+1)
✅ 3. Direction: ADX linear scaling, ADX/100 (unbounded)
✅ 4. Volatility: Raw ATR z-score, (ATR-SMA)/STDEV (unbounded)

📊 System Statistics (EUR_USD H1, 700 bars):
• ASI range: [{asi_min:.0f}, {asi_max:.0f}] USD per 100k lot
• Swing points: {hsp_mask.sum()} HSP + {lsp_mask.sum()} LSP detected
• HSP angles: {len(valid_hsp_angles)} valid calculations
• LSP angles: {len(valid_lsp_angles)} valid calculations  
• Direction: {dir_stats}
• Volatility: {vol_stats}

🎯 Modified 4-indicator system with linear scaling operational!
"""
    
    ax9.text(0.05, 0.8, implementation_text, fontsize=10, transform=ax9.transAxes,
             verticalalignment='top', fontfamily='monospace')
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
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