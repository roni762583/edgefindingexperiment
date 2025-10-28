#!/usr/bin/env python3
"""
Compare ATR/(pipsize*25) vs other pip scaling methods
Shows high-resolution pip scaling where 1.0 = 25 pips
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def main():
    print("📊 Comparing ATR/(pipsize*25) vs other pip scaling methods...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Extract OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate raw ATR
    atr_raw = TechnicalIndicators.calculate_atr(high, low, close, period=14)
    
    # For EUR/USD: 1 pip = 0.0001
    pip_size = 0.0001
    
    # Calculate different linear normalizations
    atr_pips = atr_raw / pip_size              # Direct pips
    atr_pip_100 = atr_raw / (pip_size * 100)   # 1.0 = 100 pips
    atr_pip_50 = atr_raw / (pip_size * 50)     # 1.0 = 50 pips
    atr_pip_25 = atr_raw / (pip_size * 25)     # 1.0 = 25 pips
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_pips = atr_pips[~np.isnan(atr_pips)]
    valid_100 = atr_pip_100[~np.isnan(atr_pip_100)]
    valid_50 = atr_pip_50[~np.isnan(atr_pip_50)]
    valid_25 = atr_pip_25[~np.isnan(atr_pip_25)]
    
    print(f"\n📊 Statistics:")
    print(f"ATR in Pips:")
    print(f"  Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips")
    print(f"  Mean: {valid_pips.mean():.1f} pips")
    
    print(f"\nATR/(pipsize*100) - 100-pip reference:")
    print(f"  Range: [{valid_100.min():.3f}, {valid_100.max():.3f}]")
    print(f"  Mean: {valid_100.mean():.3f}")
    
    print(f"\nATR/(pipsize*50) - 50-pip reference:")
    print(f"  Range: [{valid_50.min():.3f}, {valid_50.max():.3f}]")
    print(f"  Mean: {valid_50.mean():.3f}")
    
    print(f"\nATR/(pipsize*25) - 25-pip reference:")
    print(f"  Range: [{valid_25.min():.3f}, {valid_25.max():.3f}]")
    print(f"  Mean: {valid_25.mean():.3f}")
    print(f"  Interpretation: 1.0 = 25 pips")
    
    # Verify the relationships
    print(f"\n🔍 Scaling Relationships:")
    print(f"  ÷25 = 4 × ÷100: {valid_25.mean() / valid_100.mean():.3f} (should be 4.0)")
    print(f"  ÷25 = 2 × ÷50: {valid_25.mean() / valid_50.mean():.3f} (should be 2.0)")
    
    # Create comparison plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle('Linear ATR Scaling Comparison: 25-pip High Resolution\\nEUR/USD H1 Data', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: ATR in Pips (reference)
    ax1 = axes[0]
    ax1.plot(time_index, atr_pips, 'blue', linewidth=1.5, label='ATR in Pips')
    ax1.fill_between(time_index, atr_pips, alpha=0.3, color='blue')
    
    # Add reference lines
    ax1.axhline(10, color='green', linestyle='--', alpha=0.7, label='10 pips')
    ax1.axhline(15, color='cyan', linestyle='--', alpha=0.7, label='15 pips')
    ax1.axhline(20, color='orange', linestyle='--', alpha=0.7, label='20 pips')
    ax1.axhline(25, color='red', linestyle='--', alpha=0.7, label='25 pips')
    ax1.axhline(30, color='purple', linestyle=':', alpha=0.5, label='30 pips')
    
    ax1.set_title('1. ATR in Pips - Reference Measurement', fontweight='bold')
    ax1.set_ylabel('ATR (Pips)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_pips) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips\\nMean: {valid_pips.mean():.1f} pips', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: ATR/(pipsize*100)
    ax2 = axes[1]
    ax2.plot(time_index, atr_pip_100, 'green', linewidth=1.5, label='ATR/(pipsize*100)')
    ax2.fill_between(time_index, atr_pip_100, alpha=0.3, color='green')
    
    ax2.axhline(0.1, color='green', linestyle='--', alpha=0.7, label='0.1 (10 pips)')
    ax2.axhline(0.25, color='red', linestyle='--', alpha=0.7, label='0.25 (25 pips)')
    ax2.axhline(0.5, color='purple', linestyle=':', alpha=0.5, label='0.5 (50 pips)')
    
    ax2.set_title('2. ATR/(pipsize*100) - 100-pip Reference', fontweight='bold')
    ax2.set_ylabel('Normalized Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ATR/(pipsize*50)
    ax3 = axes[2]
    ax3.plot(time_index, atr_pip_50, 'orange', linewidth=1.5, label='ATR/(pipsize*50)')
    ax3.fill_between(time_index, atr_pip_50, alpha=0.3, color='orange')
    
    ax3.axhline(0.2, color='green', linestyle='--', alpha=0.7, label='0.2 (10 pips)')
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 (25 pips)')
    ax3.axhline(1.0, color='purple', linestyle=':', alpha=0.5, label='1.0 (50 pips)')
    
    ax3.set_title('3. ATR/(pipsize*50) - 50-pip Reference', fontweight='bold')
    ax3.set_ylabel('Normalized Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ATR/(pipsize*25)
    ax4 = axes[3]
    ax4.plot(time_index, atr_pip_25, 'red', linewidth=1.5, label='ATR/(pipsize*25)')
    ax4.fill_between(time_index, atr_pip_25, alpha=0.3, color='red')
    
    # Add reference lines for 25-pip scaling
    ax4.axhline(0.4, color='green', linestyle='--', alpha=0.7, label='0.4 (10 pips)')
    ax4.axhline(0.6, color='cyan', linestyle='--', alpha=0.7, label='0.6 (15 pips)')
    ax4.axhline(0.8, color='orange', linestyle='--', alpha=0.7, label='0.8 (20 pips)')
    ax4.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1.0 (25 pips)')
    ax4.axhline(1.2, color='purple', linestyle=':', alpha=0.5, label='1.2 (30 pips)')
    ax4.axhline(2.0, color='black', linestyle=':', alpha=0.3, label='2.0 (50 pips)')
    
    ax4.set_title('4. ATR/(pipsize*25) - 25-pip Reference (High Resolution)', fontweight='bold')
    ax4.set_ylabel('Normalized Value')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_25) > 0:
        ax4.text(0.02, 0.95, f'Range: [{valid_25.min():.3f}, {valid_25.max():.3f}]\\nMean: {valid_25.mean():.3f}\\nRef: 1.0 = 25 pips', 
                transform=ax4.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in axes:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/atr_25pip_linear_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print detailed comparison
    print(f"\n💡 High-Resolution Scaling Comparison:")
    print(f"="*60)
    
    # Create comparison table
    pip_levels = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f"{'Pips':<4} {'÷100':<8} {'÷50':<8} {'÷25':<8} {'Relative to ÷100'}")
    print("-" * 45)
    
    for pips in pip_levels:
        scale_100 = pips / 100
        scale_50 = pips / 50
        scale_25 = pips / 25
        print(f"{pips:<4} {scale_100:<8.3f} {scale_50:<8.3f} {scale_25:<8.3f} {scale_25/scale_100:<8.1f}x")
    
    print(f"\n🎯 Resolution Analysis:")
    print(f"ATR/(pipsize*25) provides:")
    print(f"  ✓ 4x higher values than ÷100 scaling")
    print(f"  ✓ 2x higher values than ÷50 scaling")
    print(f"  ✓ Excellent granularity for 10-30 pip range")
    print(f"  ✓ 1.0 represents modest volatility threshold (25 pips)")
    print(f"")
    print(f"Your data analysis:")
    print(f"  • Average ATR: {valid_pips.mean():.1f} pips")
    print(f"  • ÷100 scaling: {valid_100.mean():.3f}")
    print(f"  • ÷50 scaling: {valid_50.mean():.3f}")
    print(f"  • ÷25 scaling: {valid_25.mean():.3f}")
    print(f"  • Resolution gain: {valid_25.mean()/valid_100.mean():.1f}x higher than ÷100")
    print(f"")
    print(f"📊 Volatility Level Interpretation:")
    print(f"With ATR/(pipsize*25):")
    print(f"  • 0.0 - 0.4: Very low volatility (0-10 pips)")
    print(f"  • 0.4 - 0.6: Low volatility (10-15 pips)")
    print(f"  • 0.6 - 0.8: Medium-low volatility (15-20 pips)")
    print(f"  • 0.8 - 1.0: Medium volatility (20-25 pips)")
    print(f"  • 1.0 - 1.2: Medium-high volatility (25-30 pips)")
    print(f"  • 1.2 - 1.6: High volatility (30-40 pips)")
    print(f"  • 1.6 - 2.0: Very high volatility (40-50 pips)")
    print(f"  • > 2.0: Extreme volatility (> 50 pips)")
    print(f"")
    print(f"🎯 Use ATR/(pipsize*25) if:")
    print(f"  ✓ You want maximum granularity in normal volatility ranges")
    print(f"  ✓ Most volatility is 10-30 pips (typical for majors)")
    print(f"  ✓ You need fine discrimination for ML features")
    print(f"  ✓ Your average volatility (~16 pips) benefits from higher values")
    print(f"  ✓ 1.0 as a reasonable daily volatility threshold")
    print(f"")
    print(f"✅ Advantages of ÷25 scaling:")
    print(f"  ✓ Highest resolution for typical FX volatility")
    print(f"  ✓ Your 16-pip average becomes {valid_25.mean():.3f} (good feature range)")
    print(f"  ✓ Better ML feature discrimination")
    print(f"  ✓ Still intuitive (1.0 = 25 pips)")
    print(f"  ✓ Linear scaling preserves relationships")
    print(f"")
    print(f"⚠️ Considerations:")
    print(f"  • Higher values may need different thresholds in models")
    print(f"  • More sensitive to small volatility changes")
    print(f"  • Need to adjust volatility level expectations")
    
    plt.show()
    print("🎉 ATR/(pipsize*25) high-resolution comparison completed!")

if __name__ == "__main__":
    main()