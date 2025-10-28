#!/usr/bin/env python3
"""
Compare tanh(ATR/(pipsize*50)) vs other ATR normalizations
Shows tanh scaling with 50-pip reference point
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
    print("📊 Comparing tanh(ATR/(pipsize*50)) vs other normalizations...")
    
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
    
    # Calculate different normalizations
    atr_pips = atr_raw / pip_size
    atr_pip_100 = atr_raw / (pip_size * 100)  # 1.0 = 100 pips
    atr_pip_50 = atr_raw / (pip_size * 50)    # 1.0 = 50 pips
    
    # Apply tanh transformations
    tanh_100 = np.tanh(atr_pip_100)  # tanh(ATR/(pipsize*100))
    tanh_50 = np.tanh(atr_pip_50)    # tanh(ATR/(pipsize*50))
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_pips = atr_pips[~np.isnan(atr_pips)]
    valid_tanh_100 = tanh_100[~np.isnan(tanh_100)]
    valid_tanh_50 = tanh_50[~np.isnan(tanh_50)]
    
    print(f"\n📊 Statistics:")
    print(f"ATR in Pips:")
    print(f"  Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips")
    print(f"  Mean: {valid_pips.mean():.1f} pips")
    
    print(f"\ntanh(ATR/(pipsize*100)):")
    print(f"  Range: [{valid_tanh_100.min():.3f}, {valid_tanh_100.max():.3f}]")
    print(f"  Mean: {valid_tanh_100.mean():.3f}")
    print(f"  Saturation point: 100 pips → tanh(1.0) = 0.762")
    
    print(f"\ntanh(ATR/(pipsize*50)):")
    print(f"  Range: [{valid_tanh_50.min():.3f}, {valid_tanh_50.max():.3f}]")
    print(f"  Mean: {valid_tanh_50.mean():.3f}")
    print(f"  Saturation point: 50 pips → tanh(1.0) = 0.762")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle('ATR Tanh Scaling Comparison: 50-pip vs 100-pip Reference\\nEUR/USD H1 Data', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: ATR in Pips (reference)
    ax1 = axes[0]
    ax1.plot(time_index, atr_pips, 'blue', linewidth=1.5, label='ATR in Pips')
    ax1.fill_between(time_index, atr_pips, alpha=0.3, color='blue')
    
    # Add reference lines
    ax1.axhline(10, color='green', linestyle='--', alpha=0.7, label='10 pips')
    ax1.axhline(20, color='orange', linestyle='--', alpha=0.7, label='20 pips')
    ax1.axhline(30, color='red', linestyle='--', alpha=0.7, label='30 pips')
    ax1.axhline(50, color='purple', linestyle=':', alpha=0.5, label='50 pips')
    
    ax1.set_title('1. ATR in Pips - Reference Measurement', fontweight='bold')
    ax1.set_ylabel('ATR (Pips)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_pips) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips\\nMean: {valid_pips.mean():.1f} pips', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: tanh(ATR/(pipsize*100))
    ax2 = axes[1]
    ax2.plot(time_index, tanh_100, 'green', linewidth=1.5, label='tanh(ATR/(pipsize*100))')
    ax2.fill_between(time_index, tanh_100, alpha=0.3, color='green')
    
    # Add interpretation lines for 100-pip scaling
    ax2.axhline(np.tanh(10/100), color='green', linestyle='--', alpha=0.7, label=f'10 pips → {np.tanh(0.1):.3f}')
    ax2.axhline(np.tanh(20/100), color='orange', linestyle='--', alpha=0.7, label=f'20 pips → {np.tanh(0.2):.3f}')
    ax2.axhline(np.tanh(50/100), color='purple', linestyle=':', alpha=0.5, label=f'50 pips → {np.tanh(0.5):.3f}')
    ax2.axhline(np.tanh(100/100), color='red', linestyle=':', alpha=0.3, label=f'100 pips → {np.tanh(1.0):.3f}')
    
    ax2.set_title('2. tanh(ATR/(pipsize*100)) - 100-pip Reference Scaling', fontweight='bold')
    ax2.set_ylabel('tanh Value')
    ax2.set_ylim(-0.05, 0.8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_tanh_100) > 0:
        ax2.text(0.02, 0.95, f'Range: [{valid_tanh_100.min():.3f}, {valid_tanh_100.max():.3f}]\\nMean: {valid_tanh_100.mean():.3f}\\nSaturation: ~100 pips', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: tanh(ATR/(pipsize*50))
    ax3 = axes[2]
    ax3.plot(time_index, tanh_50, 'red', linewidth=1.5, label='tanh(ATR/(pipsize*50))')
    ax3.fill_between(time_index, tanh_50, alpha=0.3, color='red')
    
    # Add interpretation lines for 50-pip scaling
    ax3.axhline(np.tanh(10/50), color='green', linestyle='--', alpha=0.7, label=f'10 pips → {np.tanh(0.2):.3f}')
    ax3.axhline(np.tanh(20/50), color='orange', linestyle='--', alpha=0.7, label=f'20 pips → {np.tanh(0.4):.3f}')
    ax3.axhline(np.tanh(30/50), color='blue', linestyle='--', alpha=0.7, label=f'30 pips → {np.tanh(0.6):.3f}')
    ax3.axhline(np.tanh(50/50), color='red', linestyle=':', alpha=0.5, label=f'50 pips → {np.tanh(1.0):.3f}')
    
    ax3.set_title('3. tanh(ATR/(pipsize*50)) - 50-pip Reference Scaling', fontweight='bold')
    ax3.set_ylabel('tanh Value')
    ax3.set_xlabel('Time')
    ax3.set_ylim(-0.05, 0.8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_tanh_50) > 0:
        ax3.text(0.02, 0.95, f'Range: [{valid_tanh_50.min():.3f}, {valid_tanh_50.max():.3f}]\\nMean: {valid_tanh_50.mean():.3f}\\nSaturation: ~50 pips', 
                transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in axes:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/tanh_atr_50pip_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print detailed comparison
    print(f"\n💡 Scaling Comparison Analysis:")
    print(f"="*60)
    
    # Create comparison table
    pip_levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    print(f"{'Pips':<4} {'tanh(pips/100)':<12} {'tanh(pips/50)':<12} {'Difference':<10}")
    print("-" * 50)
    
    for pips in pip_levels:
        tanh_100_val = np.tanh(pips / 100)
        tanh_50_val = np.tanh(pips / 50)
        diff = tanh_50_val - tanh_100_val
        print(f"{pips:<4} {tanh_100_val:<12.3f} {tanh_50_val:<12.3f} {diff:<10.3f}")
    
    print(f"\n🎯 Key Insights:")
    print(f"tanh(ATR/(pipsize*50)) provides:")
    print(f"  ✓ More sensitivity in the 0-30 pip range")
    print(f"  ✓ Earlier saturation (starts around 50 pips)")
    print(f"  ✓ Better distinction for typical FX volatility levels")
    print(f"  ✓ Higher values for the same pip input")
    print(f"")
    print(f"Your data analysis:")
    print(f"  • Average ATR: {valid_pips.mean():.1f} pips")
    print(f"  • tanh(pips/100): {valid_tanh_100.mean():.3f}")
    print(f"  • tanh(pips/50): {valid_tanh_50.mean():.3f}")
    print(f"  • Sensitivity gain: {(valid_tanh_50.mean()/valid_tanh_100.mean()-1)*100:.1f}% higher values")
    print(f"")
    print(f"🎯 Recommendation:")
    print(f"Use tanh(ATR/(pipsize*50)) if:")
    print(f"  ✓ Most volatility is < 50 pips (typical for major pairs)")
    print(f"  ✓ You want better discrimination in normal volatility ranges")
    print(f"  ✓ You want higher baseline values for ML features")
    print(f"  ✓ Extreme volatility (>50 pips) should be treated similarly")
    print(f"")
    print(f"Use tanh(ATR/(pipsize*100)) if:")
    print(f"  ✓ You trade high volatility pairs or exotic currencies")
    print(f"  ✓ You want more linear response up to 100 pips")
    print(f"  ✓ You need to distinguish between 50-100 pip volatility levels")
    
    plt.show()
    print("🎉 tanh(ATR/(pipsize*50)) comparison completed!")

if __name__ == "__main__":
    main()