#!/usr/bin/env python3
"""
Compare ATR/(pipsize*50) vs ATR/(pipsize*100) 
Shows linear pip scaling with 50-pip vs 100-pip reference points
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
    print("📊 Comparing ATR/(pipsize*50) vs ATR/(pipsize*100)...")
    
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
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_pips = atr_pips[~np.isnan(atr_pips)]
    valid_100 = atr_pip_100[~np.isnan(atr_pip_100)]
    valid_50 = atr_pip_50[~np.isnan(atr_pip_50)]
    
    print(f"\n📊 Statistics:")
    print(f"ATR in Pips:")
    print(f"  Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips")
    print(f"  Mean: {valid_pips.mean():.1f} pips")
    
    print(f"\nATR/(pipsize*100) - 100-pip reference:")
    print(f"  Range: [{valid_100.min():.3f}, {valid_100.max():.3f}]")
    print(f"  Mean: {valid_100.mean():.3f}")
    print(f"  Interpretation: 1.0 = 100 pips")
    
    print(f"\nATR/(pipsize*50) - 50-pip reference:")
    print(f"  Range: [{valid_50.min():.3f}, {valid_50.max():.3f}]")
    print(f"  Mean: {valid_50.mean():.3f}")
    print(f"  Interpretation: 1.0 = 50 pips")
    
    # Verify the relationship
    print(f"\n🔍 Relationship Verification:")
    print(f"  ATR/(pipsize*50) = 2 × ATR/(pipsize*100)")
    print(f"  Mean ratio: {valid_50.mean() / valid_100.mean():.3f} (should be 2.0)")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Linear ATR Scaling: 50-pip vs 100-pip Reference Points\\nEUR/USD H1 Data', 
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
    
    # Plot 2: ATR/(pipsize*100)
    ax2 = axes[1]
    ax2.plot(time_index, atr_pip_100, 'green', linewidth=1.5, label='ATR/(pipsize*100)')
    ax2.fill_between(time_index, atr_pip_100, alpha=0.3, color='green')
    
    # Add reference lines
    ax2.axhline(0.1, color='green', linestyle='--', alpha=0.7, label='0.1 (10 pips)')
    ax2.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='0.2 (20 pips)')
    ax2.axhline(0.3, color='red', linestyle='--', alpha=0.7, label='0.3 (30 pips)')
    ax2.axhline(0.5, color='purple', linestyle=':', alpha=0.5, label='0.5 (50 pips)')
    ax2.axhline(1.0, color='black', linestyle=':', alpha=0.3, label='1.0 (100 pips)')
    
    ax2.set_title('2. ATR/(pipsize*100) - 100-pip Reference (Linear)', fontweight='bold')
    ax2.set_ylabel('Normalized Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_100) > 0:
        ax2.text(0.02, 0.95, f'Range: [{valid_100.min():.3f}, {valid_100.max():.3f}]\\nMean: {valid_100.mean():.3f}\\nRef: 1.0 = 100 pips', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: ATR/(pipsize*50)
    ax3 = axes[2]
    ax3.plot(time_index, atr_pip_50, 'red', linewidth=1.5, label='ATR/(pipsize*50)')
    ax3.fill_between(time_index, atr_pip_50, alpha=0.3, color='red')
    
    # Add reference lines
    ax3.axhline(0.2, color='green', linestyle='--', alpha=0.7, label='0.2 (10 pips)')
    ax3.axhline(0.4, color='orange', linestyle='--', alpha=0.7, label='0.4 (20 pips)')
    ax3.axhline(0.6, color='red', linestyle='--', alpha=0.7, label='0.6 (30 pips)')
    ax3.axhline(1.0, color='purple', linestyle=':', alpha=0.5, label='1.0 (50 pips)')
    ax3.axhline(2.0, color='black', linestyle=':', alpha=0.3, label='2.0 (100 pips)')
    
    ax3.set_title('3. ATR/(pipsize*50) - 50-pip Reference (Linear)', fontweight='bold')
    ax3.set_ylabel('Normalized Value')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_50) > 0:
        ax3.text(0.02, 0.95, f'Range: [{valid_50.min():.3f}, {valid_50.max():.3f}]\\nMean: {valid_50.mean():.3f}\\nRef: 1.0 = 50 pips', 
                transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in axes:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/atr_50pip_linear_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print detailed comparison
    print(f"\n💡 Linear Scaling Comparison:")
    print(f"="*50)
    
    # Create comparison table
    pip_levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    print(f"{'Pips':<4} {'÷100 scaling':<12} {'÷50 scaling':<12} {'Ratio':<8}")
    print("-" * 40)
    
    for pips in pip_levels:
        scale_100 = pips / 100
        scale_50 = pips / 50
        ratio = scale_50 / scale_100
        print(f"{pips:<4} {scale_100:<12.3f} {scale_50:<12.3f} {ratio:<8.1f}x")
    
    print(f"\n🎯 Key Differences:")
    print(f"ATR/(pipsize*50) provides:")
    print(f"  ✓ 2x higher values than ÷100 scaling")
    print(f"  ✓ Better resolution for typical volatility (10-30 pips)")
    print(f"  ✓ More meaningful decimal values")
    print(f"  ✓ 1.0 represents reasonable volatility threshold (50 pips)")
    print(f"")
    print(f"Your data analysis:")
    print(f"  • Average ATR: {valid_pips.mean():.1f} pips")
    print(f"  • ÷100 scaling: {valid_100.mean():.3f}")
    print(f"  • ÷50 scaling: {valid_50.mean():.3f}")
    print(f"  • Values are {valid_50.mean()/valid_100.mean():.1f}x higher with ÷50")
    print(f"")
    print(f"📊 Volatility Level Interpretation:")
    print(f"With ATR/(pipsize*50):")
    print(f"  • 0.0 - 0.2: Very low volatility (0-10 pips)")
    print(f"  • 0.2 - 0.4: Low volatility (10-20 pips)")
    print(f"  • 0.4 - 0.6: Medium volatility (20-30 pips)")
    print(f"  • 0.6 - 1.0: High volatility (30-50 pips)")
    print(f"  • 1.0 - 2.0: Very high volatility (50-100 pips)")
    print(f"  • > 2.0: Extreme volatility (> 100 pips)")
    print(f"")
    print(f"🎯 Use ATR/(pipsize*50) if:")
    print(f"  ✓ You want better granularity in normal volatility ranges")
    print(f"  ✓ Most of your volatility is < 50 pips")
    print(f"  ✓ You prefer higher baseline values for features")
    print(f"  ✓ 1.0 as a meaningful volatility threshold makes sense")
    print(f"  ✓ You want linear scaling (no saturation)")
    print(f"")
    print(f"✅ Benefits over other methods:")
    print(f"  ✓ Linear (no transformation artifacts)")
    print(f"  ✓ Intuitive pip interpretation")
    print(f"  ✓ Better resolution than ÷100")
    print(f"  ✓ Unbounded (captures all extremes)")
    print(f"  ✓ Cross-instrument comparable")
    print(f"  ✓ Simple calculation")
    
    plt.show()
    print("🎉 ATR/(pipsize*50) linear comparison completed!")

if __name__ == "__main__":
    main()