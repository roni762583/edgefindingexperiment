#!/usr/bin/env python3
"""
Simple comparison: tanh(zscore(ATR)) vs Pure ATR
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
    print("📊 Comparing tanh(zscore(ATR)) vs Pure ATR...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Extract OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate pure ATR
    atr = TechnicalIndicators.calculate_atr(high, low, close, period=14)
    
    # Calculate ATR z-score with 500-period rolling stdev
    atr_sma = np.full(len(atr), np.nan)
    for i in range(19, len(atr)):  # SMA20
        if not np.any(np.isnan(atr[i-19:i+1])):
            atr_sma[i] = np.mean(atr[i-19:i+1])
    
    atr_stdev = np.full(len(atr), np.nan)
    for i in range(499, len(atr)):  # 500-period rolling stdev
        start_idx = max(0, i - 499)
        atr_slice = atr[start_idx:i+1]
        if not np.any(np.isnan(atr_slice)) and len(atr_slice) > 1:
            atr_stdev[i] = np.std(atr_slice, ddof=1)
    
    # Calculate z-score
    zscore = np.full(len(atr), np.nan)
    valid_mask = ~(np.isnan(atr) | np.isnan(atr_sma) | np.isnan(atr_stdev)) & (atr_stdev > 0)
    zscore[valid_mask] = (atr[valid_mask] - atr_sma[valid_mask]) / atr_stdev[valid_mask]
    
    # Apply tanh transformation
    tanh_zscore = np.full(len(zscore), np.nan)
    valid_zscore = ~np.isnan(zscore)
    tanh_zscore[valid_zscore] = np.tanh(zscore[valid_zscore])
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_atr = atr[~np.isnan(atr)]
    valid_tanh = tanh_zscore[~np.isnan(tanh_zscore)]
    
    print(f"\n📊 Statistics:")
    print(f"Pure ATR:")
    print(f"  Valid: {len(valid_atr)}/{len(atr)} ({len(valid_atr)/len(atr)*100:.1f}%)")
    print(f"  Range: [{valid_atr.min():.6f}, {valid_atr.max():.6f}]")
    print(f"  Mean: {valid_atr.mean():.6f}")
    
    print(f"tanh(zscore(ATR)):")
    print(f"  Valid: {len(valid_tanh)}/{len(tanh_zscore)} ({len(valid_tanh)/len(tanh_zscore)*100:.1f}%)")
    if len(valid_tanh) > 0:
        print(f"  Range: [{valid_tanh.min():.3f}, {valid_tanh.max():.3f}]")
        print(f"  Mean: {valid_tanh.mean():.3f}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('ATR Comparison: Pure ATR vs tanh(zscore(ATR))\\nEUR/USD H1 Data', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Pure ATR
    ax1.plot(time_index, atr, 'blue', linewidth=1.5, label='Pure ATR (14-period)')
    ax1.fill_between(time_index, atr, alpha=0.3, color='blue')
    
    ax1.set_title('1. Pure Average True Range (ATR)', fontweight='bold')
    ax1.set_ylabel('ATR Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    if len(valid_atr) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_atr.min():.6f}, {valid_atr.max():.6f}]\\nMean: {valid_atr.mean():.6f}\\nValid: {len(valid_atr)}/{len(atr)}', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: tanh(zscore(ATR))
    ax2.plot(time_index, tanh_zscore, 'red', linewidth=1.5, label='tanh(zscore(ATR))')
    ax2.fill_between(time_index, tanh_zscore, alpha=0.3, color='red')
    
    # Add reference lines
    ax2.axhline(0, color='black', linestyle='-', alpha=0.7, label='0 (average)')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='±0.5')
    ax2.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(0.76, color='orange', linestyle=':', alpha=0.5, label='±0.76 (tanh(1))')
    ax2.axhline(-0.76, color='orange', linestyle=':', alpha=0.5)
    
    ax2.set_title('2. tanh(zscore(ATR)) - Normalized Volatility Indicator', fontweight='bold')
    ax2.set_ylabel('tanh(Z-Score)')
    ax2.set_xlabel('Time')
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    if len(valid_tanh) > 0:
        ax2.text(0.02, 0.95, f'Range: [{valid_tanh.min():.3f}, {valid_tanh.max():.3f}]\\nMean: {valid_tanh.mean():.3f}\\nValid: {len(valid_tanh)}/{len(tanh_zscore)}', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in [ax1, ax2]:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/tanh_atr_simple_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    plt.show()
    print("🎉 Simple tanh vs ATR comparison completed!")

if __name__ == "__main__":
    main()