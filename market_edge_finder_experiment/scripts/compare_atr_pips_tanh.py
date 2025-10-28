#!/usr/bin/env python3
"""
Compare ATR in pips vs tanh(pips/100) volatility indicator
Shows pip-normalized volatility with bounded tanh scaling
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def calculate_atr_pips_and_tanh(high, low, close, period=14, pip_size=0.0001):
    """
    Calculate ATR in pips and apply tanh(pips/100) transformation
    
    Args:
        high, low, close: Price arrays
        period: ATR calculation period
        pip_size: Pip size for the instrument (0.0001 for EUR/USD)
    
    Returns:
        atr_pips: ATR converted to pips
        tanh_pips: tanh(pips/100) bounded volatility indicator
    """
    # Calculate raw ATR
    atr_raw = TechnicalIndicators.calculate_atr(high, low, close, period)
    
    # Convert to pips
    atr_pips = atr_raw / pip_size
    
    # Apply tanh(pips/100) transformation
    tanh_pips = np.full(len(atr_pips), np.nan)
    valid_mask = ~np.isnan(atr_pips)
    tanh_pips[valid_mask] = np.tanh(atr_pips[valid_mask] / 100.0)
    
    return atr_pips, tanh_pips

def main():
    print("📊 Comparing ATR in pips vs tanh(pips/100)...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Extract OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate ATR in pips and tanh transformation
    # For EUR/USD: 1 pip = 0.0001
    pip_size = 0.0001
    atr_pips, tanh_pips = calculate_atr_pips_and_tanh(high, low, close, period=14, pip_size=pip_size)
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_pips = atr_pips[~np.isnan(atr_pips)]
    valid_tanh = tanh_pips[~np.isnan(tanh_pips)]
    
    print(f"\n📊 Statistics:")
    print(f"ATR in Pips:")
    print(f"  Valid: {len(valid_pips)}/{len(atr_pips)} ({len(valid_pips)/len(atr_pips)*100:.1f}%)")
    print(f"  Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips")
    print(f"  Mean: {valid_pips.mean():.1f} pips")
    print(f"  Interpretation: Direct volatility in pips")
    
    print(f"\ntanh(pips/100):")
    print(f"  Valid: {len(valid_tanh)}/{len(tanh_pips)} ({len(valid_tanh)/len(tanh_pips)*100:.1f}%)")
    print(f"  Range: [{valid_tanh.min():.3f}, {valid_tanh.max():.3f}]")
    print(f"  Mean: {valid_tanh.mean():.3f}")
    print(f"  Interpretation: Bounded volatility indicator [0, 1]")
    
    # Analyze volatility levels
    low_vol = np.sum(valid_pips < 10) / len(valid_pips) * 100 if len(valid_pips) > 0 else 0
    medium_vol = np.sum((valid_pips >= 10) & (valid_pips < 20)) / len(valid_pips) * 100 if len(valid_pips) > 0 else 0
    high_vol = np.sum(valid_pips >= 20) / len(valid_pips) * 100 if len(valid_pips) > 0 else 0
    
    print(f"\n📈 Volatility Distribution:")
    print(f"  Low volatility (< 10 pips): {low_vol:.1f}%")
    print(f"  Medium volatility (10-20 pips): {medium_vol:.1f}%")
    print(f"  High volatility (> 20 pips): {high_vol:.1f}%")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('ATR Volatility: Pips vs tanh(pips/100) Transformation\\nEUR/USD H1 Data', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: ATR in Pips
    ax1.plot(time_index, atr_pips, 'blue', linewidth=1.5, label='ATR in Pips (14-period)')
    ax1.fill_between(time_index, atr_pips, alpha=0.3, color='blue')
    
    # Add volatility level reference lines
    ax1.axhline(10, color='green', linestyle='--', alpha=0.7, label='10 pips (low vol threshold)')
    ax1.axhline(20, color='orange', linestyle='--', alpha=0.7, label='20 pips (high vol threshold)')
    ax1.axhline(30, color='red', linestyle=':', alpha=0.5, label='30 pips (very high vol)')
    
    ax1.set_title('1. ATR in Pips - Direct Volatility Measurement', fontweight='bold')
    ax1.set_ylabel('ATR (Pips)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_pips) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips\\nMean: {valid_pips.mean():.1f} pips\\nValid: {len(valid_pips)}/{len(atr_pips)}\\n\\nDistribution:\\n• < 10 pips: {low_vol:.1f}%\\n• 10-20 pips: {medium_vol:.1f}%\\n• > 20 pips: {high_vol:.1f}%', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: tanh(pips/100)
    ax2.plot(time_index, tanh_pips, 'red', linewidth=1.5, label='tanh(pips/100)')
    ax2.fill_between(time_index, tanh_pips, alpha=0.3, color='red')
    
    # Add interpretation reference lines
    ax2.axhline(np.tanh(10/100), color='green', linestyle='--', alpha=0.7, label=f'tanh(10/100) = {np.tanh(0.1):.3f}')
    ax2.axhline(np.tanh(20/100), color='orange', linestyle='--', alpha=0.7, label=f'tanh(20/100) = {np.tanh(0.2):.3f}')
    ax2.axhline(np.tanh(30/100), color='red', linestyle=':', alpha=0.5, label=f'tanh(30/100) = {np.tanh(0.3):.3f}')
    ax2.axhline(0.5, color='gray', linestyle='-', alpha=0.3, label='0.5 reference')
    
    ax2.set_title('2. tanh(pips/100) - Bounded Volatility Indicator', fontweight='bold')
    ax2.set_ylabel('tanh(pips/100)')
    ax2.set_xlabel('Time')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics and interpretation
    if len(valid_tanh) > 0:
        ax2.text(0.02, 0.95, f'Range: [{valid_tanh.min():.3f}, {valid_tanh.max():.3f}]\\nMean: {valid_tanh.mean():.3f}\\nValid: {len(valid_tanh)}/{len(tanh_pips)}\\n\\nInterpretation:\\n• 0.0 - 0.1: Low volatility\\n• 0.1 - 0.2: Medium volatility\\n• 0.2 - 0.3: High volatility\\n• > 0.3: Very high volatility', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in [ax1, ax2]:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/atr_pips_tanh_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print practical interpretation
    print(f"\n💡 Practical Interpretation:")
    print(f"="*50)
    print(f"ATR in Pips = {valid_pips.mean():.1f} pips means:")
    print(f"  • Average volatility of {valid_pips.mean():.1f} pips per bar")
    print(f"  • Traders can expect ~{valid_pips.mean():.1f} pip moves on average")
    print(f"  • Stop losses should account for {valid_pips.mean()*2:.1f}+ pip swings")
    print(f"")
    print(f"tanh(pips/100) = {valid_tanh.mean():.3f} means:")
    print(f"  • Normalized volatility indicator")
    print(f"  • Bounded [0, 1] for ML model stability")
    print(f"  • {valid_tanh.mean():.3f} indicates {'low' if valid_tanh.mean() < 0.15 else 'medium' if valid_tanh.mean() < 0.25 else 'high'} volatility period")
    print(f"")
    print(f"🎯 Scaling Analysis:")
    print(f"  • tanh(10 pips / 100) = {np.tanh(0.1):.3f} → Low volatility threshold")
    print(f"  • tanh(20 pips / 100) = {np.tanh(0.2):.3f} → High volatility threshold")
    print(f"  • tanh(50 pips / 100) = {np.tanh(0.5):.3f} → Very high volatility")
    print(f"  • tanh(100 pips / 100) = {np.tanh(1.0):.3f} → Extreme volatility")
    print(f"")
    print(f"✅ Benefits of tanh(pips/100):")
    print(f"  ✓ Intuitive pip-based scaling")
    print(f"  ✓ Bounded [0, 1] output for ML models")
    print(f"  ✓ Non-linear scaling emphasizes differences at low volatility")
    print(f"  ✓ Prevents extreme outliers from dominating")
    print(f"  ✓ Cross-instrument comparable (when using appropriate pip sizes)")
    
    plt.show()
    print("🎉 ATR pips vs tanh(pips/100) comparison completed!")

if __name__ == "__main__":
    main()