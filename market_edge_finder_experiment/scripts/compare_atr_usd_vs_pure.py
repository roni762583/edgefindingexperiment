#!/usr/bin/env python3
"""
Compare Pure ATR vs USD-valued ATR
Shows the difference between raw price units and USD normalization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def calculate_usd_atr(high, low, close, period=14):
    """
    Calculate ATR in USD per 100k lot (same normalization as ASI)
    For EUR/USD: 1 pip = 0.0001, so USD value = ATR * 100000
    """
    # Calculate raw ATR
    atr_raw = TechnicalIndicators.calculate_atr(high, low, close, period)
    
    # Convert to USD per 100k lot
    # For EUR/USD: 1 pip movement on 100k lot = $10
    # ATR in price units * 100000 gives ATR in USD per 100k lot
    atr_usd = atr_raw * 100000
    
    return atr_usd, atr_raw

def main():
    print("📊 Comparing Pure ATR vs USD-valued ATR...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Extract OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate both ATR versions
    atr_usd, atr_pure = calculate_usd_atr(high, low, close, period=14)
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_pure = atr_pure[~np.isnan(atr_pure)]
    valid_usd = atr_usd[~np.isnan(atr_usd)]
    
    print(f"\n📊 Statistics:")
    print(f"Pure ATR (price units):")
    print(f"  Valid: {len(valid_pure)}/{len(atr_pure)} ({len(valid_pure)/len(atr_pure)*100:.1f}%)")
    print(f"  Range: [{valid_pure.min():.6f}, {valid_pure.max():.6f}]")
    print(f"  Mean: {valid_pure.mean():.6f}")
    print(f"  Interpretation: Raw volatility in EUR/USD price units")
    
    print(f"\nUSD-valued ATR (per 100k lot):")
    print(f"  Valid: {len(valid_usd)}/{len(atr_usd)} ({len(valid_usd)/len(atr_usd)*100:.1f}%)")
    print(f"  Range: [{valid_usd.min():.2f}, {valid_usd.max():.2f}] USD")
    print(f"  Mean: {valid_usd.mean():.2f} USD")
    print(f"  Interpretation: Volatility in USD per 100k lot")
    
    # Calculate conversion factor verification
    conversion_factor = valid_usd.mean() / valid_pure.mean() if len(valid_pure) > 0 else 0
    print(f"\nConversion factor: {conversion_factor:.0f} (should be 100,000)")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('ATR Comparison: Pure vs USD-Valued\\nEUR/USD H1 Data (Cross-Instrument Normalization)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Pure ATR
    ax1.plot(time_index, atr_pure, 'blue', linewidth=1.5, label='Pure ATR (14-period)')
    ax1.fill_between(time_index, atr_pure, alpha=0.3, color='blue')
    
    ax1.set_title('1. Pure ATR - Raw Price Units (EUR/USD points)', fontweight='bold')
    ax1.set_ylabel('ATR (Price Units)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics and interpretation
    if len(valid_pure) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_pure.min():.6f}, {valid_pure.max():.6f}]\\nMean: {valid_pure.mean():.6f}\\nValid: {len(valid_pure)}/{len(atr_pure)}\\n\\nInterpretation:\\n• Raw volatility measurement\\n• Currency pair specific\\n• Not comparable across instruments', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: USD-valued ATR
    ax2.plot(time_index, atr_usd, 'green', linewidth=1.5, label='USD-valued ATR (per 100k lot)')
    ax2.fill_between(time_index, atr_usd, alpha=0.3, color='green')
    
    ax2.set_title('2. USD-Valued ATR - Normalized for Cross-Instrument Comparison', fontweight='bold')
    ax2.set_ylabel('ATR (USD per 100k lot)')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics and interpretation
    if len(valid_usd) > 0:
        # Add reference lines for common volatility levels
        ax2.axhline(100, color='orange', linestyle='--', alpha=0.5, label='$100 (moderate vol)')
        ax2.axhline(200, color='red', linestyle='--', alpha=0.5, label='$200 (high vol)')
        ax2.legend()
        
        ax2.text(0.02, 0.95, f'Range: [{valid_usd.min():.0f}, {valid_usd.max():.0f}] USD\\nMean: {valid_usd.mean():.0f} USD\\nValid: {len(valid_usd)}/{len(atr_usd)}\\n\\nInterpretation:\\n• Volatility in dollar terms\\n• Comparable across all FX pairs\\n• Useful for risk management\\n• Standard lot = 100k units', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in [ax1, ax2]:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/atr_pure_vs_usd_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print practical interpretation
    print(f"\n💡 Practical Interpretation:")
    print(f"="*40)
    print(f"Pure ATR = {valid_pure.mean():.6f} means:")
    print(f"  • Average volatility of {valid_pure.mean():.6f} EUR/USD points")
    print(f"  • About {valid_pure.mean()*10000:.1f} pips average movement")
    print(f"")
    print(f"USD ATR = ${valid_usd.mean():.0f} means:")
    print(f"  • Average volatility costs ${valid_usd.mean():.0f} per 100k lot")
    print(f"  • Risk management: set position sizes based on USD volatility")
    print(f"  • Cross-pair comparison: compare ${valid_usd.mean():.0f} vs other pairs")
    print(f"")
    print(f"🎯 Key Benefits of USD Normalization:")
    print(f"  ✓ Cross-instrument comparison (EUR/USD vs GBP/JPY, etc.)")
    print(f"  ✓ Risk management in dollar terms")
    print(f"  ✓ Position sizing based on volatility cost")
    print(f"  ✓ Portfolio volatility budgeting")
    
    plt.show()
    print("🎉 Pure vs USD ATR comparison completed!")

if __name__ == "__main__":
    main()