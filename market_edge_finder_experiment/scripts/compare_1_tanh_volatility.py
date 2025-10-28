#!/usr/bin/env python3
"""
Compare 1-tanh(zscore(ATR)) volatility implementation
Uses 500-period rolling window for standard deviation calculation
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def calculate_volatility_1_tanh(high, low, close, atr_window=14, sma_window=20, stdev_window=500):
    """
    Calculate 1-tanh(zscore(ATR)) volatility indicator
    
    Formula: 1 - tanh((ATR - SMA20(ATR)) / STDEV500(ATR))
    
    Returns values in range [0, 2] where:
    - Values near 0: Very high volatility (z-score >> 0)
    - Values near 1: Average volatility (z-score ≈ 0) 
    - Values near 2: Very low volatility (z-score << 0)
    """
    print(f"🔄 Calculating 1-tanh volatility with {stdev_window}-period rolling window...")
    
    # Calculate ATR
    atr = TechnicalIndicators.calculate_atr(high, low, close, atr_window)
    print(f"   ATR calculated: {np.sum(~np.isnan(atr))}/{len(atr)} valid values")
    
    # Calculate SMA20 of ATR
    atr_sma = np.full(len(atr), np.nan)
    for i in range(sma_window - 1, len(atr)):
        if not np.any(np.isnan(atr[i-sma_window+1:i+1])):
            atr_sma[i] = np.mean(atr[i-sma_window+1:i+1])
    
    print(f"   ATR SMA20: {np.sum(~np.isnan(atr_sma))}/{len(atr_sma)} valid values")
    
    # Calculate rolling standard deviation with specified window
    atr_stdev = np.full(len(atr), np.nan)
    for i in range(stdev_window - 1, len(atr)):
        start_idx = max(0, i - stdev_window + 1)
        atr_slice = atr[start_idx:i+1]
        if not np.any(np.isnan(atr_slice)) and len(atr_slice) > 1:
            atr_stdev[i] = np.std(atr_slice, ddof=1)
    
    print(f"   ATR STDEV{stdev_window}: {np.sum(~np.isnan(atr_stdev))}/{len(atr_stdev)} valid values")
    
    # Calculate z-score
    zscore = np.full(len(atr), np.nan)
    valid_mask = ~(np.isnan(atr) | np.isnan(atr_sma) | np.isnan(atr_stdev)) & (atr_stdev > 0)
    zscore[valid_mask] = (atr[valid_mask] - atr_sma[valid_mask]) / atr_stdev[valid_mask]
    
    print(f"   Z-Score: {np.sum(~np.isnan(zscore))}/{len(zscore)} valid values")
    
    # Apply 1 - tanh transformation
    volatility_1_tanh = np.full(len(zscore), np.nan)
    valid_zscore = ~np.isnan(zscore)
    volatility_1_tanh[valid_zscore] = 1 - np.tanh(zscore[valid_zscore])
    
    print(f"   1-tanh(zscore): {np.sum(~np.isnan(volatility_1_tanh))}/{len(volatility_1_tanh)} valid values")
    
    return volatility_1_tanh, zscore, atr, atr_sma, atr_stdev

def calculate_current_volatility(high, low, close, atr_window=14, sma_window=20, stdev_window=500):
    """Calculate current raw z-score volatility for comparison"""
    atr = TechnicalIndicators.calculate_atr(high, low, close, atr_window)
    
    # Calculate SMA20 of ATR
    atr_sma = np.full(len(atr), np.nan)
    for i in range(sma_window - 1, len(atr)):
        if not np.any(np.isnan(atr[i-sma_window+1:i+1])):
            atr_sma[i] = np.mean(atr[i-sma_window+1:i+1])
    
    # Calculate rolling standard deviation
    atr_stdev = np.full(len(atr), np.nan)
    for i in range(stdev_window - 1, len(atr)):
        start_idx = max(0, i - stdev_window + 1)
        atr_slice = atr[start_idx:i+1]
        if not np.any(np.isnan(atr_slice)) and len(atr_slice) > 1:
            atr_stdev[i] = np.std(atr_slice, ddof=1)
    
    # Calculate z-score
    zscore = np.full(len(atr), np.nan)
    valid_mask = ~(np.isnan(atr) | np.isnan(atr_sma) | np.isnan(atr_stdev)) & (atr_stdev > 0)
    zscore[valid_mask] = (atr[valid_mask] - atr_sma[valid_mask]) / atr_stdev[valid_mask]
    
    return zscore

def main():
    print("📊 Comparing 1-tanh(zscore(ATR)) vs Raw zscore volatility methods...")
    print("="*70)
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Extract OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate both volatility methods
    print(f"\n🔄 Calculating volatility indicators...")
    vol_1_tanh, zscore_1_tanh, atr, atr_sma, atr_stdev = calculate_volatility_1_tanh(high, low, close)
    vol_current = calculate_current_volatility(high, low, close)
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Analysis of results
    print(f"\n📊 STATISTICAL COMPARISON:")
    print(f"="*50)
    
    # 1-tanh method analysis
    valid_1_tanh = vol_1_tanh[~np.isnan(vol_1_tanh)]
    valid_zscore_1_tanh = zscore_1_tanh[~np.isnan(zscore_1_tanh)]
    
    print(f"1-tanh(zscore) Method:")
    print(f"  Valid values: {len(valid_1_tanh)}/{len(vol_1_tanh)} ({len(valid_1_tanh)/len(vol_1_tanh)*100:.1f}%)")
    print(f"  Range: [{valid_1_tanh.min():.3f}, {valid_1_tanh.max():.3f}]")
    print(f"  Mean: {valid_1_tanh.mean():.3f}")
    print(f"  Std: {valid_1_tanh.std():.3f}")
    print(f"  Underlying z-score range: [{valid_zscore_1_tanh.min():.3f}, {valid_zscore_1_tanh.max():.3f}]")
    
    # Current method analysis  
    valid_current = vol_current[~np.isnan(vol_current)]
    
    print(f"\nRaw Z-Score Method (Current):")
    print(f"  Valid values: {len(valid_current)}/{len(vol_current)} ({len(valid_current)/len(vol_current)*100:.1f}%)")
    print(f"  Range: [{valid_current.min():.3f}, {valid_current.max():.3f}]")
    print(f"  Mean: {valid_current.mean():.3f}")
    print(f"  Std: {valid_current.std():.3f}")
    
    # Interpretation guide
    print(f"\n🎯 INTERPRETATION GUIDE:")
    print(f"="*40)
    print(f"1-tanh(zscore) Values:")
    print(f"  • 0.0 - 0.5: Very high volatility (z-score > +1)")
    print(f"  • 0.5 - 1.0: Above average volatility (z-score 0 to +1)")
    print(f"  • 1.0: Average volatility (z-score = 0)")
    print(f"  • 1.0 - 1.5: Below average volatility (z-score 0 to -1)")
    print(f"  • 1.5 - 2.0: Very low volatility (z-score < -1)")
    print(f"")
    print(f"Raw Z-Score Values:")
    print(f"  • > +2: Extremely high volatility")
    print(f"  • +1 to +2: High volatility")
    print(f"  • -1 to +1: Normal volatility range")
    print(f"  • -1 to -2: Low volatility")
    print(f"  • < -2: Extremely low volatility")
    
    # Create comprehensive comparison chart
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle('1-tanh(zscore(ATR)) vs Raw Z-Score Volatility Comparison\\nEUR/USD H1 Data (700 bars)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Raw ATR
    ax1 = axes[0]
    valid_atr = ~np.isnan(atr)
    ax1.plot(time_index, atr, 'blue', linewidth=1.5, label='ATR (14-period)')
    ax1.fill_between(time_index, atr, alpha=0.3, color='blue')
    ax1.set_title('1. Average True Range (ATR) - Base Measurement', fontweight='bold')
    ax1.set_ylabel('ATR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if valid_atr.any():
        atr_valid = atr[valid_atr]
        ax1.text(0.02, 0.95, f'Range: [{atr_valid.min():.6f}, {atr_valid.max():.6f}]\\nMean: {atr_valid.mean():.6f}', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. ATR with SMA and STDEV bands
    ax2 = axes[1] 
    valid_sma = ~np.isnan(atr_sma)
    valid_stdev = ~np.isnan(atr_stdev)
    
    ax2.plot(time_index, atr, 'blue', alpha=0.5, linewidth=1, label='ATR')
    if valid_sma.any():
        ax2.plot(time_index, atr_sma, 'red', linewidth=2, label='SMA20(ATR)')
        
    if valid_stdev.any() and valid_sma.any():
        # Plot standard deviation bands
        upper_band = atr_sma + atr_stdev
        lower_band = atr_sma - atr_stdev
        ax2.fill_between(time_index, lower_band, upper_band, alpha=0.2, color='red', label='±1σ bands')
        
    ax2.set_title('2. ATR with SMA20 and Standard Deviation Bands', fontweight='bold')
    ax2.set_ylabel('ATR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-Score comparison
    ax3 = axes[2]
    ax3.plot(time_index, zscore_1_tanh, 'green', linewidth=2, label='Z-Score (500-period STDEV)')
    ax3.plot(time_index, vol_current, 'orange', linewidth=2, alpha=0.7, label='Z-Score (Current method)')
    
    # Add reference lines
    ax3.axhline(0, color='black', linestyle='-', alpha=0.7, label='Mean (0)')
    ax3.axhline(1, color='red', linestyle='--', alpha=0.5, label='±1σ')
    ax3.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(2, color='red', linestyle=':', alpha=0.3, label='±2σ')
    ax3.axhline(-2, color='red', linestyle=':', alpha=0.3)
    
    ax3.set_title('3. Z-Score Comparison (Both Methods)', fontweight='bold')
    ax3.set_ylabel('Z-Score')
    ax3.set_ylim(-3, 3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 1-tanh(zscore) volatility
    ax4 = axes[3]
    ax4.plot(time_index, vol_1_tanh, 'purple', linewidth=2, label='1-tanh(zscore) Volatility')
    ax4.fill_between(time_index, vol_1_tanh, alpha=0.3, color='purple')
    
    # Add interpretation reference lines
    ax4.axhline(1, color='black', linestyle='-', alpha=0.7, label='1.0 (Average vol)')
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='0.5 (High vol)')
    ax4.axhline(1.5, color='blue', linestyle='--', alpha=0.5, label='1.5 (Low vol)')
    ax4.axhline(0.2, color='red', linestyle=':', alpha=0.3, label='0.2 (Very high)')
    ax4.axhline(1.8, color='blue', linestyle=':', alpha=0.3, label='1.8 (Very low)')
    
    ax4.set_title('4. 1-tanh(zscore) Volatility Indicator', fontweight='bold')
    ax4.set_ylabel('Volatility Level')
    ax4.set_ylim(-0.1, 2.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_1_tanh) > 0:
        ax4.text(0.02, 0.95, f'Range: [{valid_1_tanh.min():.3f}, {valid_1_tanh.max():.3f}]\\nMean: {valid_1_tanh.mean():.3f}\\nStd: {valid_1_tanh.std():.3f}', 
                transform=ax4.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Side-by-side comparison
    ax5 = axes[4]
    
    # Normalize both for visual comparison
    if len(valid_current) > 0 and len(valid_1_tanh) > 0:
        # Normalize current method to [0,1] for comparison
        current_normalized = (vol_current - np.nanmin(vol_current)) / (np.nanmax(vol_current) - np.nanmin(vol_current))
        
        # Normalize 1-tanh to [0,1] for comparison  
        tanh_normalized = (vol_1_tanh - np.nanmin(vol_1_tanh)) / (np.nanmax(vol_1_tanh) - np.nanmin(vol_1_tanh))
        
        ax5.plot(time_index, current_normalized, 'orange', linewidth=2, alpha=0.8, label='Raw Z-Score (normalized)')
        ax5.plot(time_index, tanh_normalized, 'purple', linewidth=2, alpha=0.8, label='1-tanh(zscore) (normalized)')
        
        ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 midpoint')
    
    ax5.set_title('5. Normalized Comparison (Both scaled to [0,1])', fontweight='bold')
    ax5.set_ylabel('Normalized Value')
    ax5.set_xlabel('Time')
    ax5.set_ylim(-0.1, 1.1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Format time axis
    for ax in axes:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/1_tanh_volatility_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved to: {save_path}")
    
    # Correlation analysis
    if len(valid_1_tanh) > 0 and len(valid_current) > 0:
        # Find overlapping valid indices
        valid_both = ~(np.isnan(vol_1_tanh) | np.isnan(vol_current))
        if valid_both.any():
            corr = np.corrcoef(vol_1_tanh[valid_both], vol_current[valid_both])[0,1]
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"Correlation between methods: {corr:.4f}")
            
            if abs(corr) > 0.8:
                print("   → Strong correlation: Methods track similarly")
            elif abs(corr) > 0.5:
                print("   → Moderate correlation: Some similarity with differences")
            else:
                print("   → Weak correlation: Methods behave quite differently")
    
    # Practical implications
    print(f"\n💡 PRACTICAL IMPLICATIONS:")
    print(f"="*40)
    print(f"1-tanh(zscore) Method:")
    print(f"  ✓ Bounded [0, 2] output")
    print(f"  ✓ Intuitive 'volatility level' interpretation")
    print(f"  ✓ Value of 1.0 = average volatility")
    print(f"  ✓ Lower values = higher volatility (inverted)")
    print(f"  ✓ Smooth S-curve response to extremes")
    print(f"")
    print(f"Raw Z-Score Method:")
    print(f"  ✓ Preserves original statistical meaning")
    print(f"  ✓ Unbounded range captures all extremes")
    print(f"  ✓ Positive values = above average volatility")
    print(f"  ✓ Direct statistical interpretation")
    
    print(f"\n🎯 RECOMMENDATION:")
    if len(valid_1_tanh) > 0:
        high_vol_pct = np.sum(valid_1_tanh < 0.5) / len(valid_1_tanh) * 100
        low_vol_pct = np.sum(valid_1_tanh > 1.5) / len(valid_1_tanh) * 100
        
        print(f"In your data:")
        print(f"  • {high_vol_pct:.1f}% of time shows high volatility (< 0.5)")
        print(f"  • {low_vol_pct:.1f}% of time shows low volatility (> 1.5)")
        print(f"")
        print(f"Choose 1-tanh(zscore) if you prefer:")
        print(f"  → Bounded, intuitive volatility levels")
        print(f"  → 'Volatility meter' interpretation")
        print(f"  → ML models that benefit from bounded inputs")
    
    plt.show()
    print("🎉 1-tanh volatility comparison completed!")

if __name__ == "__main__":
    main()