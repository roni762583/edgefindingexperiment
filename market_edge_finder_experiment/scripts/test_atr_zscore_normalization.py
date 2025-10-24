#!/usr/bin/env python3
"""
Test ATR with Z-score Normalization using SMA(20) and STDEV(500)

Tests the formula: (ATR - SMA20(ATR)) / STDEV500(ATR) 
And then: (arctan(zscore) + π/2) / π for 0-1 mapping

Usage: python3 scripts/test_atr_zscore_normalization.py
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

def zscore_sma20_stdev500(series):
    """
    Calculate zscore using SMA(20) and STDEV(500) pattern from new_swt repo.
    
    Formula: (x - sma20(x)) / stdev500(x)
    
    Args:
        series: pandas Series of values
        
    Returns:
        pandas Series of zscore values
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    # Calculate SMA(20) 
    sma20 = series.rolling(window=20, min_periods=20).mean()
    
    # Calculate rolling STDEV(500)
    stdev500 = series.rolling(window=500, min_periods=500).std()
    
    # Calculate zscore: (x - sma20(x)) / stdev500(x)
    zscore = (series - sma20) / stdev500
    
    # Handle edge cases (inf, -inf)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    return zscore

def arctan_normalize_zscore(zscore_series):
    """
    Normalize zscore using arctan to map to 0-1 range.
    
    Formula: (arctan(zscore) + π/2) / π
    
    High volatility (positive zscore) → values closer to 1
    Low volatility (negative zscore) → values closer to 0
    """
    return (np.arctan(zscore_series) + np.pi/2) / np.pi

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) using Wilder's method
    """
    high = df['high']
    low = df['low'] 
    close = df['close']
    
    # Calculate True Range components
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    # True Range = max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR using Wilder's smoothing (EMA with alpha = 1/period)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr

def test_atr_zscore_normalization():
    """
    Test ATR with zscore normalization and arctan mapping
    """
    print("🚀 Testing ATR Z-score Normalization...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} bars")
    
    print("🔄 Calculating ATR...")
    # Calculate ATR
    df['atr'] = calculate_atr(df, period=14)
    
    print("🔄 Calculating Z-score normalization...")
    # Apply zscore normalization using SMA(20) and STDEV(500)
    df['atr_zscore'] = zscore_sma20_stdev500(df['atr'])
    
    print("🔄 Applying arctan transformation...")
    # Apply arctan normalization to map to 0-1
    df['atr_normalized'] = arctan_normalize_zscore(df['atr_zscore'])
    
    # Also calculate simple min-max normalization for comparison
    df['atr_minmax'] = (df['atr'] - df['atr'].min()) / (df['atr'].max() - df['atr'].min())
    
    print("🔄 Creating comparison chart...")
    
    # Create comparison chart
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    fig.suptitle('ATR Z-score Normalization Test', fontsize=14, fontweight='bold')
    
    # Use last 100 bars for clarity
    plot_data = df.tail(100).copy().reset_index(drop=True)
    
    # Set up simple numeric x-axis for speed
    x_data = range(len(plot_data))
    
    # 1. Price Chart
    axes[0].plot(x_data, plot_data['close'], 'b-', linewidth=1.5, label='Close Price')
    axes[0].set_title('Price Chart', fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Raw ATR
    axes[1].plot(x_data, plot_data['atr'], 'red', linewidth=2, label='Raw ATR')
    axes[1].fill_between(x_data, 0, plot_data['atr'], alpha=0.3, color='red')
    axes[1].set_title('Raw ATR (Average True Range)', fontweight='bold')
    axes[1].set_ylabel('ATR Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. ATR Z-score
    axes[2].plot(x_data, plot_data['atr_zscore'], 'purple', linewidth=2, label='ATR Z-score')
    axes[2].fill_between(x_data, 0, plot_data['atr_zscore'], alpha=0.3, color='purple')
    axes[2].set_title('ATR Z-score: (ATR - SMA20(ATR)) / STDEV500(ATR)', fontweight='bold')
    axes[2].set_ylabel('Z-score')
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='±1 std')
    axes[2].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 4. Arctan Normalized ATR (0-1)
    axes[3].plot(x_data, plot_data['atr_normalized'], 'darkgreen', linewidth=2, label='ATR Normalized (arctan)')
    axes[3].fill_between(x_data, 0, plot_data['atr_normalized'], alpha=0.3, color='darkgreen')
    axes[3].set_title('ATR Normalized: (arctan(zscore) + π/2) / π', fontweight='bold')
    axes[3].set_ylabel('Normalized Value (0-1)')
    axes[3].set_ylim(0, 1)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Add regime quartile lines
    for level, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
        axes[3].axhline(y=level, color='gray', linestyle='--', alpha=0.6)
        axes[3].text(0, level + 0.02, label, fontsize=8, color='gray')
    
    # 5. Comparison: Arctan vs Min-Max normalization
    axes[4].plot(x_data, plot_data['atr_normalized'], 'darkgreen', linewidth=2, label='Arctan Normalized', alpha=0.8)
    axes[4].plot(x_data, plot_data['atr_minmax'], 'orange', linewidth=2, label='Min-Max Normalized', alpha=0.8)
    axes[4].set_title('Comparison: Arctan vs Min-Max Normalization', fontweight='bold')
    axes[4].set_ylabel('Normalized Value (0-1)')
    axes[4].set_xlabel('Time')
    axes[4].set_ylim(0, 1)
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    # Simple numeric x-axis labels
    for ax in axes:
        ax.set_xlabel('Bar Index')
    
    plt.tight_layout(pad=2.0)
    
    # Save chart
    save_path = project_root / "data/test/atr_zscore_normalization_test.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Calculate and print statistics
    print(f"\n📊 Statistics for last {len(plot_data)} bars:")
    
    valid_data = plot_data[['atr', 'atr_zscore', 'atr_normalized', 'atr_minmax']].dropna()
    
    if len(valid_data) > 0:
        print(f"Raw ATR          - Mean: {valid_data['atr'].mean():.4f}, Std: {valid_data['atr'].std():.4f}")
        print(f"ATR Z-score      - Mean: {valid_data['atr_zscore'].mean():.4f}, Std: {valid_data['atr_zscore'].std():.4f}")
        print(f"ATR Normalized   - Mean: {valid_data['atr_normalized'].mean():.4f}, Std: {valid_data['atr_normalized'].std():.4f}")
        print(f"ATR Min-Max      - Mean: {valid_data['atr_minmax'].mean():.4f}, Std: {valid_data['atr_minmax'].std():.4f}")
        
        # Correlation between normalization methods
        corr = valid_data['atr_normalized'].corr(valid_data['atr_minmax'])
        print(f"\nCorrelation (Arctan vs Min-Max): {corr:.4f}")
        
        # Regime distribution for arctan normalized
        q25 = (valid_data['atr_normalized'] < 0.25).sum() / len(valid_data)
        q50 = ((valid_data['atr_normalized'] >= 0.25) & (valid_data['atr_normalized'] < 0.50)).sum() / len(valid_data)
        q75 = ((valid_data['atr_normalized'] >= 0.50) & (valid_data['atr_normalized'] < 0.75)).sum() / len(valid_data)
        q100 = (valid_data['atr_normalized'] >= 0.75).sum() / len(valid_data)
        
        print(f"\n📊 Volatility Regime Distribution (Arctan Normalized):")
        print(f"Low Vol (Q1):     {q25:.1%}")
        print(f"Moderate (Q2):    {q50:.1%}")
        print(f"High (Q3):        {q75:.1%}")
        print(f"Very High (Q4):   {q100:.1%}")
    
    plt.show()
    print("🎉 ATR Z-score normalization test completed!")

def main():
    """Main function"""
    test_atr_zscore_normalization()

if __name__ == "__main__":
    main()