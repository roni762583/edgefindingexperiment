#!/usr/bin/env python3
"""
Compare Direction Combination Methods

Compares three methods for combining ADX and Efficiency Ratio:
1. Geometric Mean: sqrt(ER * (ADX/100))
2. Weighted Average: 0.6 * ER + 0.4 * (ADX/100)  
3. Simple Product: ER * (ADX/100)

Usage: python3 scripts/compare_direction_methods.py
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

def calculate_efficiency_ratio(df, period=14):
    """
    Calculate Kaufman's Efficiency Ratio
    ER = |Price Change| / Sum of |Daily Changes|
    """
    close = df['close'].values
    
    # Calculate price change over period
    price_change = np.abs(close[period:] - close[:-period])
    
    # Calculate sum of daily price changes over period
    daily_changes = np.abs(np.diff(close))
    volatility = np.array([
        np.sum(daily_changes[i:i+period]) 
        for i in range(len(daily_changes) - period + 1)
    ])
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-8, volatility)
    
    # Calculate efficiency ratio
    er = price_change / volatility
    
    # Pad with NaN for first period values
    er_full = np.full(len(close), np.nan)
    er_full[period:] = er
    
    return er_full

def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index (ADX)
    Simplified implementation focusing on trend strength
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smooth using Wilder's smoothing (exponential moving average)
    alpha = 1.0 / period
    
    # Initialize arrays
    atr = np.zeros(len(tr))
    plus_di = np.zeros(len(tr))
    minus_di = np.zeros(len(tr))
    
    # Calculate first values
    atr[0] = np.mean(tr[:period])
    plus_di[0] = np.mean(plus_dm[:period])
    minus_di[0] = np.mean(minus_dm[:period])
    
    # Apply Wilder's smoothing
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        plus_di[i] = alpha * plus_dm[i] + (1 - alpha) * plus_di[i-1]
        minus_di[i] = alpha * minus_dm[i] + (1 - alpha) * minus_di[i-1]
    
    # Calculate +DI and -DI
    plus_di_pct = 100 * plus_di / atr
    minus_di_pct = 100 * minus_di / atr
    
    # Calculate DX
    dx = 100 * np.abs(plus_di_pct - minus_di_pct) / (plus_di_pct + minus_di_pct + 1e-8)
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(len(dx))
    adx[period-1] = np.mean(dx[:period])
    
    for i in range(period, len(dx)):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Pad with NaN for first period values
    adx_full = np.full(len(close), np.nan)
    adx_full[1:] = adx
    
    return adx_full

def combine_direction_indicators(er, adx):
    """
    Calculate four combination methods for ER and ADX
    """
    # Normalize ADX to 0-1 range
    adx_norm = adx / 100.0
    
    # Method 1: Geometric Mean
    geometric_mean = np.sqrt(er * adx_norm)
    
    # Method 2: Weighted Average
    weighted_avg = 0.6 * er + 0.4 * adx_norm
    
    # Method 3: Simple Product
    simple_product = er * adx_norm
    
    # Method 4: Simple Product x10
    simple_product_x10 = er * adx_norm * 10
    
    return geometric_mean, weighted_avg, simple_product, simple_product_x10

def create_comparison_chart(df, save_path=None):
    """
    Create comparison chart of the four direction combination methods
    """
    print("📊 Calculating Efficiency Ratio...")
    er = calculate_efficiency_ratio(df, period=14)
    
    print("📊 Calculating ADX...")
    adx = calculate_adx(df, period=14)
    
    print("📊 Combining indicators...")
    geometric_mean, weighted_avg, simple_product, simple_product_x10 = combine_direction_indicators(er, adx)
    
    # Create the plot
    fig, axes = plt.subplots(6, 1, figsize=(15, 24))
    fig.suptitle('Direction Indicator Combination Methods Comparison', fontsize=16, fontweight='bold')
    
    # Filter data to show more recent period for clarity
    start_idx = max(0, len(df) - 200)  # Last 200 bars
    plot_df = df.iloc[start_idx:].copy()
    plot_df['er'] = er[start_idx:]
    plot_df['adx'] = adx[start_idx:]
    plot_df['geometric_mean'] = geometric_mean[start_idx:]
    plot_df['weighted_avg'] = weighted_avg[start_idx:]
    plot_df['simple_product'] = simple_product[start_idx:]
    plot_df['simple_product_x10'] = simple_product_x10[start_idx:]
    
    # Convert index to datetime if it's not already
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        if 'time' in plot_df.columns:
            plot_df.index = pd.to_datetime(plot_df['time'])
    
    x_data = plot_df.index
    
    # 1. Price Chart
    ax1 = axes[0]
    ax1.plot(x_data, plot_df['close'], 'b-', linewidth=1.5, label='Close Price')
    ax1.set_title('Price Chart', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Individual Components (ER and ADX)
    ax2 = axes[1]
    ax2.plot(x_data, plot_df['er'], 'g-', linewidth=2, label='Efficiency Ratio', alpha=0.8)
    ax2.plot(x_data, plot_df['adx']/100, 'r-', linewidth=2, label='ADX (normalized)', alpha=0.8)
    ax2.set_title('Individual Components', fontweight='bold')
    ax2.set_ylabel('Value (0-1)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Method 1: Geometric Mean
    ax3 = axes[2]
    ax3.plot(x_data, plot_df['geometric_mean'], 'purple', linewidth=3, label='Geometric Mean')
    ax3.fill_between(x_data, 0, plot_df['geometric_mean'], alpha=0.3, color='purple')
    ax3.set_title('Method 1: Geometric Mean = sqrt(ER × ADX_norm)', fontweight='bold')
    ax3.set_ylabel('Direction Score')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add quartile lines for regime states
    for level, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
        ax3.axhline(y=level, color='gray', linestyle='--', alpha=0.6)
        ax3.text(x_data[0], level + 0.02, label, fontsize=8, color='gray')
    
    # 4. Method 2: Weighted Average
    ax4 = axes[3]
    ax4.plot(x_data, plot_df['weighted_avg'], 'orange', linewidth=3, label='Weighted Average')
    ax4.fill_between(x_data, 0, plot_df['weighted_avg'], alpha=0.3, color='orange')
    ax4.set_title('Method 2: Weighted Average = 0.6×ER + 0.4×ADX_norm', fontweight='bold')
    ax4.set_ylabel('Direction Score')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add quartile lines
    for level, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
        ax4.axhline(y=level, color='gray', linestyle='--', alpha=0.6)
        ax4.text(x_data[0], level + 0.02, label, fontsize=8, color='gray')
    
    # 5. Method 3 vs 4: Simple Product Comparison (Same Scale)
    ax5 = axes[4]
    ax5.plot(x_data, plot_df['simple_product'], 'brown', linewidth=3, label='Simple Product (ER × ADX_norm)', alpha=0.8)
    ax5.plot(x_data, plot_df['simple_product_x10'], 'darkred', linewidth=3, label='Simple Product x10 (ER × ADX_norm × 10)', alpha=0.8)
    ax5.set_title('Methods 3 & 4 Comparison: Simple Product vs Simple Product x10 (Same Scale)', fontweight='bold')
    ax5.set_ylabel('Direction Score')
    ax5.set_ylim(0, 10)  # Use same scale to show actual difference
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Add reference lines for both scales
    for level, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
        ax5.axhline(y=level, color='brown', linestyle='--', alpha=0.4, linewidth=1)
        ax5.text(x_data[0], level + 0.1, label, fontsize=8, color='brown')
    
    for level, label in [(2.5, 'Q1×10'), (5.0, 'Q2×10'), (7.5, 'Q3×10')]:
        ax5.axhline(y=level, color='darkred', linestyle='--', alpha=0.4, linewidth=1)
        ax5.text(x_data[-20], level + 0.1, label, fontsize=8, color='darkred')
    
    # 6. Method 3: Simple Product (Close-up)
    ax6 = axes[5]
    ax6.plot(x_data, plot_df['simple_product'], 'brown', linewidth=3, label='Simple Product')
    ax6.fill_between(x_data, 0, plot_df['simple_product'], alpha=0.3, color='brown')
    ax6.set_title('Method 3: Simple Product = ER × ADX_norm (Close-up View)', fontweight='bold')
    ax6.set_ylabel('Direction Score')
    ax6.set_xlabel('Time')
    ax6.set_ylim(0, 1)  # Original scale for detail
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Add quartile lines
    for level, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
        ax6.axhline(y=level, color='gray', linestyle='--', alpha=0.6)
        ax6.text(x_data[0], level + 0.02, label, fontsize=8, color='gray')
    
    # Format x-axis for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the chart
    if save_path is None:
        save_path = project_root / "data/test/direction_methods_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Calculate and print statistics
    print(f"\n📊 Statistics for last {len(plot_df)} bars:")
    
    valid_mask = ~(np.isnan(plot_df['geometric_mean']) | 
                   np.isnan(plot_df['weighted_avg']) | 
                   np.isnan(plot_df['simple_product']) |
                   np.isnan(plot_df['simple_product_x10']))
    
    if np.any(valid_mask):
        print(f"Geometric Mean    - Mean: {np.nanmean(plot_df['geometric_mean']):.3f}, Std: {np.nanstd(plot_df['geometric_mean']):.3f}")
        print(f"Weighted Average  - Mean: {np.nanmean(plot_df['weighted_avg']):.3f}, Std: {np.nanstd(plot_df['weighted_avg']):.3f}")
        print(f"Simple Product    - Mean: {np.nanmean(plot_df['simple_product']):.3f}, Std: {np.nanstd(plot_df['simple_product']):.3f}")
        print(f"Simple Product x10- Mean: {np.nanmean(plot_df['simple_product_x10']):.3f}, Std: {np.nanstd(plot_df['simple_product_x10']):.3f}")
        
        # Calculate correlations between methods
        corr_geo_wgt = np.corrcoef(plot_df['geometric_mean'][valid_mask], plot_df['weighted_avg'][valid_mask])[0,1]
        corr_geo_prod = np.corrcoef(plot_df['geometric_mean'][valid_mask], plot_df['simple_product'][valid_mask])[0,1]
        corr_geo_prod_x10 = np.corrcoef(plot_df['geometric_mean'][valid_mask], plot_df['simple_product_x10'][valid_mask])[0,1]
        corr_wgt_prod = np.corrcoef(plot_df['weighted_avg'][valid_mask], plot_df['simple_product'][valid_mask])[0,1]
        corr_prod_prod_x10 = np.corrcoef(plot_df['simple_product'][valid_mask], plot_df['simple_product_x10'][valid_mask])[0,1]
        
        print(f"\n📊 Correlations between methods:")
        print(f"Geometric vs Weighted:     {corr_geo_wgt:.3f}")
        print(f"Geometric vs Product:      {corr_geo_prod:.3f}")
        print(f"Geometric vs Product x10:  {corr_geo_prod_x10:.3f}")
        print(f"Weighted vs Product:       {corr_wgt_prod:.3f}")
        print(f"Product vs Product x10:    {corr_prod_prod_x10:.3f}")
    
    plt.show()

def main():
    """Main function"""
    print("🚀 Starting Direction Methods Comparison...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    
    if not data_path.exists():
        print(f"❌ Sample data not found: {data_path}")
        return
    
    print(f"📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing required columns. Found: {list(df.columns)}")
        return
    
    print(f"✅ Loaded {len(df)} bars")
    
    # Create comparison chart
    create_comparison_chart(df)
    
    print("🎉 Direction methods comparison completed!")

if __name__ == "__main__":
    main()