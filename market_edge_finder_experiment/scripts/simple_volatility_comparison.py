#!/usr/bin/env python3
"""
Simple Volatility Methods Comparison - Fast Version

Compares five volatility methods with optimized calculations:
1. ATR (Average True Range)
2. Returns Volatility 
3. Parkinson Estimator
4. Garman-Klass Estimator
5. Yang-Zhang Estimator

Usage: python3 scripts/simple_volatility_comparison.py
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

def calculate_simple_volatility_methods(df, atr_period=14, vol_period=20):
    """
    Calculate all 5 volatility methods efficiently using pandas
    """
    print("🔄 Preparing data...")
    
    # Convert to pandas DataFrame with proper columns
    data = df[['open', 'high', 'low', 'close']].copy()
    
    print("🔄 Calculating ATR...")
    # 1. Average True Range (ATR)
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr'] = data['true_range'].ewm(span=atr_period, adjust=False).mean()
    
    print("🔄 Calculating Returns Volatility...")
    # 2. Returns-based Volatility
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['returns_vol'] = data['log_returns'].rolling(window=vol_period).std()
    
    print("🔄 Calculating Parkinson...")
    # 3. Parkinson Estimator
    parkinson_const = 1.0 / (4.0 * np.log(2.0))
    data['ln_hl_sq'] = (np.log(data['high'] / data['low'])) ** 2
    data['parkinson'] = np.sqrt(parkinson_const * data['ln_hl_sq'].rolling(window=vol_period).mean())
    
    print("🔄 Calculating Garman-Klass...")
    # 4. Garman-Klass Estimator  
    data['ln_hc'] = np.log(data['high'] / data['close'])
    data['ln_ho'] = np.log(data['high'] / data['open'])
    data['ln_lc'] = np.log(data['low'] / data['close'])
    data['ln_lo'] = np.log(data['low'] / data['open'])
    data['gk_daily'] = data['ln_hc'] * data['ln_ho'] + data['ln_lc'] * data['ln_lo']
    data['garman_klass'] = np.sqrt(data['gk_daily'].rolling(window=vol_period).mean())
    
    print("🔄 Calculating Yang-Zhang...")
    # 5. Yang-Zhang Estimator (simplified)
    data['overnight_ret'] = np.log(data['open'] / data['prev_close'])
    data['oc_ret'] = np.log(data['close'] / data['open'])
    
    # Rolling calculations
    overnight_var = data['overnight_ret'].rolling(window=vol_period).var()
    oc_var = data['oc_ret'].rolling(window=vol_period).var()
    rs_mean = data['gk_daily'].rolling(window=vol_period).mean()  # Reuse GK for efficiency
    
    # Simplified Yang-Zhang (without full bias correction)
    k = 0.34
    data['yang_zhang'] = np.sqrt(overnight_var + k * oc_var + (1-k) * rs_mean)
    
    return data

def normalize_to_0_1(series):
    """Simple min-max normalization to 0-1 range"""
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return series
    
    min_val = valid_data.min()
    max_val = valid_data.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    return (series - min_val) / (max_val - min_val)

def create_simple_comparison_chart(df, save_path=None):
    """
    Create comparison chart of volatility methods
    """
    # Calculate volatility methods
    vol_data = calculate_simple_volatility_methods(df)
    
    print("🔄 Normalizing data...")
    # Normalize all volatility measures
    vol_methods = ['atr', 'returns_vol', 'parkinson', 'garman_klass', 'yang_zhang']
    for method in vol_methods:
        vol_data[f'{method}_norm'] = normalize_to_0_1(vol_data[method])
    
    # Filter last 150 bars for clarity
    plot_data = vol_data.tail(150).copy()
    
    # Convert index to datetime if needed
    if 'time' in df.columns:
        time_col = df['time'].tail(150).reset_index(drop=True)
        plot_data.index = pd.to_datetime(time_col)
    
    print("🔄 Creating charts...")
    # Create plots
    fig, axes = plt.subplots(6, 1, figsize=(15, 24))
    fig.suptitle('Volatility Methods Comparison', fontsize=16, fontweight='bold')
    
    x_data = plot_data.index
    
    # 1. Price
    axes[0].plot(x_data, plot_data['close'], 'b-', linewidth=1.5)
    axes[0].set_title('Price Chart')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # 2-6. Volatility methods
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    titles = ['ATR', 'Returns Volatility', 'Parkinson', 'Garman-Klass', 'Yang-Zhang']
    
    for i, (method, color, title) in enumerate(zip(vol_methods, colors, titles)):
        ax = axes[i+1]
        norm_col = f'{method}_norm'
        
        ax.plot(x_data, plot_data[norm_col], color, linewidth=2, label=title)
        ax.fill_between(x_data, 0, plot_data[norm_col], alpha=0.3, color=color)
        ax.set_title(f'{title} (Normalized)')
        ax.set_ylabel('Volatility')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add quartile lines
        for level in [0.25, 0.5, 0.75]:
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
    
    # Format dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[-1].set_xlabel('Time')
    
    plt.tight_layout()
    
    # Save chart
    if save_path is None:
        save_path = project_root / "data/test/simple_volatility_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Calculate statistics
    print(f"\n📊 Statistics (last {len(plot_data)} bars):")
    
    for method in vol_methods:
        norm_col = f'{method}_norm'
        valid_data = plot_data[norm_col].dropna()
        if len(valid_data) > 0:
            print(f"{method:12} - Mean: {valid_data.mean():.3f}, Std: {valid_data.std():.3f}")
    
    # Calculate correlation matrix
    print(f"\n📊 Correlation Matrix:")
    norm_cols = [f'{method}_norm' for method in vol_methods]
    corr_data = plot_data[norm_cols].dropna()
    
    if len(corr_data) > 10:
        corr_matrix = corr_data.corr()
        print(corr_matrix.round(3))
    
    plt.show()

def main():
    """Main function"""
    print("🚀 Starting Simple Volatility Comparison...")
    
    # Load data
    data_path = project_root / "data/test/sample_data.csv"
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} bars")
    
    # Create comparison
    create_simple_comparison_chart(df)
    
    print("🎉 Volatility comparison completed!")

if __name__ == "__main__":
    main()