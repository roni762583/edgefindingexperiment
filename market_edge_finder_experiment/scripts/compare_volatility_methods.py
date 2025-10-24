#!/usr/bin/env python3
"""
Compare Volatility Estimation Methods

Compares five methods for volatility calculation:
1. Average True Range (ATR) - Wilder's method
2. Returns-based Volatility - Standard deviation of log returns
3. Parkinson Estimator - High-Low based volatility
4. Garman-Klass Estimator - OHLC-based volatility
5. Yang-Zhang Estimator - Handles overnight gaps

Usage: python3 scripts/compare_volatility_methods.py
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

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) - Wilder's method
    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = EMA of True Range
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate True Range components
    hl = high[1:] - low[1:]  # High - Low
    hc = np.abs(high[1:] - close[:-1])  # |High - Close_prev|
    lc = np.abs(low[1:] - close[:-1])   # |Low - Close_prev|
    
    # True Range = maximum of the three
    true_range = np.maximum(hl, np.maximum(hc, lc))
    
    # Calculate ATR using Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    atr = np.zeros(len(true_range))
    
    # Initialize with first period average
    atr[0] = np.mean(true_range[:period])
    
    # Apply Wilder's smoothing
    for i in range(1, len(true_range)):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
    
    # Pad with NaN for first value
    atr_full = np.full(len(close), np.nan)
    atr_full[1:] = atr
    
    return atr_full

def calculate_returns_volatility(df, period=20):
    """
    Calculate Returns-based Volatility
    Standard deviation of log returns over rolling window
    """
    close = df['close'].values
    
    # Calculate log returns
    log_returns = np.log(close[1:] / close[:-1])
    
    # Calculate rolling standard deviation using pandas for efficiency
    returns_series = pd.Series(log_returns)
    volatility_series = returns_series.rolling(window=period, min_periods=period).std()
    
    # Convert back to numpy array and pad
    volatility = np.full(len(close), np.nan)
    volatility[1:] = volatility_series.values
    
    return volatility

def calculate_parkinson_volatility(df, period=20):
    """
    Calculate Parkinson Estimator
    More efficient volatility estimator using High-Low range
    Parkinson = sqrt((1/(4*ln(2))) * (ln(H/L))^2)
    """
    high = df['high'].values
    low = df['low'].values
    
    # Calculate Parkinson estimator for each bar
    # Avoid division by zero
    hl_ratio = np.where(low > 0, high / low, 1.0)
    ln_hl = np.log(hl_ratio)
    
    # Parkinson constant
    parkinson_constant = 1.0 / (4.0 * np.log(2.0))
    
    # Calculate rolling Parkinson volatility using pandas
    ln_hl_series = pd.Series(ln_hl)
    mean_ln_hl_sq_series = (ln_hl_series ** 2).rolling(window=period, min_periods=period).mean()
    volatility_series = np.sqrt(parkinson_constant * mean_ln_hl_sq_series)
    
    volatility = volatility_series.values
    
    return volatility

def calculate_garman_klass_volatility(df, period=20):
    """
    Calculate Garman-Klass Estimator
    Uses all OHLC data for better volatility estimation
    GK = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
    """
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Avoid division by zero
    safe_open = np.where(open_price > 0, open_price, 1e-8)
    safe_close = np.where(close > 0, close, 1e-8)
    
    # Calculate GK components
    ln_hc = np.log(high / safe_close)
    ln_ho = np.log(high / safe_open)
    ln_lc = np.log(low / safe_close)
    ln_lo = np.log(low / safe_open)
    
    # Garman-Klass estimator
    gk_daily = ln_hc * ln_ho + ln_lc * ln_lo
    
    # Calculate rolling average using pandas
    gk_series = pd.Series(gk_daily)
    volatility_series = np.sqrt(gk_series.rolling(window=period, min_periods=period).mean())
    
    volatility = volatility_series.values
    
    return volatility

def calculate_yang_zhang_volatility(df, period=20):
    """
    Calculate Yang-Zhang Estimator
    Handles overnight gaps and is unbiased
    YZ = overnight_volatility + open_to_close_volatility - close_to_close_volatility
    """
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Avoid division by zero
    safe_open = np.where(open_price > 0, open_price, 1e-8)
    safe_close = np.where(close > 0, close, 1e-8)
    
    # Overnight returns (Close[t-1] to Open[t])
    overnight_returns = np.log(safe_open[1:] / safe_close[:-1])
    
    # Open-to-close returns
    oc_returns = np.log(safe_close[1:] / safe_open[1:])
    
    # Close-to-close returns
    cc_returns = np.log(safe_close[1:] / safe_close[:-1])
    
    # Rogers-Satchell component (within-day volatility)
    ln_ho = np.log(high[1:] / safe_open[1:])
    ln_hc = np.log(high[1:] / safe_close[1:])
    ln_lo = np.log(low[1:] / safe_open[1:])
    ln_lc = np.log(low[1:] / safe_close[1:])
    
    rs_component = ln_ho * ln_hc + ln_lo * ln_lc
    
    # Calculate rolling Yang-Zhang volatility using pandas for efficiency
    k = 0.34 / (1.34 + (period + 1) / (period - 1))  # bias correction
    
    # Convert to pandas series for rolling calculations
    overnight_series = pd.Series(overnight_returns)
    oc_series = pd.Series(oc_returns)
    rs_series = pd.Series(rs_component)
    
    # Rolling variance calculations
    overnight_vol_series = overnight_series.rolling(window=period, min_periods=period).var()
    oc_vol_series = oc_series.rolling(window=period, min_periods=period).var()
    rs_vol_series = rs_series.rolling(window=period, min_periods=period).mean()
    
    # Yang-Zhang estimator
    yz_vol_series = overnight_vol_series + k * oc_vol_series + (1 - k) * rs_vol_series
    volatility_series = np.sqrt(np.maximum(yz_vol_series, 0))  # Ensure non-negative
    
    # Pad with NaN for first value
    volatility = np.full(len(close), np.nan)
    volatility[1:-1] = volatility_series.values[:-1]
    
    return volatility

def normalize_volatility_series(vol_series, method='zscore', period=100):
    """
    Normalize volatility series to 0-1 range for comparison
    """
    if method == 'zscore':
        # Z-score normalization with rolling statistics
        rolling_mean = pd.Series(vol_series).rolling(period, min_periods=20).mean()
        rolling_std = pd.Series(vol_series).rolling(period, min_periods=20).std()
        normalized = (vol_series - rolling_mean) / (rolling_std + 1e-8)
        # Convert to 0-1 range using sigmoid
        normalized = 1 / (1 + np.exp(-normalized))
    
    elif method == 'percentile':
        # Rolling percentile normalization
        normalized = np.full_like(vol_series, np.nan)
        for i in range(period, len(vol_series)):
            window = vol_series[i-period:i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                normalized[i] = np.sum(valid_window <= vol_series[i]) / len(valid_window)
    
    else:  # minmax
        # Simple min-max scaling over entire series
        valid_mask = ~np.isnan(vol_series)
        if np.any(valid_mask):
            vol_min = np.nanmin(vol_series[valid_mask])
            vol_max = np.nanmax(vol_series[valid_mask])
            normalized = (vol_series - vol_min) / (vol_max - vol_min + 1e-8)
        else:
            normalized = vol_series.copy()
    
    return normalized

def create_volatility_comparison_chart(df, save_path=None):
    """
    Create comparison chart of the five volatility estimation methods
    """
    print("📊 Calculating ATR...")
    atr = calculate_atr(df, period=14)
    
    print("📊 Calculating Returns Volatility...")
    returns_vol = calculate_returns_volatility(df, period=20)
    
    print("📊 Calculating Parkinson Volatility...")
    parkinson_vol = calculate_parkinson_volatility(df, period=20)
    
    print("📊 Calculating Garman-Klass Volatility...")
    gk_vol = calculate_garman_klass_volatility(df, period=20)
    
    print("📊 Calculating Yang-Zhang Volatility...")
    yz_vol = calculate_yang_zhang_volatility(df, period=20)
    
    print("📊 Normalizing volatility series...")
    # Normalize all series for comparison
    atr_norm = normalize_volatility_series(atr, method='zscore')
    returns_norm = normalize_volatility_series(returns_vol, method='zscore')
    parkinson_norm = normalize_volatility_series(parkinson_vol, method='zscore')
    gk_norm = normalize_volatility_series(gk_vol, method='zscore')
    yz_norm = normalize_volatility_series(yz_vol, method='zscore')
    
    # Create the plot
    fig, axes = plt.subplots(7, 1, figsize=(15, 28))
    fig.suptitle('Volatility Estimation Methods Comparison', fontsize=16, fontweight='bold')
    
    # Filter data to show more recent period for clarity
    start_idx = max(0, len(df) - 200)  # Last 200 bars
    plot_df = df.iloc[start_idx:].copy()
    
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
    
    # 2. ATR (Average True Range)
    ax2 = axes[1]
    ax2.plot(x_data, atr_norm[start_idx:], 'red', linewidth=2, label='ATR (Normalized)')
    ax2.fill_between(x_data, 0, atr_norm[start_idx:], alpha=0.3, color='red')
    ax2.set_title('Method 1: Average True Range (ATR)', fontweight='bold')
    ax2.set_ylabel('Normalized Volatility')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Returns-based Volatility
    ax3 = axes[2]
    ax3.plot(x_data, returns_norm[start_idx:], 'green', linewidth=2, label='Returns Volatility (Normalized)')
    ax3.fill_between(x_data, 0, returns_norm[start_idx:], alpha=0.3, color='green')
    ax3.set_title('Method 2: Returns-based Volatility (Rolling StdDev)', fontweight='bold')
    ax3.set_ylabel('Normalized Volatility')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Parkinson Estimator
    ax4 = axes[3]
    ax4.plot(x_data, parkinson_norm[start_idx:], 'purple', linewidth=2, label='Parkinson (Normalized)')
    ax4.fill_between(x_data, 0, parkinson_norm[start_idx:], alpha=0.3, color='purple')
    ax4.set_title('Method 3: Parkinson Estimator (High-Low based)', fontweight='bold')
    ax4.set_ylabel('Normalized Volatility')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Garman-Klass Estimator
    ax5 = axes[4]
    ax5.plot(x_data, gk_norm[start_idx:], 'orange', linewidth=2, label='Garman-Klass (Normalized)')
    ax5.fill_between(x_data, 0, gk_norm[start_idx:], alpha=0.3, color='orange')
    ax5.set_title('Method 4: Garman-Klass Estimator (OHLC-based)', fontweight='bold')
    ax5.set_ylabel('Normalized Volatility')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Yang-Zhang Estimator
    ax6 = axes[5]
    ax6.plot(x_data, yz_norm[start_idx:], 'brown', linewidth=2, label='Yang-Zhang (Normalized)')
    ax6.fill_between(x_data, 0, yz_norm[start_idx:], alpha=0.3, color='brown')
    ax6.set_title('Method 5: Yang-Zhang Estimator (Gap-adjusted)', fontweight='bold')
    ax6.set_ylabel('Normalized Volatility')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. All Methods Comparison
    ax7 = axes[6]
    ax7.plot(x_data, atr_norm[start_idx:], 'red', linewidth=2, label='ATR', alpha=0.8)
    ax7.plot(x_data, returns_norm[start_idx:], 'green', linewidth=2, label='Returns Vol', alpha=0.8)
    ax7.plot(x_data, parkinson_norm[start_idx:], 'purple', linewidth=2, label='Parkinson', alpha=0.8)
    ax7.plot(x_data, gk_norm[start_idx:], 'orange', linewidth=2, label='Garman-Klass', alpha=0.8)
    ax7.plot(x_data, yz_norm[start_idx:], 'brown', linewidth=2, label='Yang-Zhang', alpha=0.8)
    ax7.set_title('All Methods Comparison (Normalized)', fontweight='bold')
    ax7.set_ylabel('Normalized Volatility')
    ax7.set_xlabel('Time')
    ax7.set_ylim(0, 1)
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Add regime quartile lines to final chart
    for level, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
        ax7.axhline(y=level, color='gray', linestyle='--', alpha=0.6)
        ax7.text(x_data[0], level + 0.02, label, fontsize=8, color='gray')
    
    # Format x-axis for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the chart
    if save_path is None:
        save_path = project_root / "data/test/volatility_methods_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Calculate and print statistics
    print(f"\n📊 Statistics for last {len(plot_df)} bars:")
    
    # Calculate correlations between methods
    methods_data = {
        'ATR': atr_norm[start_idx:],
        'Returns': returns_norm[start_idx:],
        'Parkinson': parkinson_norm[start_idx:],
        'Garman-Klass': gk_norm[start_idx:],
        'Yang-Zhang': yz_norm[start_idx:]
    }
    
    # Print statistics for each method
    for name, data in methods_data.items():
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"{name:12} - Mean: {np.mean(valid_data):.3f}, Std: {np.std(valid_data):.3f}")
    
    # Calculate correlation matrix
    print(f"\n📊 Correlation Matrix:")
    print("             ATR   Returns  Parkinson  G-K   Y-Z")
    
    method_names = list(methods_data.keys())
    for i, name1 in enumerate(method_names):
        row = f"{name1:12}"
        for j, name2 in enumerate(method_names):
            data1 = methods_data[name1]
            data2 = methods_data[name2]
            
            # Find common valid indices
            valid_mask = ~(np.isnan(data1) | np.isnan(data2))
            if np.sum(valid_mask) > 10:
                corr = np.corrcoef(data1[valid_mask], data2[valid_mask])[0, 1]
                row += f" {corr:5.3f}"
            else:
                row += "   N/A"
        print(row)
    
    plt.show()

def main():
    """Main function"""
    print("🚀 Starting Volatility Methods Comparison...")
    
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
    create_volatility_comparison_chart(df)
    
    print("🎉 Volatility methods comparison completed!")

if __name__ == "__main__":
    main()