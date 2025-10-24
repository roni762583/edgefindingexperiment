#!/usr/bin/env python3
"""
Fast Volatility Comparison - Core Methods Only

Usage: python3 scripts/fast_volatility_comparison.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Main function"""
    print("🚀 Fast Volatility Comparison...")
    
    # Load data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} bars")
    
    # Use last 100 bars only for speed
    data = df.tail(100).copy().reset_index(drop=True)
    
    print("🔄 Calculating volatility methods...")
    
    # 1. ATR (Simple True Range)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = data['tr'].rolling(14).mean()
    
    # 2. Returns Volatility
    data['returns'] = np.log(data['close'] / data['close'].shift(1))
    data['ret_vol'] = data['returns'].rolling(20).std()
    
    # 3. Parkinson (High-Low)
    data['parkinson'] = np.sqrt(0.3606 * (np.log(data['high'] / data['low']) ** 2).rolling(20).mean())
    
    # 4. Garman-Klass (simplified)
    data['gk'] = np.sqrt((
        (np.log(data['high'] / data['close']) * np.log(data['high'] / data['open'])) +
        (np.log(data['low'] / data['close']) * np.log(data['low'] / data['open']))
    ).rolling(20).mean())
    
    # 5. Simple Range Volatility
    data['range_vol'] = ((data['high'] - data['low']) / data['close']).rolling(20).std()
    
    # Normalize to 0-1
    vol_cols = ['atr', 'ret_vol', 'parkinson', 'gk', 'range_vol']
    for col in vol_cols:
        data[f'{col}_norm'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    
    print("🔄 Creating chart...")
    
    # Plot
    fig, axes = plt.subplots(6, 1, figsize=(12, 20))
    fig.suptitle('Volatility Methods Comparison (Fast)', fontsize=14, fontweight='bold')
    
    # Price
    axes[0].plot(data['close'], 'b-', linewidth=1.5)
    axes[0].set_title('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Volatility methods
    methods = [
        ('atr_norm', 'ATR', 'red'),
        ('ret_vol_norm', 'Returns Vol', 'green'), 
        ('parkinson_norm', 'Parkinson', 'purple'),
        ('gk_norm', 'Garman-Klass', 'orange'),
        ('range_vol_norm', 'Range Vol', 'brown')
    ]
    
    for i, (col, title, color) in enumerate(methods):
        ax = axes[i+1]
        ax.plot(data[col], color, linewidth=2, label=title)
        ax.fill_between(range(len(data)), 0, data[col], alpha=0.3, color=color)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Quartile lines
        for level in [0.25, 0.5, 0.75]:
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    save_path = project_root / "data/test/fast_volatility_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Stats
    print(f"\n📊 Correlations:")
    norm_cols = [col for col, _, _ in methods]
    norm_data = data[norm_cols].dropna()
    print(norm_data.corr().round(3))
    
    plt.show()
    print("🎉 Done!")

if __name__ == "__main__":
    main()