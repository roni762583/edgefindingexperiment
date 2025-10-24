#!/usr/bin/env python3
"""
Quick ATR Z-score Test
Tests: (ATR - SMA20(ATR)) / STDEV500(ATR) and arctan normalization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("🚀 Quick ATR Z-score Test...")
    
    # Load data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} bars")
    
    # Use last 80 bars for speed
    data = df.tail(80).copy().reset_index(drop=True)
    
    print("🔄 Calculating ATR...")
    # Simple ATR calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = data['tr'].rolling(14).mean()
    
    print("🔄 Calculating Z-score...")
    # Z-score: (ATR - SMA20(ATR)) / STDEV500(ATR)
    # Since we only have 80 bars, use shorter windows for demo
    data['atr_sma20'] = data['atr'].rolling(10).mean()  # Use 10 instead of 20
    data['atr_std'] = data['atr'].rolling(30).std()     # Use 30 instead of 500
    data['atr_zscore'] = (data['atr'] - data['atr_sma20']) / data['atr_std']
    
    print("🔄 Applying arctan normalization...")
    # Arctan normalization: (arctan(zscore) + π/2) / π
    data['atr_normalized'] = (np.arctan(data['atr_zscore']) + np.pi/2) / np.pi
    
    # Simple min-max for comparison
    data['atr_minmax'] = (data['atr'] - data['atr'].min()) / (data['atr'].max() - data['atr'].min())
    
    print("🔄 Creating chart...")
    # Simple plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle('ATR Z-score Normalization Test', fontsize=12)
    
    x = range(len(data))
    
    # 1. Price
    axes[0].plot(x, data['close'], 'b-', linewidth=1.5)
    axes[0].set_title('Price')
    axes[0].grid(True, alpha=0.3)
    
    # 2. ATR
    axes[1].plot(x, data['atr'], 'r-', linewidth=2)
    axes[1].set_title('ATR')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Z-score
    axes[2].plot(x, data['atr_zscore'], 'purple', linewidth=2)
    axes[2].set_title('ATR Z-score')
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Normalized comparison
    axes[3].plot(x, data['atr_normalized'], 'darkgreen', linewidth=2, label='Arctan Norm', alpha=0.8)
    axes[3].plot(x, data['atr_minmax'], 'orange', linewidth=2, label='Min-Max Norm', alpha=0.8)
    axes[3].set_title('Normalized Comparison')
    axes[3].set_ylim(0, 1)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Add quartile lines to last chart
    for level in [0.25, 0.5, 0.75]:
        axes[3].axhline(y=level, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    save_path = project_root / "data/test/quick_atr_zscore_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Stats
    valid_data = data[['atr_normalized', 'atr_minmax']].dropna()
    if len(valid_data) > 0:
        print(f"\n📊 Statistics:")
        print(f"Arctan Normalized - Mean: {valid_data['atr_normalized'].mean():.3f}, Std: {valid_data['atr_normalized'].std():.3f}")
        print(f"Min-Max Normalized - Mean: {valid_data['atr_minmax'].mean():.3f}, Std: {valid_data['atr_minmax'].std():.3f}")
        
        corr = valid_data['atr_normalized'].corr(valid_data['atr_minmax'])
        print(f"Correlation: {corr:.3f}")
        
        # Regime distribution
        q1 = (valid_data['atr_normalized'] < 0.25).mean()
        q2 = ((valid_data['atr_normalized'] >= 0.25) & (valid_data['atr_normalized'] < 0.50)).mean()
        q3 = ((valid_data['atr_normalized'] >= 0.50) & (valid_data['atr_normalized'] < 0.75)).mean()
        q4 = (valid_data['atr_normalized'] >= 0.75).mean()
        
        print(f"\n📊 Volatility Regime Distribution:")
        print(f"Q1 (Low):     {q1:.1%}")
        print(f"Q2 (Mod):     {q2:.1%}")
        print(f"Q3 (High):    {q3:.1%}")
        print(f"Q4 (V.High):  {q4:.1%}")
    
    plt.show()
    print("🎉 Done!")

if __name__ == "__main__":
    main()