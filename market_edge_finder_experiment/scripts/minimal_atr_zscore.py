#!/usr/bin/env python3
"""
Minimal ATR Z-score Test - No plotting, just calculations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent

def main():
    print("🚀 Minimal ATR Z-score Test...")
    
    # Load data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} bars")
    
    # Use last 50 bars
    data = df.tail(50).copy()
    
    print("🔄 Calculating ATR...")
    # ATR calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = data['tr'].rolling(14).mean()
    
    print("🔄 Z-score normalization...")
    # Z-score using pattern from new_swt: (ATR - SMA20(ATR)) / STDEV500(ATR)
    # Using shorter windows for demo: SMA(10) and STDEV(20)
    data['atr_sma'] = data['atr'].rolling(10).mean()
    data['atr_std'] = data['atr'].rolling(20).std()
    data['atr_zscore'] = (data['atr'] - data['atr_sma']) / data['atr_std']
    
    print("🔄 Arctan normalization...")
    # Arctan normalization: (arctan(zscore) + π/2) / π
    data['atr_normalized'] = (np.arctan(data['atr_zscore']) + np.pi/2) / np.pi
    
    # Clean data
    clean_data = data[['atr', 'atr_zscore', 'atr_normalized']].dropna()
    
    print(f"\n📊 Results ({len(clean_data)} valid bars):")
    print(f"ATR          - Mean: {clean_data['atr'].mean():.4f}, Std: {clean_data['atr'].std():.4f}")
    print(f"ATR Z-score  - Mean: {clean_data['atr_zscore'].mean():.4f}, Std: {clean_data['atr_zscore'].std():.4f}")
    print(f"ATR Norm     - Mean: {clean_data['atr_normalized'].mean():.4f}, Std: {clean_data['atr_normalized'].std():.4f}")
    
    # Show some actual values
    print(f"\n📊 Sample values (last 5 bars):")
    sample = clean_data.tail(5)
    for i, (idx, row) in enumerate(sample.iterrows()):
        print(f"Bar {i+1}: ATR={row['atr']:.4f}, Z-score={row['atr_zscore']:.3f}, Normalized={row['atr_normalized']:.3f}")
    
    # Regime distribution
    q1 = (clean_data['atr_normalized'] < 0.25).mean()
    q2 = ((clean_data['atr_normalized'] >= 0.25) & (clean_data['atr_normalized'] < 0.50)).mean() 
    q3 = ((clean_data['atr_normalized'] >= 0.50) & (clean_data['atr_normalized'] < 0.75)).mean()
    q4 = (clean_data['atr_normalized'] >= 0.75).mean()
    
    print(f"\n📊 Volatility Regime Distribution:")
    print(f"Q1 (0.00-0.25): {q1:.1%}")
    print(f"Q2 (0.25-0.50): {q2:.1%}")
    print(f"Q3 (0.50-0.75): {q3:.1%}")
    print(f"Q4 (0.75-1.00): {q4:.1%}")
    
    print("\n🔧 Implementation for your system:")
    print("def volatility_score(atr_series):")
    print("    atr_sma20 = atr_series.rolling(20).mean()")
    print("    atr_std500 = atr_series.rolling(500).std()")
    print("    zscore = (atr_series - atr_sma20) / atr_std500")
    print("    return (np.arctan(zscore) + np.pi/2) / np.pi")
    
    print("\n🎉 ATR Z-score test completed!")

if __name__ == "__main__":
    main()