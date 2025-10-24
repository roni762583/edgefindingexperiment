#!/usr/bin/env python3
"""
Analyze ADX Range - Determine optimal scaling for direction indicator
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def main():
    print("📊 Analyzing ADX Range for Optimal Scaling...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    # Calculate ADX
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    print(f"🔄 Calculating ADX for {len(df)} bars...")
    adx = TechnicalIndicators.calculate_adx(high, low, close, period=14)
    
    # Filter valid ADX values
    valid_adx = adx[~np.isnan(adx)]
    
    print(f"\n📊 ADX Statistics:")
    print(f"Valid values: {len(valid_adx)}/{len(adx)}")
    print(f"Range: [{np.min(valid_adx):.2f}, {np.max(valid_adx):.2f}]")
    print(f"Mean: {np.mean(valid_adx):.2f}")
    print(f"Median: {np.median(valid_adx):.2f}")
    print(f"Std: {np.std(valid_adx):.2f}")
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nADX Percentiles:")
    for p in percentiles:
        val = np.percentile(valid_adx, p)
        print(f"P{p:2d}: {val:6.2f}")
    
    # Test different scaling approaches
    scaling_options = {
        'No scaling': valid_adx,
        'ADX / 100': valid_adx / 100,
        'ADX / 100 * 10': valid_adx / 100 * 10,
        'ADX / 50': valid_adx / 50,
        'Normalize [0,1]': (valid_adx - np.min(valid_adx)) / (np.max(valid_adx) - np.min(valid_adx)),
        'Z-score': (valid_adx - np.mean(valid_adx)) / np.std(valid_adx),
        'tanh(ADX/25)': np.tanh(valid_adx / 25),
        'arctan(ADX/50)': np.arctan(valid_adx / 50)
    }
    
    print(f"\n📊 Scaling Options Comparison:")
    print(f"{'Method':<20} {'Range':<20} {'Mean':<8} {'Std':<8}")
    print("-" * 60)
    
    for name, scaled in scaling_options.items():
        range_str = f"[{np.min(scaled):.3f}, {np.max(scaled):.3f}]"
        print(f"{name:<20} {range_str:<20} {np.mean(scaled):<8.3f} {np.std(scaled):<8.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('ADX Scaling Options Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    time_index = range(len(valid_adx))
    
    for i, (name, scaled) in enumerate(scaling_options.items()):
        if i < len(axes):
            ax = axes[i]
            ax.plot(time_index, scaled, 'blue', linewidth=1.5, alpha=0.8)
            ax.fill_between(time_index, scaled, alpha=0.3, color='blue')
            ax.set_title(f'{name}\nRange: [{np.min(scaled):.3f}, {np.max(scaled):.3f}]')
            ax.set_ylabel('Scaled Value')
            ax.grid(True, alpha=0.3)
            
            # Add reference lines for common ranges
            if 'tanh' in name.lower() or 'arctan' in name.lower():
                ax.axhline(0, color='black', linestyle='-', alpha=0.5)
                ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
                ax.axhline(-0.5, color='red', linestyle='--', alpha=0.5)
            elif '[0,1]' in name or '/ 100' in name:
                ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
                if np.max(scaled) <= 1:
                    ax.set_ylim(-0.1, 1.1)
    
    # Remove empty subplots
    for i in range(len(scaling_options), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/adx_scaling_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Chart saved to: {save_path}")
    
    # Recommendations
    print(f"\n🎯 Scaling Recommendations:")
    print(f"1. For [0,1] range: Normalize or ADX/100")
    print(f"2. For [-1,1] range: Z-score or tanh(ADX/25)")
    print(f"3. For bounded smooth: arctan(ADX/50)")
    print(f"4. Original tested: ADX/100*10 gives [0, {np.max(valid_adx)/100*10:.1f}]")
    
    # Suggest best option based on distribution
    if np.max(valid_adx) > 50:
        print(f"\n✅ Recommended: tanh(ADX/25) for smooth [-1,1] range")
        recommended = np.tanh(valid_adx / 25)
        print(f"   Range: [{np.min(recommended):.3f}, {np.max(recommended):.3f}]")
    else:
        print(f"\n✅ Recommended: ADX/100 for [0,1] range")
        recommended = valid_adx / 100
        print(f"   Range: [{np.min(recommended):.3f}, {np.max(recommended):.3f}]")
    
    plt.show()
    print("🎉 ADX range analysis completed!")

if __name__ == "__main__":
    main()