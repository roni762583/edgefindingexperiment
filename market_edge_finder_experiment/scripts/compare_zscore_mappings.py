#!/usr/bin/env python3
"""
Compare tanh and arctan mapping for zscore ATR

Compares different transformation methods for ATR z-score:
1. Raw z-score (current)
2. tanh(z-score) mapping
3. arctan(z-score) mapping
4. Normalized arctan mapping
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def calculate_atr_zscore(high, low, close, atr_window=14, sma_window=20, stdev_window=500):
    """Calculate ATR z-score components"""
    # Calculate ATR
    atr = TechnicalIndicators.calculate_atr(high, low, close, atr_window)
    
    # Calculate SMA20 of ATR
    atr_sma = np.full(len(atr), np.nan)
    for i in range(sma_window - 1, len(atr)):
        if not np.any(np.isnan(atr[i-sma_window+1:i+1])):
            atr_sma[i] = np.mean(atr[i-sma_window+1:i+1])
    
    # Calculate rolling standard deviation
    stdev_window = min(stdev_window, len(atr))
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

def apply_transformations(zscore):
    """Apply different transformation methods to z-score"""
    transformations = {}
    
    # 1. Raw z-score (current implementation)
    transformations['Raw Z-Score'] = zscore.copy()
    
    # 2. tanh transformation (bounded [-1, 1])
    valid_mask = ~np.isnan(zscore)
    tanh_transformed = np.full(len(zscore), np.nan)
    tanh_transformed[valid_mask] = np.tanh(zscore[valid_mask])
    transformations['tanh(z-score)'] = tanh_transformed
    
    # 3. arctan transformation (bounded [-π/2, π/2])
    arctan_transformed = np.full(len(zscore), np.nan)
    arctan_transformed[valid_mask] = np.arctan(zscore[valid_mask])
    transformations['arctan(z-score)'] = arctan_transformed
    
    # 4. Normalized arctan (bounded [0, 1])
    arctan_normalized = np.full(len(zscore), np.nan)
    arctan_normalized[valid_mask] = (np.arctan(zscore[valid_mask]) + np.pi/2) / np.pi
    transformations['Normalized arctan'] = arctan_normalized
    
    # 5. Inverted tanh (like old volatility: 1 - tanh)
    inverted_tanh = np.full(len(zscore), np.nan)
    inverted_tanh[valid_mask] = 1 - np.tanh(zscore[valid_mask])
    transformations['1 - tanh(z-score)'] = inverted_tanh
    
    return transformations

def main():
    print("📊 Comparing tanh and arctan mappings for ATR z-score...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Calculate ATR z-score
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    zscore = calculate_atr_zscore(high, low, close)
    valid_zscore = zscore[~np.isnan(zscore)]
    
    print(f"📈 ATR Z-Score Statistics:")
    print(f"  Valid values: {len(valid_zscore)}/{len(zscore)}")
    print(f"  Range: [{np.min(valid_zscore):.3f}, {np.max(valid_zscore):.3f}]")
    print(f"  Mean: {np.mean(valid_zscore):.3f}")
    print(f"  Std: {np.std(valid_zscore):.3f}")
    print()
    
    # Apply transformations
    transformations = apply_transformations(zscore)
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Create comparison chart
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('ATR Z-Score Transformation Methods Comparison\\nEUR/USD H1 Data (700 bars)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, transformed) in enumerate(transformations.items()):
        if i < len(axes):
            ax = axes[i]
            valid_data = ~np.isnan(transformed)
            
            if valid_data.any():
                # Plot the transformation
                ax.plot(time_index, transformed, color=colors[i % len(colors)], 
                       linewidth=1.5, alpha=0.8, label=name)
                ax.fill_between(time_index, transformed, alpha=0.3, color=colors[i % len(colors)])
                
                # Add statistics
                valid_vals = transformed[valid_data]
                mean_val = np.mean(valid_vals)
                std_val = np.std(valid_vals)
                min_val = np.min(valid_vals)
                max_val = np.max(valid_vals)
                
                stats_text = f"Range: [{min_val:.3f}, {max_val:.3f}]\\nMean: {mean_val:.3f}\\nStd: {std_val:.3f}"
                ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add reference lines based on transformation
                if 'tanh' in name.lower():
                    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
                    if '1 -' in name:
                        ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='1.0')
                        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='0.5')
                        ax.set_ylim(-0.1, 2.1)
                    else:
                        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='±0.5')
                        ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
                        ax.set_ylim(-1.1, 1.1)
                elif 'arctan' in name.lower():
                    if 'Normalized' in name:
                        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='0.5')
                        ax.axhline(0.75, color='gray', linestyle=':', alpha=0.5, label='0.75')
                        ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='0.25')
                        ax.set_ylim(-0.1, 1.1)
                    else:
                        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
                        ax.axhline(np.pi/4, color='gray', linestyle='--', alpha=0.5, label='π/4')
                        ax.axhline(-np.pi/4, color='gray', linestyle='--', alpha=0.5, label='-π/4')
                        ax.set_ylim(-1.8, 1.8)
                else:  # Raw z-score
                    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
                    ax.axhline(1, color='red', linestyle='--', alpha=0.5, label='±1σ')
                    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
                    ax.axhline(2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
                    ax.axhline(-2, color='orange', linestyle=':', alpha=0.5)
                    ax.set_ylim(-2.5, 2.5)
            
            ax.set_title(f'{i+1}. {name}', fontweight='bold')
            ax.set_ylabel('Transformed Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot if any
    if len(transformations) < len(axes):
        for i in range(len(transformations), len(axes)):
            fig.delaxes(axes[i])
    
    # Format x-axis
    for ax in axes[:len(transformations)]:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('Time')
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/zscore_transformations_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved to: {save_path}")
    
    # Create statistical comparison table
    print(f"\\n📊 Transformation Methods Comparison:")
    print(f"{'Method':<20} {'Range':<25} {'Mean':<8} {'Std':<8} {'Properties'}")
    print("-" * 80)
    
    for name, transformed in transformations.items():
        valid_vals = transformed[~np.isnan(transformed)]
        if len(valid_vals) > 0:
            range_str = f"[{np.min(valid_vals):.3f}, {np.max(valid_vals):.3f}]"
            mean_val = np.mean(valid_vals)
            std_val = np.std(valid_vals)
            
            # Add properties description
            if name == 'Raw Z-Score':
                properties = "Unbounded, centered at 0"
            elif name == 'tanh(z-score)':
                properties = "Bounded [-1,1], S-curve"
            elif name == 'arctan(z-score)':
                properties = f"Bounded [-π/2,π/2], gradual"
            elif name == 'Normalized arctan':
                properties = "Bounded [0,1], gradual"
            elif name == '1 - tanh(z-score)':
                properties = "Bounded [0,2], inverted S-curve"
            else:
                properties = "Custom"
            
            print(f"{name:<20} {range_str:<25} {mean_val:<8.3f} {std_val:<8.3f} {properties}")
    
    # Show theoretical comparison
    print(f"\\n🔬 Theoretical Behavior Analysis:")
    test_zscores = np.array([-3, -2, -1, 0, 1, 2, 3])
    
    print(f"\\nZ-Score → Transformations:")
    print(f"{'Z-Score':<8} {'Raw':<8} {'tanh':<8} {'arctan':<8} {'Norm.arctan':<12} {'1-tanh':<8}")
    print("-" * 60)
    
    for z in test_zscores:
        raw = z
        tanh_val = np.tanh(z)
        arctan_val = np.arctan(z)
        norm_arctan = (np.arctan(z) + np.pi/2) / np.pi
        inv_tanh = 1 - np.tanh(z)
        
        print(f"{z:<8.0f} {raw:<8.3f} {tanh_val:<8.3f} {arctan_val:<8.3f} {norm_arctan:<12.3f} {inv_tanh:<8.3f}")
    
    print(f"\\n🎯 Recommendations:")
    print(f"• Raw Z-Score: Best for preserving original statistical meaning")
    print(f"• tanh: Good for bounded [-1,1] with S-curve response to extremes")
    print(f"• arctan: Gentler bounded response, less aggressive than tanh")
    print(f"• Normalized arctan: Maps to [0,1] for easy interpretation")
    print(f"• 1-tanh: Inverted for 'volatility level' interpretation")
    
    plt.show()
    print("🎉 ATR z-score transformation comparison completed!")

if __name__ == "__main__":
    main()