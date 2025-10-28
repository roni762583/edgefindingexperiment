#!/usr/bin/env python3
"""
Compare ATR/(pipsize*25) with tanh and arctan transformations
Shows linear vs bounded transformations with 25-pip scaling
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
    print("📊 Comparing ATR/(pipsize*25) transformations: Linear, tanh, arctan...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded {len(df)} bars from sample data")
    
    # Extract OHLC data
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate raw ATR
    atr_raw = TechnicalIndicators.calculate_atr(high, low, close, period=14)
    
    # For EUR/USD: 1 pip = 0.0001
    pip_size = 0.0001
    
    # Calculate base scaling ATR/(pipsize*25)
    atr_25pip = atr_raw / (pip_size * 25)  # 1.0 = 25 pips
    
    # Apply transformations
    tanh_25pip = np.full(len(atr_25pip), np.nan)
    arctan_25pip = np.full(len(atr_25pip), np.nan)
    
    valid_mask = ~np.isnan(atr_25pip)
    tanh_25pip[valid_mask] = np.tanh(atr_25pip[valid_mask])
    arctan_25pip[valid_mask] = np.arctan(atr_25pip[valid_mask])
    
    # Also calculate ATR in pips for reference
    atr_pips = atr_raw / pip_size
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_pips = atr_pips[~np.isnan(atr_pips)]
    valid_linear = atr_25pip[~np.isnan(atr_25pip)]
    valid_tanh = tanh_25pip[~np.isnan(tanh_25pip)]
    valid_arctan = arctan_25pip[~np.isnan(arctan_25pip)]
    
    print(f"\n📊 Statistics:")
    print(f"ATR in Pips (reference):")
    print(f"  Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips")
    print(f"  Mean: {valid_pips.mean():.1f} pips")
    
    print(f"\nATR/(pipsize*25) - Linear:")
    print(f"  Range: [{valid_linear.min():.3f}, {valid_linear.max():.3f}]")
    print(f"  Mean: {valid_linear.mean():.3f}")
    print(f"  Interpretation: 1.0 = 25 pips")
    
    print(f"\ntanh(ATR/(pipsize*25)) - Bounded [-1,1]:")
    print(f"  Range: [{valid_tanh.min():.3f}, {valid_tanh.max():.3f}]")
    print(f"  Mean: {valid_tanh.mean():.3f}")
    print(f"  Saturation starts around: {np.arctanh(0.76):.1f} (≈{np.arctanh(0.76)*25:.0f} pips)")
    
    print(f"\narctan(ATR/(pipsize*25)) - Bounded [-π/2,π/2]:")
    print(f"  Range: [{valid_arctan.min():.3f}, {valid_arctan.max():.3f}]")
    print(f"  Mean: {valid_arctan.mean():.3f}")
    print(f"  Max theoretical: ±{np.pi/2:.3f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle('ATR/(pipsize*25) Transformation Comparison\\nLinear vs Bounded (tanh/arctan) - EUR/USD H1', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Linear ATR/(pipsize*25)
    ax1 = axes[0]
    ax1.plot(time_index, atr_25pip, 'blue', linewidth=1.5, label='ATR/(pipsize*25) - Linear')
    ax1.fill_between(time_index, atr_25pip, alpha=0.3, color='blue')
    
    # Add reference lines for pip levels
    ax1.axhline(0.4, color='green', linestyle='--', alpha=0.7, label='0.4 (10 pips)')
    ax1.axhline(0.6, color='cyan', linestyle='--', alpha=0.7, label='0.6 (15 pips)')
    ax1.axhline(0.8, color='orange', linestyle='--', alpha=0.7, label='0.8 (20 pips)')
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1.0 (25 pips)')
    ax1.axhline(1.2, color='purple', linestyle=':', alpha=0.5, label='1.2 (30 pips)')
    
    ax1.set_title('1. ATR/(pipsize*25) - Linear Scaling (Unbounded)', fontweight='bold')
    ax1.set_ylabel('Linear Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_linear) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_linear.min():.3f}, {valid_linear.max():.3f}]\\nMean: {valid_linear.mean():.3f}\\nInterpretation: 1.0 = 25 pips\\nLinear, unbounded scaling', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: tanh(ATR/(pipsize*25))
    ax2 = axes[1]
    ax2.plot(time_index, tanh_25pip, 'red', linewidth=1.5, label='tanh(ATR/(pipsize*25))')
    ax2.fill_between(time_index, tanh_25pip, alpha=0.3, color='red')
    
    # Add interpretation lines for tanh
    ax2.axhline(np.tanh(0.4), color='green', linestyle='--', alpha=0.7, label=f'tanh(0.4) = {np.tanh(0.4):.3f} (10 pips)')
    ax2.axhline(np.tanh(0.8), color='orange', linestyle='--', alpha=0.7, label=f'tanh(0.8) = {np.tanh(0.8):.3f} (20 pips)')
    ax2.axhline(np.tanh(1.0), color='red', linestyle='--', alpha=0.7, label=f'tanh(1.0) = {np.tanh(1.0):.3f} (25 pips)')
    ax2.axhline(0.5, color='gray', linestyle='-', alpha=0.3, label='0.5 reference')
    ax2.axhline(0.76, color='purple', linestyle=':', alpha=0.5, label='0.76 (saturation start)')
    
    ax2.set_title('2. tanh(ATR/(pipsize*25)) - S-Curve Bounded [-1,1]', fontweight='bold')
    ax2.set_ylabel('tanh Value')
    ax2.set_ylim(-0.05, 1.0)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_tanh) > 0:
        ax2.text(0.02, 0.95, f'Range: [{valid_tanh.min():.3f}, {valid_tanh.max():.3f}]\\nMean: {valid_tanh.mean():.3f}\\nBounded: [-1, 1]\\nS-curve, saturates at extremes', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: arctan(ATR/(pipsize*25))
    ax3 = axes[2]
    ax3.plot(time_index, arctan_25pip, 'green', linewidth=1.5, label='arctan(ATR/(pipsize*25))')
    ax3.fill_between(time_index, arctan_25pip, alpha=0.3, color='green')
    
    # Add interpretation lines for arctan
    ax3.axhline(np.arctan(0.4), color='green', linestyle='--', alpha=0.7, label=f'arctan(0.4) = {np.arctan(0.4):.3f} (10 pips)')
    ax3.axhline(np.arctan(0.8), color='orange', linestyle='--', alpha=0.7, label=f'arctan(0.8) = {np.arctan(0.8):.3f} (20 pips)')
    ax3.axhline(np.arctan(1.0), color='red', linestyle='--', alpha=0.7, label=f'arctan(1.0) = {np.arctan(1.0):.3f} (25 pips)')
    ax3.axhline(np.pi/4, color='gray', linestyle='-', alpha=0.3, label=f'π/4 = {np.pi/4:.3f}')
    ax3.axhline(np.pi/2, color='purple', linestyle=':', alpha=0.5, label=f'π/2 = {np.pi/2:.3f} (max)')
    
    ax3.set_title('3. arctan(ATR/(pipsize*25)) - Gradual Curve Bounded [-π/2,π/2]', fontweight='bold')
    ax3.set_ylabel('arctan Value')
    ax3.set_xlabel('Time')
    ax3.set_ylim(-0.1, 1.8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_arctan) > 0:
        ax3.text(0.02, 0.95, f'Range: [{valid_arctan.min():.3f}, {valid_arctan.max():.3f}]\\nMean: {valid_arctan.mean():.3f}\\nBounded: [-π/2, π/2] = [-1.571, 1.571]\\nGradual curve, less saturation', 
                transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in axes:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/atr_25pip_transformations_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print detailed transformation analysis
    print(f"\n💡 Transformation Behavior Analysis:")
    print(f"="*60)
    
    # Create transformation comparison table
    pip_levels = [5, 10, 15, 20, 25, 30, 40, 50]
    print(f"{'Pips':<4} {'Linear':<8} {'tanh':<8} {'arctan':<8} {'Notes'}")
    print("-" * 55)
    
    for pips in pip_levels:
        linear_val = pips / 25
        tanh_val = np.tanh(linear_val)
        arctan_val = np.arctan(linear_val)
        
        if pips <= 15:
            note = "Low volatility"
        elif pips <= 25:
            note = "Medium volatility"
        elif pips <= 40:
            note = "High volatility"
        else:
            note = "Very high volatility"
        
        print(f"{pips:<4} {linear_val:<8.3f} {tanh_val:<8.3f} {arctan_val:<8.3f} {note}")
    
    print(f"\n🎯 Key Insights:")
    print(f"")
    print(f"Linear ATR/(pipsize*25):")
    print(f"  ✓ Preserves exact pip relationships")
    print(f"  ✓ Unbounded - captures all extremes")
    print(f"  ✓ Your range [{valid_linear.min():.3f}, {valid_linear.max():.3f}] shows good spread")
    print(f"  ✗ May need special handling for ML models")
    print(f"")
    print(f"tanh(ATR/(pipsize*25)):")
    print(f"  ✓ Bounded [0, 1] for this data")
    print(f"  ✓ S-curve emphasizes differences at low volatility")
    print(f"  ✓ Strong saturation prevents outliers")
    print(f"  ✗ Compresses high volatility differences")
    print(f"")
    print(f"arctan(ATR/(pipsize*25)):")
    print(f"  ✓ Bounded but larger range than tanh")
    print(f"  ✓ Gentler curve, less aggressive saturation")
    print(f"  ✓ More linear-like in middle ranges")
    print(f"  ✗ Range depends on π, less intuitive")
    print(f"")
    print(f"📊 For Your Data (16-pip average):")
    avg_pips = valid_pips.mean()
    avg_linear = avg_pips / 25
    avg_tanh = np.tanh(avg_linear)
    avg_arctan = np.arctan(avg_linear)
    
    print(f"  • {avg_pips:.1f} pips → Linear: {avg_linear:.3f}")
    print(f"  • {avg_pips:.1f} pips → tanh: {avg_tanh:.3f}")
    print(f"  • {avg_pips:.1f} pips → arctan: {avg_arctan:.3f}")
    print(f"")
    print(f"🎯 Recommendation:")
    print(f"  • Linear: Best for statistical accuracy and interpretability")
    print(f"  • tanh: Best for neural networks needing bounded inputs")
    print(f"  • arctan: Best compromise between linear and bounded")
    print(f"")
    print(f"For TCNAE + LightGBM:")
    print(f"  • LightGBM: Can handle linear (trees split well on unbounded)")
    print(f"  • TCNAE: May benefit from tanh or arctan bounded inputs")
    print(f"  • Consider: Linear for LightGBM, tanh for TCNAE")
    
    plt.show()
    print("🎉 ATR/(pipsize*25) transformation comparison completed!")

if __name__ == "__main__":
    main()