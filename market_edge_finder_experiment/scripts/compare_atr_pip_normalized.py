#!/usr/bin/env python3
"""
Compare ATR vs ATR/(pipsize*100) 
Shows direct pip normalization vs raw ATR
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
    print("📊 Comparing ATR vs ATR/(pipsize*100)...")
    
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
    
    # Calculate ATR/(pipsize*100) - this gives ATR as a fraction where 1.0 = 100 pips
    atr_pip_normalized = atr_raw / (pip_size * 100)
    
    # Also calculate ATR in pips for reference
    atr_pips = atr_raw / pip_size
    
    # Create time index
    time_index = pd.to_datetime(df['time']) if 'time' in df.columns else range(len(df))
    
    # Print statistics
    valid_raw = atr_raw[~np.isnan(atr_raw)]
    valid_normalized = atr_pip_normalized[~np.isnan(atr_pip_normalized)]
    valid_pips = atr_pips[~np.isnan(atr_pips)]
    
    print(f"\n📊 Statistics:")
    print(f"Raw ATR (price units):")
    print(f"  Valid: {len(valid_raw)}/{len(atr_raw)} ({len(valid_raw)/len(atr_raw)*100:.1f}%)")
    print(f"  Range: [{valid_raw.min():.6f}, {valid_raw.max():.6f}]")
    print(f"  Mean: {valid_raw.mean():.6f}")
    
    print(f"\nATR/(pipsize*100) (normalized):")
    print(f"  Valid: {len(valid_normalized)}/{len(atr_pip_normalized)} ({len(valid_normalized)/len(atr_pip_normalized)*100:.1f}%)")
    print(f"  Range: [{valid_normalized.min():.3f}, {valid_normalized.max():.3f}]")
    print(f"  Mean: {valid_normalized.mean():.3f}")
    print(f"  Interpretation: 1.0 = 100 pips, 0.1 = 10 pips")
    
    print(f"\nATR in Pips (reference):")
    print(f"  Range: [{valid_pips.min():.1f}, {valid_pips.max():.1f}] pips")
    print(f"  Mean: {valid_pips.mean():.1f} pips")
    
    # Verify the normalization
    print(f"\n🔍 Normalization Verification:")
    print(f"  ATR/(pipsize*100) mean: {valid_normalized.mean():.3f}")
    print(f"  ATR in pips mean / 100: {valid_pips.mean()/100:.3f}")
    print(f"  Match: {abs(valid_normalized.mean() - valid_pips.mean()/100) < 0.001}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('ATR Comparison: Raw vs Pip-Normalized\\nEUR/USD H1 Data (1.0 = 100 pips)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Raw ATR
    ax1.plot(time_index, atr_raw, 'blue', linewidth=1.5, label='Raw ATR (14-period)')
    ax1.fill_between(time_index, atr_raw, alpha=0.3, color='blue')
    
    ax1.set_title('1. Raw ATR - Price Units', fontweight='bold')
    ax1.set_ylabel('ATR (Price Units)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    if len(valid_raw) > 0:
        ax1.text(0.02, 0.95, f'Range: [{valid_raw.min():.6f}, {valid_raw.max():.6f}]\\nMean: {valid_raw.mean():.6f}\\nValid: {len(valid_raw)}/{len(atr_raw)}\\n\\nInterpretation:\\n• Raw EUR/USD price units\\n• Instrument specific\\n• Not normalized', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: ATR/(pipsize*100)
    ax2.plot(time_index, atr_pip_normalized, 'green', linewidth=1.5, label='ATR/(pipsize*100)')
    ax2.fill_between(time_index, atr_pip_normalized, alpha=0.3, color='green')
    
    # Add reference lines
    ax2.axhline(0.1, color='blue', linestyle='--', alpha=0.7, label='0.1 (10 pips)')
    ax2.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='0.2 (20 pips)')
    ax2.axhline(0.3, color='red', linestyle=':', alpha=0.5, label='0.3 (30 pips)')
    ax2.axhline(0.5, color='purple', linestyle=':', alpha=0.5, label='0.5 (50 pips)')
    
    ax2.set_title('2. ATR/(pipsize*100) - Pip-Normalized (1.0 = 100 pips)', fontweight='bold')
    ax2.set_ylabel('Normalized ATR')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics and interpretation
    if len(valid_normalized) > 0:
        # Calculate volatility level distribution
        low_vol = np.sum(valid_normalized < 0.1) / len(valid_normalized) * 100
        med_vol = np.sum((valid_normalized >= 0.1) & (valid_normalized < 0.2)) / len(valid_normalized) * 100
        high_vol = np.sum(valid_normalized >= 0.2) / len(valid_normalized) * 100
        
        ax2.text(0.02, 0.95, f'Range: [{valid_normalized.min():.3f}, {valid_normalized.max():.3f}]\\nMean: {valid_normalized.mean():.3f}\\nValid: {len(valid_normalized)}/{len(atr_pip_normalized)}\\n\\nInterpretation:\\n• 1.0 = 100 pips volatility\\n• 0.1 = 10 pips volatility\\n• Mean {valid_normalized.mean():.3f} = {valid_normalized.mean()*100:.1f} pips\\n\\nDistribution:\\n• < 10 pips: {low_vol:.1f}%\\n• 10-20 pips: {med_vol:.1f}%\\n• > 20 pips: {high_vol:.1f}%', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format time axis
    for ax in [ax1, ax2]:
        if hasattr(time_index, 'dtype') and 'datetime' in str(time_index.dtype):
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/atr_pip_normalized_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print practical interpretation
    print(f"\n💡 Practical Interpretation:")
    print(f"="*50)
    print(f"Raw ATR = {valid_raw.mean():.6f} means:")
    print(f"  • {valid_raw.mean():.6f} EUR/USD price units")
    print(f"  • Currency pair specific measurement")
    print(f"  • Requires pip conversion for interpretation")
    print(f"")
    print(f"ATR/(pipsize*100) = {valid_normalized.mean():.3f} means:")
    print(f"  • {valid_normalized.mean()*100:.1f} pips average volatility")
    print(f"  • Direct pip interpretation (multiply by 100)")
    print(f"  • Cross-instrument comparable")
    print(f"  • 1.0 = 100 pips (major volatility)")
    print(f"  • 0.1 = 10 pips (low volatility)")
    print(f"")
    print(f"🎯 Scaling Examples:")
    print(f"  • 0.05 = 5 pips (very low volatility)")
    print(f"  • 0.10 = 10 pips (low volatility)")
    print(f"  • 0.15 = 15 pips (medium volatility)")
    print(f"  • 0.20 = 20 pips (high volatility)")
    print(f"  • 0.30 = 30 pips (very high volatility)")
    print(f"  • 0.50 = 50 pips (extreme volatility)")
    print(f"  • 1.00 = 100 pips (crisis-level volatility)")
    print(f"")
    print(f"✅ Benefits of ATR/(pipsize*100):")
    print(f"  ✓ Direct pip interpretation")
    print(f"  ✓ Linear scaling (no transformation)")
    print(f"  ✓ Cross-instrument comparable")
    print(f"  ✓ Intuitive for traders")
    print(f"  ✓ Unbounded (captures all extremes)")
    print(f"  ✓ Simple calculation")
    print(f"")
    print(f"🤔 Considerations:")
    print(f"  • Unbounded output (may need handling for ML)")
    print(f"  • Linear scaling (no compression of extremes)")
    print(f"  • Requires correct pip size for each instrument")
    
    plt.show()
    print("🎉 ATR vs pip-normalized comparison completed!")

if __name__ == "__main__":
    main()