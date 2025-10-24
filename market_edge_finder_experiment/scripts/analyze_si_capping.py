#!/usr/bin/env python3
"""
Analyze SI Capping - Examine SI values before and after capping

Shows raw SI values vs capped values and explores different capping strategies.

Usage: python3 scripts/analyze_si_capping.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators
from configs.instruments import get_pip_value

def calculate_si_with_analysis(open_prices, high, low, close, instrument):
    """
    Calculate SI values with detailed analysis of pre/post capping
    """
    EPSILON = 1e-10
    
    if len(close) < 2:
        return None
    
    # Step 1: Normalize OHLC to USD values
    open_usd, high_usd, low_usd, close_usd = TechnicalIndicators.normalize_ohlc_to_usd(
        open_prices, high, low, close, instrument)
    
    # Step 2: Calculate ATR in USD terms
    atr_usd = TechnicalIndicators.calculate_atr_usd(high_usd, low_usd, close_usd, 14)
    
    # Initialize arrays
    si_raw = np.full(len(close), 0.0)
    si_capped = np.full(len(close), 0.0)
    limit_moves = np.full(len(close), np.nan)
    n_values = np.full(len(close), 0.0)
    r_values = np.full(len(close), 0.0)
    k_values = np.full(len(close), 0.0)
    
    for i in range(1, len(close)):
        # Current and previous values
        C2, O2, H2, L2 = close_usd[i], open_usd[i], high_usd[i], low_usd[i]
        C1, O1 = close_usd[i-1], open_usd[i-1]
        
        # Dynamic limit move
        if not np.isnan(atr_usd[i]) and atr_usd[i] > 0:
            L = 3.0 * atr_usd[i]
        else:
            recent_range = np.nanmean([abs(high_usd[j] - low_usd[j]) for j in range(max(0, i-10), i+1)])
            L = 3.0 * recent_range if not np.isnan(recent_range) else 1.0
        
        limit_moves[i] = L
        
        # Wilder's formulas
        N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
        
        term1 = abs(H2 - C1) - 0.5 * abs(L2 - C1) + 0.25 * abs(C1 - O1)
        term2 = abs(L2 - C1) - 0.5 * abs(H2 - C1) + 0.25 * abs(C1 - O1)
        term3 = (H2 - L2) + 0.25 * abs(C1 - O1)
        R = max(term1, term2, term3)
        
        if R <= 0:
            R = EPSILON
        
        K = max(abs(H2 - C1), abs(L2 - C1))
        
        # Store intermediate values
        n_values[i] = N
        r_values[i] = R
        k_values[i] = K
        
        # Calculate raw SI (before capping)
        if L > EPSILON:
            SI_raw = 100.0 * (N / R) * (K / L)
            si_raw[i] = SI_raw
            
            # Apply capping
            SI_capped = round(SI_raw)
            SI_capped = max(-100, min(100, SI_capped))
            si_capped[i] = SI_capped
    
    return {
        'si_raw': si_raw,
        'si_capped': si_capped, 
        'limit_moves': limit_moves,
        'n_values': n_values,
        'r_values': r_values,
        'k_values': k_values
    }

def analyze_capping_strategies(si_raw):
    """
    Analyze different capping strategies
    """
    strategies = {}
    
    # Strategy 1: Hard cap at ±100 (current)
    strategies['Hard Cap ±100'] = np.clip(si_raw, -100, 100)
    
    # Strategy 2: Soft cap using tanh
    strategies['Tanh Soft Cap'] = 100 * np.tanh(si_raw / 100)
    
    # Strategy 3: Sigmoid capping
    strategies['Sigmoid Cap'] = 200 / (1 + np.exp(-si_raw / 50)) - 100
    
    # Strategy 4: Percentile-based capping
    p99 = np.nanpercentile(np.abs(si_raw[si_raw != 0]), 99)
    strategies['Percentile Cap'] = np.clip(si_raw, -p99, p99)
    
    # Strategy 5: Standard deviation capping
    si_std = np.nanstd(si_raw[si_raw != 0])
    si_mean = np.nanmean(si_raw[si_raw != 0])
    cap_level = si_mean + 3 * si_std
    strategies['3-Sigma Cap'] = np.clip(si_raw, -cap_level, cap_level)
    
    return strategies

def create_si_analysis_chart():
    """
    Create comprehensive SI capping analysis chart
    """
    print("🚀 Analyzing SI capping strategies...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    # Calculate SI with analysis
    print("📊 Calculating SI values with detailed analysis...")
    results = calculate_si_with_analysis(
        df['open'].values, df['high'].values, df['low'].values, df['close'].values, 'EUR_USD'
    )
    
    si_raw = results['si_raw']
    si_capped = results['si_capped']
    
    # Analyze capping strategies
    strategies = analyze_capping_strategies(si_raw)
    
    # Filter out zero values for analysis
    non_zero_mask = si_raw != 0
    si_raw_nz = si_raw[non_zero_mask]
    
    # Create analysis chart
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('SI Capping Analysis - Raw vs Capped Values', fontsize=16, fontweight='bold')
    
    # 1. Raw SI histogram
    ax1 = axes[0, 0]
    ax1.hist(si_raw_nz, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(-100, color='red', linestyle='--', label='±100 cap lines')
    ax1.axvline(100, color='red', linestyle='--')
    ax1.set_title('Raw SI Values Distribution (Non-Zero)')
    ax1.set_xlabel('SI Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(0.02, 0.95, f'Count: {len(si_raw_nz)}\nMean: {np.mean(si_raw_nz):.2f}\nStd: {np.std(si_raw_nz):.2f}\nMin: {np.min(si_raw_nz):.2f}\nMax: {np.max(si_raw_nz):.2f}', 
             transform=ax1.transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Capped SI histogram
    ax2 = axes[0, 1]
    si_capped_nz = si_capped[non_zero_mask]
    ax2.hist(si_capped_nz, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Capped SI Values Distribution (±100)')
    ax2.set_xlabel('SI Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Count capped values
    capped_count = np.sum((si_raw_nz > 100) | (si_raw_nz < -100))
    ax2.text(0.02, 0.95, f'Capped: {capped_count}/{len(si_raw_nz)} ({capped_count/len(si_raw_nz)*100:.1f}%)', 
             transform=ax2.transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Time series comparison
    ax3 = axes[1, 0]
    last_200 = slice(-200, None)
    x_range = range(len(si_raw[last_200]))
    ax3.plot(x_range, si_raw[last_200], 'b-', alpha=0.7, label='Raw SI', linewidth=1)
    ax3.plot(x_range, si_capped[last_200], 'r-', alpha=0.9, label='Capped SI', linewidth=2)
    ax3.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(-100, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('SI Time Series (Last 200 bars)')
    ax3.set_xlabel('Bar Index')
    ax3.set_ylabel('SI Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot: Raw vs Capped
    ax4 = axes[1, 1]
    ax4.scatter(si_raw_nz, si_capped_nz, alpha=0.6, s=20)
    ax4.plot([-200, 200], [-200, 200], 'r--', alpha=0.8, label='No capping line')
    ax4.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(-100, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(100, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(-100, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Raw SI vs Capped SI')
    ax4.set_xlabel('Raw SI')
    ax4.set_ylabel('Capped SI')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Alternative capping strategies comparison
    ax5 = axes[2, 0]
    for name, values in strategies.items():
        if name != 'Hard Cap ±100':  # Skip current method
            values_nz = values[non_zero_mask]
            ax5.hist(values_nz, bins=30, alpha=0.5, label=name, density=True)
    ax5.set_title('Alternative Capping Strategies')
    ax5.set_xlabel('SI Value')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Capping impact summary
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Calculate statistics for all strategies
    stats_text = "Capping Strategy Comparison:\n\n"
    stats_text += f"{'Strategy':<15} {'Mean':<8} {'Std':<8} {'Range':<12} {'Capped%':<8}\n"
    stats_text += "-" * 60 + "\n"
    
    for name, values in strategies.items():
        values_nz = values[non_zero_mask]
        mean_val = np.mean(values_nz)
        std_val = np.std(values_nz)
        range_val = f"[{np.min(values_nz):.1f},{np.max(values_nz):.1f}]"
        
        # Calculate how many were "capped" (changed from original)
        if name == 'Hard Cap ±100':
            capped_pct = capped_count / len(si_raw_nz) * 100
        else:
            diff_count = np.sum(np.abs(values_nz - si_raw_nz) > 0.1)
            capped_pct = diff_count / len(si_raw_nz) * 100
        
        stats_text += f"{name:<15} {mean_val:<8.2f} {std_val:<8.2f} {range_val:<12} {capped_pct:<8.1f}%\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save chart
    save_path = project_root / "data/test/si_capping_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved to: {save_path}")
    
    # Print summary
    print(f"\n📊 SI Capping Analysis Summary:")
    print(f"Total non-zero SI values: {len(si_raw_nz)}")
    print(f"Raw SI range: [{np.min(si_raw_nz):.2f}, {np.max(si_raw_nz):.2f}]")
    print(f"Values exceeding ±100: {capped_count} ({capped_count/len(si_raw_nz)*100:.1f}%)")
    print(f"Raw SI mean: {np.mean(si_raw_nz):.2f}")
    print(f"Raw SI std: {np.std(si_raw_nz):.2f}")
    
    plt.show()
    print("🎉 SI capping analysis completed!")

def main():
    """Main function"""
    create_si_analysis_chart()

if __name__ == "__main__":
    main()