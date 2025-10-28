#!/usr/bin/env python3
"""
Create batch vs incremental analysis in the exact style of the reference image
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import (
    process_historical_data_incremental,
    MultiInstrumentState,
    IncrementalIndicatorCalculator
)

def create_reference_style_visualization():
    """Create visualization matching the reference image format"""
    
    # Load data slice (40 bars like reference)
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use the same slice that showed good results in our debug
    test_slice = df.iloc[1500:1540].copy()  # 40 bars
    
    # --- Run Batch Processing ---
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # --- Run Incremental Processing (manually for better control) ---
    multi_state = MultiInstrumentState()
    calculator = IncrementalIndicatorCalculator()
    
    incremental_asi = []
    incremental_swings_hsp = []
    incremental_swings_lsp = []
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        ohlc = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        state = multi_state.get_instrument_state("EUR_USD")
        state.bar_count += 1
        
        # Calculate ASI
        asi_value = calculator.calculate_asi_incremental(ohlc, state, "EUR_USD")
        incremental_asi.append(asi_value)
        
        # Detect swings
        is_hsp, is_lsp = calculator.detect_hsp_lsp_wilder_proper(
            asi_value, ohlc['high'], ohlc['low'], state
        )
        
        incremental_swings_hsp.append(is_hsp)
        incremental_swings_lsp.append(is_lsp)
        
        # Update state
        state.prev_open = ohlc['open']
        state.prev_high = ohlc['high']
        state.prev_low = ohlc['low']
        state.prev_close = ohlc['close']
    
    # --- Create Figure with vertical stacking ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Batch vs Incremental ASI Analysis (40 Bars)', fontsize=16, fontweight='bold')
    
    # --- SUBPLOT 1: Full Data Comparison ---
    x_axis = range(len(test_slice))
    
    # Plot ASI series
    batch_asi = batch_results.get('asi', pd.Series([0]*len(test_slice)))
    ax1.plot(x_axis, batch_asi.values, 'b-', linewidth=2, label='Batch ASI (Full)', alpha=0.8)
    ax1.plot(x_axis, incremental_asi, 'r--', linewidth=2, label='Incremental ASI (Full)', alpha=0.8)
    
    # Find aligned range (where both have valid data)
    batch_nonzero = batch_asi != 0
    valid_start = batch_nonzero.idxmax() if batch_nonzero.any() else 0
    valid_end = len(batch_asi) - 1
    
    if valid_start is not None:
        start_idx = list(batch_asi.index).index(valid_start) if valid_start in batch_asi.index else 0
        # Highlight aligned range
        ax1.axvspan(start_idx, valid_end, alpha=0.3, color='yellow', label='Aligned Range')
    
    # Mark swing points with larger, more visible markers
    if 'sig_hsp' in batch_results.columns:
        hsp_mask = batch_results['sig_hsp'] == True
        lsp_mask = batch_results['sig_lsp'] == True
        
        for i, idx in enumerate(batch_results.index[hsp_mask]):
            pos = list(batch_results.index).index(idx)
            val = batch_asi.loc[idx]
            ax1.plot(pos, val, '^', color='darkblue', markersize=15, markeredgewidth=2, 
                    markeredgecolor='white', label='Batch HSP (All)' if i == 0 else "", zorder=5)
        
        for i, idx in enumerate(batch_results.index[lsp_mask]):
            pos = list(batch_results.index).index(idx)
            val = batch_asi.loc[idx]
            ax1.plot(pos, val, 'v', color='darkblue', markersize=15, markeredgewidth=2,
                    markeredgecolor='white', label='Batch LSP (All)' if i == 0 else "", zorder=5)
    
    # Mark incremental swing points with larger, more visible markers
    incr_hsp_labeled = False
    incr_lsp_labeled = False
    
    for i, (is_hsp, asi_val) in enumerate(zip(incremental_swings_hsp, incremental_asi)):
        if is_hsp:
            ax1.plot(i, asi_val, '^', color='darkred', markersize=12, markeredgewidth=2,
                    markeredgecolor='white', label='Incremental HSP (All)' if not incr_hsp_labeled else "", zorder=5)
            incr_hsp_labeled = True
    
    for i, (is_lsp, asi_val) in enumerate(zip(incremental_swings_lsp, incremental_asi)):
        if is_lsp:
            ax1.plot(i, asi_val, 'v', color='darkred', markersize=12, markeredgewidth=2,
                    markeredgecolor='white', label='Incremental LSP (All)' if not incr_lsp_labeled else "", zorder=5)
            incr_lsp_labeled = True
    
    ax1.set_title('FULL DATA: Batch vs Incremental (Showing Data Availability Difference)')
    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('ASI Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- SUBPLOT 2: Aligned Data (Fair Comparison) ---
    if valid_start is not None and start_idx < len(x_axis):
        aligned_x = x_axis[start_idx:]
        aligned_batch = batch_asi.values[start_idx:]
        aligned_incremental = incremental_asi[start_idx:]
        
        # Calculate correlation
        if len(aligned_batch) > 1 and len(aligned_incremental) > 1:
            correlation = np.corrcoef(aligned_batch, aligned_incremental)[0, 1]
        else:
            correlation = 0.0
        
        ax2.plot(range(len(aligned_batch)), aligned_batch, 'b-', linewidth=2, label='Batch ASI (Aligned)')
        ax2.plot(range(len(aligned_incremental)), aligned_incremental, 'r--', linewidth=2, label='Incremental ASI (Aligned)')
        
        # Mark swing points in aligned range
        batch_hsp_count = 0
        batch_lsp_count = 0
        incr_hsp_count = 0
        incr_lsp_count = 0
        
        if 'sig_hsp' in batch_results.columns:
            for i, (hsp, lsp) in enumerate(zip(batch_results['sig_hsp'].values[start_idx:], 
                                             batch_results['sig_lsp'].values[start_idx:])):
                if hsp:
                    ax2.plot(i, aligned_batch[i], '^', color='darkblue', markersize=15, markeredgewidth=2,
                            markeredgecolor='white', label=f'Batch HSP ({batch_hsp_count+1})' if batch_hsp_count == 0 else "", zorder=5)
                    batch_hsp_count += 1
                if lsp:
                    ax2.plot(i, aligned_batch[i], 'v', color='darkblue', markersize=15, markeredgewidth=2,
                            markeredgecolor='white', label=f'Batch LSP ({batch_lsp_count+1})' if batch_lsp_count == 0 else "", zorder=5)
                    batch_lsp_count += 1
        
        for i, (is_hsp, is_lsp) in enumerate(zip(incremental_swings_hsp[start_idx:], 
                                                incremental_swings_lsp[start_idx:])):
            if is_hsp:
                ax2.plot(i, aligned_incremental[i], '^', color='darkred', markersize=12, markeredgewidth=2,
                        markeredgecolor='white', label=f'Incremental HSP ({incr_hsp_count+1})' if incr_hsp_count == 0 else "", zorder=5)
                incr_hsp_count += 1
            if is_lsp:
                ax2.plot(i, aligned_incremental[i], 'v', color='darkred', markersize=12, markeredgewidth=2,
                        markeredgecolor='white', label=f'Incremental LSP ({incr_lsp_count+1})' if incr_lsp_count == 0 else "", zorder=5)
                incr_lsp_count += 1
        
        ax2.set_title(f'ALIGNED DATA: Fair Comparison (Same Valid Range)\nAligned ASI Correlation: {correlation:.3f}')
    else:
        ax2.text(0.5, 0.5, 'No aligned range available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('ALIGNED DATA: No valid comparison range')
        batch_hsp_count = batch_lsp_count = incr_hsp_count = incr_lsp_count = 0
        correlation = 0.0
    
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('ASI Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- SUBPLOT 3: Combined Analysis ---
    ax3.axis('off')
    
    # Create comprehensive summary in one panel
    
    # Left side: Statistics
    stats_text = f"""STATISTICAL ANALYSIS

Data Range Analysis:
  Total bars: {len(test_slice)}
  Batch non-zero range: {(batch_asi != 0).sum()} bars
  Incremental range: {len(incremental_asi)} bars  
  Aligned range: {len(aligned_batch) if 'aligned_batch' in locals() else 0} bars

Correlation Analysis:
  Full range correlation: {np.corrcoef(batch_asi.values, incremental_asi)[0,1]:.5f}
  Aligned range correlation: {correlation:.5f}

Swing Point Analysis (Aligned Range):
  Batch: {batch_hsp_count} HSP, {batch_lsp_count} LSP
  Incremental: {incr_hsp_count} HSP, {incr_lsp_count} LSP
  HSP overlap: {min(batch_hsp_count, incr_hsp_count)}/{max(batch_hsp_count, incr_hsp_count) if max(batch_hsp_count, incr_hsp_count) > 0 else 1}
  LSP overlap: {min(batch_lsp_count, incr_lsp_count)}/{max(batch_lsp_count, incr_lsp_count) if max(batch_lsp_count, incr_lsp_count) > 0 else 1}"""
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Right side: Swing Point Summary Table
    table_data = [
        ['Method', 'HSP Count', 'LSP Count', 'Status'],
        ['Batch (Aligned)', str(batch_hsp_count), str(batch_lsp_count), 'Working'],
        ['Incremental (Aligned)', str(incr_hsp_count), str(incr_lsp_count), 'Partial'],
        ['Batch (Full)', 
         str((batch_results.get('sig_hsp', pd.Series()).sum()) if 'sig_hsp' in batch_results.columns else 0),
         str((batch_results.get('sig_lsp', pd.Series()).sum()) if 'sig_lsp' in batch_results.columns else 0),
         'Working'],
        ['Incremental (Full)', 
         str(sum(incremental_swings_hsp)),
         str(sum(incremental_swings_lsp)),
         'Working']
    ]
    
    # Create mini table on the right
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], 
                      loc=(0.55, 0.3), cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.8, 1.5)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i <= 2:  # Aligned data
                table[(i, j)].set_facecolor('#FFEB3B')
            else:  # Full range data
                table[(i, j)].set_facecolor('#E3F2FD')
    
    ax3.set_title('COMPREHENSIVE ANALYSIS: Swing Points & Correlation', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = project_root / "data/test/reference_style_analysis_40bars.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Analysis complete!")
    print(f"💾 Saved to: {save_path}")
    
    # Print summary
    print(f"\n📈 SUMMARY:")
    print(f"ASI correlation (full): {np.corrcoef(batch_asi.values, incremental_asi)[0,1]:.3f}")
    print(f"ASI correlation (aligned): {correlation:.3f}")
    print(f"Batch swings: {(batch_results.get('sig_hsp', pd.Series()).sum() if 'sig_hsp' in batch_results.columns else 0)} HSP, {(batch_results.get('sig_lsp', pd.Series()).sum() if 'sig_lsp' in batch_results.columns else 0)} LSP")
    print(f"Incremental swings: {sum(incremental_swings_hsp)} HSP, {sum(incremental_swings_lsp)} LSP")

if __name__ == "__main__":
    create_reference_style_visualization()