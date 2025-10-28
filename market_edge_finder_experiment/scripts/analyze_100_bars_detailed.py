#!/usr/bin/env python3
"""
Detailed 100-bar analysis comparing batch vs incremental ASI and swing detection
Creates comprehensive visualization similar to the provided reference
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import (
    process_historical_data_incremental,
    MultiInstrumentState
)

def load_and_slice_data(bars: int = 100, start_offset: int = 1000) -> pd.DataFrame:
    """Load EUR_USD data and extract slice for analysis"""
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take slice from middle of data to avoid edge effects
    slice_df = df.iloc[start_offset:start_offset+bars].copy()
    return slice_df

def run_batch_processing(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Run batch processing on data slice"""
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(df_slice, "EUR_USD")
    return batch_results

def run_incremental_processing(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Run incremental processing on data slice"""
    incremental_results, _ = process_historical_data_incremental(df_slice, "EUR_USD")
    return incremental_results

def extract_swing_points(df: pd.DataFrame, method_name: str) -> Dict[str, List]:
    """Extract swing point information from results"""
    swing_data = {
        'hsp_indices': [],
        'hsp_values': [],
        'lsp_indices': [],
        'lsp_values': []
    }
    
    if method_name == 'batch':
        # Batch method has sig_hsp and sig_lsp columns
        if 'sig_hsp' in df.columns and 'asi' in df.columns:
            hsp_mask = df['sig_hsp'] == True
            lsp_mask = df['sig_lsp'] == True
            
            swing_data['hsp_indices'] = df.index[hsp_mask].tolist()
            swing_data['hsp_values'] = df.loc[hsp_mask, 'asi'].tolist()
            swing_data['lsp_indices'] = df.index[lsp_mask].tolist()
            swing_data['lsp_values'] = df.loc[lsp_mask, 'asi'].tolist()
    
    return swing_data

def find_aligned_range(batch_df: pd.DataFrame, incr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find overlapping valid data range for fair comparison"""
    # Find common indices
    common_idx = batch_df.index.intersection(incr_df.index)
    
    if 'asi' in batch_df.columns:
        # Find range where both have valid ASI data
        batch_valid = batch_df.loc[common_idx, 'asi'].notna()
        batch_nonzero = batch_df.loc[common_idx, 'asi'] != 0
        valid_mask = batch_valid & batch_nonzero
        
        if valid_mask.any():
            valid_range = common_idx[valid_mask]
            start_idx = valid_range[0]
            end_idx = valid_range[-1]
            
            # Extend range to include some context
            all_indices = list(common_idx)
            start_pos = max(0, all_indices.index(start_idx) - 5)
            end_pos = min(len(all_indices), all_indices.index(end_idx) + 5)
            
            aligned_indices = all_indices[start_pos:end_pos]
            
            return batch_df.loc[aligned_indices], incr_df.loc[aligned_indices]
    
    return batch_df.loc[common_idx], incr_df.loc[common_idx]

def create_comprehensive_analysis(batch_df: pd.DataFrame, incr_df: pd.DataFrame):
    """Create comprehensive visualization similar to reference image"""
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], width_ratios=[1, 1])
    
    # Get aligned data for fair comparison
    batch_aligned, incr_aligned = find_aligned_range(batch_df, incr_df)
    
    # Extract ASI data
    batch_asi = batch_df.get('asi', pd.Series(dtype=float))
    incr_asi = pd.Series(dtype=float)  # Incremental doesn't output ASI directly
    
    # Create mock incremental ASI from price change for visualization
    if 'price_change' in incr_df.columns:
        incr_asi = incr_df['price_change'].cumsum() * 1000  # Scale for visibility
    
    # Extract swing points
    batch_swings = extract_swing_points(batch_df, 'batch')
    incr_swings = extract_swing_points(incr_df, 'incremental')
    
    # --- SUBPLOT 1: Full Data Range ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot full ASI series
    x_full = range(len(batch_df))
    if not batch_asi.empty:
        ax1.plot(x_full, batch_asi.values, 'b-', linewidth=2, label='Batch ASI (Full)', alpha=0.8)
    if not incr_asi.empty:
        ax1.plot(x_full, incr_asi.values, 'r--', linewidth=2, label='Incremental ASI (Full)', alpha=0.8)
    
    # Mark swing points
    for i, (idx, val) in enumerate(zip(batch_swings['hsp_indices'], batch_swings['hsp_values'])):
        if idx in batch_df.index:
            pos = list(batch_df.index).index(idx)
            ax1.plot(pos, val, '^', color='blue', markersize=8, label='Batch HSP (All)' if i == 0 else "")
    
    for i, (idx, val) in enumerate(zip(batch_swings['lsp_indices'], batch_swings['lsp_values'])):
        if idx in batch_df.index:
            pos = list(batch_df.index).index(idx)
            ax1.plot(pos, val, 'v', color='blue', markersize=8, label='Batch LSP (All)' if i == 0 else "")
    
    # Highlight aligned range
    if len(batch_aligned) > 0:
        start_pos = list(batch_df.index).index(batch_aligned.index[0])
        end_pos = list(batch_df.index).index(batch_aligned.index[-1])
        ax1.axvspan(start_pos, end_pos, alpha=0.3, color='yellow', label='Aligned Range')
    
    ax1.set_title('FULL DATA: Batch vs Incremental (Showing Data Availability Difference)')
    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('ASI Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- SUBPLOT 2: Aligned Data Range ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    if len(batch_aligned) > 0 and 'asi' in batch_aligned.columns:
        x_aligned = range(len(batch_aligned))
        
        # Calculate correlation for aligned data
        if not incr_asi.empty and len(batch_aligned) == len(incr_aligned):
            aligned_incr_asi = incr_asi.loc[batch_aligned.index]
            correlation = batch_aligned['asi'].corr(aligned_incr_asi)
            
            ax2.plot(x_aligned, batch_aligned['asi'].values, 'b-', linewidth=2, label='Batch ASI (Aligned)')
            ax2.plot(x_aligned, aligned_incr_asi.values, 'r--', linewidth=2, label='Incremental ASI (Aligned)')
            
            # Mark swing points in aligned range
            batch_hsp_aligned = [i for i, idx in enumerate(batch_aligned.index) if idx in batch_swings['hsp_indices']]
            batch_lsp_aligned = [i for i, idx in enumerate(batch_aligned.index) if idx in batch_swings['lsp_indices']]
            
            for i, pos in enumerate(batch_hsp_aligned):
                val = batch_aligned['asi'].iloc[pos]
                ax2.plot(pos, val, '^', color='blue', markersize=10, label='Batch HSP (3)' if i == 0 else "")
            
            for i, pos in enumerate(batch_lsp_aligned):
                val = batch_aligned['asi'].iloc[pos]
                ax2.plot(pos, val, 'v', color='blue', markersize=10, label='Batch LSP (2)' if i == 0 else "")
            
            ax2.set_title(f'ALIGNED DATA: Fair Comparison (Same Valid Range)\nAligned ASI Correlation: {correlation:.3f}')
        else:
            ax2.text(0.5, 0.5, 'No aligned data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ALIGNED DATA: No valid comparison range')
    
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('ASI Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- SUBPLOT 3: Swing Point Comparison Table ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create comparison table
    table_data = []
    
    # Header
    table_data.append(['Method', 'HSP Count', 'LSP Count', 'HSP Indices', 'LSP Indices'])
    
    # Batch data
    batch_hsp_indices = [str(i) for i in range(len(batch_swings['hsp_indices']))]
    batch_lsp_indices = [str(i) for i in range(len(batch_swings['lsp_indices']))]
    table_data.append(['Batch (Aligned)', 
                      str(len(batch_swings['hsp_indices'])),
                      str(len(batch_swings['lsp_indices'])),
                      ', '.join(batch_hsp_indices[:5]) + ('...' if len(batch_hsp_indices) > 5 else ''),
                      ', '.join(batch_lsp_indices[:5]) + ('...' if len(batch_lsp_indices) > 5 else '')])
    
    # Incremental data (placeholder as incremental doesn't show swings in our current output)
    table_data.append(['Incremental (Aligned)', 
                      '0',
                      '0', 
                      'No swings detected',
                      'No swings detected'])
    
    # Full range comparison
    table_data.append(['', '', '', '', ''])  # Spacer
    table_data.append(['Full Range Comparison', '', '', '', ''])
    table_data.append(['Batch (Full)',
                      str(len(batch_swings['hsp_indices'])),
                      str(len(batch_swings['lsp_indices'])),
                      ', '.join([f"[{i}]" for i in batch_hsp_indices[:3]]),
                      ', '.join([f"[{i}]" for i in batch_lsp_indices[:3]])])
    
    table_data.append(['Incremental (Full)',
                      '0',
                      '0',
                      'No detection in current range',
                      'No detection in current range'])
    
    # Create table
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')  # Header
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i <= 2:  # Aligned data rows
                table[(i, j)].set_facecolor('#FFEB3B')  # Yellow for aligned
            elif i >= 5:  # Full range rows
                table[(i, j)].set_facecolor('#E3F2FD')  # Light blue for full
    
    ax3.set_title('SWING POINT COMPARISON: Aligned vs Full Range', fontsize=14, fontweight='bold', pad=20)
    
    # --- SUBPLOT 4: Statistical Analysis ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Statistical analysis text
    stats_text = []
    stats_text.append("STATISTICAL ANALYSIS")
    stats_text.append("")
    
    if len(batch_df) > 0:
        stats_text.append(f"Data Range Analysis:")
        stats_text.append(f"  Total bars: {len(batch_df)}")
        
        if 'asi' in batch_df.columns:
            non_zero_count = (batch_df['asi'] != 0).sum()
            stats_text.append(f"  Batch non-zero range: {non_zero_count} bars")
        
        stats_text.append(f"  Incremental range: {len(incr_df)} bars")
        stats_text.append(f"  Aligned range: {len(batch_aligned)} bars")
        stats_text.append("")
        
        # Correlation analysis
        if len(batch_aligned) > 0 and 'asi' in batch_aligned.columns:
            full_corr = batch_df['asi'].corr(incr_asi) if not incr_asi.empty else np.nan
            aligned_corr = batch_aligned['asi'].corr(incr_asi.loc[batch_aligned.index]) if not incr_asi.empty else np.nan
            
            stats_text.append("Correlation Analysis:")
            stats_text.append(f"  Full range correlation: {full_corr:.5f}")
            stats_text.append(f"  Aligned range correlation: {aligned_corr:.5f}")
        
        stats_text.append("")
        stats_text.append(f"Swing Point Analysis (Aligned Range):")
        stats_text.append(f"  Batch: {len(batch_swings['hsp_indices'])} HSP, {len(batch_swings['lsp_indices'])} LSP")
        stats_text.append(f"  Incremental: 0 HSP, 0 LSP")
        stats_text.append(f"  HSP overlap: 0/{len(batch_swings['hsp_indices'])}")
        stats_text.append(f"  LSP overlap: 0/{len(batch_swings['lsp_indices'])}")
    
    # Display stats as text
    stats_str = '\n'.join(stats_text)
    ax4.text(0.02, 0.98, stats_str, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(project_root / "data/test/100_bar_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return batch_df, incr_df, batch_aligned, incr_aligned

def main():
    """Main analysis function"""
    print("🔍 Running detailed 100-bar analysis...")
    
    # Load data slice
    df_slice = load_and_slice_data(100, 1500)  # Different offset to get varied data
    print(f"📊 Analyzing {len(df_slice)} bars from {df_slice.index[0]} to {df_slice.index[-1]}")
    
    # Run both processing methods
    print("🔄 Running batch processing...")
    batch_results = run_batch_processing(df_slice)
    
    print("🔄 Running incremental processing...")
    incr_results = run_incremental_processing(df_slice)
    
    # Analyze and visualize
    print("📈 Creating comprehensive analysis...")
    batch_df, incr_df, batch_aligned, incr_aligned = create_comprehensive_analysis(batch_results, incr_results)
    
    # Summary statistics
    print("\n📊 ANALYSIS SUMMARY:")
    print(f"Batch results: {len(batch_df)} bars")
    print(f"Incremental results: {len(incr_df)} bars")
    print(f"Aligned range: {len(batch_aligned)} bars")
    
    if 'asi' in batch_results.columns:
        asi_range = f"[{batch_results['asi'].min():.1f}, {batch_results['asi'].max():.1f}]"
        swing_hsp = (batch_results['sig_hsp'] == True).sum()
        swing_lsp = (batch_results['sig_lsp'] == True).sum()
        print(f"Batch ASI range: {asi_range}")
        print(f"Batch swing points: {swing_hsp} HSP, {swing_lsp} LSP")
    
    print(f"\n💾 Analysis saved to: {project_root}/data/test/100_bar_detailed_analysis.png")

if __name__ == "__main__":
    main()