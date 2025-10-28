#!/usr/bin/env python3
"""
Graph 200 points with swing point markers
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import process_historical_data_incremental

def graph_200_points():
    """Graph 200 data points with all swing markers visible"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take 200 bars from a different section to get more swing activity
    test_slice = df.iloc[2000:2200].copy()  # 200 bars from different section
    
    print(f"🔍 Graphing {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # --- Run batch processing ---
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # --- Count and locate all swing points ---
    swing_points = []
    if 'sig_hsp' in batch_results.columns and 'sig_lsp' in batch_results.columns:
        for i, (hsp, lsp, asi) in enumerate(zip(batch_results['sig_hsp'], batch_results['sig_lsp'], batch_results['asi'])):
            if hsp:
                swing_points.append((i, 'HSP', asi, 'red', '^'))
            if lsp:
                swing_points.append((i, 'LSP', asi, 'blue', 'v'))
    
    print(f"\n📊 SWING POINTS FOUND:")
    print(f"Total swing points: {len(swing_points)}")
    for bar, type_, asi, color, marker in swing_points:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    # --- Create visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Plot ASI line
    x_axis = range(len(test_slice))
    asi_values = batch_results['asi'].values
    ax.plot(x_axis, asi_values, 'b-', linewidth=2, label='Batch ASI', alpha=0.8)
    
    # Plot ALL swing point markers with giant size
    marker_size = 300
    
    for bar, type_, asi, color, marker_shape in swing_points:
        ax.scatter(bar, asi, marker=marker_shape, s=marker_size, c=color, 
                  edgecolors='white', linewidth=3, zorder=10,
                  label=f'{type_} Points' if bar == swing_points[0][0] else "")
        
        # Add text annotation for each marker
        ax.annotate(f'{type_}\n{bar}', xy=(bar, asi), 
                   xytext=(bar, asi + (30 if type_ == 'HSP' else -30)),
                   fontsize=8, ha='center', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Customize the plot
    ax.set_title(f'200-Bar ASI Analysis with ALL Swing Points Marked\nTotal Swing Points: {len(swing_points)}', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('ASI Value', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""📊 SWING POINT STATISTICS
    
🎯 Detection Results:
  • Total swing points: {len(swing_points)}
  • HSP count: {sum(1 for sp in swing_points if sp[1] == 'HSP')}
  • LSP count: {sum(1 for sp in swing_points if sp[1] == 'LSP')}
  • ASI range: [{asi_values.min():.1f}, {asi_values.max():.1f}]
  
📅 Data Info:
  • Bars analyzed: {len(test_slice)}
  • Period: {test_slice.index[0].strftime('%Y-%m-%d')} to {test_slice.index[-1].strftime('%Y-%m-%d')}
  • Swing frequency: {len(swing_points)/len(test_slice)*100:.1f}% of bars
    
✅ Status: ALL MARKERS VISIBLE"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    plt.tight_layout()
    
    # Save with high resolution
    save_path = project_root / "data/test/200_points_all_markers.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n💾 200-point analysis saved to: {save_path}")
    print(f"🎉 All {len(swing_points)} swing point markers should be visible!")
    
    return batch_results, swing_points

if __name__ == "__main__":
    graph_200_points()