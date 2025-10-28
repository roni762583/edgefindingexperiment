#!/usr/bin/env python3
"""
Create 200-bar graph showing both Wilder and Simple swing detection methods
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.simple_swing_detection import detect_simple_swings_batch

def graph_both_methods():
    """Create comparison graph with both methods"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use 200-bar slice
    test_slice = df.iloc[2000:2200].copy()
    print(f"🔍 Creating comparison graph for both methods")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Get ASI and Wilder swings
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    asi_values = batch_results['asi'].values
    
    # Wilder method results
    hsp_wilder = np.where(batch_results['sig_hsp'])[0]
    lsp_wilder = np.where(batch_results['sig_lsp'])[0]
    
    # Simple method results
    hsp_flags, lsp_flags = detect_simple_swings_batch(asi_values, min_distance=1, use_exceeding_filter=True)
    hsp_simple = np.where(hsp_flags)[0]
    lsp_simple = np.where(lsp_flags)[0]
    
    print(f"\n📊 Results:")
    print(f"Wilder method:  {len(hsp_wilder)} HSP + {len(lsp_wilder)} LSP = {len(hsp_wilder)+len(lsp_wilder)} total")
    print(f"Simple method:  {len(hsp_simple)} HSP + {len(lsp_simple)} LSP = {len(hsp_simple)+len(lsp_simple)} total")
    
    # Create dual-panel visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    x = np.arange(len(asi_values))
    
    # Top panel: Wilder method
    ax1.plot(x, asi_values, 'b-', linewidth=1.5, alpha=0.8, label='ASI')
    
    # Wilder HSP markers and connectors
    if len(hsp_wilder) > 0:
        ax1.scatter(hsp_wilder, asi_values[hsp_wilder], marker='^', s=200, c='red', 
                   edgecolors='white', linewidth=2, label=f'Wilder HSP ({len(hsp_wilder)})', zorder=10)
        
        if len(hsp_wilder) >= 2:
            ax1.plot(hsp_wilder, asi_values[hsp_wilder], 'r--', linewidth=2, alpha=0.8, zorder=5)
            
        # Annotate Wilder HSP
        for idx in hsp_wilder:
            ax1.annotate(f'HSP\n{idx}', xy=(idx, asi_values[idx]), xytext=(idx, asi_values[idx]+20),
                        fontsize=9, ha='center', color='red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Wilder LSP markers and connectors
    if len(lsp_wilder) > 0:
        ax1.scatter(lsp_wilder, asi_values[lsp_wilder], marker='v', s=200, c='blue', 
                   edgecolors='white', linewidth=2, label=f'Wilder LSP ({len(lsp_wilder)})', zorder=10)
        
        if len(lsp_wilder) >= 2:
            ax1.plot(lsp_wilder, asi_values[lsp_wilder], 'b--', linewidth=2, alpha=0.8, zorder=5)
            
        # Annotate Wilder LSP
        for idx in lsp_wilder:
            ax1.annotate(f'LSP\n{idx}', xy=(idx, asi_values[idx]), xytext=(idx, asi_values[idx]-20),
                        fontsize=9, ha='center', color='blue', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax1.set_title('Wilder Method: Complex, Strict Alternation (1978 Specification)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ASI Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Simple method
    ax2.plot(x, asi_values, 'b-', linewidth=1.5, alpha=0.8, label='ASI')
    
    # Simple HSP markers and connectors
    if len(hsp_simple) > 0:
        ax2.scatter(hsp_simple, asi_values[hsp_simple], marker='^', s=150, c='darkgreen', 
                   edgecolors='white', linewidth=2, label=f'Simple HSP ({len(hsp_simple)})', zorder=10)
        
        if len(hsp_simple) >= 2:
            ax2.plot(hsp_simple, asi_values[hsp_simple], 'g--', linewidth=2, alpha=0.8, zorder=5)
    
    # Simple LSP markers and connectors
    if len(lsp_simple) > 0:
        ax2.scatter(lsp_simple, asi_values[lsp_simple], marker='v', s=150, c='purple', 
                   edgecolors='white', linewidth=2, label=f'Simple LSP ({len(lsp_simple)})', zorder=10)
        
        if len(lsp_simple) >= 2:
            ax2.plot(lsp_simple, asi_values[lsp_simple], 'm--', linewidth=2, alpha=0.8, zorder=5)
    
    ax2.set_title('Simple Method: Practical, Better Coverage (Reference Style)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('ASI Value')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add method comparison text
    wilder_last = max(max(hsp_wilder, default=0), max(lsp_wilder, default=0))
    simple_last = max(max(hsp_simple, default=0), max(lsp_simple, default=0))
    
    fig.suptitle(f'ASI Swing Detection Comparison - 200 Bars\n' +
                f'Wilder: {len(hsp_wilder)+len(lsp_wilder)} swings (last at bar {wilder_last}) | ' +
                f'Simple: {len(hsp_simple)+len(lsp_simple)} swings (last at bar {simple_last})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    # Save the graph
    save_path = project_root / "data/test/wilder_vs_simple_methods_200bars.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n💾 Comparison graph saved to: {save_path}")
    
    # Show coverage statistics
    print(f"\n📈 Coverage Analysis:")
    print(f"Wilder method: Last swing at bar {wilder_last} ({wilder_last/199*100:.1f}% coverage)")
    print(f"Simple method: Last swing at bar {simple_last} ({simple_last/199*100:.1f}% coverage)")
    
    # Method recommendations
    print(f"\n💡 Use Case Recommendations:")
    print(f"🎯 For ML training & trend analysis:")
    print(f"   → Use Simple method (better coverage, more training data)")
    print(f"📚 For Wilder 1978 specification compliance:")
    print(f"   → Use Wilder method (strict alternation, fewer but precise swings)")
    print(f"⚖️  Both methods available - choose based on your needs!")
    
    return save_path

if __name__ == "__main__":
    graph_both_methods()