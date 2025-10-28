#!/usr/bin/env python3
"""
Graph 200 bars of incremental processing with HSP/LSP markers and connector lines
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.incremental_indicators import MultiInstrumentState, update_indicators

def graph_incremental_200_with_connectors():
    """Generate 200-bar incremental graph with swing point connectors"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 200-bar slice as before
    test_slice = df.iloc[2000:2200].copy()
    print(f"🔍 Graphing incremental processing on 200 bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Initialize state
    multi_state = MultiInstrumentState()
    
    # Storage for results
    results = []
    hsp_points = []  # (bar_index, asi_value)
    lsp_points = []  # (bar_index, asi_value)
    
    # Process each bar incrementally
    prev_hsp_count = 0
    prev_lsp_count = 0
    
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        # Update indicators
        indicators, multi_state = update_indicators(new_ohlc, multi_state, 'EUR_USD')
        
        # Get current state
        state = multi_state.get_instrument_state('EUR_USD')
        
        # Check for new swing points (more efficient)
        current_hsp_count = len(state.hsp_indices)
        current_lsp_count = len(state.lsp_indices)
        
        # Check if new HSP was detected
        if current_hsp_count > prev_hsp_count:
            new_hsp_bar = state.hsp_indices[-1]  # Most recent HSP bar index
            new_hsp_asi = state.hsp_values[-1]   # Most recent HSP ASI value
            hsp_points.append((new_hsp_bar, new_hsp_asi))
            print(f"  Bar {i:3d}: NEW HSP detected at bar {new_hsp_bar}, ASI={new_hsp_asi:.1f}")
            prev_hsp_count = current_hsp_count
        
        # Check if new LSP was detected
        if current_lsp_count > prev_lsp_count:
            new_lsp_bar = state.lsp_indices[-1]  # Most recent LSP bar index
            new_lsp_asi = state.lsp_values[-1]   # Most recent LSP ASI value
            lsp_points.append((new_lsp_bar, new_lsp_asi))
            print(f"  Bar {i:3d}: NEW LSP detected at bar {new_lsp_bar}, ASI={new_lsp_asi:.1f}")
            prev_lsp_count = current_lsp_count
        
        # Store current ASI value
        current_asi = state.asi_history[-1] if state.asi_history else np.nan
        results.append({
            'bar': i,
            'timestamp': timestamp,
            'asi': current_asi,
            'volatility': indicators['volatility'],
            'direction': indicators['direction']
        })
        
        # Progress indicator every 50 bars
        if i % 50 == 0:
            print(f"  Processed {i+1}/200 bars...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 INCREMENTAL SWING POINTS FOUND:")
    print(f"Total HSP: {len(hsp_points)}")
    for bar, asi in hsp_points:
        print(f"  Bar {bar:3d}: HSP at ASI={asi:6.1f}")
    
    print(f"Total LSP: {len(lsp_points)}")
    for bar, asi in lsp_points:
        print(f"  Bar {bar:3d}: LSP at ASI={asi:6.1f}")
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top plot: ASI with swing points and connectors
    x = results_df['bar'].values
    asi_values = results_df['asi'].values
    
    # Plot ASI line
    ax1.plot(x, asi_values, 'b-', linewidth=1.5, alpha=0.7, label='ASI')
    
    # Plot HSP markers and connect them
    if len(hsp_points) >= 1:
        hsp_bars, hsp_asis = zip(*hsp_points)
        ax1.scatter(hsp_bars, hsp_asis, marker='^', s=200, c='red', 
                   edgecolors='white', linewidth=2, label=f'HSP ({len(hsp_points)})', zorder=10)
        
        # Connect HSP points with red dashed lines
        if len(hsp_points) >= 2:
            ax1.plot(hsp_bars, hsp_asis, 'r--', linewidth=2, alpha=0.8, 
                    label='HSP Trend', zorder=5)
            
            # Annotate HSP points
            for bar, asi in hsp_points:
                ax1.annotate(f'HSP\n{bar}', xy=(bar, asi), xytext=(bar, asi+20),
                           fontsize=9, ha='center', color='red', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Plot LSP markers and connect them
    if len(lsp_points) >= 1:
        lsp_bars, lsp_asis = zip(*lsp_points)
        ax1.scatter(lsp_bars, lsp_asis, marker='v', s=200, c='blue', 
                   edgecolors='white', linewidth=2, label=f'LSP ({len(lsp_points)})', zorder=10)
        
        # Connect LSP points with blue dashed lines
        if len(lsp_points) >= 2:
            ax1.plot(lsp_bars, lsp_asis, 'b--', linewidth=2, alpha=0.8, 
                    label='LSP Trend', zorder=5)
            
            # Annotate LSP points
            for bar, asi in lsp_points:
                ax1.annotate(f'LSP\n{bar}', xy=(bar, asi), xytext=(bar, asi-20),
                           fontsize=9, ha='center', color='blue', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax1.set_title('Incremental ASI Processing - 200 Bars with Swing Point Connectors', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('ASI Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Volatility and Direction indicators
    valid_vol = ~np.isnan(results_df['volatility'])
    valid_dir = ~np.isnan(results_df['direction'])
    
    ax2_twin = ax2.twinx()
    
    # Plot volatility
    ax2.plot(x[valid_vol], results_df['volatility'][valid_vol], 'g-', 
            linewidth=1.5, alpha=0.8, label='Volatility (ATR scaled)')
    ax2.set_ylabel('Volatility [0,1]', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Plot direction
    ax2_twin.plot(x[valid_dir], results_df['direction'][valid_dir], 'orange', 
                 linewidth=1.5, alpha=0.8, label='Direction (ADX scaled)')
    ax2_twin.set_ylabel('Direction [0,1]', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    ax2.set_xlabel('Bar Index')
    ax2.set_title('Incremental Volatility & Direction Indicators', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save the graph
    save_path = project_root / "data/test/incremental_200_bars_with_connectors.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n💾 Incremental 200-bar analysis with connectors saved to: {save_path}")
    print(f"📊 Summary:")
    print(f"  - Total bars processed: {len(results_df)}")
    print(f"  - HSP detected: {len(hsp_points)} (connected with red dashed lines)")
    print(f"  - LSP detected: {len(lsp_points)} (connected with blue dashed lines)")
    print(f"  - ASI coverage: {np.sum(~np.isnan(asi_values))} / {len(asi_values)} bars")
    print(f"  - Volatility coverage: {np.sum(valid_vol)} / {len(results_df)} bars")
    print(f"  - Direction coverage: {np.sum(valid_dir)} / {len(results_df)} bars")
    
    # Check alternation
    all_swings = []
    for bar, asi in hsp_points:
        all_swings.append((bar, 'HSP', asi))
    for bar, asi in lsp_points:
        all_swings.append((bar, 'LSP', asi))
    
    # Sort by bar index
    all_swings.sort(key=lambda x: x[0])
    
    if len(all_swings) > 1:
        violations = 0
        for i in range(1, len(all_swings)):
            if all_swings[i][1] == all_swings[i-1][1]:
                violations += 1
        
        alternation_rate = (len(all_swings) - violations) / len(all_swings) * 100
        print(f"  - Alternation rate: {alternation_rate:.1f}% ({violations} violations)")
    
    return results_df, hsp_points, lsp_points

if __name__ == "__main__":
    graph_incremental_200_with_connectors()