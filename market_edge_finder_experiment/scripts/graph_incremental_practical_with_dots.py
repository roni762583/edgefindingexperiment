#!/usr/bin/env python3
"""
Regenerate graph using incremental practical method with ASI as medium dots
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.practical_incremental import update_practical_indicators, PracticalMultiInstrumentState

def create_practical_incremental_graph():
    """Create graph using incremental practical method with ASI dots"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 200-bar slice as before
    test_slice = df.iloc[2000:2200].copy()
    print(f"🎯 Creating Incremental Practical Graph: 200 bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Incremental processing with practical method
    multi_state = PracticalMultiInstrumentState()
    incremental_results = []
    asi_values = []
    close_prices = []
    raw_adx_values = []
    
    print(f"\n🔄 Running incremental practical processing...")
    for i, (timestamp, row) in enumerate(test_slice.iterrows()):
        new_ohlc = {
            'open': row['open'],
            'high': row['high'], 
            'low': row['low'],
            'close': row['close']
        }
        
        indicators, multi_state = update_practical_indicators(new_ohlc, multi_state, 'EUR_USD')
        incremental_results.append(indicators)
        
        # Get ASI value from state
        state = multi_state.get_instrument_state('EUR_USD')
        current_asi = state.asi_history[-1] if state.asi_history else 0.0
        asi_values.append(current_asi)
        
        # Store close price
        close_prices.append(row['close'])
        
        # Get raw ADX value for comparison
        raw_adx = state.adx_history[-1] if state.adx_history else np.nan
        raw_adx_values.append(raw_adx)
        
        if i % 50 == 0:
            print(f"  Processed {i+1}/200 bars...")
    
    # Get swing points from state
    state = multi_state.get_instrument_state('EUR_USD')
    
    print(f"\n📊 Incremental Practical Results:")
    hsp_count = len(state.hsp_indices)
    lsp_count = len(state.lsp_indices)
    print(f"HSP: {hsp_count}, LSP: {lsp_count}, Total: {hsp_count + lsp_count}")
    
    # Create the visualization - 4 panels
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 20))
    
    x = np.arange(len(asi_values))
    
    # Top plot: Close Price
    ax1.plot(x, close_prices, 'k-', linewidth=1.5, alpha=0.8, label='Close Price')
    ax1.set_title('Close Price - EUR/USD', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: ASI with swing points and connectors
    # Plot ASI as medium dots (not line)
    ax2.scatter(x, asi_values, c='blue', s=8, alpha=0.6, label='ASI (dots)', zorder=3)
    
    # Plot HSP markers and connect them
    if len(state.hsp_indices) >= 1:
        hsp_bars = state.hsp_indices
        hsp_asis = [state.asi_history[idx] if idx < len(state.asi_history) else state.hsp_values[i] 
                   for i, idx in enumerate(hsp_bars)]
        
        ax2.scatter(hsp_bars, hsp_asis, marker='^', s=200, c='red', 
                   edgecolors='white', linewidth=2, label=f'HSP ({len(hsp_bars)})', zorder=10)
        
        # Connect HSP points with red dashed lines
        if len(hsp_bars) >= 2:
            ax2.plot(hsp_bars, hsp_asis, 'r--', linewidth=2, alpha=0.8, 
                    label='HSP Trend', zorder=5)
            
            # Annotate HSP points
            for bar, asi in zip(hsp_bars, hsp_asis):
                ax2.annotate(f'HSP\n{bar}', xy=(bar, asi), xytext=(bar, asi+20),
                           fontsize=9, ha='center', color='red', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # Plot LSP markers and connect them
    if len(state.lsp_indices) >= 1:
        lsp_bars = state.lsp_indices
        lsp_asis = [state.asi_history[idx] if idx < len(state.asi_history) else state.lsp_values[i] 
                   for i, idx in enumerate(lsp_bars)]
        
        ax2.scatter(lsp_bars, lsp_asis, marker='v', s=200, c='blue', 
                   edgecolors='white', linewidth=2, label=f'LSP ({len(lsp_bars)})', zorder=10)
        
        # Connect LSP points with blue dashed lines
        if len(lsp_bars) >= 2:
            ax2.plot(lsp_bars, lsp_asis, 'b--', linewidth=2, alpha=0.8, 
                    label='LSP Trend', zorder=5)
            
            # Annotate LSP points
            for bar, asi in zip(lsp_bars, lsp_asis):
                ax2.annotate(f'LSP\n{bar}', xy=(bar, asi), xytext=(bar, asi-20),
                           fontsize=9, ha='center', color='blue', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax2.set_title('Incremental Practical Method - ASI as Dots with Swing Point Connectors', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('ASI Value')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: Volatility, Direction, and Slope Angles
    incremental_df = pd.DataFrame(incremental_results)
    
    valid_vol = ~incremental_df['volatility'].isna()
    valid_dir = ~incremental_df['direction'].isna()
    valid_slope_high = ~incremental_df['slope_high'].isna()
    valid_slope_low = ~incremental_df['slope_low'].isna()
    
    ax3_twin = ax3.twinx()
    
    # Plot volatility (ATR scaled)
    if valid_vol.sum() > 0:
        ax3.plot(x[valid_vol], incremental_df['volatility'][valid_vol], 'g-', 
                linewidth=1.5, alpha=0.8, label='Volatility (ATR scaled)')
        ax3.set_ylabel('Volatility [0,1]', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
    
    # Plot direction (ADX scaled)
    if valid_dir.sum() > 0:
        ax3_twin.plot(x[valid_dir], incremental_df['direction'][valid_dir], 'orange', 
                     linewidth=1.5, alpha=0.8, label='Direction (ADX scaled)')
        ax3_twin.set_ylabel('Direction [0,1]', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    # Plot raw slopes on secondary axis
    ax3_slopes = ax3.twinx()
    ax3_slopes.spines['right'].set_position(('outward', 60))
    
    if valid_slope_high.sum() > 0:
        ax3_slopes.plot(x[valid_slope_high], incremental_df['slope_high'][valid_slope_high], 'r-', 
                       linewidth=1.2, alpha=0.7, label='High Slope')
    
    if valid_slope_low.sum() > 0:
        ax3_slopes.plot(x[valid_slope_low], incremental_df['slope_low'][valid_slope_low], 'b-', 
                       linewidth=1.2, alpha=0.7, label='Low Slope')
    
    ax3_slopes.set_ylabel('Raw Slopes (ASI/bar)', color='red')
    ax3_slopes.tick_params(axis='y', labelcolor='red')
    # Auto-scale for raw slope values
    if valid_slope_high.sum() > 0 or valid_slope_low.sum() > 0:
        all_slopes = []
        if valid_slope_high.sum() > 0:
            all_slopes.extend(incremental_df['slope_high'][valid_slope_high].dropna().values)
        if valid_slope_low.sum() > 0:
            all_slopes.extend(incremental_df['slope_low'][valid_slope_low].dropna().values)
        if all_slopes:
            slope_min, slope_max = np.min(all_slopes), np.max(all_slopes)
            slope_range = slope_max - slope_min
            ax3_slopes.set_ylim(slope_min - 0.1*slope_range, slope_max + 0.1*slope_range)
    
    ax3.set_xlabel('Bar Index')
    ax3.set_title('Incremental Practical Method - Volatility, Direction & Slope Indicators', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add legends
    if valid_vol.sum() > 0:
        ax3.legend(loc='upper left')
    if valid_dir.sum() > 0:
        ax3_twin.legend(loc='upper center')
    if valid_slope_high.sum() > 0 or valid_slope_low.sum() > 0:
        ax3_slopes.legend(loc='upper right')
    
    # Fourth plot: Log Close Feature (Price Change)
    valid_price_change = ~incremental_df['price_change'].isna()
    
    if valid_price_change.sum() > 0:
        ax4.plot(x[valid_price_change], incremental_df['price_change'][valid_price_change], 'purple', 
                linewidth=1.5, alpha=0.8, label='Price Change (Scaled [0,1])')
        ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50th Percentile')
        ax4.set_ylabel('Price Change [0,1]', color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4.set_ylim(0, 1)  # Fixed [0,1] range for percentile scaled data
    
    ax4.set_xlabel('Bar Index')
    ax4.set_title('Price Change Feature (Percentile Scaled [0,1])', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save the graph
    save_path = project_root / "data/test/incremental_practical_4panel_complete.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n💾 4-panel complete incremental practical graph saved to: {save_path}")
    print(f"📊 Summary:")
    print(f"  - Method: Incremental Practical (min_distance=3)")
    print(f"  - Panel 1: Close price trace")
    print(f"  - Panel 2: ASI dots with swing point connectors")
    print(f"  - Panel 3: Volatility, direction, and slopes")
    print(f"  - Panel 4: Log close feature (price change)")
    print(f"  - Total bars processed: 200")
    print(f"  - HSP detected: {len(state.hsp_indices)} (red triangles)")
    print(f"  - LSP detected: {len(state.lsp_indices)} (blue triangles)")
    print(f"  - Coverage: {max(max(state.hsp_indices, default=0), max(state.lsp_indices, default=0))/199*100:.1f}%")
    
    # Show some swing points
    all_swings = []
    for i, idx in enumerate(state.hsp_indices):
        asi_val = state.asi_history[idx] if idx < len(state.asi_history) else state.hsp_values[i]
        all_swings.append((idx, 'HSP', asi_val))
    for i, idx in enumerate(state.lsp_indices):
        asi_val = state.asi_history[idx] if idx < len(state.asi_history) else state.lsp_values[i]
        all_swings.append((idx, 'LSP', asi_val))
    
    all_swings.sort()
    
    print(f"\nFirst 10 swings:")
    for bar, type_, asi in all_swings[:10]:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\nLast 10 swings:")
    for bar, type_, asi in all_swings[-10:]:
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    return save_path, len(all_swings)

if __name__ == "__main__":
    create_practical_incremental_graph()