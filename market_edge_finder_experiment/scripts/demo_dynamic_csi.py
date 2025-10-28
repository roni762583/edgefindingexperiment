#!/usr/bin/env python3
"""
Dynamic CSI Demo Script

Demonstrates how to enable dynamic OANDA API fetching for CSI calculations
and compare the results with static configuration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from dotenv import load_dotenv
from configs.csi_config import set_dynamic_csi, is_dynamic_csi_enabled
from features.practical_incremental import PracticalIncrementalCalculator, PracticalInstrumentState

def demo_dynamic_csi():
    """Demonstrate dynamic CSI calculation"""
    load_dotenv()
    
    print("🚀 DYNAMIC CSI DEMONSTRATION")
    print("=" * 50)
    
    # Load some sample EUR_USD data
    csv_file = "data/raw/EUR_USD_3years_H1.csv"
    if not Path(csv_file).exists():
        print(f"❌ Sample data file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Take a small sample for testing
    sample_df = df.tail(50).copy()
    
    print(f"📊 Using {len(sample_df)} EUR_USD bars for CSI calculation")
    print()
    
    # Initialize calculator and state
    calculator = PracticalIncrementalCalculator()
    instrument = "EUR_USD"
    
    # Test both static and dynamic CSI
    for mode in ["Static", "Dynamic"]:
        print(f"🧮 {mode.upper()} CSI CALCULATION")
        print("-" * 30)
        
        # Configure CSI mode
        use_dynamic = (mode == "Dynamic")
        set_dynamic_csi(use_dynamic)
        
        print(f"Dynamic CSI enabled: {is_dynamic_csi_enabled()}")
        
        # Initialize fresh state
        state = PracticalInstrumentState()
        csi_values = []
        
        # Process bars to build up state
        for i, row in sample_df.iterrows():
            ohlc = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            
            # Add OHLC to history
            state.ohlc_history.append(ohlc)
            
            # Calculate indicators (simplified)
            if len(state.ohlc_history) >= 2:
                # Calculate ATR for CSI
                curr = ohlc
                prev = state.ohlc_history[-2]
                
                tr1 = curr['high'] - curr['low']
                tr2 = abs(curr['high'] - prev['close'])
                tr3 = abs(curr['low'] - prev['close'])
                true_range = max(tr1, tr2, tr3)
                
                # Simple ATR calculation for demo
                if len(state.atr_raw_history) == 0:
                    atr_raw = true_range
                else:
                    alpha = 2.0 / 15  # 14-period EMA
                    prev_atr = state.atr_raw_history[-1]
                    atr_raw = alpha * true_range + (1 - alpha) * prev_atr
                
                state.atr_raw_history.append(atr_raw)
                
                # Add some dummy ADX values for CSI calculation
                state.adx_history.append(25.0)  # Constant for demo
                
                # Calculate CSI if we have enough data
                if len(state.adx_history) >= 14 and len(state.atr_raw_history) > 0:
                    csi_value = calculator.calculate_csi_wilder_original(
                        ohlc, state, instrument, ohlc['close']
                    )
                    if not pd.isna(csi_value):
                        csi_values.append(csi_value)
        
        # Show results
        if csi_values:
            print(f"CSI values calculated: {len(csi_values)}")
            print(f"CSI mean: {sum(csi_values)/len(csi_values):.1f}")
            print(f"CSI range: [{min(csi_values):.1f}, {max(csi_values):.1f}]")
            print(f"Last 3 CSI values: {[f'{v:.1f}' for v in csi_values[-3:]]}")
        else:
            print("No CSI values calculated (insufficient data)")
        
        print()

def show_configuration_options():
    """Show how to configure dynamic CSI"""
    print("⚙️  DYNAMIC CSI CONFIGURATION OPTIONS")
    print("=" * 50)
    
    print("Environment Variables:")
    print("  CSI_USE_DYNAMIC_API=true     # Enable dynamic API fetching")
    print("  CSI_API_TIMEOUT=30           # API timeout in seconds")
    print("  CSI_CACHE_DURATION=300       # Cache duration in seconds")
    print()
    
    print("Programmatic Configuration:")
    print("  from configs.csi_config import set_dynamic_csi")
    print("  set_dynamic_csi(True)        # Enable dynamic fetching")
    print("  set_dynamic_csi(False)       # Use static configuration")
    print()
    
    print("Current Status:")
    print(f"  Dynamic CSI enabled: {is_dynamic_csi_enabled()}")
    print()

if __name__ == "__main__":
    show_configuration_options()
    demo_dynamic_csi()