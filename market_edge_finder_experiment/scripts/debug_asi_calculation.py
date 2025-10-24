#!/usr/bin/env python3
"""
Debug ASI Calculation - Direct test to verify 65x multiplier and no capping
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import TechnicalIndicators

def main():
    print("🔧 Debug ASI Calculation...")
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print(f"📊 Testing direct ASI calculation on {len(df)} bars...")
    
    # Call the ASI calculation directly
    open_prices = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    print("🚀 Calling TechnicalIndicators.calculate_asi_grok_spec...")
    
    try:
        asi, atr_usd, si_values = TechnicalIndicators.calculate_asi_grok_spec(
            open_prices, high, low, close, 'EUR_USD'
        )
        
        print(f"✅ ASI calculation completed!")
        print(f"ASI range: [{np.nanmin(asi):.2f}, {np.nanmax(asi):.2f}]")
        
        # Analyze SI values (differences between consecutive ASI values)
        si_diff = np.diff(asi)
        si_nonzero = si_diff[si_diff != 0]
        
        print(f"\n📊 SI Values from ASI differences:")
        print(f"Non-zero SI count: {len(si_nonzero)}")
        print(f"SI range: [{np.min(si_nonzero):.2f}, {np.max(si_nonzero):.2f}]")
        
        # Check for values exceeding ±100
        exceeds_100 = np.sum(np.abs(si_nonzero) > 100)
        print(f"Values exceeding ±100: {exceeds_100}/{len(si_nonzero)} ({exceeds_100/len(si_nonzero)*100:.1f}%)")
        
        # Analyze the raw si_values returned by the function
        if si_values is not None:
            si_raw_nonzero = si_values[si_values != 0]
            print(f"\n📊 Raw SI Values from function:")
            print(f"Non-zero SI count: {len(si_raw_nonzero)}")
            if len(si_raw_nonzero) > 0:
                print(f"SI range: [{np.min(si_raw_nonzero):.2f}, {np.max(si_raw_nonzero):.2f}]")
                
                exceeds_100_raw = np.sum(np.abs(si_raw_nonzero) > 100)
                print(f"Raw values exceeding ±100: {exceeds_100_raw}/{len(si_raw_nonzero)} ({exceeds_100_raw/len(si_raw_nonzero)*100:.1f}%)")
                
                if exceeds_100_raw > 0:
                    print(f"🎉 SUCCESS: Found uncapped SI values!")
                    extremes = si_raw_nonzero[np.abs(si_raw_nonzero) > 100]
                    print(f"Extreme values: {extremes[:10]}")
                else:
                    print(f"❌ Raw SI values still capped")
            else:
                print("❌ No non-zero raw SI values found")
        else:
            print("❌ No raw SI values returned")
            
    except Exception as e:
        print(f"❌ Error in ASI calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()