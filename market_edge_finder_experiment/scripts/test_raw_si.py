#!/usr/bin/env python3
"""
Test Raw SI Values - Direct calculation to verify no capping
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.instruments import get_pip_value

def test_si_calculation():
    """Test SI calculation directly with a few sample bars"""
    
    # Load sample data
    data_path = project_root / "data/test/sample_data.csv"
    df = pd.read_csv(data_path)
    
    print("🔍 Testing Raw SI Calculation...")
    
    # Take first 20 bars for testing
    test_df = df.head(20).copy()
    
    # Get instrument parameters  
    pip_size, pip_value_usd = get_pip_value('EUR_USD')
    print(f"EUR_USD: pip_size={pip_size}, pip_value_usd=${pip_value_usd}")
    
    # USD normalization
    open_usd = (test_df['open'] / pip_size) * pip_value_usd
    high_usd = (test_df['high'] / pip_size) * pip_value_usd
    low_usd = (test_df['low'] / pip_size) * pip_value_usd
    close_usd = (test_df['close'] / pip_size) * pip_value_usd
    
    print(f"\nSample USD values:")
    print(f"Close[0]: ${close_usd.iloc[0]:.0f}")
    print(f"Close[1]: ${close_usd.iloc[1]:.0f}")
    
    # Calculate one SI value manually
    i = 5  # Test bar 5
    
    C2, O2, H2, L2 = close_usd.iloc[i], open_usd.iloc[i], high_usd.iloc[i], low_usd.iloc[i]
    C1, O1 = close_usd.iloc[i-1], open_usd.iloc[i-1]
    
    # Simple ATR estimate for L
    recent_ranges = []
    for j in range(max(0, i-5), i):
        tr = max(
            high_usd.iloc[j] - low_usd.iloc[j],
            abs(high_usd.iloc[j] - close_usd.iloc[j-1]) if j > 0 else 0,
            abs(low_usd.iloc[j] - close_usd.iloc[j-1]) if j > 0 else 0
        )
        recent_ranges.append(tr)
    
    atr_estimate = np.mean(recent_ranges) if recent_ranges else 50
    L = 3.0 * atr_estimate
    
    print(f"\nTest calculation for bar {i}:")
    print(f"C2=${C2:.0f}, C1=${C1:.0f}, O2=${O2:.0f}, O1=${O1:.0f}")
    print(f"H2=${H2:.0f}, L2=${L2:.0f}")
    print(f"L (limit move) = ${L:.0f}")
    
    # Wilder's formulas
    N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
    
    term1 = abs(H2 - C1) - 0.5 * abs(L2 - C1) + 0.25 * abs(C1 - O1)
    term2 = abs(L2 - C1) - 0.5 * abs(H2 - C1) + 0.25 * abs(C1 - O1)
    term3 = (H2 - L2) + 0.25 * abs(C1 - O1)
    R = max(term1, term2, term3)
    
    K = max(abs(H2 - C1), abs(L2 - C1))
    
    print(f"N = {N:.2f}")
    print(f"R = {R:.2f} (term1={term1:.2f}, term2={term2:.2f}, term3={term3:.2f})")
    print(f"K = {K:.2f}")
    
    # Calculate raw SI with different multipliers
    if R > 0 and L > 0:
        SI_100 = 100.0 * (N / R) * (K / L)
        SI_65 = 65.0 * (N / R) * (K / L)
        
        print(f"\nRaw SI calculations:")
        print(f"SI (100x) = {SI_100:.2f}")
        print(f"SI (65x)  = {SI_65:.2f}")
        print(f"SI (65x) rounded = {round(SI_65)}")
        
        # Check if would be capped
        if abs(SI_100) > 100:
            print(f"⚠️  SI_100 would be capped: {SI_100:.2f} -> ±100")
        if abs(SI_65) > 100:
            print(f"⚠️  SI_65 would be capped: {SI_65:.2f} -> ±100")
        else:
            print(f"✅ SI_65 within ±100 range")
    
    print("\n" + "="*50)
    
    # Test with extreme example
    print("Testing with extreme values...")
    
    # Create an extreme price move
    C1_extreme = 10000  # $10,000 
    C2_extreme = 10500  # $10,500 (large move)
    O1_extreme = 10000
    O2_extreme = 10100
    H2_extreme = 10600
    L2_extreme = 10050
    L_extreme = 100  # Small limit move
    
    N_extreme = (C2_extreme - C1_extreme) + 0.5 * (C2_extreme - O2_extreme) + 0.25 * (C1_extreme - O1_extreme)
    
    term1_extreme = abs(H2_extreme - C1_extreme) - 0.5 * abs(L2_extreme - C1_extreme) + 0.25 * abs(C1_extreme - O1_extreme)
    term2_extreme = abs(L2_extreme - C1_extreme) - 0.5 * abs(H2_extreme - C1_extreme) + 0.25 * abs(C1_extreme - O1_extreme)
    term3_extreme = (H2_extreme - L2_extreme) + 0.25 * abs(C1_extreme - O1_extreme)
    R_extreme = max(term1_extreme, term2_extreme, term3_extreme)
    
    K_extreme = max(abs(H2_extreme - C1_extreme), abs(L2_extreme - C1_extreme))
    
    SI_extreme_100 = 100.0 * (N_extreme / R_extreme) * (K_extreme / L_extreme)
    SI_extreme_65 = 65.0 * (N_extreme / R_extreme) * (K_extreme / L_extreme)
    
    print(f"Extreme case:")
    print(f"N={N_extreme:.0f}, R={R_extreme:.0f}, K={K_extreme:.0f}, L={L_extreme:.0f}")
    print(f"SI (100x) = {SI_extreme_100:.0f}")
    print(f"SI (65x)  = {SI_extreme_65:.0f}")
    
    print(f"\n🎯 This should show if 65x multiplier reduces extreme values!")

if __name__ == "__main__":
    test_si_calculation()