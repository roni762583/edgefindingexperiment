#!/usr/bin/env python3
"""
Test USD pairs (EUR_USD, USD_JPY) to show dynamic pip value calculation working
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.instruments import extract_instrument_from_filename, calculate_pip_value_usd, get_pip_size

def test_usd_pairs_pip_calculation():
    """Test dynamic pip calculation with USD pairs"""
    
    print("🧪 TESTING: Dynamic Pip Values for USD Pairs")
    print("=" * 60)
    
    # Test EUR_USD (quote currency = USD)
    print("\n📊 Testing EUR_USD (Quote = USD):")
    eur_usd_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    if eur_usd_path.exists():
        df = pd.read_csv(eur_usd_path).head(20)
        instrument = extract_instrument_from_filename(str(eur_usd_path))
        
        for i, row in df.iterrows():
            rate = row['close']
            pip_value = calculate_pip_value_usd(instrument, rate)
            if i < 5:  # Show first 5
                print(f"  Rate: {rate:.5f}, Pip Value: ${pip_value:.2f}")
        
        pip_values = [calculate_pip_value_usd(instrument, row['close']) for _, row in df.iterrows()]
        print(f"  Range: ${min(pip_values):.2f} - ${max(pip_values):.2f}")
    
    # Test USD_JPY (base currency = USD)
    print("\n📊 Testing USD_JPY (Base = USD):")
    usd_jpy_path = project_root / "data/raw/USD_JPY_3years_H1.csv"
    if usd_jpy_path.exists():
        df = pd.read_csv(usd_jpy_path).head(20)
        instrument = extract_instrument_from_filename(str(usd_jpy_path))
        
        for i, row in df.iterrows():
            rate = row['close']
            pip_value = calculate_pip_value_usd(instrument, rate)
            if i < 5:  # Show first 5
                print(f"  Rate: {rate:.3f}, Pip Value: ${pip_value:.2f}")
        
        pip_values = [calculate_pip_value_usd(instrument, row['close']) for _, row in df.iterrows()]
        print(f"  Range: ${min(pip_values):.2f} - ${max(pip_values):.2f}")
    
    # Test AUD_USD (quote currency = USD)
    print("\n📊 Testing AUD_USD (Quote = USD):")
    aud_usd_path = project_root / "data/raw/AUD_USD_3years_H1.csv"
    if aud_usd_path.exists():
        df = pd.read_csv(aud_usd_path).head(20)
        instrument = extract_instrument_from_filename(str(aud_usd_path))
        
        for i, row in df.iterrows():
            rate = row['close']
            pip_value = calculate_pip_value_usd(instrument, rate)
            if i < 5:  # Show first 5
                print(f"  Rate: {rate:.5f}, Pip Value: ${pip_value:.2f}")
        
        pip_values = [calculate_pip_value_usd(instrument, row['close']) for _, row in df.iterrows()]
        print(f"  Range: ${min(pip_values):.2f} - ${max(pip_values):.2f}")
    
    print("\n💡 For cross-currency pairs like AUD_CHF:")
    print("   Need USD rates (AUD_USD, CHF_USD) for accurate pip value calculation")
    print("   Currently falls back to $10.00 (simplified calculation)")
    print("   In production, feed multiple USD rates to calculate_pip_value_usd()")

if __name__ == "__main__":
    test_usd_pairs_pip_calculation()