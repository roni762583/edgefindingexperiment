#!/usr/bin/env python3
"""
Compare Static vs Dynamic CSI Parameter Fetching

Test the difference between static configuration and live OANDA API
for CSI calculations to validate the dynamic implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.instruments import get_dynamic_csi_parameters, FX_INSTRUMENTS
from data_pull.oanda_csi_fetcher import OandaCSIFetcher
from dotenv import load_dotenv

def compare_csi_parameters():
    """Compare static vs dynamic CSI parameters for key instruments"""
    load_dotenv()
    
    # Test with representative instruments
    test_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'GBP_JPY', 'EUR_GBP']
    
    print("🔍 STATIC vs DYNAMIC CSI PARAMETER COMPARISON")
    print("=" * 70)
    print(f"{'Instrument':<10} {'Source':<8} {'Margin':<10} {'Commission':<12} {'USD/Pip':<8}")
    print("-" * 70)
    
    for instrument in test_instruments:
        print(f"{instrument:<10}", end="")
        
        # Static parameters
        try:
            margin_static, commission_static, usd_pip_static = get_dynamic_csi_parameters(
                instrument, use_live_api=False
            )
            print(f" Static   ${margin_static:<9.0f} ${commission_static:<11.2f} ${usd_pip_static:<7.2f}")
        except Exception as e:
            print(f" Static   ERROR: {e}")
        
        # Dynamic parameters
        print(f"{'':<10}", end="")
        try:
            margin_dynamic, commission_dynamic, usd_pip_dynamic = get_dynamic_csi_parameters(
                instrument, use_live_api=True
            )
            print(f" Dynamic  ${margin_dynamic:<9.0f} ${commission_dynamic:<11.2f} ${usd_pip_dynamic:<7.2f}")
            
            # Calculate differences
            margin_diff = ((margin_dynamic - margin_static) / margin_static) * 100
            commission_diff = ((commission_dynamic - commission_static) / commission_static) * 100
            
            print(f"{'':<10} Diff%    {margin_diff:<+9.1f}% {commission_diff:<+11.1f}% {'':8}")
            
        except Exception as e:
            print(f" Dynamic  ERROR: {e}")
        
        print()

def test_csi_calculation_difference():
    """Test actual CSI calculation differences"""
    load_dotenv()
    
    print("🧮 CSI CALCULATION COMPARISON")
    print("=" * 50)
    
    # Simulate typical values for EUR_USD
    adxr = 25.0
    atr_pips = 20.0  # 20 pips ATR
    
    print(f"Test conditions: ADXR={adxr}, ATR={atr_pips} pips")
    print()
    
    # Static CSI calculation
    try:
        margin_static, commission_static, usd_pip_static = get_dynamic_csi_parameters(
            'EUR_USD', use_live_api=False
        )
        
        sqrt_M_static = (margin_static ** 0.5)
        commission_factor_static = 1.0 / (150.0 + commission_static)
        economic_factor_static = usd_pip_static / sqrt_M_static * commission_factor_static
        csi_static = adxr * atr_pips * economic_factor_static * 100.0
        
        print(f"Static CSI:  {csi_static:.1f}")
        print(f"  M=${margin_static:.0f}, C=${commission_static:.2f}, V=${usd_pip_static:.2f}")
        
    except Exception as e:
        print(f"Static CSI calculation failed: {e}")
    
    # Dynamic CSI calculation  
    try:
        margin_dynamic, commission_dynamic, usd_pip_dynamic = get_dynamic_csi_parameters(
            'EUR_USD', use_live_api=True
        )
        
        sqrt_M_dynamic = (margin_dynamic ** 0.5)
        commission_factor_dynamic = 1.0 / (150.0 + commission_dynamic)
        economic_factor_dynamic = usd_pip_dynamic / sqrt_M_dynamic * commission_factor_dynamic
        csi_dynamic = adxr * atr_pips * economic_factor_dynamic * 100.0
        
        print(f"Dynamic CSI: {csi_dynamic:.1f}")
        print(f"  M=${margin_dynamic:.0f}, C=${commission_dynamic:.2f}, V=${usd_pip_dynamic:.2f}")
        
        # Calculate CSI difference
        if 'csi_static' in locals():
            csi_diff = ((csi_dynamic - csi_static) / csi_static) * 100
            print(f"CSI Difference: {csi_diff:+.1f}%")
        
    except Exception as e:
        print(f"Dynamic CSI calculation failed: {e}")

if __name__ == "__main__":
    compare_csi_parameters()
    print()
    test_csi_calculation_difference()