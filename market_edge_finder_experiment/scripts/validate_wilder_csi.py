#!/usr/bin/env python3
"""
Validate Wilder CSI calculation with manual examples
"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.instruments import get_csi_margin_requirement, get_csi_commission_cost, calculate_pip_value_usd

def validate_wilder_csi_soybeans():
    """Validate with Wilder's Soybeans example from the book"""
    print("🧪 VALIDATING WILDER'S SOYBEANS EXAMPLE")
    print("=" * 50)
    
    # Wilder's Soybeans example values
    adxr = 50.0
    atr_14_cents = 15.0  # 15¢ - let's try keeping it in cents rather than converting
    atr_14_decimal = 0.15  # 15¢ in price units
    V = 50.0       # $50 value of 1¢ move
    M = 3000.0     # $3000 margin
    C = 45.0       # $45 commission
    
    # Calculate CSI - Check bracket interpretation
    sqrt_M = np.sqrt(M)
    commission_factor = 1.0 / (150.0 + C)
    
    # Try different interpretations and ATR units
    print(f"Testing ATR in cents (15) vs decimal (0.15)")
    
    # Test with ATR in cents (correct for commodities)
    economic_factor_2 = (V / sqrt_M) * commission_factor
    csi_cents = adxr * atr_14_cents * economic_factor_2 * 100.0
    
    # Test with ATR in decimal  
    csi_decimal = adxr * atr_14_decimal * economic_factor_2 * 100.0
    
    # Also test if we need to remove the extra ×100
    csi_no_scale = adxr * atr_14_cents * economic_factor_2
    
    print(f"ADXR: {adxr}")
    print(f"ATR_14 (cents): {atr_14_cents}")
    print(f"ATR_14 (decimal): {atr_14_decimal}")
    print(f"V (value per cent): ${V}")
    print(f"M (margin): ${M}")
    print(f"C (commission): ${C}")
    print(f"√M: {sqrt_M:.2f}")
    print(f"Commission factor: {commission_factor:.4f}")
    print(f"Economic factor: {economic_factor_2:.6f}")
    print(f"CSI (ATR in cents): {csi_cents:.1f}")
    print(f"CSI (ATR decimal): {csi_decimal:.1f}")
    print(f"CSI (no ×100): {csi_no_scale:.1f}")
    print(f"Expected CSI: 348")
    
    # Check which is closest
    closest_diff = min(abs(csi_cents - 348), abs(csi_decimal - 348), abs(csi_no_scale - 348))
    if abs(csi_cents - 348) == closest_diff:
        print(f"✅ ATR in cents is closest (diff: {abs(csi_cents - 348):.1f})")
    elif abs(csi_decimal - 348) == closest_diff:
        print(f"✅ ATR decimal is closest (diff: {abs(csi_decimal - 348):.1f})")
    else:
        print(f"✅ No ×100 scaling is closest (diff: {abs(csi_no_scale - 348):.1f})")
    print()

def validate_eur_usd_csi():
    """Validate EUR_USD CSI calculation"""
    print("🧪 VALIDATING EUR_USD CSI CALCULATION")
    print("=" * 50)
    
    # Typical EUR_USD values
    adxr = 25.0            # Typical ADX for EUR_USD
    atr_14_raw = 0.0020    # Typical ATR in EUR_USD price units (about 20 pips)
    atr_14_pips = atr_14_raw / 0.0001  # Convert to pips: 0.0020 / 0.0001 = 20 pips
    current_rate = 1.1650  # EUR_USD rate
    
    # Get FX economic factors
    V = calculate_pip_value_usd("EUR_USD", current_rate)
    M = get_csi_margin_requirement("EUR_USD")
    C = get_csi_commission_cost("EUR_USD")
    
    # Calculate CSI using both raw and pip-converted ATR
    sqrt_M = np.sqrt(M)
    commission_factor = 1.0 / (150.0 + C)
    economic_factor = V / sqrt_M * commission_factor
    csi_raw = adxr * atr_14_raw * economic_factor * 100.0
    csi_pips = adxr * atr_14_pips * economic_factor * 100.0
    
    print(f"ADXR: {adxr}")
    print(f"ATR_14 (raw): {atr_14_raw} ({atr_14_pips:.0f} pips)")
    print(f"ATR_14 (pips): {atr_14_pips}")
    print(f"Current rate: {current_rate}")
    print(f"V (pip value): ${V}")
    print(f"M (margin): ${M}")
    print(f"C (commission): ${C}")
    print(f"√M: {sqrt_M:.2f}")
    print(f"Commission factor: {commission_factor:.4f}")
    print(f"Economic factor: {economic_factor:.6f}")
    print(f"CSI (raw ATR): {csi_raw:.4f}")
    print(f"CSI (pip ATR): {csi_pips:.4f}")
    print()

def compare_fx_instruments():
    """Compare CSI across different FX instruments"""
    print("🧪 COMPARING CSI ACROSS FX INSTRUMENTS")
    print("=" * 50)
    
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "GBP_JPY", "EUR_GBP"]
    adxr = 25.0  # Same directional strength for comparison
    
    for instrument in instruments:
        # Typical ATR values (approximated)
        atr_typical = {
            "EUR_USD": 0.0015,   # ~15 pips
            "GBP_USD": 0.0025,   # ~25 pips  
            "USD_JPY": 0.15,     # ~15 pips (JPY scale)
            "GBP_JPY": 0.35,     # ~35 pips (JPY scale)
            "EUR_GBP": 0.0012    # ~12 pips
        }
        
        atr_14_raw = atr_typical.get(instrument, 0.0020)
        current_rate = 1.0 if "USD" in instrument.split("_")[1] else 1.20  # Simplified
        
        # Convert ATR to pips
        pip_size = 0.01 if 'JPY' in instrument else 0.0001
        atr_14_pips = atr_14_raw / pip_size
        
        # Get economic factors
        V = calculate_pip_value_usd(instrument, current_rate)
        M = get_csi_margin_requirement(instrument)
        C = get_csi_commission_cost(instrument)
        
        # Calculate CSI using ATR in pips
        sqrt_M = np.sqrt(M)
        commission_factor = 1.0 / (150.0 + C)
        economic_factor = V / sqrt_M * commission_factor
        csi = adxr * atr_14_pips * economic_factor * 100.0
        
        print(f"{instrument:<8}: CSI={csi:6.1f} (ATR={atr_14_pips:5.1f}pips, M=${M:4.0f}, C=${C:2.0f})")

if __name__ == "__main__":
    validate_wilder_csi_soybeans()
    validate_eur_usd_csi()
    compare_fx_instruments()