#!/usr/bin/env python3
"""
FX Instruments Configuration for Edge Finding Experiment
20 major FX pairs for cross-instrument context and prediction
"""

from typing import List, Dict

# 24 FX pairs for edge discovery system - includes original 20 plus 4 additional pairs
FX_INSTRUMENTS: List[str] = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
    'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
    'AUD_JPY', 'CAD_JPY', 'AUD_CHF', 'AUD_NZD', 'CHF_JPY', 'NZD_JPY'
]

# Instrument groups for context tensor organization  
CURRENCY_GROUPS: Dict[str, List[str]] = {
    'USD_MAJORS': ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD'],
    'EUR_CROSSES': ['EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD'],
    'GBP_CROSSES': ['GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD'],
    'JPY_CROSSES': ['AUD_JPY', 'CAD_JPY', 'CHF_JPY', 'NZD_JPY'],
    'OTHER_CROSSES': ['AUD_CHF', 'AUD_NZD']
}

# Market session times (UTC) for regime detection
MARKET_SESSIONS: Dict[str, Dict[str, int]] = {
    'SYDNEY': {'start': 22, 'end': 7},
    'TOKYO': {'start': 0, 'end': 9},
    'LONDON': {'start': 8, 'end': 17},
    'NEW_YORK': {'start': 13, 'end': 22}
}

# Typical spreads (pips) for transaction cost calculation
TYPICAL_SPREADS: Dict[str, float] = {
    'EUR_USD': 0.1, 'GBP_USD': 0.2, 'USD_JPY': 0.1, 'USD_CHF': 0.2,
    'AUD_USD': 0.2, 'USD_CAD': 0.2, 'NZD_USD': 0.3,
    'EUR_GBP': 0.2, 'EUR_JPY': 0.2, 'EUR_CHF': 0.3, 'EUR_AUD': 0.3,
    'EUR_CAD': 0.3, 'EUR_NZD': 0.4, 'GBP_JPY': 0.3, 'GBP_CHF': 0.4,
    'GBP_AUD': 0.4, 'GBP_CAD': 0.4, 'GBP_NZD': 0.5, 'AUD_JPY': 0.3, 'CAD_JPY': 0.3,
    'AUD_CHF': 0.4, 'AUD_NZD': 0.4, 'CHF_JPY': 0.3, 'NZD_JPY': 0.4
}

def get_pip_size(instrument: str) -> float:
    """
    Get pip size for instrument.
    
    Returns:
        float: 0.0001 for most pairs, 0.01 for JPY pairs
    """
    if 'JPY' in instrument:
        return 0.01
    else:
        return 0.0001

def calculate_pip_value_usd(instrument: str, current_rate: float, usd_rates: dict = None) -> float:
    """
    Calculate dynamic pip value in USD per standard lot for current rates.
    
    Args:
        instrument: FX pair (e.g., 'EUR_USD', 'USD_JPY', 'AUD_CHF')
        current_rate: Current exchange rate for the instrument
        usd_rates: Dict of current USD rates for cross-currency calculation
                  Required keys: 'EUR_USD', 'GBP_USD', 'AUD_USD', etc.
    
    Returns:
        float: USD value of 1 pip per 100,000 units
    """
    pip_size = get_pip_size(instrument)
    base_currency, quote_currency = instrument.split('_')
    
    # Standard lot size
    lot_size = 100000
    
    # Calculate pip value in quote currency
    pip_value_quote = lot_size * pip_size
    
    # Convert to USD
    if quote_currency == 'USD':
        # Quote currency is USD, so pip value is already in USD
        pip_value_usd = pip_value_quote
    elif base_currency == 'USD':
        # Base currency is USD (e.g., USD_JPY), pip value in quote currency / current rate
        pip_value_usd = pip_value_quote / current_rate
    else:
        # Cross currency pair (e.g., EUR_GBP, AUD_CHF)
        # Need to convert quote currency to USD
        if usd_rates is None:
            # Fallback to simplified calculation
            return 10.0 if 'JPY' not in instrument else 7.0
        
        quote_to_usd_pair = f"{quote_currency}_USD"
        usd_to_quote_pair = f"USD_{quote_currency}"
        
        if quote_to_usd_pair in usd_rates:
            # Direct quote currency to USD rate
            pip_value_usd = pip_value_quote * usd_rates[quote_to_usd_pair]
        elif usd_to_quote_pair in usd_rates:
            # Inverse: USD to quote currency rate
            pip_value_usd = pip_value_quote / usd_rates[usd_to_quote_pair]
        else:
            # Fallback to simplified calculation
            pip_value_usd = 10.0 if 'JPY' not in instrument else 7.0
    
    return pip_value_usd

def get_pip_value(instrument: str) -> tuple:
    """
    Legacy function for backwards compatibility.
    Returns static pip values - use calculate_pip_value_usd() for dynamic calculation.
    """
    pip_size = get_pip_size(instrument)
    
    if 'JPY' in instrument:
        pip_value_usd = 7.0  # Simplified
    else:
        pip_value_usd = 10.0  # Simplified
    
    return pip_size, pip_value_usd

def extract_instrument_from_filename(filename: str) -> str:
    """
    Extract FX instrument name from CSV filename.
    
    Args:
        filename: e.g., 'AUD_CHF_3years_H1.csv', '/path/to/EUR_USD_3years_H1.csv'
    
    Returns:
        str: Instrument name, e.g., 'AUD_CHF', 'EUR_USD'
    """
    import os
    basename = os.path.basename(filename)
    
    # Remove .csv extension
    if basename.endswith('.csv'):
        basename = basename[:-4]
    
    # Split by underscore and take first two parts (base_quote)
    parts = basename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        raise ValueError(f"Cannot extract instrument from filename: {filename}")

def get_instrument_priority() -> Dict[str, int]:
    """Get processing priority (1=highest) for instruments"""
    priority = {}
    for i, instrument in enumerate(FX_INSTRUMENTS):
        priority[instrument] = i + 1
    return priority