#!/usr/bin/env python3
"""
FX Instruments Configuration for Edge Finding Experiment
20 major FX pairs for cross-instrument context and prediction
"""

from typing import List, Dict

# 20 FX pairs for edge discovery system
FX_INSTRUMENTS: List[str] = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
    'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
    'AUD_JPY', 'CAD_JPY'
]

# Instrument groups for context tensor organization
CURRENCY_GROUPS: Dict[str, List[str]] = {
    'USD_MAJORS': ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD'],
    'EUR_CROSSES': ['EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD'],
    'GBP_CROSSES': ['GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD'],
    'JPY_CROSSES': ['AUD_JPY', 'CAD_JPY']
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
    'GBP_AUD': 0.4, 'GBP_CAD': 0.4, 'GBP_NZD': 0.5, 'AUD_JPY': 0.3, 'CAD_JPY': 0.3
}

def get_pip_value(instrument: str) -> float:
    """Get pip value for instrument (0.0001 for most, 0.01 for JPY pairs)"""
    if 'JPY' in instrument:
        return 0.01
    return 0.0001

def get_instrument_priority() -> Dict[str, int]:
    """Get processing priority (1=highest) for instruments"""
    priority = {}
    for i, instrument in enumerate(FX_INSTRUMENTS):
        priority[instrument] = i + 1
    return priority