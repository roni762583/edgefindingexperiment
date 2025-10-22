"""
Unit tests for instruments configuration module
"""

import pytest
from configs.instruments import (
    FX_INSTRUMENTS,
    CURRENCY_GROUPS,
    MARKET_SESSIONS,
    TYPICAL_SPREADS,
    get_pip_value,
    get_instrument_priority
)


class TestInstrumentsConfig:
    """Test suite for instruments configuration"""

    def test_fx_instruments_count(self):
        """Test that we have exactly 20 FX instruments"""
        assert len(FX_INSTRUMENTS) == 20, f"Expected 20 instruments, got {len(FX_INSTRUMENTS)}"

    def test_fx_instruments_format(self):
        """Test that all instruments follow OANDA format (XXX_YYY)"""
        for instrument in FX_INSTRUMENTS:
            assert isinstance(instrument, str), f"Instrument {instrument} is not a string"
            assert '_' in instrument, f"Instrument {instrument} missing underscore"
            
            parts = instrument.split('_')
            assert len(parts) == 2, f"Instrument {instrument} should have exactly 2 parts"
            
            base, quote = parts
            assert len(base) == 3, f"Base currency {base} should be 3 characters"
            assert len(quote) == 3, f"Quote currency {quote} should be 3 characters"
            assert base.isupper(), f"Base currency {base} should be uppercase"
            assert quote.isupper(), f"Quote currency {quote} should be uppercase"

    def test_no_duplicate_instruments(self):
        """Test that there are no duplicate instruments"""
        assert len(FX_INSTRUMENTS) == len(set(FX_INSTRUMENTS)), "Duplicate instruments found"

    def test_major_pairs_included(self):
        """Test that major FX pairs are included"""
        major_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD']
        for pair in major_pairs:
            assert pair in FX_INSTRUMENTS, f"Major pair {pair} not found in instruments"

    def test_currency_groups_coverage(self):
        """Test that currency groups cover all instruments"""
        all_grouped_instruments = []
        for group_instruments in CURRENCY_GROUPS.values():
            all_grouped_instruments.extend(group_instruments)
        
        # Remove duplicates
        unique_grouped = set(all_grouped_instruments)
        
        # Check that all instruments are covered
        for instrument in FX_INSTRUMENTS:
            assert instrument in unique_grouped, f"Instrument {instrument} not in any group"

    def test_currency_groups_structure(self):
        """Test currency groups have expected structure"""
        expected_groups = ['USD_MAJORS', 'EUR_CROSSES', 'GBP_CROSSES', 'JPY_CROSSES']
        
        for group in expected_groups:
            assert group in CURRENCY_GROUPS, f"Expected group {group} not found"
            assert isinstance(CURRENCY_GROUPS[group], list), f"Group {group} should be a list"
            assert len(CURRENCY_GROUPS[group]) > 0, f"Group {group} should not be empty"

    def test_market_sessions_structure(self):
        """Test market sessions have correct structure"""
        expected_sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NEW_YORK']
        
        for session in expected_sessions:
            assert session in MARKET_SESSIONS, f"Session {session} not found"
            
            session_data = MARKET_SESSIONS[session]
            assert 'start' in session_data, f"Session {session} missing start time"
            assert 'end' in session_data, f"Session {session} missing end time"
            
            assert 0 <= session_data['start'] <= 23, f"Session {session} start time invalid"
            assert 0 <= session_data['end'] <= 23, f"Session {session} end time invalid"

    def test_typical_spreads_coverage(self):
        """Test that all instruments have typical spreads defined"""
        for instrument in FX_INSTRUMENTS:
            assert instrument in TYPICAL_SPREADS, f"No spread defined for {instrument}"
            
            spread = TYPICAL_SPREADS[instrument]
            assert isinstance(spread, (int, float)), f"Spread for {instrument} should be numeric"
            assert spread > 0, f"Spread for {instrument} should be positive"
            assert spread < 10, f"Spread for {instrument} seems too large: {spread}"

    def test_pip_value_calculation(self):
        """Test pip value calculation for different instrument types"""
        # Test JPY pairs
        jpy_pairs = [inst for inst in FX_INSTRUMENTS if 'JPY' in inst]
        for pair in jpy_pairs:
            assert get_pip_value(pair) == 0.01, f"JPY pair {pair} should have pip value 0.01"
        
        # Test non-JPY pairs
        non_jpy_pairs = [inst for inst in FX_INSTRUMENTS if 'JPY' not in inst]
        for pair in non_jpy_pairs:
            assert get_pip_value(pair) == 0.0001, f"Non-JPY pair {pair} should have pip value 0.0001"

    def test_instrument_priority_unique(self):
        """Test that instrument priorities are unique"""
        priorities = get_instrument_priority()
        
        assert len(priorities) == len(FX_INSTRUMENTS), "Priority count mismatch"
        
        priority_values = list(priorities.values())
        assert len(priority_values) == len(set(priority_values)), "Duplicate priorities found"
        
        # Check priorities are sequential starting from 1
        expected_priorities = set(range(1, len(FX_INSTRUMENTS) + 1))
        actual_priorities = set(priority_values)
        assert actual_priorities == expected_priorities, "Priorities should be sequential from 1"

    def test_instrument_priority_coverage(self):
        """Test that all instruments have priorities"""
        priorities = get_instrument_priority()
        
        for instrument in FX_INSTRUMENTS:
            assert instrument in priorities, f"No priority for instrument {instrument}"
            
            priority = priorities[instrument]
            assert isinstance(priority, int), f"Priority for {instrument} should be integer"
            assert 1 <= priority <= len(FX_INSTRUMENTS), f"Priority {priority} out of range"

    @pytest.mark.parametrize("instrument,expected_pip", [
        ("EUR_USD", 0.0001),
        ("GBP_USD", 0.0001),
        ("USD_JPY", 0.01),
        ("GBP_JPY", 0.01),
        ("EUR_JPY", 0.01),
        ("AUD_JPY", 0.01)
    ])
    def test_pip_values_parametrized(self, instrument, expected_pip):
        """Parametrized test for pip values"""
        assert get_pip_value(instrument) == expected_pip