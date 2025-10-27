#!/usr/bin/env python3
"""
Production-ready incremental indicator calculation system

Implements the single-source-of-truth incremental update function that ensures
identical calculations between historical training and live production processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import bisect
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from configs.instruments import get_pip_value

logger = logging.getLogger(__name__)

@dataclass 
class InstrumentState:
    """State for a single instrument within the combined state structure"""
    # ASI/Swing Point State
    asi_value: float = 0.0
    recent_highs: deque = field(default_factory=lambda: deque(maxlen=50))  # (index, price)
    recent_lows: deque = field(default_factory=lambda: deque(maxlen=50))   # (index, price)
    
    # ATR State (Dollar Scaling)  
    atr_ema: float = 0.0
    prev_close_usd: float = 0.0
    atr_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # ADX State
    adx_value: float = 0.0
    di_plus_ema: float = 0.0
    di_minus_ema: float = 0.0
    tr_ema: float = 0.0
    dm_plus_ema: float = 0.0
    dm_minus_ema: float = 0.0
    adx_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # Price Change State
    prev_close: float = 0.0
    
    # General State
    bar_count: int = 0
    pip_size: float = 0.0001
    pip_value: float = 10.0
    
    # Historical data for slope calculation
    high_history: deque = field(default_factory=lambda: deque(maxlen=100))
    low_history: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class MultiInstrumentState:
    """Combined state management for all 20 FX instruments"""
    instruments: Dict[str, InstrumentState] = field(default_factory=dict)
    
    # Cross-instrument context tensor state
    context_matrix: np.ndarray = field(default_factory=lambda: np.zeros((20, 5)))  # 20 instruments × 5 features
    context_timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    # Global market regime state
    market_regime: int = 0  # 16-state regime framework
    regime_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def get_instrument_state(self, instrument: str) -> InstrumentState:
        """Get or create state for an instrument"""
        if instrument not in self.instruments:
            pip_size, pip_value = get_pip_value(instrument)
            self.instruments[instrument] = InstrumentState(pip_size=pip_size, pip_value=pip_value)
        return self.instruments[instrument]

class IncrementalIndicatorCalculator:
    """
    Production-ready incremental indicator calculator
    
    Ensures identical results between batch and incremental processing
    """
    
    def __init__(self, atr_period: int = 14, adx_period: int = 14):
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.atr_alpha = 2.0 / (atr_period + 1)
        self.adx_alpha = 2.0 / (adx_period + 1)
    
    def calculate_atr_usd_incremental(
        self, 
        ohlc: Dict[str, float], 
        state: InstrumentState
    ) -> float:
        """Calculate ATR in USD terms incrementally"""
        
        # Convert to USD
        high_usd = ohlc['high'] / state.pip_size * state.pip_value
        low_usd = ohlc['low'] / state.pip_size * state.pip_value
        close_usd = ohlc['close'] / state.pip_size * state.pip_value
        
        if state.bar_count == 0:
            # First bar
            state.prev_close_usd = close_usd
            return 0.0
        
        # Calculate True Range
        tr1 = high_usd - low_usd
        tr2 = abs(high_usd - state.prev_close_usd)
        tr3 = abs(low_usd - state.prev_close_usd)
        true_range = max(tr1, tr2, tr3)
        
        # Update ATR EMA
        if state.atr_ema == 0.0:
            state.atr_ema = true_range
        else:
            state.atr_ema = self.atr_alpha * true_range + (1 - self.atr_alpha) * state.atr_ema
        
        # Update previous close
        state.prev_close_usd = close_usd
        
        return state.atr_ema
    
    def calculate_adx_incremental(
        self, 
        ohlc: Dict[str, float], 
        state: InstrumentState
    ) -> float:
        """Calculate ADX incrementally"""
        
        if state.bar_count == 0:
            # Store first bar data
            state.prev_close = ohlc['close']
            return 0.0
        
        # Calculate directional movement
        high_diff = ohlc['high'] - (state.high_history[-1] if state.high_history else ohlc['high'])
        low_diff = (state.low_history[-1] if state.low_history else ohlc['low']) - ohlc['low']
        
        dm_plus = high_diff if high_diff > low_diff and high_diff > 0 else 0
        dm_minus = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        # Calculate True Range
        prev_close = state.prev_close
        tr1 = ohlc['high'] - ohlc['low']
        tr2 = abs(ohlc['high'] - prev_close)
        tr3 = abs(ohlc['low'] - prev_close)
        true_range = max(tr1, tr2, tr3)
        
        # Update EMAs
        if state.tr_ema == 0.0:
            state.tr_ema = true_range
            state.dm_plus_ema = dm_plus
            state.dm_minus_ema = dm_minus
        else:
            state.tr_ema = self.adx_alpha * true_range + (1 - self.adx_alpha) * state.tr_ema
            state.dm_plus_ema = self.adx_alpha * dm_plus + (1 - self.adx_alpha) * state.dm_plus_ema
            state.dm_minus_ema = self.adx_alpha * dm_minus + (1 - self.adx_alpha) * state.dm_minus_ema
        
        # Calculate DI+ and DI-
        if state.tr_ema > 0:
            di_plus = 100 * state.dm_plus_ema / state.tr_ema
            di_minus = 100 * state.dm_minus_ema / state.tr_ema
        else:
            di_plus = di_minus = 0
        
        # Calculate DX
        di_sum = di_plus + di_minus
        if di_sum > 0:
            dx = 100 * abs(di_plus - di_minus) / di_sum
        else:
            dx = 0
        
        # Update ADX EMA
        if state.adx_value == 0.0:
            state.adx_value = dx
        else:
            state.adx_value = self.adx_alpha * dx + (1 - self.adx_alpha) * state.adx_value
        
        # Update previous close
        state.prev_close = ohlc['close']
        
        return state.adx_value
    
    def calculate_percentile_scaling(self, value: float, history: deque) -> float:
        """Apply percentile scaling to [0,1] range"""
        if len(history) < 2:
            return 0.5  # Default to middle until we have history
        
        # Calculate percentile rank
        sorted_values = sorted(history)
        rank = bisect.bisect_left(sorted_values, value) / len(history)
        
        # Cap at 99th percentile
        return min(0.995, rank)
    
    def detect_swing_points(
        self, 
        ohlc: Dict[str, float], 
        state: InstrumentState,
        min_distance: int = 3
    ) -> Tuple[bool, bool]:
        """Detect swing highs and lows incrementally"""
        
        # Add current bar to history
        current_idx = state.bar_count
        state.high_history.append(ohlc['high'])
        state.low_history.append(ohlc['low'])
        
        if len(state.high_history) < min_distance * 2 + 1:
            return False, False
        
        # Check for swing high (middle point higher than surrounding points)
        check_idx = len(state.high_history) - min_distance - 1
        if check_idx >= min_distance:
            is_swing_high = True
            check_high = state.high_history[check_idx]
            
            # Check left side
            for i in range(check_idx - min_distance, check_idx):
                if state.high_history[i] >= check_high:
                    is_swing_high = False
                    break
            
            # Check right side
            if is_swing_high:
                for i in range(check_idx + 1, min(len(state.high_history), check_idx + min_distance + 1)):
                    if state.high_history[i] >= check_high:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                state.recent_highs.append((current_idx - min_distance, check_high))
        
        # Check for swing low (similar logic)
        is_swing_low = False
        if check_idx >= min_distance:
            is_swing_low = True
            check_low = state.low_history[check_idx]
            
            # Check left side
            for i in range(check_idx - min_distance, check_idx):
                if state.low_history[i] <= check_low:
                    is_swing_low = False
                    break
            
            # Check right side
            if is_swing_low:
                for i in range(check_idx + 1, min(len(state.low_history), check_idx + min_distance + 1)):
                    if state.low_history[i] <= check_low:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                state.recent_lows.append((current_idx - min_distance, check_low))
        
        return len(state.recent_highs) > 0 and state.recent_highs[-1][0] == current_idx - min_distance, \
               len(state.recent_lows) > 0 and state.recent_lows[-1][0] == current_idx - min_distance
    
    def calculate_slope(self, points: deque, min_points: int = 3) -> float:
        """Calculate regression slope from swing points"""
        if len(points) < min_points:
            return np.nan
        
        # Take last min_points for slope calculation
        recent_points = list(points)[-min_points:]
        
        if len(recent_points) < 2:
            return np.nan
        
        # Extract x (indices) and y (prices)
        x = np.array([p[0] for p in recent_points])
        y = np.array([p[1] for p in recent_points])
        
        # Calculate slope using least squares
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        return np.nan

def update_indicators(
    new_ohlc: Dict[str, float],
    multi_state: MultiInstrumentState,
    instrument: str
) -> Tuple[Dict[str, float], MultiInstrumentState]:
    """
    Production-ready incremental update of 5 indicators
    
    Args:
        new_ohlc: {'open': x, 'high': y, 'low': z, 'close': w}
        multi_state: Combined state for all instruments
        instrument: FX pair identifier
        
    Returns:
        indicators: {'slope_high': x, 'slope_low': y, 'volatility': z, 'direction': w, 'price_change': v}
        updated_state: New state for next calculation
    """
    calculator = IncrementalIndicatorCalculator()
    
    # Get instrument-specific state
    state = multi_state.get_instrument_state(instrument)
    
    # Update bar count
    state.bar_count += 1
    
    # 1. Calculate ATR in USD
    atr_usd = calculator.calculate_atr_usd_incremental(new_ohlc, state)
    
    # Update ATR history and calculate volatility
    if atr_usd > 0:
        state.atr_history.append(atr_usd)
    volatility = calculator.calculate_percentile_scaling(atr_usd, state.atr_history) if atr_usd > 0 else np.nan
    
    # 2. Calculate ADX
    adx_value = calculator.calculate_adx_incremental(new_ohlc, state)
    
    # Update ADX history and calculate direction
    if adx_value > 0:
        state.adx_history.append(adx_value)
    direction = calculator.calculate_percentile_scaling(adx_value, state.adx_history) if adx_value > 0 else np.nan
    
    # 3. Detect swing points and calculate slopes
    is_swing_high, is_swing_low = calculator.detect_swing_points(new_ohlc, state)
    
    slope_high = calculator.calculate_slope(state.recent_highs)
    slope_low = calculator.calculate_slope(state.recent_lows)
    
    # 4. Calculate price change (5th indicator)
    if state.bar_count > 1 and state.prev_close > 0:
        price_change = (new_ohlc['close'] - state.prev_close) / state.prev_close
    else:
        price_change = 0.0
    
    # Update previous close for next iteration
    state.prev_close = new_ohlc['close']
    
    # Prepare indicators output
    indicators = {
        'slope_high': slope_high,
        'slope_low': slope_low,
        'volatility': volatility,
        'direction': direction,
        'price_change': price_change
    }
    
    # Update cross-instrument context tensor (placeholder for now)
    # In production, this would update the actual context matrix
    multi_state.context_timestamp = pd.Timestamp.now()
    
    return indicators, multi_state

def process_historical_data_incremental(
    df: pd.DataFrame, 
    instrument: str,
    multi_state: Optional[MultiInstrumentState] = None
) -> Tuple[pd.DataFrame, MultiInstrumentState]:
    """
    Process historical data using incremental updates
    
    This ensures identical calculations to live processing
    """
    if multi_state is None:
        multi_state = MultiInstrumentState()
    
    results = []
    
    for idx, row in df.iterrows():
        ohlc = {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        indicators, multi_state = update_indicators(ohlc, multi_state, instrument)
        
        # Add timestamp and combine with original data
        result = {
            'time': idx,
            **indicators
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.set_index('time', inplace=True)
    
    return results_df, multi_state

# For testing and validation
if __name__ == "__main__":
    # Test with sample data
    test_data = {
        'open': [1.1000, 1.1010, 1.1020],
        'high': [1.1015, 1.1025, 1.1035],
        'low': [0.9995, 1.1005, 1.1015],
        'close': [1.1010, 1.1020, 1.1030]
    }
    
    df = pd.DataFrame(test_data)
    df.index = pd.date_range('2025-01-01', periods=len(df), freq='H')
    
    results, state = process_historical_data_incremental(df, "EUR_USD")
    print("Test results:")
    print(results)
    print(f"Final state bar count: {state.get_instrument_state('EUR_USD').bar_count}")