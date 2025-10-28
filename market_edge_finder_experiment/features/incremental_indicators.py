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
    
    # WILDER'S PROPER ASI SWING DETECTION STATE
    # Two-step process: Candidate Detection → Breakout Confirmation
    
    # Last CONFIRMED significant points (only after breakout confirmation)
    last_sig_hsp_index: Optional[int] = None      # Index of last significant HSP
    last_sig_hsp_asi: Optional[float] = None      # ASI value at last significant HSP  
    last_sig_hsp_price: Optional[float] = None    # HIP (High Price) at significant HSP
    
    last_sig_lsp_index: Optional[int] = None      # Index of last significant LSP
    last_sig_lsp_asi: Optional[float] = None      # ASI value at last significant LSP
    last_sig_lsp_price: Optional[float] = None    # LOP (Low Price) at significant LSP
    
    # PENDING candidates (awaiting breakout confirmation)
    pending_hsp_index: Optional[int] = None       # Index of pending HSP candidate
    pending_hsp_asi: Optional[float] = None       # ASI value of pending HSP
    pending_hsp_price: Optional[float] = None     # High price of pending HSP
    
    pending_lsp_index: Optional[int] = None       # Index of pending LSP candidate  
    pending_lsp_asi: Optional[float] = None       # ASI value of pending LSP
    pending_lsp_price: Optional[float] = None     # Low price of pending LSP
    
    # LEGACY: Old flawed tracking for slope calculation (backward compatibility)
    hsp_indices: deque = field(default_factory=lambda: deque(maxlen=100))   # Indices of HSPs
    hsp_values: deque = field(default_factory=lambda: deque(maxlen=100))    # ASI values at HSPs
    lsp_indices: deque = field(default_factory=lambda: deque(maxlen=100))   # Indices of LSPs  
    lsp_values: deque = field(default_factory=lambda: deque(maxlen=100))    # ASI values at LSPs
    
    # Last calculated angles for forward-filling (like batch method)
    last_hsp_angle: float = np.nan
    last_lsp_angle: float = np.nan
    
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
    prev_open: float = 0.0
    prev_high: float = 0.0
    prev_low: float = 0.0
    
    # General State
    bar_count: int = 0
    pip_size: float = 0.0001
    pip_value: float = 10.0
    
    # Historical data for slope calculation
    high_history: deque = field(default_factory=lambda: deque(maxlen=100))
    low_history: deque = field(default_factory=lambda: deque(maxlen=100))
    asi_history: deque = field(default_factory=lambda: deque(maxlen=100))

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
        # Use the EXACT same smoothing methods as batch processing
        self.atr_alpha = 2.0 / (atr_period + 1)  # Standard EMA for ATR (like batch)
        self.adx_alpha = 1.0 / adx_period        # Wilder's smoothing for ADX (like batch)
    
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
            # Initialize history
            state.high_history.append(ohlc['high'])
            state.low_history.append(ohlc['low'])
            return 0.0
        
        # Calculate directional movement
        prev_high = state.high_history[-1] if state.high_history else ohlc['high']
        prev_low = state.low_history[-1] if state.low_history else ohlc['low']
        
        high_diff = ohlc['high'] - prev_high
        low_diff = prev_low - ohlc['low']
        
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
        
        # Update previous close and add current values to history
        state.prev_close = ohlc['close']
        state.high_history.append(ohlc['high'])
        state.low_history.append(ohlc['low'])
        
        return state.adx_value
    
    def calculate_percentile_scaling(self, value: float, history: deque, window: int = 200) -> float:
        """Apply percentile scaling to [0,1] range using 200-bar rolling window (matching batch)"""
        if len(history) < 2:
            return 0.5  # Default to middle until we have history
        
        # Use last 'window' values for scaling (matching batch rolling window logic)
        recent_values = list(history)[-window:] if len(history) > window else list(history)
        
        # Filter out NaN/invalid values
        valid_values = [v for v in recent_values if not np.isnan(v) and v is not None]
        
        if len(valid_values) < 2:
            return 0.5
        
        # Calculate percentile rank (exact match to batch method)
        count_below = sum(1 for v in valid_values if v < value)
        percentile_rank = count_below / len(valid_values)
        
        # Scale to [0,1] and cap at 1.0 (99th percentile protection, matching batch)
        return min(1.0, percentile_rank)
    
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
    
    def calculate_asi_incremental(
        self, 
        ohlc: Dict[str, float], 
        state: InstrumentState,
        instrument: str,
        limit_multiplier: float = 3.0
    ) -> float:
        """Calculate ASI (Accumulation Swing Index) incrementally with USD normalization"""
        
        if state.bar_count == 0:
            # Initialize first bar
            state.asi_value = 0.0
            return 0.0
        
        # Get previous OHLC - need at least 1 previous bar
        if state.bar_count < 2:
            return state.asi_value
            
        # Convert current OHLC to USD (matching batch calculation)
        # Convert to pips first, then to USD
        current_ohlc_pips = {
            'open': ohlc['open'] / state.pip_size,
            'high': ohlc['high'] / state.pip_size,
            'low': ohlc['low'] / state.pip_size,
            'close': ohlc['close'] / state.pip_size
        }
        
        current_ohlc_usd = {
            'open': current_ohlc_pips['open'] * state.pip_value,
            'high': current_ohlc_pips['high'] * state.pip_value,
            'low': current_ohlc_pips['low'] * state.pip_value,
            'close': current_ohlc_pips['close'] * state.pip_value
        }
        
        # Convert previous OHLC to USD 
        prev_ohlc_pips = {
            'open': state.prev_open / state.pip_size,
            'high': state.prev_high / state.pip_size,
            'low': state.prev_low / state.pip_size,
            'close': state.prev_close / state.pip_size
        }
        
        prev_ohlc_usd = {
            'open': prev_ohlc_pips['open'] * state.pip_value,
            'high': prev_ohlc_pips['high'] * state.pip_value,
            'low': prev_ohlc_pips['low'] * state.pip_value,
            'close': prev_ohlc_pips['close'] * state.pip_value
        }
        
        # Current and previous values in USD (matching batch notation)
        C2 = current_ohlc_usd['close']  # Current close (USD)
        O2 = current_ohlc_usd['open']   # Current open (USD)
        H2 = current_ohlc_usd['high']   # Current high (USD)
        L2 = current_ohlc_usd['low']    # Current low (USD)
        
        C1 = prev_ohlc_usd['close']     # Previous close (USD)
        O1 = prev_ohlc_usd['open']      # Previous open (USD)
        
        # Calculate ATR in USD for limit move (matching batch calculation exactly)
        # Check if we have enough data for proper ATR (like batch method)
        if state.bar_count >= 14 and state.atr_ema > 0:
            # Use ATR if available (already in USD from calculate_atr_usd_incremental)
            L = limit_multiplier * state.atr_ema
        else:
            # Fallback: use recent range average (exact match to batch logic)
            # Calculate recent range over available bars (up to 10 bars like batch)
            recent_ranges = []
            
            # Add current range
            current_range_usd = abs(H2 - L2)
            recent_ranges.append(current_range_usd)
            
            # Add ranges from history if available 
            history_len = len(state.high_history)
            for j in range(max(0, history_len - 10), history_len):
                if j < len(state.high_history) and j < len(state.low_history):
                    # Convert historical ranges to USD
                    hist_high_usd = (state.high_history[j] / state.pip_size) * state.pip_value
                    hist_low_usd = (state.low_history[j] / state.pip_size) * state.pip_value
                    hist_range_usd = abs(hist_high_usd - hist_low_usd)
                    recent_ranges.append(hist_range_usd)
            
            # Calculate mean recent range (matching batch logic)
            if recent_ranges:
                recent_range_mean = np.mean(recent_ranges)
                L = limit_multiplier * recent_range_mean
            else:
                L = 1.0  # Final fallback
        
        # Swing Index calculation (Wilder's method) - exact match to batch
        EPSILON = 1e-10
        
        # Numerator (N) - same as batch
        N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
        
        # Range (R) - Wilder's original specification (3 scenarios, take maximum)
        term1 = abs(H2 - C1) - 0.5 * abs(L2 - C1) + 0.25 * abs(C1 - O1)
        term2 = abs(L2 - C1) - 0.5 * abs(H2 - C1) + 0.25 * abs(C1 - O1)
        term3 = (H2 - L2) + 0.25 * abs(C1 - O1)
        
        R = max(term1, term2, term3)
        if R <= 0:
            R = EPSILON
            
        # Limit Factor (K) - Wilder's specification
        K = max(abs(H2 - C1), abs(L2 - C1))
        
        # SI Calculation with Wilder's 50x multiplier (exact match to batch)
        if L > EPSILON:
            SI = 50.0 * (N / R) * (K / L)
            SI = round(SI)  # Round to nearest integer (no capping)
        else:
            SI = 0
            
        # Accumulate swing index (exact match to batch)
        state.asi_value += SI
        
        return state.asi_value
    
    def detect_hsp_lsp_incremental(
        self,
        asi_value: float,
        state: InstrumentState
    ) -> Tuple[bool, bool]:
        """
        Detect HSP/LSP using 3-bar pattern to match batch method exactly
        Uses same logic as batch: middle bar higher/lower than both neighbors
        """
        current_idx = state.bar_count
        
        # Add current ASI to history (we need this for swing detection)
        state.asi_history.append(asi_value)
        
        # Need at least 3 bars for 3-bar pattern detection
        if len(state.asi_history) < 3:
            return False, False
            
        # Check 3-bar pattern: left, middle, right (like batch method)
        # We check the pattern that was just completed with the new bar
        if len(state.asi_history) >= 3:
            right_idx = len(state.asi_history) - 1  # Current bar (just added)
            middle_idx = len(state.asi_history) - 2  # Previous bar
            left_idx = len(state.asi_history) - 3   # Bar before that
            
            left_asi = state.asi_history[left_idx]
            middle_asi = state.asi_history[middle_idx] 
            right_asi = state.asi_history[right_idx]
            
            # Skip NaN values (like batch method)
            if np.isnan(left_asi) or np.isnan(middle_asi) or np.isnan(right_asi):
                return False, False
            
            # HSP: middle bar higher than both neighbors (exact match to batch)
            is_hsp = middle_asi > left_asi and middle_asi > right_asi
            
            # LSP: middle bar lower than both neighbors (exact match to batch)
            is_lsp = middle_asi < left_asi and middle_asi < right_asi
            
            # Apply alternating constraint (like batch method)
            detected_hsp = False
            detected_lsp = False
            
            # Determine the most recent swing type
            last_hsp_idx = state.hsp_indices[-1] if state.hsp_indices else -1
            last_lsp_idx = state.lsp_indices[-1] if state.lsp_indices else -1
            
            # FIXED: Calculate correct dataset index for the middle bar
            # current_idx is the current bar being processed (state.bar_count)
            # The middle bar of the 3-bar pattern is 2 bars ago in the dataset
            middle_dataset_idx = current_idx - 2
            
            if is_hsp:
                # Can add HSP if: 1) no previous swings, OR 2) last swing was LSP
                if len(state.hsp_indices) == 0 and len(state.lsp_indices) == 0:
                    # First swing point ever
                    state.hsp_indices.append(middle_dataset_idx)
                    state.hsp_values.append(middle_asi)
                    detected_hsp = True
                elif last_lsp_idx > last_hsp_idx:
                    # Last swing was LSP, so we can add HSP
                    state.hsp_indices.append(middle_dataset_idx)
                    state.hsp_values.append(middle_asi)
                    detected_hsp = True
                    
            if is_lsp:
                # Can add LSP if: 1) no previous swings, OR 2) last swing was HSP
                if len(state.hsp_indices) == 0 and len(state.lsp_indices) == 0:
                    # First swing point ever
                    state.lsp_indices.append(middle_dataset_idx)
                    state.lsp_values.append(middle_asi)
                    detected_lsp = True
                elif last_hsp_idx > last_lsp_idx:
                    # Last swing was HSP, so we can add LSP
                    state.lsp_indices.append(middle_dataset_idx)
                    state.lsp_values.append(middle_asi)
                    detected_lsp = True
                    
            return detected_hsp, detected_lsp
        
        return False, False
    
    def detect_hsp_lsp_wilder_proper(
        self,
        asi_value: float,
        high_price: float,
        low_price: float,
        state: InstrumentState
    ) -> Tuple[bool, bool]:
        """
        Proper Wilder ASI swing detection with two-step process:
        1. Candidate Detection: Local high/low in ASI (3-bar pattern)  
        2. Breakout Confirmation: Only confirmed after breaking opposite significant point
        
        Reference: New Concepts in Technical Trading Systems (1978), pp. 96-102
        """
        current_idx = state.bar_count
        
        # Add current ASI to history for candidate detection
        state.asi_history.append(asi_value)
        
        # Step 1: CANDIDATE DETECTION (3-bar pattern)
        detected_candidate = False
        
        if len(state.asi_history) >= 3:
            left_asi = state.asi_history[-3]
            middle_asi = state.asi_history[-2] 
            right_asi = state.asi_history[-1]  # Current bar
            
            # Skip NaN values
            if not (np.isnan(left_asi) or np.isnan(middle_asi) or np.isnan(right_asi)):
                # FIXED: Match batch method exactly - middle bar is at current_idx - 2
                # In batch: when processing bar i, middle is at i-1
                # In incremental: when processing bar current_idx, middle should be at current_idx - 2
                middle_dataset_idx = current_idx - 2  # Middle bar index in dataset
                
                # HSP Candidate: middle bar higher than both neighbors
                if middle_asi > left_asi and middle_asi > right_asi:
                    # Store as pending HSP candidate (needs breakout confirmation)
                    state.pending_hsp_index = middle_dataset_idx
                    state.pending_hsp_asi = middle_asi
                    state.pending_hsp_price = high_price  # HIP (High Price)
                    detected_candidate = True
                
                # LSP Candidate: middle bar lower than both neighbors  
                if middle_asi < left_asi and middle_asi < right_asi:
                    # Store as pending LSP candidate (needs breakout confirmation)
                    state.pending_lsp_index = middle_dataset_idx
                    state.pending_lsp_asi = middle_asi
                    state.pending_lsp_price = low_price  # LOP (Low Price)
                    detected_candidate = True
        
        # Step 2: BREAKOUT CONFIRMATION
        confirmed_hsp = False
        confirmed_lsp = False
        
        # Confirm pending HSP: ASI must drop BELOW last significant LSP
        if (state.pending_hsp_index is not None and 
            state.last_sig_lsp_asi is not None and
            asi_value < state.last_sig_lsp_asi):
            
            # Confirm the pending HSP as significant
            state.last_sig_hsp_index = state.pending_hsp_index
            state.last_sig_hsp_asi = state.pending_hsp_asi
            state.last_sig_hsp_price = state.pending_hsp_price
            
            # Add to legacy tracking for slope calculation compatibility
            state.hsp_indices.append(state.pending_hsp_index)
            state.hsp_values.append(state.pending_hsp_asi)
            
            # Clear pending state
            state.pending_hsp_index = None
            state.pending_hsp_asi = None
            state.pending_hsp_price = None
            
            confirmed_hsp = True
        
        # Confirm pending LSP: ASI must rise ABOVE last significant HSP
        if (state.pending_lsp_index is not None and
            state.last_sig_hsp_asi is not None and
            asi_value > state.last_sig_hsp_asi):
            
            # Confirm the pending LSP as significant
            state.last_sig_lsp_index = state.pending_lsp_index
            state.last_sig_lsp_asi = state.pending_lsp_asi
            state.last_sig_lsp_price = state.pending_lsp_price
            
            # Add to legacy tracking for slope calculation compatibility
            state.lsp_indices.append(state.pending_lsp_index)
            state.lsp_values.append(state.pending_lsp_asi)
            
            # Clear pending state
            state.pending_lsp_index = None
            state.pending_lsp_asi = None
            state.pending_lsp_price = None
            
            confirmed_lsp = True
        
        # Special case: First swing point (no prior significant point to break)
        if (state.last_sig_hsp_asi is None and state.last_sig_lsp_asi is None):
            # Allow first candidate to be confirmed immediately
            if state.pending_hsp_index is not None:
                state.last_sig_hsp_index = state.pending_hsp_index
                state.last_sig_hsp_asi = state.pending_hsp_asi
                state.last_sig_hsp_price = state.pending_hsp_price
                state.hsp_indices.append(state.pending_hsp_index)
                state.hsp_values.append(state.pending_hsp_asi)
                state.pending_hsp_index = None
                state.pending_hsp_asi = None
                state.pending_hsp_price = None
                confirmed_hsp = True
                
            elif state.pending_lsp_index is not None:
                state.last_sig_lsp_index = state.pending_lsp_index
                state.last_sig_lsp_asi = state.pending_lsp_asi
                state.last_sig_lsp_price = state.pending_lsp_price
                state.lsp_indices.append(state.pending_lsp_index)
                state.lsp_values.append(state.pending_lsp_asi)
                state.pending_lsp_index = None
                state.pending_lsp_asi = None
                state.pending_lsp_price = None
                confirmed_lsp = True
        
        return confirmed_hsp, confirmed_lsp
    
    def calculate_angle_slopes_incremental(
        self,
        state: InstrumentState
    ) -> Tuple[float, float]:
        """
        Calculate angle slopes from last two HSP/LSP points
        Exactly matches batch method's _calculate_angle_slopes_between_last_two_hsp_lsp
        """
        # Calculate HSP angle
        if len(state.hsp_indices) >= 2:
            # Get last two HSPs
            idx1, idx2 = state.hsp_indices[-2], state.hsp_indices[-1]
            y1, y2 = state.hsp_values[-2], state.hsp_values[-1]
            
            if idx2 != idx1:  # Avoid division by zero
                slope = (y2 - y1) / (idx2 - idx1)  # ASI units per bar
                angle_rad = np.arctan(slope)  # (-π/2, π/2)
                angle_deg = np.degrees(angle_rad)  # (-90, +90)
                # Linear mapping: (-90, +90) to (-1, +1)
                state.last_hsp_angle = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
        
        # Calculate LSP angle  
        if len(state.lsp_indices) >= 2:
            # Get last two LSPs
            idx1, idx2 = state.lsp_indices[-2], state.lsp_indices[-1]
            y1, y2 = state.lsp_values[-2], state.lsp_values[-1]
            
            if idx2 != idx1:  # Avoid division by zero
                slope = (y2 - y1) / (idx2 - idx1)  # ASI units per bar
                angle_rad = np.arctan(slope)  # (-π/2, π/2)
                angle_deg = np.degrees(angle_rad)  # (-90, +90)
                # Linear mapping: (-90, +90) to (-1, +1)  
                state.last_lsp_angle = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
        
        # Return current angles (forward-filled like batch method)
        return state.last_hsp_angle, state.last_lsp_angle

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
    
    # 3. Calculate ASI and detect HSP/LSP for slope calculation
    asi_value = calculator.calculate_asi_incremental(new_ohlc, state, instrument)
    state.asi_value = asi_value
    
    # Detect HSP/LSP using proper Wilder method (like batch method)
    is_hsp, is_lsp = calculator.detect_hsp_lsp_wilder_proper(
        asi_value, new_ohlc['high'], new_ohlc['low'], state
    )
    
    # Calculate angle slopes from last two HSP/LSP points (like batch method)
    slope_high, slope_low = calculator.calculate_angle_slopes_incremental(state)
    
    # 4. Calculate price change (5th indicator)
    if state.bar_count > 1 and state.prev_close > 0:
        price_change = (new_ohlc['close'] - state.prev_close) / state.prev_close
    else:
        price_change = 0.0
    
    # Update previous OHLC for next iteration (needed for ASI calculation)
    state.prev_open = new_ohlc['open']
    state.prev_high = new_ohlc['high']
    state.prev_low = new_ohlc['low']
    state.prev_close = new_ohlc['close']
    
    # Update historical data for range calculations
    state.high_history.append(new_ohlc['high'])
    state.low_history.append(new_ohlc['low'])
    
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