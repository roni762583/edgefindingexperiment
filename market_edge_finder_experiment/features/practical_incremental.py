#!/usr/bin/env python3
"""
Practical incremental indicator processing
Uses the practical swing detection method (min_distance=3) instead of complex Wilder
This should achieve high correlation with batch processing
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import logging

@dataclass
class PracticalInstrumentState:
    """
    State for practical incremental processing
    Uses simple swing detection with min_distance=3
    """
    bar_count: int = 0
    
    # Price and indicator histories
    ohlc_history: List[Dict[str, float]] = field(default_factory=list)
    asi_history: List[float] = field(default_factory=list)
    atr_history: List[float] = field(default_factory=list)
    adx_history: List[float] = field(default_factory=list)
    
    # ADX calculation components
    di_plus_history: List[float] = field(default_factory=list)
    di_minus_history: List[float] = field(default_factory=list)
    dx_history: List[float] = field(default_factory=list)
    
    # Smoothed DM and TR for proper ADX calculation
    dm_plus_smooth: float = 0.0
    dm_minus_smooth: float = 0.0
    tr_smooth: float = 0.0
    
    # Practical swing points (simple min_distance=3)
    hsp_indices: List[int] = field(default_factory=list)
    hsp_values: List[float] = field(default_factory=list)
    lsp_indices: List[int] = field(default_factory=list)
    lsp_values: List[float] = field(default_factory=list)
    
    # Current slope values (forward-filled)
    current_hsp_slope: float = np.nan
    current_lsp_slope: float = np.nan
    
    # Price change history for percentile scaling
    price_change_history: List[float] = field(default_factory=list)
    
    # Target calculation (next hour pip movement in USD)
    next_hour_targets: List[float] = field(default_factory=list)
    
    # CSI (Commodity Selection Index) components
    csi_history: List[float] = field(default_factory=list)
    current_csi: float = np.nan

class PracticalIncrementalCalculator:
    """
    Practical incremental calculator using simple swing detection
    Designed to match batch practical method with high correlation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_practical_swing_incremental(
        self, 
        asi_value: float, 
        state: PracticalInstrumentState,
        min_distance: int = 3
    ) -> Tuple[bool, bool]:
        """
        Practical swing detection - incremental version
        Uses simple 3-bar patterns with minimum distance filtering
        """
        # Add current ASI to history
        state.asi_history.append(asi_value)
        current_idx = state.bar_count
        
        detected_hsp = False
        detected_lsp = False
        
        # Need at least 3 bars for pattern detection
        if len(state.asi_history) >= 3:
            # Check 3-bar pattern for middle bar (current_idx - 1)
            left_asi = state.asi_history[-3]
            middle_asi = state.asi_history[-2] 
            right_asi = state.asi_history[-1]  # Current
            middle_idx = current_idx - 1
            
            # Skip NaN values
            if not (np.isnan(left_asi) or np.isnan(middle_asi) or np.isnan(right_asi)):
                
                # HSP: middle bar higher than both neighbors
                if middle_asi > left_asi and middle_asi > right_asi:
                    # Check minimum distance from last HSP
                    if not state.hsp_indices or (middle_idx - state.hsp_indices[-1]) >= min_distance:
                        state.hsp_indices.append(middle_idx)
                        state.hsp_values.append(middle_asi)
                        detected_hsp = True
                        
                        # Update HSP slope if we have at least 2 HSPs
                        if len(state.hsp_indices) >= 2:
                            idx1, idx2 = state.hsp_indices[-2], state.hsp_indices[-1]
                            y1, y2 = state.hsp_values[-2], state.hsp_values[-1]
                            
                            if idx2 != idx1:  # Avoid division by zero
                                raw_slope = (y2 - y1) / (idx2 - idx1)
                                # Store the raw slope directly (not angle converted)
                                state.current_hsp_slope = raw_slope
                
                # LSP: middle bar lower than both neighbors  
                if middle_asi < left_asi and middle_asi < right_asi:
                    # Check minimum distance from last LSP
                    if not state.lsp_indices or (middle_idx - state.lsp_indices[-1]) >= min_distance:
                        state.lsp_indices.append(middle_idx)
                        state.lsp_values.append(middle_asi)
                        detected_lsp = True
                        
                        # Update LSP slope if we have at least 2 LSPs
                        if len(state.lsp_indices) >= 2:
                            idx1, idx2 = state.lsp_indices[-2], state.lsp_indices[-1]
                            y1, y2 = state.lsp_values[-2], state.lsp_values[-1]
                            
                            if idx2 != idx1:  # Avoid division by zero
                                raw_slope = (y2 - y1) / (idx2 - idx1)
                                # Store the raw slope directly (not angle converted)
                                state.current_lsp_slope = raw_slope
        
        return detected_hsp, detected_lsp
    
    def calculate_atr_usd_incremental(self, new_ohlc: Dict[str, float], state: PracticalInstrumentState, instrument: str) -> float:
        """Calculate ATR in USD using direct implementation"""
        # Add current OHLC to history
        # ATR calculation: True Range with EMA smoothing
        
        if len(state.ohlc_history) < 2:
            return 0.0
        
        # Get current and previous OHLC
        curr = new_ohlc
        prev = state.ohlc_history[-2]  # Previous bar
        
        # Calculate True Range
        tr1 = curr['high'] - curr['low']
        tr2 = abs(curr['high'] - prev['close'])
        tr3 = abs(curr['low'] - prev['close'])
        true_range = max(tr1, tr2, tr3)
        
        # Convert to USD using dynamic pip values
        from configs.instruments import get_pip_size, calculate_pip_value_usd
        pip_size = get_pip_size(instrument)
        
        # Calculate dynamic pip value based on current rate
        current_rate = curr['close']  # Use close price as current rate
        pip_value_usd = calculate_pip_value_usd(instrument, current_rate)
        
        # Convert to pips then to USD
        tr_pips = true_range / pip_size
        tr_usd = tr_pips * pip_value_usd
        
        # EMA smoothing (14-period)
        period = 14
        alpha = 2.0 / (period + 1)
        
        if len(state.atr_history) == 0:
            # First ATR value
            atr_usd = tr_usd
        else:
            # EMA calculation
            prev_atr = state.atr_history[-1]
            atr_usd = alpha * tr_usd + (1 - alpha) * prev_atr
        
        return atr_usd
    
    def calculate_adx_incremental(self, new_ohlc: Dict[str, float], state: PracticalInstrumentState) -> float:
        """Calculate ADX using proper implementation"""
        if len(state.ohlc_history) < 2:
            return 0.0
        
        # Get current and previous OHLC
        curr = new_ohlc
        prev = state.ohlc_history[-2]
        
        # Calculate directional movement
        dm_plus = max(curr['high'] - prev['high'], 0) if curr['high'] > prev['high'] else 0
        dm_minus = max(prev['low'] - curr['low'], 0) if curr['low'] < prev['low'] else 0
        
        # Calculate True Range (same as ATR)
        tr1 = curr['high'] - curr['low']
        tr2 = abs(curr['high'] - prev['close'])
        tr3 = abs(curr['low'] - prev['close'])
        true_range = max(tr1, tr2, tr3)
        
        # EMA smoothing (14-period)
        period = 14
        alpha = 2.0 / (period + 1)
        
        # Initialize or update DM and TR smoothed values
        if len(state.ohlc_history) == 2:  # First calculation
            state.dm_plus_smooth = dm_plus
            state.dm_minus_smooth = dm_minus
            state.tr_smooth = true_range
        else:
            # EMA update using proper smoothed values
            state.dm_plus_smooth = alpha * dm_plus + (1 - alpha) * state.dm_plus_smooth
            state.dm_minus_smooth = alpha * dm_minus + (1 - alpha) * state.dm_minus_smooth
            state.tr_smooth = alpha * true_range + (1 - alpha) * state.tr_smooth
        
        # Calculate DI+ and DI-
        di_plus = (state.dm_plus_smooth / state.tr_smooth * 100) if state.tr_smooth > 0 else 0
        di_minus = (state.dm_minus_smooth / state.tr_smooth * 100) if state.tr_smooth > 0 else 0
        
        # Store DI values for reference
        state.di_plus_history.append(di_plus)
        state.di_minus_history.append(di_minus)
        
        # Calculate DX
        di_sum = di_plus + di_minus
        di_diff = abs(di_plus - di_minus)
        dx = (di_diff / di_sum * 100) if di_sum > 0 else 0
        
        state.dx_history.append(dx)
        
        # Calculate ADX (EMA of DX)
        if len(state.dx_history) == 1:
            adx = dx
        else:
            prev_adx = state.adx_history[-1] if state.adx_history else dx
            adx = alpha * dx + (1 - alpha) * prev_adx
        
        return adx
    
    def calculate_next_hour_target(self, current_ohlc: Dict[str, float], next_ohlc: Dict[str, float], 
                                 instrument: str) -> float:
        """
        Calculate dollar-scaled pip movement for next hour (target for ML prediction)
        
        Args:
            current_ohlc: Current bar OHLC
            next_ohlc: Next bar OHLC (future data, for training only)
            instrument: FX pair name
            
        Returns:
            float: Dollar value of pip movement to next hour
        """
        from configs.instruments import get_pip_size, calculate_pip_value_usd
        
        current_close = current_ohlc['close']
        next_close = next_ohlc['close']
        
        # Calculate price change in native currency units
        price_diff = next_close - current_close
        
        # Convert to pips
        pip_size = get_pip_size(instrument)
        pip_movement = price_diff / pip_size
        
        # Convert pips to USD value using current rate
        pip_value_usd = calculate_pip_value_usd(instrument, current_close)
        dollar_target = pip_movement * pip_value_usd
        
        return dollar_target
    
    def calculate_percentile_scaling(self, current_value: float, history: List[float], window: int = 200) -> float:
        """Percentile scaling [0,1]"""
        if np.isnan(current_value) or current_value <= 0:
            return np.nan
        
        # Use rolling window
        relevant_history = history[-window:] if len(history) > window else history
        
        if len(relevant_history) < 10:  # Need minimum history
            return np.nan
        
        # Calculate percentile rank
        rank = np.sum(np.array(relevant_history) <= current_value)
        percentile = rank / len(relevant_history)
        
        # Cap at 99th percentile to prevent outliers
        return min(percentile, 0.99)
    
    def calculate_asi_incremental(self, new_ohlc: Dict[str, float], state: PracticalInstrumentState, instrument: str) -> float:
        """Calculate ASI incrementally - direct implementation"""
        if len(state.ohlc_history) < 2:
            asi_value = 0.0
        else:
            # Get current and previous OHLC
            curr = new_ohlc
            prev = state.ohlc_history[-2]
            
            # Calculate SI using simplified approach
            # SI = 50 * ((C - C_prev) + 0.5 * (C - O) + 0.25 * (C_prev - O_prev)) / R
            c = curr['close']
            c_prev = prev['close']
            o = curr['open']
            o_prev = prev['open']
            h = curr['high']
            l = curr['low']
            h_prev = prev['high']
            l_prev = prev['low']
            
            # Calculate R (Reference value)
            r1 = h - c_prev
            r2 = l - c_prev
            r3 = h - l
            
            R = max(abs(r1), abs(r2), r3)
            
            if R > 0:
                # Calculate SI
                si = 50 * ((c - c_prev) + 0.5 * (c - o) + 0.25 * (c_prev - o_prev)) / R
                
                # Add to ASI accumulation
                if len(state.asi_history) == 0:
                    asi_value = si
                else:
                    asi_value = state.asi_history[-1] + si
            else:
                # No movement, maintain previous ASI
                asi_value = state.asi_history[-1] if state.asi_history else 0.0
        
        return asi_value
    
    def calculate_csi_incremental(self, new_ohlc: Dict[str, float], state: PracticalInstrumentState, 
                                instrument: str) -> float:
        """
        Calculate CSI (Commodity Selection Index) incrementally using Wilder's formula.
        
        Original Wilder CSI Formula: CSI = ADX × ATR × (Volume / 1000)
        
        For FX markets without volume data, we use price velocity as volume proxy:
        CSI = ADX × ATR_USD × (Price_Velocity / 100)
        
        Where Price_Velocity = |Close - Open| * 1000 (normalized price movement)
        
        This provides Wilder's market selection tool for ranking instrument activity.
        NOTE: CSI is for reference only and SHALL NOT be used in the edge finding system.
        
        Args:
            new_ohlc: Current bar OHLC
            state: Instrument state with ADX and ATR history
            instrument: FX pair name
            
        Returns:
            float: CSI value for market selection ranking (reference only)
        """
        # Need both ADX and ATR for CSI calculation
        if len(state.adx_history) == 0 or len(state.atr_history) == 0:
            return np.nan
        
        # Get current ADX and ATR values
        current_adx = state.adx_history[-1]
        current_atr_usd = state.atr_history[-1]
        
        if np.isnan(current_adx) or np.isnan(current_atr_usd) or current_adx <= 0 or current_atr_usd <= 0:
            return np.nan
        
        # Calculate price velocity as volume proxy (Wilder CSI adaptation for FX)
        price_velocity = abs(new_ohlc['close'] - new_ohlc['open']) * 1000
        volume_proxy = price_velocity / 100  # Scale to approximate volume/1000 factor
        
        # Apply Wilder's CSI formula with volume proxy
        # CSI = ADX × ATR × (Volume_Proxy / scaling_factor)
        csi_value = current_adx * current_atr_usd * volume_proxy
        
        return csi_value

@dataclass
class PracticalMultiInstrumentState:
    """Multi-instrument state for practical processing"""
    instruments: Dict[str, PracticalInstrumentState] = field(default_factory=dict)
    context_matrix: Optional[np.ndarray] = None
    market_regime: int = 0
    
    def get_instrument_state(self, instrument: str) -> PracticalInstrumentState:
        """Get or create instrument state"""
        if instrument not in self.instruments:
            self.instruments[instrument] = PracticalInstrumentState()
        return self.instruments[instrument]

def update_practical_indicators(
    new_ohlc: Dict[str, float],
    multi_state: PracticalMultiInstrumentState,
    instrument: str,
    next_ohlc: Dict[str, float] = None
) -> Tuple[Dict[str, float], PracticalMultiInstrumentState]:
    """
    Practical incremental update using simple swing detection
    Designed to match batch practical method
    """
    calculator = PracticalIncrementalCalculator()
    
    # Get instrument-specific state
    state = multi_state.get_instrument_state(instrument)
    
    # Store OHLC
    state.ohlc_history.append(new_ohlc)
    state.bar_count += 1
    
    # 1. Calculate ASI
    asi_value = calculator.calculate_asi_incremental(new_ohlc, state, instrument)
    
    # 2. Practical swing detection
    detected_hsp, detected_lsp = calculator.detect_practical_swing_incremental(asi_value, state)
    
    # 3. Calculate ATR in USD
    atr_usd = calculator.calculate_atr_usd_incremental(new_ohlc, state, instrument)
    
    # Update ATR history and calculate volatility
    if atr_usd > 0:
        state.atr_history.append(atr_usd)
    volatility = calculator.calculate_percentile_scaling(atr_usd, state.atr_history) if atr_usd > 0 else np.nan
    
    # 4. Calculate ADX
    adx_value = calculator.calculate_adx_incremental(new_ohlc, state)
    
    # Update ADX history and calculate direction
    if adx_value > 0:
        state.adx_history.append(adx_value)
    direction = calculator.calculate_percentile_scaling(adx_value, state.adx_history) if adx_value > 0 else np.nan
    
    # 5. Calculate price change (log returns) with percentile scaling
    price_change_raw = np.nan
    price_change_scaled = np.nan
    
    if len(state.ohlc_history) >= 2:
        prev_close = state.ohlc_history[-2]['close']
        curr_close = state.ohlc_history[-1]['close']
        if prev_close > 0 and curr_close > 0:
            price_change_raw = np.log(curr_close / prev_close)
            
            # Store raw price change for percentile scaling
            if not np.isnan(price_change_raw):
                state.price_change_history.append(price_change_raw)
                
                # Apply percentile scaling [0,1] - same as volatility/direction
                price_change_scaled = calculator.calculate_percentile_scaling(
                    price_change_raw, state.price_change_history
                )
    
    # 6. Get current slopes (forward-filled)
    slope_high = state.current_hsp_slope
    slope_low = state.current_lsp_slope
    
    # 7. Calculate CSI (Commodity Selection Index) for market selection
    csi_value = calculator.calculate_csi_incremental(new_ohlc, state, instrument)
    if not np.isnan(csi_value):
        state.csi_history.append(csi_value)
        state.current_csi = csi_value
    
    # 8. Calculate target (next hour dollar-scaled pip movement) if future data available
    target_next_hour = np.nan
    if next_ohlc is not None:
        target_next_hour = calculator.calculate_next_hour_target(new_ohlc, next_ohlc, instrument)
        state.next_hour_targets.append(target_next_hour)
    
    return {
        'slope_high': slope_high,
        'slope_low': slope_low,
        'volatility': volatility,
        'direction': direction,
        'price_change': price_change_scaled,  # Feature: current bar price change [0,1]
        'csi': state.current_csi,            # CSI: market selection index (ADX * ATR_USD)
        'target_next_hour': target_next_hour   # Target: next hour pip movement in USD
    }, multi_state