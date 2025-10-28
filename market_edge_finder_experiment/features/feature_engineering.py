#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Market Edge Finder Experiment
Implements 4 causal indicators per instrument: slope_high, slope_low, volatility, direction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import zscore
import warnings
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys
from collections import deque

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.instruments import FX_INSTRUMENTS, get_pip_value

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering parameters"""
    volatility_window: int = 14  # ATR calculation window
    direction_window: int = 14  # ADX calculation window
    outlier_threshold: float = 3.0  # Z-score threshold for outlier removal
    normalization_window: int = 500  # Rolling window for normalization


class SwingPointDetector:
    """
    Detects swing highs and lows using strict causal methodology
    No lookahead bias - only uses confirmed pivots
    """
    
    def __init__(self, min_distance: int = 3, lookback: int = 20):
        """
        Initialize swing point detector
        
        Args:
            min_distance: Minimum bars between swing points
            lookback: Maximum bars to look back for swing detection
        """
        self.min_distance = min_distance
        self.lookback = lookback
    
    def find_swing_highs(self, high_prices: np.ndarray) -> List[Tuple[int, float]]:
        """
        Find swing highs in price series (causal - no lookahead)
        
        Args:
            high_prices: Array of high prices
            
        Returns:
            List of (index, price) tuples for swing highs
        """
        if len(high_prices) < self.min_distance * 2 + 1:
            return []
        
        swing_highs = []
        
        # Start from min_distance to ensure we can look back/forward
        for i in range(self.min_distance, len(high_prices) - self.min_distance):
            # Check if current point is higher than surrounding points
            is_swing_high = True
            
            # Check left side (confirmed)
            for j in range(max(0, i - self.min_distance), i):
                if high_prices[j] >= high_prices[i]:
                    is_swing_high = False
                    break
            
            # Check right side (only confirmed bars)
            if is_swing_high:
                for j in range(i + 1, min(len(high_prices), i + self.min_distance + 1)):
                    if high_prices[j] >= high_prices[i]:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                swing_highs.append((i, high_prices[i]))
        
        return swing_highs
    
    def find_swing_lows(self, low_prices: np.ndarray) -> List[Tuple[int, float]]:
        """
        Find swing lows in price series (causal - no lookahead)
        
        Args:
            low_prices: Array of low prices
            
        Returns:
            List of (index, price) tuples for swing lows
        """
        if len(low_prices) < self.min_distance * 2 + 1:
            return []
        
        swing_lows = []
        
        for i in range(self.min_distance, len(low_prices) - self.min_distance):
            is_swing_low = True
            
            # Check left side (confirmed)
            for j in range(max(0, i - self.min_distance), i):
                if low_prices[j] <= low_prices[i]:
                    is_swing_low = False
                    break
            
            # Check right side (only confirmed bars)
            if is_swing_low:
                for j in range(i + 1, min(len(low_prices), i + self.min_distance + 1)):
                    if low_prices[j] <= low_prices[i]:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                swing_lows.append((i, low_prices[i]))
        
        return swing_lows


@dataclass
class WilderSwingDetector:
    """
    Proper swing point detection based on new_muzero methodology.
    Uses a lookback window to ensure the swing point is the absolute highest/lowest
    in the window, not just higher/lower than immediate neighbors.
    """
    lookback: int = 2  # Bars before and after (total window = 5)
    asi_buffer: deque = field(default_factory=deque)
    last_hsp_value: Optional[float] = None
    last_lsp_value: Optional[float] = None
    
    def __post_init__(self):
        # Need lookback*2 + 1 total bars (e.g., 2+1+2 = 5 bars)
        self.asi_buffer = deque(maxlen=self.lookback * 2 + 1)
    
    def detect_swing_points(self, asi_value: float) -> Dict[str, Optional[float]]:
        """
        Detect HSP/LSP using proper swing detection methodology.
        
        Args:
            asi_value: Current ASI value
            
        Returns:
            Dict with 'hsp' and 'lsp' keys, values are ASI levels or None.
            The swing point detected is from the middle of the buffer (lookback bars ago).
        """
        self.asi_buffer.append(asi_value)
        
        result = {'hsp': None, 'lsp': None}
        
        # Need full window to detect swing points
        if len(self.asi_buffer) < self.lookback * 2 + 1:
            return result
        
        # Middle bar is the potential swing point
        mid_idx = self.lookback
        middle_asi = self.asi_buffer[mid_idx]
        
        # Check if middle bar is highest in entire window (HSP)
        is_hsp = all(
            middle_asi >= self.asi_buffer[i] 
            for i in range(len(self.asi_buffer)) if i != mid_idx
        )
        
        # Check if middle bar is lowest in entire window (LSP)
        is_lsp = all(
            middle_asi <= self.asi_buffer[i]
            for i in range(len(self.asi_buffer)) if i != mid_idx
        )
        
        # HSP: Only register if exceeded previous HSP
        if is_hsp and middle_asi > (self.last_hsp_value or float('-inf')):
            self.last_hsp_value = middle_asi
            result['hsp'] = middle_asi
        
        # LSP: Only register if exceeded (below) previous LSP  
        if is_lsp and middle_asi < (self.last_lsp_value or float('inf')):
            self.last_lsp_value = middle_asi
            result['lsp'] = middle_asi
        
        return result
    
    def get_swing_point_offset(self) -> int:
        """Return the offset where swing points are detected (lookback bars ago)."""
        return self.lookback


class TechnicalIndicators:
    """
    Causal technical indicators implementation
    All calculations are strictly non-lookahead
    """
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # Set first value to high - low (no previous close)
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using Wilder's smoothing
        atr = np.full(len(high), np.nan)
        atr[period-1] = np.mean(true_range[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
        
        return atr
    
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            ADX values
        """
        if len(high) < period * 2:
            return np.full(len(high), np.nan)
        
        # Calculate directional movements
        plus_dm = np.maximum(high - np.roll(high, 1), 0)
        minus_dm = np.maximum(np.roll(low, 1) - low, 0)
        
        # Remove first element (invalid due to shift)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Where plus_dm <= minus_dm, set plus_dm to 0
        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
        
        # Calculate True Range
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        # Smooth DM values
        plus_di = np.full(len(high), np.nan)
        minus_di = np.full(len(high), np.nan)
        
        for i in range(period, len(high)):
            if not np.isnan(atr[i]) and atr[i] != 0:
                plus_di[i] = 100 * np.mean(plus_dm[i-period+1:i+1]) / atr[i]
                minus_di[i] = 100 * np.mean(minus_dm[i-period+1:i+1]) / atr[i]
        
        # Calculate DX
        dx = np.full(len(high), np.nan)
        for i in range(period, len(high)):
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                sum_di = plus_di[i] + minus_di[i]
                if sum_di != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / sum_di
        
        # Calculate ADX (smoothed DX)
        adx = np.full(len(high), np.nan)
        start_idx = period * 2 - 1
        if start_idx < len(dx):
            adx[start_idx] = np.nanmean(dx[period:start_idx+1])
            
            for i in range(start_idx + 1, len(high)):
                if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    @staticmethod
    def normalize_ohlc_to_usd(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                             close: np.ndarray, instrument: str) -> tuple:
        """
        Convert raw OHLC prices to USD values per standard lot (100,000 units)
        
        Args:
            open_prices, high, low, close: Raw price arrays
            instrument: FX pair name (e.g., 'EUR_USD')
            
        Returns:
            tuple: (open_usd, high_usd, low_usd, close_usd) normalized to USD
        """
        pip_size, pip_value_usd = get_pip_value(instrument)
        
        # Convert prices to pips
        open_pips = open_prices / pip_size
        high_pips = high / pip_size  
        low_pips = low / pip_size
        close_pips = close / pip_size
        
        # Convert pips to USD values
        open_usd = open_pips * pip_value_usd
        high_usd = high_pips * pip_value_usd
        low_usd = low_pips * pip_value_usd
        close_usd = close_pips * pip_value_usd
        
        return open_usd, high_usd, low_usd, close_usd
    
    @staticmethod
    def calculate_atr_usd(high_usd: np.ndarray, low_usd: np.ndarray, close_usd: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range in USD terms
        
        Args:
            high_usd, low_usd, close_usd: USD-normalized price arrays
            period: ATR calculation period (default 14)
            
        Returns:
            ATR values in USD
        """
        if len(close_usd) < 2:
            return np.full(len(close_usd), np.nan)
        
        true_range = np.full(len(close_usd), np.nan)
        
        for i in range(1, len(close_usd)):
            # True Range components
            tr1 = high_usd[i] - low_usd[i]                    # H - L
            tr2 = abs(high_usd[i] - close_usd[i-1])          # |H - C_prev|
            tr3 = abs(low_usd[i] - close_usd[i-1])           # |L - C_prev|
            
            true_range[i] = max(tr1, tr2, tr3)
        
        # Calculate ATR using exponential moving average
        atr = np.full(len(close_usd), np.nan)
        alpha = 2.0 / (period + 1)  # EMA smoothing factor
        
        # Initialize with first period simple average
        first_valid = period
        if first_valid < len(true_range):
            atr[first_valid] = np.nanmean(true_range[1:first_valid+1])
            
            # Apply EMA for subsequent values
            for i in range(first_valid + 1, len(true_range)):
                if not np.isnan(true_range[i]) and not np.isnan(atr[i-1]):
                    atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
        
        return atr
    
    @staticmethod
    def calculate_asi_grok_spec(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, instrument: str, atr_period: int = 14, 
                               atr_multiplier: float = 3.0) -> tuple:
        """
        Calculate Accumulation Swing Index (ASI) per Grok specification
        
        Implements USD normalization, dynamic limit move, and proper Wilder formulas
        
        Args:
            open_prices, high, low, close: Raw price arrays
            instrument: FX pair name for USD normalization
            atr_period: Period for ATR calculation (default 14)
            atr_multiplier: Multiplier for limit move (default 3.0)
            
        Returns:
            tuple: (asi, atr_usd, si_values) - ASI values, ATR in USD, individual SI values
        """
        EPSILON = 1e-10
        
        if len(close) < 2:
            return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
        
        # Step 1: Normalize OHLC to USD values
        open_usd, high_usd, low_usd, close_usd = TechnicalIndicators.normalize_ohlc_to_usd(
            open_prices, high, low, close, instrument)
        
        # Step 2: Calculate ATR in USD terms
        atr_usd = TechnicalIndicators.calculate_atr_usd(high_usd, low_usd, close_usd, atr_period)
        
        # Initialize output arrays
        asi = np.full(len(close), 0.0, dtype=np.float64)
        si_values = np.full(len(close), 0.0, dtype=np.float64)
        asi[0] = 0.0
        si_values[0] = 0.0
        
        for i in range(1, len(close)):
            # Current bar values (subscript 2 in Grok notation)
            C2 = close_usd[i]     # Current close (USD)
            O2 = open_usd[i]      # Current open (USD)
            H2 = high_usd[i]      # Current high (USD)
            L2 = low_usd[i]       # Current low (USD)
            
            # Previous bar values (subscript 1 in Grok notation)
            C1 = close_usd[i-1]   # Previous close (USD)
            O1 = open_usd[i-1]    # Previous open (USD)
            
            # Step 3: Set dynamic limit move L = 3 × ATR
            if not np.isnan(atr_usd[i]) and atr_usd[i] > 0:
                L = atr_multiplier * atr_usd[i]
            else:
                # Fallback: use simple range if ATR not available
                recent_range = np.nanmean([abs(high_usd[j] - low_usd[j]) for j in range(max(0, i-10), i+1)])
                L = atr_multiplier * recent_range if not np.isnan(recent_range) else 1.0
            
            # Step 4: Calculate Swing Index (SI) per Grok specification
            
            # Numerator (N) - same as Wilder
            N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
            
            # Range (R) - Wilder's original specification (3 scenarios, take maximum)
            term1 = abs(H2 - C1) - 0.5 * abs(L2 - C1) + 0.25 * abs(C1 - O1)
            term2 = abs(L2 - C1) - 0.5 * abs(H2 - C1) + 0.25 * abs(C1 - O1)
            term3 = (H2 - L2) + 0.25 * abs(C1 - O1)
            
            R = max(term1, term2, term3)
            
            # Ensure R > 0
            if R <= 0:
                R = EPSILON
            
            # Limit Factor (K) - Wilder's specification
            K = max(abs(H2 - C1), abs(L2 - C1))
            
            # SI Calculation with Wilder's 50x multiplier (no capping)
            if L > EPSILON:
                SI = 50.0 * (N / R) * (K / L)
                
                # Round to nearest integer (no capping)
                SI = round(SI)
                # SI = max(-100, min(100, SI))  # COMMENTED OUT: No capping
            else:
                SI = 0
            
            si_values[i] = SI
            
            # Step 5: Accumulate swing index
            asi[i] = asi[i-1] + SI
        
        return asi, atr_usd, si_values
    
    @staticmethod
    def calculate_swing_index(open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, limit_move: float = 0.01) -> np.ndarray:
        """
        Calculate Swing Index (SI) - non-accumulated version of ASI
        
        Args:
            open_prices: Open prices
            high: High prices  
            low: Low prices
            close: Close prices
            limit_move: Maximum one-day move as percentage
            
        Returns:
            Swing Index values
        """
        if len(close) < 2:
            return np.full(len(close), np.nan)
        
        si = np.full(len(close), np.nan)
        si[0] = 0.0
        
        for i in range(1, len(close)):
            c = close[i]
            h = high[i]
            l = low[i]
            o = open_prices[i]
            
            c_prev = close[i-1]
            h_prev = high[i-1]
            l_prev = low[i-1]
            o_prev = open_prices[i-1]
            
            # True Range components
            tr1 = abs(h - c_prev)
            tr2 = abs(l - c_prev)
            tr3 = abs(h - l)
            
            tr = max(tr1, tr2, tr3)
            
            if tr == 0:
                si[i] = 0
                continue
            
            k = max(tr1, tr2)
            
            # R calculation
            if tr1 >= max(tr2, tr3):
                r = tr1 - 0.5 * tr2 + 0.25 * (c_prev - o_prev)
            elif tr2 >= max(tr1, tr3):
                r = tr2 - 0.5 * tr1 + 0.25 * (c_prev - o_prev)
            else:
                r = tr3 + 0.25 * (c_prev - o_prev)
            
            # Swing Index for this bar
            if k != 0 and limit_move != 0 and r != 0:
                si[i] = 50 * ((c - c_prev + 0.5 * (c - o) + 0.25 * (c_prev - o_prev)) / r) * (k / limit_move)
            else:
                si[i] = 0
        
        return si


class FXFeatureGenerator:
    """
    Main feature generator for FX instruments
    Produces ASI (Accumulation Swing Index) only
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature generator
        
        Args:
            config: Feature configuration parameters
        """
        self.config = config or FeatureConfig()
        
        logger.info(f"FXFeatureGenerator initialized with config: {self.config}")
    
    def calculate_swing_slopes(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                             timestamps: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate regression slopes of swing highs and lows
        
        Args:
            high_prices: High price array
            low_prices: Low price array
            timestamps: Optional timestamps for slope calculation
            
        Returns:
            Tuple of (slope_high, slope_low) arrays
        """
        if timestamps is None:
            timestamps = np.arange(len(high_prices))
        else:
            # Convert to numeric for regression
            timestamps = (timestamps - timestamps[0]).total_seconds() / 3600  # Hours
        
        slope_high = np.full(len(high_prices), np.nan)
        slope_low = np.full(len(low_prices), np.nan)
        
        # Calculate rolling slopes using confirmed swing points only
        for i in range(self.config.swing_lookback, len(high_prices)):
            # Get swing points in lookback window
            window_highs = high_prices[max(0, i - self.config.swing_lookback):i]
            window_lows = low_prices[max(0, i - self.config.swing_lookback):i]
            window_times = timestamps[max(0, i - self.config.swing_lookback):i]
            
            # Find swing highs in window
            swing_highs = self.swing_detector.find_swing_highs(window_highs)
            if len(swing_highs) >= 2:
                # Extract swing high prices and their timestamps
                swing_times = [window_times[idx] for idx, _ in swing_highs]
                swing_prices = [price for _, price in swing_highs]
                
                # Calculate linear regression slope
                if len(swing_times) >= 2:
                    slope, _, r_value, _, _ = stats.linregress(swing_times, swing_prices)
                    # Only use if correlation is reasonable
                    if abs(r_value) > 0.1:  # Minimum correlation threshold
                        slope_high[i] = slope
            
            # Find swing lows in window
            swing_lows = self.swing_detector.find_swing_lows(window_lows)
            if len(swing_lows) >= 2:
                swing_times = [window_times[idx] for idx, _ in swing_lows]
                swing_prices = [price for _, price in swing_lows]
                
                if len(swing_times) >= 2:
                    slope, _, r_value, _, _ = stats.linregress(swing_times, swing_prices)
                    if abs(r_value) > 0.1:
                        slope_low[i] = slope
        
        return slope_high, slope_low
    
    def calculate_normalized_asi(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, instrument: str) -> tuple:
        """
        Calculate normalized ASI using Grok specification
        
        Args:
            open_prices, high, low, close: Raw price arrays
            instrument: FX pair name for USD normalization
            
        Returns:
            tuple: (normalized_asi, angle_slopes) - ASI values and angle slopes
        """
        # Use the new Grok specification ASI calculation
        asi, atr_usd, si_values = TechnicalIndicators.calculate_asi_grok_spec(
            open_prices, high, low, close, instrument
        )
        
        # For now, angle_slopes is not used but kept for compatibility
        angle_slopes = np.full(len(asi), np.nan)
        
        return asi, angle_slopes
    
    def calculate_volatility(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, instrument: str = "EUR_USD") -> np.ndarray:
        """
        Calculate volatility using ATR/(pipsize*50) scaling
        
        Formula: ATR/(pipsize*50) where 1.0 = 50 pips
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            instrument: FX instrument pair for pip size calculation
            
        Returns:
            ATR/(pipsize*50) volatility array (linear scaling)
        """
        # Calculate ATR
        atr = TechnicalIndicators.calculate_atr(high, low, close, self.config.volatility_window)
        
        # Get pip size for the instrument
        pip_size, _ = get_pip_value(instrument)
        
        # Calculate ATR/(pipsize*50) scaling
        volatility = np.full(len(atr), np.nan)
        valid_mask = ~np.isnan(atr)
        volatility[valid_mask] = atr[valid_mask] / (pip_size * 50)
        
        return volatility
    
    def calculate_direction(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate directional movement intensity using simple ADX scaling
        
        Formula: ADX/100 for direct linear scaling
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Direction intensity array with ADX/100 scaling
        """
        # Calculate ADX with configured window
        adx = TechnicalIndicators.calculate_adx(high, low, close, self.config.direction_window)
        
        # Apply simple ADX/100 scaling (no hyperbolic transformation)
        direction = np.full(len(adx), np.nan)
        valid_mask = ~np.isnan(adx)
        direction[valid_mask] = adx[valid_mask] / 100.0
        
        return direction
    
    def calculate_log_returns(self, close: np.ndarray) -> np.ndarray:
        """
        Calculate log returns using log(c/c₋₁)
        
        Formula: log(close[t] / close[t-1]) = log(close[t]) - log(close[t-1])
        
        Args:
            close: Close prices
            
        Returns:
            Log returns array (first value is NaN)
        """
        log_returns = np.full(len(close), np.nan)
        
        # Calculate log returns starting from index 1
        for i in range(1, len(close)):
            if not (np.isnan(close[i]) or np.isnan(close[i-1])) and close[i-1] > 0:
                log_returns[i] = np.log(close[i] / close[i-1])
        
        return log_returns
    
    def calculate_atr_dollar_scaled(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, instrument: str) -> np.ndarray:
        """
        Calculate ATR with dollar scaling (USD per standard lot)
        
        Process:
        1. Convert OHLC to pips
        2. Convert pips to USD per standard lot (100,000 units)
        3. Calculate ATR in USD terms
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            instrument: FX instrument pair
            
        Returns:
            ATR array in USD per standard lot
        """
        # Get pip size and pip value for the instrument
        pip_size, pip_value_usd = get_pip_value(instrument)
        
        # Step 1: Convert OHLC to pips
        high_pips = high / pip_size
        low_pips = low / pip_size
        close_pips = close / pip_size
        
        # Step 2: Convert pips to USD (pip_value_usd is already per standard lot)
        high_usd = high_pips * pip_value_usd
        low_usd = low_pips * pip_value_usd
        close_usd = close_pips * pip_value_usd
        
        # Step 3: Calculate ATR in USD terms
        atr_usd = TechnicalIndicators.calculate_atr_usd(high_usd, low_usd, close_usd, self.config.volatility_window)
        
        return atr_usd
    
    def calculate_percentile_scaling(self, values: np.ndarray, window: int = 200) -> np.ndarray:
        """
        Apply percentile scaling (100 bins) with rolling window
        
        Maps values to [0,1] based on their percentile rank within rolling window
        
        Args:
            values: Input values to scale
            window: Rolling window size (default 200 bars)
            
        Returns:
            Scaled values in [0,1] range
        """
        scaled = np.full(len(values), np.nan)
        
        # Start scaling after we have enough data
        for i in range(window - 1, len(values)):
            if not np.isnan(values[i]):
                # Get window of historical values
                start_idx = max(0, i - window + 1)
                window_values = values[start_idx:i+1]
                
                # Remove NaN values
                valid_values = window_values[~np.isnan(window_values)]
                
                if len(valid_values) > 0:
                    # Calculate percentile rank
                    current_value = values[i]
                    count_below = np.sum(valid_values < current_value)
                    percentile_rank = count_below / len(valid_values)
                    
                    # Scale to [0,1] and cap at 1.0 (99th percentile protection)
                    scaled[i] = min(1.0, percentile_rank)
        
        return scaled
    
    def calculate_volatility_dollar_percentile(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, instrument: str) -> np.ndarray:
        """
        Calculate volatility using dollar-scaled ATR with percentile scaling
        
        Process per specification:
        1. Dollar-scale ATR (USD per standard lot)
        2. Apply percentile scaling (100 bins, 200-bar window)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            instrument: FX instrument pair
            
        Returns:
            Volatility scaled to [0,1] range
        """
        # Step 1: Calculate dollar-scaled ATR
        atr_usd = self.calculate_atr_dollar_scaled(high, low, close, instrument)
        
        # Step 2: Apply percentile scaling
        volatility_scaled = self.calculate_percentile_scaling(atr_usd, window=200)
        
        return volatility_scaled
    
    def calculate_direction_percentile(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate direction using ADX with percentile scaling
        
        Process per specification:
        1. Calculate raw ADX (0-100)
        2. Apply percentile scaling (100 bins, 200-bar window)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Direction scaled to [0,1] range
        """
        # Step 1: Calculate raw ADX
        adx = TechnicalIndicators.calculate_adx(high, low, close, self.config.direction_window)
        
        # Step 2: Apply percentile scaling  
        direction_scaled = self.calculate_percentile_scaling(adx, window=200)
        
        return direction_scaled
    
    def calculate_asi(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, limit_move: float = 1.0) -> np.ndarray:
        """
        Calculate Accumulation Swing Index using Wilder's exact specification.
        
        Args:
            open_prices: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            limit_move: Maximum one-day move (L=1.0, so K/L = K unscaled)
            
        Returns:
            Raw ASI array (no normalization per spec)
        """
        asi = TechnicalIndicators.calculate_asi(open_prices, high, low, close, limit_move)
        return asi
    
    def calculate_normalized_asi(self, open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, instrument: str, atr_period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Normalized ASI per specification:
        - Dollar normalization (USD per standard lot)
        - L = 3 × ATR adaptive limit
        - SI scaling with 100 multiplier and [-100, +100] capping
        - Angle normalization for regression slopes between swing points
        
        Args:
            open_prices: Open prices
            high: High prices  
            low: Low prices
            close: Close prices
            instrument: Instrument name (e.g., 'EUR_USD')
            atr_period: ATR calculation period
            
        Returns:
            Tuple of (normalized_asi, angle_normalized_slopes)
        """
        
        # Step 1: Get instrument parameters
        pip_size, pip_value = get_pip_value(instrument)
        
        # Step 2: Convert OHLC to USD per standard lot
        ohlc_pips = {
            'open': open_prices / pip_size,
            'high': high / pip_size, 
            'low': low / pip_size,
            'close': close / pip_size
        }
        
        ohlc_usd = {
            'open': ohlc_pips['open'] * pip_value,
            'high': ohlc_pips['high'] * pip_value,
            'low': ohlc_pips['low'] * pip_value, 
            'close': ohlc_pips['close'] * pip_value
        }
        
        # Step 3: Calculate ATR in USD
        atr_usd = self._calculate_atr_usd(ohlc_usd['high'], ohlc_usd['low'], ohlc_usd['close'], atr_period)
        
        # Step 4: Set adaptive limit move L = 3 × ATR
        limit_move = 3.0 * atr_usd
        
        # Step 5: Calculate normalized SI with USD values
        normalized_asi = self._calculate_si_usd(ohlc_usd, limit_move)
        
        # Step 6: Calculate angle-normalized slopes between swing points
        angle_slopes = self._calculate_angle_normalized_slopes(normalized_asi)
        
        return normalized_asi, angle_slopes
    
    def _calculate_atr_usd(self, high_usd: np.ndarray, low_usd: np.ndarray, close_usd: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range in USD terms."""
        n = len(high_usd)
        tr = np.full(n, np.nan)
        
        for i in range(1, n):
            tr1 = high_usd[i] - low_usd[i]  # High - Low
            tr2 = abs(high_usd[i] - close_usd[i-1])  # High - Prev Close  
            tr3 = abs(low_usd[i] - close_usd[i-1])   # Low - Prev Close
            tr[i] = max(tr1, tr2, tr3)
        
        # Calculate EMA of True Range
        atr = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        
        # Initialize with simple average of first 'period' values
        first_valid = period
        if first_valid < n:
            atr[first_valid] = np.nanmean(tr[1:first_valid+1])
            
            # EMA for subsequent values
            for i in range(first_valid + 1, n):
                if not np.isnan(tr[i]):
                    atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
                else:
                    atr[i] = atr[i-1]
        
        return atr
    
    def _calculate_si_usd(self, ohlc_usd: Dict[str, np.ndarray], limit_move: np.ndarray) -> np.ndarray:
        """Calculate Swing Index using USD-normalized values."""
        EPSILON = 1e-10
        
        close = ohlc_usd['close']
        open_prices = ohlc_usd['open']
        high = ohlc_usd['high'] 
        low = ohlc_usd['low']
        
        n = len(close)
        si = np.full(n, 0.0)
        asi = np.full(n, 0.0)
        
        for i in range(1, n):
            if np.isnan(limit_move[i]) or limit_move[i] <= 0:
                continue
                
            # Current and previous values (USD)
            C2, O2, H2, L2 = close[i], open_prices[i], high[i], low[i]
            C1, O1 = close[i-1], open_prices[i-1]
            
            # Numerator N
            N = (C2 - C1) + 0.5 * (C2 - O2) + 0.25 * (C1 - O1)
            
            # Range R (Wilder's formula)
            term1 = abs(H2 - C1)
            term2 = abs(L2 - C1)
            term3 = H2 - L2
            max_term = max(term1, term2, term3)
            min_term = min(term1, term2)
            R = max_term - 0.5 * min_term + 0.25 * abs(C1 - O1)
            R = max(R, EPSILON)  # Avoid division by zero
            
            # Limit factor K
            K = max(term1, term2)
            
            # SI calculation with 100 multiplier and capping
            L = limit_move[i]
            si_raw = 100.0 * (N / R) * (K / L) if L > 0 else 0.0
            
            # Round and cap SI to [-100, +100]
            si[i] = max(-100.0, min(100.0, round(si_raw)))
            
            # Accumulate ASI
            asi[i] = asi[i-1] + si[i]
        
        return asi
    
    def _calculate_angle_normalized_slopes(self, asi: np.ndarray) -> np.ndarray:
        """
        Calculate angle-normalized slopes between swing points.
        Maps arctan(slope) from (-90°, +90°) to (-1, +1).
        """
        n = len(asi)
        angle_slopes = np.full(n, np.nan)
        
        # Detect swing points (simplified - using existing method)
        swing_highs, swing_lows = self._detect_swing_points_simple(asi)
        
        # Combine and sort swing points
        all_swings = []
        for i in range(n):
            if swing_highs[i]:
                all_swings.append((i, asi[i], 'H'))
            elif swing_lows[i]:
                all_swings.append((i, asi[i], 'L'))
        
        # Calculate slopes between consecutive swing points
        for j in range(1, len(all_swings)):
            x1, y1, _ = all_swings[j-1]
            x2, y2, _ = all_swings[j]
            
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)  # USD per bar
                
                # Method 1: Angle-normalized (recommended)
                angle_rad = np.arctan(slope)  # (-π/2, π/2)
                angle_deg = np.degrees(angle_rad)  # (-90, +90)
                # Linear mapping: mappedValue = outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin))
                # Map (-90, +90) to (-1, +1)
                angle_norm = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
                
                # Fill in the slope for all bars between swing points
                for k in range(x1, min(x2 + 1, n)):
                    angle_slopes[k] = angle_norm
        
        return angle_slopes
    
    def _detect_swing_points_simple(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple 3-bar swing point detection for slope calculation."""
        n = len(values)
        highs = np.full(n, False)
        lows = np.full(n, False)
        
        for i in range(2, n):
            if i >= 3:
                middle_idx = i - 2
                left_idx = i - 3
                right_idx = i - 1
                
                # HSP: middle > both neighbors
                if (not np.isnan(values[middle_idx]) and 
                    not np.isnan(values[left_idx]) and 
                    not np.isnan(values[right_idx])):
                    
                    if (values[middle_idx] > values[left_idx] and 
                        values[middle_idx] > values[right_idx]):
                        highs[middle_idx] = True
                    
                    # LSP: middle < both neighbors
                    if (values[middle_idx] < values[left_idx] and 
                        values[middle_idx] < values[right_idx]):
                        lows[middle_idx] = True
        
        return highs, lows
    
    def _detect_swing_points_with_alternating_constraint(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect swing points with alternating constraint - NO LOOK-AHEAD BIAS.
        
        Returns:
            Tuple of (local_hsp, local_lsp, sig_hsp, sig_lsp) boolean arrays
        """
        n = len(values)
        
        # Step 1: Find local extremes - NO LOOK-AHEAD BIAS
        local_hsp = np.full(n, False)
        local_lsp = np.full(n, False)
        for i in range(2, n):  # Start from bar 2 to check pattern on bar i-2
            # Check if bar (i-2) was a 3-bar pattern using bars (i-3), (i-2), (i-1)
            if i >= 3:  # Need at least 3 bars for the pattern
                middle_idx = i - 2
                left_idx = i - 3  
                right_idx = i - 1
                
                # Skip NaN values
                if (np.isnan(values[middle_idx]) or np.isnan(values[left_idx]) or np.isnan(values[right_idx])):
                    continue
                
                # HSP: middle bar higher than both neighbors
                if values[middle_idx] > values[left_idx] and values[middle_idx] > values[right_idx]:
                    local_hsp[middle_idx] = True
                
                # LSP: middle bar lower than both neighbors  
                if values[middle_idx] < values[left_idx] and values[middle_idx] < values[right_idx]:
                    local_lsp[middle_idx] = True

        # Step 2: Apply alternating constraint - NO LOOK-AHEAD BIAS
        sig_hsp = np.full(n, False)
        sig_lsp = np.full(n, False)
        last_hsp_idx = None
        last_lsp_idx = None
        for i in range(n):  # Check all bars for local extremes
            if local_hsp[i]:
                if last_hsp_idx is None or (last_lsp_idx is not None and last_lsp_idx > last_hsp_idx):
                    sig_hsp[i] = True
                    last_hsp_idx = i
            if local_lsp[i]:
                if last_lsp_idx is None or (last_hsp_idx is not None and last_hsp_idx > last_lsp_idx):
                    sig_lsp[i] = True
                    last_lsp_idx = i
        
        return local_hsp, local_lsp, sig_hsp, sig_lsp
    
    def _detect_swing_points_wilder_proper(self, asi_values: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proper Wilder ASI swing detection with two-step process (batch version).
        
        Step 1: Candidate Detection - Local high/low in ASI (3-bar pattern)
        Step 2: Breakout Confirmation - Only confirmed after breaking opposite significant point
        
        Reference: New Concepts in Technical Trading Systems (1978), pp. 96-102
        
        Args:
            asi_values: ASI time series
            high_prices: High prices for HIP tracking
            low_prices: Low prices for LOP tracking
            
        Returns:
            sig_hsp: Boolean array marking confirmed significant HSPs
            sig_lsp: Boolean array marking confirmed significant LSPs
        """
        n = len(asi_values)
        sig_hsp = np.full(n, False)
        sig_lsp = np.full(n, False)
        
        # State tracking for Wilder method
        last_sig_hsp_idx = None
        last_sig_hsp_asi = None
        last_sig_hsp_price = None  # HIP (High Price)
        
        last_sig_lsp_idx = None
        last_sig_lsp_asi = None
        last_sig_lsp_price = None  # LOP (Low Price)
        
        pending_hsp_idx = None
        pending_hsp_asi = None
        pending_hsp_price = None
        
        pending_lsp_idx = None
        pending_lsp_asi = None
        pending_lsp_price = None
        
        # Process each bar sequentially (like incremental)
        for i in range(2, n):  # Start from bar 2 (need 3 bars for pattern)
            current_asi = asi_values[i]
            
            # Step 1: CANDIDATE DETECTION (3-bar pattern)
            # Check if the previous bar (i-1) was a local extreme using bars (i-2), (i-1), (i)
            if i >= 2:
                left_asi = asi_values[i-2]
                middle_asi = asi_values[i-1]
                right_asi = asi_values[i]
                middle_idx = i-1
                
                # Skip NaN values
                if not (np.isnan(left_asi) or np.isnan(middle_asi) or np.isnan(right_asi)):
                    
                    # HSP Candidate: middle bar higher than both neighbors
                    if middle_asi > left_asi and middle_asi > right_asi:
                        # Store as pending HSP candidate (needs breakout confirmation)
                        pending_hsp_idx = middle_idx
                        pending_hsp_asi = middle_asi
                        pending_hsp_price = high_prices[middle_idx]  # HIP
                    
                    # LSP Candidate: middle bar lower than both neighbors
                    if middle_asi < left_asi and middle_asi < right_asi:
                        # Store as pending LSP candidate (needs breakout confirmation)
                        pending_lsp_idx = middle_idx
                        pending_lsp_asi = middle_asi
                        pending_lsp_price = low_prices[middle_idx]  # LOP
            
            # Step 2: BREAKOUT CONFIRMATION
            
            # Confirm pending HSP: ASI must drop BELOW last significant LSP
            if (pending_hsp_idx is not None and 
                last_sig_lsp_asi is not None and
                current_asi < last_sig_lsp_asi):
                
                # Confirm the pending HSP as significant
                sig_hsp[pending_hsp_idx] = True
                last_sig_hsp_idx = pending_hsp_idx
                last_sig_hsp_asi = pending_hsp_asi
                last_sig_hsp_price = pending_hsp_price
                
                # Clear pending state
                pending_hsp_idx = None
                pending_hsp_asi = None
                pending_hsp_price = None
            
            # Confirm pending LSP: ASI must rise ABOVE last significant HSP
            if (pending_lsp_idx is not None and
                last_sig_hsp_asi is not None and
                current_asi > last_sig_hsp_asi):
                
                # Confirm the pending LSP as significant
                sig_lsp[pending_lsp_idx] = True
                last_sig_lsp_idx = pending_lsp_idx
                last_sig_lsp_asi = pending_lsp_asi
                last_sig_lsp_price = pending_lsp_price
                
                # Clear pending state
                pending_lsp_idx = None
                pending_lsp_asi = None
                pending_lsp_price = None
            
            # Special case: First swing point (no prior significant point to break)
            if last_sig_hsp_asi is None and last_sig_lsp_asi is None:
                # Allow first candidate to be confirmed immediately
                if pending_hsp_idx is not None:
                    sig_hsp[pending_hsp_idx] = True
                    last_sig_hsp_idx = pending_hsp_idx
                    last_sig_hsp_asi = pending_hsp_asi
                    last_sig_hsp_price = pending_hsp_price
                    pending_hsp_idx = None
                    pending_hsp_asi = None
                    pending_hsp_price = None
                    
                elif pending_lsp_idx is not None:
                    sig_lsp[pending_lsp_idx] = True
                    last_sig_lsp_idx = pending_lsp_idx
                    last_sig_lsp_asi = pending_lsp_asi
                    last_sig_lsp_price = pending_lsp_price
                    pending_lsp_idx = None
                    pending_lsp_asi = None
                    pending_lsp_price = None
        
        return sig_hsp, sig_lsp
    
    def _calculate_angle_slopes_between_last_two_hsp_lsp(self, values: np.ndarray, sig_hsp: np.ndarray, sig_lsp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate angle-normalized slopes for regression lines between:
        1. Last two HSPs (High Swing Points)
        2. Last two LSPs (Low Swing Points)
        Maps arctan(slope) from (-90°, +90°) to (-1, +1).
        Forward-fills previous values until new slopes are available.
        """
        n = len(values)
        hsp_angles = np.full(n, np.nan)
        lsp_angles = np.full(n, np.nan)
        
        # Get HSP indices and values
        hsp_indices = np.where(sig_hsp)[0]
        hsp_values = values[hsp_indices]
        
        # Get LSP indices and values  
        lsp_indices = np.where(sig_lsp)[0]
        lsp_values = values[lsp_indices]
        
        # Track last calculated angles for forward-filling
        last_hsp_angle = np.nan
        last_lsp_angle = np.nan
        
        # Calculate HSP regression angles with forward-filling
        for i in range(n):
            # Find last two HSPs before current bar
            valid_hsp_idx = hsp_indices[hsp_indices <= i]
            if len(valid_hsp_idx) >= 2:
                # Get last two HSPs
                idx1, idx2 = valid_hsp_idx[-2], valid_hsp_idx[-1]
                y1, y2 = values[idx1], values[idx2]
                
                if not np.isnan(y1) and not np.isnan(y2) and idx2 != idx1:
                    slope = (y2 - y1) / (idx2 - idx1)  # ASI units per bar
                    angle_rad = np.arctan(slope)  # (-π/2, π/2)
                    angle_deg = np.degrees(angle_rad)  # (-90, +90)
                    # Linear mapping: mappedValue = outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin))
                    # Map (-90, +90) to (-1, +1)
                    last_hsp_angle = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
            
            # Forward-fill: use last calculated angle if available
            if not np.isnan(last_hsp_angle):
                hsp_angles[i] = last_hsp_angle
        
        # Calculate LSP regression angles with forward-filling
        for i in range(n):
            # Find last two LSPs before current bar
            valid_lsp_idx = lsp_indices[lsp_indices <= i]
            if len(valid_lsp_idx) >= 2:
                # Get last two LSPs
                idx1, idx2 = valid_lsp_idx[-2], valid_lsp_idx[-1]
                y1, y2 = values[idx1], values[idx2]
                
                if not np.isnan(y1) and not np.isnan(y2) and idx2 != idx1:
                    slope = (y2 - y1) / (idx2 - idx1)  # ASI units per bar
                    angle_rad = np.arctan(slope)  # (-π/2, π/2)
                    angle_deg = np.degrees(angle_rad)  # (-90, +90)
                    # Linear mapping: mappedValue = outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin))
                    # Map (-90, +90) to (-1, +1)
                    last_lsp_angle = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
            
            # Forward-fill: use last calculated angle if available
            if not np.isnan(last_lsp_angle):
                lsp_angles[i] = last_lsp_angle
        
        return hsp_angles, lsp_angles
    
    def normalize_features(self, feature_array: np.ndarray, method: str = 'arctan') -> np.ndarray:
        """
        Normalize features using specified method
        
        Args:
            feature_array: Raw feature values
            method: Normalization method ('arctan', 'zscore', 'tanh')
            
        Returns:
            Normalized feature array
        """
        if method == 'arctan':
            # Apply z-score first, then arctan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z_scores = zscore(feature_array, nan_policy='omit')
                normalized = np.arctan(z_scores)
        
        elif method == 'zscore':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalized = zscore(feature_array, nan_policy='omit')
        
        elif method == 'tanh':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z_scores = zscore(feature_array, nan_policy='omit')
                normalized = np.tanh(z_scores)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def generate_features_single_instrument(self, df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        Generate all 5 features for a single instrument
        
        Features:
        1. slope_high: Linear regression slope of swing highs
        2. slope_low: Linear regression slope of swing lows  
        3. volatility: Dollar-scaled ATR with percentile scaling [0,1]
        4. direction: ADX with percentile scaling [0,1]
        5. log_returns: log(c/c₋₁) log returns
        
        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            instrument: Instrument name for logging
            
        Returns:
            DataFrame with 5 feature columns added
        """
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError(f"DataFrame missing required OHLC columns for {instrument}")
        
        logger.info(f"Generating features for {instrument}, {len(df)} bars")
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Extract price arrays
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        
        # Calculate new normalized ASI (per specification) - PRIMARY METHOD
        logger.debug(f"Calculating normalized ASI for {instrument}")
        try:
            normalized_asi, angle_slopes = self.calculate_normalized_asi(
                df['open'].values, high_prices, low_prices, close_prices, instrument
            )
        except Exception as e:
            logger.warning(f"Failed to calculate normalized ASI for {instrument}: {e}")
            # Fall back to NaN arrays if calculation fails
            normalized_asi = np.full(len(df), np.nan)
            angle_slopes = np.full(len(df), np.nan)
        
        # # COMMENTED OUT: Original ASI method (kept for reference)
        # logger.debug(f"Calculating original ASI for {instrument}")
        # asi = self.calculate_asi(df['open'].values, high_prices, low_prices, close_prices)
        
        # Apply swing point algorithm to normalized ASI (PRIMARY METHOD - WILDER PROPER)
        logger.debug(f"Calculating swing points for {instrument} using proper Wilder method")
        
        # Apply proper Wilder method to normalized ASI with price data for HIP/LOP tracking
        sig_hsp, sig_lsp = self._detect_swing_points_wilder_proper(normalized_asi, high_prices, low_prices)
        
        # Calculate angle slopes between last two HSPs and LSPs
        hsp_angles, lsp_angles = self._calculate_angle_slopes_between_last_two_hsp_lsp(normalized_asi, sig_hsp, sig_lsp)
        
        # Calculate direction indicator (ADX with percentile scaling)
        logger.debug(f"Calculating direction indicator for {instrument}")
        try:
            direction = self.calculate_direction_percentile(high_prices, low_prices, close_prices)
        except Exception as e:
            logger.warning(f"Failed to calculate direction for {instrument}: {e}")
            direction = np.full(len(df), np.nan)
        
        # Calculate volatility indicator (Dollar-scaled ATR with percentile scaling)
        logger.debug(f"Calculating volatility indicator for {instrument}")
        try:
            volatility = self.calculate_volatility_dollar_percentile(high_prices, low_prices, close_prices, instrument)
        except Exception as e:
            logger.warning(f"Failed to calculate volatility for {instrument}: {e}")
            volatility = np.full(len(df), np.nan)
        
        # Calculate log returns (5th indicator)
        logger.debug(f"Calculating log returns for {instrument}")
        try:
            log_returns = self.calculate_log_returns(close_prices)
        except Exception as e:
            logger.warning(f"Failed to calculate log returns for {instrument}: {e}")
            log_returns = np.full(len(df), np.nan)
        
        # # COMMENTED OUT: Original ASI swing point detection (kept for reference)
        # local_hsp_orig, local_lsp_orig, sig_hsp_orig, sig_lsp_orig = self._detect_swing_points_with_alternating_constraint(asi)
        # angle_slopes_original = self._calculate_angle_slopes_from_swing_points(asi, sig_hsp_orig, sig_lsp_orig)

        # Add all 5 indicator columns (complete feature set)
        result_df['asi'] = normalized_asi  # ASI: Wilder's accumulative swing index (USD normalized)
        # Note: Wilder method doesn't distinguish local vs significant - all detected points are significant
        result_df['local_hsp'] = sig_hsp  # For backward compatibility
        result_df['local_lsp'] = sig_lsp  # For backward compatibility
        result_df['sig_hsp'] = sig_hsp
        result_df['sig_lsp'] = sig_lsp
        result_df['slope_high'] = hsp_angles  # 1. Slope High: Regression slopes of swing highs
        result_df['slope_low'] = lsp_angles   # 2. Slope Low: Regression slopes of swing lows
        result_df['volatility'] = volatility  # 3. Volatility: Dollar-scaled ATR + percentile [0,1]
        result_df['direction'] = direction    # 4. Direction: ADX + percentile scaling [0,1]
        result_df['log_returns'] = log_returns # 5. Log Returns: log(c/c₋₁)
        
        # Forward-fill slope values once they exist (don't fill initial NaN values)
        result_df['slope_high'] = result_df['slope_high'].ffill()
        result_df['slope_low'] = result_df['slope_low'].ffill()
        
        # # COMMENTED OUT: Original ASI columns (kept for reference)
        # result_df[f'{instrument}_asi_original'] = asi
        # result_df[f'{instrument}_local_hsp_orig'] = local_hsp_orig
        # result_df[f'{instrument}_local_lsp_orig'] = local_lsp_orig
        # result_df[f'{instrument}_sig_hsp_orig'] = sig_hsp_orig
        # result_df[f'{instrument}_sig_lsp_orig'] = sig_lsp_orig
        # result_df[f'{instrument}_angle_slopes_original'] = angle_slopes_original
        
        # REMOVED: Redundant hsp/lsp value columns (use sig_hsp/sig_lsp + asi instead)
        
        # # COMMENTED OUT: Original ASI chart columns (kept for reference)
        # hsp_values_orig = np.where(sig_hsp_orig, asi, np.nan)
        # lsp_values_orig = np.where(sig_lsp_orig, asi, np.nan)
        # result_df[f'{instrument}_hsp_orig'] = hsp_values_orig
        # result_df[f'{instrument}_lsp_orig'] = lsp_values_orig
        
        # Log normalized ASI statistics (PRIMARY METHOD)
        valid_count = np.sum(~np.isnan(normalized_asi))
        hsp_count = np.sum(sig_hsp)
        lsp_count = np.sum(sig_lsp)
        
        if valid_count > 0:
            logger.debug(f"{instrument}_asi (normalized): {valid_count} valid values, "
                       f"range [{np.nanmin(normalized_asi):.1f}, {np.nanmax(normalized_asi):.1f}]")
            logger.debug(f"{instrument} swing points: {hsp_count} HSP, {lsp_count} LSP detected")
        
        # # COMMENTED OUT: Original ASI logging (kept for reference)
        # if valid_count_orig > 0:
        #     logger.debug(f"{instrument}_asi (original): {valid_count_orig} valid values, "
        #                f"range [{np.nanmin(asi):.3f}, {np.nanmax(asi):.3f}]")
        #     logger.debug(f"{instrument} original swing points: {hsp_count_orig} HSP, {lsp_count_orig} LSP detected")
        
        return result_df
    
    def validate_features(self, df: pd.DataFrame, instrument: str) -> Dict[str, bool]:
        """
        Validate generated features for quality and completeness
        
        Args:
            df: DataFrame with generated features
            instrument: Instrument name
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        feature_columns = [f'{instrument}_asi']
        
        for col in feature_columns:
            if col not in df.columns:
                validation_results[f'{col}_exists'] = False
                continue
            
            validation_results[f'{col}_exists'] = True
            
            # Check for reasonable coverage (at least 50% non-NaN)
            valid_ratio = df[col].notna().sum() / len(df)
            validation_results[f'{col}_coverage'] = valid_ratio >= 0.5
            
            # Check for reasonable range (normalized features should be roughly [-3, 3])
            if valid_ratio > 0:
                feature_range = df[col].max() - df[col].min()
                validation_results[f'{col}_range'] = 0.1 <= feature_range <= 10.0
                
                # Check for extreme outliers
                extreme_outliers = np.sum(np.abs(df[col]) > 5) / len(df)
                validation_results[f'{col}_outliers'] = extreme_outliers < 0.05
        
        return validation_results


def validate_ohlc_data(df: pd.DataFrame, instrument: str) -> bool:
    """
    Validate OHLC data quality before feature generation
    
    Args:
        df: OHLC DataFrame
        instrument: Instrument name for logging
        
    Returns:
        True if data is valid, False otherwise
    """
    if df.empty:
        logger.error(f"Empty DataFrame for {instrument}")
        return False
    
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns for {instrument}: {missing_cols}")
        return False
    
    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found in {instrument}: {nan_counts.to_dict()}")
    
    # Check OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    
    if invalid_ohlc > 0:
        logger.error(f"Invalid OHLC relationships in {instrument}: {invalid_ohlc} bars")
        return False
    
    # Check for zero or negative prices
    negative_prices = (df[required_cols] <= 0).any(axis=1).sum()
    if negative_prices > 0:
        logger.error(f"Zero or negative prices in {instrument}: {negative_prices} bars")
        return False
    
    logger.info(f"OHLC validation passed for {instrument}: {len(df)} bars")
    return True


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Generate realistic FX price data
    base_price = 1.3000
    returns = np.random.normal(0, 0.0001, 1000)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from closes
    opens = np.roll(close_prices, 1)
    opens[0] = base_price
    
    spreads = np.random.uniform(0.0001, 0.0005, 1000)
    highs = np.maximum(opens, close_prices) + spreads / 2
    lows = np.minimum(opens, close_prices) - spreads / 2
    volumes = np.random.randint(100, 1000, 1000)
    
    test_df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Test feature generation
    generator = FXFeatureGenerator()
    
    # Validate data first
    if validate_ohlc_data(test_df, 'EUR_USD'):
        # Generate features
        result = generator.generate_features_single_instrument(test_df, 'EUR_USD')
        
        # Validate features
        validation = generator.validate_features(result, 'EUR_USD')
        
        print("Feature validation results:")
        for check, passed in validation.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
        
        # Display sample features
        feature_cols = ['EUR_USD_slope_high', 'EUR_USD_slope_low', 'EUR_USD_volatility', 'EUR_USD_direction']
        print(f"\nSample features (last 10 rows):")
        print(result[feature_cols].tail(10))
        
        print(f"\nFeature statistics:")
        print(result[feature_cols].describe())
    else:
        print("❌ Data validation failed")