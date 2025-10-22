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
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.instruments import FX_INSTRUMENTS, get_pip_value

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering parameters"""
    swing_lookback: int = 20  # Bars to look back for swing highs/lows
    volatility_window: int = 14  # ATR calculation window
    direction_window: int = 14  # ADX calculation window
    min_swing_distance: int = 3  # Minimum bars between swing points
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


class FXFeatureGenerator:
    """
    Main feature generator for FX instruments
    Produces 4 causal indicators per instrument: slope_high, slope_low, volatility, direction
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature generator
        
        Args:
            config: Feature configuration parameters
        """
        self.config = config or FeatureConfig()
        self.swing_detector = SwingPointDetector(
            min_distance=self.config.min_swing_distance,
            lookback=self.config.swing_lookback
        )
        
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
    
    def calculate_volatility(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate normalized volatility using ATR
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Normalized volatility array
        """
        atr = TechnicalIndicators.calculate_atr(high, low, close, self.config.volatility_window)
        
        # Normalize by price to get percentage volatility
        volatility = np.where(close > 0, atr / close, np.nan)
        
        return volatility
    
    def calculate_direction(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate directional movement intensity using ADX
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Direction intensity array
        """
        adx = TechnicalIndicators.calculate_adx(high, low, close, self.config.direction_window)
        
        # Normalize ADX to 0-1 range
        direction = adx / 100.0
        
        return direction
    
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
        Generate all 4 features for a single instrument
        
        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            instrument: Instrument name for logging
            
        Returns:
            DataFrame with 4 feature columns added
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
        
        # Calculate raw features
        logger.debug(f"Calculating swing slopes for {instrument}")
        slope_high, slope_low = self.calculate_swing_slopes(high_prices, low_prices, timestamps)
        
        logger.debug(f"Calculating volatility for {instrument}")
        volatility = self.calculate_volatility(high_prices, low_prices, close_prices)
        
        logger.debug(f"Calculating direction for {instrument}")
        direction = self.calculate_direction(high_prices, low_prices, close_prices)
        
        # Normalize features
        logger.debug(f"Normalizing features for {instrument}")
        slope_high_norm = self.normalize_features(slope_high, 'arctan')
        slope_low_norm = self.normalize_features(slope_low, 'arctan')
        volatility_norm = self.normalize_features(volatility, 'zscore')
        direction_norm = self.normalize_features(direction, 'zscore')
        
        # Add to result DataFrame
        result_df[f'{instrument}_slope_high'] = slope_high_norm
        result_df[f'{instrument}_slope_low'] = slope_low_norm
        result_df[f'{instrument}_volatility'] = volatility_norm
        result_df[f'{instrument}_direction'] = direction_norm
        
        # Log feature statistics
        for feature_name, feature_data in [
            ('slope_high', slope_high_norm),
            ('slope_low', slope_low_norm),
            ('volatility', volatility_norm),
            ('direction', direction_norm)
        ]:
            valid_count = np.sum(~np.isnan(feature_data))
            if valid_count > 0:
                logger.debug(f"{instrument}_{feature_name}: {valid_count} valid values, "
                           f"range [{np.nanmin(feature_data):.3f}, {np.nanmax(feature_data):.3f}]")
        
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
        
        feature_columns = [f'{instrument}_slope_high', f'{instrument}_slope_low', 
                          f'{instrument}_volatility', f'{instrument}_direction']
        
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