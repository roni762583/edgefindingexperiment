#!/usr/bin/env python3
"""
Advanced normalization techniques for Market Edge Finder features
Implements rolling normalization, regime-aware scaling, and cross-instrument alignment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.instruments import FX_INSTRUMENTS

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for normalization parameters"""
    rolling_window: int = 500  # Rolling window for adaptive normalization
    min_periods: int = 100  # Minimum periods for valid normalization
    outlier_threshold: float = 3.0  # Z-score threshold for outlier clipping
    quantile_range: Tuple[float, float] = (0.01, 0.99)  # Quantile range for robust scaling
    regime_sensitivity: float = 0.8  # Sensitivity to regime changes (0-1)
    cross_instrument_alpha: float = 0.1  # Weight for cross-instrument normalization


class RollingNormalizer:
    """
    Rolling window normalizer with adaptive parameters
    Prevents lookahead bias while maintaining responsiveness to market changes
    """
    
    def __init__(self, window: int = 500, min_periods: int = 100, method: str = 'zscore'):
        """
        Initialize rolling normalizer
        
        Args:
            window: Rolling window size
            min_periods: Minimum periods required
            method: Normalization method ('zscore', 'robust', 'quantile')
        """
        self.window = window
        self.min_periods = min_periods
        self.method = method
        
        if method not in ['zscore', 'robust', 'quantile']:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def normalize_series(self, series: pd.Series) -> pd.Series:
        """
        Apply rolling normalization to a pandas Series
        
        Args:
            series: Input time series
            
        Returns:
            Normalized series
        """
        if len(series) < self.min_periods:
            logger.warning(f"Series too short for normalization: {len(series)} < {self.min_periods}")
            return pd.Series(np.nan, index=series.index)
        
        normalized = pd.Series(np.nan, index=series.index)
        
        if self.method == 'zscore':
            # Rolling z-score normalization
            rolling_mean = series.rolling(window=self.window, min_periods=self.min_periods).mean()
            rolling_std = series.rolling(window=self.window, min_periods=self.min_periods).std()
            
            # Avoid division by zero
            valid_mask = (rolling_std > 1e-8) & rolling_mean.notna()
            normalized[valid_mask] = (series[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]
        
        elif self.method == 'robust':
            # Rolling robust normalization using median and MAD
            rolling_median = series.rolling(window=self.window, min_periods=self.min_periods).median()
            rolling_mad = series.rolling(window=self.window, min_periods=self.min_periods).apply(
                lambda x: np.median(np.abs(x - np.median(x))) if len(x) > 0 else np.nan
            )
            
            valid_mask = (rolling_mad > 1e-8) & rolling_median.notna()
            normalized[valid_mask] = (series[valid_mask] - rolling_median[valid_mask]) / (1.4826 * rolling_mad[valid_mask])
        
        elif self.method == 'quantile':
            # Rolling quantile normalization
            def quantile_normalize(window_data):
                if len(window_data) < self.min_periods:
                    return np.nan
                
                q01, q99 = np.percentile(window_data, [1, 99])
                q50 = np.median(window_data)
                
                if q99 - q01 < 1e-8:
                    return 0.0
                
                current_value = window_data.iloc[-1]
                return 2 * (current_value - q50) / (q99 - q01)
            
            normalized = series.rolling(window=self.window, min_periods=self.min_periods).apply(quantile_normalize)
        
        return normalized


class RegimeAwareNormalizer:
    """
    Normalizer that adapts to different market regimes
    Uses volatility and trend to detect regime changes
    """
    
    def __init__(self, sensitivity: float = 0.8, lookback: int = 100):
        """
        Initialize regime-aware normalizer
        
        Args:
            sensitivity: Regime change sensitivity (0-1)
            lookback: Lookback period for regime detection
        """
        self.sensitivity = sensitivity
        self.lookback = lookback
    
    def detect_regimes(self, price_data: pd.Series) -> pd.Series:
        """
        Detect market regimes based on volatility and trend
        
        Args:
            price_data: Price time series
            
        Returns:
            Series with regime indicators (0=low vol, 1=high vol, 2=trending, 3=choppy)
        """
        if len(price_data) < self.lookback:
            return pd.Series(0, index=price_data.index)
        
        # Calculate rolling statistics
        returns = price_data.pct_change()
        rolling_vol = returns.rolling(window=self.lookback).std()
        rolling_trend = price_data.rolling(window=self.lookback).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        # Classify regimes
        vol_threshold = rolling_vol.rolling(window=self.lookback * 2).median()
        trend_threshold = 0.3  # Correlation threshold for trending
        
        regimes = pd.Series(0, index=price_data.index)
        
        # High volatility regimes
        high_vol_mask = rolling_vol > (vol_threshold * (1 + self.sensitivity))
        regimes[high_vol_mask] = 1
        
        # Trending regimes
        trending_mask = np.abs(rolling_trend) > trend_threshold
        regimes[trending_mask & high_vol_mask] = 2
        regimes[trending_mask & ~high_vol_mask] = 2
        
        # Choppy regimes (high vol but no trend)
        choppy_mask = high_vol_mask & (np.abs(rolling_trend) <= trend_threshold)
        regimes[choppy_mask] = 3
        
        return regimes
    
    def normalize_by_regime(self, series: pd.Series, price_data: pd.Series) -> pd.Series:
        """
        Apply regime-specific normalization
        
        Args:
            series: Feature series to normalize
            price_data: Price data for regime detection
            
        Returns:
            Regime-normalized series
        """
        regimes = self.detect_regimes(price_data)
        normalized = pd.Series(np.nan, index=series.index)
        
        for regime in regimes.unique():
            if np.isnan(regime):
                continue
            
            regime_mask = regimes == regime
            regime_data = series[regime_mask]
            
            if len(regime_data) > 10:  # Minimum data points for normalization
                # Use different normalization parameters per regime
                if regime == 0:  # Low volatility
                    normalizer = RollingNormalizer(window=200, method='zscore')
                elif regime == 1:  # High volatility
                    normalizer = RollingNormalizer(window=100, method='robust')
                elif regime == 2:  # Trending
                    normalizer = RollingNormalizer(window=300, method='zscore')
                else:  # Choppy
                    normalizer = RollingNormalizer(window=50, method='quantile')
                
                normalized[regime_mask] = normalizer.normalize_series(regime_data)
        
        return normalized


class CrossInstrumentNormalizer:
    """
    Normalizer that considers cross-instrument relationships
    Helps align features across different FX pairs
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize cross-instrument normalizer
        
        Args:
            alpha: Weight for cross-instrument adjustment (0-1)
        """
        self.alpha = alpha
    
    def compute_cross_instrument_stats(self, features_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Compute cross-instrument statistics for each feature type
        
        Args:
            features_dict: Dictionary of feature DataFrames per instrument
            
        Returns:
            Dictionary of cross-instrument statistics
        """
        feature_types = ['slope_high', 'slope_low', 'volatility', 'direction']
        cross_stats = {}
        
        for feature_type in feature_types:
            # Collect all series for this feature type
            all_series = []
            for instrument, df in features_dict.items():
                col_name = f'{instrument}_{feature_type}'
                if col_name in df.columns:
                    all_series.append(df[col_name])
            
            if all_series:
                # Concatenate and compute global statistics
                combined_series = pd.concat(all_series, axis=0)
                cross_stats[f'{feature_type}_global_mean'] = combined_series.rolling(window=1000).mean()
                cross_stats[f'{feature_type}_global_std'] = combined_series.rolling(window=1000).std()
        
        return cross_stats
    
    def normalize_with_cross_adjustment(self, features_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply cross-instrument normalization adjustment
        
        Args:
            features_dict: Dictionary of feature DataFrames per instrument
            
        Returns:
            Dictionary of cross-adjusted feature DataFrames
        """
        cross_stats = self.compute_cross_instrument_stats(features_dict)
        adjusted_features = {}
        
        for instrument, df in features_dict.items():
            adjusted_df = df.copy()
            
            for col in df.columns:
                # Extract feature type from column name
                feature_type = col.split('_', 1)[1]  # Remove instrument prefix
                
                global_mean_key = f'{feature_type}_global_mean'
                global_std_key = f'{feature_type}_global_std'
                
                if global_mean_key in cross_stats and global_std_key in cross_stats:
                    # Apply cross-instrument adjustment
                    local_normalized = df[col]
                    global_mean = cross_stats[global_mean_key].reindex(df.index, method='ffill')
                    global_std = cross_stats[global_std_key].reindex(df.index, method='ffill')
                    
                    # Weighted combination of local and global normalization
                    valid_mask = global_std.notna() & (global_std > 1e-8)
                    global_normalized = (local_normalized - global_mean) / global_std
                    
                    adjusted_df.loc[valid_mask, col] = (
                        (1 - self.alpha) * local_normalized[valid_mask] + 
                        self.alpha * global_normalized[valid_mask]
                    )
            
            adjusted_features[instrument] = adjusted_df
        
        return adjusted_features


class AdvancedFeatureNormalizer:
    """
    Main normalizer class combining multiple normalization techniques
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize advanced feature normalizer
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        
        self.rolling_normalizer = RollingNormalizer(
            window=self.config.rolling_window,
            min_periods=self.config.min_periods
        )
        
        self.regime_normalizer = RegimeAwareNormalizer(
            sensitivity=self.config.regime_sensitivity
        )
        
        self.cross_normalizer = CrossInstrumentNormalizer(
            alpha=self.config.cross_instrument_alpha
        )
        
        logger.info(f"AdvancedFeatureNormalizer initialized with config: {self.config}")
    
    def clip_outliers(self, series: pd.Series, threshold: float = None) -> pd.Series:
        """
        Clip extreme outliers using z-score threshold
        
        Args:
            series: Input series
            threshold: Z-score threshold (default from config)
            
        Returns:
            Series with outliers clipped
        """
        threshold = threshold or self.config.outlier_threshold
        
        # Calculate rolling z-scores
        rolling_mean = series.rolling(window=self.config.rolling_window).mean()
        rolling_std = series.rolling(window=self.config.rolling_window).std()
        
        z_scores = np.abs((series - rolling_mean) / rolling_std)
        
        # Clip outliers
        clipped = series.copy()
        outlier_mask = z_scores > threshold
        
        # Replace outliers with threshold values
        upper_bound = rolling_mean + threshold * rolling_std
        lower_bound = rolling_mean - threshold * rolling_std
        
        clipped[outlier_mask & (series > rolling_mean)] = upper_bound[outlier_mask & (series > rolling_mean)]
        clipped[outlier_mask & (series < rolling_mean)] = lower_bound[outlier_mask & (series < rolling_mean)]
        
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            logger.debug(f"Clipped {outlier_count} outliers ({outlier_count/len(series):.1%})")
        
        return clipped
    
    def normalize_features_comprehensive(self, features_dict: Dict[str, pd.DataFrame], 
                                       price_data_dict: Optional[Dict[str, pd.Series]] = None) -> Dict[str, pd.DataFrame]:
        """
        Apply comprehensive normalization pipeline
        
        Args:
            features_dict: Dictionary of feature DataFrames per instrument
            price_data_dict: Optional price data for regime detection
            
        Returns:
            Dictionary of fully normalized feature DataFrames
        """
        logger.info(f"Starting comprehensive normalization for {len(features_dict)} instruments")
        
        # Step 1: Clip outliers
        logger.debug("Step 1: Clipping outliers")
        clipped_features = {}
        for instrument, df in features_dict.items():
            clipped_df = df.copy()
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32]:
                    clipped_df[col] = self.clip_outliers(df[col])
            clipped_features[instrument] = clipped_df
        
        # Step 2: Rolling normalization
        logger.debug("Step 2: Applying rolling normalization")
        rolling_normalized = {}
        for instrument, df in clipped_features.items():
            normalized_df = df.copy()
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32]:
                    normalized_df[col] = self.rolling_normalizer.normalize_series(df[col])
            rolling_normalized[instrument] = normalized_df
        
        # Step 3: Regime-aware adjustment (if price data available)
        if price_data_dict:
            logger.debug("Step 3: Applying regime-aware normalization")
            regime_normalized = {}
            for instrument, df in rolling_normalized.items():
                if instrument in price_data_dict:
                    regime_df = df.copy()
                    price_series = price_data_dict[instrument]
                    
                    for col in df.columns:
                        if df[col].dtype in [np.float64, np.float32]:
                            regime_df[col] = self.regime_normalizer.normalize_by_regime(
                                df[col], price_series
                            )
                    regime_normalized[instrument] = regime_df
                else:
                    regime_normalized[instrument] = df
        else:
            regime_normalized = rolling_normalized
        
        # Step 4: Cross-instrument adjustment
        logger.debug("Step 4: Applying cross-instrument normalization")
        final_normalized = self.cross_normalizer.normalize_with_cross_adjustment(regime_normalized)
        
        # Step 5: Final validation and statistics
        logger.debug("Step 5: Final validation")
        self._log_normalization_stats(final_normalized)
        
        return final_normalized
    
    def _log_normalization_stats(self, normalized_features: Dict[str, pd.DataFrame]):
        """Log normalization statistics for validation"""
        all_stats = []
        
        for instrument, df in normalized_features.items():
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32]:
                    series = df[col].dropna()
                    if len(series) > 0:
                        stats = {
                            'instrument': instrument,
                            'feature': col,
                            'mean': series.mean(),
                            'std': series.std(),
                            'min': series.min(),
                            'max': series.max(),
                            'coverage': len(series) / len(df)
                        }
                        all_stats.append(stats)
        
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            
            logger.info("Normalization Statistics Summary:")
            logger.info(f"  Mean of means: {stats_df['mean'].mean():.3f}")
            logger.info(f"  Mean of stds: {stats_df['std'].mean():.3f}")
            logger.info(f"  Average coverage: {stats_df['coverage'].mean():.1%}")
            logger.info(f"  Extreme values: [{stats_df['min'].min():.3f}, {stats_df['max'].max():.3f}]")


# Utility functions
def apply_arctan_normalization(series: pd.Series) -> pd.Series:
    """Apply arctan normalization after z-scoring"""
    z_scores = zscore(series.dropna())
    return pd.Series(np.arctan(z_scores), index=series.dropna().index).reindex(series.index)


def apply_tanh_normalization(series: pd.Series) -> pd.Series:
    """Apply tanh normalization after z-scoring"""
    z_scores = zscore(series.dropna())
    return pd.Series(np.tanh(z_scores), index=series.dropna().index).reindex(series.index)


def robust_quantile_normalization(series: pd.Series, quantile_range: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Apply robust quantile-based normalization"""
    q_low, q_high = quantile_range
    
    rolling_q_low = series.rolling(window=500).quantile(q_low)
    rolling_q_high = series.rolling(window=500).quantile(q_high)
    rolling_median = series.rolling(window=500).median()
    
    # Normalize to [-1, 1] range
    range_size = rolling_q_high - rolling_q_low
    valid_mask = range_size > 1e-8
    
    normalized = pd.Series(np.nan, index=series.index)
    normalized[valid_mask] = 2 * (series[valid_mask] - rolling_median[valid_mask]) / range_size[valid_mask]
    
    return normalized


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Create test features for multiple instruments
    test_features = {}
    test_prices = {}
    
    for instrument in ['EUR_USD', 'GBP_USD', 'USD_JPY']:
        # Generate features with different characteristics
        slope_high = np.random.normal(0, 0.1, len(dates)) + 0.01 * np.sin(np.arange(len(dates)) / 100)
        slope_low = np.random.normal(0, 0.08, len(dates)) - 0.01 * np.cos(np.arange(len(dates)) / 80)
        volatility = np.random.exponential(0.5, len(dates))
        direction = np.random.normal(0.5, 0.3, len(dates))
        
        # Add some outliers
        outlier_indices = np.random.choice(len(dates), size=10, replace=False)
        slope_high[outlier_indices] *= 5
        
        test_features[instrument] = pd.DataFrame({
            f'{instrument}_slope_high': slope_high,
            f'{instrument}_slope_low': slope_low,
            f'{instrument}_volatility': volatility,
            f'{instrument}_direction': direction
        }, index=dates)
        
        # Generate corresponding price data
        base_price = 1.3 if 'JPY' not in instrument else 110
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        test_prices[instrument] = pd.Series(prices, index=dates)
    
    # Test comprehensive normalization
    normalizer = AdvancedFeatureNormalizer()
    
    print("Testing comprehensive feature normalization...")
    normalized_features = normalizer.normalize_features_comprehensive(test_features, test_prices)
    
    print("\n=== Normalization Results ===")
    for instrument, df in normalized_features.items():
        print(f"\n{instrument}:")
        print(df.describe())
        
        # Check for outliers
        for col in df.columns:
            outliers = np.sum(np.abs(df[col]) > 3)
            print(f"  {col}: {outliers} outliers (>{3} std)")
    
    # Test individual normalization methods
    test_series = test_features['EUR_USD']['EUR_USD_slope_high']
    
    print(f"\n=== Individual Normalization Methods ===")
    print(f"Original: mean={test_series.mean():.3f}, std={test_series.std():.3f}")
    
    # Rolling normalization
    rolling_norm = RollingNormalizer(window=200, method='zscore')
    normalized = rolling_norm.normalize_series(test_series)
    print(f"Rolling Z-score: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Arctan normalization
    arctan_norm = apply_arctan_normalization(test_series)
    print(f"Arctan: mean={arctan_norm.mean():.3f}, std={arctan_norm.std():.3f}, range=[{arctan_norm.min():.3f}, {arctan_norm.max():.3f}]")
    
    # Robust quantile normalization
    quantile_norm = robust_quantile_normalization(test_series)
    print(f"Quantile: mean={quantile_norm.mean():.3f}, std={quantile_norm.std():.3f}, range=[{quantile_norm.min():.3f}, {quantile_norm.max():.3f}]")