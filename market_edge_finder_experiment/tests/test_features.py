"""
Test suite for feature engineering components.

Tests for feature generation, normalization, and multiprocessing pipeline.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Import feature components
from ..features.feature_engineering import FeatureEngineer, FeatureConfig
from ..features.normalization import FeatureNormalizer, NormalizationConfig
from ..features.multiprocessor import FeatureMultiprocessor

# Suppress warnings for tests
warnings.filterwarnings('ignore')


class TestFeatureEngineer(unittest.TestCase):
    """Test suite for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FeatureConfig(
            atr_period=14,
            adx_period=14,
            volatility_window=24,
            direction_window=6,
            swing_lookback=12
        )
        self.engineer = FeatureEngineer(config=self.config)
        
        # Create test data
        np.random.seed(42)  # For reproducible tests
        self.test_data = self._create_test_ohlc_data()
    
    def _create_test_ohlc_data(self, n_rows=100):
        """Create synthetic OHLC data for testing."""
        dates = pd.date_range('2023-01-01', periods=n_rows, freq='H')
        
        # Generate realistic price movements
        price_base = 1.1000
        returns = np.random.normal(0, 0.001, n_rows)
        prices = [price_base]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC from prices
        data = []
        for i, price in enumerate(prices):
            volatility = np.random.uniform(0.0005, 0.002)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test FeatureEngineer initializes correctly."""
        self.assertIsInstance(self.engineer, FeatureEngineer)
        self.assertEqual(self.engineer.config.atr_period, 14)
        self.assertEqual(self.engineer.config.adx_period, 14)
    
    def test_feature_generation_output_structure(self):
        """Test that feature generation produces correct output structure."""
        features = self.engineer.generate_features_single_instrument(
            self.test_data, 'EUR_USD'
        )
        
        # Check that output is a DataFrame
        self.assertIsInstance(features, pd.DataFrame)
        
        # Check required columns are present
        required_columns = ['slope_high', 'slope_low', 'volatility', 'direction']
        for col in required_columns:
            self.assertIn(col, features.columns)
        
        # Check timestamp is preserved
        self.assertIn('timestamp', features.columns)
        
        # Check data types
        for col in required_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(features[col]))
    
    def test_swing_slope_calculation(self):
        """Test swing slope calculation produces reasonable values."""
        prices = self.test_data['high'].values
        timestamps = self.test_data['timestamp'].values
        
        slopes = self.engineer.calculate_swing_slopes(
            prices, prices, timestamps  # Using same for high/low for simplicity
        )
        
        slope_high, slope_low = slopes
        
        # Check output shapes
        self.assertEqual(len(slope_high), len(prices))
        self.assertEqual(len(slope_low), len(prices))
        
        # Check for reasonable values (not all NaN or infinite)
        self.assertFalse(np.all(np.isnan(slope_high)))
        self.assertFalse(np.all(np.isnan(slope_low)))
        self.assertTrue(np.all(np.isfinite(slope_high[~np.isnan(slope_high)])))
        self.assertTrue(np.all(np.isfinite(slope_low[~np.isnan(slope_low)])))
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        high_prices = self.test_data['high'].values
        low_prices = self.test_data['low'].values
        close_prices = self.test_data['close'].values
        
        volatility = self.engineer.calculate_volatility(
            high_prices, low_prices, close_prices
        )
        
        # Check output shape
        self.assertEqual(len(volatility), len(high_prices))
        
        # Volatility should be non-negative
        valid_vol = volatility[~np.isnan(volatility)]
        self.assertTrue(np.all(valid_vol >= 0))
        
        # Should have some variation
        self.assertGreater(np.std(valid_vol), 0)
    
    def test_direction_calculation(self):
        """Test direction calculation."""
        high_prices = self.test_data['high'].values
        low_prices = self.test_data['low'].values
        close_prices = self.test_data['close'].values
        
        direction = self.engineer.calculate_direction(
            high_prices, low_prices, close_prices
        )
        
        # Check output shape
        self.assertEqual(len(direction), len(high_prices))
        
        # Direction should be between -1 and 1
        valid_dir = direction[~np.isnan(direction)]
        self.assertTrue(np.all(valid_dir >= -1))
        self.assertTrue(np.all(valid_dir <= 1))
    
    def test_atr_calculation(self):
        """Test ATR (Average True Range) calculation."""
        high_prices = self.test_data['high'].values
        low_prices = self.test_data['low'].values
        close_prices = self.test_data['close'].values
        
        atr = self.engineer.calculate_atr(high_prices, low_prices, close_prices)
        
        # Check output shape
        self.assertEqual(len(atr), len(high_prices))
        
        # ATR should be non-negative
        valid_atr = atr[~np.isnan(atr)]
        self.assertTrue(np.all(valid_atr >= 0))
    
    def test_adx_calculation(self):
        """Test ADX (Average Directional Index) calculation."""
        high_prices = self.test_data['high'].values
        low_prices = self.test_data['low'].values
        close_prices = self.test_data['close'].values
        
        adx = self.engineer.calculate_adx(high_prices, low_prices, close_prices)
        
        # Check output shape
        self.assertEqual(len(adx), len(high_prices))
        
        # ADX should be between 0 and 100
        valid_adx = adx[~np.isnan(adx)]
        self.assertTrue(np.all(valid_adx >= 0))
        self.assertTrue(np.all(valid_adx <= 100))
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple calls."""
        features1 = self.engineer.generate_features_single_instrument(
            self.test_data, 'EUR_USD'
        )
        features2 = self.engineer.generate_features_single_instrument(
            self.test_data, 'EUR_USD'
        )
        
        # Features should be identical for same input
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_missing_data_handling(self):
        """Test handling of missing data in input."""
        # Create data with missing values
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[10:15, 'high'] = np.nan
        data_with_missing.loc[20:25, 'close'] = np.nan
        
        # Should not raise exception
        features = self.engineer.generate_features_single_instrument(
            data_with_missing, 'EUR_USD'
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(data_with_missing))
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Very small dataset
        small_data = self.test_data.head(5)
        
        # Should still work but may have many NaN values
        features = self.engineer.generate_features_single_instrument(
            small_data, 'EUR_USD'
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(small_data))


class TestFeatureNormalizer(unittest.TestCase):
    """Test suite for FeatureNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NormalizationConfig(
            method='rolling_zscore',
            window=24,
            outlier_method='iqr',
            outlier_threshold=2.0,
            regime_window=168
        )
        self.normalizer = FeatureNormalizer(config=self.config)
        
        # Create test feature data
        np.random.seed(42)
        self.test_features = self._create_test_features()
    
    def _create_test_features(self, n_rows=200):
        """Create synthetic feature data for testing."""
        dates = pd.date_range('2023-01-01', periods=n_rows, freq='H')
        
        data = {
            'timestamp': dates,
            'slope_high': np.random.normal(0.1, 0.5, n_rows),
            'slope_low': np.random.normal(-0.1, 0.4, n_rows),
            'volatility': np.random.exponential(0.02, n_rows),
            'direction': np.random.uniform(-1, 1, n_rows)
        }
        
        # Add some outliers
        outlier_indices = np.random.choice(n_rows, size=5, replace=False)
        data['slope_high'][outlier_indices] = np.random.normal(0, 5, 5)
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test FeatureNormalizer initializes correctly."""
        self.assertIsInstance(self.normalizer, FeatureNormalizer)
        self.assertEqual(self.normalizer.config.method, 'rolling_zscore')
        self.assertEqual(self.normalizer.config.window, 24)
    
    def test_outlier_clipping(self):
        """Test outlier clipping functionality."""
        original_data = self.test_features[['slope_high', 'slope_low', 'volatility', 'direction']].copy()
        clipped_data = self.normalizer._clip_outliers(original_data)
        
        # Check that extreme outliers are reduced
        for col in original_data.columns:
            original_range = original_data[col].max() - original_data[col].min()
            clipped_range = clipped_data[col].max() - clipped_data[col].min()
            # Clipped range should generally be smaller or equal
            self.assertLessEqual(clipped_range, original_range * 1.1)  # Allow small tolerance
    
    def test_rolling_normalization(self):
        """Test rolling normalization."""
        feature_cols = ['slope_high', 'slope_low', 'volatility', 'direction']
        normalized = self.normalizer._apply_rolling_normalization(
            self.test_features[feature_cols]
        )
        
        # Check output shape
        self.assertEqual(normalized.shape, self.test_features[feature_cols].shape)
        
        # Check that normalization reduces variance in later periods
        window = self.config.window
        if len(normalized) > window:
            later_period = normalized.iloc[window:]
            for col in feature_cols:
                std_dev = later_period[col].std()
                # Should be approximately normalized (std close to 1)
                self.assertLess(abs(std_dev - 1.0), 2.0)  # Allow some tolerance
    
    def test_normalize_single_instrument(self):
        """Test normalization for single instrument."""
        normalized = self.normalizer.normalize_single_instrument(
            self.test_features, 'EUR_USD'
        )
        
        # Check output structure
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertEqual(normalized.shape, self.test_features.shape)
        
        # Check that timestamp is preserved
        pd.testing.assert_series_equal(
            normalized['timestamp'], 
            self.test_features['timestamp']
        )
    
    def test_comprehensive_normalization(self):
        """Test comprehensive normalization with multiple instruments."""
        # Create feature dict for multiple instruments
        features_dict = {
            'EUR_USD': self.test_features.copy(),
            'GBP_USD': self.test_features.copy() * 1.1,  # Slightly different scaling
            'USD_JPY': self.test_features.copy() * 0.9
        }
        
        normalized_dict = self.normalizer.normalize_features_comprehensive(features_dict)
        
        # Check that all instruments are present
        self.assertEqual(set(normalized_dict.keys()), set(features_dict.keys()))
        
        # Check that shapes are preserved
        for instrument in features_dict:
            self.assertEqual(
                normalized_dict[instrument].shape,
                features_dict[instrument].shape
            )
    
    def test_cross_instrument_alignment(self):
        """Test cross-instrument alignment functionality."""
        # Create features with different scaling
        features_dict = {
            'EUR_USD': self.test_features.copy(),
            'GBP_USD': self.test_features.copy() * 2.0,  # Double scaling
        }
        
        aligned_features = self.normalizer._apply_cross_instrument_alignment(features_dict)
        
        # After alignment, the relative scaling should be reduced
        eur_std = aligned_features['EUR_USD'][['slope_high', 'slope_low', 'volatility', 'direction']].std()
        gbp_std = aligned_features['GBP_USD'][['slope_high', 'slope_low', 'volatility', 'direction']].std()
        
        # Standard deviations should be more similar after alignment
        std_ratio = (gbp_std / eur_std).mean()
        self.assertLess(abs(std_ratio - 1.0), 1.0)  # Should be closer to 1
    
    def test_regime_adjustment(self):
        """Test regime-aware normalization adjustment."""
        # Create price data with regime changes
        price_data = pd.Series(np.random.randn(len(self.test_features)).cumsum() + 100)
        
        adjusted_features = self.normalizer._apply_regime_adjustment(
            self.test_features[['slope_high', 'slope_low', 'volatility', 'direction']],
            price_data
        )
        
        # Should return DataFrame with same shape
        self.assertEqual(
            adjusted_features.shape,
            self.test_features[['slope_high', 'slope_low', 'volatility', 'direction']].shape
        )
    
    def test_state_persistence(self):
        """Test that normalizer can save and load state."""
        # Normalize some data to build state
        self.normalizer.normalize_single_instrument(self.test_features, 'EUR_USD')
        
        # Get state
        state = self.normalizer.get_state()
        self.assertIsInstance(state, dict)
        
        # Create new normalizer and load state
        new_normalizer = FeatureNormalizer(config=self.config)
        new_normalizer.load_state(state)
        
        # Should produce same results
        result1 = self.normalizer.normalize_single_instrument(self.test_features, 'EUR_USD')
        result2 = new_normalizer.normalize_single_instrument(self.test_features, 'EUR_USD')
        
        pd.testing.assert_frame_equal(result1, result2)


class TestFeatureMultiprocessor(unittest.TestCase):
    """Test suite for FeatureMultiprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.multiprocessor = FeatureMultiprocessor(max_workers=2)
        
        # Create test data for multiple instruments
        np.random.seed(42)
        self.test_data_dict = {}
        
        instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        for instrument in instruments:
            self.test_data_dict[instrument] = self._create_test_ohlc_data()
    
    def _create_test_ohlc_data(self, n_rows=100):
        """Create synthetic OHLC data for testing."""
        dates = pd.date_range('2023-01-01', periods=n_rows, freq='H')
        
        price_base = np.random.uniform(0.5, 2.0)
        returns = np.random.normal(0, 0.001, n_rows)
        prices = [price_base]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, price in enumerate(prices):
            volatility = np.random.uniform(0.0005, 0.002)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test FeatureMultiprocessor initializes correctly."""
        self.assertIsInstance(self.multiprocessor, FeatureMultiprocessor)
        self.assertEqual(self.multiprocessor.max_workers, 2)
    
    def test_parallel_feature_computation(self):
        """Test parallel feature computation."""
        features_dict, stats_dict = self.multiprocessor.compute_features_parallel(
            self.test_data_dict
        )
        
        # Check that all instruments are processed
        self.assertEqual(set(features_dict.keys()), set(self.test_data_dict.keys()))
        self.assertEqual(set(stats_dict.keys()), set(self.test_data_dict.keys()))
        
        # Check feature structure
        for instrument, features in features_dict.items():
            self.assertIsInstance(features, pd.DataFrame)
            
            # Check required columns
            required_columns = ['slope_high', 'slope_low', 'volatility', 'direction']
            for col in required_columns:
                self.assertIn(col, features.columns)
        
        # Check statistics structure
        for instrument, stats in stats_dict.items():
            self.assertIsInstance(stats, dict)
            self.assertIn('processing_time', stats)
            self.assertIn('feature_count', stats)
    
    def test_error_handling_in_parallel_processing(self):
        """Test error handling when some instruments fail."""
        # Create problematic data
        bad_data_dict = self.test_data_dict.copy()
        bad_data_dict['BAD_INSTRUMENT'] = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
            'open': [None] * 5,  # All None values
            'high': [None] * 5,
            'low': [None] * 5,
            'close': [None] * 5,
            'volume': [None] * 5
        })
        
        # Should handle errors gracefully
        features_dict, stats_dict = self.multiprocessor.compute_features_parallel(
            bad_data_dict
        )
        
        # Good instruments should still be processed
        good_instruments = [k for k in self.test_data_dict.keys()]
        for instrument in good_instruments:
            if instrument in features_dict:  # May skip due to errors
                self.assertIsInstance(features_dict[instrument], pd.DataFrame)
    
    def test_performance_comparison(self):
        """Test that parallel processing provides expected performance characteristics."""
        import time
        
        # Time sequential processing (simulate)
        start_time = time.time()
        sequential_features = {}
        engineer = FeatureEngineer()
        
        for instrument, data in self.test_data_dict.items():
            sequential_features[instrument] = engineer.generate_features_single_instrument(
                data, instrument
            )
        sequential_time = time.time() - start_time
        
        # Time parallel processing
        start_time = time.time()
        parallel_features, _ = self.multiprocessor.compute_features_parallel(
            self.test_data_dict
        )
        parallel_time = time.time() - start_time
        
        # Results should be equivalent (order might differ)
        self.assertEqual(set(sequential_features.keys()), set(parallel_features.keys()))
        
        # For small datasets and overhead, parallel might not be faster,
        # but it should complete successfully
        self.assertGreater(parallel_time, 0)
        self.assertGreater(sequential_time, 0)
    
    def test_worker_count_configuration(self):
        """Test different worker count configurations."""
        # Test with 1 worker (essentially sequential)
        mp_single = FeatureMultiprocessor(max_workers=1)
        features_single, _ = mp_single.compute_features_parallel(self.test_data_dict)
        
        # Test with multiple workers
        mp_multi = FeatureMultiprocessor(max_workers=3)
        features_multi, _ = mp_multi.compute_features_parallel(self.test_data_dict)
        
        # Results should be the same regardless of worker count
        self.assertEqual(set(features_single.keys()), set(features_multi.keys()))
        
        for instrument in features_single.keys():
            if instrument in features_multi:
                # Features should be very similar (may have tiny numerical differences)
                pd.testing.assert_frame_equal(
                    features_single[instrument], 
                    features_multi[instrument],
                    check_exact=False,
                    atol=1e-10
                )


class TestFeatureIntegration(unittest.TestCase):
    """Integration tests for feature pipeline components."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Initialize all components
        self.engineer = FeatureEngineer()
        self.normalizer = FeatureNormalizer()
        self.multiprocessor = FeatureMultiprocessor(max_workers=2)
        
        # Create test data
        np.random.seed(42)
        self.test_data_dict = {
            'EUR_USD': self._create_test_ohlc_data(),
            'GBP_USD': self._create_test_ohlc_data(),
            'USD_JPY': self._create_test_ohlc_data()
        }
    
    def _create_test_ohlc_data(self, n_rows=150):
        """Create synthetic OHLC data for testing."""
        dates = pd.date_range('2023-01-01', periods=n_rows, freq='H')
        
        price_base = np.random.uniform(0.8, 1.5)
        returns = np.random.normal(0, 0.002, n_rows)
        prices = [price_base]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, price in enumerate(prices):
            volatility = np.random.uniform(0.001, 0.003)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    def test_complete_feature_pipeline(self):
        """Test complete feature pipeline from raw data to normalized features."""
        # Step 1: Generate features in parallel
        features_dict, stats_dict = self.multiprocessor.compute_features_parallel(
            self.test_data_dict
        )
        
        # Step 2: Normalize features
        normalized_features = self.normalizer.normalize_features_comprehensive(
            features_dict
        )
        
        # Verify pipeline completion
        self.assertEqual(set(normalized_features.keys()), set(self.test_data_dict.keys()))
        
        for instrument, features in normalized_features.items():
            # Check structure
            self.assertIsInstance(features, pd.DataFrame)
            
            # Check required columns
            required_columns = ['slope_high', 'slope_low', 'volatility', 'direction']
            for col in required_columns:
                self.assertIn(col, features.columns)
            
            # Check that features are properly normalized
            for col in required_columns:
                # Should have reasonable variance (not all zeros)
                if not features[col].isna().all():
                    variance = features[col].var()
                    self.assertGreater(variance, 0)
    
    def test_pipeline_consistency(self):
        """Test that pipeline produces consistent results."""
        # Run pipeline twice
        features1, _ = self.multiprocessor.compute_features_parallel(self.test_data_dict)
        normalized1 = self.normalizer.normalize_features_comprehensive(features1)
        
        features2, _ = self.multiprocessor.compute_features_parallel(self.test_data_dict)
        normalized2 = self.normalizer.normalize_features_comprehensive(features2)
        
        # Results should be identical
        for instrument in normalized1.keys():
            pd.testing.assert_frame_equal(
                normalized1[instrument],
                normalized2[instrument],
                check_exact=False,
                atol=1e-10
            )
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline robustness with missing data."""
        # Introduce missing data
        corrupted_data = {}
        for instrument, data in self.test_data_dict.items():
            corrupted = data.copy()
            # Remove some random rows
            corrupted = corrupted.drop(corrupted.index[10:15])
            # Add some NaN values
            corrupted.loc[corrupted.index[20:25], 'high'] = np.nan
            corrupted_data[instrument] = corrupted
        
        # Pipeline should handle this gracefully
        features_dict, _ = self.multiprocessor.compute_features_parallel(corrupted_data)
        normalized_features = self.normalizer.normalize_features_comprehensive(features_dict)
        
        # Should still produce results
        self.assertGreater(len(normalized_features), 0)
        
        for instrument, features in normalized_features.items():
            self.assertIsInstance(features, pd.DataFrame)
            self.assertGreater(len(features), 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFeatureEngineer,
        TestFeatureNormalizer,
        TestFeatureMultiprocessor,
        TestFeatureIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")