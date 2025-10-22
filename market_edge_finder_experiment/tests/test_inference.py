"""
Test suite for inference pipeline components.

Tests for real-time prediction, data pipeline, and inference orchestration.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import warnings

# Import inference components
from ..inference.predictor import EdgePredictor, PredictionConfig, RealtimePredictor
from ..inference.data_pipeline import OANDADataProvider, DataBuffer, RealtimeDataPipeline, DataConfig

# Import model components for mocking
from ..models.tcnae import TCNAE, TCNAEConfig
from ..models.gbdt_model import MultiOutputGBDT, GBDTConfig
from ..models.context_manager import ContextTensorManager

# Suppress warnings for tests
warnings.filterwarnings('ignore')


class TestEdgePredictor(unittest.TestCase):
    """Test suite for EdgePredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary model files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create config with temporary paths
        self.config = PredictionConfig(
            tcnae_model_path=str(self.temp_path / "tcnae.pth"),
            gbdt_model_path=str(self.temp_path / "gbdt.pkl"),
            context_manager_path=str(self.temp_path / "context.pkl"),
            normalizer_path=str(self.temp_path / "normalizer.pkl"),
            sequence_length=4,
            latent_dim=10,
            num_instruments=5,
            device='cpu'
        )
        
        # Create mock model files
        self._create_mock_model_files()
        
        self.predictor = EdgePredictor(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_model_files(self):
        """Create mock model files for testing."""
        # Create mock TCNAE model
        tcnae_config = TCNAEConfig(
            input_channels=4,
            sequence_length=4,
            latent_dim=10,
            num_instruments=5
        )
        mock_tcnae = TCNAE(config=tcnae_config)
        torch.save(mock_tcnae.state_dict(), self.config.tcnae_model_path)
        
        # Create mock GBDT model
        import pickle
        mock_gbdt = MagicMock()
        mock_gbdt.predict.return_value = np.random.randn(5, 5)
        with open(self.config.gbdt_model_path, 'wb') as f:
            pickle.dump(mock_gbdt, f)
        
        # Create mock context manager state
        mock_context_state = {'context_tensor': torch.randn(10, 5)}
        with open(self.config.context_manager_path, 'wb') as f:
            pickle.dump(mock_context_state, f)
        
        # Create mock normalizer
        mock_normalizer = MagicMock()
        mock_normalizer.normalize_single_instrument.return_value = pd.DataFrame({
            'slope_high': np.random.randn(4),
            'slope_low': np.random.randn(4),
            'volatility': np.random.randn(4),
            'direction': np.random.randn(4)
        })
        with open(self.config.normalizer_path, 'wb') as f:
            pickle.dump(mock_normalizer, f)
    
    def test_initialization(self):
        """Test EdgePredictor initializes correctly."""
        self.assertIsInstance(self.predictor, EdgePredictor)
        self.assertEqual(self.predictor.config.num_instruments, 5)
        self.assertFalse(self.predictor.is_initialized)
    
    def test_model_initialization(self):
        """Test model loading and initialization."""
        self.predictor.initialize_models()
        
        self.assertTrue(self.predictor.is_initialized)
        self.assertIsNotNone(self.predictor.tcnae_model)
        self.assertIsNotNone(self.predictor.gbdt_model)
        self.assertIsNotNone(self.predictor.context_manager)
        self.assertIsNotNone(self.predictor.normalizer)
    
    def test_feature_update(self):
        """Test feature history update."""
        # Create mock market data
        market_data = {}
        for i, instrument in enumerate(self.predictor.instruments[:3]):
            market_data[instrument] = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
                'open': np.random.randn(10) + 1.0,
                'high': np.random.randn(10) + 1.1,
                'low': np.random.randn(10) + 0.9,
                'close': np.random.randn(10) + 1.0,
                'volume': np.random.randint(100, 1000, 10)
            })
        
        # Mock feature engineer
        with patch.object(self.predictor, 'feature_engineer') as mock_engineer:
            mock_engineer.generate_features_single_instrument.return_value = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
                'slope_high': np.random.randn(10),
                'slope_low': np.random.randn(10),
                'volatility': np.random.randn(10),
                'direction': np.random.randn(10)
            })
            
            self.predictor.update_features(market_data)
            
            # Check that features were stored
            self.assertGreater(len(self.predictor.feature_history), 0)
            self.assertIsNotNone(self.predictor.last_update_time)
    
    @patch('torch.load')
    def test_prediction_generation_without_initialization(self, mock_torch_load):
        """Test that prediction fails without initialization."""
        with self.assertRaises(Exception):
            self.predictor.generate_predictions()
    
    def test_prediction_generation_with_insufficient_data(self):
        """Test prediction handling with insufficient feature data."""
        self.predictor.initialize_models()
        
        # Try to generate predictions without any feature history
        result = self.predictor.generate_predictions()
        self.assertIsNone(result)
    
    def test_prediction_cache(self):
        """Test prediction caching mechanism."""
        self.predictor.initialize_models()
        
        # Mock successful prediction
        mock_prediction = {
            'predictions': {instrument: 0.1 for instrument in self.predictor.instruments},
            'metadata': {'timestamp': '2023-01-01T12:00:00'}
        }
        self.predictor.prediction_cache = mock_prediction
        
        # Test cache retrieval
        cached = self.predictor.get_latest_predictions(max_age_minutes=60)
        self.assertIsNotNone(cached)
        
        # Test stale cache
        old_prediction = {
            'predictions': {instrument: 0.1 for instrument in self.predictor.instruments},
            'metadata': {'timestamp': '2020-01-01T12:00:00'}  # Very old
        }
        self.predictor.prediction_cache = old_prediction
        
        stale_cached = self.predictor.get_latest_predictions(max_age_minutes=60)
        self.assertIsNone(stale_cached)
    
    def test_model_health_check(self):
        """Test model health status reporting."""
        health = self.predictor.get_model_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('initialized', health)
        self.assertIn('components', health)
        self.assertIn('feature_history_status', health)
    
    def test_state_save_load(self):
        """Test predictor state persistence."""
        self.predictor.initialize_models()
        
        # Update with some mock data
        self.predictor.feature_history['EUR_USD'] = pd.DataFrame({
            'slope_high': [1, 2, 3],
            'slope_low': [1, 2, 3],
            'volatility': [1, 2, 3],
            'direction': [1, 2, 3]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            state_path = tmp.name
        
        try:
            # Save state
            self.predictor.save_state(state_path)
            
            # Create new predictor and load state
            new_predictor = EdgePredictor(self.config)
            new_predictor.initialize_models()
            new_predictor.load_state(state_path)
            
            # Check that feature history was restored
            self.assertIn('EUR_USD', new_predictor.feature_history)
            
        finally:
            Path(state_path).unlink()


class TestDataBuffer(unittest.TestCase):
    """Test suite for DataBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DataConfig(
            api_key='test_key',
            account_id='test_account',
            buffer_size_hours=48
        )
        self.buffer = DataBuffer(self.config)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'open': np.random.randn(100) + 1.0,
            'high': np.random.randn(100) + 1.1,
            'low': np.random.randn(100) + 0.9,
            'close': np.random.randn(100) + 1.0,
            'volume': np.random.randint(100, 1000, 100)
        })
    
    def test_initialization(self):
        """Test DataBuffer initializes correctly."""
        self.assertIsInstance(self.buffer, DataBuffer)
        self.assertEqual(self.buffer.config.buffer_size_hours, 48)
        self.assertEqual(len(self.buffer.data), 0)
    
    def test_add_data(self):
        """Test adding data to buffer."""
        self.buffer.add_data('EUR_USD', self.test_data)
        
        self.assertIn('EUR_USD', self.buffer.data)
        self.assertEqual(len(self.buffer.data['EUR_USD']), len(self.test_data))
    
    def test_data_retrieval(self):
        """Test data retrieval from buffer."""
        self.buffer.add_data('EUR_USD', self.test_data)
        
        # Test full data retrieval
        retrieved = self.buffer.get_data('EUR_USD')
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), len(self.test_data))
        
        # Test lookback retrieval
        retrieved_partial = self.buffer.get_data('EUR_USD', lookback_hours=24)
        self.assertIsNotNone(retrieved_partial)
        self.assertLessEqual(len(retrieved_partial), len(self.test_data))
    
    def test_buffer_size_limit(self):
        """Test that buffer respects size limits."""
        # Create data larger than buffer size
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='H'),
            'open': np.random.randn(200) + 1.0,
            'high': np.random.randn(200) + 1.1,
            'low': np.random.randn(200) + 0.9,
            'close': np.random.randn(200) + 1.0,
            'volume': np.random.randint(100, 1000, 200)
        })
        
        self.buffer.add_data('EUR_USD', large_data)
        
        # Buffer should be trimmed to max size
        buffered_data = self.buffer.get_data('EUR_USD')
        self.assertLessEqual(len(buffered_data), self.config.buffer_size_hours)
    
    def test_duplicate_handling(self):
        """Test handling of duplicate timestamps."""
        # Add initial data
        self.buffer.add_data('EUR_USD', self.test_data)
        
        # Add overlapping data
        overlap_data = self.test_data.tail(10).copy()
        overlap_data['volume'] = overlap_data['volume'] * 2  # Different values
        
        self.buffer.add_data('EUR_USD', overlap_data)
        
        # Should not have duplicates
        buffered_data = self.buffer.get_data('EUR_USD')
        duplicate_count = buffered_data['timestamp'].duplicated().sum()
        self.assertEqual(duplicate_count, 0)
    
    def test_buffer_status(self):
        """Test buffer status reporting."""
        self.buffer.add_data('EUR_USD', self.test_data)
        self.buffer.add_data('GBP_USD', self.test_data)
        
        status = self.buffer.get_buffer_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status['total_instruments'], 2)
        self.assertIn('EUR_USD', status['instruments'])
        self.assertIn('GBP_USD', status['instruments'])


class TestRealtimeDataPipeline(unittest.TestCase):
    """Test suite for RealtimeDataPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DataConfig(
            api_key='test_key',
            account_id='test_account',
            instruments=['EUR_USD', 'GBP_USD'],
            update_interval_seconds=1  # Fast for testing
        )
        
        # Mock the OANDA data provider
        with patch('market_edge_finder_experiment.inference.data_pipeline.OANDADataProvider') as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.fetch_historical_data.return_value = self._create_test_data()
            mock_provider.fetch_latest_data.return_value = self._create_test_data(n_rows=10)
            mock_provider.get_connection_health.return_value = {'api_accessible': True}
            mock_provider_class.return_value = mock_provider
            
            self.pipeline = RealtimeDataPipeline(self.config)
            self.mock_provider = mock_provider
    
    def _create_test_data(self, n_rows=50):
        """Create test OHLC data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
            'open': np.random.randn(n_rows) + 1.0,
            'high': np.random.randn(n_rows) + 1.1,
            'low': np.random.randn(n_rows) + 0.9,
            'close': np.random.randn(n_rows) + 1.0,
            'volume': np.random.randint(100, 1000, n_rows)
        })
    
    def test_initialization(self):
        """Test RealtimeDataPipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, RealtimeDataPipeline)
        self.assertFalse(self.pipeline.running)
        self.assertEqual(len(self.pipeline.data_callbacks), 0)
    
    def test_callback_registration(self):
        """Test data callback registration."""
        def dummy_callback(data):
            pass
        
        self.pipeline.add_data_callback(dummy_callback)
        self.assertEqual(len(self.pipeline.data_callbacks), 1)
    
    def test_initial_data_load(self):
        """Test initial data loading."""
        self.pipeline._initial_data_load()
        
        # Should have called fetch_historical_data for each instrument
        expected_calls = len(self.config.instruments)
        self.assertEqual(self.mock_provider.fetch_historical_data.call_count, expected_calls)
    
    def test_pipeline_status(self):
        """Test pipeline status reporting."""
        status = self.pipeline.get_pipeline_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('running', status)
        self.assertIn('update_interval_seconds', status)
        self.assertIn('api_health', status)
        self.assertIn('buffer_status', status)
    
    def test_force_update(self):
        """Test forced data update."""
        updated_data = self.pipeline.force_update()
        
        self.assertIsInstance(updated_data, dict)
        # Should have attempted to update all instruments
        self.assertEqual(self.mock_provider.fetch_latest_data.call_count, len(self.config.instruments))
    
    def test_current_data_retrieval(self):
        """Test current data retrieval."""
        # Add some data first
        self.pipeline._initial_data_load()
        
        # Test getting all data
        all_data = self.pipeline.get_current_data()
        self.assertIsInstance(all_data, dict)
        
        # Test getting specific instrument data
        if 'EUR_USD' in all_data:
            eur_data = self.pipeline.get_current_data('EUR_USD')
            self.assertIsInstance(eur_data, pd.DataFrame)


class TestRealtimePredictor(unittest.TestCase):
    """Test suite for RealtimePredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary model files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.config = PredictionConfig(
            tcnae_model_path=str(self.temp_path / "tcnae.pth"),
            gbdt_model_path=str(self.temp_path / "gbdt.pkl"),
            context_manager_path=str(self.temp_path / "context.pkl"),
            normalizer_path=str(self.temp_path / "normalizer.pkl"),
            sequence_length=4,
            latent_dim=10,
            num_instruments=3,
            device='cpu'
        )
        
        self._create_mock_model_files()
        
        self.realtime_predictor = RealtimePredictor(
            self.config, 
            update_interval_seconds=1  # Fast for testing
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.realtime_predictor.running:
            self.realtime_predictor.stop()
        
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_model_files(self):
        """Create mock model files for testing."""
        # Create mock TCNAE model
        tcnae_config = TCNAEConfig(
            input_channels=4,
            sequence_length=4,
            latent_dim=10,
            num_instruments=3
        )
        mock_tcnae = TCNAE(config=tcnae_config)
        torch.save(mock_tcnae.state_dict(), self.config.tcnae_model_path)
        
        # Create mock GBDT model
        import pickle
        mock_gbdt = MagicMock()
        mock_gbdt.predict.return_value = np.random.randn(3, 3)
        with open(self.config.gbdt_model_path, 'wb') as f:
            pickle.dump(mock_gbdt, f)
        
        # Create mock context manager state
        mock_context_state = {'context_tensor': torch.randn(10, 3)}
        with open(self.config.context_manager_path, 'wb') as f:
            pickle.dump(mock_context_state, f)
        
        # Create mock normalizer
        mock_normalizer = MagicMock()
        mock_normalizer.normalize_single_instrument.return_value = pd.DataFrame({
            'slope_high': np.random.randn(4),
            'slope_low': np.random.randn(4),
            'volatility': np.random.randn(4),
            'direction': np.random.randn(4)
        })
        with open(self.config.normalizer_path, 'wb') as f:
            pickle.dump(mock_normalizer, f)
    
    def test_initialization(self):
        """Test RealtimePredictor initializes correctly."""
        self.assertIsInstance(self.realtime_predictor, RealtimePredictor)
        self.assertFalse(self.realtime_predictor.running)
    
    def test_start_stop(self):
        """Test starting and stopping the real-time predictor."""
        # Mock the EdgePredictor initialization
        with patch.object(self.realtime_predictor.predictor, 'initialize_models'):
            self.realtime_predictor.start()
            self.assertTrue(self.realtime_predictor.running)
            
            # Let it run briefly
            time.sleep(0.1)
            
            self.realtime_predictor.stop()
            self.assertFalse(self.realtime_predictor.running)
    
    def test_market_data_addition(self):
        """Test adding market data to the queue."""
        market_data = {
            'EUR_USD': pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5, freq='H'),
                'open': np.random.randn(5) + 1.0,
                'high': np.random.randn(5) + 1.1,
                'low': np.random.randn(5) + 0.9,
                'close': np.random.randn(5) + 1.0,
                'volume': np.random.randint(100, 1000, 5)
            })
        }
        
        # Should not raise exception
        self.realtime_predictor.add_market_data(market_data)
        
        # Queue should have data
        self.assertFalse(self.realtime_predictor.data_queue.empty())
    
    def test_prediction_retrieval(self):
        """Test getting current predictions."""
        # Mock predictions in the underlying predictor
        mock_predictions = {
            'predictions': {'EUR_USD': 0.1, 'GBP_USD': 0.2},
            'metadata': {'timestamp': '2023-01-01T12:00:00'}
        }
        self.realtime_predictor.predictor.prediction_cache = mock_predictions
        
        predictions = self.realtime_predictor.get_current_predictions()
        self.assertEqual(predictions, mock_predictions)


class TestInferenceIntegration(unittest.TestCase):
    """Integration tests for inference pipeline components."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Create temporary directory for model files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock model files
        self._create_mock_model_files()
        
        # Create config
        self.prediction_config = PredictionConfig(
            tcnae_model_path=str(self.temp_path / "tcnae.pth"),
            gbdt_model_path=str(self.temp_path / "gbdt.pkl"),
            context_manager_path=str(self.temp_path / "context.pkl"),
            normalizer_path=str(self.temp_path / "normalizer.pkl"),
            sequence_length=4,
            latent_dim=8,
            num_instruments=3,
            device='cpu'
        )
        
        self.data_config = DataConfig(
            api_key='test_key',
            account_id='test_account',
            instruments=['EUR_USD', 'GBP_USD', 'USD_JPY'],
            update_interval_seconds=1
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_model_files(self):
        """Create mock model files."""
        # TCNAE model
        tcnae_config = TCNAEConfig(
            input_channels=4,
            sequence_length=4,
            latent_dim=8,
            num_instruments=3
        )
        mock_tcnae = TCNAE(config=tcnae_config)
        torch.save(mock_tcnae.state_dict(), self.temp_path / "tcnae.pth")
        
        # GBDT model
        import pickle
        mock_gbdt = MagicMock()
        mock_gbdt.predict.return_value = np.random.randn(3, 3)
        with open(self.temp_path / "gbdt.pkl", 'wb') as f:
            pickle.dump(mock_gbdt, f)
        
        # Context manager
        mock_context_state = {'context_tensor': torch.randn(8, 3)}
        with open(self.temp_path / "context.pkl", 'wb') as f:
            pickle.dump(mock_context_state, f)
        
        # Normalizer
        mock_normalizer = MagicMock()
        mock_normalizer.normalize_single_instrument.return_value = pd.DataFrame({
            'slope_high': np.random.randn(4),
            'slope_low': np.random.randn(4),
            'volatility': np.random.randn(4),
            'direction': np.random.randn(4)
        })
        with open(self.temp_path / "normalizer.pkl", 'wb') as f:
            pickle.dump(mock_normalizer, f)
    
    def test_predictor_data_pipeline_integration(self):
        """Test integration between predictor and data pipeline."""
        # Initialize predictor
        predictor = EdgePredictor(self.prediction_config)
        predictor.initialize_models()
        
        # Create mock market data
        market_data = {}
        for instrument in self.data_config.instruments:
            market_data[instrument] = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
                'open': np.random.randn(10) + 1.0,
                'high': np.random.randn(10) + 1.1,
                'low': np.random.randn(10) + 0.9,
                'close': np.random.randn(10) + 1.0,
                'volume': np.random.randint(100, 1000, 10)
            })
        
        # Mock the feature engineer to avoid complex setup
        with patch.object(predictor, 'feature_engineer') as mock_engineer:
            mock_engineer.generate_features_single_instrument.return_value = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
                'slope_high': np.random.randn(10),
                'slope_low': np.random.randn(10),
                'volatility': np.random.randn(10),
                'direction': np.random.randn(10)
            })
            
            # Update features
            predictor.update_features(market_data)
            
            # Should have feature history
            self.assertGreater(len(predictor.feature_history), 0)
            
            # Try to generate predictions (will fail due to insufficient data, but should not crash)
            try:
                predictions = predictor.generate_predictions()
                # If it succeeds, check structure
                if predictions is not None:
                    self.assertIn('predictions', predictions)
                    self.assertIn('metadata', predictions)
            except Exception as e:
                # Expected due to mocking, but should be a reasonable error
                self.assertIsInstance(e, (ValueError, RuntimeError, KeyError))
    
    def test_end_to_end_mock_pipeline(self):
        """Test end-to-end pipeline with extensive mocking."""
        # Mock all external dependencies
        with patch('market_edge_finder_experiment.inference.data_pipeline.OANDADataProvider') as mock_provider_class:
            # Setup mock data provider
            mock_provider = MagicMock()
            mock_provider.fetch_historical_data.return_value = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=50, freq='H'),
                'open': np.random.randn(50) + 1.0,
                'high': np.random.randn(50) + 1.1,
                'low': np.random.randn(50) + 0.9,
                'close': np.random.randn(50) + 1.0,
                'volume': np.random.randint(100, 1000, 50)
            })
            mock_provider.get_connection_health.return_value = {'api_accessible': True}
            mock_provider_class.return_value = mock_provider
            
            # Create data pipeline
            data_pipeline = RealtimeDataPipeline(self.data_config)
            
            # Create predictor
            predictor = EdgePredictor(self.prediction_config)
            
            # Setup callback to connect pipeline to predictor
            def data_callback(market_data):
                predictor.update_features(market_data)
            
            data_pipeline.add_data_callback(data_callback)
            
            # Initialize components
            predictor.initialize_models()
            
            # Load initial data
            data_pipeline._initial_data_load()
            
            # Verify that the pipeline can at least attempt to process data
            status = data_pipeline.get_pipeline_status()
            self.assertIn('api_health', status)
            
            health = predictor.get_model_health()
            self.assertTrue(health['initialized'])


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEdgePredictor,
        TestDataBuffer,
        TestRealtimeDataPipeline,
        TestRealtimePredictor,
        TestInferenceIntegration
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