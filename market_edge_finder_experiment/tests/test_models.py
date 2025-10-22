"""
Comprehensive test suite for model components.

Tests for TCNAE, LightGBM, Context Manager, and hybrid training pipeline.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path
import warnings

# Import model components
from ..models.tcnae import TCNAE, TCNAEConfig
from ..models.gbdt_model import MultiOutputGBDT, GBDTConfig
from ..models.context_manager import ContextTensorManager

# Suppress warnings for tests
warnings.filterwarnings('ignore')


class TestTCNAE(unittest.TestCase):
    """Test suite for TCNAE model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TCNAEConfig(
            input_channels=4,
            sequence_length=4,
            latent_dim=10,
            num_instruments=5,
            hidden_channels=[8, 16],
            kernel_size=3,
            dropout=0.1
        )
        self.model = TCNAE(config=self.config)
        self.batch_size = 2
        
        # Create test input tensor: (batch, features, sequence, instruments)
        self.test_input = torch.randn(
            self.batch_size, 
            self.config.input_channels, 
            self.config.sequence_length, 
            self.config.num_instruments
        )
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model, TCNAE)
        self.assertEqual(self.model.config.latent_dim, 10)
        self.assertEqual(self.model.config.num_instruments, 5)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        self.model.eval()
        with torch.no_grad():
            reconstruction, latent = self.model(self.test_input)
        
        # Check output shapes
        expected_reconstruction_shape = self.test_input.shape
        expected_latent_shape = (self.batch_size, self.config.latent_dim, self.config.num_instruments)
        
        self.assertEqual(reconstruction.shape, expected_reconstruction_shape)
        self.assertEqual(latent.shape, expected_latent_shape)
    
    def test_encoder_output(self):
        """Test encoder produces correct latent representation."""
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(self.test_input)
        
        expected_shape = (self.batch_size, self.config.latent_dim, self.config.num_instruments)
        self.assertEqual(latent.shape, expected_shape)
    
    def test_decoder_output(self):
        """Test decoder reconstructs from latent representation."""
        self.model.eval()
        
        # Create test latent representation
        test_latent = torch.randn(
            self.batch_size, 
            self.config.latent_dim, 
            self.config.num_instruments
        )
        
        with torch.no_grad():
            reconstruction = self.model.decode(test_latent)
        
        expected_shape = (
            self.batch_size, 
            self.config.input_channels, 
            self.config.sequence_length, 
            self.config.num_instruments
        )
        self.assertEqual(reconstruction.shape, expected_shape)
    
    def test_loss_calculation(self):
        """Test loss calculation works correctly."""
        self.model.eval()
        with torch.no_grad():
            reconstruction, latent = self.model(self.test_input)
            loss = self.model.calculate_loss(reconstruction, self.test_input, latent)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Loss should be positive
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pth"
            
            # Save model
            torch.save(self.model.state_dict(), save_path)
            
            # Create new model and load state
            new_model = TCNAE(config=self.config)
            new_model.load_state_dict(torch.load(save_path))
            
            # Test that models produce same output
            self.model.eval()
            new_model.eval()
            
            with torch.no_grad():
                original_output = self.model(self.test_input)
                loaded_output = new_model(self.test_input)
            
            torch.testing.assert_close(original_output[0], loaded_output[0])
            torch.testing.assert_close(original_output[1], loaded_output[1])
    
    def test_device_compatibility(self):
        """Test model works on different devices."""
        # Test CPU
        self.model.eval()
        with torch.no_grad():
            output_cpu = self.model(self.test_input)
        
        self.assertIsInstance(output_cpu[0], torch.Tensor)
        self.assertIsInstance(output_cpu[1], torch.Tensor)
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = self.model.cuda()
            input_gpu = self.test_input.cuda()
            
            with torch.no_grad():
                output_gpu = model_gpu(input_gpu)
            
            self.assertEqual(output_gpu[0].device.type, 'cuda')
            self.assertEqual(output_gpu[1].device.type, 'cuda')


class TestMultiOutputGBDT(unittest.TestCase):
    """Test suite for MultiOutputGBDT model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GBDTConfig(
            num_instruments=5,
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            verbosity=-1  # Suppress LightGBM output
        )
        self.model = MultiOutputGBDT(config=self.config)
        
        # Create test data
        self.n_samples = 100
        self.n_features = 20
        
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.randn(self.n_samples, self.config.num_instruments)
        
        self.X_test = np.random.randn(50, self.n_features)
        self.y_test = np.random.randn(50, self.config.num_instruments)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model, MultiOutputGBDT)
        self.assertEqual(self.model.config.num_instruments, 5)
        self.assertFalse(self.model.is_fitted)
    
    def test_training(self):
        """Test model training works correctly."""
        # Train model
        training_history = self.model.fit(self.X_train, self.y_train)
        
        self.assertTrue(self.model.is_fitted)
        self.assertIsInstance(training_history, dict)
        self.assertIn('training_loss', training_history)
        self.assertEqual(len(self.model.models), self.config.num_instruments)
    
    def test_prediction(self):
        """Test model prediction after training."""
        # Train model first
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        expected_shape = (len(self.X_test), self.config.num_instruments)
        self.assertEqual(predictions.shape, expected_shape)
        self.assertFalse(np.any(np.isnan(predictions)))
    
    def test_validation_during_training(self):
        """Test validation during training."""
        training_history = self.model.fit(
            self.X_train, self.y_train,
            X_val=self.X_test, y_val=self.y_test
        )
        
        self.assertIn('validation_loss', training_history)
        self.assertTrue(len(training_history['validation_loss']) > 0)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Train model first
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        self.assertEqual(importance.shape, (self.n_features, self.config.num_instruments))
        self.assertTrue(np.all(importance >= 0))  # Importance should be non-negative
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Train model first
        self.model.fit(self.X_train, self.y_train)
        original_predictions = self.model.predict(self.X_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_gbdt.pkl"
            
            # Save model
            self.model.save_model(str(save_path))
            
            # Load model
            loaded_model = MultiOutputGBDT.load_model(str(save_path))
            loaded_predictions = loaded_model.predict(self.X_test)
            
            # Check predictions are the same
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
    
    def test_incremental_learning(self):
        """Test incremental learning capability."""
        # Initial training
        self.model.fit(self.X_train, self.y_train)
        initial_predictions = self.model.predict(self.X_test)
        
        # Incremental training
        X_new = np.random.randn(50, self.n_features)
        y_new = np.random.randn(50, self.config.num_instruments)
        
        self.model.fit_incremental(X_new, y_new)
        updated_predictions = self.model.predict(self.X_test)
        
        # Predictions should be different after incremental training
        self.assertFalse(np.array_equal(initial_predictions, updated_predictions))


class TestContextTensorManager(unittest.TestCase):
    """Test suite for ContextTensorManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_instruments = 5
        self.context_dim = 10
        self.device = torch.device('cpu')
        
        self.manager = ContextTensorManager(
            num_instruments=self.num_instruments,
            context_dim=self.context_dim,
            device=self.device
        )
        
        # Create test tensors
        self.test_predictions = torch.randn(1, self.num_instruments)
        self.test_targets = torch.randn(1, self.num_instruments)
    
    def test_initialization(self):
        """Test context manager initializes correctly."""
        self.assertEqual(self.manager.num_instruments, self.num_instruments)
        self.assertEqual(self.manager.context_dim, self.context_dim)
        
        context = self.manager.get_current_context()
        expected_shape = (self.context_dim, self.num_instruments)
        self.assertEqual(context.shape, expected_shape)
    
    def test_context_update(self):
        """Test context updating with new predictions."""
        initial_context = self.manager.get_current_context().clone()
        
        # Update context
        updated_context = self.manager.update_context(self.test_predictions)
        
        # Context should have changed
        self.assertFalse(torch.equal(initial_context, updated_context))
        self.assertEqual(updated_context.shape, initial_context.shape)
    
    def test_teacher_forcing(self):
        """Test teacher forcing mechanism."""
        # Enable teacher forcing
        context_with_teacher = self.manager.update_context(
            self.test_predictions, 
            self.test_targets, 
            use_teacher_forcing=True
        )
        
        # Disable teacher forcing
        context_without_teacher = self.manager.update_context(
            self.test_predictions, 
            use_teacher_forcing=False
        )
        
        # Contexts should be different
        self.assertFalse(torch.equal(context_with_teacher, context_without_teacher))
    
    def test_adaptive_teacher_forcing(self):
        """Test adaptive teacher forcing based on correlation."""
        # Setup with known correlation patterns
        predictions = torch.ones(1, self.num_instruments) * 0.5
        targets = torch.ones(1, self.num_instruments) * 0.5  # High correlation
        
        # Update several times to build correlation history
        for _ in range(10):
            self.manager.update_context(predictions, targets, use_teacher_forcing=True)
        
        # Check that teacher forcing ratios are updated
        ratios = self.manager.get_teacher_forcing_ratios()
        self.assertEqual(len(ratios), self.num_instruments)
        self.assertTrue(all(0 <= ratio <= 1 for ratio in ratios))
    
    def test_cross_instrument_attention(self):
        """Test cross-instrument attention mechanism."""
        # Create predictions with different patterns
        predictions = torch.zeros(1, self.num_instruments)
        predictions[0, 0] = 1.0  # Strong signal in first instrument
        
        initial_context = self.manager.get_current_context().clone()
        updated_context = self.manager.update_context(predictions)
        
        # All instruments should be influenced by the strong signal
        context_diff = updated_context - initial_context
        self.assertTrue(torch.any(torch.abs(context_diff) > 0))
    
    def test_context_persistence(self):
        """Test context maintains information over time."""
        contexts = []
        
        for i in range(5):
            prediction = torch.ones(1, self.num_instruments) * (i + 1)
            context = self.manager.update_context(prediction)
            contexts.append(context.clone())
        
        # Later contexts should be different from earlier ones
        self.assertFalse(torch.equal(contexts[0], contexts[-1]))
        
        # But should maintain some historical influence
        # (This is harder to test directly, but we check that context evolves smoothly)
        for i in range(1, len(contexts)):
            diff = torch.norm(contexts[i] - contexts[i-1])
            self.assertGreater(diff.item(), 0)  # Should change
            self.assertLess(diff.item(), 10)    # But not drastically
    
    def test_state_save_load(self):
        """Test saving and loading context state."""
        # Update context with some data
        for i in range(3):
            prediction = torch.randn(1, self.num_instruments)
            self.manager.update_context(prediction)
        
        original_context = self.manager.get_current_context().clone()
        original_state = self.manager.get_state()
        
        # Create new manager and load state
        new_manager = ContextTensorManager(
            num_instruments=self.num_instruments,
            context_dim=self.context_dim,
            device=self.device
        )
        new_manager.load_state(original_state)
        
        restored_context = new_manager.get_current_context()
        
        # Contexts should be identical
        torch.testing.assert_close(original_context, restored_context)
    
    def test_device_compatibility(self):
        """Test context manager works on different devices."""
        # Test CPU (already default)
        cpu_context = self.manager.get_current_context()
        self.assertEqual(cpu_context.device.type, 'cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            gpu_manager = ContextTensorManager(
                num_instruments=self.num_instruments,
                context_dim=self.context_dim,
                device=torch.device('cuda')
            )
            
            gpu_context = gpu_manager.get_current_context()
            self.assertEqual(gpu_context.device.type, 'cuda')


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components working together."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.num_instruments = 3
        self.sequence_length = 4
        self.input_channels = 4
        self.latent_dim = 8
        self.batch_size = 2
        
        # Initialize models
        tcnae_config = TCNAEConfig(
            input_channels=self.input_channels,
            sequence_length=self.sequence_length,
            latent_dim=self.latent_dim,
            num_instruments=self.num_instruments,
            hidden_channels=[6, 8],
            kernel_size=3,
            dropout=0.1
        )
        self.tcnae = TCNAE(config=tcnae_config)
        
        gbdt_config = GBDTConfig(
            num_instruments=self.num_instruments,
            n_estimators=10,
            max_depth=3,
            verbosity=-1
        )
        self.gbdt = MultiOutputGBDT(config=gbdt_config)
        
        self.context_manager = ContextTensorManager(
            num_instruments=self.num_instruments,
            context_dim=self.latent_dim,
            device=torch.device('cpu')
        )
        
        # Create test data
        self.test_sequences = torch.randn(
            self.batch_size,
            self.input_channels,
            self.sequence_length,
            self.num_instruments
        )
        
        self.test_targets = torch.randn(self.batch_size, self.num_instruments)
    
    def test_tcnae_to_gbdt_pipeline(self):
        """Test TCNAE features can be used by GBDT."""
        # Generate TCNAE features
        self.tcnae.eval()
        with torch.no_grad():
            _, latent_features = self.tcnae(self.test_sequences)
        
        # Prepare features for GBDT (reshape and convert to numpy)
        latent_np = latent_features.cpu().numpy()
        # Reshape from (batch, latent_dim, instruments) to (batch*instruments, latent_dim)
        X_train = latent_np.transpose(0, 2, 1).reshape(-1, self.latent_dim)
        y_train = self.test_targets.cpu().numpy().reshape(-1, 1)
        
        # Train GBDT
        self.gbdt.fit(X_train.reshape(-1, self.latent_dim), 
                     y_train.reshape(self.batch_size, self.num_instruments))
        
        # Make predictions
        predictions = self.gbdt.predict(X_train.reshape(-1, self.latent_dim))
        
        self.assertEqual(predictions.shape, (len(X_train), self.num_instruments))
    
    def test_context_integration(self):
        """Test context manager integration with TCNAE and GBDT."""
        # Generate TCNAE features
        self.tcnae.eval()
        with torch.no_grad():
            _, latent_features = self.tcnae(self.test_sequences)
        
        # Get context
        context = self.context_manager.get_current_context()
        
        # Combine latent features with context
        # latent_features: (batch, latent_dim, instruments)
        # context: (latent_dim, instruments)
        combined_features = latent_features + context.unsqueeze(0)
        
        # Verify shapes
        self.assertEqual(combined_features.shape, latent_features.shape)
        
        # Update context with mock predictions
        mock_predictions = torch.randn(1, self.num_instruments)
        updated_context = self.context_manager.update_context(mock_predictions)
        
        self.assertEqual(updated_context.shape, context.shape)
    
    def test_end_to_end_prediction_flow(self):
        """Test complete prediction flow from input to output."""
        # 1. TCNAE feature extraction
        self.tcnae.eval()
        with torch.no_grad():
            _, latent_features = self.tcnae(self.test_sequences)
        
        # 2. Get context
        context = self.context_manager.get_current_context()
        
        # 3. Combine features with context
        combined_features = latent_features + context.unsqueeze(0)
        
        # 4. Prepare for GBDT
        features_np = combined_features.cpu().numpy()
        X_for_gbdt = features_np.transpose(0, 2, 1).reshape(-1, self.latent_dim)
        
        # 5. Train GBDT (in real scenario, this would be pre-trained)
        y_dummy = np.random.randn(len(X_for_gbdt), self.num_instruments)
        self.gbdt.fit(X_for_gbdt, y_dummy)
        
        # 6. Generate predictions
        predictions = self.gbdt.predict(X_for_gbdt)
        
        # 7. Update context with predictions
        pred_tensor = torch.FloatTensor(predictions[:self.num_instruments]).unsqueeze(0)
        self.context_manager.update_context(pred_tensor)
        
        # Verify the flow completed successfully
        self.assertEqual(predictions.shape, (len(X_for_gbdt), self.num_instruments))
        
        # Verify context was updated
        new_context = self.context_manager.get_current_context()
        self.assertFalse(torch.equal(context, new_context))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTCNAE,
        TestMultiOutputGBDT,
        TestContextTensorManager,
        TestModelIntegration
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