#!/usr/bin/env python3
"""
PROPER EDGE DISCOVERY EXPERIMENT
================================

Correct implementation that predicts:
1. USD pips per standard lot (actual trading value)
2. Direction probability (0-1 for upward movement)
3. Confidence measures (prediction uncertainty)

NO SHORTCUTS - Complete rebuild with proper targets.

Author: Claude Code Assistant  
Date: 2025-10-30
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.tcnae import TCNAE, TCNAEConfig, TCNAETrainer
from models.pip_direction_gbdt import PipDirectionPredictor, PipDirectionConfig
from cache_latent_features import LatentFeatureCache
from utils.proper_data_loader import load_proper_data
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proper_experiment")

class ProperEdgeDiscoveryExperiment:
    """
    Complete rebuild of the edge discovery experiment with proper targets:
    - USD pip prediction (not scaled price_change)
    - Direction classification (not regression)
    - Confidence measures
    """
    
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id or f"proper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = Path("results") / f"proper_experiment_{self.experiment_id}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tcnae_model = None
        self.pip_direction_model = None
        self.cache_system = LatentFeatureCache()
        
        logger.info(f"🔬 Initializing PROPER experiment: {self.experiment_id}")
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load data with proper raw returns (calculated from OHLC).
        
        Returns:
            Tuple of (features, raw_returns, instrument_names)
        """
        logger.info("📊 Loading data with PROPER raw returns calculation...")
        
        # Load data with proper returns calculation
        features, raw_returns, instruments = load_proper_data()
        
        n_samples, n_instruments, n_features = features.shape
        logger.info(f"Loaded {n_samples:,} samples, {n_instruments} instruments, {n_features} features")
        
        # Reshape features for TCNAE: (N, sequence_length=4, instruments*features)
        # We'll use a sliding window approach
        sequence_length = 4
        reshaped_features = []
        reshaped_returns = []
        
        for i in range(sequence_length, n_samples):
            # Create 4-hour sequence
            seq_features = features[i-sequence_length:i]  # (4, n_instruments, 5)
            seq_features = seq_features.reshape(sequence_length, -1)  # (4, n_instruments*5)
            reshaped_features.append(seq_features)
            
            # Target is next hour's returns
            reshaped_returns.append(raw_returns[i])  # (n_instruments,)
        
        final_features = np.array(reshaped_features)  # (N-4, 4, n_instruments*5)
        final_returns = np.array(reshaped_returns)   # (N-4, n_instruments)
        
        logger.info(f"✅ Prepared {len(final_features):,} sequences for training")
        logger.info(f"✅ Using ACTUAL returns calculated from OHLC close prices")
        
        return final_features, final_returns, instruments
    
    def create_temporal_splits(self, features: np.ndarray, returns: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Create proper temporal splits (no lookahead bias).
        
        Returns:
            Tuple of train/val/test features and returns
        """
        n_samples = len(features)
        
        # Temporal splits: 70% train, 15% val, 15% test
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        X_train = features[:train_end]
        X_val = features[train_end:val_end]
        X_test = features[val_end:]
        
        y_train = returns[:train_end]
        y_val = returns[train_end:val_end]
        y_test = returns[val_end:]
        
        logger.info(f"📊 Temporal splits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def stage1_train_tcnae(self, X_train: np.ndarray, X_val: np.ndarray, 
                          final_features: np.ndarray) -> str:
        """
        Stage 1: Train TCNAE autoencoder for latent feature extraction.
        
        Returns:
            Cache name for latent features
        """
        logger.info("🔥 Stage 1: Training TCNAE autoencoder")
        
        # TCNAE configuration
        input_features = final_features.shape[2]  # n_instruments * 5
        config = TCNAEConfig(
            input_dim=input_features,  # Actual number of features
            latent_dim=120,  # Compress to 120D latent space
            sequence_length=4,
            encoder_channels=[input_features, 128, 96, 64],
            dropout=0.1
        )
        
        # Initialize TCNAE and trainer
        self.tcnae_model = TCNAE(config)
        trainer = TCNAETrainer(self.tcnae_model)
        
        # Convert to tensors and transpose for TCNAE: (batch, sequence, features) -> (batch, features, sequence)
        X_train_tensor = torch.FloatTensor(X_train).transpose(1, 2)  # (N, 4, 120) -> (N, 120, 4)
        X_val_tensor = torch.FloatTensor(X_val).transpose(1, 2)      # (N, 4, 120) -> (N, 120, 4)
        
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # Autoencoder: input = target
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Train the autoencoder
        training_history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            early_stopping_patience=15
        )
        
        # Save trained model
        model_path = self.results_dir / f"tcnae_model_{self.experiment_id}.pth"
        torch.save({
            'model_state_dict': self.tcnae_model.state_dict(),
            'config': config.__dict__,
            'training_history': training_history,
            'experiment_id': self.experiment_id
        }, model_path)
        
        logger.info(f"💾 Saved TCNAE model to {model_path}")
        
        # Extract and cache latent features
        cache_name = f"proper_{self.experiment_id}"
        
        logger.info("🔄 Extracting latent features from trained TCNAE...")
        
        # Extract latents for all data
        with torch.no_grad():
            X_train_latents = self.tcnae_model.encode(X_train_tensor).numpy()
            X_val_latents = self.tcnae_model.encode(X_val_tensor).numpy()
        
        # Cache latents for later stages (simple approach)
        cache_file = self.results_dir / f"cached_latents_{cache_name}.npz"
        np.savez(cache_file, 
                train_latents=X_train_latents, 
                val_latents=X_val_latents)
        logger.info(f"💾 Cached latent features to {cache_file}")
        
        logger.info(f"✅ Stage 1 complete: TCNAE trained and latents cached as '{cache_name}'")
        
        return cache_name
    
    def stage2_train_pip_direction_models(self, cache_name: str, y_train: np.ndarray, 
                                        y_val: np.ndarray, instruments: List[str]) -> Dict[str, Any]:
        """
        Stage 2: Train USD pip and direction prediction models.
        
        Args:
            cache_name: Name of cached latent features
            y_train: Training returns for pip calculation
            y_val: Validation returns for pip calculation
            instruments: List of instrument names
            
        Returns:
            Training results and metrics
        """
        logger.info("🎯 Stage 2: Training USD pip & direction prediction models")
        
        # Load cached latent features
        cache_file = self.results_dir / f"cached_latents_{cache_name}.npz"
        cached_data = np.load(cache_file)
        train_latents = cached_data['train_latents']
        val_latents = cached_data['val_latents']
        
        # Configure pip & direction model
        config = PipDirectionConfig(
            num_leaves=100,
            learning_rate=0.05,
            num_boost_round=1000,
            early_stopping_rounds=50
        )
        
        # Initialize model
        self.pip_direction_model = PipDirectionPredictor(config)
        
        # Create feature names
        feature_names = [f"latent_{i}" for i in range(train_latents.shape[1])]
        
        # Train the model
        training_results = self.pip_direction_model.fit(
            X_train=train_latents,
            X_val=val_latents,
            raw_returns_train=y_train,
            raw_returns_val=y_val,
            instruments=instruments,
            feature_names=feature_names
        )
        
        # Save trained model
        model_path = self.results_dir / f"pip_direction_model_{self.experiment_id}.pkl"
        self.pip_direction_model.save(model_path)
        
        logger.info(f"💾 Saved pip & direction models to {model_path}")
        
        # Save training results
        results_path = self.results_dir / f"stage2_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"✅ Stage 2 complete: Pip & direction models trained")
        
        return training_results
    
    def stage3_evaluate_performance(self, cache_name: str, X_test: np.ndarray, 
                                   y_test: np.ndarray, instruments: List[str]) -> Dict[str, Any]:
        """
        Stage 3: Comprehensive evaluation on test data.
        
        Args:
            cache_name: Name of cached latent features
            X_test: Test input features
            y_test: Test returns for evaluation
            instruments: List of instrument names
            
        Returns:
            Comprehensive performance metrics
        """
        logger.info("📊 Stage 3: Comprehensive performance evaluation")
        
        # Load TCNAE model if not already loaded
        if self.tcnae_model is None:
            logger.info("🔄 Loading trained TCNAE model for evaluation...")
            model_path = self.results_dir / f"tcnae_model_{self.experiment_id}.pth"
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Recreate TCNAE config
            config_dict = checkpoint['config']
            config = TCNAEConfig(**config_dict)
            
            # Load model
            self.tcnae_model = TCNAE(config)
            self.tcnae_model.load_state_dict(checkpoint['model_state_dict'])
            self.tcnae_model.eval()
            logger.info("✅ TCNAE model loaded successfully")
        
        # Extract test latents
        X_test_tensor = torch.FloatTensor(X_test).transpose(1, 2)  # (N, 4, 120) -> (N, 120, 4)
        with torch.no_grad():
            X_test_latents = self.tcnae_model.encode(X_test_tensor).numpy()
        
        # Load pip & direction model if not already loaded
        if self.pip_direction_model is None:
            logger.info("🔄 Loading trained pip & direction models for evaluation...")
            model_path = self.results_dir / f"pip_direction_model_{self.experiment_id}.pkl"
            self.pip_direction_model = PipDirectionPredictor.load(model_path)
            logger.info("✅ Pip & direction models loaded successfully")
        
        # Get predictions
        pip_predictions, direction_probabilities, confidence_scores = self.pip_direction_model.predict(X_test_latents)
        
        # Evaluate performance
        performance_results = self.pip_direction_model.evaluate_performance(X_test_latents, y_test)
        
        # Additional analysis
        analysis_results = {
            'experiment_summary': {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'n_instruments': len(instruments),
                'n_test_samples': len(X_test),
                'model_architecture': 'TCNAE(120D) -> PipDirection(USD+Probability)'
            },
            'performance_by_instrument': performance_results,
            'aggregate_metrics': {
                'avg_direction_accuracy': np.mean([r['direction_accuracy'] for r in performance_results.values()]),
                'avg_pip_correlation': np.mean([r['pip_correlation'] for r in performance_results.values()]),
                'total_simulated_pnl': sum([r['simulated_pnl_usd'] for r in performance_results.values()]),
                'best_performer': max(performance_results.keys(), 
                                    key=lambda k: performance_results[k]['direction_accuracy']),
                'worst_performer': min(performance_results.keys(), 
                                     key=lambda k: performance_results[k]['direction_accuracy'])
            },
            'prediction_samples': {
                'instruments': instruments,
                'pip_predictions_sample': pip_predictions[:10].tolist(),  # First 10 samples
                'direction_probabilities_sample': direction_probabilities[:10].tolist(),
                'confidence_scores_sample': confidence_scores[:10].tolist()
            }
        }
        
        # Save comprehensive results
        results_path = self.results_dir / f"comprehensive_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"💾 Saved evaluation results to {results_path}")
        
        # Print summary
        logger.info("🎯 EXPERIMENT RESULTS SUMMARY:")
        logger.info(f"Average Direction Accuracy: {analysis_results['aggregate_metrics']['avg_direction_accuracy']:.3f}")
        logger.info(f"Average Pip Correlation: {analysis_results['aggregate_metrics']['avg_pip_correlation']:.4f}")
        logger.info(f"Total Simulated P&L: ${analysis_results['aggregate_metrics']['total_simulated_pnl']:.2f}")
        logger.info(f"Best Performer: {analysis_results['aggregate_metrics']['best_performer']}")
        
        return analysis_results
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        Run the complete proper experiment pipeline.
        
        Returns:
            Final experiment results
        """
        try:
            logger.info("🚀 Starting PROPER edge discovery experiment")
            
            # Load and prepare data
            features, returns, instruments = self.load_and_prepare_data()
            
            # Create temporal splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_temporal_splits(features, returns)
            
            # Stage 1: Train TCNAE
            cache_name = self.stage1_train_tcnae(X_train, X_val, features)
            
            # Stage 2: Train pip & direction models
            stage2_results = self.stage2_train_pip_direction_models(cache_name, y_train, y_val, instruments)
            
            # Stage 3: Comprehensive evaluation
            final_results = self.stage3_evaluate_performance(cache_name, X_test, y_test, instruments)
            
            # Save experiment summary
            experiment_summary = {
                'experiment_id': self.experiment_id,
                'status': 'COMPLETED',
                'timestamp': datetime.now().isoformat(),
                'stages_completed': ['tcnae_training', 'pip_direction_training', 'evaluation'],
                'data_summary': {
                    'total_samples': len(features),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'instruments': len(instruments)
                },
                'final_results': final_results
            }
            
            summary_path = self.results_dir / f"experiment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(experiment_summary, f, indent=2, default=str)
            
            logger.info(f"✅ COMPLETE EXPERIMENT FINISHED: {self.experiment_id}")
            logger.info(f"📁 Results saved to: {self.results_dir}")
            
            return experiment_summary
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            raise

def main():
    """Main execution function."""
    logger.info("🔬 Starting PROPER Edge Discovery Experiment")
    logger.info("=" * 60)
    
    # Initialize experiment
    experiment = ProperEdgeDiscoveryExperiment()
    
    # Run complete experiment
    results = experiment.run_complete_experiment()
    
    logger.info("🎉 PROPER EXPERIMENT COMPLETE!")
    return results

if __name__ == "__main__":
    results = main()