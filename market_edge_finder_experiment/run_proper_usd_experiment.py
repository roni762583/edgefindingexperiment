#!/usr/bin/env python3
"""
PROPER USD PIP EXPERIMENT - DIRECT TRAINING ON USD VALUES
========================================================

Train models to predict actual USD pip movements for next 1-hour bar.
No log returns, no post-conversion - direct USD pip values as training targets.

Author: Claude Code Assistant  
Date: 2025-10-31
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
from utils.usd_pip_data_loader import load_usd_pip_data
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proper_usd_experiment")

class ProperUSDPipExperiment:
    """
    Direct USD pip prediction experiment with proper training targets.
    """
    
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id or f"usd_proper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = Path("results") / f"usd_experiment_{self.experiment_id}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tcnae_model = None
        self.models = {}  # Store LightGBM models
        
        logger.info(f"🔬 Initializing PROPER USD PIP experiment: {self.experiment_id}")
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load data with proper USD pip movements as targets.
        """
        logger.info("📊 Loading data with USD pip movement targets...")
        
        # Load data with USD pip targets
        features, usd_movements, instruments = load_usd_pip_data()
        
        n_samples, n_instruments, n_features = features.shape
        logger.info(f"Loaded {n_samples:,} samples, {n_instruments} instruments, {n_features} features")
        
        # Log USD movement statistics
        logger.info(f"USD pip statistics:")
        logger.info(f"  Mean absolute movement: ${np.mean(np.abs(usd_movements)):.2f}")
        logger.info(f"  Max movement: ${np.max(np.abs(usd_movements)):.2f}")
        logger.info(f"  Standard deviation: ${np.std(usd_movements):.2f}")
        
        # Reshape features for TCNAE: (N, sequence_length=4, instruments*features)
        sequence_length = 4
        reshaped_features = []
        reshaped_usd_targets = []
        
        for i in range(sequence_length, n_samples):
            # Create 4-hour sequence
            seq_features = features[i-sequence_length:i]  # (4, n_instruments, 5)
            seq_features = seq_features.reshape(sequence_length, -1)  # (4, n_instruments*5)
            reshaped_features.append(seq_features)
            
            # Target is next hour's USD pip movements
            reshaped_usd_targets.append(usd_movements[i])  # (n_instruments,)
        
        final_features = np.array(reshaped_features)  # (N-4, 4, n_instruments*5)
        final_usd_targets = np.array(reshaped_usd_targets)   # (N-4, n_instruments)
        
        logger.info(f"✅ Prepared {len(final_features):,} sequences for training")
        logger.info(f"✅ Using ACTUAL USD pip movements as training targets")
        
        return final_features, final_usd_targets, instruments
    
    def create_temporal_splits(self, features: np.ndarray, usd_targets: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Create proper temporal splits (no lookahead bias).
        """
        n_samples = len(features)
        
        # Temporal splits: 70% train, 15% val, 15% test
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        X_train = features[:train_end]
        X_val = features[train_end:val_end]
        X_test = features[val_end:]
        
        y_train = usd_targets[:train_end]
        y_val = usd_targets[train_end:val_end]
        y_test = usd_targets[val_end:]
        
        logger.info(f"📊 Temporal splits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def stage1_train_tcnae(self, X_train: np.ndarray, X_val: np.ndarray, 
                          final_features: np.ndarray) -> str:
        """
        Stage 1: Train TCNAE autoencoder for latent feature extraction.
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
        cache_name = f"usd_{self.experiment_id}"
        
        logger.info("🔄 Extracting latent features from trained TCNAE...")
        
        # Extract latents for all data
        with torch.no_grad():
            X_train_latents = self.tcnae_model.encode(X_train_tensor).numpy()
            X_val_latents = self.tcnae_model.encode(X_val_tensor).numpy()
        
        # Cache latents for later stages
        cache_file = self.results_dir / f"cached_latents_{cache_name}.npz"
        np.savez(cache_file, 
                train_latents=X_train_latents, 
                val_latents=X_val_latents)
        logger.info(f"💾 Cached latent features to {cache_file}")
        
        logger.info(f"✅ Stage 1 complete: TCNAE trained and latents cached as '{cache_name}'")
        
        return cache_name
    
    def stage2_train_usd_models(self, cache_name: str, y_train: np.ndarray, 
                               y_val: np.ndarray, instruments: List[str]) -> Dict[str, Any]:
        """
        Stage 2: Train models to predict USD pip movements directly.
        """
        logger.info("🎯 Stage 2: Training USD pip prediction models")
        
        # Load cached latent features
        cache_file = self.results_dir / f"cached_latents_{cache_name}.npz"
        cached_data = np.load(cache_file)
        train_latents = cached_data['train_latents']
        val_latents = cached_data['val_latents']
        
        logger.info(f"Training {len(instruments)} USD pip prediction models")
        
        # Train one model per instrument
        for i, instrument in enumerate(instruments):
            logger.info(f"Training USD pip model for {instrument}")
            
            # Extract USD targets for this instrument
            y_train_instrument = y_train[:, i]
            y_val_instrument = y_val[:, i]
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(train_latents, label=y_train_instrument)
            val_data = lgb.Dataset(val_latents, label=y_val_instrument, reference=train_data)
            
            # LightGBM parameters for regression
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            self.models[instrument] = model
            
            # Log training stats
            train_pred = model.predict(train_latents)
            val_pred = model.predict(val_latents)
            
            train_rmse = np.sqrt(np.mean((y_train_instrument - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val_instrument - val_pred) ** 2))
            
            logger.info(f"  {instrument}: Train RMSE=${train_rmse:.2f}, Val RMSE=${val_rmse:.2f}")
        
        # Save all models
        models_path = self.results_dir / f"usd_models_{self.experiment_id}.json"
        model_strings = {}
        for instrument, model in self.models.items():
            model_strings[instrument] = model.model_to_string()
        
        with open(models_path, 'w') as f:
            json.dump(model_strings, f)
        
        logger.info(f"💾 Saved all USD models to {models_path}")
        logger.info(f"✅ Stage 2 complete: {len(instruments)} USD pip models trained")
        
        return {"models_trained": len(instruments)}
    
    def stage3_evaluate_performance(self, cache_name: str, X_test: np.ndarray, 
                                   y_test: np.ndarray, instruments: List[str]) -> Dict[str, Any]:
        """
        Stage 3: Comprehensive evaluation on test data.
        """
        logger.info("📊 Stage 3: Comprehensive USD pip prediction evaluation")
        
        # Load TCNAE model if needed
        if self.tcnae_model is None:
            logger.info("🔄 Loading trained TCNAE model for evaluation...")
            model_path = self.results_dir / f"tcnae_model_{self.experiment_id}.pth"
            checkpoint = torch.load(model_path, map_location='cpu')
            
            config_dict = checkpoint['config']
            config = TCNAEConfig(**config_dict)
            
            self.tcnae_model = TCNAE(config)
            self.tcnae_model.load_state_dict(checkpoint['model_state_dict'])
            self.tcnae_model.eval()
            logger.info("✅ TCNAE model loaded successfully")
        
        # Extract test latents
        X_test_tensor = torch.FloatTensor(X_test).transpose(1, 2)
        with torch.no_grad():
            X_test_latents = self.tcnae_model.encode(X_test_tensor).numpy()
        
        # Evaluate each instrument
        results = {}
        all_predictions = []
        
        for i, instrument in enumerate(instruments):
            model = self.models[instrument]
            y_true = y_test[:, i]
            y_pred = model.predict(X_test_latents)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            # Direction accuracy
            direction_accuracy = np.mean((y_true > 0) == (y_pred > 0))
            
            # Economic metrics
            total_actual_movement = np.sum(np.abs(y_true))
            total_predicted_movement = np.sum(np.abs(y_pred))
            
            results[instrument] = {
                'rmse_usd': rmse,
                'mae_usd': mae,
                'correlation': correlation,
                'direction_accuracy': direction_accuracy,
                'total_actual_usd': total_actual_movement,
                'total_predicted_usd': total_predicted_movement,
                'n_samples': len(y_true)
            }
            
            all_predictions.extend(y_pred.tolist())
            
            logger.info(f"{instrument}: RMSE=${rmse:.2f}, Correlation={correlation:.4f}, Direction={direction_accuracy:.1%}")
        
        # Aggregate metrics
        aggregate_results = {
            'experiment_summary': {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'n_instruments': len(instruments),
                'n_test_samples': len(X_test),
                'model_architecture': 'TCNAE(120D) -> LightGBM(USD_direct)'
            },
            'performance_by_instrument': results,
            'aggregate_metrics': {
                'avg_rmse_usd': np.mean([r['rmse_usd'] for r in results.values()]),
                'avg_correlation': np.mean([r['correlation'] for r in results.values()]),
                'avg_direction_accuracy': np.mean([r['direction_accuracy'] for r in results.values()]),
                'total_actual_movement': sum([r['total_actual_usd'] for r in results.values()]),
                'total_predicted_movement': sum([r['total_predicted_usd'] for r in results.values()]),
                'best_performer': max(results.keys(), key=lambda k: results[k]['correlation']),
                'worst_performer': min(results.keys(), key=lambda k: results[k]['correlation'])
            },
            'prediction_samples': {
                'instruments': instruments,
                'sample_predictions': all_predictions[:100]  # First 100 predictions
            }
        }
        
        # Save results
        results_path = self.results_dir / f"usd_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(aggregate_results, f, indent=2, default=str)
        
        logger.info(f"💾 Saved evaluation results to {results_path}")
        
        # Print summary
        logger.info("🎯 USD PIP EXPERIMENT RESULTS:")
        logger.info(f"Average RMSE: ${aggregate_results['aggregate_metrics']['avg_rmse_usd']:.2f}")
        logger.info(f"Average Correlation: {aggregate_results['aggregate_metrics']['avg_correlation']:.4f}")
        logger.info(f"Average Direction Accuracy: {aggregate_results['aggregate_metrics']['avg_direction_accuracy']:.1%}")
        logger.info(f"Best Performer: {aggregate_results['aggregate_metrics']['best_performer']}")
        
        return aggregate_results
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        Run the complete USD pip prediction experiment.
        """
        try:
            logger.info("🚀 Starting PROPER USD pip prediction experiment")
            
            # Load and prepare data
            features, usd_targets, instruments = self.load_and_prepare_data()
            
            # Create temporal splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_temporal_splits(features, usd_targets)
            
            # Stage 1: Train TCNAE
            cache_name = self.stage1_train_tcnae(X_train, X_val, features)
            
            # Stage 2: Train USD pip models
            stage2_results = self.stage2_train_usd_models(cache_name, y_train, y_val, instruments)
            
            # Stage 3: Comprehensive evaluation
            final_results = self.stage3_evaluate_performance(cache_name, X_test, y_test, instruments)
            
            # Save experiment summary
            experiment_summary = {
                'experiment_id': self.experiment_id,
                'status': 'COMPLETED',
                'timestamp': datetime.now().isoformat(),
                'experiment_type': 'USD_PIP_DIRECT_TRAINING',
                'stages_completed': ['tcnae_training', 'usd_model_training', 'evaluation'],
                'data_summary': {
                    'total_samples': len(features),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'instruments': len(instruments)
                },
                'final_results': final_results
            }
            
            summary_path = self.results_dir / f"usd_experiment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(experiment_summary, f, indent=2, default=str)
            
            logger.info(f"✅ COMPLETE USD EXPERIMENT FINISHED: {self.experiment_id}")
            logger.info(f"📁 Results saved to: {self.results_dir}")
            
            return experiment_summary
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            raise

def main():
    """Main execution function."""
    logger.info("🔬 Starting PROPER USD Pip Prediction Experiment")
    logger.info("=" * 60)
    
    # Initialize experiment
    experiment = ProperUSDPipExperiment()
    
    # Run complete experiment
    results = experiment.run_complete_experiment()
    
    logger.info("🎉 PROPER USD EXPERIMENT COMPLETE!")
    return results

if __name__ == "__main__":
    results = main()