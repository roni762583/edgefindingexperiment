#!/usr/bin/env python3
"""
Training script for Market Edge Finder Experiment.

Executes the 4-stage hybrid training pipeline combining TCNAE and LightGBM
with context tensor management and adaptive teacher forcing.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from typing import Dict, Optional, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.config import load_config, SystemConfig
from training.train_hybrid import HybridTrainer, HybridDataLoader
from models.tcnae import TCNAE, TCNAEConfig
from models.gbdt_model import MultiOutputGBDT, GBDTConfig
from models.context_manager import ContextTensorManager
from utils.logger import setup_logging
import pickle

# Configure logging
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Complete training orchestration for the hybrid model system.
    
    Manages data loading, model initialization, training execution,
    evaluation, and model persistence.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the training orchestrator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.device = self._get_device()
        
        # Initialize components
        self.data_loader: Optional[HybridDataLoader] = None
        self.trainer: Optional[HybridTrainer] = None
        
        # Ensure output directories exist
        Path(config.models_path).mkdir(parents=True, exist_ok=True)
        Path(config.results_path).mkdir(parents=True, exist_ok=True)
        Path(config.logs_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training orchestrator initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device for training."""
        device_config = self.config.model.device.lower()
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Auto-selected Metal Performance Shaders (MPS) device")
            else:
                device = torch.device('cpu')
                logger.info("Auto-selected CPU device")
        else:
            device = torch.device(device_config)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def load_training_data(self) -> None:
        """Load and prepare training data."""
        logger.info("Loading training data...")
        
        data_path = Path(self.config.data_path)
        
        # Check if preprocessed data exists
        train_dir = data_path / "train"
        val_dir = data_path / "validation"
        
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed training data not found. "
                f"Please run preprocessing first: python scripts/run_preprocessing.py"
            )
        
        # Load training data
        train_features = {}
        val_features = {}
        train_targets = {}
        val_targets = {}
        
        for instrument in self.config.data.instruments:
            try:
                # Load features
                train_feature_file = train_dir / f"{instrument}.parquet"
                val_feature_file = val_dir / f"{instrument}.parquet"
                
                if train_feature_file.exists() and val_feature_file.exists():
                    train_features[instrument] = pd.read_parquet(train_feature_file)
                    val_features[instrument] = pd.read_parquet(val_feature_file)
                    
                    # Load targets
                    targets_file = data_path / f"{instrument}_targets.parquet"
                    if targets_file.exists():
                        targets = pd.read_parquet(targets_file)
                        
                        # Split targets based on timestamps
                        train_end_time = train_features[instrument]['timestamp'].max()
                        
                        train_targets[instrument] = targets[targets['timestamp'] <= train_end_time]
                        val_targets[instrument] = targets[targets['timestamp'] > train_end_time]
                    else:
                        logger.warning(f"No targets found for {instrument}")
                        continue
                else:
                    logger.warning(f"No training data found for {instrument}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error loading data for {instrument}: {str(e)}")
                continue
        
        # Initialize data loader
        self.data_loader = HybridDataLoader(
            train_features=train_features,
            val_features=val_features,
            train_targets=train_targets,
            val_targets=val_targets,
            sequence_length=self.config.model.tcnae_sequence_length,
            batch_size=self.config.training.batch_size,
            device=self.device
        )
        
        logger.info(f"Training data loaded for {len(train_features)} instruments")
        logger.info(f"Training sequences: {len(self.data_loader.train_sequences)}")
        logger.info(f"Validation sequences: {len(self.data_loader.val_sequences)}")
    
    def initialize_models(self) -> None:
        """Initialize all model components."""
        logger.info("Initializing models...")
        
        # TCNAE configuration
        tcnae_config = TCNAEConfig(
            input_channels=self.config.model.tcnae_input_channels,
            sequence_length=self.config.model.tcnae_sequence_length,
            latent_dim=self.config.model.tcnae_latent_dim,
            num_instruments=self.config.model.num_instruments,
            hidden_channels=self.config.model.tcnae_hidden_channels,
            kernel_size=self.config.model.tcnae_kernel_size,
            dropout=self.config.model.tcnae_dropout
        )
        
        # LightGBM configuration
        gbdt_config = GBDTConfig(
            num_instruments=self.config.model.num_instruments,
            n_estimators=self.config.model.gbdt_n_estimators,
            max_depth=self.config.model.gbdt_max_depth,
            learning_rate=self.config.model.gbdt_learning_rate,
            subsample=self.config.model.gbdt_subsample,
            colsample_bytree=self.config.model.gbdt_colsample_bytree,
            reg_alpha=self.config.model.gbdt_reg_alpha,
            reg_lambda=self.config.model.gbdt_reg_lambda
        )
        
        # Initialize trainer
        self.trainer = HybridTrainer(
            tcnae_config=tcnae_config,
            gbdt_config=gbdt_config,
            training_config=self.config.training,
            device=self.device
        )
        
        logger.info("Models initialized successfully")
    
    def run_training(self) -> Dict[str, Any]:
        """
        Execute the complete 4-stage training pipeline.
        
        Returns:
            Training results and metrics
        """
        logger.info("Starting 4-stage hybrid training pipeline...")
        
        if not self.data_loader or not self.trainer:
            raise ValueError("Data loader and trainer must be initialized first")
        
        # Execute training
        training_results = self.trainer.train_full_pipeline(self.data_loader)
        
        logger.info("Training pipeline completed successfully")
        return training_results
    
    def save_models(self, training_results: Dict[str, Any]) -> None:
        """
        Save all trained models and components.
        
        Args:
            training_results: Results from training
        """
        logger.info("Saving trained models...")
        
        models_path = Path(self.config.models_path)
        
        # Save TCNAE model
        tcnae_path = models_path / "tcnae_best.pth"
        torch.save({
            'model_state_dict': self.trainer.tcnae_model.state_dict(),
            'config': self.trainer.tcnae_config.__dict__,
            'training_results': training_results.get('stage1_results', {})
        }, tcnae_path)
        logger.info(f"TCNAE model saved to {tcnae_path}")
        
        # Save LightGBM model
        gbdt_path = models_path / "gbdt_best.pkl"
        self.trainer.gbdt_model.save_model(str(gbdt_path))
        logger.info(f"LightGBM model saved to {gbdt_path}")
        
        # Save context manager
        context_path = models_path / "context_manager.pkl"
        context_state = self.trainer.context_manager.get_state()
        with open(context_path, 'wb') as f:
            pickle.dump(context_state, f)
        logger.info(f"Context manager saved to {context_path}")
        
        # Save normalizer (if available)
        normalizer_source = Path(self.config.data_path) / "feature_normalizer.pkl"
        normalizer_dest = models_path / "normalizer.pkl"
        if normalizer_source.exists():
            import shutil
            shutil.copy2(normalizer_source, normalizer_dest)
            logger.info(f"Normalizer copied to {normalizer_dest}")
        
        # Save training configuration and results
        results_path = Path(self.config.results_path) / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_json_serializable(training_results)
        
        with open(results_path, 'w') as f:
            json.dump({
                'config': {
                    'model': self.config.model.__dict__,
                    'training': self.config.training.__dict__,
                    'data': self.config.data.__dict__
                },
                'device': str(self.device),
                'training_results': serializable_results
            }, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate trained models on validation data.
        
        Returns:
            Evaluation metrics and results
        """
        logger.info("Evaluating trained models...")
        
        if not self.trainer or not self.data_loader:
            raise ValueError("Trainer and data loader must be initialized")
        
        # Get validation predictions
        val_predictions = []
        val_targets = []
        
        # Evaluate in batches
        self.trainer.tcnae_model.eval()
        
        with torch.no_grad():
            for batch_features, batch_targets in self.data_loader.get_validation_batches():
                # Generate TCNAE features
                _, latent_features = self.trainer.tcnae_model(batch_features)
                
                # Get context
                context = self.trainer.context_manager.get_current_context()
                
                # Combine features
                combined_features = latent_features + context.unsqueeze(0)
                
                # Prepare for LightGBM
                features_np = combined_features.cpu().numpy()
                features_flat = features_np.transpose(0, 2, 1).reshape(-1, features_np.shape[1])
                
                # Generate predictions
                predictions = self.trainer.gbdt_model.predict(features_flat)
                
                val_predictions.append(predictions)
                val_targets.append(batch_targets.cpu().numpy())
        
        # Combine all predictions and targets
        all_predictions = np.vstack(val_predictions)
        all_targets = np.vstack(val_targets)
        
        # Calculate evaluation metrics
        evaluation_results = self._calculate_evaluation_metrics(all_predictions, all_targets)
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def _calculate_evaluation_metrics(self, 
                                    predictions: np.ndarray, 
                                    targets: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {}
        
        # Overall metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Correlation
        correlation_matrix = np.corrcoef(predictions.flatten(), targets.flatten())
        correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        target_direction = np.sign(targets)
        directional_accuracy = np.mean(pred_direction == target_direction)
        
        metrics['overall'] = {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation),
            'directional_accuracy': float(directional_accuracy)
        }
        
        # Per-instrument metrics
        metrics['per_instrument'] = {}
        
        for i in range(predictions.shape[1]):
            instrument = self.config.data.instruments[i] if i < len(self.config.data.instruments) else f"instrument_{i}"
            
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            mse_i = np.mean((pred_i - target_i) ** 2)
            mae_i = np.mean(np.abs(pred_i - target_i))
            corr_i = np.corrcoef(pred_i, target_i)[0, 1] if not np.isnan(np.corrcoef(pred_i, target_i)[0, 1]) else 0.0
            acc_i = np.mean(np.sign(pred_i) == np.sign(target_i))
            
            metrics['per_instrument'][instrument] = {
                'mse': float(mse_i),
                'mae': float(mae_i),
                'correlation': float(corr_i),
                'directional_accuracy': float(acc_i)
            }
        
        return metrics
    
    def run_complete_training(self) -> None:
        """Run the complete training pipeline from start to finish."""
        logger.info("Starting complete training pipeline...")
        
        try:
            # Step 1: Load data
            self.load_training_data()
            
            # Step 2: Initialize models
            self.initialize_models()
            
            # Step 3: Execute training
            training_results = self.run_training()
            
            # Step 4: Evaluate models
            evaluation_results = self.evaluate_models()
            
            # Step 5: Save everything
            combined_results = {
                **training_results,
                'evaluation': evaluation_results
            }
            self.save_models(combined_results)
            
            # Print summary
            self._print_training_summary(combined_results)
            
            logger.info("Complete training pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def _print_training_summary(self, results: Dict[str, Any]) -> None:
        """Print training summary to console."""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        # Stage results
        for stage in ['stage1_results', 'stage2_results', 'stage3_results', 'stage4_results']:
            if stage in results:
                stage_num = stage.split('_')[0].replace('stage', '')
                stage_name = {
                    '1': 'TCNAE Pretraining',
                    '2': 'Hybrid Training', 
                    '3': 'Cooperative Learning',
                    '4': 'Adaptive Teacher Forcing'
                }.get(stage_num, stage)
                
                print(f"\nStage {stage_num}: {stage_name}")
                print("-" * 40)
                
                stage_results = results[stage]
                if 'final_loss' in stage_results:
                    print(f"Final Loss: {stage_results['final_loss']:.6f}")
                if 'best_loss' in stage_results:
                    print(f"Best Loss: {stage_results['best_loss']:.6f}")
                if 'training_time' in stage_results:
                    print(f"Training Time: {stage_results['training_time']:.2f}s")
        
        # Evaluation results
        if 'evaluation' in results:
            eval_results = results['evaluation']
            print(f"\nFINAL EVALUATION")
            print("-" * 40)
            
            overall = eval_results.get('overall', {})
            print(f"Overall Correlation: {overall.get('correlation', 0):.4f}")
            print(f"Overall Directional Accuracy: {overall.get('directional_accuracy', 0):.4f}")
            print(f"Overall MSE: {overall.get('mse', 0):.6f}")
            print(f"Overall MAE: {overall.get('mae', 0):.6f}")
            
            # Top performing instruments
            per_instrument = eval_results.get('per_instrument', {})
            if per_instrument:
                correlations = [(inst, metrics['correlation']) for inst, metrics in per_instrument.items()]
                correlations.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\nTop 5 Instruments by Correlation:")
                for i, (instrument, corr) in enumerate(correlations[:5]):
                    print(f"  {i+1}. {instrument}: {corr:.4f}")
        
        print("\n" + "="*80)


def main():
    """Main training execution function."""
    parser = argparse.ArgumentParser(description='Market Edge Finder Hybrid Training')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--device', type=str, help='Override device selection (cpu, cuda, mps)')
    parser.add_argument('--epochs', type=int, help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config parameters if specified
        if args.device:
            config.model.device = args.device
        if args.epochs:
            config.training.num_epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        
        # Setup logging
        setup_logging(
            level=args.log_level,
            log_file=Path(config.logs_path) / "training.log"
        )
        
        logger.info("Starting Market Edge Finder hybrid training")
        logger.info(f"Configuration: {args.config or 'default'}")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Device: {config.model.device}")
        logger.info(f"Training epochs: {config.training.num_epochs}")
        
        # Initialize orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Run complete training
        orchestrator.run_complete_training()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()