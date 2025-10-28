#!/usr/bin/env python3
"""
LightGBM GBDT Integration for Market Edge Finder Experiment
Maps TCNAE latent features to 20 instrument predictions with advanced boosting
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.instruments import FX_INSTRUMENTS

logger = logging.getLogger(__name__)


@dataclass
class GBDTConfig:
    """Configuration for LightGBM GBDT model"""
    # Core model parameters
    objective: str = 'regression'
    metric: str = 'rmse'
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    
    # Training parameters
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    verbose_eval: int = 100
    
    # Regularization
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    min_data_in_leaf: int = 20
    min_gain_to_split: float = 0.0
    
    # Advanced parameters
    max_depth: int = -1  # No limit
    min_child_weight: float = 1e-3
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    # Multi-output specific
    num_outputs: int = 20  # 20 FX instruments
    output_names: List[str] = field(default_factory=lambda: FX_INSTRUMENTS)
    
    # Incremental learning
    enable_incremental: bool = True
    incremental_batch_size: int = 1000
    
    def to_lgb_params(self) -> Dict[str, Any]:
        """Convert to LightGBM parameters dictionary"""
        return {
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'min_data_in_leaf': self.min_data_in_leaf,
            'min_gain_to_split': self.min_gain_to_split,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'verbose': -1,
            'seed': 42
        }


class MultiOutputGBDT:
    """
    Multi-output GBDT model for predicting 20 FX instrument returns
    Uses separate LightGBM models for each instrument with shared features
    """
    
    def __init__(self, config: Optional[GBDTConfig] = None):
        """
        Initialize multi-output GBDT
        
        Args:
            config: GBDT configuration
        """
        self.config = config or GBDTConfig()
        self.models: Dict[str, lgb.Booster] = {}
        self.feature_importances: Dict[str, np.ndarray] = {}
        self.training_history: Dict[str, Dict[str, List[float]]] = {}
        self.is_trained = False
        
        logger.info(f"MultiOutputGBDT initialized for {self.config.num_outputs} outputs")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    instrument_names: Optional[List[str]] = None) -> Dict[str, lgb.Dataset]:
        """
        Prepare data for LightGBM training
        
        Args:
            X: Feature matrix [samples, latent_dim]
            y: Target matrix [samples, num_instruments]
            instrument_names: Optional instrument names
            
        Returns:
            Dictionary of LightGBM datasets per instrument
        """
        if instrument_names is None:
            instrument_names = self.config.output_names
        
        if y.shape[1] != len(instrument_names):
            raise ValueError(f"Target shape {y.shape[1]} doesn't match instruments {len(instrument_names)}")
        
        datasets = {}
        
        for i, instrument in enumerate(instrument_names):
            # Extract target for this instrument
            y_instrument = y[:, i]
            
            # Remove NaN values
            valid_mask = ~np.isnan(y_instrument) & ~np.isnan(X).any(axis=1)
            
            if np.sum(valid_mask) < 100:
                logger.warning(f"Insufficient valid data for {instrument}: {np.sum(valid_mask)} samples")
                continue
            
            X_clean = X[valid_mask]
            y_clean = y_instrument[valid_mask]
            
            # Create LightGBM dataset
            lgb_data = lgb.Dataset(
                X_clean,
                label=y_clean,
                feature_name=[f'latent_{j}' for j in range(X.shape[1])],
                free_raw_data=False
            )
            
            datasets[instrument] = lgb_data
            
            logger.debug(f"Prepared dataset for {instrument}: {X_clean.shape[0]} samples, "
                        f"feature range [{X_clean.min():.3f}, {X_clean.max():.3f}], "
                        f"target range [{y_clean.min():.6f}, {y_clean.max():.6f}]")
        
        return datasets
    
    def train_single_instrument(self, instrument: str, train_data: lgb.Dataset, 
                               valid_data: Optional[lgb.Dataset] = None) -> lgb.Booster:
        """
        Train GBDT model for single instrument
        
        Args:
            instrument: Instrument name
            train_data: Training dataset
            valid_data: Optional validation dataset
            
        Returns:
            Trained LightGBM booster
        """
        logger.info(f"Training GBDT for {instrument}")
        
        # Prepare validation sets
        valid_sets = [train_data]
        valid_names = ['train']
        
        if valid_data is not None:
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Initialize callbacks
        callbacks = []
        
        # Early stopping if validation data provided
        if valid_data is not None:
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds))
        
        # Record evaluation results
        eval_results = {}
        callbacks.append(lgb.record_evaluation(eval_results))
        
        try:
            # Train model
            model = lgb.train(
                params=self.config.to_lgb_params(),
                train_set=train_data,
                num_boost_round=self.config.num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            
            # Store training history
            self.training_history[instrument] = eval_results
            
            # Store feature importance
            self.feature_importances[instrument] = model.feature_importance(importance_type='gain')
            
            logger.info(f"✅ {instrument}: Training completed, {model.num_trees()} trees, "
                       f"best iteration: {model.best_iteration}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ {instrument}: Training failed - {e}")
            raise
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            instrument_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train GBDT models for all instruments
        
        Args:
            X: Training features [samples, latent_dim]
            y: Training targets [samples, num_instruments]
            X_val: Optional validation features
            y_val: Optional validation targets
            instrument_names: Optional instrument names
            
        Returns:
            Training summary dictionary
        """
        logger.info(f"Starting multi-output GBDT training: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Prepare training data
        train_datasets = self.prepare_data(X, y, instrument_names)
        
        # Prepare validation data if provided
        val_datasets = {}
        if X_val is not None and y_val is not None:
            val_datasets = self.prepare_data(X_val, y_val, instrument_names)
        
        # Train models for each instrument
        training_summary = {
            'successful_instruments': [],
            'failed_instruments': [],
            'training_metrics': {}
        }
        
        for instrument in train_datasets.keys():
            try:
                train_data = train_datasets[instrument]
                val_data = val_datasets.get(instrument, None)
                
                # Train model
                model = self.train_single_instrument(instrument, train_data, val_data)
                self.models[instrument] = model
                
                # Calculate training metrics
                train_pred = model.predict(train_data.get_data())
                train_mse = mean_squared_error(train_data.get_label(), train_pred)
                train_mae = mean_absolute_error(train_data.get_label(), train_pred)
                
                metrics = {
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'num_trees': model.num_trees(),
                    'best_iteration': model.best_iteration
                }
                
                # Validation metrics if available
                if val_data is not None:
                    val_pred = model.predict(val_data.get_data())
                    val_mse = mean_squared_error(val_data.get_label(), val_pred)
                    val_mae = mean_absolute_error(val_data.get_label(), val_pred)
                    
                    metrics.update({
                        'val_mse': val_mse,
                        'val_mae': val_mae
                    })
                
                training_summary['training_metrics'][instrument] = metrics
                training_summary['successful_instruments'].append(instrument)
                
            except Exception as e:
                logger.error(f"Training failed for {instrument}: {e}")
                training_summary['failed_instruments'].append(instrument)
        
        self.is_trained = len(self.models) > 0
        
        logger.info(f"Multi-output GBDT training completed: "
                   f"{len(training_summary['successful_instruments'])}/{len(train_datasets)} successful")
        
        return training_summary
    
    def predict(self, X: np.ndarray, instrument_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate predictions for all instruments
        
        Args:
            X: Feature matrix [samples, latent_dim]
            instrument_names: Optional instrument names
            
        Returns:
            Predictions matrix [samples, num_instruments]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if instrument_names is None:
            instrument_names = self.config.output_names
        
        num_samples = X.shape[0]
        predictions = np.full((num_samples, len(instrument_names)), np.nan)
        
        for i, instrument in enumerate(instrument_names):
            if instrument in self.models:
                try:
                    pred = self.models[instrument].predict(X)
                    predictions[:, i] = pred
                except Exception as e:
                    logger.warning(f"Prediction failed for {instrument}: {e}")
            else:
                logger.warning(f"No trained model for {instrument}")
        
        return predictions
    
    def predict_single(self, X: np.ndarray, instrument: str) -> np.ndarray:
        """
        Generate predictions for single instrument
        
        Args:
            X: Feature matrix [samples, latent_dim]
            instrument: Instrument name
            
        Returns:
            Predictions array [samples]
        """
        if instrument not in self.models:
            raise ValueError(f"No trained model for {instrument}")
        
        return self.models[instrument].predict(X)
    
    def update_incremental(self, X_new: np.ndarray, y_new: np.ndarray, 
                          instrument_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Update models incrementally with new data
        
        Args:
            X_new: New feature matrix [samples, latent_dim]
            y_new: New target matrix [samples, num_instruments]
            instrument_names: Optional instrument names
            
        Returns:
            Dictionary indicating success for each instrument
        """
        if not self.config.enable_incremental:
            raise ValueError("Incremental learning not enabled in config")
        
        if instrument_names is None:
            instrument_names = self.config.output_names
        
        update_results = {}
        
        for i, instrument in enumerate(instrument_names):
            if instrument not in self.models:
                update_results[instrument] = False
                continue
            
            try:
                # Prepare new data
                y_instrument = y_new[:, i]
                valid_mask = ~np.isnan(y_instrument) & ~np.isnan(X_new).any(axis=1)
                
                if np.sum(valid_mask) < 10:
                    logger.warning(f"Insufficient new data for {instrument}: {np.sum(valid_mask)} samples")
                    update_results[instrument] = False
                    continue
                
                X_clean = X_new[valid_mask]
                y_clean = y_instrument[valid_mask]
                
                # Create new dataset
                new_data = lgb.Dataset(
                    X_clean,
                    label=y_clean,
                    feature_name=[f'latent_{j}' for j in range(X_new.shape[1])],
                    reference=None  # Important for incremental learning
                )
                
                # Update model (retraining approach since LightGBM doesn't support true incremental learning)
                updated_model = lgb.train(
                    params=self.config.to_lgb_params(),
                    train_set=new_data,
                    num_boost_round=min(100, self.config.num_boost_round // 10),  # Fewer iterations
                    init_model=self.models[instrument],
                    keep_training_booster=True
                )
                
                self.models[instrument] = updated_model
                update_results[instrument] = True
                
                logger.debug(f"Updated model for {instrument} with {X_clean.shape[0]} new samples")
                
            except Exception as e:
                logger.error(f"Incremental update failed for {instrument}: {e}")
                update_results[instrument] = False
        
        return update_results
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance across all instruments
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'cover')
            
        Returns:
            DataFrame with feature importance per instrument
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_data = {}
        
        for instrument, model in self.models.items():
            importance = model.feature_importance(importance_type=importance_type)
            importance_data[instrument] = importance
        
        # Create DataFrame
        feature_names = [f'latent_{i}' for i in range(len(next(iter(importance_data.values()))))]
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        
        return importance_df
    
    def save_models(self, save_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
            
        Returns:
            Dictionary mapping instruments to saved file paths
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save individual models
        for instrument, model in self.models.items():
            model_path = save_dir / f'{instrument}_gbdt_model.txt'
            model.save_model(str(model_path))
            saved_paths[instrument] = str(model_path)
        
        # Save configuration and metadata
        config_path = save_dir / 'gbdt_config.json'
        with open(config_path, 'w') as f:
            # Convert config to dict for JSON serialization
            config_dict = {
                'objective': self.config.objective,
                'num_outputs': self.config.num_outputs,
                'output_names': self.config.output_names,
                'learning_rate': self.config.learning_rate,
                'num_boost_round': self.config.num_boost_round
            }
            json.dump(config_dict, f, indent=2)
        
        # Save feature importances
        if self.feature_importances:
            importance_path = save_dir / 'feature_importances.pkl'
            with open(importance_path, 'wb') as f:
                pickle.dump(self.feature_importances, f)
        
        logger.info(f"Saved {len(saved_paths)} GBDT models to {save_dir}")
        return saved_paths
    
    def load_models(self, save_dir: Union[str, Path]) -> bool:
        """
        Load trained models
        
        Args:
            save_dir: Directory containing saved models
            
        Returns:
            True if successful, False otherwise
        """
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            logger.error(f"Save directory does not exist: {save_dir}")
            return False
        
        try:
            # Load configuration
            config_path = save_dir / 'gbdt_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    # Update current config with saved values
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            
            # Load individual models
            loaded_count = 0
            for instrument in self.config.output_names:
                model_path = save_dir / f'{instrument}_gbdt_model.txt'
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
                    self.models[instrument] = model
                    loaded_count += 1
            
            # Load feature importances
            importance_path = save_dir / 'feature_importances.pkl'
            if importance_path.exists():
                with open(importance_path, 'rb') as f:
                    self.feature_importances = pickle.load(f)
            
            self.is_trained = loaded_count > 0
            
            logger.info(f"Loaded {loaded_count} GBDT models from {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False


# Utility functions for cooperative learning
class CooperativeGBDTManager:
    """
    Manager for cooperative learning between TCNAE and GBDT
    Handles residual learning and model coordination
    """
    
    def __init__(self, gbdt_model: MultiOutputGBDT, update_frequency: int = 500):
        """
        Initialize cooperative manager
        
        Args:
            gbdt_model: Multi-output GBDT model
            update_frequency: How often to update GBDT (in batches)
        """
        self.gbdt_model = gbdt_model
        self.update_frequency = update_frequency
        self.batch_counter = 0
        self.residual_buffer = {
            'features': [],
            'residuals': []
        }
        
        logger.info(f"CooperativeGBDTManager initialized with update frequency: {update_frequency}")
    
    def accumulate_residuals(self, latent_features: np.ndarray, tcn_predictions: np.ndarray, 
                           true_targets: np.ndarray) -> bool:
        """
        Accumulate residuals for batch update
        
        Args:
            latent_features: Latent features from TCNAE [batch_size, latent_dim]
            tcn_predictions: Predictions from TCN [batch_size, num_instruments]
            true_targets: True target values [batch_size, num_instruments]
            
        Returns:
            True if update should be triggered, False otherwise
        """
        # Calculate residuals
        residuals = true_targets - tcn_predictions
        
        # Store in buffer
        self.residual_buffer['features'].append(latent_features)
        self.residual_buffer['residuals'].append(residuals)
        
        self.batch_counter += 1
        
        # Check if update should be triggered
        return self.batch_counter % self.update_frequency == 0
    
    def update_gbdt_residuals(self) -> bool:
        """
        Update GBDT model with accumulated residuals
        
        Returns:
            True if update successful, False otherwise
        """
        if not self.residual_buffer['features']:
            logger.warning("No residuals accumulated for GBDT update")
            return False
        
        try:
            # Concatenate buffered data
            X_residual = np.vstack(self.residual_buffer['features'])
            y_residual = np.vstack(self.residual_buffer['residuals'])
            
            # Update GBDT with residuals
            if self.gbdt_model.is_trained:
                update_results = self.gbdt_model.update_incremental(X_residual, y_residual)
                success_rate = sum(update_results.values()) / len(update_results)
                
                logger.info(f"GBDT residual update: {success_rate:.1%} instruments successful")
            else:
                # Initial training if not yet trained
                training_summary = self.gbdt_model.fit(X_residual, y_residual)
                success_rate = len(training_summary['successful_instruments']) / len(FX_INSTRUMENTS)
                
                logger.info(f"GBDT initial training: {success_rate:.1%} instruments successful")
            
            # Clear buffer
            self.residual_buffer = {'features': [], 'residuals': []}
            
            return success_rate > 0.5  # Consider successful if >50% of instruments updated
            
        except Exception as e:
            logger.error(f"GBDT residual update failed: {e}")
            return False
    
    def get_combined_predictions(self, latent_features: np.ndarray, tcn_predictions: np.ndarray) -> np.ndarray:
        """
        Get combined TCN + GBDT predictions
        
        Args:
            latent_features: Latent features from TCNAE [batch_size, latent_dim]
            tcn_predictions: Predictions from TCN [batch_size, num_instruments]
            
        Returns:
            Combined predictions [batch_size, num_instruments]
        """
        if not self.gbdt_model.is_trained:
            return tcn_predictions
        
        try:
            # Get GBDT residual predictions
            gbdt_residuals = self.gbdt_model.predict(latent_features)
            
            # Combine predictions
            combined_predictions = tcn_predictions + gbdt_residuals
            
            return combined_predictions
            
        except Exception as e:
            logger.warning(f"Failed to get GBDT predictions, using TCN only: {e}")
            return tcn_predictions


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    latent_dim = 120
    n_instruments = 24
    
    # Simulate latent features from TCNAE
    X = np.random.randn(n_samples, latent_dim)
    
    # Simulate correlated targets (FX returns)
    correlation_matrix = np.random.uniform(0.1, 0.9, (n_instruments, latent_dim))
    noise = np.random.normal(0, 0.001, (n_samples, n_instruments))
    y = X @ correlation_matrix.T + noise
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Testing Multi-output GBDT:")
    print(f"Training data: {X_train.shape}, targets: {y_train.shape}")
    print(f"Validation data: {X_val.shape}, targets: {y_val.shape}")
    
    # Test GBDT model
    config = GBDTConfig(
        num_boost_round=100,
        early_stopping_rounds=20,
        learning_rate=0.1
    )
    
    gbdt = MultiOutputGBDT(config)
    
    # Train model
    training_summary = gbdt.fit(X_train, y_train, X_val, y_val)
    
    print(f"\nTraining Results:")
    print(f"Successful instruments: {len(training_summary['successful_instruments'])}")
    print(f"Failed instruments: {len(training_summary['failed_instruments'])}")
    
    # Test predictions
    predictions = gbdt.predict(X_val)
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Calculate validation metrics
    val_mse = []
    val_mae = []
    
    for i, instrument in enumerate(FX_INSTRUMENTS):
        if not np.isnan(predictions[:, i]).all():
            mse = mean_squared_error(y_val[:, i], predictions[:, i])
            mae = mean_absolute_error(y_val[:, i], predictions[:, i])
            val_mse.append(mse)
            val_mae.append(mae)
    
    if val_mse:
        print(f"Average validation MSE: {np.mean(val_mse):.6f}")
        print(f"Average validation MAE: {np.mean(val_mae):.6f}")
    
    # Test feature importance
    importance_df = gbdt.get_feature_importance()
    print(f"\nFeature importance shape: {importance_df.shape}")
    print(f"Top 5 most important features (average):")
    avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    print(avg_importance.head())
    
    # Test save/load
    save_dir = Path('test_gbdt_models')
    if save_dir.exists():
        import shutil
        shutil.rmtree(save_dir)
    
    saved_paths = gbdt.save_models(save_dir)
    print(f"\nSaved {len(saved_paths)} models")
    
    # Test loading
    gbdt_loaded = MultiOutputGBDT(config)
    load_success = gbdt_loaded.load_models(save_dir)
    print(f"Load successful: {load_success}")
    
    if load_success:
        # Test loaded model predictions
        loaded_predictions = gbdt_loaded.predict(X_val[:10])
        original_predictions = predictions[:10]
        
        diff = np.abs(loaded_predictions - original_predictions)
        print(f"Max prediction difference after load: {np.max(diff):.8f}")
    
    # Clean up
    if save_dir.exists():
        import shutil
        shutil.rmtree(save_dir)
    
    print("✅ Multi-output GBDT test completed successfully")