"""
Simple LightGBM wrapper for Market Edge Finder Experiment

Multi-output gradient boosting for 24-instrument prediction using TCNAE latent features.
Production-ready implementation with proper error handling and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import pickle
import yaml
import joblib
from dataclasses import dataclass

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM not installed. Run: pip install lightgbm")

logger = logging.getLogger(__name__)


@dataclass
class GBDTConfig:
    """Configuration for LightGBM model"""
    
    # Core parameters
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_child_samples: int = 20
    feature_fraction: float = 0.8
    
    # Training parameters
    objective: str = 'regression'
    metric: str = 'rmse'
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    
    # Multi-output parameters
    num_instruments: int = 24
    latent_dim: int = 120
    
    # Training settings
    early_stopping_rounds: int = 50
    verbose: int = 100
    random_state: int = 42
    
    # Validation
    validation_fraction: float = 0.2
    cross_validation_folds: int = 5


class MultiOutputGBDT:
    """
    Multi-output LightGBM for predicting 24 FX instrument returns.
    
    Uses TCNAE latent features (120-dim) to predict USD-scaled pip targets
    for all 24 instruments simultaneously.
    """
    
    def __init__(self, config: Optional[GBDTConfig] = None):
        """
        Initialize multi-output GBDT.
        
        Args:
            config: GBDT configuration
        """
        self.config = config or GBDTConfig()
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.instruments: List[str] = []
        self.is_fitted = False
        self.feature_importance_: Optional[Dict[str, np.ndarray]] = None
        self.training_history_: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized MultiOutputGBDT for {self.config.num_instruments} instruments")
    
    def _create_base_model(self) -> lgb.LGBMRegressor:
        """Create a base LightGBM model with configured parameters."""
        return lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            min_child_samples=self.config.min_child_samples,
            feature_fraction=self.config.feature_fraction,
            objective=self.config.objective,
            metric=self.config.metric,
            boosting_type=self.config.boosting_type,
            num_leaves=self.config.num_leaves,
            random_state=self.config.random_state,
            verbose=-1  # Suppress individual model output
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            instruments: List[str],
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'MultiOutputGBDT':
        """
        Fit multi-output GBDT models.
        
        Args:
            X: Feature matrix [n_samples, latent_dim]
            y: Target matrix [n_samples, num_instruments]
            instruments: List of instrument names
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (optional)
            
        Returns:
            Fitted model instance
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        # Input validation
        if X.shape[1] != self.config.latent_dim:
            raise ValueError(f"Expected {self.config.latent_dim} features, got {X.shape[1]}")
        
        if y.shape[1] != len(instruments):
            raise ValueError(f"Target shape {y.shape[1]} doesn't match instruments {len(instruments)}")
        
        if len(instruments) != self.config.num_instruments:
            logger.warning(f"Expected {self.config.num_instruments} instruments, got {len(instruments)}")
        
        self.instruments = instruments.copy()
        self.models = {}
        self.training_history_ = {}
        
        logger.info(f"Training GBDT for {len(instruments)} instruments on {len(X)} samples")
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, None)]  # Will be set per instrument
        
        # Train one model per instrument
        for i, instrument in enumerate(instruments):
            logger.debug(f"Training model {i+1}/{len(instruments)}: {instrument}")
            
            # Extract targets for this instrument
            y_instrument = y[:, i]
            
            # Skip if all targets are NaN
            valid_mask = ~np.isnan(y_instrument)
            if valid_mask.sum() == 0:
                logger.warning(f"No valid targets for {instrument}, skipping")
                continue
            
            # Filter to valid samples
            X_valid = X[valid_mask]
            y_valid = y_instrument[valid_mask]
            weights_valid = sample_weight[valid_mask] if sample_weight is not None else None
            
            # Prepare validation set for this instrument
            current_eval_set = None
            if eval_set is not None:
                y_val_instrument = y_val[:, i]
                val_valid_mask = ~np.isnan(y_val_instrument)
                if val_valid_mask.sum() > 0:
                    current_eval_set = [(X_val[val_valid_mask], y_val_instrument[val_valid_mask])]
            
            # Create and train model
            model = self._create_base_model()
            
            try:
                model.fit(
                    X_valid, y_valid,
                    sample_weight=weights_valid,
                    eval_set=current_eval_set,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
                        lgb.log_evaluation(period=self.config.verbose)
                    ] if current_eval_set else None
                )
                
                self.models[instrument] = model
                
                # Store training history
                if hasattr(model, 'evals_result_') and model.evals_result_:
                    self.training_history_[instrument] = model.evals_result_
                
                logger.debug(f"✅ {instrument}: {model.best_iteration_} iterations, "
                           f"score: {model.best_score_:.6f}")
                
            except Exception as e:
                logger.error(f"Failed to train model for {instrument}: {e}")
                continue
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        self.is_fitted = True
        logger.info(f"✅ MultiOutputGBDT training completed: {len(self.models)}/{len(instruments)} models")
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict targets for all instruments.
        
        Args:
            X: Feature matrix [n_samples, latent_dim]
            return_std: Whether to return prediction uncertainty (not implemented)
            
        Returns:
            Predictions [n_samples, num_instruments] or tuple with uncertainties
            
        Raises:
            ValueError: If model not fitted or invalid input shape
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X.shape[1] != self.config.latent_dim:
            raise ValueError(f"Expected {self.config.latent_dim} features, got {X.shape[1]}")
        
        n_samples = X.shape[0]
        predictions = np.full((n_samples, len(self.instruments)), np.nan, dtype=np.float32)
        
        # Generate predictions for each instrument
        for i, instrument in enumerate(self.instruments):
            if instrument in self.models:
                try:
                    pred = self.models[instrument].predict(X)
                    predictions[:, i] = pred
                except Exception as e:
                    logger.warning(f"Prediction failed for {instrument}: {e}")
        
        if return_std:
            # Placeholder for uncertainty estimation
            # Could implement using ensemble methods or quantile regression
            std = np.zeros_like(predictions)
            return predictions, std
        
        return predictions
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance across all models."""
        if not self.models:
            self.feature_importance_ = None
            return
        
        importance_dict = {}
        
        for instrument, model in self.models.items():
            importance_dict[instrument] = model.feature_importances_
        
        self.feature_importance_ = importance_dict
        
        # Calculate average importance across all models
        if importance_dict:
            all_importances = np.array(list(importance_dict.values()))
            avg_importance = np.mean(all_importances, axis=0)
            importance_dict['average'] = avg_importance
            
            logger.info(f"Feature importance calculated for {len(self.models)} models")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, np.ndarray]:
        """
        Get feature importance for all models.
        
        Args:
            importance_type: Type of importance ('gain', 'split', etc.)
            
        Returns:
            Dictionary of feature importances per instrument
        """
        if self.feature_importance_ is None:
            logger.warning("Feature importance not available")
            return {}
        
        return self.feature_importance_.copy()
    
    def validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Validate model performance.
        
        Args:
            X: Validation features
            y: Validation targets
            
        Returns:
            Dictionary of validation metrics per instrument
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before validation")
        
        predictions = self.predict(X)
        metrics = {}
        
        for i, instrument in enumerate(self.instruments):
            if instrument not in self.models:
                continue
            
            y_true = y[:, i]
            y_pred = predictions[:, i]
            
            # Filter valid predictions
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            
            if valid_mask.sum() == 0:
                metrics[instrument] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
                continue
            
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))
            mae = np.mean(np.abs(y_true_valid - y_pred_valid))
            
            # R² calculation
            ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
            ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            metrics[instrument] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_samples': len(y_true_valid)
            }
        
        return metrics
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        save_data = {
            'config': self.config,
            'models': self.models,
            'instruments': self.instruments,
            'feature_importance_': self.feature_importance_,
            'training_history_': self.training_history_,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MultiOutputGBDT':
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        model = cls(save_data['config'])
        model.models = save_data['models']
        model.instruments = save_data['instruments']
        model.feature_importance_ = save_data['feature_importance_']
        model.training_history_ = save_data['training_history_']
        model.is_fitted = save_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return model


def load_config_from_yaml(config_path: Union[str, Path]) -> GBDTConfig:
    """
    Load GBDT configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        GBDT configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model configuration
    model_config = config_dict.get('model', {})
    
    # Map YAML keys to GBDTConfig parameters
    gbdt_params = {
        'n_estimators': model_config.get('gbdt_n_estimators', 1000),
        'max_depth': model_config.get('gbdt_max_depth', 6),
        'learning_rate': model_config.get('gbdt_learning_rate', 0.05),
        'subsample': model_config.get('gbdt_subsample', 0.8),
        'colsample_bytree': model_config.get('gbdt_colsample_bytree', 0.8),
        'reg_alpha': model_config.get('gbdt_reg_alpha', 0.1),
        'reg_lambda': model_config.get('gbdt_reg_lambda', 0.1),
        'min_child_samples': model_config.get('gbdt_min_child_samples', 20),
        'feature_fraction': model_config.get('gbdt_feature_fraction', 0.8),
        'num_instruments': model_config.get('num_instruments', 24),
        'latent_dim': model_config.get('tcnae_latent_dim', 120)
    }
    
    return GBDTConfig(**gbdt_params)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test MultiOutputGBDT
    config = GBDTConfig(
        n_estimators=100,  # Smaller for testing
        num_instruments=3,  # Test with 3 instruments
        latent_dim=10      # Smaller latent dimension
    )
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, config.latent_dim)
    y = np.random.randn(n_samples, config.num_instruments)
    
    # Add some NaN values to test robustness
    y[np.random.rand(*y.shape) < 0.1] = np.nan
    
    instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    
    print(f"MultiOutputGBDT Test:")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Instruments: {instruments}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    model = MultiOutputGBDT(config)
    model.fit(X_train, y_train, instruments, X_val, y_val)
    
    # Test prediction
    predictions = model.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    
    # Validate
    metrics = model.validate(X_val, y_val)
    print(f"Validation metrics: {metrics}")
    
    # Test save/load
    model.save('test_gbdt_model.pkl')
    loaded_model = MultiOutputGBDT.load('test_gbdt_model.pkl')
    
    print("✅ MultiOutputGBDT test completed successfully")