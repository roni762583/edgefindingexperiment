#!/usr/bin/env python3
"""
USD PIP & DIRECTION PREDICTION MODEL
====================================

Proper implementation of LightGBM for edge discovery with:
1. USD pip prediction (actual dollar movement per standard lot)
2. Direction classification (probability of upward movement)
3. Confidence measures (prediction uncertainty)

This replaces the incorrect price_change regression with proper trading targets.

Author: Claude Code Assistant
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import lightgbm as lgb
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PipDirectionConfig:
    """Configuration for pip and direction prediction model."""
    # Model hyperparameters
    num_leaves: int = 100
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    # Training parameters
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    verbose_eval: int = 100
    
    # Pip calculation
    pip_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pip_multipliers is None:
            # Standard pip values for major FX pairs (units per pip)
            self.pip_multipliers = {
                # Major pairs (most trade in 0.0001 increments)
                'EUR_USD': 0.0001, 'GBP_USD': 0.0001, 'AUD_USD': 0.0001, 'NZD_USD': 0.0001,
                'USD_CAD': 0.0001, 'USD_CHF': 0.0001,
                
                # JPY pairs (trade in 0.01 increments)  
                'USD_JPY': 0.01, 'EUR_JPY': 0.01, 'GBP_JPY': 0.01, 'AUD_JPY': 0.01,
                'CAD_JPY': 0.01, 'CHF_JPY': 0.01, 'NZD_JPY': 0.01,
                
                # Cross pairs
                'EUR_GBP': 0.0001, 'EUR_AUD': 0.0001, 'EUR_CAD': 0.0001, 'EUR_CHF': 0.0001, 'EUR_NZD': 0.0001,
                'GBP_AUD': 0.0001, 'GBP_CAD': 0.0001, 'GBP_CHF': 0.0001, 'GBP_NZD': 0.0001,
                'AUD_CAD': 0.0001, 'AUD_CHF': 0.0001, 'AUD_NZD': 0.0001,
                'CAD_CHF': 0.0001, 'NZD_CAD': 0.0001, 'NZD_CHF': 0.0001
            }

class PipDirectionPredictor:
    """
    Multi-target LightGBM model that predicts:
    1. USD pips per standard lot (100,000 units)
    2. Direction probability (0-1 for up movement)
    """
    
    def __init__(self, config: PipDirectionConfig):
        self.config = config
        self.instruments = []
        self.pip_models = {}  # One model per instrument for pip prediction
        self.direction_models = {}  # One model per instrument for direction
        self.feature_names = []
        self.training_stats = {}
        
    def calculate_usd_pips(self, price_changes: np.ndarray, instrument: str, 
                          current_rates: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Convert raw price changes to USD pips per standard lot (100,000 units).
        
        Args:
            price_changes: Raw price changes (e.g., 0.0001 for EUR_USD)
            instrument: FX pair (e.g., 'EUR_USD')
            current_rates: Current exchange rates for USD conversion
            
        Returns:
            USD pip values per standard lot
        """
        pip_size = self.config.pip_multipliers.get(instrument, 0.0001)
        pips = price_changes / pip_size
        
        # Convert to USD value per standard lot (100,000 units)
        if instrument.endswith('_USD'):
            # Direct USD pairs: 1 pip = $10 per standard lot
            usd_pips = pips * 10.0
        elif instrument.startswith('USD_'):
            # USD base pairs: 1 pip = $10 / current_rate
            base_currency = instrument.split('_')[1]
            if current_rates and f'USD_{base_currency}' in current_rates:
                rate = current_rates[f'USD_{base_currency}']
                usd_pips = pips * (10.0 / rate)
            else:
                # Fallback approximation
                usd_pips = pips * 8.0  # Rough average
        else:
            # Cross pairs: Convert via USD rates
            base_currency = instrument.split('_')[0]
            if current_rates and f'{base_currency}_USD' in current_rates:
                base_to_usd = current_rates[f'{base_currency}_USD']
                usd_pips = pips * 10.0 * base_to_usd
            else:
                # Fallback approximation
                usd_pips = pips * 10.0  # Assume roughly USD equivalent
        
        return usd_pips
    
    def prepare_targets(self, features: np.ndarray, raw_returns: np.ndarray, 
                       instruments: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare USD pip and direction targets from raw returns.
        
        Args:
            features: Input features (N, feature_dim)
            raw_returns: Raw price returns (N, num_instruments) 
            instruments: List of instrument names
            
        Returns:
            Tuple of (usd_pip_targets, direction_targets)
        """
        n_samples, n_instruments = raw_returns.shape
        usd_pip_targets = np.zeros_like(raw_returns)
        direction_targets = np.zeros_like(raw_returns)
        
        for i, instrument in enumerate(instruments):
            # Convert to USD pips
            usd_pips = self.calculate_usd_pips(raw_returns[:, i], instrument)
            usd_pip_targets[:, i] = usd_pips
            
            # Direction: 1 for up, 0 for down
            direction_targets[:, i] = (raw_returns[:, i] > 0).astype(float)
            
        return usd_pip_targets, direction_targets
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray,
            raw_returns_train: np.ndarray, raw_returns_val: np.ndarray,
            instruments: List[str], feature_names: List[str]) -> Dict[str, Any]:
        """
        Train separate models for pip prediction and direction classification.
        
        Args:
            X_train: Training features (N, latent_dim)
            X_val: Validation features (N, latent_dim) 
            raw_returns_train: Training returns for USD pip calculation
            raw_returns_val: Validation returns for USD pip calculation
            instruments: List of FX instrument names
            feature_names: Names of input features
            
        Returns:
            Training statistics and metrics
        """
        self.instruments = instruments
        self.feature_names = feature_names
        
        logger.info(f"🎯 Training pip & direction models for {len(instruments)} instruments")
        
        # Prepare targets
        usd_pip_train, direction_train = self.prepare_targets(X_train, raw_returns_train, instruments)
        usd_pip_val, direction_val = self.prepare_targets(X_val, raw_returns_val, instruments)
        
        training_results = {}
        
        for i, instrument in enumerate(instruments):
            logger.info(f"Training models for {instrument}")
            
            # Prepare data for this instrument
            y_pip_train = usd_pip_train[:, i]
            y_dir_train = direction_train[:, i]
            y_pip_val = usd_pip_val[:, i]
            y_dir_val = direction_val[:, i]
            
            # Create LightGBM datasets
            train_data_pip = lgb.Dataset(X_train, label=y_pip_train, feature_name=feature_names)
            val_data_pip = lgb.Dataset(X_val, label=y_pip_val, reference=train_data_pip)
            
            train_data_dir = lgb.Dataset(X_train, label=y_dir_train, feature_name=feature_names)
            val_data_dir = lgb.Dataset(X_val, label=y_dir_val, reference=train_data_dir)
            
            # Pip prediction model (regression)
            pip_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'feature_fraction': self.config.feature_fraction,
                'bagging_fraction': self.config.bagging_fraction,
                'bagging_freq': self.config.bagging_freq,
                'min_child_samples': self.config.min_child_samples,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'verbose': -1
            }
            
            self.pip_models[instrument] = lgb.train(
                pip_params,
                train_data_pip,
                valid_sets=[val_data_pip],
                num_boost_round=self.config.num_boost_round,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds),
                    lgb.log_evaluation(self.config.verbose_eval)
                ]
            )
            
            # Direction prediction model (binary classification)
            dir_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': self.config.num_leaves,
                'learning_rate': self.config.learning_rate,
                'feature_fraction': self.config.feature_fraction,
                'bagging_fraction': self.config.bagging_fraction,
                'bagging_freq': self.config.bagging_freq,
                'min_child_samples': self.config.min_child_samples,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'verbose': -1
            }
            
            self.direction_models[instrument] = lgb.train(
                dir_params,
                train_data_dir,
                valid_sets=[val_data_dir],
                num_boost_round=self.config.num_boost_round,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds),
                    lgb.log_evaluation(self.config.verbose_eval)
                ]
            )
            
            # Calculate metrics
            pip_pred_val = self.pip_models[instrument].predict(X_val)
            dir_pred_val = self.direction_models[instrument].predict(X_val)
            
            pip_rmse = np.sqrt(np.mean((pip_pred_val - y_pip_val) ** 2))
            pip_mae = np.mean(np.abs(pip_pred_val - y_pip_val))
            
            direction_accuracy = np.mean((dir_pred_val > 0.5) == (y_dir_val > 0.5))
            direction_correlation = np.corrcoef(dir_pred_val, y_dir_val)[0, 1]
            
            training_results[instrument] = {
                'pip_rmse': pip_rmse,
                'pip_mae': pip_mae,
                'direction_accuracy': direction_accuracy,
                'direction_correlation': direction_correlation,
                'pip_pred_range': [pip_pred_val.min(), pip_pred_val.max()],
                'actual_pip_range': [y_pip_val.min(), y_pip_val.max()],
                'n_samples': len(y_pip_val)
            }
            
        self.training_stats = training_results
        return training_results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for USD pips, direction, and confidence.
        
        Args:
            X: Input features (N, latent_dim)
            
        Returns:
            Tuple of (usd_pip_predictions, direction_probabilities, confidence_scores)
        """
        n_samples = X.shape[0]
        n_instruments = len(self.instruments)
        
        pip_predictions = np.zeros((n_samples, n_instruments))
        direction_probabilities = np.zeros((n_samples, n_instruments))
        confidence_scores = np.zeros((n_samples, n_instruments))
        
        for i, instrument in enumerate(self.instruments):
            # Pip predictions
            pip_pred = self.pip_models[instrument].predict(X)
            pip_predictions[:, i] = pip_pred
            
            # Direction probabilities
            dir_prob = self.direction_models[instrument].predict(X)
            direction_probabilities[:, i] = dir_prob
            
            # Confidence (distance from 0.5 for direction, normalized pip magnitude)
            dir_confidence = np.abs(dir_prob - 0.5) * 2  # 0-1 scale
            pip_confidence = np.clip(np.abs(pip_pred) / 20.0, 0, 1)  # Normalize by ~20 pip typical range
            confidence_scores[:, i] = (dir_confidence + pip_confidence) / 2
            
        return pip_predictions, direction_probabilities, confidence_scores
    
    def save(self, filepath: Path):
        """Save the trained model to disk."""
        model_data = {
            'config': self.config,
            'instruments': self.instruments,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'pip_models': {inst: model.model_to_string() for inst, model in self.pip_models.items()},
            'direction_models': {inst: model.model_to_string() for inst, model in self.direction_models.items()}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Saved pip & direction models to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'PipDirectionPredictor':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_data['config'])
        predictor.instruments = model_data['instruments']
        predictor.feature_names = model_data['feature_names']
        predictor.training_stats = model_data['training_stats']
        
        # Reconstruct LightGBM models
        for instrument in predictor.instruments:
            predictor.pip_models[instrument] = lgb.Booster(model_str=model_data['pip_models'][instrument])
            predictor.direction_models[instrument] = lgb.Booster(model_str=model_data['direction_models'][instrument])
        
        logger.info(f"📂 Loaded pip & direction models from {filepath}")
        return predictor
    
    def evaluate_performance(self, X_test: np.ndarray, raw_returns_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model performance.
        
        Args:
            X_test: Test features
            raw_returns_test: Test returns for ground truth
            
        Returns:
            Detailed performance metrics
        """
        usd_pip_true, direction_true = self.prepare_targets(X_test, raw_returns_test, self.instruments)
        pip_pred, dir_prob, confidence = self.predict(X_test)
        
        results = {}
        
        for i, instrument in enumerate(self.instruments):
            # Pip prediction metrics
            pip_rmse = np.sqrt(np.mean((pip_pred[:, i] - usd_pip_true[:, i]) ** 2))
            pip_mae = np.mean(np.abs(pip_pred[:, i] - usd_pip_true[:, i]))
            pip_corr = np.corrcoef(pip_pred[:, i], usd_pip_true[:, i])[0, 1]
            
            # Direction prediction metrics
            dir_pred_binary = (dir_prob[:, i] > 0.5).astype(float)
            direction_accuracy = np.mean(dir_pred_binary == direction_true[:, i])
            direction_precision = np.mean(direction_true[:, i][dir_pred_binary == 1]) if np.sum(dir_pred_binary) > 0 else 0
            direction_recall = np.mean(dir_pred_binary[direction_true[:, i] == 1]) if np.sum(direction_true[:, i]) > 0 else 0
            
            # Trading metrics
            # Simulate trades: long when dir_prob > 0.6, short when dir_prob < 0.4
            long_signals = dir_prob[:, i] > 0.6
            short_signals = dir_prob[:, i] < 0.4
            
            long_pnl = np.sum(usd_pip_true[:, i][long_signals]) if np.sum(long_signals) > 0 else 0
            short_pnl = -np.sum(usd_pip_true[:, i][short_signals]) if np.sum(short_signals) > 0 else 0
            total_pnl = long_pnl + short_pnl
            
            results[instrument] = {
                'pip_rmse': pip_rmse,
                'pip_mae': pip_mae,
                'pip_correlation': pip_corr,
                'direction_accuracy': direction_accuracy,
                'direction_precision': direction_precision,
                'direction_recall': direction_recall,
                'simulated_pnl_usd': total_pnl,
                'n_long_trades': np.sum(long_signals),
                'n_short_trades': np.sum(short_signals),
                'avg_confidence': np.mean(confidence[:, i]),
                'n_samples': len(usd_pip_true)
            }
        
        return results