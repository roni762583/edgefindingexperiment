"""
Real-time inference pipeline for the Market Edge Finder Experiment.

This module provides the production inference system that combines TCNAE and LightGBM
for real-time FX return predictions. Includes context management, feature preprocessing,
and prediction post-processing with proper error handling and logging.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import threading
import time
from queue import Queue, Empty

# Import our models and utilities
from ..models.tcnae import TCNAE
from ..models.gbdt_model import MultiOutputGBDT
from ..models.context_manager import ContextTensorManager
from ..features.feature_engineering import FeatureEngineer
from ..features.normalization import FeatureNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for production
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class PredictionConfig:
    """Configuration for the inference pipeline."""
    
    # Model paths
    tcnae_model_path: str
    gbdt_model_path: str
    context_manager_path: str
    normalizer_path: str
    
    # Model parameters
    sequence_length: int = 4
    latent_dim: int = 100
    num_instruments: int = 20
    
    # Feature configuration
    feature_cols: List[str] = None
    
    # Prediction parameters
    confidence_threshold: float = 0.1  # Minimum prediction confidence
    max_prediction_age_minutes: int = 60  # Maximum age for cached predictions
    
    # Performance settings
    device: str = 'cpu'  # 'cuda', 'mps', or 'cpu'
    batch_size: int = 1
    num_threads: int = 1
    
    # Safety limits
    max_position_size: float = 1.0  # Maximum position size per instrument
    max_total_exposure: float = 10.0  # Maximum total portfolio exposure
    
    def __post_init__(self):
        if self.feature_cols is None:
            self.feature_cols = ['slope_high', 'slope_low', 'volatility', 'direction']


class ModelLoadingError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised during prediction generation."""
    pass


class EdgePredictor:
    """
    Production inference pipeline for real-time FX edge prediction.
    
    This class coordinates TCNAE feature extraction, LightGBM prediction,
    and context tensor management for real-time trading decisions.
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize the edge predictor.
        
        Args:
            config: Prediction configuration
        """
        self.config = config
        self.device = self._get_device()
        self.is_initialized = False
        
        # Model components
        self.tcnae_model: Optional[TCNAE] = None
        self.gbdt_model: Optional[MultiOutputGBDT] = None
        self.context_manager: Optional[ContextTensorManager] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.normalizer: Optional[FeatureNormalizer] = None
        
        # State management
        self.feature_history: Dict[str, pd.DataFrame] = {}
        self.prediction_cache: Dict[str, Dict] = {}
        self.last_update_time: Optional[datetime] = None
        self.lock = threading.Lock()
        
        # Instruments
        self.instruments = [
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
            'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
            'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD',
            'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD'
        ]
        
        logger.info(f"EdgePredictor initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device for inference."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.config.device)
    
    def initialize_models(self) -> None:
        """
        Load all models and components required for inference.
        
        Raises:
            ModelLoadingError: If any model fails to load
        """
        try:
            logger.info("Loading models for inference...")
            
            # 1. Load TCNAE model
            logger.info("Loading TCNAE model...")
            self.tcnae_model = TCNAE(
                input_channels=len(self.config.feature_cols),
                sequence_length=self.config.sequence_length,
                latent_dim=self.config.latent_dim,
                num_instruments=self.config.num_instruments
            )
            
            if Path(self.config.tcnae_model_path).exists():
                checkpoint = torch.load(self.config.tcnae_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.tcnae_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.tcnae_model.load_state_dict(checkpoint)
                logger.info("TCNAE model loaded successfully")
            else:
                raise ModelLoadingError(f"TCNAE model not found at {self.config.tcnae_model_path}")
            
            self.tcnae_model.to(self.device)
            self.tcnae_model.eval()
            
            # 2. Load LightGBM model
            logger.info("Loading LightGBM model...")
            if Path(self.config.gbdt_model_path).exists():
                with open(self.config.gbdt_model_path, 'rb') as f:
                    self.gbdt_model = pickle.load(f)
                logger.info("LightGBM model loaded successfully")
            else:
                raise ModelLoadingError(f"LightGBM model not found at {self.config.gbdt_model_path}")
            
            # 3. Load context manager
            logger.info("Loading context manager...")
            if Path(self.config.context_manager_path).exists():
                with open(self.config.context_manager_path, 'rb') as f:
                    context_state = pickle.load(f)
                    
                self.context_manager = ContextTensorManager(
                    num_instruments=self.config.num_instruments,
                    context_dim=self.config.latent_dim,
                    device=self.device
                )
                self.context_manager.load_state(context_state)
                logger.info("Context manager loaded successfully")
            else:
                logger.warning(f"Context manager not found at {self.config.context_manager_path}, initializing new one")
                self.context_manager = ContextTensorManager(
                    num_instruments=self.config.num_instruments,
                    context_dim=self.config.latent_dim,
                    device=self.device
                )
            
            # 4. Load normalizer
            logger.info("Loading feature normalizer...")
            if Path(self.config.normalizer_path).exists():
                with open(self.config.normalizer_path, 'rb') as f:
                    self.normalizer = pickle.load(f)
                logger.info("Feature normalizer loaded successfully")
            else:
                logger.warning(f"Normalizer not found at {self.config.normalizer_path}, initializing new one")
                self.normalizer = FeatureNormalizer()
            
            # 5. Initialize feature engineer
            self.feature_engineer = FeatureEngineer()
            
            self.is_initialized = True
            logger.info("All models loaded successfully - ready for inference")
            
        except Exception as e:
            error_msg = f"Failed to initialize models: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg)
    
    def update_features(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Update feature history with new market data.
        
        Args:
            market_data: Dictionary with instrument names and OHLC data
        """
        with self.lock:
            try:
                for instrument, data in market_data.items():
                    if instrument not in self.instruments:
                        continue
                    
                    # Generate features for this instrument
                    features = self.feature_engineer.generate_features_single_instrument(
                        data, instrument
                    )
                    
                    # Update feature history
                    if instrument not in self.feature_history:
                        self.feature_history[instrument] = features
                    else:
                        # Append new features and keep only recent history
                        combined = pd.concat([self.feature_history[instrument], features])
                        # Keep last 1000 rows for efficiency
                        self.feature_history[instrument] = combined.tail(1000).reset_index(drop=True)
                
                self.last_update_time = datetime.now()
                logger.debug(f"Features updated for {len(market_data)} instruments")
                
            except Exception as e:
                logger.error(f"Error updating features: {str(e)}")
    
    def _prepare_features_for_inference(self) -> Optional[torch.Tensor]:
        """
        Prepare features for model inference.
        
        Returns:
            Feature tensor ready for TCNAE inference, or None if insufficient data
        """
        try:
            # Check if we have enough feature history
            if not self.feature_history:
                logger.warning("No feature history available")
                return None
            
            # Collect features for all instruments
            instrument_features = []
            
            for instrument in self.instruments:
                if instrument not in self.feature_history:
                    logger.warning(f"No features available for {instrument}")
                    return None
                
                features_df = self.feature_history[instrument]
                
                if len(features_df) < self.config.sequence_length:
                    logger.warning(f"Insufficient feature history for {instrument}: {len(features_df)} < {self.config.sequence_length}")
                    return None
                
                # Get last sequence_length rows
                recent_features = features_df[self.config.feature_cols].tail(self.config.sequence_length)
                
                # Normalize features
                normalized_features = self.normalizer.normalize_single_instrument(
                    recent_features, instrument
                )
                
                instrument_features.append(normalized_features.values)
            
            # Stack features: (sequence_length, features, instruments)
            feature_array = np.stack(instrument_features, axis=-1)
            
            # Convert to tensor and reshape for TCNAE: (batch, features, sequence_length, instruments)
            feature_tensor = torch.FloatTensor(feature_array).permute(1, 0, 2).unsqueeze(0)
            
            return feature_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preparing features for inference: {str(e)}")
            return None
    
    def generate_predictions(self, use_context: bool = True) -> Optional[Dict[str, Any]]:
        """
        Generate predictions for all instruments.
        
        Args:
            use_context: Whether to use context tensor for prediction
            
        Returns:
            Dictionary with predictions and metadata, or None if prediction fails
        """
        if not self.is_initialized:
            raise PredictionError("Models not initialized. Call initialize_models() first.")
        
        try:
            with self.lock:
                # Prepare features
                feature_tensor = self._prepare_features_for_inference()
                if feature_tensor is None:
                    return None
                
                # Generate TCNAE features
                with torch.no_grad():
                    _, latent_features = self.tcnae_model(feature_tensor)
                
                # Convert to numpy for LightGBM
                latent_np = latent_features.cpu().numpy().squeeze()  # Remove batch dimension
                
                # Add context if requested
                if use_context and self.context_manager is not None:
                    context_tensor = self.context_manager.get_current_context()
                    context_np = context_tensor.cpu().numpy()
                    
                    # Concatenate latent features with context
                    combined_features = np.concatenate([latent_np, context_np], axis=-1)
                else:
                    combined_features = latent_np
                
                # Reshape for LightGBM: (instruments, features)
                if combined_features.ndim == 2:
                    combined_features = combined_features.T  # Transpose to (instruments, features)
                
                # Generate LightGBM predictions
                predictions = self.gbdt_model.predict(combined_features)
                
                # Apply confidence filtering
                confidence_scores = np.abs(predictions)
                mask = confidence_scores >= self.config.confidence_threshold
                filtered_predictions = np.where(mask, predictions, 0.0)
                
                # Apply position size limits
                filtered_predictions = np.clip(
                    filtered_predictions,
                    -self.config.max_position_size,
                    self.config.max_position_size
                )
                
                # Check total exposure
                total_exposure = np.sum(np.abs(filtered_predictions))
                if total_exposure > self.config.max_total_exposure:
                    scaling_factor = self.config.max_total_exposure / total_exposure
                    filtered_predictions *= scaling_factor
                    logger.info(f"Scaled predictions by {scaling_factor:.3f} to limit total exposure")
                
                # Update context with new predictions
                if self.context_manager is not None:
                    pred_tensor = torch.FloatTensor(filtered_predictions).unsqueeze(0).to(self.device)
                    self.context_manager.update_context(pred_tensor)
                
                # Create prediction dictionary
                prediction_dict = {
                    'predictions': {
                        instrument: float(pred) 
                        for instrument, pred in zip(self.instruments, filtered_predictions)
                    },
                    'confidence_scores': {
                        instrument: float(score) 
                        for instrument, score in zip(self.instruments, confidence_scores)
                    },
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_exposure': float(total_exposure),
                        'high_confidence_count': int(np.sum(mask)),
                        'used_context': use_context,
                        'model_version': 'hybrid_tcnae_lgb',
                        'feature_sequence_length': self.config.sequence_length
                    }
                }
                
                # Cache predictions
                self.prediction_cache = prediction_dict
                
                logger.info(f"Generated predictions for {len(self.instruments)} instruments")
                return prediction_dict
                
        except Exception as e:
            error_msg = f"Error generating predictions: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def get_latest_predictions(self, max_age_minutes: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest cached predictions if they're still fresh.
        
        Args:
            max_age_minutes: Maximum age in minutes for cached predictions
            
        Returns:
            Latest predictions if fresh enough, otherwise None
        """
        if max_age_minutes is None:
            max_age_minutes = self.config.max_prediction_age_minutes
        
        if not self.prediction_cache:
            return None
        
        try:
            pred_time = datetime.fromisoformat(self.prediction_cache['metadata']['timestamp'])
            age_minutes = (datetime.now() - pred_time).total_seconds() / 60
            
            if age_minutes <= max_age_minutes:
                return self.prediction_cache
            else:
                logger.debug(f"Cached predictions too old: {age_minutes:.1f} minutes")
                return None
                
        except Exception as e:
            logger.error(f"Error checking prediction cache: {str(e)}")
            return None
    
    def predict_for_instruments(self, 
                               instrument_list: List[str],
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Generate predictions for a specific list of instruments.
        
        Args:
            instrument_list: List of instruments to predict
            market_data: Current market data
            
        Returns:
            Dictionary with instrument predictions
        """
        # Update features with new market data
        self.update_features(market_data)
        
        # Generate full predictions
        full_predictions = self.generate_predictions()
        
        if full_predictions is None:
            return {instrument: 0.0 for instrument in instrument_list}
        
        # Filter for requested instruments
        filtered_predictions = {}
        for instrument in instrument_list:
            if instrument in full_predictions['predictions']:
                filtered_predictions[instrument] = full_predictions['predictions'][instrument]
            else:
                filtered_predictions[instrument] = 0.0
        
        return filtered_predictions
    
    def get_model_health(self) -> Dict[str, Any]:
        """
        Get health status of all model components.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            'initialized': self.is_initialized,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'components': {
                'tcnae_model': self.tcnae_model is not None,
                'gbdt_model': self.gbdt_model is not None,
                'context_manager': self.context_manager is not None,
                'feature_engineer': self.feature_engineer is not None,
                'normalizer': self.normalizer is not None
            },
            'feature_history_status': {
                instrument: len(self.feature_history.get(instrument, []))
                for instrument in self.instruments
            },
            'cache_status': {
                'has_cached_predictions': bool(self.prediction_cache),
                'cache_age_minutes': None
            }
        }
        
        # Calculate cache age
        if self.prediction_cache and 'metadata' in self.prediction_cache:
            try:
                pred_time = datetime.fromisoformat(self.prediction_cache['metadata']['timestamp'])
                age_minutes = (datetime.now() - pred_time).total_seconds() / 60
                health_status['cache_status']['cache_age_minutes'] = age_minutes
            except:
                pass
        
        return health_status
    
    def save_state(self, state_path: str) -> None:
        """
        Save current predictor state for recovery.
        
        Args:
            state_path: Path to save state file
        """
        try:
            state = {
                'config': self.config.__dict__,
                'feature_history': self.feature_history,
                'prediction_cache': self.prediction_cache,
                'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
                'context_state': self.context_manager.get_state() if self.context_manager else None
            }
            
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Predictor state saved to {state_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictor state: {str(e)}")
    
    def load_state(self, state_path: str) -> None:
        """
        Load predictor state from file.
        
        Args:
            state_path: Path to state file
        """
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.feature_history = state.get('feature_history', {})
            self.prediction_cache = state.get('prediction_cache', {})
            
            if state.get('last_update_time'):
                self.last_update_time = datetime.fromisoformat(state['last_update_time'])
            
            # Restore context manager state
            if self.context_manager and state.get('context_state'):
                self.context_manager.load_state(state['context_state'])
            
            logger.info(f"Predictor state loaded from {state_path}")
            
        except Exception as e:
            logger.error(f"Error loading predictor state: {str(e)}")


class RealtimePredictor:
    """
    Wrapper for real-time prediction with background processing.
    
    This class provides a higher-level interface for real-time predictions
    with automatic feature updates and background processing.
    """
    
    def __init__(self, config: PredictionConfig, update_interval_seconds: int = 60):
        """
        Initialize the real-time predictor.
        
        Args:
            config: Prediction configuration
            update_interval_seconds: How often to generate new predictions
        """
        self.predictor = EdgePredictor(config)
        self.update_interval = update_interval_seconds
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.data_queue: Queue = Queue()
        
    def start(self) -> None:
        """Start the real-time prediction service."""
        if not self.predictor.is_initialized:
            self.predictor.initialize_models()
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._prediction_worker)
        self.worker_thread.start()
        logger.info("Real-time predictor started")
    
    def stop(self) -> None:
        """Stop the real-time prediction service."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Real-time predictor stopped")
    
    def add_market_data(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Add new market data for processing.
        
        Args:
            market_data: Dictionary with instrument market data
        """
        try:
            self.data_queue.put(market_data, timeout=1.0)
        except:
            logger.warning("Market data queue full, dropping oldest data")
            try:
                self.data_queue.get_nowait()  # Remove oldest
                self.data_queue.put(market_data, timeout=1.0)
            except Empty:
                pass
    
    def get_current_predictions(self) -> Optional[Dict[str, Any]]:
        """Get the most recent predictions."""
        return self.predictor.get_latest_predictions()
    
    def _prediction_worker(self) -> None:
        """Background worker for generating predictions."""
        while self.running:
            try:
                # Process any new market data
                while not self.data_queue.empty():
                    try:
                        market_data = self.data_queue.get_nowait()
                        self.predictor.update_features(market_data)
                    except Empty:
                        break
                
                # Generate new predictions
                predictions = self.predictor.generate_predictions()
                if predictions:
                    logger.debug("Generated new predictions in background")
                
                # Wait before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in prediction worker: {str(e)}")
                time.sleep(self.update_interval)


if __name__ == "__main__":
    # Example usage
    logger.info("Market Edge Finder Inference Module")
    logger.info("This module provides real-time prediction capabilities for the hybrid TCNAE+LightGBM system.")