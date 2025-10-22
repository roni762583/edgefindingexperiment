#!/usr/bin/env python3
"""
Context Tensor Management for Market Edge Finder Experiment
Handles cross-instrument awareness and adaptive teacher forcing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from collections import deque
import warnings
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.instruments import FX_INSTRUMENTS

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context tensor system"""
    num_instruments: int = 20
    context_dim: int = 4  # Features per instrument in context
    sequence_length: int = 4  # Hours of context history
    
    # Adaptive teacher forcing parameters
    initial_teacher_forcing: float = 1.0  # Start with 100% true context
    min_teacher_forcing: float = 0.1  # Minimum teacher forcing ratio
    correlation_threshold: float = 0.3  # Minimum correlation for adaptive switching
    
    # EMA smoothing parameters
    correlation_ema_alpha: float = 0.9  # EMA smoothing for correlation tracking
    context_ema_alpha: float = 0.1  # EMA smoothing for context updates
    
    # Context attention parameters
    enable_attention: bool = True
    attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Cross-instrument correlation parameters
    correlation_window: int = 100  # Rolling window for correlation computation
    correlation_update_freq: int = 10  # Batches between correlation updates


class ContextTensor:
    """
    Context tensor for maintaining cross-instrument state
    Tracks predictions and correlations across all FX instruments
    """
    
    def __init__(self, config: ContextConfig, device: str = 'cpu'):
        """
        Initialize context tensor
        
        Args:
            config: Context configuration
            device: Device for tensor operations
        """
        self.config = config
        self.device = device
        
        # Current context state [num_instruments, context_dim]
        self.current_context = torch.zeros(
            config.num_instruments, config.context_dim, device=device
        )
        
        # Context history [sequence_length, num_instruments, context_dim]
        self.context_history = torch.zeros(
            config.sequence_length, config.num_instruments, config.context_dim, device=device
        )
        
        # Correlation tracking
        self.correlation_matrix = torch.eye(config.num_instruments, device=device)
        self.prediction_history = deque(maxlen=config.correlation_window)
        self.target_history = deque(maxlen=config.correlation_window)
        
        # Teacher forcing state
        self.teacher_forcing_ratio = config.initial_teacher_forcing
        self.correlation_ema = torch.zeros(config.num_instruments, device=device)
        
        logger.info(f"ContextTensor initialized: {config.num_instruments} instruments, "
                   f"context_dim={config.context_dim}, device={device}")
    
    def update_context(self, new_predictions: torch.Tensor, 
                      true_targets: Optional[torch.Tensor] = None,
                      use_teacher_forcing: bool = True) -> torch.Tensor:
        """
        Update context tensor with new predictions/targets
        
        Args:
            new_predictions: New predictions [batch_size, num_instruments]
            true_targets: Optional true targets for teacher forcing
            use_teacher_forcing: Whether to apply teacher forcing
            
        Returns:
            Updated context tensor [num_instruments, context_dim]
        """
        batch_size = new_predictions.size(0)
        
        # Determine what to use for context update
        if use_teacher_forcing and true_targets is not None:
            # Apply adaptive teacher forcing
            context_source = self._apply_teacher_forcing(new_predictions, true_targets)
        else:
            # Use predictions only
            context_source = new_predictions
        
        # Update context with batch average
        if batch_size > 1:
            context_update = context_source.mean(dim=0)  # [num_instruments]
        else:
            context_update = context_source.squeeze(0)  # [num_instruments]
        
        # Expand to context_dim if needed
        if self.config.context_dim > 1:
            # Create multi-dimensional context from single prediction
            # Include: current prediction, momentum, volatility, confidence
            momentum = context_update - self.current_context[:, 0]  # Change from previous
            volatility = torch.abs(momentum)  # Volatility proxy
            confidence = torch.sigmoid(torch.abs(context_update))  # Confidence proxy
            
            new_context = torch.stack([
                context_update,  # Current prediction
                momentum,        # Momentum/change
                volatility,      # Volatility
                confidence       # Confidence
            ], dim=1)  # [num_instruments, context_dim]
        else:
            new_context = context_update.unsqueeze(1)  # [num_instruments, 1]
        
        # Update context with EMA smoothing
        alpha = self.config.context_ema_alpha
        self.current_context = (1 - alpha) * self.current_context + alpha * new_context
        
        # Update context history (shift and append)
        self.context_history = torch.cat([
            self.context_history[1:],  # Remove oldest
            self.current_context.unsqueeze(0)  # Add current
        ], dim=0)
        
        return self.current_context.clone()
    
    def _apply_teacher_forcing(self, predictions: torch.Tensor, 
                              targets: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive teacher forcing based on correlation
        
        Args:
            predictions: Model predictions [batch_size, num_instruments]
            targets: True targets [batch_size, num_instruments]
            
        Returns:
            Mixed predictions/targets based on teacher forcing ratio
        """
        # Calculate per-instrument teacher forcing ratios
        tf_ratios = torch.full((self.config.num_instruments,), 
                              self.teacher_forcing_ratio, device=self.device)
        
        # Apply adaptive ratios based on correlation
        for i in range(self.config.num_instruments):
            if self.correlation_ema[i] > self.config.correlation_threshold:
                # Good correlation - reduce teacher forcing
                adaptive_ratio = max(
                    self.config.min_teacher_forcing,
                    self.teacher_forcing_ratio * (1 - self.correlation_ema[i])
                )
            else:
                # Poor correlation - increase teacher forcing
                adaptive_ratio = min(
                    1.0,
                    self.teacher_forcing_ratio * (2 - self.correlation_ema[i])
                )
            tf_ratios[i] = adaptive_ratio
        
        # Apply teacher forcing per instrument
        batch_size = predictions.size(0)
        mixed_context = torch.zeros_like(predictions)
        
        for i in range(self.config.num_instruments):
            ratio = tf_ratios[i]
            # Use Bernoulli sampling for stochastic teacher forcing
            use_target_mask = torch.rand(batch_size, device=self.device) < ratio
            
            mixed_context[:, i] = torch.where(
                use_target_mask,
                targets[:, i],
                predictions[:, i]
            )
        
        return mixed_context
    
    def update_correlations(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update correlation tracking between predictions and targets
        
        Args:
            predictions: Model predictions [batch_size, num_instruments]
            targets: True targets [batch_size, num_instruments]
        """
        # Store predictions and targets for correlation computation
        self.prediction_history.append(predictions.detach().cpu().numpy())
        self.target_history.append(targets.detach().cpu().numpy())
        
        # Compute correlations if we have enough data
        if len(self.prediction_history) >= 10:  # Minimum samples for correlation
            # Concatenate recent history
            recent_preds = np.concatenate(list(self.prediction_history), axis=0)
            recent_targets = np.concatenate(list(self.target_history), axis=0)
            
            # Compute per-instrument correlations
            correlations = []
            for i in range(self.config.num_instruments):
                pred_series = recent_preds[:, i]
                target_series = recent_targets[:, i]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pred_series) | np.isnan(target_series))
                if np.sum(valid_mask) > 5:
                    corr = np.corrcoef(pred_series[valid_mask], target_series[valid_mask])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                
                correlations.append(abs(corr))  # Use absolute correlation
            
            # Update EMA correlations
            new_correlations = torch.tensor(correlations, device=self.device)
            alpha = self.config.correlation_ema_alpha
            self.correlation_ema = alpha * self.correlation_ema + (1 - alpha) * new_correlations
            
            # Update teacher forcing ratio based on average correlation
            avg_correlation = self.correlation_ema.mean().item()
            if avg_correlation > self.config.correlation_threshold:
                # Decrease teacher forcing as model improves
                decay_factor = min(0.99, 1 - 0.01 * avg_correlation)
                self.teacher_forcing_ratio *= decay_factor
                self.teacher_forcing_ratio = max(
                    self.config.min_teacher_forcing, 
                    self.teacher_forcing_ratio
                )
    
    def get_context_for_instrument(self, instrument_idx: int) -> torch.Tensor:
        """
        Get context vector for specific instrument
        
        Args:
            instrument_idx: Index of instrument
            
        Returns:
            Context vector including cross-instrument information
        """
        # Current instrument context
        local_context = self.current_context[instrument_idx]  # [context_dim]
        
        # Cross-instrument context (summary of other instruments)
        other_instruments = torch.cat([
            self.current_context[:instrument_idx],
            self.current_context[instrument_idx + 1:]
        ], dim=0)  # [num_instruments-1, context_dim]
        
        # Summarize other instruments (mean pooling)
        global_context = other_instruments.mean(dim=0)  # [context_dim]
        
        # Combine local and global context
        combined_context = torch.cat([local_context, global_context], dim=0)  # [2 * context_dim]
        
        return combined_context
    
    def get_full_context_matrix(self) -> torch.Tensor:
        """
        Get full context matrix for all instruments
        
        Returns:
            Context matrix [num_instruments, context_features]
        """
        context_matrix = []
        
        for i in range(self.config.num_instruments):
            instrument_context = self.get_context_for_instrument(i)
            context_matrix.append(instrument_context)
        
        return torch.stack(context_matrix, dim=0)  # [num_instruments, context_features]
    
    def reset_context(self):
        """Reset context tensor to initial state"""
        self.current_context.zero_()
        self.context_history.zero_()
        self.correlation_matrix = torch.eye(self.config.num_instruments, device=self.device)
        self.prediction_history.clear()
        self.target_history.clear()
        self.teacher_forcing_ratio = self.config.initial_teacher_forcing
        self.correlation_ema.zero_()
        
        logger.info("Context tensor reset to initial state")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get context tensor statistics for monitoring
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'average_correlation': self.correlation_ema.mean().item(),
            'context_norm': torch.norm(self.current_context).item(),
            'context_std': torch.std(self.current_context).item(),
            'per_instrument_correlations': self.correlation_ema.cpu().numpy().tolist(),
            'history_length': len(self.prediction_history)
        }
        
        return stats


class ContextAttentionModule(nn.Module):
    """
    Attention module for context tensor processing
    Enables instruments to selectively attend to relevant cross-instrument information
    """
    
    def __init__(self, config: ContextConfig):
        """
        Initialize context attention module
        
        Args:
            config: Context configuration
        """
        super(ContextAttentionModule, self).__init__()
        
        self.config = config
        self.context_dim = config.context_dim
        self.num_heads = config.attention_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.context_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.context_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.context_dim, config.context_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.context_dim * 2, config.context_dim)
        )
        
        logger.info(f"ContextAttentionModule initialized: {config.attention_heads} heads, "
                   f"context_dim={config.context_dim}")
    
    def forward(self, context_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to context matrix
        
        Args:
            context_matrix: Context matrix [batch_size, num_instruments, context_dim]
            
        Returns:
            Attended context matrix [batch_size, num_instruments, context_dim]
        """
        # Apply multi-head attention
        attended_context, attention_weights = self.attention(
            query=context_matrix,
            key=context_matrix,
            value=context_matrix
        )
        
        # Residual connection and layer norm
        context_matrix = self.layer_norm(context_matrix + attended_context)
        
        # Feed-forward network
        ff_output = self.feed_forward(context_matrix)
        
        # Another residual connection
        context_matrix = self.layer_norm(context_matrix + ff_output)
        
        return context_matrix


class ContextManager:
    """
    High-level context manager for the entire system
    Coordinates context tensors, attention, and adaptive teacher forcing
    """
    
    def __init__(self, config: Optional[ContextConfig] = None, device: str = 'cpu'):
        """
        Initialize context manager
        
        Args:
            config: Context configuration
            device: Device for tensor operations
        """
        self.config = config or ContextConfig()
        self.device = device
        
        # Initialize context tensor
        self.context_tensor = ContextTensor(self.config, device)
        
        # Initialize attention module if enabled
        self.attention_module = None
        if self.config.enable_attention:
            self.attention_module = ContextAttentionModule(self.config).to(device)
        
        # Training state
        self.training_step = 0
        self.correlation_update_counter = 0
        
        logger.info(f"ContextManager initialized with attention: {self.config.enable_attention}")
    
    def prepare_features_with_context(self, base_features: torch.Tensor, 
                                    batch_size: int) -> torch.Tensor:
        """
        Prepare features augmented with context information
        
        Args:
            base_features: Base features [batch_size, sequence_length, feature_dim]
            batch_size: Batch size
            
        Returns:
            Augmented features with context [batch_size, sequence_length, feature_dim + context_dim]
        """
        # Get current context matrix
        context_matrix = self.context_tensor.get_full_context_matrix()  # [num_instruments, context_features]
        
        # Apply attention if enabled
        if self.attention_module is not None:
            # Expand for batch dimension
            context_batch = context_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            context_matrix = self.attention_module(context_batch).mean(dim=0)  # Average over batch
        
        # Flatten context for concatenation
        context_flattened = context_matrix.flatten()  # [num_instruments * context_features]
        
        # Expand context to match batch size and sequence length
        seq_len = base_features.size(1)
        context_expanded = context_flattened.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )  # [batch_size, sequence_length, context_features]
        
        # Concatenate base features with context
        augmented_features = torch.cat([base_features, context_expanded], dim=-1)
        
        return augmented_features
    
    def update_training_step(self, predictions: torch.Tensor, 
                           targets: torch.Tensor,
                           use_teacher_forcing: bool = True) -> Dict[str, Any]:
        """
        Update context manager for one training step
        
        Args:
            predictions: Model predictions [batch_size, num_instruments]
            targets: True targets [batch_size, num_instruments]
            use_teacher_forcing: Whether to use teacher forcing
            
        Returns:
            Update statistics
        """
        # Update context tensor
        updated_context = self.context_tensor.update_context(
            predictions, targets, use_teacher_forcing
        )
        
        # Update correlations periodically
        self.correlation_update_counter += 1
        if self.correlation_update_counter >= self.config.correlation_update_freq:
            self.context_tensor.update_correlations(predictions, targets)
            self.correlation_update_counter = 0
        
        self.training_step += 1
        
        # Get statistics
        stats = self.context_tensor.get_statistics()
        stats['training_step'] = self.training_step
        stats['context_updated'] = True
        
        return stats
    
    def get_adaptive_teacher_forcing_schedule(self, total_steps: int) -> List[float]:
        """
        Generate adaptive teacher forcing schedule
        
        Args:
            total_steps: Total training steps
            
        Returns:
            List of teacher forcing ratios for each step
        """
        schedule = []
        current_ratio = self.config.initial_teacher_forcing
        
        for step in range(total_steps):
            # Exponential decay with floor
            decay_rate = 0.999  # Slow decay
            current_ratio *= decay_rate
            current_ratio = max(self.config.min_teacher_forcing, current_ratio)
            
            schedule.append(current_ratio)
        
        return schedule
    
    def save_state(self, filepath: Union[str, Path]) -> bool:
        """
        Save context manager state
        
        Args:
            filepath: Path to save state
            
        Returns:
            True if successful
        """
        try:
            state = {
                'config': self.config,
                'current_context': self.context_tensor.current_context.cpu(),
                'context_history': self.context_tensor.context_history.cpu(),
                'correlation_matrix': self.context_tensor.correlation_matrix.cpu(),
                'correlation_ema': self.context_tensor.correlation_ema.cpu(),
                'teacher_forcing_ratio': self.context_tensor.teacher_forcing_ratio,
                'training_step': self.training_step
            }
            
            # Save attention module if present
            if self.attention_module is not None:
                state['attention_state_dict'] = self.attention_module.state_dict()
            
            torch.save(state, filepath)
            logger.info(f"Context manager state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save context manager state: {e}")
            return False
    
    def load_state(self, filepath: Union[str, Path]) -> bool:
        """
        Load context manager state
        
        Args:
            filepath: Path to load state from
            
        Returns:
            True if successful
        """
        try:
            state = torch.load(filepath, map_location=self.device)
            
            # Restore context tensor state
            self.context_tensor.current_context = state['current_context'].to(self.device)
            self.context_tensor.context_history = state['context_history'].to(self.device)
            self.context_tensor.correlation_matrix = state['correlation_matrix'].to(self.device)
            self.context_tensor.correlation_ema = state['correlation_ema'].to(self.device)
            self.context_tensor.teacher_forcing_ratio = state['teacher_forcing_ratio']
            self.training_step = state['training_step']
            
            # Restore attention module if present
            if 'attention_state_dict' in state and self.attention_module is not None:
                self.attention_module.load_state_dict(state['attention_state_dict'])
            
            logger.info(f"Context manager state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load context manager state: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test context tensor system
    config = ContextConfig(
        num_instruments=20,
        context_dim=4,
        sequence_length=4,
        enable_attention=True,
        attention_heads=4
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    context_manager = ContextManager(config, device)
    
    print(f"Testing Context Manager on device: {device}")
    
    # Generate test data
    batch_size = 32
    num_instruments = 20
    feature_dim = 80  # Base features before context
    
    # Simulate base features and predictions/targets
    base_features = torch.randn(batch_size, 4, feature_dim, device=device)
    predictions = torch.randn(batch_size, num_instruments, device=device) * 0.001
    targets = predictions + torch.randn_like(predictions) * 0.0005  # Correlated targets
    
    print(f"Base features shape: {base_features.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Test context augmentation
    augmented_features = context_manager.prepare_features_with_context(base_features, batch_size)
    print(f"Augmented features shape: {augmented_features.shape}")
    
    # Test training step updates
    print("\nTesting training step updates:")
    for step in range(5):
        stats = context_manager.update_training_step(predictions, targets)
        
        print(f"Step {step}: TF ratio={stats['teacher_forcing_ratio']:.3f}, "
              f"Avg correlation={stats['average_correlation']:.3f}, "
              f"Context norm={stats['context_norm']:.3f}")
    
    # Test adaptive teacher forcing schedule
    schedule = context_manager.get_adaptive_teacher_forcing_schedule(100)
    print(f"\nTeacher forcing schedule (first 10 steps): {schedule[:10]}")
    print(f"Teacher forcing schedule (last 10 steps): {schedule[-10:]}")
    
    # Test context tensor operations
    print(f"\nContext tensor statistics:")
    final_stats = context_manager.context_tensor.get_statistics()
    for key, value in final_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        elif isinstance(value, list) and len(value) <= 5:
            print(f"  {key}: {value}")
    
    # Test save/load
    save_path = 'test_context_state.pth'
    save_success = context_manager.save_state(save_path)
    print(f"\nSave successful: {save_success}")
    
    if save_success:
        # Create new manager and load state
        new_manager = ContextManager(config, device)
        load_success = new_manager.load_state(save_path)
        print(f"Load successful: {load_success}")
        
        if load_success:
            # Verify state preservation
            original_context = context_manager.context_tensor.current_context
            loaded_context = new_manager.context_tensor.current_context
            
            diff = torch.abs(original_context - loaded_context).max().item()
            print(f"Max context difference after load: {diff:.8f}")
        
        # Clean up
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
    
    print("✅ Context manager test completed successfully")