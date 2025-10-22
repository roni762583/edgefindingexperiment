#!/usr/bin/env python3
"""
Temporal Convolutional Autoencoder (TCNAE) for Market Edge Finder Experiment
Compresses 4-hour sequences of features into 100-dimensional latent representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class TCNAEConfig:
    """Configuration for TCNAE model"""
    input_dim: int = 100  # 20 instruments × 4 features + context
    sequence_length: int = 4  # 4-hour lookback
    latent_dim: int = 100  # Compressed representation size
    
    # Encoder parameters
    encoder_channels: List[int] = None  # [100, 128, 96, 64]
    encoder_kernel_sizes: List[int] = None  # [3, 3, 3, 3]
    encoder_dilations: List[int] = None  # [1, 2, 4, 8]
    
    # Decoder parameters (mirror encoder)
    decoder_channels: List[int] = None  # [64, 96, 128, 100]
    decoder_kernel_sizes: List[int] = None  # [3, 3, 3, 3]
    decoder_dilations: List[int] = None  # [8, 4, 2, 1]
    
    # Regularization
    dropout: float = 0.1
    batch_norm: bool = True
    
    # Training parameters
    reconstruction_weight: float = 1.0
    regularization_weight: float = 0.01
    
    def __post_init__(self):
        """Set default values for channel configurations"""
        if self.encoder_channels is None:
            self.encoder_channels = [self.input_dim, 128, 96, 64]
        if self.encoder_kernel_sizes is None:
            self.encoder_kernel_sizes = [3, 3, 3, 3]
        if self.encoder_dilations is None:
            self.encoder_dilations = [1, 2, 4, 8]
        
        if self.decoder_channels is None:
            self.decoder_channels = [64, 96, 128, self.input_dim]
        if self.decoder_kernel_sizes is None:
            self.decoder_kernel_sizes = [3, 3, 3, 3]
        if self.decoder_dilations is None:
            self.decoder_dilations = [8, 4, 2, 1]


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution - ensures no lookahead bias
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, bias: bool = True):
        """
        Initialize causal convolution
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            bias: Whether to use bias
        """
        super(CausalConv1d, self).__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             dilation=dilation, padding=self.padding, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding
        
        Args:
            x: Input tensor [batch_size, channels, sequence_length]
            
        Returns:
            Output tensor with same temporal dimension
        """
        out = self.conv(x)
        
        # Remove future information (right padding)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connections
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.1, batch_norm: bool = True):
        """
        Initialize TCN block
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super(TCNBlock, self).__init__()
        
        # First causal convolution
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second causal convolution
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN block
        
        Args:
            x: Input tensor [batch_size, in_channels, sequence_length]
            
        Returns:
            Output tensor [batch_size, out_channels, sequence_length]
        """
        # Store input for residual connection
        residual = self.residual(x)
        
        # First convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        # Second convolution path
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        # Add residual connection
        out = out + residual
        
        return self.activation(out)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder
    Compresses temporal sequences into latent representation
    """
    
    def __init__(self, config: TCNAEConfig):
        """
        Initialize TCN encoder
        
        Args:
            config: TCNAE configuration
        """
        super(TCNEncoder, self).__init__()
        
        self.config = config
        self.blocks = nn.ModuleList()
        
        # Build encoder blocks
        for i in range(len(config.encoder_channels) - 1):
            in_channels = config.encoder_channels[i]
            out_channels = config.encoder_channels[i + 1]
            kernel_size = config.encoder_kernel_sizes[i]
            dilation = config.encoder_dilations[i]
            
            block = TCNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=config.dropout,
                batch_norm=config.batch_norm
            )
            self.blocks.append(block)
        
        # Final compression to latent dimension
        final_channels = config.encoder_channels[-1]
        self.latent_conv = nn.Conv1d(final_channels, config.latent_dim, 1)
        self.latent_pool = nn.AdaptiveAvgPool1d(1)  # Global temporal pooling
        
        logger.info(f"TCNEncoder initialized with {len(self.blocks)} blocks")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation
        
        Args:
            x: Input tensor [batch_size, input_dim, sequence_length]
            
        Returns:
            Latent tensor [batch_size, latent_dim]
        """
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Compress to latent space
        x = self.latent_conv(x)
        x = self.latent_pool(x)  # [batch_size, latent_dim, 1]
        x = x.squeeze(-1)  # [batch_size, latent_dim]
        
        return x


class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network Decoder
    Reconstructs temporal sequences from latent representation
    """
    
    def __init__(self, config: TCNAEConfig):
        """
        Initialize TCN decoder
        
        Args:
            config: TCNAE configuration
        """
        super(TCNDecoder, self).__init__()
        
        self.config = config
        
        # Expand latent to initial temporal representation
        self.latent_expand = nn.Linear(config.latent_dim, 
                                     config.decoder_channels[0] * config.sequence_length)
        
        # Build decoder blocks
        self.blocks = nn.ModuleList()
        for i in range(len(config.decoder_channels) - 1):
            in_channels = config.decoder_channels[i]
            out_channels = config.decoder_channels[i + 1]
            kernel_size = config.decoder_kernel_sizes[i]
            dilation = config.decoder_dilations[i]
            
            block = TCNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=config.dropout,
                batch_norm=config.batch_norm
            )
            self.blocks.append(block)
        
        # Final output layer
        self.output_conv = nn.Conv1d(config.decoder_channels[-1], config.input_dim, 1)
        
        logger.info(f"TCNDecoder initialized with {len(self.blocks)} blocks")
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to temporal sequence
        
        Args:
            latent: Latent tensor [batch_size, latent_dim]
            
        Returns:
            Reconstructed tensor [batch_size, input_dim, sequence_length]
        """
        batch_size = latent.size(0)
        
        # Expand latent to temporal representation
        x = self.latent_expand(latent)  # [batch_size, channels * sequence_length]
        x = x.view(batch_size, self.config.decoder_channels[0], self.config.sequence_length)
        
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Generate final output
        x = self.output_conv(x)
        
        return x


class TCNAE(nn.Module):
    """
    Complete Temporal Convolutional Autoencoder
    Combines encoder and decoder for sequence compression and reconstruction
    """
    
    def __init__(self, config: Optional[TCNAEConfig] = None):
        """
        Initialize TCNAE
        
        Args:
            config: TCNAE configuration
        """
        super(TCNAE, self).__init__()
        
        self.config = config or TCNAEConfig()
        
        # Initialize encoder and decoder
        self.encoder = TCNEncoder(self.config)
        self.decoder = TCNDecoder(self.config)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"TCNAE initialized: {self.count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation
        
        Args:
            x: Input tensor [batch_size, input_dim, sequence_length]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output
        
        Args:
            latent: Latent tensor [batch_size, latent_dim]
            
        Returns:
            Reconstructed tensor [batch_size, input_dim, sequence_length]
        """
        return self.decoder(latent)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass (encode + decode)
        
        Args:
            x: Input tensor [batch_size, input_dim, sequence_length]
            
        Returns:
            Tuple of (reconstructed, latent)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        return reconstructed, latent
    
    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor, 
                    latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute TCNAE loss components
        
        Args:
            x: Original input [batch_size, input_dim, sequence_length]
            reconstructed: Reconstructed output [batch_size, input_dim, sequence_length]
            latent: Latent representation [batch_size, latent_dim]
            
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        
        # Regularization loss (L2 on latent)
        regularization_loss = torch.mean(latent.pow(2))
        
        # Total loss
        total_loss = (self.config.reconstruction_weight * reconstruction_loss + 
                     self.config.regularization_weight * regularization_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'regularization_loss': regularization_loss
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_latent_statistics(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Compute statistics of latent representations
        
        Args:
            dataloader: DataLoader for computing statistics
            
        Returns:
            Dictionary of latent statistics
        """
        self.eval()
        latents = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                latent = self.encode(x)
                latents.append(latent.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        
        stats = {
            'mean': np.mean(latents),
            'std': np.std(latents),
            'min': np.min(latents),
            'max': np.max(latents),
            'latent_dim_variance': np.var(latents, axis=0).mean()
        }
        
        self.train()
        return stats


class TCNAETrainer:
    """
    Trainer class for TCNAE model
    """
    
    def __init__(self, model: TCNAE, device: str = 'cpu'):
        """
        Initialize TCNAE trainer
        
        Args:
            model: TCNAE model instance
            device: Training device ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        
        # Default optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'regularization_loss': []
        }
        
        logger.info(f"TCNAETrainer initialized on device: {device}")
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'regularization_loss': 0.0
        }
        
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Forward pass
            reconstructed, latent = self.model(x)
            
            # Compute loss
            losses = self.model.compute_loss(x, reconstructed, latent)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'regularization_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Forward pass
                reconstructed, latent = self.model(x)
                
                # Compute loss
                losses = self.model.compute_loss(x, reconstructed, latent)
                
                # Accumulate losses
                for key, value in losses.items():
                    val_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def fit(self, train_loader: torch.utils.data.DataLoader, 
            val_loader: torch.utils.data.DataLoader,
            epochs: int = 100, early_stopping_patience: int = 15) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        logger.info(f"Starting TCNAE training for {epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['reconstruction_loss'].append(train_metrics['reconstruction_loss'])
            self.history['regularization_loss'].append(train_metrics['regularization_loss'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Early stopping
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'best_tcnae_model.pth')
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_metrics['total_loss']:.6f}, "
                          f"Val Loss={val_metrics['total_loss']:.6f}, "
                          f"Recon Loss={train_metrics['reconstruction_loss']:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return self.history


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test TCNAE model
    config = TCNAEConfig(
        input_dim=100,
        sequence_length=4,
        latent_dim=100,
        dropout=0.1
    )
    
    model = TCNAE(config)
    
    # Generate test data
    batch_size = 32
    test_input = torch.randn(batch_size, config.input_dim, config.sequence_length)
    
    print(f"TCNAE Model Test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    with torch.no_grad():
        reconstructed, latent = model(test_input)
        print(f"Latent shape: {latent.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Test loss computation
        losses = model.compute_loss(test_input, reconstructed, latent)
        print(f"Losses: {losses}")
        
        # Test individual components
        encoded = model.encode(test_input)
        decoded = model.decode(encoded)
        print(f"Encode shape: {encoded.shape}")
        print(f"Decode shape: {decoded.shape}")
    
    print("✅ TCNAE model test completed successfully")