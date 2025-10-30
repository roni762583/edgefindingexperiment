#!/usr/bin/env python3
"""
Pre-calculate and Cache TCNAE Latent Features

After TCNAE training (Stage 1), extract and cache all latent representations
for efficient use in subsequent stages (LightGBM training, evaluation, etc.).
"""

import torch
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.tcnae import TCNAE, TCNAEConfig
from training.basic_trainer import FeatureTargetDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LatentFeatureCache:
    """
    Cache system for pre-calculated TCNAE latent features.
    
    Allows efficient reuse of latent representations across multiple
    training stages without re-computing through TCNAE.
    """
    
    def __init__(self, cache_dir: str = "data/latent_cache"):
        """
        Initialize latent cache system.
        
        Args:
            cache_dir: Directory to store cached latent features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LatentFeatureCache initialized: {self.cache_dir}")
    
    def extract_and_cache_latents(self, 
                                 tcnae_model: TCNAE,
                                 features: np.ndarray,
                                 targets: np.ndarray,
                                 instruments: List[str],
                                 sequence_length: int = 4,
                                 batch_size: int = 64,
                                 cache_name: str = None) -> str:
        """
        Extract latent features from trained TCNAE and cache them.
        
        Args:
            tcnae_model: Trained TCNAE model
            features: Feature matrix [n_samples, n_instruments, n_features]
            targets: Target matrix [n_samples, n_instruments]
            instruments: List of instrument names
            sequence_length: Sequence length for TCNAE input
            batch_size: Batch size for processing
            cache_name: Optional cache identifier
            
        Returns:
            Cache file path
        """
        logger.info("🔧 Extracting latent features from trained TCNAE...")
        
        if cache_name is None:
            cache_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set model to evaluation mode
        tcnae_model.eval()
        device = next(tcnae_model.parameters()).device
        
        # Create dataset
        dataset = FeatureTargetDataset(features, targets, sequence_length)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        
        logger.info(f"Processing {len(dataset)} samples in {len(dataloader)} batches")
        
        # Extract latent features
        all_latents = []
        all_targets = []
        sample_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, tuple):
                    sequences, batch_targets = batch
                    all_targets.append(batch_targets.cpu().numpy())
                else:
                    sequences = batch
                    batch_targets = None
                
                # Handle both tensor and list cases for sequences
                if isinstance(sequences, list):
                    sequences = torch.stack(sequences)
                
                # Move to device
                sequences = sequences.to(device)
                
                # Extract latent representations
                latent_features = tcnae_model.encode(sequences)
                all_latents.append(latent_features.cpu().numpy())
                
                # Track sample indices
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(sequences)
                sample_indices.extend(range(start_idx, end_idx))
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        # Concatenate all latents
        latent_matrix = np.vstack(all_latents)  # [n_samples, latent_dim]
        
        if all_targets:
            target_matrix = np.vstack(all_targets)  # [n_samples, n_instruments]
        else:
            target_matrix = targets[sequence_length:]  # Align with valid samples
        
        logger.info(f"Extracted latent features: {latent_matrix.shape}")
        logger.info(f"Aligned targets: {target_matrix.shape}")
        
        # Create cache metadata
        cache_metadata = {
            'cache_name': cache_name,
            'timestamp': datetime.now().isoformat(),
            'tcnae_config': {
                'input_dim': tcnae_model.config.input_dim,
                'latent_dim': tcnae_model.config.latent_dim,
                'sequence_length': tcnae_model.config.sequence_length
            },
            'data_info': {
                'n_samples': len(latent_matrix),
                'latent_dim': latent_matrix.shape[1],
                'n_instruments': len(instruments),
                'instruments': instruments,
                'sequence_length': sequence_length
            },
            'sample_indices': sample_indices,
            'original_feature_shape': features.shape,
            'original_target_shape': targets.shape
        }
        
        # Prepare cache data
        cache_data = {
            'latent_features': latent_matrix,
            'targets': target_matrix,
            'instruments': instruments,
            'metadata': cache_metadata
        }
        
        # Save cache files
        cache_file = self.cache_dir / f"latent_cache_{cache_name}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata separately for quick inspection
        metadata_file = self.cache_dir / f"latent_metadata_{cache_name}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(cache_metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Latent features cached: {cache_file}")
        logger.info(f"✅ Metadata saved: {metadata_file}")
        
        # Log statistics
        logger.info(f"Latent statistics:")
        logger.info(f"  Shape: {latent_matrix.shape}")
        logger.info(f"  Mean: {np.mean(latent_matrix):.6f}")
        logger.info(f"  Std: {np.std(latent_matrix):.6f}")
        logger.info(f"  Range: [{np.min(latent_matrix):.6f}, {np.max(latent_matrix):.6f}]")
        
        return str(cache_file)
    
    def load_cached_latents(self, cache_name: str) -> Dict:
        """
        Load cached latent features.
        
        Args:
            cache_name: Cache identifier
            
        Returns:
            Dictionary with latent features, targets, and metadata
        """
        cache_file = self.cache_dir / f"latent_cache_{cache_name}.pkl"
        
        if not cache_file.exists():
            available_caches = list(self.cache_dir.glob("latent_cache_*.pkl"))
            raise FileNotFoundError(
                f"Cache not found: {cache_file}\n"
                f"Available caches: {[f.stem for f in available_caches]}"
            )
        
        logger.info(f"Loading cached latents: {cache_file}")
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        logger.info(f"✅ Loaded latent cache: {cache_data['latent_features'].shape}")
        
        return cache_data
    
    def list_available_caches(self) -> List[Dict]:
        """
        List all available latent caches.
        
        Returns:
            List of cache information dictionaries
        """
        cache_files = list(self.cache_dir.glob("latent_cache_*.pkl"))
        metadata_files = list(self.cache_dir.glob("latent_metadata_*.json"))
        
        cache_info = []
        
        for cache_file in sorted(cache_files):
            cache_name = cache_file.stem.replace("latent_cache_", "")
            
            # Try to load metadata
            metadata_file = self.cache_dir / f"latent_metadata_{cache_name}.json"
            
            if metadata_file.exists():
                import json
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cache_info.append({
                        'cache_name': cache_name,
                        'cache_file': str(cache_file),
                        'metadata_file': str(metadata_file),
                        'timestamp': metadata.get('timestamp'),
                        'n_samples': metadata.get('data_info', {}).get('n_samples'),
                        'latent_dim': metadata.get('data_info', {}).get('latent_dim'),
                        'n_instruments': metadata.get('data_info', {}).get('n_instruments')
                    })
                except Exception as e:
                    logger.warning(f"Could not load metadata for {cache_name}: {e}")
            else:
                # Minimal info without metadata
                cache_info.append({
                    'cache_name': cache_name,
                    'cache_file': str(cache_file),
                    'metadata_file': None,
                    'file_size': cache_file.stat().st_size
                })
        
        return cache_info
    
    def split_cached_latents(self, cache_name: str, 
                           train_ratio: float = 0.7, 
                           val_ratio: float = 0.15) -> Tuple[Dict, Dict, Dict]:
        """
        Split cached latents into train/validation/test sets.
        
        Args:
            cache_name: Cache identifier
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        cache_data = self.load_cached_latents(cache_name)
        
        latent_features = cache_data['latent_features']
        targets = cache_data['targets']
        instruments = cache_data['instruments']
        
        n_samples = len(latent_features)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split data temporally (important for financial data)
        train_data = {
            'latent_features': latent_features[:train_end],
            'targets': targets[:train_end],
            'instruments': instruments,
            'split': 'train',
            'indices': list(range(0, train_end))
        }
        
        val_data = {
            'latent_features': latent_features[train_end:val_end],
            'targets': targets[train_end:val_end],
            'instruments': instruments,
            'split': 'validation',
            'indices': list(range(train_end, val_end))
        }
        
        test_data = {
            'latent_features': latent_features[val_end:],
            'targets': targets[val_end:],
            'instruments': instruments,
            'split': 'test',
            'indices': list(range(val_end, n_samples))
        }
        
        logger.info(f"Split cached latents:")
        logger.info(f"  Train: {train_data['latent_features'].shape}")
        logger.info(f"  Validation: {val_data['latent_features'].shape}")
        logger.info(f"  Test: {test_data['latent_features'].shape}")
        
        return train_data, val_data, test_data


def create_latent_cache_from_experiment():
    """
    Example: Create latent cache after TCNAE training.
    """
    # This would typically be called after Stage 1 (TCNAE training)
    
    logger.info("🚀 Creating latent cache from trained TCNAE...")
    
    # Load the trained TCNAE model (example)
    # model_path = "results/ml_experiment_YYYYMMDD_HHMMSS/tcnae_model.pth"
    # tcnae_model = TCNAE.load_state_dict(torch.load(model_path))
    
    # Load clean data
    logger.info("Loading clean feature data...")
    
    # This is a placeholder - replace with actual data loading
    # For now, simulate having clean data available
    n_samples, n_instruments, n_features = 2000, 24, 5
    features = np.random.randn(n_samples, n_instruments, n_features)
    targets = np.random.randn(n_samples, n_instruments) * 0.01
    instruments = [f"INST_{i:02d}" for i in range(n_instruments)]
    
    # Initialize cache system
    cache_system = LatentFeatureCache()
    
    # For demonstration, create a dummy TCNAE model
    from models.tcnae import TCNAEConfig
    config = TCNAEConfig(
        input_dim=n_instruments * n_features,
        latent_dim=120,
        sequence_length=4
    )
    tcnae_model = TCNAE(config)
    
    # Extract and cache latents
    cache_file = cache_system.extract_and_cache_latents(
        tcnae_model=tcnae_model,
        features=features,
        targets=targets,
        instruments=instruments,
        cache_name="demo_experiment"
    )
    
    logger.info(f"✅ Latent cache created: {cache_file}")
    
    # Demonstrate loading
    cached_data = cache_system.load_cached_latents("demo_experiment")
    logger.info(f"✅ Successfully loaded cache: {cached_data['latent_features'].shape}")
    
    # Demonstrate splitting
    train_data, val_data, test_data = cache_system.split_cached_latents("demo_experiment")
    
    return cache_file


if __name__ == "__main__":
    try:
        # Demonstrate latent caching system
        cache_file = create_latent_cache_from_experiment()
        
        # List available caches
        cache_system = LatentFeatureCache()
        available_caches = cache_system.list_available_caches()
        
        print("\n🗃️  Available Latent Caches:")
        for cache_info in available_caches:
            print(f"  • {cache_info['cache_name']}: {cache_info.get('n_samples', 'Unknown')} samples")
        
        print(f"\n✅ Latent caching system demonstration completed!")
        
    except Exception as e:
        print(f"\n❌ Latent caching failed: {e}")
        import traceback
        traceback.print_exc()