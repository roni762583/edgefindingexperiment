#!/usr/bin/env python3
"""
Temporal Data Splitting Module for Market Edge Finder

Handles creation of temporal train/validation/test splits while maintaining
proper temporal order to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class TemporalSplitterConfig:
    """Configuration for temporal data splitting."""
    splits_path: Path = Path("data/splits")
    
    # Data splits (temporal percentages)
    train_pct: float = 0.70    # 70% for training
    val_pct: float = 0.15      # 15% for validation
    test_pct: float = 0.15     # 15% for testing
    
    # Minimum data requirements
    lookback_window: int = 256  # Minimum lookback for sequences
    min_val_samples: int = 10   # Minimum validation samples
    min_test_samples: int = 10  # Minimum test samples
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.splits_path.mkdir(parents=True, exist_ok=True)


class TemporalSplitter:
    """
    Temporal data splitter for Market Edge Finder.
    
    Creates train/validation/test splits while maintaining temporal order
    to prevent data leakage in financial time series.
    """
    
    def __init__(self, config: TemporalSplitterConfig):
        self.config = config
    
    def create_temporal_splits(self, all_features: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create temporal train/validation/test splits.
        
        CRITICAL: This maintains temporal order - no data leakage!
        - Train: Earliest portion of data
        - Validation: Middle portion of data  
        - Test: Latest portion of data
        
        Args:
            all_features: Features for all instruments
            
        Returns:
            Nested dict: {split: {instrument: DataFrame}}
        """
        logger.info("✂️ Creating temporal train/validation/test splits...")
        
        splits = {'train': {}, 'validation': {}, 'test': {}}
        successful_splits = 0
        
        for instrument, df in all_features.items():
            logger.info(f"📊 Creating splits for {instrument}...")
            
            try:
                split_result = self._split_single_instrument(instrument, df)
                
                if split_result is not None:
                    train_df, val_df, test_df = split_result
                    
                    splits['train'][instrument] = train_df
                    splits['validation'][instrument] = val_df
                    splits['test'][instrument] = test_df
                    successful_splits += 1
                    
                    logger.info(
                        f"✅ {instrument}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
                    )
                else:
                    logger.warning(f"⚠️ {instrument}: Split creation failed")
                    
            except Exception as e:
                logger.error(f"❌ {instrument}: Split creation error - {str(e)}")
        
        # Save splits to disk
        self._save_splits(splits)
        
        logger.info(f"🎉 Temporal splits created for {successful_splits}/{len(all_features)} instruments")
        return splits
    
    def _split_single_instrument(self, instrument: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal splits for a single instrument.
        
        Args:
            instrument: Instrument name
            df: Features DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df) or None if insufficient data
        """
        if len(df) < 100:
            logger.warning(f"⚠️ Insufficient data for splits: {instrument} ({len(df)} rows)")
            return None
        
        # Calculate split points (temporal order preserved)
        total_rows = len(df)
        train_end = int(total_rows * self.config.train_pct)
        val_end = int(total_rows * (self.config.train_pct + self.config.val_pct))
        
        # Ensure minimum lookback window for validation and test
        train_end = max(train_end, self.config.lookback_window)
        val_end = max(val_end, train_end + self.config.lookback_window)
        
        # Create temporal splits
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Validate splits
        if len(train_df) < self.config.lookback_window:
            logger.warning(f"⚠️ Insufficient training data: {instrument}")
            return None
        
        if len(val_df) < self.config.min_val_samples:
            logger.warning(f"⚠️ Insufficient validation data: {instrument}")
            return None
        
        if len(test_df) < self.config.min_test_samples:
            logger.warning(f"⚠️ Insufficient test data: {instrument}")
            return None
        
        return train_df, val_df, test_df
    
    def _save_splits(self, splits: Dict[str, Dict[str, pd.DataFrame]]):
        """Save train/validation/test splits to disk."""
        for split_name, split_data in splits.items():
            split_dir = self.config.splits_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            for instrument, df in split_data.items():
                file_path = split_dir / f"{instrument}_{split_name}.parquet"
                df.to_parquet(file_path, compression='snappy')
            
            logger.info(f"💾 Saved {split_name} split: {len(split_data)} instruments")
    
    def load_splits(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load existing splits from disk.
        
        Returns:
            Nested dict: {split: {instrument: DataFrame}}
        """
        logger.info("📂 Loading existing splits from disk...")
        
        splits = {'train': {}, 'validation': {}, 'test': {}}
        
        for split_name in splits.keys():
            split_dir = self.config.splits_path / split_name
            
            if not split_dir.exists():
                logger.warning(f"⚠️ Split directory not found: {split_dir}")
                continue
            
            parquet_files = list(split_dir.glob("*.parquet"))
            
            for file_path in parquet_files:
                instrument = file_path.stem.replace(f"_{split_name}", "")
                
                try:
                    df = pd.read_parquet(file_path)
                    splits[split_name][instrument] = df
                    logger.debug(f"✅ Loaded {split_name} split for {instrument}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {split_name} split for {instrument}: {str(e)}")
        
        total_instruments = len(splits['train'])
        logger.info(f"📂 Loaded splits for {total_instruments} instruments")
        
        return splits
    
    def get_split_summary(self, splits: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Generate summary statistics for data splits.
        
        Args:
            splits: Dictionary of splits
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_instruments': len(splits['train']),
            'instruments': list(splits['train'].keys()),
            'split_sizes': {},
            'date_ranges': {}
        }
        
        # Calculate split sizes
        for split_name in ['train', 'validation', 'test']:
            total_samples = sum(len(df) for df in splits[split_name].values())
            summary['split_sizes'][split_name] = total_samples
        
        # Calculate date ranges for each instrument
        for instrument in summary['instruments']:
            if instrument in splits['train']:
                train_df = splits['train'][instrument]
                test_df = splits['test'][instrument]
                
                summary['date_ranges'][instrument] = {
                    'train_start': train_df.index[0].strftime('%Y-%m-%d'),
                    'train_end': train_df.index[-1].strftime('%Y-%m-%d'),
                    'test_start': test_df.index[0].strftime('%Y-%m-%d'),
                    'test_end': test_df.index[-1].strftime('%Y-%m-%d')
                }
        
        return summary