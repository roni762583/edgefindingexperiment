#!/usr/bin/env python3
"""
Data preprocessing script for Market Edge Finder Experiment.

Downloads historical data from OANDA, generates features, and prepares
datasets for training with proper validation splits and normalization.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.config import load_config, SystemConfig
from data_pull.oanda_v20_connector import OANDAConnector
from features.feature_engineering import FeatureEngineer
from features.normalization import FeatureNormalizer
from features.multiprocessor import FeatureMultiprocessor
from utils.logger import setup_logging
import pickle

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Complete data preprocessing pipeline.
    
    Handles data download, feature generation, normalization, and
    dataset preparation for training and validation.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the data preprocessor.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.oanda_connector = OANDAConnector(
            api_key=config.oanda.api_key,
            account_id=config.oanda.account_id,
            environment=config.oanda.environment
        )
        self.feature_engineer = FeatureEngineer()
        self.feature_normalizer = FeatureNormalizer()
        self.feature_multiprocessor = FeatureMultiprocessor(
            max_workers=config.data.max_workers
        )
        
        # Ensure output directories exist
        Path(config.data_path).mkdir(parents=True, exist_ok=True)
        Path(config.models_path).mkdir(parents=True, exist_ok=True)
    
    def download_historical_data(self, 
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                force_redownload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for all instruments.
        
        Args:
            start_date: Start date for data download
            end_date: End date for data download
            force_redownload: Force redownload even if data exists
            
        Returns:
            Dictionary with instrument data
        """
        logger.info("Starting historical data download...")
        
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.config.data.lookback_days)
        
        data_dict = {}
        
        for instrument in self.config.data.instruments:
            try:
                # Check if data already exists
                data_file = Path(self.config.data_path) / f"{instrument}_raw.parquet"
                
                if data_file.exists() and not force_redownload:
                    logger.info(f"Loading existing data for {instrument}")
                    data = pd.read_parquet(data_file)
                else:
                    logger.info(f"Downloading data for {instrument}")
                    data = self.oanda_connector.get_historical_data(
                        instrument=instrument,
                        granularity=self.config.data.granularity,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Save raw data
                    data.to_parquet(data_file)
                    logger.info(f"Saved raw data for {instrument}: {len(data)} records")
                
                data_dict[instrument] = data
                
            except Exception as e:
                logger.error(f"Failed to download data for {instrument}: {str(e)}")
                continue
        
        logger.info(f"Downloaded data for {len(data_dict)} instruments")
        return data_dict
    
    def generate_features(self, 
                         data_dict: Dict[str, pd.DataFrame],
                         save_features: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate features for all instruments in parallel.
        
        Args:
            data_dict: Raw OHLC data for each instrument
            save_features: Whether to save features to disk
            
        Returns:
            Dictionary with feature data
        """
        logger.info("Starting feature generation...")
        
        # Generate features in parallel
        features_dict, stats_dict = self.feature_multiprocessor.compute_features_parallel(
            data_dict
        )
        
        # Log statistics
        for instrument, stats in stats_dict.items():
            logger.info(f"{instrument}: {stats['feature_count']} features generated in {stats['processing_time']:.2f}s")
        
        if save_features:
            # Save individual feature files
            for instrument, features in features_dict.items():
                feature_file = Path(self.config.data_path) / f"{instrument}_features.parquet"
                features.to_parquet(feature_file)
                logger.info(f"Saved features for {instrument}: {len(features)} records")
        
        logger.info(f"Feature generation completed for {len(features_dict)} instruments")
        return features_dict
    
    def normalize_features(self, 
                          features_dict: Dict[str, pd.DataFrame],
                          save_normalizer: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Normalize features across all instruments.
        
        Args:
            features_dict: Feature data for each instrument
            save_normalizer: Whether to save normalizer state
            
        Returns:
            Dictionary with normalized features
        """
        logger.info("Starting feature normalization...")
        
        # Normalize features comprehensively
        normalized_features = self.feature_normalizer.normalize_features_comprehensive(
            features_dict
        )
        
        if save_normalizer:
            # Save normalizer state
            normalizer_file = Path(self.config.models_path) / "feature_normalizer.pkl"
            normalizer_state = self.feature_normalizer.get_state()
            with open(normalizer_file, 'wb') as f:
                pickle.dump(normalizer_state, f)
            logger.info(f"Saved normalizer state to {normalizer_file}")
        
        # Save normalized features
        for instrument, features in normalized_features.items():
            normalized_file = Path(self.config.data_path) / f"{instrument}_normalized.parquet"
            features.to_parquet(normalized_file)
            logger.info(f"Saved normalized features for {instrument}: {len(features)} records")
        
        logger.info(f"Feature normalization completed for {len(normalized_features)} instruments")
        return normalized_features
    
    def create_training_datasets(self, 
                               normalized_features: Dict[str, pd.DataFrame],
                               validation_split: Optional[float] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create training and validation datasets with proper temporal splits.
        
        Args:
            normalized_features: Normalized feature data
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with train/validation splits
        """
        logger.info("Creating training datasets...")
        
        if validation_split is None:
            validation_split = self.config.training.validation_split
        
        datasets = {
            'train': {},
            'validation': {}
        }
        
        for instrument, features in normalized_features.items():
            if len(features) == 0:
                logger.warning(f"No features available for {instrument}")
                continue
            
            # Sort by timestamp to ensure temporal order
            features = features.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate split point (temporal split, not random)
            split_idx = int(len(features) * (1 - validation_split))
            
            # Create splits
            train_data = features.iloc[:split_idx].copy()
            val_data = features.iloc[split_idx:].copy()
            
            datasets['train'][instrument] = train_data
            datasets['validation'][instrument] = val_data
            
            logger.info(f"{instrument}: Train={len(train_data)}, Validation={len(val_data)}")
        
        # Save datasets
        for split_name, split_data in datasets.items():
            split_dir = Path(self.config.data_path) / split_name
            split_dir.mkdir(exist_ok=True)
            
            for instrument, data in split_data.items():
                split_file = split_dir / f"{instrument}.parquet"
                data.to_parquet(split_file)
        
        logger.info("Training datasets created and saved")
        return datasets
    
    def generate_targets(self, 
                        data_dict: Dict[str, pd.DataFrame],
                        horizon_hours: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Generate target variables (future returns) for training.
        
        Args:
            data_dict: Raw OHLC data
            horizon_hours: Prediction horizon in hours
            
        Returns:
            Dictionary with target data
        """
        logger.info(f"Generating targets with {horizon_hours}h horizon...")
        
        targets_dict = {}
        
        for instrument, data in data_dict.items():
            if len(data) < horizon_hours + 1:
                logger.warning(f"Insufficient data for targets: {instrument}")
                continue
            
            # Calculate returns
            close_prices = data['close'].values
            
            # Future returns (shifted by horizon)
            future_returns = np.full(len(close_prices), np.nan)
            
            for i in range(len(close_prices) - horizon_hours):
                current_price = close_prices[i]
                future_price = close_prices[i + horizon_hours]
                future_returns[i] = (future_price - current_price) / current_price
            
            # Create targets DataFrame
            targets = pd.DataFrame({
                'timestamp': data['timestamp'],
                'target_return': future_returns
            })
            
            # Remove NaN targets
            targets = targets.dropna().reset_index(drop=True)
            
            targets_dict[instrument] = targets
            
            # Save targets
            targets_file = Path(self.config.data_path) / f"{instrument}_targets.parquet"
            targets.to_parquet(targets_file)
            
            logger.info(f"Generated {len(targets)} targets for {instrument}")
        
        logger.info(f"Target generation completed for {len(targets_dict)} instruments")
        return targets_dict
    
    def run_complete_preprocessing(self, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 force_redownload: bool = False) -> None:
        """
        Run complete preprocessing pipeline.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            force_redownload: Force redownload of data
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        try:
            # Step 1: Download historical data
            data_dict = self.download_historical_data(
                start_date=start_date,
                end_date=end_date,
                force_redownload=force_redownload
            )
            
            # Step 2: Generate targets
            targets_dict = self.generate_targets(data_dict)
            
            # Step 3: Generate features
            features_dict = self.generate_features(data_dict)
            
            # Step 4: Normalize features
            normalized_features = self.normalize_features(features_dict)
            
            # Step 5: Create training datasets
            datasets = self.create_training_datasets(normalized_features)
            
            # Step 6: Create summary
            self._create_preprocessing_summary(data_dict, features_dict, normalized_features, datasets)
            
            logger.info("Complete preprocessing pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise
    
    def _create_preprocessing_summary(self, 
                                    data_dict: Dict[str, pd.DataFrame],
                                    features_dict: Dict[str, pd.DataFrame],
                                    normalized_features: Dict[str, pd.DataFrame],
                                    datasets: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """Create preprocessing summary report."""
        summary = {
            'preprocessing_date': datetime.now().isoformat(),
            'config': {
                'instruments': self.config.data.instruments,
                'lookback_days': self.config.data.lookback_days,
                'granularity': self.config.data.granularity,
                'validation_split': self.config.training.validation_split
            },
            'data_summary': {},
            'feature_summary': {},
            'dataset_summary': {}
        }
        
        # Data summary
        for instrument, data in data_dict.items():
            if len(data) > 0:
                summary['data_summary'][instrument] = {
                    'records': len(data),
                    'start_date': data['timestamp'].min().isoformat(),
                    'end_date': data['timestamp'].max().isoformat(),
                    'missing_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns))
                }
        
        # Feature summary
        for instrument, features in features_dict.items():
            if len(features) > 0:
                summary['feature_summary'][instrument] = {
                    'records': len(features),
                    'features': list(features.columns),
                    'missing_ratio': features.isnull().sum().sum() / (len(features) * len(features.columns))
                }
        
        # Dataset summary
        for split_name, split_data in datasets.items():
            summary['dataset_summary'][split_name] = {}
            for instrument, data in split_data.items():
                summary['dataset_summary'][split_name][instrument] = {
                    'records': len(data),
                    'start_date': data['timestamp'].min().isoformat() if len(data) > 0 else None,
                    'end_date': data['timestamp'].max().isoformat() if len(data) > 0 else None
                }
        
        # Save summary
        summary_file = Path(self.config.data_path) / "preprocessing_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Preprocessing summary saved to {summary_file}")


def main():
    """Main preprocessing execution function."""
    parser = argparse.ArgumentParser(description='Market Edge Finder Data Preprocessing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force-redownload', action='store_true', help='Force redownload of data')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(
            level=args.log_level,
            log_file=Path(config.logs_path) / "preprocessing.log"
        )
        
        logger.info("Starting Market Edge Finder data preprocessing")
        logger.info(f"Configuration: {args.config or 'default'}")
        logger.info(f"Environment: {config.environment}")
        
        # Parse dates
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Run preprocessing
        preprocessor.run_complete_preprocessing(
            start_date=start_date,
            end_date=end_date,
            force_redownload=args.force_redownload
        )
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()