#!/usr/bin/env python3
"""
Multiprocessing utilities for parallel feature computation
Handles 20 FX instruments efficiently using Python multiprocessing
"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
import time
from functools import partial
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.feature_engineering import FXFeatureGenerator, FeatureConfig, validate_ohlc_data
from configs.instruments import FX_INSTRUMENTS

logger = logging.getLogger(__name__)


class FeatureComputeTask:
    """Single feature computation task for multiprocessing"""
    
    def __init__(self, instrument: str, data: pd.DataFrame, config: FeatureConfig):
        self.instrument = instrument
        self.data = data
        self.config = config
    
    def __call__(self) -> Tuple[str, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Execute feature computation task
        
        Returns:
            Tuple of (instrument, features_df, metadata)
        """
        try:
            start_time = time.time()
            
            # Validate input data
            if not validate_ohlc_data(self.data, self.instrument):
                return self.instrument, None, {'error': 'Data validation failed'}
            
            # Generate features
            generator = FXFeatureGenerator(self.config)
            result_df = generator.generate_features_single_instrument(self.data, self.instrument)
            
            # Validate features
            validation_results = generator.validate_features(result_df, self.instrument)
            
            # Extract only feature columns
            feature_columns = [f'{self.instrument}_slope_high', f'{self.instrument}_slope_low',
                             f'{self.instrument}_volatility', f'{self.instrument}_direction']
            
            features_df = result_df[feature_columns].copy()
            
            computation_time = time.time() - start_time
            
            metadata = {
                'computation_time': computation_time,
                'num_bars': len(self.data),
                'valid_features': sum(1 for col in feature_columns if col in result_df.columns),
                'validation_results': validation_results,
                'success': True
            }
            
            return self.instrument, features_df, metadata
            
        except Exception as e:
            logger.error(f"Feature computation failed for {self.instrument}: {e}")
            return self.instrument, None, {'error': str(e), 'success': False}


def compute_features_single_instrument(task_data: Tuple[str, pd.DataFrame, FeatureConfig]) -> Tuple[str, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Wrapper function for multiprocessing feature computation
    
    Args:
        task_data: Tuple of (instrument, data, config)
        
    Returns:
        Tuple of (instrument, features_df, metadata)
    """
    instrument, data, config = task_data
    task = FeatureComputeTask(instrument, data, config)
    return task()


class ParallelFeatureProcessor:
    """
    Parallel feature processor for multiple FX instruments
    Efficiently computes features using multiprocessing
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, max_workers: Optional[int] = None):
        """
        Initialize parallel feature processor
        
        Args:
            config: Feature configuration
            max_workers: Maximum number of worker processes (default: CPU count)
        """
        self.config = config or FeatureConfig()
        self.max_workers = max_workers or min(mp.cpu_count(), len(FX_INSTRUMENTS))
        
        logger.info(f"ParallelFeatureProcessor initialized with {self.max_workers} workers")
    
    def compute_features_parallel(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Compute features for multiple instruments in parallel
        
        Args:
            data_dict: Dictionary mapping instrument names to OHLCV DataFrames
            
        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        if not data_dict:
            logger.warning("Empty data dictionary provided")
            return {}, {}
        
        logger.info(f"Starting parallel feature computation for {len(data_dict)} instruments")
        start_time = time.time()
        
        # Prepare tasks
        tasks = []
        for instrument, data in data_dict.items():
            if instrument in FX_INSTRUMENTS:
                tasks.append((instrument, data, self.config))
            else:
                logger.warning(f"Unknown instrument {instrument}, skipping")
        
        if not tasks:
            logger.error("No valid instruments found for processing")
            return {}, {}
        
        # Execute tasks in parallel
        features_dict = {}
        metadata_dict = {}
        
        try:
            with Pool(processes=self.max_workers) as pool:
                results = pool.map(compute_features_single_instrument, tasks)
            
            # Process results
            for instrument, features_df, metadata in results:
                metadata_dict[instrument] = metadata
                
                if features_df is not None and metadata.get('success', False):
                    features_dict[instrument] = features_df
                    logger.info(f"✅ {instrument}: Features computed successfully "
                              f"({metadata['computation_time']:.2f}s, {metadata['num_bars']} bars)")
                else:
                    error_msg = metadata.get('error', 'Unknown error')
                    logger.error(f"❌ {instrument}: Feature computation failed - {error_msg}")
        
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            return {}, {}
        
        total_time = time.time() - start_time
        success_count = len(features_dict)
        
        logger.info(f"Parallel feature computation completed: {success_count}/{len(tasks)} successful "
                   f"in {total_time:.2f}s")
        
        return features_dict, metadata_dict
    
    def compute_features_batch(self, data_dict: Dict[str, pd.DataFrame], 
                              batch_size: int = 5) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Compute features in batches to manage memory usage
        
        Args:
            data_dict: Dictionary mapping instrument names to OHLCV DataFrames
            batch_size: Number of instruments to process per batch
            
        Returns:
            Tuple of (features_dict, metadata_dict)
        """
        instruments = list(data_dict.keys())
        all_features = {}
        all_metadata = {}
        
        logger.info(f"Processing {len(instruments)} instruments in batches of {batch_size}")
        
        for i in range(0, len(instruments), batch_size):
            batch_instruments = instruments[i:i + batch_size]
            batch_data = {inst: data_dict[inst] for inst in batch_instruments}
            
            logger.info(f"Processing batch {i//batch_size + 1}: {batch_instruments}")
            
            batch_features, batch_metadata = self.compute_features_parallel(batch_data)
            
            all_features.update(batch_features)
            all_metadata.update(batch_metadata)
            
            # Brief pause between batches to avoid overwhelming system
            time.sleep(0.1)
        
        return all_features, all_metadata
    
    def validate_all_features(self, features_dict: Dict[str, pd.DataFrame], 
                            metadata_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Validate all computed features across instruments
        
        Args:
            features_dict: Dictionary of feature DataFrames
            metadata_dict: Dictionary of computation metadata
            
        Returns:
            Dictionary of validation results per instrument
        """
        validation_summary = {}
        
        for instrument, features_df in features_dict.items():
            if instrument not in metadata_dict:
                continue
            
            validation_results = metadata_dict[instrument].get('validation_results', {})
            
            # Additional cross-instrument validation
            feature_columns = [col for col in features_df.columns if col.startswith(instrument)]
            
            summary = {
                'instrument': instrument,
                'num_features': len(feature_columns),
                'num_bars': len(features_df),
                'coverage': {},
                'ranges': {},
                'validation_passed': True
            }
            
            for col in feature_columns:
                if col in features_df.columns:
                    valid_ratio = features_df[col].notna().sum() / len(features_df)
                    feature_range = features_df[col].max() - features_df[col].min()
                    
                    summary['coverage'][col] = valid_ratio
                    summary['ranges'][col] = feature_range
                    
                    # Check if validation passed
                    if valid_ratio < 0.5 or feature_range < 0.1:
                        summary['validation_passed'] = False
            
            # Include original validation results
            summary['detailed_validation'] = validation_results
            validation_summary[instrument] = summary
        
        return validation_summary
    
    def create_unified_feature_matrix(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create unified feature matrix from individual instrument features
        
        Args:
            features_dict: Dictionary of feature DataFrames per instrument
            
        Returns:
            Unified DataFrame with all features aligned by timestamp
        """
        if not features_dict:
            return pd.DataFrame()
        
        logger.info(f"Creating unified feature matrix from {len(features_dict)} instruments")
        
        # Get common time index
        all_indices = [df.index for df in features_dict.values()]
        common_index = all_indices[0]
        
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
        
        if len(common_index) == 0:
            logger.error("No common timestamps found across instruments")
            return pd.DataFrame()
        
        logger.info(f"Common time range: {len(common_index)} timestamps from "
                   f"{common_index[0]} to {common_index[-1]}")
        
        # Combine features
        unified_features = []
        
        for instrument in FX_INSTRUMENTS:
            if instrument in features_dict:
                df = features_dict[instrument].reindex(common_index)
                unified_features.append(df)
            else:
                logger.warning(f"Missing features for {instrument}, filling with NaN")
                # Create empty features for missing instrument
                feature_cols = [f'{instrument}_slope_high', f'{instrument}_slope_low',
                               f'{instrument}_volatility', f'{instrument}_direction']
                empty_df = pd.DataFrame(np.nan, index=common_index, columns=feature_cols)
                unified_features.append(empty_df)
        
        # Concatenate all features
        unified_df = pd.concat(unified_features, axis=1)
        
        logger.info(f"Unified feature matrix created: {unified_df.shape[0]} rows × {unified_df.shape[1]} columns")
        
        # Log feature coverage statistics
        coverage_stats = {}
        for col in unified_df.columns:
            valid_ratio = unified_df[col].notna().sum() / len(unified_df)
            coverage_stats[col] = valid_ratio
        
        avg_coverage = np.mean(list(coverage_stats.values()))
        logger.info(f"Average feature coverage: {avg_coverage:.1%}")
        
        return unified_df


def get_optimal_worker_count() -> int:
    """
    Determine optimal number of workers based on system resources
    
    Returns:
        Optimal worker count
    """
    cpu_count = mp.cpu_count()
    instrument_count = len(FX_INSTRUMENTS)
    
    # Use at most the number of instruments, but don't exceed CPU count
    optimal_count = min(cpu_count, instrument_count)
    
    # For memory-intensive operations, reduce worker count on systems with limited RAM
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate ~1GB per worker for feature computation
        memory_limited_count = max(1, int(available_gb))
        optimal_count = min(optimal_count, memory_limited_count)
        
    except ImportError:
        # psutil not available, use conservative estimate
        optimal_count = min(optimal_count, 4)
    
    logger.info(f"Optimal worker count: {optimal_count} (CPU: {cpu_count}, Instruments: {instrument_count})")
    return optimal_count


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data for multiple instruments
    test_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'GBP_JPY']
    test_data = {}
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='H')
    
    for instrument in test_instruments:
        # Generate realistic price data
        base_price = np.random.uniform(1.0, 2.0) if 'JPY' not in instrument else np.random.uniform(100, 150)
        returns = np.random.normal(0, 0.0001, len(dates))
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        opens = np.roll(close_prices, 1)
        opens[0] = base_price
        
        spreads = np.random.uniform(0.0001, 0.0005, len(dates))
        highs = np.maximum(opens, close_prices) + spreads / 2
        lows = np.minimum(opens, close_prices) - spreads / 2
        volumes = np.random.randint(100, 1000, len(dates))
        
        test_data[instrument] = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
    
    # Test parallel feature computation
    processor = ParallelFeatureProcessor(max_workers=get_optimal_worker_count())
    
    print(f"Testing parallel feature computation with {len(test_instruments)} instruments...")
    
    features_dict, metadata_dict = processor.compute_features_parallel(test_data)
    
    # Validate results
    validation_summary = processor.validate_all_features(features_dict, metadata_dict)
    
    print("\n=== Parallel Feature Computation Results ===")
    for instrument, summary in validation_summary.items():
        status = "✅ PASS" if summary['validation_passed'] else "❌ FAIL"
        print(f"{instrument}: {status} - {summary['num_features']} features, "
              f"{summary['num_bars']} bars, avg coverage: "
              f"{np.mean(list(summary['coverage'].values())):.1%}")
    
    # Create unified feature matrix
    unified_df = processor.create_unified_feature_matrix(features_dict)
    print(f"\nUnified feature matrix: {unified_df.shape}")
    print("Sample features:")
    print(unified_df.head())
    
    print(f"\nFeature completion:")
    for col in unified_df.columns:
        coverage = unified_df[col].notna().sum() / len(unified_df)
        print(f"  {col}: {coverage:.1%}")