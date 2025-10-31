#!/usr/bin/env python3
"""
PROPER DATA LOADER FOR USD PIP CALCULATION
==========================================

Loads raw OHLC data and processed features to calculate:
1. Actual price returns (not scaled price_change)
2. USD pip values per standard lot
3. Proper feature sequences for TCNAE

Author: Claude Code Assistant
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ProperDataLoader:
    """
    Load and prepare data with proper USD pip calculation.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # FX instrument list (24 pairs)
        self.instruments = [
            'AUD_CHF', 'AUD_JPY', 'AUD_NZD', 'AUD_USD', 'CAD_JPY', 'CHF_JPY',
            'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY', 'EUR_NZD', 'EUR_USD',
            'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_JPY', 'GBP_NZD', 'GBP_USD',
            'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_JPY'
        ]
        
    def load_raw_ohlc_data(self, instrument: str) -> pd.DataFrame:
        """Load raw OHLC data for an instrument."""
        file_path = self.raw_dir / f"{instrument}_3years_H1.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def load_processed_features(self, instrument: str) -> pd.DataFrame:
        """Load processed features for an instrument."""
        file_path = self.processed_dir / f"{instrument}_H1_precomputed_features.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed features file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def calculate_returns(self, ohlc_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate 1-hour forward returns from OHLC data.
        
        Args:
            ohlc_df: OHLC dataframe with 'close' column
            
        Returns:
            Array of 1-hour forward log returns
        """
        close_prices = ohlc_df['close'].values
        
        # Calculate 1-hour forward returns: log(close[t+1] / close[t])
        forward_returns = np.log(close_prices[1:] / close_prices[:-1])
        
        # Pad with NaN for the last observation (no forward return available)
        returns = np.append(forward_returns, np.nan)
        
        return returns
    
    def align_data(self, ohlc_df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align OHLC and features data by timestamp.
        
        Args:
            ohlc_df: Raw OHLC data
            features_df: Processed features data
            
        Returns:
            Tuple of aligned (ohlc, features) dataframes
        """
        if 'time' not in ohlc_df.columns or 'time' not in features_df.columns:
            logger.warning("No time column found, using index alignment")
            min_len = min(len(ohlc_df), len(features_df))
            return ohlc_df.iloc[:min_len].copy(), features_df.iloc[:min_len].copy()
        
        # Merge on time with inner join to get aligned data
        merged = pd.merge(ohlc_df, features_df, on='time', how='inner', suffixes=('_ohlc', '_feat'))
        
        if len(merged) == 0:
            raise ValueError("No overlapping timestamps between OHLC and features data")
        
        # Split back into separate dataframes
        ohlc_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        ohlc_aligned = merged[ohlc_cols].copy()
        
        features_cols = [col for col in merged.columns if col not in ohlc_cols]
        features_aligned = merged[features_cols].copy()
        
        logger.info(f"Aligned data: {len(merged)} samples")
        
        return ohlc_aligned, features_aligned
    
    def load_all_instruments_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load raw OHLC and processed features for all instruments.
        
        Returns:
            Tuple of (ohlc_data_dict, features_data_dict)
        """
        ohlc_data = {}
        features_data = {}
        
        logger.info(f"Loading data for {len(self.instruments)} instruments...")
        
        for instrument in self.instruments:
            try:
                # Load raw OHLC
                ohlc_df = self.load_raw_ohlc_data(instrument)
                
                # Load processed features
                features_df = self.load_processed_features(instrument)
                
                # Align data
                ohlc_aligned, features_aligned = self.align_data(ohlc_df, features_df)
                
                ohlc_data[instrument] = ohlc_aligned
                features_data[instrument] = features_aligned
                
                logger.info(f"✅ {instrument}: {len(ohlc_aligned)} aligned samples")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {instrument}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(ohlc_data)} instruments")
        
        return ohlc_data, features_data
    
    def create_aligned_arrays(self, ohlc_data: Dict[str, pd.DataFrame], 
                            features_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create aligned numpy arrays for all instruments.
        
        Args:
            ohlc_data: Dictionary of OHLC dataframes
            features_data: Dictionary of features dataframes
            
        Returns:
            Tuple of (features_array, returns_array, valid_instruments)
        """
        # Find minimum length across all instruments
        min_length = min([len(df) for df in ohlc_data.values()])
        logger.info(f"Minimum length across instruments: {min_length}")
        
        # Extract features and calculate returns
        valid_instruments = []
        all_features = []
        all_returns = []
        
        for instrument in self.instruments:
            if instrument not in ohlc_data or instrument not in features_data:
                logger.warning(f"Skipping {instrument}: missing data")
                continue
            
            ohlc_df = ohlc_data[instrument].iloc[:min_length]
            features_df = features_data[instrument].iloc[:min_length]
            
            # Calculate returns
            returns = self.calculate_returns(ohlc_df)
            
            # Extract 5 features: slope_high, slope_low, volatility, direction, price_change
            feature_cols = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change']
            
            # Check if all required features are present
            missing_cols = [col for col in feature_cols if col not in features_df.columns]
            if missing_cols:
                logger.warning(f"Skipping {instrument}: missing features {missing_cols}")
                continue
            
            features = features_df[feature_cols].values  # Shape: (min_length, 5)
            
            # Check for NaN values
            if np.isnan(features).any() or np.isnan(returns[:-1]).any():  # Exclude last return (NaN)
                logger.warning(f"Skipping {instrument}: contains NaN values")
                continue
            
            all_features.append(features)
            all_returns.append(returns)
            valid_instruments.append(instrument)
        
        if len(valid_instruments) == 0:
            raise ValueError("No valid instruments with complete data")
        
        # Stack into arrays
        # Features: (n_samples, n_instruments, 5)
        features_array = np.stack(all_features, axis=1)  # Shape: (min_length, n_instruments, 5)
        
        # Returns: (n_samples, n_instruments)
        returns_array = np.stack(all_returns, axis=1)    # Shape: (min_length, n_instruments)
        
        logger.info(f"✅ Created arrays: Features {features_array.shape}, Returns {returns_array.shape}")
        logger.info(f"Valid instruments: {valid_instruments}")
        
        return features_array, returns_array, valid_instruments
    
    def load_complete_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load complete dataset with proper returns calculation.
        
        Returns:
            Tuple of (features, returns, instruments)
        """
        logger.info("🔄 Loading complete dataset with proper returns calculation...")
        
        # Load all data
        ohlc_data, features_data = self.load_all_instruments_data()
        
        # Create aligned arrays
        features, returns, instruments = self.create_aligned_arrays(ohlc_data, features_data)
        
        # Remove samples with NaN returns (last sample for each instrument)
        valid_mask = ~np.isnan(returns).any(axis=1)
        features = features[valid_mask]
        returns = returns[valid_mask]
        
        logger.info(f"✅ Final dataset: {features.shape[0]} samples, {len(instruments)} instruments")
        
        return features, returns, instruments

def load_proper_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to load properly formatted data.
    
    Returns:
        Tuple of (features, returns, instruments)
    """
    loader = ProperDataLoader()
    return loader.load_complete_dataset()