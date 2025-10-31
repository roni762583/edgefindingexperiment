#!/usr/bin/env python3
"""
USD PIP VALUE DATA LOADER - PROPER TRAINING TARGETS
==================================================

Calculates actual USD pip movements for next 1-hour bar as training labels.
No log returns, no conversions - direct USD pip values.

Author: Claude Code Assistant
Date: 2025-10-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class USDPipDataLoader:
    """
    Load data with proper USD pip movement calculation as training targets.
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
        
        # Pip sizes by instrument
        self.pip_sizes = {
            # Standard 4-decimal pairs
            'EUR_USD': 0.0001, 'GBP_USD': 0.0001, 'AUD_USD': 0.0001, 'NZD_USD': 0.0001,
            'USD_CHF': 0.0001, 'USD_CAD': 0.0001,
            'EUR_GBP': 0.0001, 'EUR_AUD': 0.0001, 'EUR_CAD': 0.0001, 'EUR_CHF': 0.0001, 'EUR_NZD': 0.0001,
            'GBP_AUD': 0.0001, 'GBP_CAD': 0.0001, 'GBP_CHF': 0.0001, 'GBP_NZD': 0.0001,
            'AUD_CAD': 0.0001, 'AUD_CHF': 0.0001, 'AUD_NZD': 0.0001,
            'CAD_CHF': 0.0001, 'NZD_CAD': 0.0001, 'NZD_CHF': 0.0001,
            
            # JPY pairs (2-decimal)
            'USD_JPY': 0.01, 'EUR_JPY': 0.01, 'GBP_JPY': 0.01, 'AUD_JPY': 0.01,
            'CAD_JPY': 0.01, 'CHF_JPY': 0.01, 'NZD_JPY': 0.01
        }
        
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
    
    def calculate_usd_pip_movements(self, ohlc_df: pd.DataFrame, instrument: str) -> np.ndarray:
        """
        Calculate actual USD pip movements for next 1-hour bar.
        
        Args:
            ohlc_df: OHLC dataframe with 'close' column
            instrument: FX pair name
            
        Returns:
            Array of USD pip movements for next hour
        """
        close_prices = ohlc_df['close'].values
        pip_size = self.pip_sizes.get(instrument, 0.0001)
        
        # Calculate next-hour price movements
        price_movements = close_prices[1:] - close_prices[:-1]  # Raw price change
        
        # Convert to pips
        pip_movements = price_movements / pip_size
        
        # Convert to USD value per standard lot (100,000 units)
        if instrument.endswith('_USD'):
            # Direct USD pairs: 1 pip = $10 per standard lot
            usd_pip_movements = pip_movements * 10.0
        elif instrument.startswith('USD_'):
            # USD base pairs: Use approximate conversion
            # For USD_JPY: 1 pip ≈ $9.09 (varies with rate)
            # For USD_CHF: 1 pip ≈ $9.80 (varies with rate)
            # For USD_CAD: 1 pip ≈ $7.50 (varies with rate)
            conversion_factors = {
                'USD_JPY': 9.0,   # Approximate average
                'USD_CHF': 9.8,   # Approximate average  
                'USD_CAD': 7.5    # Approximate average
            }
            factor = conversion_factors.get(instrument, 9.0)
            usd_pip_movements = pip_movements * factor
        else:
            # Cross pairs: Use base currency approximation
            # Most cross pairs roughly equivalent to $10/pip for standard lot
            base_currency = instrument.split('_')[0]
            if base_currency in ['EUR', 'GBP']:
                # EUR and GBP typically stronger than USD
                usd_pip_movements = pip_movements * 12.0
            elif base_currency in ['AUD', 'CAD', 'NZD']:
                # AUD, CAD, NZD typically weaker than USD
                usd_pip_movements = pip_movements * 8.0
            else:
                # CHF approximately equivalent
                usd_pip_movements = pip_movements * 10.0
        
        # Pad with NaN for the last observation (no next-hour movement available)
        usd_movements = np.append(usd_pip_movements, np.nan)
        
        return usd_movements
    
    def align_data(self, ohlc_df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align OHLC and features data by timestamp.
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
        Create aligned numpy arrays with USD pip movements as targets.
        """
        # Find minimum length across all instruments
        min_length = min([len(df) for df in ohlc_data.values()])
        logger.info(f"Minimum length across instruments: {min_length}")
        
        # Extract features and calculate USD pip movements
        valid_instruments = []
        all_features = []
        all_usd_movements = []
        
        for instrument in self.instruments:
            if instrument not in ohlc_data or instrument not in features_data:
                logger.warning(f"Skipping {instrument}: missing data")
                continue
            
            ohlc_df = ohlc_data[instrument].iloc[:min_length]
            features_df = features_data[instrument].iloc[:min_length]
            
            # Calculate USD pip movements as targets
            usd_movements = self.calculate_usd_pip_movements(ohlc_df, instrument)
            
            # Extract 5 features: slope_high, slope_low, volatility, direction, price_change
            feature_cols = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change']
            
            # Check if all required features are present
            missing_cols = [col for col in feature_cols if col not in features_df.columns]
            if missing_cols:
                logger.warning(f"Skipping {instrument}: missing features {missing_cols}")
                continue
            
            features = features_df[feature_cols].values  # Shape: (min_length, 5)
            
            # Check for NaN values
            if np.isnan(features).any() or np.isnan(usd_movements[:-1]).any():  # Exclude last movement (NaN)
                logger.warning(f"Skipping {instrument}: contains NaN values")
                continue
            
            all_features.append(features)
            all_usd_movements.append(usd_movements)
            valid_instruments.append(instrument)
            
            # Log USD pip movement statistics
            valid_movements = usd_movements[~np.isnan(usd_movements)]
            logger.info(f"✅ {instrument}: USD pip range [{np.min(valid_movements):.1f}, {np.max(valid_movements):.1f}]")
        
        if len(valid_instruments) == 0:
            raise ValueError("No valid instruments with complete data")
        
        # Stack into arrays
        # Features: (n_samples, n_instruments, 5)
        features_array = np.stack(all_features, axis=1)  # Shape: (min_length, n_instruments, 5)
        
        # USD movements: (n_samples, n_instruments)
        usd_movements_array = np.stack(all_usd_movements, axis=1)    # Shape: (min_length, n_instruments)
        
        logger.info(f"✅ Created arrays: Features {features_array.shape}, USD Movements {usd_movements_array.shape}")
        logger.info(f"Valid instruments: {valid_instruments}")
        
        return features_array, usd_movements_array, valid_instruments
    
    def load_complete_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load complete dataset with USD pip movements as targets.
        """
        logger.info("🔄 Loading complete dataset with USD pip movement targets...")
        
        # Load all data
        ohlc_data, features_data = self.load_all_instruments_data()
        
        # Create aligned arrays
        features, usd_movements, instruments = self.create_aligned_arrays(ohlc_data, features_data)
        
        # Remove samples with NaN USD movements (last sample for each instrument)
        valid_mask = ~np.isnan(usd_movements).any(axis=1)
        features = features[valid_mask]
        usd_movements = usd_movements[valid_mask]
        
        logger.info(f"✅ Final dataset: {features.shape[0]} samples, {len(instruments)} instruments")
        logger.info(f"✅ USD pip targets: mean abs movement = {np.mean(np.abs(usd_movements)):.2f}")
        
        return features, usd_movements, instruments

def load_usd_pip_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convenience function to load data with USD pip targets.
    """
    loader = USDPipDataLoader()
    return loader.load_complete_dataset()