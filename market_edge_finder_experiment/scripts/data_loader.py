#!/usr/bin/env python3
"""
Data Loading Module for Market Edge Finder

Handles loading and validation of OHLCV data from CSV files and OANDA API.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    raw_data_path: Path = Path("data/raw")
    instruments: List[str] = None
    granularity: str = "H1"
    days_lookback: int = 1095  # 3 years
    
    def __post_init__(self):
        if self.instruments is None:
            # 20 major FX pairs as specified in CLAUDE.md
            self.instruments = [
                "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD",
                "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
                "AUD_JPY", "EUR_CHF", "GBP_CHF", "CHF_JPY", "EUR_AUD",
                "GBP_AUD", "AUD_CHF", "NZD_JPY", "CAD_JPY", "AUD_NZD"
            ]


class DataLoader:
    """
    Data loader for Market Edge Finder.
    
    Handles loading OHLCV data from CSV files and validates data integrity.
    """
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        
        # Create directories if they don't exist
        self.config.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    def load_existing_csv_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load existing CSV data for all instruments.
        
        Returns:
            Dictionary mapping instrument -> raw OHLCV DataFrame
        """
        logger.info(f"📊 Loading CSV data for all instruments")
        
        all_data = {}
        csv_files = list(self.config.raw_data_path.glob("*_3years_H1.csv"))
        
        logger.info(f"📁 Found {len(csv_files)} CSV files")
        
        for csv_file in sorted(csv_files):
            instrument = csv_file.stem.replace("_3years_H1", "")
            
            try:
                # Load CSV
                df = pd.read_csv(csv_file)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time').sort_index()
                
                # Validate required columns
                if self._validate_ohlcv_data(df):
                    all_data[instrument] = df
                    logger.info(f"✅ {instrument}: {len(df)} bars loaded")
                else:
                    logger.warning(f"⚠️ {instrument}: Missing required columns")
                    
            except Exception as e:
                logger.error(f"❌ {instrument}: Failed to load - {str(e)}")
        
        logger.info(f"📊 Loaded {len(all_data)} instruments successfully")
        return all_data
    
    def load_sample_data(self, sample_file: Path) -> pd.DataFrame:
        """
        Load sample data for testing.
        
        Args:
            sample_file: Path to sample CSV file
            
        Returns:
            DataFrame with time-indexed OHLCV data
        """
        logger.info(f"📊 Loading sample data from {sample_file}")
        
        if not sample_file.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_file}")
        
        df = pd.read_csv(sample_file)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()
        
        if not self._validate_ohlcv_data(df):
            raise ValueError("Sample data validation failed")
        
        logger.info(f"✅ Sample data loaded: {len(df)} bars")
        return df
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns exist
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            return False
        
        # Check for empty data
        if len(df) == 0:
            logger.error("DataFrame is empty")
            return False
        
        # Check for NaN values in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        for col in critical_cols:
            if df[col].isna().any():
                logger.warning(f"NaN values found in {col}")
        
        # Basic OHLC consistency checks
        inconsistent_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if inconsistent_mask.any():
            inconsistent_count = inconsistent_mask.sum()
            logger.warning(f"Found {inconsistent_count} bars with inconsistent OHLC data")
        
        return True
    
    def get_data_summary(self, all_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate summary statistics for loaded data.
        
        Args:
            all_data: Dictionary of instrument -> DataFrame
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_instruments': len(all_data),
            'instruments': list(all_data.keys()),
            'date_ranges': {},
            'total_bars': 0
        }
        
        for instrument, df in all_data.items():
            if len(df) > 0:
                summary['date_ranges'][instrument] = {
                    'start': df.index[0].strftime('%Y-%m-%d %H:%M'),
                    'end': df.index[-1].strftime('%Y-%m-%d %H:%M'),
                    'bars': len(df)
                }
                summary['total_bars'] += len(df)
        
        return summary