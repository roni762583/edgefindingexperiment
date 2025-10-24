#!/usr/bin/env python3
"""
Feature Generation Pipeline for Market Edge Finder

Handles feature engineering and processing for trading data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class FeaturePipelineConfig:
    """Configuration for feature generation pipeline."""
    features_path: Path = Path("data/features")
    processed_data_path: Path = Path("data/processed")
    lookback_window: int = 256  # 4 hours for sequence features
    target_horizon: int = 1     # 1 hour prediction
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.features_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)


class FeaturePipeline:
    """
    Feature generation pipeline for Market Edge Finder.
    
    Handles ASI calculation, swing point detection, and other technical indicators.
    """
    
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        
        # Import feature generator
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from features.feature_engineering import FXFeatureGenerator, FeatureConfig
        
        self.feature_generator = FXFeatureGenerator(FeatureConfig())
    
    def generate_features_single_instrument(self, df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        Generate features for a single instrument.
        
        Args:
            df: OHLCV DataFrame
            instrument: Instrument name (e.g., 'AUD_CHF')
            
        Returns:
            DataFrame with original data plus generated features
        """
        logger.info(f"⚙️ Generating features for {instrument}...")
        
        try:
            # Generate features using the feature engine
            features_df = self.feature_generator.generate_features_single_instrument(df, instrument)
            
            if features_df is None or len(features_df) == 0:
                logger.warning(f"⚠️ No features generated for {instrument}")
                return df
            
            # Add target variable if not present
            if 'target_return' not in features_df.columns:
                features_df = self._add_target_variable(features_df)
            
            logger.info(f"✅ {instrument}: Generated {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature generation failed for {instrument}: {str(e)}")
            return df
    
    def generate_features_all_instruments(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate features for all instruments.
        
        Args:
            all_data: Dictionary mapping instrument -> OHLCV DataFrame
            
        Returns:
            Dictionary mapping instrument -> DataFrame with features
        """
        logger.info(f"⚙️ Generating features for {len(all_data)} instruments")
        
        all_features = {}
        processed_count = 0
        
        for i, (instrument, df) in enumerate(all_data.items(), 1):
            logger.info(f"📊 [{i}/{len(all_data)}] Processing {instrument}...")
            
            try:
                # Generate complete feature set
                features_df = self.generate_features_single_instrument(df, instrument)
                
                if features_df is None or len(features_df) == 0:
                    logger.warning(f"⚠️ No features generated for {instrument}")
                    continue
                
                # Remove any NaN rows (from indicator calculations)
                initial_rows = len(features_df)
                features_df = features_df.dropna()
                final_rows = len(features_df)
                
                if initial_rows != final_rows:
                    logger.info(f"🧹 {instrument}: Removed {initial_rows - final_rows} NaN rows")
                
                all_features[instrument] = features_df
                processed_count += 1
                
                # Save processed features
                self._save_features(instrument, features_df)
                
                logger.info(f"✅ {instrument}: {len(features_df)} rows, {len(features_df.columns)} features")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {instrument}: {str(e)}")
                continue
        
        logger.info(f"🎉 Feature engineering completed for {processed_count}/{len(all_data)} instruments")
        return all_features
    
    def generate_unified_asi(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate unified ASI dataset for all instruments.
        
        Args:
            all_data: Dictionary of instrument -> OHLCV DataFrame
            
        Returns:
            Unified DataFrame with ASI for all instruments
        """
        logger.info(f"⚙️ Generating unified ASI for {len(all_data)} instruments")
        
        unified_asi = None
        processed_count = 0
        
        for instrument, df in all_data.items():
            logger.info(f"📊 Processing {instrument} ASI...")
            
            try:
                # Generate ASI features
                result = self.feature_generator.generate_features_single_instrument(df, instrument)
                
                # Extract ASI column
                asi_col = f'{instrument}_asi'
                if asi_col in result.columns:
                    asi_series = result[asi_col]
                    
                    if unified_asi is None:
                        # Initialize unified DataFrame
                        unified_asi = pd.DataFrame(index=asi_series.index)
                    
                    # Add this instrument's ASI
                    unified_asi[asi_col] = asi_series
                    processed_count += 1
                    
                    final_asi = asi_series.dropna().iloc[-1]
                    logger.info(f"✅ {instrument}: {len(asi_series)} bars (final ASI: {final_asi:.2f})")
                else:
                    logger.warning(f"⚠️ {instrument}: ASI column missing")
                    
            except Exception as e:
                logger.error(f"❌ {instrument}: ASI generation failed - {str(e)}")
        
        logger.info(f"🎉 Generated ASI for {processed_count}/{len(all_data)} instruments")
        
        # Save unified ASI
        if unified_asi is not None:
            unified_asi_file = self.config.features_path / "unified_asi_all_instruments.csv"
            unified_asi.to_csv(unified_asi_file)
            logger.info(f"💾 Saved unified ASI to {unified_asi_file}")
        
        return unified_asi
    
    def process_sample_data(self, df: pd.DataFrame, instrument: str, output_file: Path) -> pd.DataFrame:
        """
        Process sample data and save with features.
        
        Args:
            df: Sample OHLCV DataFrame
            instrument: Instrument name
            output_file: Output file path
            
        Returns:
            DataFrame with features added
        """
        logger.info(f"🚀 Processing sample data for {instrument}...")
        
        # Generate complete features including ASI, HSP, LSP swing points
        df_with_features = self.feature_generator.generate_features_single_instrument(df, instrument)
        
        # Extract the new feature columns
        feature_columns = [col for col in df_with_features.columns if col not in df.columns]
        for col in feature_columns:
            df[col] = df_with_features[col]
        
        # Reset index to get time column back
        df = df.reset_index()
        
        # Save updated CSV with features
        df.to_csv(output_file, index=False)
        
        # Log completion with feature counts
        sig_hsp_count = df[f'{instrument}_sig_hsp'].sum()
        sig_lsp_count = df[f'{instrument}_sig_lsp'].sum()
        local_hsp_count = df[f'{instrument}_local_hsp'].sum()
        local_lsp_count = df[f'{instrument}_local_lsp'].sum()
        
        logger.info(f"✅ Created {output_file} with all features")
        logger.info(f"📊 Features: {sig_hsp_count} sig HSP, {sig_lsp_count} sig LSP, {local_hsp_count} local HSP, {local_lsp_count} local LSP")
        
        return df
    
    def _add_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable (1-hour forward return)."""
        if 'close' not in df.columns:
            raise ValueError("Close price not found in DataFrame")
        
        # Calculate 1-hour forward return
        df['target_return'] = df['close'].pct_change(periods=self.config.target_horizon).shift(-self.config.target_horizon)
        
        # Create binary classification target (positive/negative return)
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        return df
    
    def _save_features(self, instrument: str, features_df: pd.DataFrame):
        """Save processed features to parquet file."""
        features_file = self.config.features_path / f"{instrument}_features.parquet"
        features_df.to_parquet(features_file, compression='snappy')
        logger.debug(f"💾 Saved features for {instrument} to {features_file}")
    
    def get_feature_summary(self, all_features: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate summary of generated features.
        
        Args:
            all_features: Dictionary of instrument -> features DataFrame
            
        Returns:
            Feature summary statistics
        """
        summary = {
            'total_instruments': len(all_features),
            'instruments': list(all_features.keys()),
            'feature_counts': {},
            'total_samples': 0
        }
        
        for instrument, df in all_features.items():
            summary['feature_counts'][instrument] = {
                'samples': len(df),
                'features': len(df.columns)
            }
            summary['total_samples'] += len(df)
        
        # Calculate common feature names across instruments
        if all_features:
            all_columns = set()
            for df in all_features.values():
                all_columns.update(df.columns)
            summary['total_unique_features'] = len(all_columns)
        
        return summary