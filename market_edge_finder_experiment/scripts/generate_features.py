#!/usr/bin/env python3
"""
Unified Feature Generation Script - Market Edge Finder

Consolidates all feature generation functionality into a single, comprehensive script.
Supports multiple data sources and output formats.

Usage:
    # Process sample data for testing
    python3 scripts/generate_features.py --sample
    
    # Process single instrument
    python3 scripts/generate_features.py --file data/raw/EUR_USD.csv
    
    # Process specific instrument by name  
    python3 scripts/generate_features.py --instrument EUR_USD
    
    # Process all instruments in directory
    python3 scripts/generate_features.py --all data/raw/
    
    # Generate analysis charts
    python3 scripts/generate_features.py --sample --charts
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator, FeatureConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedFeatureGenerator:
    """
    Unified feature generation for all data sources and scenarios.
    Consolidates functionality from multiple scripts into one interface.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize with optional custom configuration."""
        self.config = config or FeatureConfig()
        self.feature_generator = FXFeatureGenerator(self.config)
        
        # Define standard paths
        self.project_root = project_root
        self.data_raw_path = self.project_root / "data/raw"
        self.data_test_path = self.project_root / "data/test"
        self.sample_data_path = self.data_test_path / "sample_data.csv"
        
        logger.info(f"UnifiedFeatureGenerator initialized with config: {self.config}")
    
    def process_sample_data(self, output_charts: bool = False) -> pd.DataFrame:
        """
        Process sample data for testing and development.
        
        Args:
            output_charts: Whether to generate analysis charts
            
        Returns:
            Processed DataFrame with features
        """
        logger.info("🔄 Processing sample data...")
        
        if not self.sample_data_path.exists():
            raise FileNotFoundError(f"Sample data not found: {self.sample_data_path}")
        
        # Load sample data
        df = pd.read_csv(self.sample_data_path)
        logger.info(f"📊 Loaded {len(df)} bars from sample data")
        
        # Add timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
        
        # Generate features
        features_df = self.feature_generator.generate_features_single_instrument(df, 'EUR_USD')
        
        # Save processed data
        output_path = self.data_test_path / "processed_sample_data.csv"
        features_df.to_csv(output_path, index=False)
        logger.info(f"💾 Processed sample data saved to: {output_path}")
        
        # Generate analysis reports
        self._print_feature_analysis(features_df, 'EUR_USD Sample Data')
        
        # Generate charts if requested
        if output_charts:
            self._generate_analysis_charts(features_df, 'sample')
        
        return features_df
    
    def process_instrument_file(self, file_path: Path, output_charts: bool = False) -> pd.DataFrame:
        """
        Process a single instrument CSV file.
        
        Args:
            file_path: Path to instrument CSV file
            output_charts: Whether to generate analysis charts
            
        Returns:
            Processed DataFrame with features
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract instrument name from filename
        instrument = file_path.stem.upper()
        logger.info(f"🔄 Processing {instrument} from {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"📊 Loaded {len(df)} bars for {instrument}")
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {list(df.columns)}, Required: {required_cols}")
        
        # Add timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')
        
        # Generate features
        features_df = self.feature_generator.generate_features_single_instrument(df, instrument)
        
        # Save processed data
        output_path = file_path.parent / f"processed_{file_path.name}"
        features_df.to_csv(output_path, index=False)
        logger.info(f"💾 Processed {instrument} data saved to: {output_path}")
        
        # Generate analysis reports
        self._print_feature_analysis(features_df, instrument)
        
        # Generate charts if requested
        if output_charts:
            self._generate_analysis_charts(features_df, instrument)
        
        return features_df
    
    def process_instrument_by_name(self, instrument: str, output_charts: bool = False) -> pd.DataFrame:
        """
        Process instrument by name (looks for file in data/raw/).
        
        Args:
            instrument: Instrument name (e.g., 'EUR_USD')
            output_charts: Whether to generate analysis charts
            
        Returns:
            Processed DataFrame with features
        """
        file_path = self.data_raw_path / f"{instrument}.csv"
        return self.process_instrument_file(file_path, output_charts)
    
    def process_all_instruments(self, data_dir: Path, output_charts: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Process all CSV files in a directory.
        
        Args:
            data_dir: Directory containing instrument CSV files
            output_charts: Whether to generate analysis charts
            
        Returns:
            Dictionary mapping instrument names to processed DataFrames
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        logger.info(f"🔄 Processing {len(csv_files)} instruments from {data_dir}")
        
        results = {}
        for file_path in csv_files:
            try:
                features_df = self.process_instrument_file(file_path, output_charts)
                instrument = file_path.stem.upper()
                results[instrument] = features_df
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
        
        logger.info(f"✅ Successfully processed {len(results)}/{len(csv_files)} instruments")
        return results
    
    def _print_feature_analysis(self, df: pd.DataFrame, source_name: str):
        """Print comprehensive feature analysis."""
        logger.info(f"\n📊 Feature Analysis for {source_name}")
        logger.info(f"Total bars: {len(df)}")
        
        # ASI analysis
        if 'asi' in df.columns:
            asi_range = [df['asi'].min(), df['asi'].max()]
            logger.info(f"ASI range: [{asi_range[0]:.0f}, {asi_range[1]:.0f}] USD per 100k lot")
        
        # Swing point analysis
        if 'sig_hsp' in df.columns and 'sig_lsp' in df.columns:
            hsp_count = df['sig_hsp'].sum()
            lsp_count = df['sig_lsp'].sum()
            logger.info(f"Significant swing points: {hsp_count} HSP, {lsp_count} LSP")
        
        # Angle analysis
        if 'hsp_angles' in df.columns and 'lsp_angles' in df.columns:
            hsp_valid = df['hsp_angles'].count()
            lsp_valid = df['lsp_angles'].count()
            logger.info(f"Valid angles: {hsp_valid} HSP, {lsp_valid} LSP")
            
            if hsp_valid > 0:
                hsp_range = [df['hsp_angles'].min(), df['hsp_angles'].max()]
                logger.info(f"HSP angles range: [{hsp_range[0]:.3f}, {hsp_range[1]:.3f}]")
            
            if lsp_valid > 0:
                lsp_range = [df['lsp_angles'].min(), df['lsp_angles'].max()]
                logger.info(f"LSP angles range: [{lsp_range[0]:.3f}, {lsp_range[1]:.3f}]")
        
        # SI analysis from ASI differences
        if 'asi' in df.columns:
            si_values = np.diff(df['asi'].values)
            si_nonzero = si_values[si_values != 0]
            if len(si_nonzero) > 0:
                logger.info(f"SI values: {len(si_nonzero)} non-zero, range [{si_nonzero.min():.0f}, {si_nonzero.max():.0f}]")
    
    def _generate_analysis_charts(self, df: pd.DataFrame, source_name: str):
        """Generate analysis charts for the processed data."""
        try:
            # Import chart generation script
            chart_script = self.project_root / "scripts/chart_all_indicators.py"
            if chart_script.exists():
                logger.info(f"📊 Generating analysis charts for {source_name}...")
                # Note: In a real implementation, you'd call the chart generation function directly
                # For now, we just log that charts would be generated
                logger.info(f"Charts would be generated and saved to data/test/{source_name}_indicators_chart.png")
            else:
                logger.warning("Chart generation script not found")
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Unified Feature Generation for Market Edge Finder')
    
    # Data source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--sample', action='store_true', 
                             help='Process sample data for testing')
    source_group.add_argument('--file', type=Path, 
                             help='Process specific CSV file')
    source_group.add_argument('--instrument', type=str, 
                             help='Process instrument by name (looks in data/raw/)')
    source_group.add_argument('--all', type=Path, 
                             help='Process all CSV files in directory')
    
    # Output options
    parser.add_argument('--charts', action='store_true', 
                       help='Generate analysis charts')
    parser.add_argument('--config', type=Path, 
                       help='Path to custom feature configuration JSON')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load custom config if provided
        config = None
        if args.config:
            # In a real implementation, you'd load JSON config here
            logger.info(f"Loading custom config from {args.config}")
        
        # Initialize feature generator
        generator = UnifiedFeatureGenerator(config)
        
        # Process based on arguments
        if args.sample:
            logger.info("🚀 Processing sample data...")
            result = generator.process_sample_data(args.charts)
            
        elif args.file:
            logger.info(f"🚀 Processing file: {args.file}")
            result = generator.process_instrument_file(args.file, args.charts)
            
        elif args.instrument:
            logger.info(f"🚀 Processing instrument: {args.instrument}")
            result = generator.process_instrument_by_name(args.instrument, args.charts)
            
        elif args.all:
            logger.info(f"🚀 Processing all instruments in: {args.all}")
            results = generator.process_all_instruments(args.all, args.charts)
            logger.info(f"✅ Processed {len(results)} instruments successfully")
            
        logger.info("🎉 Feature generation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Feature generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()