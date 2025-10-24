#!/usr/bin/env python3
"""
Complete Data Pipeline for Market Edge Finder

Refactored to use separate modules for data loading, feature generation,
and temporal splitting. Maintains clean separation of concerns.

Workflow:
1. Load data using DataLoader
2. Generate features using FeaturePipeline
3. Create temporal splits using TemporalSplitter
4. Generate summary reports

This ensures no look-ahead bias and proper temporal ordering.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our separated modules
from data_loader import DataLoader, DataLoaderConfig
from feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from temporal_splitter import TemporalSplitter, TemporalSplitterConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataPipelineConfig:
    """Configuration for the complete data pipeline."""
    
    # Storage paths
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    features_path: Path = Path("data/features")
    splits_path: Path = Path("data/splits")
    
    # Data splits (temporal percentages)
    train_pct: float = 0.70    # 70% for training
    val_pct: float = 0.15      # 15% for validation
    test_pct: float = 0.15     # 15% for testing
    
    # Feature engineering
    lookback_window: int = 256  # 4 hours for sequence features
    target_horizon: int = 1     # 1 hour prediction
    
    def __post_init__(self):
        # Create directories
        for path in [self.raw_data_path, self.processed_data_path, 
                    self.features_path, self.splits_path]:
            path.mkdir(parents=True, exist_ok=True)


class CompleteDataPipeline:
    """
    Complete data pipeline for Market Edge Finder.
    
    Refactored to use separate modules for clean separation of concerns:
    1. DataLoader: Handles data loading and validation
    2. FeaturePipeline: Handles feature generation
    3. TemporalSplitter: Handles temporal data splitting
    """
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        
        # Initialize component modules
        self.data_loader = DataLoader(DataLoaderConfig(
            raw_data_path=config.raw_data_path
        ))
        
        self.feature_pipeline = FeaturePipeline(FeaturePipelineConfig(
            features_path=config.features_path,
            processed_data_path=config.processed_data_path,
            lookback_window=config.lookback_window,
            target_horizon=config.target_horizon
        ))
        
        self.temporal_splitter = TemporalSplitter(TemporalSplitterConfig(
            splits_path=config.splits_path,
            train_pct=config.train_pct,
            val_pct=config.val_pct,
            test_pct=config.test_pct,
            lookback_window=config.lookback_window
        ))
        
        # Simple progress tracking (not training checkpoints)
        self.progress_file = self.config.processed_data_path / "pipeline_progress.json"
    
    def load_existing_csv_data(self) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Load existing CSV data for all instruments.
        
        Returns:
            Dictionary mapping instrument -> raw OHLCV DataFrame
        """
        return self.data_loader.load_existing_csv_data()
    
    def generate_all_features(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Step 2: Generate ASI features for all instruments and create unified dataset.
        
        Args:
            all_data: Dictionary of instrument -> OHLCV DataFrame
            
        Returns:
            Unified DataFrame with ASI for all instruments
        """
        return self.feature_pipeline.generate_unified_asi(all_data)
    
    def calculate_all_indicators(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 2: Calculate ALL indicators on the complete dataset.
        
        This is crucial - we calculate indicators on the FULL dataset first,
        then split later to avoid look-ahead bias.
        
        Args:
            all_data: Raw OHLCV data for all instruments
            
        Returns:
            Dictionary mapping instrument -> DataFrame with all features
        """
        return self.feature_pipeline.generate_features_all_instruments(all_data)
    
    def create_temporal_splits(self, all_features: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Step 3: Create temporal train/validation/test splits.
        
        CRITICAL: This maintains temporal order - no data leakage!
        - Train: Earliest portion of data
        - Validation: Middle portion of data  
        - Test: Latest portion of data
        
        Args:
            all_features: Features for all instruments
            
        Returns:
            Nested dict: {split: {instrument: DataFrame}}
        """
        return self.temporal_splitter.create_temporal_splits(all_features)
    
    
    def generate_summary_report(self, splits: Dict[str, Dict[str, pd.DataFrame]]):
        """Generate comprehensive summary report."""
        logger.info("📋 Generating pipeline summary report...")
        
        report = []
        report.append("=" * 60)
        report.append("MARKET EDGE FINDER - DATA PIPELINE SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Configuration summary
        report.append("📊 CONFIGURATION:")
        report.append(f"  Instruments: {len(self.config.instruments)}")
        report.append(f"  Granularity: {self.config.granularity}")
        report.append(f"  Lookback: {self.config.days_lookback} days")
        report.append(f"  Splits: Train={self.config.train_pct:.0%}, Val={self.config.val_pct:.0%}, Test={self.config.test_pct:.0%}")
        report.append("")
        
        # Data summary
        total_train_rows = sum(len(df) for df in splits['train'].values())
        total_val_rows = sum(len(df) for df in splits['validation'].values())
        total_test_rows = sum(len(df) for df in splits['test'].values())
        
        report.append("📈 DATA SUMMARY:")
        report.append(f"  Training samples: {total_train_rows:,}")
        report.append(f"  Validation samples: {total_val_rows:,}")
        report.append(f"  Test samples: {total_test_rows:,}")
        report.append(f"  Total samples: {total_train_rows + total_val_rows + total_test_rows:,}")
        report.append("")
        
        # Per-instrument breakdown
        report.append("🔍 PER-INSTRUMENT BREAKDOWN:")
        for instrument in sorted(splits['train'].keys()):
            train_rows = len(splits['train'][instrument])
            val_rows = len(splits['validation'][instrument])
            test_rows = len(splits['test'][instrument])
            
            # Date ranges
            train_start = splits['train'][instrument].index[0].strftime('%Y-%m-%d')
            train_end = splits['train'][instrument].index[-1].strftime('%Y-%m-%d')
            test_start = splits['test'][instrument].index[0].strftime('%Y-%m-%d')
            test_end = splits['test'][instrument].index[-1].strftime('%Y-%m-%d')
            
            report.append(f"  {instrument}:")
            report.append(f"    Train: {train_rows:,} samples ({train_start} to {train_end})")
            report.append(f"    Val:   {val_rows:,} samples")
            report.append(f"    Test:  {test_rows:,} samples ({test_start} to {test_end})")
        
        report.append("")
        report.append("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        report.append("")
        report.append("📁 Data saved to:")
        report.append(f"  Raw data: {self.config.raw_data_path}")
        report.append(f"  Features: {self.config.features_path}")
        report.append(f"  Splits: {self.config.splits_path}")
        report.append("")
        
        # Print and save report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report to file
        report_file = self.config.processed_data_path / "pipeline_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"📋 Summary report saved to {report_file}")
    
    async def run_complete_pipeline(self):
        """Run the complete data pipeline."""
        logger.info("🚀 Starting Market Edge Finder data pipeline...")
        
        try:
            # Initialize API
            await self.initialize_api()
            
            # Step 1: Download all data
            all_data = await self.download_all_data()
            if not all_data:
                raise RuntimeError("No data downloaded successfully")
            
            # Step 2: Calculate all indicators on complete dataset
            all_features = self.calculate_all_indicators(all_data)
            if not all_features:
                raise RuntimeError("No features generated successfully")
            
            # Step 3: Create temporal splits
            splits = self.create_temporal_splits(all_features)
            if not splits['train']:
                raise RuntimeError("No training data created")
            
            # Generate summary report
            self.generate_summary_report(splits)
            
            logger.info("🎉 Complete data pipeline finished successfully!")
            return splits
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Generate features for sample data - feature engineering ONLY."""
    logger.info("🚀 Generating features for sample data...")
    
    # Load sample data using DataLoader
    config = DataPipelineConfig()
    data_loader = DataLoader(DataLoaderConfig())
    
    sample_file = Path("data/test/sample_data.csv")
    df = data_loader.load_sample_data(sample_file)
    
    logger.info(f"📊 Loaded sample data: {len(df)} bars")
    
    # Generate features using FeaturePipeline
    feature_pipeline = FeaturePipeline(FeaturePipelineConfig())
    output_file = Path("data/test/sample_data_with_features.csv")
    
    # Process sample data and save with features
    df_with_features = feature_pipeline.process_sample_data(df, 'AUD_CHF', output_file)
    
    return df_with_features


def create_price_chart_with_asi(df):
    """Create candlestick chart with ASI overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    fig.suptitle('AUD/CHF Sample Data - Candlestick Chart (300 bars)', fontsize=16, fontweight='bold')
    
    # Prepare data
    times = df.index
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    asi_values = df['AUD_CHF_asi'].values
    
    # Price statistics
    price_range = highs.max() - lows.min()
    price_change = closes[-1] - opens[0]
    price_change_pct = (price_change / opens[0]) * 100
    
    # Plot candlesticks
    for i in range(len(df)):
        x = times[i]
        open_price = opens[i]
        high_price = highs[i]
        low_price = lows[i]
        close_price = closes[i]
        
        # Color: green if close > open, red otherwise
        color = 'green' if close_price > open_price else 'red'
        
        # High-low line
        ax1.plot([x, x], [low_price, high_price], color='black', linewidth=0.8)
        
        # Candlestick body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        rect = Rectangle((mdates.date2num(x) - 0.0003, body_bottom), 
                        0.0006, body_height, 
                        facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
        ax1.add_patch(rect)
    
    # Add red close price line
    ax1.plot(times, closes, color='red', linewidth=1.5, alpha=0.8, label='Close Price')
    
    # Price chart formatting
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add price statistics box
    stats_text = f"""Price Stats:
Open: {opens[0]:.5f}
Close: {closes[-1]:.5f}
High: {highs.max():.5f}
Low: {lows.min():.5f}
Change: {price_change_pct:+.2f}%"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Plot ASI as dots
    ax2.scatter(times, asi_values, color='blue', s=8, alpha=0.7, label='ASI')
    
    # Plot HSP and LSP points if they exist
    if 'AUD_CHF_hsp' in df.columns:
        hsp_mask = ~pd.isna(df['AUD_CHF_hsp'])
        if hsp_mask.any():
            hsp_times = times[hsp_mask]
            hsp_values = df.loc[hsp_mask, 'AUD_CHF_hsp']
            
            # Draw vertical lines from ASI dots to HSP markers
            for t, hsp_val in zip(hsp_times, hsp_values):
                asi_val = df.loc[df.index[times == t], 'AUD_CHF_asi'].iloc[0]
                ax2.plot([t, t], [asi_val, hsp_val], color='red', linewidth=1, alpha=0.6)
            
            # Smaller HSP markers
            ax2.scatter(hsp_times, hsp_values, color='red', s=20, marker='^', 
                       label='HSP', zorder=5, edgecolors='darkred', linewidth=0.5)
    
    if 'AUD_CHF_lsp' in df.columns:
        lsp_mask = ~pd.isna(df['AUD_CHF_lsp'])
        if lsp_mask.any():
            lsp_times = times[lsp_mask]
            lsp_values = df.loc[lsp_mask, 'AUD_CHF_lsp']
            
            # Draw vertical lines from ASI dots to LSP markers
            for t, lsp_val in zip(lsp_times, lsp_values):
                asi_val = df.loc[df.index[times == t], 'AUD_CHF_asi'].iloc[0]
                ax2.plot([t, t], [asi_val, lsp_val], color='green', linewidth=1, alpha=0.6)
            
            # Smaller LSP markers
            ax2.scatter(lsp_times, lsp_values, color='green', s=20, marker='v', 
                       label='LSP', zorder=5, edgecolors='darkgreen', linewidth=0.5)
    
    ax2.set_ylabel('ASI Value', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ASI statistics
    asi_stats_text = f"""ASI Stats:
Start: {asi_values[0]:.2f}
Final: {asi_values[-1]:.2f}
Range: [{asi_values.min():.2f}, {asi_values.max():.2f}]
Mean: {asi_values.mean():.2f}"""
    
    ax2.text(0.02, 0.98, asi_stats_text, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Second subplot title
    ax2.set_title('Accumulation Swing Index (ASI)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    output_path = Path("data/test/price_chart_with_asi_updated.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Chart saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run pipeline (no API key needed for CSV mode)
    main()