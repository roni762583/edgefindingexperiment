#!/usr/bin/env python3
"""
Download Historical Data for All 20 FX Pairs
Parallel downloader for Edge Finding Experiment using OANDA v20 API
"""

import os
import sys
import logging
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.instruments import FX_INSTRUMENTS, get_instrument_priority
from data_pull.download_oanda_data import EdgeFindingOandaDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiInstrumentDownloader:
    """
    Parallel downloader for all 20 FX instruments
    Uses process pool to download multiple instruments simultaneously
    """
    
    def __init__(self, 
                 start_date: str = "2022-01-01", 
                 end_date: str = "2025-10-22",
                 granularity: str = "M1",
                 max_workers: int = 4):
        """
        Initialize multi-instrument downloader
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data  
            granularity: Timeframe (M1, M5, H1, H4, D)
            max_workers: Maximum parallel download processes
        """
        self.start_date = start_date
        self.end_date = end_date
        self.granularity = granularity
        self.max_workers = min(max_workers, mp.cpu_count())
        
        # Create data directory
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Multi-instrument downloader initialized")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Granularity: {granularity}")
        logger.info(f"Max workers: {max_workers}")
        logger.info(f"Data directory: {self.data_dir}")

    def download_all_instruments(self) -> dict:
        """
        Download historical data for all 20 FX instruments in parallel
        
        Returns:
            Dictionary with download results per instrument
        """
        logger.info("🚀 Starting parallel download for all 20 FX instruments")
        
        # Prepare download tasks
        download_tasks = []
        for instrument in FX_INSTRUMENTS:
            output_path = self.data_dir / f"{instrument}_{self.granularity}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}.csv"
            download_tasks.append((instrument, str(output_path)))
        
        # Execute downloads in parallel
        results = {}
        completed_count = 0
        failed_count = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_instrument = {
                executor.submit(
                    download_single_instrument,
                    instrument,
                    output_path,
                    self.start_date,
                    self.end_date,
                    self.granularity
                ): instrument for instrument, output_path in download_tasks
            }
            
            # Process completed downloads
            for future in as_completed(future_to_instrument):
                instrument = future_to_instrument[future]
                try:
                    success, error_msg = future.result()
                    results[instrument] = {
                        'success': success,
                        'error': error_msg
                    }
                    
                    if success:
                        completed_count += 1
                        logger.info(f"✅ {instrument}: Download completed ({completed_count}/{len(FX_INSTRUMENTS)})")
                    else:
                        failed_count += 1
                        logger.error(f"❌ {instrument}: Download failed - {error_msg}")
                        
                except Exception as e:
                    failed_count += 1
                    results[instrument] = {
                        'success': False,
                        'error': str(e)
                    }
                    logger.error(f"❌ {instrument}: Download exception - {e}")
        
        # Summary report
        logger.info("📊 Download Summary:")
        logger.info(f"   Total instruments: {len(FX_INSTRUMENTS)}")
        logger.info(f"   Successful: {completed_count}")
        logger.info(f"   Failed: {failed_count}")
        logger.info(f"   Success rate: {(completed_count/len(FX_INSTRUMENTS)*100):.1f}%")
        
        # List failed downloads
        if failed_count > 0:
            logger.warning("Failed downloads:")
            for instrument, result in results.items():
                if not result['success']:
                    logger.warning(f"   {instrument}: {result['error']}")
        
        return results

    def validate_downloads(self) -> dict:
        """
        Validate all downloaded data files
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating downloaded data files...")
        
        validation_results = {}
        for instrument in FX_INSTRUMENTS:
            output_path = self.data_dir / f"{instrument}_{self.granularity}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}.csv"
            
            if output_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(output_path)
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    
                    validation_results[instrument] = {
                        'exists': True,
                        'rows': len(df),
                        'size_mb': round(file_size_mb, 2),
                        'date_range': f"{df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}" if len(df) > 0 else "No data",
                        'valid': len(df) > 1000  # Minimum threshold
                    }
                    
                    if len(df) > 1000:
                        logger.info(f"✅ {instrument}: {len(df):,} rows, {file_size_mb:.2f} MB")
                    else:
                        logger.warning(f"⚠️  {instrument}: Only {len(df)} rows - may be incomplete")
                        
                except Exception as e:
                    validation_results[instrument] = {
                        'exists': True,
                        'valid': False,
                        'error': str(e)
                    }
                    logger.error(f"❌ {instrument}: Validation error - {e}")
            else:
                validation_results[instrument] = {
                    'exists': False,
                    'valid': False,
                    'error': 'File not found'
                }
                logger.error(f"❌ {instrument}: File not found")
        
        return validation_results

def download_single_instrument(instrument: str, 
                              output_path: str,
                              start_date: str,
                              end_date: str, 
                              granularity: str) -> Tuple[bool, str]:
    """
    Download data for a single instrument (used by process pool)
    
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Create downloader instance
        downloader = EdgeFindingOandaDownloader.from_env_file(
            env_file=str(Path(__file__).parent.parent / ".env")
        )
        
        # Download data
        success = downloader.download_historical_csv(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            granularity=granularity
        )
        
        if success:
            return True, ""
        else:
            return False, "Download returned False"
            
    except Exception as e:
        return False, str(e)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download all 20 FX instruments")
    parser.add_argument("--start-date", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-10-22", help="End date (YYYY-MM-DD)")
    parser.add_argument("--granularity", default="M1", help="Timeframe (M1, M5, H1)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing files")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = MultiInstrumentDownloader(
        start_date=args.start_date,
        end_date=args.end_date,
        granularity=args.granularity,
        max_workers=args.max_workers
    )
    
    if args.validate_only:
        # Only validate existing downloads
        validation_results = downloader.validate_downloads()
        valid_count = sum(1 for r in validation_results.values() if r.get('valid', False))
        logger.info(f"Validation complete: {valid_count}/{len(FX_INSTRUMENTS)} files valid")
    else:
        # Download all instruments
        download_results = downloader.download_all_instruments()
        
        # Validate downloads
        validation_results = downloader.validate_downloads()
        
        # Final summary
        successful_downloads = sum(1 for r in download_results.values() if r['success'])
        valid_files = sum(1 for r in validation_results.values() if r.get('valid', False))
        
        logger.info("🎉 Multi-instrument download complete!")
        logger.info(f"   Downloads successful: {successful_downloads}/{len(FX_INSTRUMENTS)}")
        logger.info(f"   Files validated: {valid_files}/{len(FX_INSTRUMENTS)}")
        
        if valid_files == len(FX_INSTRUMENTS):
            logger.info("✅ All instruments downloaded and validated successfully!")
            return 0
        else:
            logger.warning("⚠️  Some downloads incomplete or invalid")
            return 1

if __name__ == "__main__":
    sys.exit(main())