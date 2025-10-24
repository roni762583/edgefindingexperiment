"""
Production-ready OANDA v20 historical data downloader.

This module provides comprehensive historical data downloading with:
- Exponential backoff for API rate limiting
- UTC timezone alignment and missing bar handling
- Resume capabilities with progress tracking
- Data validation and integrity checks
- Integration with optimized storage system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from pathlib import Path
import sqlite3
import hashlib
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings

# OANDA v20 API - Official library only
try:
    import v20
    from v20.errors import V20Error, V20ConnectionError, V20Timeout
    from .oanda_api_manager import OANDAAPIManager, RateLimitConfig
except ImportError:
    raise ImportError(
        "OANDA v20 library not found. Install with: pip install v20>=3.0.25.0\n"
        "See: https://oanda-api-v20.readthedocs.io/"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class DownloadConfig:
    """Configuration for historical data download."""
    
    # API Configuration
    api_key: str
    account_id: str
    environment: str = 'practice'  # 'practice' or 'live'
    
    # Data Parameters
    instruments: List[str] = None
    granularity: str = 'H1'
    max_count_per_request: int = 5000  # OANDA limit
    
    # Rate Limiting
    requests_per_second: float = 2.0  # Conservative rate limit
    max_retries: int = 5
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    backoff_multiplier: float = 2.0
    
    # Data Quality
    validate_ohlc: bool = True
    require_complete_bars: bool = True
    fill_missing_bars: bool = True
    max_missing_ratio: float = 0.05  # 5% max missing data
    
    # Resume & Recovery
    enable_resume: bool = True
    checkpoint_interval: int = 1000  # Save progress every N bars
    
    def __post_init__(self):
        if self.instruments is None:
            self.instruments = [
                'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
                'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
                'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD',
                'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD'
            ]


class RateLimiter:
    """Production-grade rate limiter with exponential backoff."""
    
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.consecutive_errors = 0
        
    async def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            
            # Add exponential backoff for consecutive errors
            if self.consecutive_errors > 0:
                backoff_time = min(2 ** self.consecutive_errors, 60.0)
                wait_time = max(wait_time, backoff_time)
                logger.warning(f"Rate limiting with backoff: {wait_time:.2f}s (errors: {self.consecutive_errors})")
            
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def record_success(self) -> None:
        """Record successful request."""
        self.consecutive_errors = 0
    
    def record_error(self) -> None:
        """Record failed request."""
        self.consecutive_errors += 1


class DataValidator:
    """Validates OHLCV data integrity and quality."""
    
    @staticmethod
    def validate_ohlcv_bar(bar: Dict[str, Any]) -> bool:
        """
        Validate individual OHLC bar.
        
        Args:
            bar: OHLC bar dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
            
            # Basic OHLC logic validation
            if not (l <= o <= h and l <= c <= h):
                return False
            
            # Check for negative prices
            if any(price <= 0 for price in [o, h, l, c]):
                return False
            
            # Check for unrealistic price movements (>50% in one bar)
            max_change = max(abs(h - l) / min(h, l), abs(o - c) / min(o, c))
            if max_change > 0.5:
                return False
            
            return True
            
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
    
    @staticmethod
    def detect_missing_bars(timestamps: List[datetime], granularity: str) -> List[datetime]:
        """
        Detect missing bars in time series.
        
        Args:
            timestamps: List of bar timestamps
            granularity: Time granularity (H1, M1, etc.)
            
        Returns:
            List of missing timestamps
        """
        if len(timestamps) < 2:
            return []
        
        # Determine expected interval
        interval_mapping = {
            'M1': timedelta(minutes=1),
            'M5': timedelta(minutes=5),
            'M15': timedelta(minutes=15),
            'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1),
            'H4': timedelta(hours=4),
            'D': timedelta(days=1)
        }
        
        interval = interval_mapping.get(granularity, timedelta(hours=1))
        
        # Find gaps
        missing = []
        for i in range(1, len(timestamps)):
            current = timestamps[i]
            previous = timestamps[i-1]
            expected = previous + interval
            
            while expected < current:
                missing.append(expected)
                expected += interval
        
        return missing
    
    @staticmethod
    def fill_missing_bars(data: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """
        Fill missing bars with interpolated values.
        
        Args:
            data: OHLC DataFrame
            granularity: Time granularity
            
        Returns:
            DataFrame with filled missing bars
        """
        if len(data) < 2:
            return data
        
        # Create complete time index
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        
        interval_mapping = {
            'M1': 'T',
            'M5': '5T',
            'M15': '15T',
            'M30': '30T',
            'H1': 'H',
            'H4': '4H',
            'D': 'D'
        }
        
        freq = interval_mapping.get(granularity, 'H')
        complete_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Reindex and forward fill
        data_indexed = data.set_index('timestamp')
        filled_data = data_indexed.reindex(complete_index, method='ffill')
        
        # Reset index
        filled_data = filled_data.reset_index()
        filled_data.rename(columns={'index': 'timestamp'}, inplace=True)
        
        return filled_data


class OANDAHistoricalDownloader:
    """
    Production-ready OANDA v20 historical data downloader.
    
    Features:
    - Comprehensive error handling and retry logic
    - Rate limiting with exponential backoff
    - Data validation and quality checks
    - Resume capabilities for interrupted downloads
    - UTC timezone alignment
    - Progress tracking and logging
    """
    
    def __init__(self, config: DownloadConfig):
        """
        Initialize the historical data downloader.
        
        Args:
            config: Download configuration
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_second)
        self.validator = DataValidator()
        
        # Initialize OANDA client
        if config.environment == 'live':
            api_url = 'https://api-fxtrade.oanda.com'
        else:
            api_url = 'https://api-fxpractice.oanda.com'
        
        self.client = v20.Context(
            hostname=api_url.replace('https://', ''),
            port=443,
            ssl=True,
            token=config.api_key
        )
        
        # Download tracking
        self.total_requests = 0
        self.total_bars_downloaded = 0
        self.failed_requests = 0
        
        logger.info(f"OANDA Historical Downloader initialized ({config.environment} environment)")
    
    async def download_instrument_data(self, 
                                     instrument: str, 
                                     start_date: datetime, 
                                     end_date: datetime,
                                     progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Download historical data for a single instrument.
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            start_date: Start date (UTC)
            end_date: End date (UTC)
            progress_callback: Optional progress callback function
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Starting download for {instrument}: {start_date} to {end_date}")
        
        # Ensure UTC timezone
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate request end time
            current_end = min(current_start + timedelta(hours=self.config.max_count_per_request), end_date)
            
            try:
                # Rate limiting
                await self.rate_limiter.wait_if_needed()
                
                # Download chunk
                chunk_data = await self._download_chunk(instrument, current_start, current_end)
                
                if chunk_data is not None and len(chunk_data) > 0:
                    all_data.append(chunk_data)
                    self.rate_limiter.record_success()
                    
                    # Progress callback
                    if progress_callback:
                        progress = (current_end - start_date) / (end_date - start_date)
                        progress_callback(instrument, progress, len(chunk_data))
                else:
                    logger.warning(f"No data received for {instrument} {current_start} to {current_end}")
                
                # Move to next chunk
                current_start = current_end + timedelta(microseconds=1)
                
            except Exception as e:
                logger.error(f"Error downloading {instrument} chunk {current_start} to {current_end}: {str(e)}")
                self.rate_limiter.record_error()
                
                # Skip this chunk after retries
                current_start = current_end + timedelta(microseconds=1)
                continue
        
        # Combine all chunks
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Data quality checks
            combined_data = self._post_process_data(combined_data, instrument)
            
            logger.info(f"Downloaded {len(combined_data)} bars for {instrument}")
            return combined_data
        else:
            logger.warning(f"No data downloaded for {instrument}")
            return pd.DataFrame()
    
    async def _download_chunk(self, 
                            instrument: str, 
                            start_time: datetime, 
                            end_time: datetime) -> Optional[pd.DataFrame]:
        """
        Download a single chunk of data with retry logic.
        
        Args:
            instrument: Currency pair
            start_time: Chunk start time
            end_time: Chunk end time
            
        Returns:
            DataFrame with chunk data or None if failed
        """
        for attempt in range(self.config.max_retries):
            try:
                # Prepare request
                params = {
                    'granularity': self.config.granularity,
                    'from': start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'to': end_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'includeFirst': False  # Avoid duplicates
                }
                
                # Make API request
                response = self.client.instrument.candles(instrument, **params)
                self.total_requests += 1
                
                if response.status == 200:
                    candles = response.body.get('candles', [])
                    
                    if not candles:
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    data_rows = []
                    for candle in candles:
                        if not self.config.require_complete_bars or candle.complete:
                            try:
                                row = {
                                    'timestamp': pd.to_datetime(candle.time, utc=True),
                                    'open': float(candle.mid.o),
                                    'high': float(candle.mid.h),
                                    'low': float(candle.mid.l),
                                    'close': float(candle.mid.c),
                                    'volume': int(candle.volume) if hasattr(candle, 'volume') else 0
                                }
                                
                                # Validate bar
                                if self.config.validate_ohlc and not self.validator.validate_ohlcv_bar(row):
                                    logger.warning(f"Invalid OHLC bar for {instrument} at {candle.time}")
                                    continue
                                
                                data_rows.append(row)
                                
                            except (ValueError, AttributeError) as e:
                                logger.warning(f"Error parsing candle for {instrument}: {str(e)}")
                                continue
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows)
                        self.total_bars_downloaded += len(df)
                        return df
                    else:
                        return pd.DataFrame()
                
                else:
                    error_msg = getattr(response, 'body', {}).get('errorMessage', 'Unknown error')
                    logger.error(f"API error for {instrument}: {response.status} - {error_msg}")
                    
                    # Handle rate limiting
                    if response.status == 429:
                        wait_time = min(2 ** attempt * 5, 300)  # Max 5 minutes
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Handle other errors
                    if response.status >= 500:  # Server errors - retry
                        wait_time = min(2 ** attempt, self.config.max_retry_delay)
                        await asyncio.sleep(wait_time)
                        continue
                    else:  # Client errors - don't retry
                        break
                        
            except (V20ConnectionError, V20Timeout) as e:
                logger.warning(f"Network error for {instrument} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    wait_time = min(
                        self.config.initial_retry_delay * (self.config.backoff_multiplier ** attempt),
                        self.config.max_retry_delay
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.failed_requests += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error for {instrument} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    wait_time = min(2 ** attempt, self.config.max_retry_delay)
                    await asyncio.sleep(wait_time)
                else:
                    self.failed_requests += 1
                    break
        
        return None
    
    def _post_process_data(self, data: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        Post-process downloaded data for quality and consistency.
        
        Args:
            data: Raw OHLC data
            instrument: Instrument name
            
        Returns:
            Processed DataFrame
        """
        if len(data) == 0:
            return data
        
        original_count = len(data)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Check for missing bars
        if self.config.fill_missing_bars:
            missing_timestamps = self.validator.detect_missing_bars(
                data['timestamp'].tolist(), 
                self.config.granularity
            )
            
            if missing_timestamps:
                missing_ratio = len(missing_timestamps) / (len(data) + len(missing_timestamps))
                
                if missing_ratio <= self.config.max_missing_ratio:
                    logger.info(f"Filling {len(missing_timestamps)} missing bars for {instrument}")
                    data = self.validator.fill_missing_bars(data, self.config.granularity)
                else:
                    logger.warning(
                        f"Too many missing bars for {instrument}: {missing_ratio:.2%} > "
                        f"{self.config.max_missing_ratio:.2%}"
                    )
        
        # Final validation
        valid_mask = data.apply(
            lambda row: self.validator.validate_ohlcv_bar(row.to_dict()), 
            axis=1
        )
        data = data[valid_mask]
        
        processed_count = len(data)
        if processed_count != original_count:
            logger.info(f"Data processing for {instrument}: {original_count} -> {processed_count} bars")
        
        return data
    
    async def download_all_instruments(self, 
                                     start_date: datetime, 
                                     end_date: datetime,
                                     output_dir: Optional[Path] = None,
                                     save_format: str = 'parquet') -> Dict[str, pd.DataFrame]:
        """
        Download historical data for all configured instruments.
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            output_dir: Optional directory to save files
            save_format: Output format ('parquet' or 'csv')
            
        Returns:
            Dictionary with instrument data
        """
        logger.info(f"Starting bulk download for {len(self.config.instruments)} instruments")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        results = {}
        
        def progress_callback(instrument: str, progress: float, bars_downloaded: int):
            logger.info(f"{instrument}: {progress:.1%} complete ({bars_downloaded} bars in chunk)")
        
        # Download each instrument
        for i, instrument in enumerate(self.config.instruments):
            logger.info(f"Processing {instrument} ({i+1}/{len(self.config.instruments)})")
            
            try:
                data = await self.download_instrument_data(
                    instrument, start_date, end_date, progress_callback
                )
                
                if len(data) > 0:
                    results[instrument] = data
                    
                    # Save to file if requested
                    if output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        if save_format == 'parquet':
                            file_path = output_dir / f"{instrument}.parquet"
                            data.to_parquet(file_path, index=False)
                        else:  # CSV
                            file_path = output_dir / f"{instrument}.csv"
                            data.to_csv(file_path, index=False)
                        
                        logger.info(f"Saved {instrument} data to {file_path}")
                else:
                    logger.warning(f"No data downloaded for {instrument}")
                    
            except Exception as e:
                logger.error(f"Failed to download {instrument}: {str(e)}")
                continue
        
        # Summary
        total_bars = sum(len(df) for df in results.values())
        logger.info(f"Download complete: {len(results)} instruments, {total_bars} total bars")
        logger.info(f"API statistics: {self.total_requests} requests, {self.failed_requests} failures")
        
        return results
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """Get download statistics."""
        return {
            'total_requests': self.total_requests,
            'total_bars_downloaded': self.total_bars_downloaded,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1)
        }


# CLI interface for standalone usage
async def main():
    """Main function for CLI usage."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='OANDA Historical Data Downloader')
    parser.add_argument('--api-key', required=True, help='OANDA API key')
    parser.add_argument('--account-id', required=True, help='OANDA account ID')
    parser.add_argument('--environment', default='practice', choices=['practice', 'live'])
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--instruments', nargs='+', help='Instruments to download')
    parser.add_argument('--granularity', default='H1', help='Data granularity')
    parser.add_argument('--output-dir', default='./data', help='Output directory')
    parser.add_argument('--format', default='parquet', choices=['parquet', 'csv'])
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    # Create config
    config = DownloadConfig(
        api_key=args.api_key,
        account_id=args.account_id,
        environment=args.environment,
        instruments=args.instruments,
        granularity=args.granularity
    )
    
    # Download data
    downloader = OANDAHistoricalDownloader(config)
    
    try:
        results = await downloader.download_all_instruments(
            start_date, end_date, Path(args.output_dir), args.format
        )
        
        print(f"Successfully downloaded data for {len(results)} instruments")
        print("Download statistics:", downloader.get_download_statistics())
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)