"""
Real-time data pipeline for inference.

This module handles real-time data ingestion from OANDA API, data validation,
feature preprocessing, and delivery to the prediction engine.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import aiohttp
import time
from queue import Queue, Empty
import threading

# OANDA v20 API
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for real-time data pipeline."""
    
    # OANDA API configuration
    api_key: str
    account_id: str
    environment: str = 'practice'  # 'practice' or 'live'
    
    # Data parameters
    instruments: List[str] = None
    granularity: str = 'H1'  # Hourly data
    count: int = 5000  # Number of historical candles to fetch
    
    # Real-time settings
    update_interval_seconds: int = 60  # How often to fetch new data
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Data validation
    min_data_points: int = 100
    max_missing_ratio: float = 0.05  # Maximum 5% missing data
    
    # Buffering
    buffer_size_hours: int = 48  # Keep 48 hours of data in memory
    
    def __post_init__(self):
        if self.instruments is None:
            self.instruments = [
                'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
                'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
                'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD',
                'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD'
            ]


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class APIConnectionError(Exception):
    """Exception raised when API connection fails."""
    pass


class OANDADataProvider:
    """
    OANDA v20 API data provider for real-time FX data.
    
    Handles connection management, data fetching, and error recovery
    with the official OANDA v20 Python library.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the OANDA data provider.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.client = self._initialize_client()
        self.last_fetch_time: Dict[str, datetime] = {}
        
        logger.info(f"OANDA data provider initialized for {len(config.instruments)} instruments")
    
    def _initialize_client(self) -> oandapyV20.API:
        """Initialize OANDA API client."""
        try:
            client = oandapyV20.API(
                access_token=self.config.api_key,
                environment=self.config.environment
            )
            return client
        except Exception as e:
            raise APIConnectionError(f"Failed to initialize OANDA client: {str(e)}")
    
    def _validate_response_data(self, data: List[Dict], instrument: str) -> None:
        """
        Validate API response data.
        
        Args:
            data: Candle data from API
            instrument: Instrument name
            
        Raises:
            DataValidationError: If data validation fails
        """
        if not data:
            raise DataValidationError(f"No data received for {instrument}")
        
        if len(data) < self.config.min_data_points:
            raise DataValidationError(
                f"Insufficient data for {instrument}: {len(data)} < {self.config.min_data_points}"
            )
        
        # Check for missing data
        missing_count = sum(1 for candle in data if not candle.get('complete', True))
        missing_ratio = missing_count / len(data)
        
        if missing_ratio > self.config.max_missing_ratio:
            raise DataValidationError(
                f"Too much missing data for {instrument}: {missing_ratio:.2%} > {self.config.max_missing_ratio:.2%}"
            )
    
    def fetch_historical_data(self, instrument: str, from_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical data for an instrument.
        
        Args:
            instrument: Instrument name (e.g., 'EUR_USD')
            from_time: Optional start time for data
            
        Returns:
            DataFrame with OHLC data
            
        Raises:
            APIConnectionError: If API request fails
            DataValidationError: If data validation fails
        """
        try:
            # Prepare request parameters
            params = {
                'granularity': self.config.granularity,
                'count': self.config.count
            }
            
            if from_time:
                params['from'] = from_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # Create and execute request
            request = instruments.InstrumentsCandles(instrument=instrument, params=params)
            
            for attempt in range(self.config.max_retries):
                try:
                    response = self.client.request(request)
                    break
                except V20Error as e:
                    if attempt == self.config.max_retries - 1:
                        raise APIConnectionError(f"OANDA API error for {instrument}: {str(e)}")
                    logger.warning(f"API retry {attempt + 1} for {instrument}: {str(e)}")
                    time.sleep(self.config.retry_delay_seconds)
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise APIConnectionError(f"API connection error for {instrument}: {str(e)}")
                    time.sleep(self.config.retry_delay_seconds)
            
            # Extract candle data
            candles = response.get('candles', [])
            self._validate_response_data(candles, instrument)
            
            # Convert to DataFrame
            data_rows = []
            for candle in candles:
                if candle.get('complete', True):  # Only use complete candles
                    mid = candle['mid']
                    data_rows.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'open': float(mid['o']),
                        'high': float(mid['h']),
                        'low': float(mid['l']),
                        'close': float(mid['c']),
                        'volume': int(candle.get('volume', 0))
                    })
            
            df = pd.DataFrame(data_rows)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Update last fetch time
            self.last_fetch_time[instrument] = datetime.now()
            
            logger.debug(f"Fetched {len(df)} candles for {instrument}")
            return df
            
        except (APIConnectionError, DataValidationError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Unexpected error fetching data for {instrument}: {str(e)}")
    
    def fetch_latest_data(self, instrument: str, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Fetch latest data for an instrument.
        
        Args:
            instrument: Instrument name
            lookback_hours: How many hours of recent data to fetch
            
        Returns:
            DataFrame with recent OHLC data
        """
        from_time = datetime.now() - timedelta(hours=lookback_hours)
        return self.fetch_historical_data(instrument, from_time)
    
    def get_connection_health(self) -> Dict[str, Any]:
        """
        Check API connection health.
        
        Returns:
            Dictionary with connection status information
        """
        health_status = {
            'api_accessible': False,
            'last_successful_fetch': None,
            'instruments_status': {}
        }
        
        try:
            # Test connection with a simple request
            test_instrument = self.config.instruments[0]
            params = {'granularity': 'H1', 'count': 1}
            request = instruments.InstrumentsCandles(instrument=test_instrument, params=params)
            
            response = self.client.request(request)
            health_status['api_accessible'] = True
            
            # Check last fetch times
            if self.last_fetch_time:
                latest_fetch = max(self.last_fetch_time.values())
                health_status['last_successful_fetch'] = latest_fetch.isoformat()
            
            # Check individual instrument status
            for instrument in self.config.instruments:
                last_fetch = self.last_fetch_time.get(instrument)
                health_status['instruments_status'][instrument] = {
                    'last_fetch': last_fetch.isoformat() if last_fetch else None,
                    'minutes_since_fetch': (datetime.now() - last_fetch).total_seconds() / 60 if last_fetch else None
                }
            
        except Exception as e:
            health_status['error'] = str(e)
            logger.error(f"API health check failed: {str(e)}")
        
        return health_status


class DataBuffer:
    """
    In-memory buffer for market data with automatic cleanup.
    
    Maintains a rolling buffer of market data for each instrument,
    automatically removing old data to prevent memory issues.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data buffer.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.data: Dict[str, pd.DataFrame] = {}
        self.lock = threading.Lock()
        
        # Calculate buffer size in number of rows
        hours_per_day = 24
        self.max_rows = self.config.buffer_size_hours
        
        logger.info(f"Data buffer initialized with {self.config.buffer_size_hours} hour capacity")
    
    def add_data(self, instrument: str, new_data: pd.DataFrame) -> None:
        """
        Add new data to the buffer.
        
        Args:
            instrument: Instrument name
            new_data: New market data to add
        """
        with self.lock:
            if instrument not in self.data:
                self.data[instrument] = new_data.copy()
            else:
                # Append new data
                combined = pd.concat([self.data[instrument], new_data], ignore_index=True)
                
                # Remove duplicates based on timestamp
                combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                combined = combined.sort_values('timestamp').reset_index(drop=True)
                
                # Trim to buffer size
                if len(combined) > self.max_rows:
                    combined = combined.tail(self.max_rows).reset_index(drop=True)
                
                self.data[instrument] = combined
            
            logger.debug(f"Buffer updated for {instrument}: {len(self.data[instrument])} rows")
    
    def get_data(self, instrument: str, lookback_hours: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get data from buffer for an instrument.
        
        Args:
            instrument: Instrument name
            lookback_hours: How many hours of data to return
            
        Returns:
            DataFrame with requested data, or None if not available
        """
        with self.lock:
            if instrument not in self.data:
                return None
            
            data = self.data[instrument].copy()
            
            if lookback_hours and len(data) > 0:
                # Filter to last N hours
                cutoff_time = data['timestamp'].iloc[-1] - timedelta(hours=lookback_hours)
                data = data[data['timestamp'] >= cutoff_time].reset_index(drop=True)
            
            return data if len(data) > 0 else None
    
    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get all buffered data.
        
        Returns:
            Dictionary with all instrument data
        """
        with self.lock:
            return {instrument: df.copy() for instrument, df in self.data.items()}
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Get buffer status information.
        
        Returns:
            Dictionary with buffer status
        """
        with self.lock:
            status = {
                'total_instruments': len(self.data),
                'buffer_capacity_hours': self.config.buffer_size_hours,
                'instruments': {}
            }
            
            for instrument, df in self.data.items():
                if len(df) > 0:
                    oldest_time = df['timestamp'].iloc[0]
                    newest_time = df['timestamp'].iloc[-1]
                    hours_span = (newest_time - oldest_time).total_seconds() / 3600
                    
                    status['instruments'][instrument] = {
                        'rows': len(df),
                        'oldest_timestamp': oldest_time.isoformat(),
                        'newest_timestamp': newest_time.isoformat(),
                        'hours_span': hours_span
                    }
                else:
                    status['instruments'][instrument] = {
                        'rows': 0,
                        'hours_span': 0
                    }
            
            return status


class RealtimeDataPipeline:
    """
    Complete real-time data pipeline for inference.
    
    Orchestrates data fetching, validation, buffering, and delivery
    to prediction engines with automatic error recovery.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the real-time data pipeline.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.data_provider = OANDADataProvider(config)
        self.data_buffer = DataBuffer(config)
        
        # Pipeline state
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.last_update_time: Optional[datetime] = None
        
        # Callbacks
        self.data_callbacks: List[Callable[[Dict[str, pd.DataFrame]], None]] = []
        
        logger.info("Real-time data pipeline initialized")
    
    def add_data_callback(self, callback: Callable[[Dict[str, pd.DataFrame]], None]) -> None:
        """
        Add callback function to be called when new data arrives.
        
        Args:
            callback: Function to call with new data
        """
        self.data_callbacks.append(callback)
        logger.info(f"Added data callback: {callback.__name__}")
    
    def start(self) -> None:
        """Start the real-time data pipeline."""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting real-time data pipeline...")
        
        # Initialize with historical data
        try:
            self._initial_data_load()
            logger.info("Initial data load completed")
        except Exception as e:
            logger.error(f"Initial data load failed: {str(e)}")
            raise
        
        # Start real-time updates
        self.running = True
        self.worker_thread = threading.Thread(target=self._update_worker)
        self.worker_thread.start()
        
        logger.info("Real-time data pipeline started successfully")
    
    def stop(self) -> None:
        """Stop the real-time data pipeline."""
        logger.info("Stopping real-time data pipeline...")
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        
        logger.info("Real-time data pipeline stopped")
    
    def _initial_data_load(self) -> None:
        """Load initial historical data for all instruments."""
        logger.info(f"Loading initial data for {len(self.config.instruments)} instruments...")
        
        for instrument in self.config.instruments:
            try:
                logger.debug(f"Fetching initial data for {instrument}")
                data = self.data_provider.fetch_historical_data(instrument)
                self.data_buffer.add_data(instrument, data)
                
                # Small delay to avoid hitting API rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to load initial data for {instrument}: {str(e)}")
                # Continue with other instruments
    
    def _update_worker(self) -> None:
        """Background worker for real-time data updates."""
        logger.info("Data update worker started")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Fetch latest data for all instruments
                updated_data = {}
                
                for instrument in self.config.instruments:
                    try:
                        # Fetch recent data (last few hours to catch any gaps)
                        latest_data = self.data_provider.fetch_latest_data(instrument, lookback_hours=6)
                        self.data_buffer.add_data(instrument, latest_data)
                        updated_data[instrument] = latest_data
                        
                        # Small delay between instruments
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Failed to update data for {instrument}: {str(e)}")
                        continue
                
                # Notify callbacks with updated data
                if updated_data and self.data_callbacks:
                    all_buffered_data = self.data_buffer.get_all_data()
                    for callback in self.data_callbacks:
                        try:
                            callback(all_buffered_data)
                        except Exception as e:
                            logger.error(f"Data callback error: {str(e)}")
                
                self.last_update_time = datetime.now()
                
                # Calculate how long to wait before next update
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.config.update_interval_seconds - elapsed_time)
                
                logger.debug(f"Data update completed in {elapsed_time:.1f}s, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in data update worker: {str(e)}")
                time.sleep(self.config.update_interval_seconds)
        
        logger.info("Data update worker stopped")
    
    def get_current_data(self, instrument: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get current buffered data.
        
        Args:
            instrument: Specific instrument to get data for, or None for all
            
        Returns:
            DataFrame for specific instrument or dict of all data
        """
        if instrument:
            return self.data_buffer.get_data(instrument)
        else:
            return self.data_buffer.get_all_data()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        status = {
            'running': self.running,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'update_interval_seconds': self.config.update_interval_seconds,
            'callback_count': len(self.data_callbacks),
            'api_health': self.data_provider.get_connection_health(),
            'buffer_status': self.data_buffer.get_buffer_status()
        }
        
        return status
    
    def force_update(self) -> Dict[str, pd.DataFrame]:
        """
        Force an immediate data update.
        
        Returns:
            Dictionary with updated data
        """
        logger.info("Forcing immediate data update...")
        
        updated_data = {}
        for instrument in self.config.instruments:
            try:
                latest_data = self.data_provider.fetch_latest_data(instrument, lookback_hours=3)
                self.data_buffer.add_data(instrument, latest_data)
                updated_data[instrument] = latest_data
            except Exception as e:
                logger.error(f"Force update failed for {instrument}: {str(e)}")
        
        self.last_update_time = datetime.now()
        logger.info(f"Force update completed for {len(updated_data)} instruments")
        
        return updated_data


if __name__ == "__main__":
    # Example usage
    logger.info("Market Edge Finder Real-time Data Pipeline")
    logger.info("This module provides real-time data ingestion from OANDA API for inference.")