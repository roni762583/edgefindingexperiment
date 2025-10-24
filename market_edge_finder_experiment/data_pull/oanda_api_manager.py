#!/usr/bin/env python3
"""
OANDA v20 API Manager with Rate Limiting and Error Handling

Production-ready API manager for OANDA v20 with comprehensive rate limiting,
quota management, and robust error handling.

Uses ONLY the official OANDA v20 Python library.
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import threading
from dataclasses import dataclass, field
import v20
from v20.context import Context
from v20.request import Request
import v20.response

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for OANDA API rate limits."""
    
    # OANDA v20 API Rate Limits (conservative estimates)
    requests_per_second: int = 10  # Conservative limit
    requests_per_minute: int = 600  # 10 req/sec * 60 sec
    requests_per_hour: int = 36000  # 600 req/min * 60 min
    
    # Burst handling
    burst_capacity: int = 20  # Allow brief bursts
    burst_refill_rate: float = 1.0  # Tokens per second
    
    # Backoff configuration
    initial_backoff: float = 1.0  # Initial backoff in seconds
    max_backoff: float = 300.0  # Maximum backoff (5 minutes)
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    
    # Request timeouts
    request_timeout: float = 30.0  # Request timeout in seconds
    connection_timeout: float = 10.0  # Connection timeout


@dataclass
class APIQuotaTracker:
    """Track API usage quotas and limits."""
    
    # Request tracking windows
    second_requests: deque = field(default_factory=lambda: deque(maxlen=100))
    minute_requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    hour_requests: deque = field(default_factory=lambda: deque(maxlen=50000))
    
    # Error tracking
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    
    # Rate limit status
    rate_limited_until: Optional[datetime] = None
    last_reset_time: datetime = field(default_factory=datetime.now)


class OANDAAPIManager:
    """
    Production-ready OANDA v20 API manager with comprehensive rate limiting.
    
    Features:
    - Official v20 library integration
    - Multi-window rate limiting (second/minute/hour)
    - Exponential backoff with jitter
    - API quota tracking and management
    - Comprehensive error handling
    - Request retry logic with circuit breaker
    """
    
    def __init__(self, 
                 api_key: str,
                 account_id: str,
                 environment: str = "practice",
                 rate_config: Optional[RateLimitConfig] = None):
        """
        Initialize OANDA API manager.
        
        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            environment: 'practice' or 'live'
            rate_config: Rate limiting configuration
        """
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        self.rate_config = rate_config or RateLimitConfig()
        
        # Initialize v20 context
        if environment == "live":
            hostname = "api-fxtrade.oanda.com"
            stream_hostname = "stream-fxtrade.oanda.com"
        else:
            hostname = "api-fxpractice.oanda.com"
            stream_hostname = "stream-fxpractice.oanda.com"
            
        self.ctx = Context(
            hostname=hostname,
            port=443,
            ssl=True,
            application="MarketEdgeFinder",
            token=api_key,
            datetime_format="RFC3339"
        )
        
        # Rate limiting components
        self.quota_tracker = APIQuotaTracker()
        self._lock = threading.Lock()
        self._token_bucket = self.rate_config.burst_capacity
        self._last_refill = time.time()
        
        logger.info(f"🔌 OANDA v20 API Manager initialized ({environment} environment)")
    
    def _refill_token_bucket(self) -> None:
        """Refill the token bucket for burst capacity management."""
        now = time.time()
        time_passed = now - self._last_refill
        tokens_to_add = time_passed * self.rate_config.burst_refill_rate
        
        self._token_bucket = min(
            self.rate_config.burst_capacity,
            self._token_bucket + tokens_to_add
        )
        self._last_refill = now
    
    def _can_make_request(self) -> Tuple[bool, Optional[float]]:
        """
        Check if we can make a request based on rate limits.
        
        Returns:
            Tuple of (can_make_request, wait_time_seconds)
        """
        now = datetime.now()
        current_time = time.time()
        
        with self._lock:
            # Check if we're in a rate-limited state
            if (self.quota_tracker.rate_limited_until and 
                now < self.quota_tracker.rate_limited_until):
                wait_time = (self.quota_tracker.rate_limited_until - now).total_seconds()
                return False, wait_time
            
            # Refill token bucket
            self._refill_token_bucket()
            
            # Check token bucket (burst capacity)
            if self._token_bucket < 1:
                return False, 1.0 / self.rate_config.burst_refill_rate
            
            # Clean old requests from tracking windows
            cutoff_second = current_time - 1.0
            cutoff_minute = current_time - 60.0
            cutoff_hour = current_time - 3600.0
            
            # Remove old entries
            while (self.quota_tracker.second_requests and 
                   self.quota_tracker.second_requests[0] < cutoff_second):
                self.quota_tracker.second_requests.popleft()
                
            while (self.quota_tracker.minute_requests and 
                   self.quota_tracker.minute_requests[0] < cutoff_minute):
                self.quota_tracker.minute_requests.popleft()
                
            while (self.quota_tracker.hour_requests and 
                   self.quota_tracker.hour_requests[0] < cutoff_hour):
                self.quota_tracker.hour_requests.popleft()
            
            # Check rate limits
            if len(self.quota_tracker.second_requests) >= self.rate_config.requests_per_second:
                return False, 1.0
                
            if len(self.quota_tracker.minute_requests) >= self.rate_config.requests_per_minute:
                return False, 60.0
                
            if len(self.quota_tracker.hour_requests) >= self.rate_config.requests_per_hour:
                return False, 3600.0
            
            return True, None
    
    def _record_request(self) -> None:
        """Record a successful request in tracking windows."""
        current_time = time.time()
        
        with self._lock:
            # Consume token from bucket
            self._token_bucket -= 1
            
            # Record request in all windows
            self.quota_tracker.second_requests.append(current_time)
            self.quota_tracker.minute_requests.append(current_time)
            self.quota_tracker.hour_requests.append(current_time)
            
            # Reset consecutive error count on successful request
            self.quota_tracker.consecutive_errors = 0
    
    def _record_error(self, error: Exception) -> None:
        """Record an API error for tracking and backoff calculations."""
        with self._lock:
            self.quota_tracker.consecutive_errors += 1
            self.quota_tracker.last_error_time = datetime.now()
            
            # Implement circuit breaker logic
            if self.quota_tracker.consecutive_errors >= 5:
                # Rate limit for increasing duration based on consecutive errors
                backoff_duration = min(
                    self.rate_config.initial_backoff * 
                    (self.rate_config.backoff_multiplier ** (self.quota_tracker.consecutive_errors - 5)),
                    self.rate_config.max_backoff
                )
                
                self.quota_tracker.rate_limited_until = (
                    datetime.now() + timedelta(seconds=backoff_duration)
                )
                
                logger.warning(
                    f"⚠️ Circuit breaker activated: {self.quota_tracker.consecutive_errors} "
                    f"consecutive errors, backing off for {backoff_duration:.1f}s"
                )
    
    async def _execute_request_with_retry(self, request_func, *args, **kwargs) -> Any:
        """
        Execute a request with retry logic and rate limiting.
        
        Args:
            request_func: Function to execute the request
            *args, **kwargs: Arguments for the request function
            
        Returns:
            Response from the API
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Check rate limits
                can_make_request, wait_time = self._can_make_request()
                
                if not can_make_request:
                    if wait_time:
                        logger.info(f"⏳ Rate limit reached, waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Execute request
                response = await asyncio.get_event_loop().run_in_executor(
                    None, request_func, *args, **kwargs
                )
                
                # Check for API errors in response
                if hasattr(response, 'status') and response.status != 200:
                    raise Exception(f"API error: {response.status} - {getattr(response, 'body', 'Unknown error')}")
                
                # Record successful request
                self._record_request()
                
                return response
                
            except Exception as e:
                self._record_error(e)
                retry_count += 1
                
                if retry_count > max_retries:
                    logger.error(f"❌ Request failed after {max_retries} retries: {str(e)}")
                    raise
                
                # Calculate backoff with jitter
                backoff_time = (
                    self.rate_config.initial_backoff * 
                    (self.rate_config.backoff_multiplier ** (retry_count - 1))
                )
                
                # Add jitter (±25%)
                import random
                jitter = backoff_time * 0.25 * (2 * random.random() - 1)
                backoff_time += jitter
                
                logger.warning(
                    f"⚠️ Request failed (attempt {retry_count}/{max_retries}), "
                    f"retrying in {backoff_time:.1f}s: {str(e)}"
                )
                
                await asyncio.sleep(backoff_time)
    
    async def get_candles(self, 
                         instrument: str,
                         granularity: str = "H1",
                         count: Optional[int] = None,
                         from_time: Optional[str] = None,
                         to_time: Optional[str] = None) -> List[Dict]:
        """
        Get historical candles with rate limiting and error handling.
        
        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Candle timeframe (M1, M5, H1, etc.)
            count: Number of candles (max 5000)
            from_time: Start time (RFC3339 format)
            to_time: End time (RFC3339 format)
            
        Returns:
            List of candle dictionaries
        """
        # Prepare request parameters
        kwargs = {
            "instrument": instrument,
            "granularity": granularity,
            "price": "M"  # Mid prices
        }
        
        if count:
            kwargs["count"] = min(count, 5000)  # OANDA limit
        if from_time:
            kwargs["from"] = from_time
        if to_time:
            kwargs["to"] = to_time
        
        def make_request():
            return self.ctx.instrument.candles(**kwargs)
        
        try:
            response = await self._execute_request_with_retry(make_request)
            
            if hasattr(response, 'body') and 'candles' in response.body:
                candles = []
                for candle in response.body['candles']:
                    if candle.get('complete', True):
                        candles.append({
                            'time': candle['time'],
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle.get('volume', 0))
                        })
                
                logger.info(f"📊 Retrieved {len(candles)} candles for {instrument}")
                return candles
            else:
                logger.error(f"❌ Invalid response format for {instrument}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get candles for {instrument}: {str(e)}")
            return []
    
    async def get_current_prices(self, instruments: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple instruments.
        
        Args:
            instruments: List of currency pairs
            
        Returns:
            Dictionary mapping instruments to price data
        """
        def make_request():
            return self.ctx.pricing.get(
                accountID=self.account_id,
                instruments=",".join(instruments)
            )
        
        try:
            response = await self._execute_request_with_retry(make_request)
            
            if hasattr(response, 'body') and 'prices' in response.body:
                prices = {}
                for price_data in response.body['prices']:
                    instrument = price_data['instrument']
                    
                    if 'bids' in price_data and 'asks' in price_data:
                        bid = float(price_data['bids'][0]['price'])
                        ask = float(price_data['asks'][0]['price'])
                        
                        prices[instrument] = {
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2,
                            'spread': ask - bid,
                            'time': price_data.get('time', datetime.now().isoformat())
                        }
                
                return prices
            else:
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get current prices: {str(e)}")
            return {}
    
    def get_quota_status(self) -> Dict[str, Any]:
        """
        Get current API quota usage status.
        
        Returns:
            Dictionary with quota information
        """
        now = datetime.now()
        current_time = time.time()
        
        with self._lock:
            # Clean old requests
            cutoff_second = current_time - 1.0
            cutoff_minute = current_time - 60.0
            cutoff_hour = current_time - 3600.0
            
            while (self.quota_tracker.second_requests and 
                   self.quota_tracker.second_requests[0] < cutoff_second):
                self.quota_tracker.second_requests.popleft()
                
            while (self.quota_tracker.minute_requests and 
                   self.quota_tracker.minute_requests[0] < cutoff_minute):
                self.quota_tracker.minute_requests.popleft()
                
            while (self.quota_tracker.hour_requests and 
                   self.quota_tracker.hour_requests[0] < cutoff_hour):
                self.quota_tracker.hour_requests.popleft()
            
            return {
                'requests_last_second': len(self.quota_tracker.second_requests),
                'requests_last_minute': len(self.quota_tracker.minute_requests),
                'requests_last_hour': len(self.quota_tracker.hour_requests),
                'consecutive_errors': self.quota_tracker.consecutive_errors,
                'rate_limited_until': self.quota_tracker.rate_limited_until,
                'token_bucket_level': self._token_bucket,
                'limits': {
                    'requests_per_second': self.rate_config.requests_per_second,
                    'requests_per_minute': self.rate_config.requests_per_minute,
                    'requests_per_hour': self.rate_config.requests_per_hour
                }
            }


# Example usage and testing
async def test_api_manager():
    """Test the API manager functionality."""
    import os
    
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    if not api_key or not account_id:
        logger.error("Please set OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables")
        return
    
    # Initialize API manager
    manager = OANDAAPIManager(
        api_key=api_key,
        account_id=account_id,
        environment="practice"
    )
    
    # Test candle retrieval
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
    
    for instrument in instruments:
        candles = await manager.get_candles(
            instrument=instrument,
            granularity="H1",
            count=100
        )
        
        logger.info(f"Retrieved {len(candles)} candles for {instrument}")
        
        # Test rate limiting by making requests in rapid succession
        await asyncio.sleep(0.1)
    
    # Test current prices
    prices = await manager.get_current_prices(instruments)
    logger.info(f"Current prices: {prices}")
    
    # Check quota status
    status = manager.get_quota_status()
    logger.info(f"Quota status: {status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_api_manager())