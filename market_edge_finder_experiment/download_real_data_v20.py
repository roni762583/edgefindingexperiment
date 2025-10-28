#!/usr/bin/env python3
"""
Official OANDA v20 Data Download Script

Downloads 3 years of hourly data for 24 FX pairs using the official OANDA v20 API.
Includes the original 20 instruments plus 4 additional pairs for complete coverage.
Follows OANDA v20 documentation exactly - no shortcuts.
"""

import os
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import v20
from v20 import Context

def check_credentials():
    """Check OANDA credentials are properly configured."""
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    environment = os.getenv("OANDA_ENVIRONMENT", "live")
    
    if not api_key or api_key == "your_key_here":
        print("❌ OANDA_API_KEY not configured")
        return False
    
    if not account_id or account_id == "your_account_here":
        print("❌ OANDA_ACCOUNT_ID not configured") 
        return False
    
    print(f"✅ OANDA credentials configured")
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"🏦 Account: {account_id}")
    print(f"🌐 Environment: {environment.upper()}")
    
    return True

def create_v20_context():
    """Create official OANDA v20 API context."""
    environment = os.getenv("OANDA_ENVIRONMENT", "live")
    
    # Use live endpoints as specified in CLAUDE.md
    if environment == "live":
        hostname = "api-fxtrade.oanda.com"
    else:
        hostname = "api-fxpractice.oanda.com"
    
    print(f"🔌 Connecting to {hostname} (Official OANDA v20 API)")
    
    # Create v20 context (official API)
    api = Context(
        hostname=hostname,
        port=443,
        ssl=True,
        application="MarketEdgeFinder",
        token=os.getenv("OANDA_API_KEY"),
        datetime_format="RFC3339"
    )
    
    return api

def download_instrument_data(api, instrument, start_date, end_date):
    """Download historical data using 5000 bar increments (OANDA v20 max limit)."""
    all_candles = []
    
    # OANDA v20 API limit: 5000 candles per request (official limit)
    max_candles_per_request = 5000
    
    # For H1 data: 5000 hours = ~208 days, but we'll use count-based requests
    current_end = end_date
    chunk_num = 1
    
    # Work backwards from end_date to collect 3 years of data in 5000-bar increments
    while len(all_candles) < 26280 and chunk_num <= 10:  # Allow more chunks for complete collection
        try:
            # Use count parameter to get exactly 5000 candles
            to_time = current_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            
            # Use official v20 API call with count parameter
            response = api.instrument.candles(
                instrument=instrument,
                price="M",  # Mid prices
                granularity="H1",  # 1 hour
                count=max_candles_per_request,
                to=to_time
            )
            
            if response.status != 200:
                error_msg = getattr(response, 'body', {}).get('errorMessage', 'Unknown error')
                print(f"   ❌ Chunk {chunk_num} failed: {response.status} - {error_msg}")
                break
            
            chunk_candles = response.body.get("candles", [])
            
            if not chunk_candles:
                print(f"   ⚠️ No more data available")
                break
            
            # Process candles from this chunk (in reverse chronological order)
            processed_candles = []
            for candle in chunk_candles:
                if candle.complete:  # Only use complete candles
                    candle_time = pd.to_datetime(candle.time)
                    
                    # Only include candles within our date range
                    if candle_time >= start_date and candle_time <= end_date:
                        processed_candles.append({
                            'time': candle.time,
                            'open': float(candle.mid.o),
                            'high': float(candle.mid.h),
                            'low': float(candle.mid.l),
                            'close': float(candle.mid.c),
                            'volume': int(candle.volume) if hasattr(candle, 'volume') else 0
                        })
                    elif candle_time < start_date:
                        # We've gone past our start date - stop processing this chunk
                        break
            
            if not processed_candles:
                print(f"   ⚠️ Reached start date")
                break
            
            # Add to beginning of all_candles (since we're working backwards)
            all_candles = processed_candles + all_candles
            
            # Progress indicator
            if chunk_num == 1:
                print(f"   📊 Chunk {chunk_num}: {len(processed_candles)} candles", end="")
            else:
                print(f", {len(processed_candles)}", end="")
            
            # Update current_end to the earliest time in this chunk
            if processed_candles:
                earliest_time = min(pd.to_datetime(candle['time']) for candle in processed_candles)
                current_end = earliest_time - timedelta(hours=1)  # Move back 1 hour
                
                # Check if we've collected enough data or reached start date
                if len(all_candles) >= 26280 or current_end < start_date:
                    print(f" (collected {len(all_candles)} bars)")
                    break
            else:
                break
            
            chunk_num += 1
            
            # Rate limiting between chunks (respect OANDA limits: 100 req/sec)
            time.sleep(0.1)  # 10 requests per second (conservative)
            
        except Exception as e:
            print(f"   ❌ Chunk {chunk_num} failed: {str(e)}")
            break
    
    if not all_candles:
        print(" - No data")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    # Remove duplicates and filter to exact date range
    df = df[~df.index.duplicated(keep='first')]
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    return df

def main():
    """Main download function."""
    print("🚀 OANDA v20 Live Data Download")
    print("=" * 50)
    
    # Check credentials
    if not check_credentials():
        print("\n🔧 Configure OANDA credentials in .env file:")
        print("   OANDA_API_KEY=your_live_api_key")
        print("   OANDA_ACCOUNT_ID=your_live_account_id") 
        print("   OANDA_ENVIRONMENT=live")
        return False
    
    # Create v20 API context
    try:
        api = create_v20_context()
        print("✅ Official OANDA v20 API context created")
    except Exception as e:
        print(f"❌ Failed to create v20 context: {e}")
        return False
    
    # Test connection
    try:
        account_response = api.account.get(os.getenv("OANDA_ACCOUNT_ID"))
        if account_response.status == 200:
            print(f"✅ Connected to account: {account_response.body['account'].alias}")
        else:
            print(f"❌ Account connection failed: {account_response.status}")
            return False
    except Exception as e:
        print(f"❌ Account test failed: {e}")
        return False
    
    # 24 FX pairs - original 20 plus 4 missing instruments
    instruments = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD",
        "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", 
        "AUD_JPY", "EUR_CHF", "GBP_CHF", "CHF_JPY", "EUR_AUD",
        "GBP_AUD", "AUD_CHF", "NZD_JPY", "CAD_JPY", "AUD_NZD",
        # 4 missing instruments added
        "EUR_CAD", "EUR_NZD", "GBP_CAD", "GBP_NZD"
    ]
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range: 3 years back (timezone-aware)
    from datetime import timezone
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1095)  # 3 years
    
    print(f"\n📊 Download Configuration:")
    print(f"   Instruments: {len(instruments)} FX pairs")
    print(f"   Timeframe: 3 years (H1 granularity)")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    print(f"   Data directory: {data_dir}")
    print("\n🚀 Starting download...\n")
    
    # Download data for each instrument
    successful_downloads = 0
    total_bars = 0
    
    for i, instrument in enumerate(instruments, 1):
        print(f"📈 [{i:2d}/{len(instruments)}] {instrument}...", end=" ")
        
        # Download data
        df = download_instrument_data(api, instrument, start_date, end_date)
        
        if df is not None:
            # Save to CSV (human readable and debuggable)
            file_path = data_dir / f"{instrument}_3years_H1.csv"
            df.to_csv(file_path)
            
            successful_downloads += 1
            total_bars += len(df)
            
            # Calculate coverage
            days_span = (df.index[-1] - df.index[0]).days
            expected_bars = days_span * 24  # 24 hours per day
            coverage = (len(df) / expected_bars * 100) if expected_bars > 0 else 0
            
            print(f"✅ {len(df):,} bars ({days_span} days, {coverage:.1f}% coverage)")
        else:
            print("❌ Failed")
        
        # Rate limiting - respect OANDA limits (100 req/sec)  
        time.sleep(0.1)  # 10 requests per second (conservative)
    
    # Summary
    print(f"\n🎉 Download Complete!")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(instruments)} instruments")
    print(f"📊 Total bars collected: {total_bars:,}")
    print(f"💾 Data saved to: {data_dir}")
    
    if successful_downloads > 0:
        avg_bars = total_bars // successful_downloads
        avg_days = avg_bars // 24
        print(f"📈 Average per instrument: {avg_bars:,} bars (~{avg_days} days)")
        
        # Estimate data size
        estimated_size_mb = total_bars * len(instruments) * 6 * 8 / (1024 * 1024)  # Rough estimate
        print(f"💾 Estimated raw data size: ~{estimated_size_mb:.1f} MB")
    
    print(f"\n🚀 Ready for feature engineering and model training!")
    return successful_downloads > 0

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run download
    success = main()
    
    if not success:
        print("\n❌ Download failed. Check credentials and try again.")
        exit(1)
    else:
        print("\n✅ Download completed successfully!")
        exit(0)