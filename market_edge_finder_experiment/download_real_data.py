#!/usr/bin/env python3
"""
Real OANDA Data Download Script

Downloads 3 years of hourly data for 20 major FX pairs.
No simulations - real market data from OANDA v20 API.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Check if we have real OANDA credentials
def check_credentials():
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    if not api_key or api_key == "your_key_here":
        print("❌ OANDA_API_KEY not configured")
        print("📝 Please set your real OANDA API key in .env file")
        return False
    
    if not account_id or account_id == "your_account_here":
        print("❌ OANDA_ACCOUNT_ID not configured") 
        print("📝 Please set your real OANDA account ID in .env file")
        return False
    
    print(f"✅ OANDA credentials configured")
    print(f"🔑 API Key: {api_key[:8]}...")
    print(f"🏦 Account: {account_id}")
    return True

async def download_real_fx_data():
    """Download real FX data from OANDA."""
    
    if not check_credentials():
        print("\n🔧 To configure OANDA credentials:")
        print("1. Get API credentials from OANDA (practice or live account)")
        print("2. Edit .env file with your real credentials:")
        print("   OANDA_API_KEY=your_real_api_key")
        print("   OANDA_ACCOUNT_ID=your_real_account_id")
        print("   OANDA_ENVIRONMENT=practice")
        return False
    
    # Import OANDA v20 library directly (official API)
    try:
        import v20
        from v20 import Context
        print("✅ Official OANDA v20 library imported")
    except ImportError as e:
        print(f"❌ v20 library import error: {e}")
        print("🔧 Install with: pip install v20>=3.0.25.0")
        return False
    
    # 20 major FX pairs
    instruments = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD",
        "USD_CAD", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", 
        "AUD_JPY", "EUR_CHF", "GBP_CHF", "CHF_JPY", "EUR_AUD",
        "GBP_AUD", "AUD_CHF", "NZD_JPY", "CAD_JPY", "AUD_NZD"
    ]
    
    print(f"🎯 Target: {len(instruments)} FX pairs")
    print(f"📅 Timeframe: 3 years of hourly data")
    print(f"📊 Granularity: H1 (1 hour)")
    
    # Initialize API with LIVE environment
    environment = os.getenv("OANDA_ENVIRONMENT", "live")
    print(f"🏦 OANDA Environment: {environment.upper()}")
    if environment != "live":
        print("⚠️  WARNING: Not using live environment!")
    
    api_manager = OANDAAPIManager(
        api_key=os.getenv("OANDA_API_KEY"),
        account_id=os.getenv("OANDA_ACCOUNT_ID"),
        environment=environment
    )
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range: 3 years back
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # 3 years
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print("🚀 Starting real data download...\n")
    
    successful_downloads = 0
    total_bars = 0
    
    for i, instrument in enumerate(instruments, 1):
        print(f"📈 [{i:2d}/{len(instruments)}] Downloading {instrument}...")
        
        try:
            # Download historical candles
            candles = await api_manager.get_candles(
                instrument=instrument,
                granularity="H1",
                from_time=start_date.isoformat(),
                to_time=end_date.isoformat()
            )
            
            if not candles:
                print(f"   ❌ No data received for {instrument}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()
            
            # Save raw data
            file_path = data_dir / f"{instrument}_3years_H1.parquet"
            df.to_parquet(file_path, compression='snappy')
            
            successful_downloads += 1
            total_bars += len(df)
            
            # Calculate data coverage
            days_coverage = (df.index[-1] - df.index[0]).days
            expected_bars = days_coverage * 24  # 24 hours per day
            coverage_pct = (len(df) / expected_bars) * 100 if expected_bars > 0 else 0
            
            print(f"   ✅ {len(df):,} bars saved ({days_coverage} days, {coverage_pct:.1f}% coverage)")
            
            # Rate limiting pause
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
    
    print(f"\n🎉 Download Complete!")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(instruments)} instruments")
    print(f"📊 Total bars collected: {total_bars:,}")
    print(f"💾 Data saved to: {data_dir}")
    
    if successful_downloads > 0:
        print(f"\n📈 Average bars per instrument: {total_bars // successful_downloads:,}")
        print(f"🗓️  Estimated data span: {total_bars // successful_downloads // 24:.0f} days per instrument")
    
    return successful_downloads > 0

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run download
    success = asyncio.run(download_real_fx_data())
    
    if success:
        print("\n🚀 Ready for feature engineering and model training!")
    else:
        print("\n❌ Download failed. Please check credentials and try again.")