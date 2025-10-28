#!/usr/bin/env python3
"""
Download the 4 missing instruments only
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the download functionality
from download_real_data_v20 import check_credentials, create_v20_context, download_instrument_data
from datetime import datetime, timedelta
import pandas as pd

def download_missing_instruments():
    """Download only the 4 missing instruments"""
    
    if not check_credentials():
        print("❌ OANDA credentials not configured properly")
        return False
    
    # Create API context
    api = create_v20_context()
    if not api:
        return False
    
    # 4 missing instruments
    missing_instruments = ["EUR_CAD", "EUR_NZD", "GBP_CAD", "GBP_NZD"]
    
    # Date range: 3 years back from now
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=3*365)  # 3 years
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📈 Downloading {len(missing_instruments)} missing instruments...")
    print("=" * 70)
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_bars = 0
    
    for i, instrument in enumerate(missing_instruments, 1):
        print(f"📈 [{i:2d}/{len(missing_instruments)}] {instrument}...", end=" ")
        
        # Download data
        df = download_instrument_data(api, instrument, start_date, end_date)
        
        if df is not None and len(df) > 0:
            # Save to CSV
            filename = f"{instrument}_3years_H1.csv"
            filepath = data_dir / filename
            df.to_csv(filepath, index=False)
            
            file_size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"✅ {len(df):,} bars ({file_size_mb:.1f} MB)")
            
            success_count += 1
            total_bars += len(df)
        else:
            print(f"❌ Failed")
    
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD SUMMARY:")
    print(f"✅ Successful: {success_count}/{len(missing_instruments)}")
    print(f"📈 Total bars: {total_bars:,}")
    print(f"📁 Files saved to: {data_dir.absolute()}")
    
    if success_count == len(missing_instruments):
        print("\n🎉 All missing instruments downloaded successfully!")
        return True
    else:
        print(f"\n⚠️ {len(missing_instruments) - success_count} instruments failed to download")
        return False

if __name__ == "__main__":
    success = download_missing_instruments()
    sys.exit(0 if success else 1)