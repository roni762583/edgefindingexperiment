#!/usr/bin/env python3
'''
Auto-generated download script for missing/incomplete data
Generated on: 2025-10-29 18:20:08
'''

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_pull.download_oanda_data import EdgeFindingOandaDownloader
import os

def main():
    # Get OANDA credentials from environment
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
    
    if not api_key or not account_id:
        print("❌ OANDA API credentials not found in environment variables")
        print("Please set OANDA_API_KEY and OANDA_ACCOUNT_ID")
        return False
    
    # Initialize downloader
    downloader = EdgeFindingOandaDownloader(api_key, account_id, environment)
    
    # Instruments to download
    instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD', 'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD', 'EUR_CAD', 'EUR_NZD', 'GBP_CAD', 'GBP_NZD']
    
    print(f"📥 Downloading 3 years of H1 data for {len(instruments)} instruments...")
    
    for instrument in instruments:
        try:
            print(f"\n🔄 Downloading {instrument}...")
            
            # Download 3 years of H1 data
            success = downloader.download_historical_data(
                instrument=instrument,
                granularity='H1',
                count=26280,  # ~3 years of hourly data
                output_dir='data/raw',
                filename_prefix=f"{instrument}_3years"
            )
            
            if success:
                print(f"✅ {instrument} download completed")
            else:
                print(f"❌ {instrument} download failed")
                
        except Exception as e:
            print(f"❌ {instrument} download error: {e}")
    
    print(f"\n🎉 Download script completed!")
    print(f"Run check_data_completeness.py again to verify downloads")

if __name__ == "__main__":
    main()
