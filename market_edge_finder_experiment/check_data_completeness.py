#!/usr/bin/env python3
"""
Check Data Completeness - Verify 3 Years of Data

Checks all 24 instruments for complete 3-year coverage and identifies
any missing data that needs to be downloaded.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_completeness():
    """
    Check if we have complete 3-year data for all 24 instruments.
    """
    
    raw_data_dir = Path("data/raw")
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return False
    
    # Expected 24 instruments
    expected_instruments = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_CHF", "GBP_CHF", "CHF_JPY",
        "EUR_AUD", "GBP_AUD", "AUD_CHF", "NZD_JPY", "CAD_JPY", "AUD_NZD",
        "EUR_CAD", "EUR_NZD", "GBP_CAD", "GBP_NZD"
    ]
    
    logger.info(f"Checking data completeness for {len(expected_instruments)} instruments...")
    
    # Expected 3 years = ~26,280 hours (365.25 * 3 * 24)
    expected_hours = int(365.25 * 3 * 24)
    logger.info(f"Expected ~{expected_hours:,} hours of data per instrument")
    
    # Current date for reference
    current_date = datetime.now()
    three_years_ago = current_date - timedelta(days=3*365.25)
    
    logger.info(f"Expected date range: {three_years_ago.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
    
    data_summary = []
    missing_instruments = []
    incomplete_instruments = []
    
    for instrument in expected_instruments:
        data_file = raw_data_dir / f"{instrument}_3years_H1.csv"
        
        if not data_file.exists():
            logger.warning(f"❌ Missing data file: {instrument}")
            missing_instruments.append(instrument)
            data_summary.append({
                'instrument': instrument,
                'status': 'MISSING',
                'file_exists': False,
                'rows': 0,
                'start_date': None,
                'end_date': None,
                'coverage_days': 0
            })
            continue
        
        try:
            # Load and check the data
            df = pd.read_csv(data_file)
            
            if 'timestamp' not in df.columns:
                logger.warning(f"⚠️  No 'timestamp' column in {instrument}")
                continue
            
            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get date range
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            coverage_days = (end_date - start_date).days
            
            rows = len(df)
            
            # Check if we have reasonable coverage  
            # Forex markets close weekends, so ~18,672 bars for 3 years is excellent
            expected_min_rows = 18000  # Realistic minimum for 3 years with weekends/holidays
            
            if rows < expected_min_rows:
                status = 'INCOMPLETE'
                incomplete_instruments.append(instrument)
                logger.warning(f"⚠️  {instrument}: {rows:,} rows (expected ~{expected_hours:,})")
            else:
                status = 'COMPLETE'
                logger.info(f"✅ {instrument}: {rows:,} rows, {coverage_days} days")
            
            data_summary.append({
                'instrument': instrument,
                'status': status,
                'file_exists': True,
                'rows': rows,
                'start_date': start_date.strftime('%Y-%m-%d %H:%M'),
                'end_date': end_date.strftime('%Y-%m-%d %H:%M'),
                'coverage_days': coverage_days
            })
            
        except Exception as e:
            logger.error(f"❌ Error reading {instrument}: {e}")
            data_summary.append({
                'instrument': instrument,
                'status': 'ERROR',
                'file_exists': True,
                'rows': 0,
                'error': str(e)
            })
    
    # Generate summary report
    logger.info("="*60)
    logger.info("DATA COMPLETENESS REPORT")
    logger.info("="*60)
    
    complete_count = sum(1 for item in data_summary if item['status'] == 'COMPLETE')
    incomplete_count = len(incomplete_instruments)
    missing_count = len(missing_instruments)
    
    logger.info(f"Complete instruments: {complete_count}/24")
    logger.info(f"Incomplete instruments: {incomplete_count}/24")
    logger.info(f"Missing instruments: {missing_count}/24")
    
    if missing_instruments:
        logger.info(f"\nMissing instruments:")
        for instrument in missing_instruments:
            logger.info(f"  ❌ {instrument}")
    
    if incomplete_instruments:
        logger.info(f"\nIncomplete instruments:")
        for instrument in incomplete_instruments:
            item = next(item for item in data_summary if item['instrument'] == instrument)
            logger.info(f"  ⚠️  {instrument}: {item['rows']:,} rows, {item['coverage_days']} days")
    
    # Check date alignment across instruments
    if complete_count > 0:
        logger.info(f"\nDate range analysis:")
        complete_items = [item for item in data_summary if item['status'] == 'COMPLETE']
        
        start_dates = [pd.to_datetime(item['start_date']) for item in complete_items]
        end_dates = [pd.to_datetime(item['end_date']) for item in complete_items]
        
        earliest_start = min(start_dates)
        latest_start = max(start_dates)
        earliest_end = min(end_dates)
        latest_end = max(end_dates)
        
        logger.info(f"  Earliest start: {earliest_start.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Latest start: {latest_start.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Earliest end: {earliest_end.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Latest end: {latest_end.strftime('%Y-%m-%d %H:%M')}")
        
        # Check for alignment issues
        start_range = (latest_start - earliest_start).days
        end_range = (latest_end - earliest_end).days
        
        if start_range > 7 or end_range > 7:
            logger.warning(f"⚠️  Date alignment issues: start range {start_range} days, end range {end_range} days")
        else:
            logger.info(f"✅ Good date alignment: start range {start_range} days, end range {end_range} days")
    
    # Save detailed report
    report_file = Path("data/data_completeness_report.csv")
    df_report = pd.DataFrame(data_summary)
    df_report.to_csv(report_file, index=False)
    logger.info(f"\n📄 Detailed report saved: {report_file}")
    
    # Determine what needs to be downloaded
    needs_download = missing_instruments + incomplete_instruments
    
    if needs_download:
        logger.info(f"\n🔄 DOWNLOAD REQUIRED for {len(needs_download)} instruments:")
        for instrument in needs_download:
            logger.info(f"  📥 {instrument}")
        
        return False, needs_download
    else:
        logger.info(f"\n✅ ALL DATA COMPLETE - Ready for 3-year experiment!")
        return True, []


def generate_download_script(instruments_to_download: list):
    """
    Generate a download script for missing/incomplete instruments.
    """
    if not instruments_to_download:
        return
    
    logger.info(f"🔧 Generating download script for {len(instruments_to_download)} instruments...")
    
    script_content = f"""#!/usr/bin/env python3
'''
Auto-generated download script for missing/incomplete data
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
    instruments = {instruments_to_download}
    
    print(f"📥 Downloading 3 years of H1 data for {{len(instruments)}} instruments...")
    
    for instrument in instruments:
        try:
            print(f"\\n🔄 Downloading {{instrument}}...")
            
            # Download 3 years of H1 data
            success = downloader.download_historical_data(
                instrument=instrument,
                granularity='H1',
                count=26280,  # ~3 years of hourly data
                output_dir='data/raw',
                filename_prefix=f"{{instrument}}_3years"
            )
            
            if success:
                print(f"✅ {{instrument}} download completed")
            else:
                print(f"❌ {{instrument}} download failed")
                
        except Exception as e:
            print(f"❌ {{instrument}} download error: {{e}}")
    
    print(f"\\n🎉 Download script completed!")
    print(f"Run check_data_completeness.py again to verify downloads")

if __name__ == "__main__":
    main()
"""
    
    script_file = Path("download_missing_data.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_file.chmod(0o755)
    
    logger.info(f"📜 Download script created: {script_file}")
    logger.info(f"💡 To run: python {script_file}")


if __name__ == "__main__":
    try:
        is_complete, needs_download = check_data_completeness()
        
        if is_complete:
            print(f"\n🎉 All data is complete! Ready for 3-year ML experiment.")
        else:
            print(f"\n⚠️  Data incomplete. {len(needs_download)} instruments need downloading.")
            
            # Generate download script
            generate_download_script(needs_download)
            
            print(f"\n📋 Next steps:")
            print(f"1. Set OANDA API credentials: export OANDA_API_KEY=your_key")
            print(f"2. Run download script: python download_missing_data.py")
            print(f"3. Re-run this check: python check_data_completeness.py")
        
    except Exception as e:
        print(f"\n❌ Data completeness check failed: {e}")
        import traceback
        traceback.print_exc()