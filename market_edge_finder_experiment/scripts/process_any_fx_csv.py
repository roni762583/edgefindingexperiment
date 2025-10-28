#!/usr/bin/env python3
"""
Production script to process any FX CSV file with automatic instrument detection
and dynamic pip values that update every bar (H1 timeframe)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.practical_incremental import update_practical_indicators, PracticalMultiInstrumentState
from configs.instruments import extract_instrument_from_filename

def trim_incomplete_leading_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim incomplete leading rows where not all indicator values are present.
    
    Keeps the first row where all core indicators (volatility, direction, price_change, csi) 
    have valid values, indicating the indicators have enough historical data to compute properly.
    
    Args:
        df: DataFrame with indicator columns
        
    Returns:
        DataFrame with incomplete leading rows removed
    """
    # Core indicators that should all be present for a complete row
    core_indicators = ['volatility', 'direction', 'price_change', 'csi']
    
    # Find the first row where all core indicators have valid (non-NaN) values
    complete_mask = df[core_indicators].notna().all(axis=1)
    
    if not complete_mask.any():
        print("⚠️  Warning: No complete rows found with all indicators present")
        return df
    
    # Find the first complete row index
    first_complete_idx = complete_mask.idxmax()
    first_complete_pos = df.index.get_loc(first_complete_idx)
    
    # Trim the DataFrame from the first complete row onwards
    trimmed_df = df.iloc[first_complete_pos:].copy()
    
    rows_removed = len(df) - len(trimmed_df)
    if rows_removed > 0:
        print(f"🔧 Trimmed {rows_removed} incomplete leading rows")
        print(f"   First complete row: {first_complete_idx}")
        print(f"   Remaining rows: {len(trimmed_df):,}")
    
    return trimmed_df

def process_fx_csv(csv_path: str, output_path: str = None, batch_size: int = 1000):
    """
    Process any FX CSV file with automatic instrument detection and dynamic pip values.
    
    Args:
        csv_path: Path to FX CSV file (format: PAIR_3years_H1.csv)
        output_path: Path to save results CSV (optional)
        batch_size: Batch size for memory-efficient processing
    
    Returns:
        pandas.DataFrame: Results with all 5 indicators
    """
    
    # Auto-extract instrument from filename
    csv_path = Path(csv_path)
    instrument = extract_instrument_from_filename(str(csv_path))
    
    print(f"🔍 AUTO-DETECTED INSTRUMENT: {instrument}")
    print(f"📂 Input file: {csv_path}")
    print(f"📊 Batch size: {batch_size}")
    print("=" * 60)
    
    # Initialize state
    multi_state = PracticalMultiInstrumentState()
    all_results = []
    
    # Process CSV in chunks to handle large files
    chunk_size = 10000  # Read 10k rows at a time
    total_processed = 0
    
    print(f"🚀 Starting processing...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        print(f"📦 Processing chunk {chunk_num + 1}...")
        
        # Prepare chunk
        chunk['time'] = pd.to_datetime(chunk['time'])
        chunk.set_index('time', inplace=True)
        
        batch_results = []
        
        for i, (timestamp, row) in enumerate(chunk.iterrows()):
            # Prepare OHLC data
            new_ohlc = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            
            # Generate all 5 indicators with dynamic pip values
            indicators, multi_state = update_practical_indicators(new_ohlc, multi_state, instrument)
            
            # Store result
            result = {
                'timestamp': timestamp,
                'bar_index': total_processed,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'slope_high': indicators['slope_high'],
                'slope_low': indicators['slope_low'],
                'volatility': indicators['volatility'],
                'direction': indicators['direction'],
                'price_change': indicators['price_change'],
                'csi': indicators['csi']
            }
            
            batch_results.append(result)
            total_processed += 1
            
            # Process in batches to manage memory
            if len(batch_results) >= batch_size:
                all_results.extend(batch_results)
                batch_results = []
                print(f"  Processed {total_processed} bars...")
        
        # Add remaining results from chunk
        if batch_results:
            all_results.extend(batch_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.set_index('timestamp', inplace=True)
    
    # Trim incomplete leading rows where not all indicator values are present
    results_df = trim_incomplete_leading_rows(results_df)
    
    # Final statistics
    state = multi_state.get_instrument_state(instrument)
    
    print(f"\n🎯 PROCESSING COMPLETE:")
    print(f"  Total bars processed: {total_processed:,}")
    print(f"  HSP detected: {len(state.hsp_indices):,}")
    print(f"  LSP detected: {len(state.lsp_indices):,}")
    print(f"  Total swings: {len(state.hsp_indices) + len(state.lsp_indices):,}")
    
    # Indicator coverage statistics
    print(f"\n📊 INDICATOR COVERAGE:")
    for col in ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change', 'csi']:
        valid_count = (~results_df[col].isna()).sum()
        coverage_pct = valid_count / len(results_df) * 100
        print(f"  {col}: {valid_count:,}/{len(results_df):,} ({coverage_pct:.1f}%)")
    
    # Indicator statistics
    print(f"\n📈 INDICATOR STATISTICS:")
    for col in ['volatility', 'direction', 'price_change']:
        valid_data = results_df[col].dropna()
        if len(valid_data) > 0:
            print(f"  {col}: mean={valid_data.mean():.3f}, std={valid_data.std():.3f}, "
                  f"range=[{valid_data.min():.3f}, {valid_data.max():.3f}]")
    
    # CSI statistics (different scale - raw ADX * ATR_USD values)
    csi_data = results_df['csi'].dropna()
    if len(csi_data) > 0:
        print(f"  csi: mean={csi_data.mean():.1f}, std={csi_data.std():.1f}, "
              f"range=[{csi_data.min():.1f}, {csi_data.max():.1f}]")
    
    # Save results if output path specified
    if output_path:
        output_path = Path(output_path)
        results_df.to_csv(output_path)
        print(f"\n💾 Results saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n✅ SUCCESS: {instrument} processed with all 5 indicators generated!")
    
    return results_df

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Process FX CSV with incremental indicators')
    parser.add_argument('csv_file', help='Path to FX CSV file (e.g., AUD_CHF_3years_H1.csv)')
    parser.add_argument('-o', '--output', help='Output CSV path (optional)')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='Batch size (default: 1000)')
    
    args = parser.parse_args()
    
    # Process the CSV
    results = process_fx_csv(args.csv_file, args.output, args.batch_size)
    
    # Show sample results
    print(f"\n📋 SAMPLE RESULTS (Last 5 bars):")
    print(results[['open', 'high', 'low', 'close', 'volatility', 'direction', 'price_change', 'csi']].tail().round(4))

if __name__ == "__main__":
    main()