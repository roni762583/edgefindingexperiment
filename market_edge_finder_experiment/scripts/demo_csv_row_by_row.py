#!/usr/bin/env python3
"""
Demo: Row-by-row CSV processing with incremental indicator generation
Shows how the system can process any FX CSV and generate all 5 indicators efficiently
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.practical_incremental import update_practical_indicators, PracticalMultiInstrumentState
from configs.instruments import extract_instrument_from_filename

def process_fx_csv_row_by_row(csv_path: str, instrument: str = None, batch_size: int = 100):
    """
    Demo: Process FX CSV row by row with memory-efficient batching
    
    Args:
        csv_path: Path to FX CSV file (time,open,high,low,close)
        instrument: FX pair name (e.g., 'EUR_USD'). If None, extracts from filename
        batch_size: Process this many rows before yielding results
    
    Returns:
        Generator yielding (batch_results, current_state) tuples
    """
    
    # Auto-extract instrument from filename if not provided
    if instrument is None:
        instrument = extract_instrument_from_filename(csv_path)
        print(f"🔍 Auto-detected instrument from filename: {instrument}")
    
    print(f"🚀 Starting row-by-row processing: {csv_path}")
    print(f"📊 Instrument: {instrument}, Batch size: {batch_size}")
    
    # Initialize state (persistent across all rows)
    multi_state = PracticalMultiInstrumentState()
    
    # Read CSV in chunks to avoid memory issues
    chunk_size = 1000  # Read 1000 rows at a time from disk
    total_processed = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk['time'] = pd.to_datetime(chunk['time'])
        chunk.set_index('time', inplace=True)
        
        batch_results = []
        
        for i, (timestamp, row) in enumerate(chunk.iterrows()):
            # Prepare OHLC data for current bar
            new_ohlc = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            
            # Generate all 5 indicators incrementally
            indicators, multi_state = update_practical_indicators(new_ohlc, multi_state, instrument)
            
            # Add metadata
            result = {
                'timestamp': timestamp,
                'bar_index': total_processed,
                **indicators  # slope_high, slope_low, volatility, direction, price_change
            }
            
            batch_results.append(result)
            total_processed += 1
            
            # Yield batch when ready
            if len(batch_results) >= batch_size:
                yield pd.DataFrame(batch_results), multi_state
                batch_results = []  # Clear for next batch
        
        # Yield remaining results from chunk
        if batch_results:
            yield pd.DataFrame(batch_results), multi_state
    
    # Final state summary
    state = multi_state.get_instrument_state(instrument)
    print(f"\n🎯 Processing Complete:")
    print(f"  Total bars processed: {total_processed}")
    print(f"  HSP detected: {len(state.hsp_indices)}")
    print(f"  LSP detected: {len(state.lsp_indices)}")
    print(f"  Memory efficient: ✅ (batched processing)")
    print(f"  State persistent: ✅ (across all batches)")

def demo_incremental_processing():
    """Demo the incremental processing capabilities"""
    
    # Test with AUD_CHF as requested
    csv_path = project_root / "data/raw/AUD_CHF_3years_H1.csv"
    
    print("🔍 DEMO: Incremental Row-by-Row Processing")
    print("=" * 60)
    
    # Process first 500 rows as demo
    df = pd.read_csv(csv_path)
    demo_rows = 500
    
    print(f"📂 Loading first {demo_rows} rows for demo...")
    
    # Save demo subset with proper filename for auto-detection
    demo_csv = project_root / "data/test/AUD_CHF_demo_500_rows.csv"
    df.head(demo_rows).to_csv(demo_csv, index=False)
    
    batch_count = 0
    total_bars = 0
    
    # Process row by row with 50-bar batches (instrument auto-detected from filename)
    for batch_df, current_state in process_fx_csv_row_by_row(demo_csv, instrument=None, batch_size=50):
        batch_count += 1
        batch_size = len(batch_df)
        total_bars += batch_size
        
        print(f"📦 Batch {batch_count}: {batch_size} bars processed (Total: {total_bars})")
        
        # Show sample of latest indicators
        if not batch_df.empty:
            latest = batch_df.iloc[-1]
            print(f"    Latest indicators: vol={latest['volatility']:.3f}, "
                  f"dir={latest['direction']:.3f}, pc={latest['price_change']:.3f}")
        
        # Memory management: each batch is independent
        del batch_df  # Free memory for next batch
    
    print(f"\n✅ Demo complete: {total_bars} bars processed in {batch_count} batches")
    print(f"🧠 Memory efficient: Each batch freed after processing")
    print(f"📈 State persistent: Swing detection continues across batches")
    
    return True

if __name__ == "__main__":
    demo_incremental_processing()