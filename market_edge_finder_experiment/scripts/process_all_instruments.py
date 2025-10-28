#!/usr/bin/env python3
"""
Multi-instrument coordinator for processing all 20 available FX pairs
Generates precomputed feature files for all instruments with standardized output format
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.process_any_fx_csv import process_fx_csv

def process_single_instrument(csv_path: Path) -> dict:
    """
    Process a single instrument CSV file.
    
    Args:
        csv_path: Path to the FX CSV file
        
    Returns:
        dict: Processing results with timing and statistics
    """
    start_time = time.time()
    
    # Define output path
    instrument_name = csv_path.stem.split('_3years_H1')[0]  # Extract EUR_USD from EUR_USD_3years_H1
    output_path = project_root / "data" / "processed" / f"{instrument_name}_H1_precomputed_features.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process the instrument
        results_df = process_fx_csv(str(csv_path), str(output_path), batch_size=1000)
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        total_bars = len(results_df)
        indicator_coverage = {}
        for col in ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change', 'csi']:
            valid_count = (~results_df[col].isna()).sum()
            coverage_pct = valid_count / total_bars * 100
            indicator_coverage[col] = coverage_pct
        
        # File size
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        
        return {
            'instrument': instrument_name,
            'status': 'SUCCESS',
            'processing_time': processing_time,
            'total_bars': total_bars,
            'output_file': str(output_path),
            'file_size_mb': file_size_mb,
            'indicator_coverage': indicator_coverage,
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'instrument': instrument_name,
            'status': 'FAILED',
            'processing_time': processing_time,
            'total_bars': 0,
            'output_file': None,
            'file_size_mb': 0,
            'indicator_coverage': {},
            'error': str(e)
        }

def process_all_instruments_parallel(max_workers: int = None):
    """
    Process all 20 FX instruments in parallel.
    
    Args:
        max_workers: Maximum number of parallel workers (default: CPU count)
    """
    print("🚀 MULTI-INSTRUMENT COORDINATOR: Processing All 20 FX Pairs")
    print("=" * 70)
    
    # Find all available CSV files
    data_dir = project_root / "data" / "raw"
    csv_files = list(data_dir.glob("*_3years_H1.csv"))
    csv_files.sort()  # Alphabetical order
    
    print(f"📂 Found {len(csv_files)} CSV files to process")
    print(f"⚡ Using {max_workers or mp.cpu_count()} parallel workers")
    print()
    
    # Process instruments in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_instrument, csv_path): csv_path 
                         for csv_path in csv_files}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            csv_path = future_to_file[future]
            result = future.result()
            results.append(result)
            completed += 1
            
            # Progress update
            status = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status} [{completed:2d}/{len(csv_files)}] {result['instrument']:<8} | "
                  f"{result['processing_time']:5.1f}s | {result['total_bars']:,} bars")
            
            if result['error']:
                print(f"    Error: {result['error']}")
    
    total_time = time.time() - start_time
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"⚡ Average time per instrument: {total_time/len(csv_files):.1f} seconds")
    
    if successful:
        total_bars = sum(r['total_bars'] for r in successful)
        total_size_mb = sum(r['file_size_mb'] for r in successful)
        
        print(f"📈 Total bars processed: {total_bars:,}")
        print(f"💾 Total output size: {total_size_mb:.1f} MB")
        print(f"📁 Output directory: {project_root}/data/processed/")
        
        # Coverage statistics
        print(f"\n📊 INDICATOR COVERAGE ACROSS ALL INSTRUMENTS:")
        indicator_names = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change', 'csi']
        
        for indicator in indicator_names:
            coverages = [r['indicator_coverage'].get(indicator, 0) for r in successful]
            if coverages:
                avg_coverage = sum(coverages) / len(coverages)
                min_coverage = min(coverages)
                max_coverage = max(coverages)
                print(f"  {indicator:<12}: avg={avg_coverage:5.1f}%, range=[{min_coverage:5.1f}%, {max_coverage:5.1f}%]")
    
    if failed:
        print(f"\n❌ FAILED INSTRUMENTS:")
        for result in failed:
            print(f"  {result['instrument']}: {result['error']}")
    
    print(f"\n🎯 All processing complete! Check data/processed/ for output files.")
    
    return results

def process_all_instruments_sequential():
    """Process all instruments sequentially (fallback for debugging)"""
    print("🐌 SEQUENTIAL PROCESSING MODE")
    print("=" * 70)
    
    data_dir = project_root / "data" / "raw"
    csv_files = list(data_dir.glob("*_3years_H1.csv"))
    csv_files.sort()
    
    results = []
    start_time = time.time()
    
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n[{i:2d}/{len(csv_files)}] Processing {csv_path.stem}...")
        result = process_single_instrument(csv_path)
        results.append(result)
        
        status = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status} {result['instrument']}: {result['processing_time']:.1f}s, {result['total_bars']:,} bars")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total sequential time: {total_time:.1f} seconds")
    
    return results

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process all 20 FX instruments')
    parser.add_argument('--sequential', action='store_true', 
                       help='Use sequential processing instead of parallel')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    if args.sequential:
        results = process_all_instruments_sequential()
    else:
        results = process_all_instruments_parallel(max_workers=args.workers)
    
    # Generate instrument ranking by CSI
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if len(successful) >= 3:
        print(f"\n🏆 TOP 5 INSTRUMENTS BY PROCESSING EFFICIENCY:")
        by_speed = sorted(successful, key=lambda x: x['processing_time'])[:5]
        for i, result in enumerate(by_speed, 1):
            bars_per_sec = result['total_bars'] / result['processing_time']
            print(f"  {i}. {result['instrument']:<8}: {bars_per_sec:,.0f} bars/sec")

if __name__ == "__main__":
    main()