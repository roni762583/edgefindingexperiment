#!/usr/bin/env python3
"""
Verify Data Completeness - Remove Incomplete Rows

Checks all 24 pre-computed feature files and ensures only complete rows 
(no NaN, no missing values) are kept for ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_and_clean_feature_files():
    """
    Verify all pre-computed feature files and remove incomplete rows.
    """
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    
    # Get all feature files
    feature_files = list(processed_dir.glob("*_H1_precomputed_features.csv"))
    
    if len(feature_files) != 24:
        logger.warning(f"Expected 24 files, found {len(feature_files)}")
    
    logger.info(f"Verifying {len(feature_files)} feature files...")
    
    summary_stats = []
    
    for file_path in sorted(feature_files):
        instrument = file_path.stem.replace("_H1_precomputed_features", "")
        logger.info(f"Processing {instrument}...")
        
        # Load the data
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        logger.info(f"  Original rows: {original_rows}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Check for missing values in each column
        missing_counts = df.isnull().sum()
        logger.info(f"  Missing values per column:")
        for col, count in missing_counts.items():
            if count > 0:
                logger.info(f"    {col}: {count} missing ({count/len(df)*100:.1f}%)")
        
        # Focus on the 5 key feature columns
        key_features = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change']
        available_features = [col for col in key_features if col in df.columns]
        
        if len(available_features) != 5:
            logger.warning(f"  Missing key features in {instrument}: expected {key_features}, found {available_features}")
        
        # Check for infinite values
        inf_counts = {}
        for col in available_features:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
                logger.info(f"    {col}: {inf_count} infinite values")
        
        # Remove rows with any NaN or infinite values in key features
        df_clean = df.copy()
        
        # Remove rows with NaN in key features
        for col in available_features:
            before_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=[col])
            after_rows = len(df_clean)
            if before_rows != after_rows:
                logger.info(f"    Removed {before_rows - after_rows} rows with NaN in {col}")
        
        # Remove rows with infinite values in key features
        for col in available_features:
            before_rows = len(df_clean)
            df_clean = df_clean[np.isfinite(df_clean[col])]
            after_rows = len(df_clean)
            if before_rows != after_rows:
                logger.info(f"    Removed {before_rows - after_rows} rows with infinite values in {col}")
        
        # Additional checks for realistic value ranges
        for col in available_features:
            col_values = df_clean[col]
            
            if col in ['volatility', 'direction']:
                # These should be [0,1] scaled
                out_of_range = ((col_values < 0) | (col_values > 1)).sum()
                if out_of_range > 0:
                    logger.warning(f"    {col}: {out_of_range} values outside [0,1] range")
                    # Remove out of range values
                    df_clean = df_clean[(df_clean[col] >= 0) & (df_clean[col] <= 1)]
            
            elif col in ['slope_high', 'slope_low']:
                # These should be reasonable slope values
                extreme_values = (np.abs(col_values) > 1000).sum()
                if extreme_values > 0:
                    logger.warning(f"    {col}: {extreme_values} extreme slope values (>1000)")
        
        final_rows = len(df_clean)
        rows_removed = original_rows - final_rows
        
        logger.info(f"  Final rows: {final_rows}")
        logger.info(f"  Rows removed: {rows_removed} ({rows_removed/original_rows*100:.1f}%)")
        
        # Save cleaned data if any rows were removed
        if rows_removed > 0:
            backup_file = file_path.parent / f"{instrument}_H1_precomputed_features_backup.csv"
            
            # Create backup of original
            df.to_csv(backup_file, index=False)
            logger.info(f"  Backup saved: {backup_file}")
            
            # Save cleaned version
            df_clean.to_csv(file_path, index=False)
            logger.info(f"  Cleaned file saved: {file_path}")
        else:
            logger.info(f"  ✅ No cleaning needed for {instrument}")
        
        # Collect summary statistics
        feature_stats = {}
        for col in available_features:
            if col in df_clean.columns:
                values = df_clean[col]
                feature_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': int(len(values))
                }
        
        summary_stats.append({
            'instrument': instrument,
            'original_rows': original_rows,
            'final_rows': final_rows,
            'rows_removed': rows_removed,
            'removal_rate': rows_removed / original_rows if original_rows > 0 else 0,
            'features': feature_stats
        })
        
        logger.info(f"  ✅ {instrument} completed\n")
    
    # Generate summary report
    logger.info("="*60)
    logger.info("DATA CLEANING SUMMARY REPORT")
    logger.info("="*60)
    
    total_original = sum(stat['original_rows'] for stat in summary_stats)
    total_final = sum(stat['final_rows'] for stat in summary_stats)
    total_removed = total_original - total_final
    
    logger.info(f"Total original rows: {total_original:,}")
    logger.info(f"Total final rows: {total_final:,}")
    logger.info(f"Total removed rows: {total_removed:,} ({total_removed/total_original*100:.1f}%)")
    logger.info(f"Files processed: {len(summary_stats)}")
    
    # Show per-instrument summary
    logger.info("\nPer-instrument summary:")
    for stat in summary_stats:
        logger.info(f"  {stat['instrument']}: {stat['final_rows']:,} rows ({stat['removal_rate']*100:.1f}% removed)")
    
    # Find minimum row count for alignment
    min_rows = min(stat['final_rows'] for stat in summary_stats)
    max_rows = max(stat['final_rows'] for stat in summary_stats)
    
    logger.info(f"\nRow count range: {min_rows:,} to {max_rows:,}")
    logger.info(f"For ML training alignment, use: {min_rows:,} rows")
    
    # Check feature value ranges
    logger.info("\nFeature value ranges (across all instruments):")
    all_features = ['slope_high', 'slope_low', 'volatility', 'direction', 'price_change']
    
    for feature in all_features:
        values = []
        for stat in summary_stats:
            if feature in stat['features']:
                feat_stat = stat['features'][feature]
                values.extend([feat_stat['min'], feat_stat['max']])
        
        if values:
            logger.info(f"  {feature}: [{min(values):.6f}, {max(values):.6f}]")
    
    # Save summary to file
    summary_file = Path("data/processed/data_cleaning_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("DATA CLEANING SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total original rows: {total_original:,}\n")
        f.write(f"Total final rows: {total_final:,}\n") 
        f.write(f"Total removed rows: {total_removed:,} ({total_removed/total_original*100:.1f}%)\n")
        f.write(f"Files processed: {len(summary_stats)}\n\n")
        
        f.write("Per-instrument summary:\n")
        for stat in summary_stats:
            f.write(f"  {stat['instrument']}: {stat['final_rows']:,} rows ({stat['removal_rate']*100:.1f}% removed)\n")
        
        f.write(f"\nFor ML training: use {min_rows:,} aligned rows\n")
    
    logger.info(f"\n✅ Summary saved to: {summary_file}")
    logger.info("="*60)
    
    return summary_stats


if __name__ == "__main__":
    try:
        summary = verify_and_clean_feature_files()
        print(f"\n🎉 Data verification and cleaning completed!")
        print(f"All {len(summary)} files processed and cleaned.")
        
    except Exception as e:
        print(f"\n❌ Data verification failed: {e}")
        import traceback
        traceback.print_exc()