#!/usr/bin/env python3
"""
Implement practical swing detection method in both batch and incremental processing
This should fix slope correlations and provide the recommended 69-swing approach
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import MultiInstrumentState, update_indicators

def detect_practical_swings_batch(asi_values, min_distance=3):
    """
    Practical swing detection for batch processing
    Returns boolean arrays like the existing batch method
    """
    n = len(asi_values)
    hsp_flags = np.full(n, False)
    lsp_flags = np.full(n, False)
    
    hsp_indices = []
    lsp_indices = []
    
    # Find all 3-bar patterns with minimum distance
    for i in range(1, n-1):
        if np.isnan(asi_values[i-1]) or np.isnan(asi_values[i]) or np.isnan(asi_values[i+1]):
            continue
            
        # HSP: local maximum
        if asi_values[i] > asi_values[i-1] and asi_values[i] > asi_values[i+1]:
            # Check minimum distance from last HSP
            if not hsp_indices or (i - hsp_indices[-1]) >= min_distance:
                hsp_indices.append(i)
        
        # LSP: local minimum  
        if asi_values[i] < asi_values[i-1] and asi_values[i] < asi_values[i+1]:
            # Check minimum distance from last LSP
            if not lsp_indices or (i - lsp_indices[-1]) >= min_distance:
                lsp_indices.append(i)
    
    # Convert to boolean arrays
    for idx in hsp_indices:
        hsp_flags[idx] = True
    for idx in lsp_indices:
        lsp_flags[idx] = True
        
    return hsp_flags, lsp_flags

def calculate_practical_slopes_batch(asi_values, hsp_flags, lsp_flags):
    """
    Calculate slopes using practical swing detection results
    """
    n = len(asi_values)
    slope_high = np.full(n, np.nan)
    slope_low = np.full(n, np.nan)
    
    # Get HSP and LSP indices
    hsp_indices = np.where(hsp_flags)[0]
    lsp_indices = np.where(lsp_flags)[0]
    
    # Track current slopes for forward filling
    current_hsp_slope = np.nan
    current_lsp_slope = np.nan
    
    # Calculate slopes as swing points are detected
    for i in range(n):
        
        # Update HSP slope if we have a new HSP
        if i in hsp_indices:
            # Find how many HSPs we have up to this point
            hsp_count = np.sum(hsp_indices <= i)
            if hsp_count >= 2:
                # Get last two HSPs
                last_two_hsp = hsp_indices[hsp_indices <= i][-2:]
                idx1, idx2 = last_two_hsp[0], last_two_hsp[1]
                y1, y2 = asi_values[idx1], asi_values[idx2]
                
                if idx2 != idx1:  # Avoid division by zero
                    raw_slope = (y2 - y1) / (idx2 - idx1)
                    angle_rad = np.arctan(raw_slope)
                    angle_deg = np.degrees(angle_rad)
                    # Linear mapping: (-90, +90) to (-1, +1)
                    current_hsp_slope = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
        
        # Update LSP slope if we have a new LSP
        if i in lsp_indices:
            # Find how many LSPs we have up to this point
            lsp_count = np.sum(lsp_indices <= i)
            if lsp_count >= 2:
                # Get last two LSPs
                last_two_lsp = lsp_indices[lsp_indices <= i][-2:]
                idx1, idx2 = last_two_lsp[0], last_two_lsp[1]
                y1, y2 = asi_values[idx1], asi_values[idx2]
                
                if idx2 != idx1:  # Avoid division by zero
                    raw_slope = (y2 - y1) / (idx2 - idx1)
                    angle_rad = np.arctan(raw_slope)
                    angle_deg = np.degrees(angle_rad)
                    # Linear mapping: (-90, +90) to (-1, +1)
                    current_lsp_slope = -1.0 + (1.0 - (-1.0)) * ((angle_deg - (-90.0)) / (90.0 - (-90.0)))
        
        # Forward fill current slopes
        slope_high[i] = current_hsp_slope
        slope_low[i] = current_lsp_slope
    
    return slope_high, slope_low

def create_practical_batch_processor():
    """
    Create a batch processor that uses practical swing detection
    """
    def process_with_practical_swings(test_slice, instrument):
        # Get basic features using existing batch processor
        generator = FXFeatureGenerator()
        batch_results = generator.generate_features_single_instrument(test_slice, instrument)
        
        # Replace swing detection with practical method
        asi_values = batch_results['asi'].values
        hsp_practical, lsp_practical = detect_practical_swings_batch(asi_values, min_distance=3)
        
        # Replace slopes with practical calculation
        slope_high_practical, slope_low_practical = calculate_practical_slopes_batch(
            asi_values, hsp_practical, lsp_practical
        )
        
        # Update results with practical values
        batch_results['sig_hsp'] = hsp_practical
        batch_results['sig_lsp'] = lsp_practical
        batch_results['slope_high'] = slope_high_practical
        batch_results['slope_low'] = slope_low_practical
        
        return batch_results
    
    return process_with_practical_swings

def test_practical_method_correlation():
    """Test correlation with practical method in both batch and incremental"""
    
    # Load data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use same 2000-bar slice as correlation test
    test_slice = df.iloc[1666:3666].copy()
    print(f"🎯 Testing Practical Method Correlation: {len(test_slice)} bars")
    print(f"Date range: {test_slice.index[0]} to {test_slice.index[-1]}")
    
    # Batch processing with practical method
    practical_processor = create_practical_batch_processor()
    batch_results = practical_processor(test_slice, "EUR_USD")
    
    # Incremental processing (needs to be updated to use practical method too)
    # For now, let's see the batch practical results
    
    print(f"\n📊 Practical Batch Results:")
    
    # Count swing points
    hsp_count = batch_results['sig_hsp'].sum()
    lsp_count = batch_results['sig_lsp'].sum()
    print(f"HSP detected: {hsp_count}")
    print(f"LSP detected: {lsp_count}")
    print(f"Total swings: {hsp_count + lsp_count}")
    
    # Check coverage
    last_hsp = np.where(batch_results['sig_hsp'])[0]
    last_lsp = np.where(batch_results['sig_lsp'])[0]
    
    if len(last_hsp) > 0 or len(last_lsp) > 0:
        last_swing = max(
            last_hsp[-1] if len(last_hsp) > 0 else 0,
            last_lsp[-1] if len(last_lsp) > 0 else 0
        )
        coverage = last_swing / (len(test_slice) - 1) * 100
        print(f"Coverage: {coverage:.1f}% (last swing at bar {last_swing})")
    
    # Check slope coverage
    slope_high_valid = (~batch_results['slope_high'].isna()).sum()
    slope_low_valid = (~batch_results['slope_low'].isna()).sum()
    print(f"Slope high coverage: {slope_high_valid/len(batch_results)*100:.1f}%")
    print(f"Slope low coverage: {slope_low_valid/len(batch_results)*100:.1f}%")
    
    # Show some swing points
    all_swings = []
    hsp_indices = np.where(batch_results['sig_hsp'])[0]
    lsp_indices = np.where(batch_results['sig_lsp'])[0]
    
    for idx in hsp_indices:
        all_swings.append((idx, 'HSP', batch_results['asi'].iloc[idx]))
    for idx in lsp_indices:
        all_swings.append((idx, 'LSP', batch_results['asi'].iloc[idx]))
    
    all_swings.sort()
    
    print(f"\nFirst 10 swings:")
    for i, (bar, type_, asi) in enumerate(all_swings[:10]):
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\nLast 10 swings:")
    for i, (bar, type_, asi) in enumerate(all_swings[-10:]):
        print(f"  Bar {bar:3d}: {type_} at ASI={asi:6.1f}")
    
    print(f"\n✅ Practical batch method implemented!")
    print(f"Next step: Update incremental method to match")
    
    return batch_results

if __name__ == "__main__":
    test_practical_method_correlation()