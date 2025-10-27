#!/usr/bin/env python3
"""
Compare batch vs incremental feature generation for consistency validation
Tests both old and new methods on same 2000-bar EUR_USD data slice
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator
from features.incremental_indicators import (
    update_indicators, 
    process_historical_data_incremental,
    MultiInstrumentState
)
from configs.instruments import get_pip_value

@dataclass 
class InstrumentState:
    """State for a single instrument within the combined state structure"""
    # ASI/Swing Point State
    asi_value: float = 0.0
    recent_highs: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_lows: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # ATR State (Dollar Scaling)  
    atr_ema: float = 0.0
    prev_close_usd: float = 0.0
    atr_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # ADX State
    adx_value: float = 0.0
    di_plus_ema: float = 0.0
    di_minus_ema: float = 0.0
    tr_ema: float = 0.0
    adx_history: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # Price Change State
    prev_close: float = 0.0
    
    # General State
    bar_count: int = 0
    pip_size: float = 0.0001
    pip_value: float = 10.0

@dataclass
class MultiInstrumentState:
    """Combined state management for all 20 FX instruments"""
    instruments: Dict[str, InstrumentState] = field(default_factory=dict)
    
    # Cross-instrument context tensor state
    context_matrix: np.ndarray = field(default_factory=lambda: np.zeros((20, 5)))  # 20 instruments × 5 features
    context_timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    # Global market regime state
    market_regime: int = 0  # 16-state regime framework
    regime_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def get_instrument_state(self, instrument: str) -> InstrumentState:
        """Get or create state for an instrument"""
        if instrument not in self.instruments:
            from configs.instruments import get_pip_value
            pip_size, pip_value = get_pip_value(instrument)
            self.instruments[instrument] = InstrumentState(pip_size=pip_size, pip_value=pip_value)
        return self.instruments[instrument]
    
    def update_context_tensor(self, instrument: str, indicators: Dict[str, float]):
        """Update the cross-instrument context tensor"""
        # This would map instrument to index and update the context matrix
        # For prototype, just store the indicators
        pass

def update_indicators_prototype(
    new_ohlc: Dict[str, float],
    multi_state: MultiInstrumentState,
    instrument: str
) -> Tuple[Dict[str, float], MultiInstrumentState]:
    """
    Prototype incremental update function with combined multi-instrument state
    
    Advantages of combined state:
    1. Single file persistence for all 20 instruments
    2. Cross-instrument context tensor integration
    3. Global market regime tracking
    4. Efficient memory usage and I/O operations
    """
    # Get instrument-specific state
    instrument_state = multi_state.get_instrument_state(instrument)
    
    # Update instrument state
    instrument_state.bar_count += 1
    
    # Price change calculation (5th indicator)
    if instrument_state.bar_count > 1 and instrument_state.prev_close > 0:
        price_change = (new_ohlc['close'] - instrument_state.prev_close) / instrument_state.prev_close
    else:
        price_change = 0.0
    
    # Update previous close
    instrument_state.prev_close = new_ohlc['close']
    
    # For prototype: placeholder for other indicators
    # Real implementation would calculate ATR, ADX, slopes incrementally
    indicators = {
        'slope_high': np.nan,  # Would use instrument_state.recent_highs
        'slope_low': np.nan,   # Would use instrument_state.recent_lows
        'volatility': np.nan,  # Would use instrument_state.atr_history
        'direction': np.nan,   # Would use instrument_state.adx_history
        'price_change': price_change
    }
    
    # Update cross-instrument context tensor
    multi_state.update_context_tensor(instrument, indicators)
    
    return indicators, multi_state

def test_batch_processing(df_slice: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Test existing batch processing method"""
    print(f"\n🔄 Testing batch processing on {len(df_slice)} bars...")
    
    generator = FXFeatureGenerator()
    
    try:
        # Generate features using existing methods
        features_df = generator.generate_features_single_instrument(df_slice, instrument)
        
        print(f"✅ Batch processing completed: {len(features_df)} rows generated")
        print(f"Columns: {list(features_df.columns)}")
        
        # Check for NaN values
        for col in features_df.columns:
            nan_count = features_df[col].isna().sum()
            coverage = (len(features_df) - nan_count) / len(features_df) * 100
            print(f"  {col}: {coverage:.1f}% coverage ({nan_count} NaN)")
            
        return features_df
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return pd.DataFrame()

def test_incremental_processing(df_slice: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Test incremental processing method with production implementation"""
    print(f"\n🔄 Testing incremental processing on {len(df_slice)} bars...")
    print(f"✅ Using production-ready incremental implementation")
    
    try:
        # Use the actual production incremental processing
        incremental_df, final_state = process_historical_data_incremental(df_slice, instrument)
        
        # Print state information
        instr_state = final_state.get_instrument_state(instrument)
        print(f"✅ Final state: {instr_state.bar_count} bars processed")
        print(f"📊 ATR history: {len(instr_state.atr_history)} values")
        print(f"📊 ADX history: {len(instr_state.adx_history)} values")
        print(f"📊 Swing highs: {len(instr_state.recent_highs)} detected")
        print(f"📊 Swing lows: {len(instr_state.recent_lows)} detected")
        
        print(f"✅ Incremental processing completed: {len(incremental_df)} rows generated")
        print(f"Columns: {list(incremental_df.columns)}")
        
        # Check for NaN values
        for col in incremental_df.columns:
            nan_count = incremental_df[col].isna().sum()
            coverage = (len(incremental_df) - nan_count) / len(incremental_df) * 100
            print(f"  {col}: {coverage:.1f}% coverage ({nan_count} NaN)")
            
        return incremental_df
        
    except Exception as e:
        print(f"❌ Incremental processing failed: {e}")
        return pd.DataFrame()

def compare_results(batch_df: pd.DataFrame, incremental_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare batch vs incremental results"""
    print(f"\n📊 Comparing batch vs incremental results...")
    
    if batch_df.empty or incremental_df.empty:
        print("❌ Cannot compare - one or both DataFrames are empty")
        return {}
    
    comparison_results = {}
    
    # Find common columns (excluding time-based columns)
    batch_cols = set(batch_df.columns)
    incr_cols = set(incremental_df.columns)
    common_cols = batch_cols.intersection(incr_cols)
    
    print(f"Common columns for comparison: {sorted(common_cols)}")
    
    if not common_cols:
        print("⚠️  No common columns found for comparison")
        return {}
    
    # Align indices for comparison
    common_index = batch_df.index.intersection(incremental_df.index)
    
    if len(common_index) == 0:
        print("⚠️  No common time periods found for comparison")
        return {}
    
    print(f"Comparing {len(common_index)} common time periods...")
    
    for col in sorted(common_cols):
        if col in ['time', 'timestamp']:
            continue
            
        try:
            batch_values = batch_df.loc[common_index, col]
            incr_values = incremental_df.loc[common_index, col]
            
            # Remove NaN values for comparison
            valid_mask = ~(batch_values.isna() | incr_values.isna())
            
            if valid_mask.sum() == 0:
                print(f"  {col}: No valid values for comparison")
                continue
                
            batch_valid = batch_values[valid_mask]
            incr_valid = incr_values[valid_mask]
            
            # Calculate comparison metrics
            if len(batch_valid) > 0:
                correlation = batch_valid.corr(incr_valid) if len(batch_valid) > 1 else np.nan
                mean_diff = (batch_valid - incr_valid).abs().mean()
                max_diff = (batch_valid - incr_valid).abs().max()
                
                comparison_results[col] = {
                    'correlation': correlation,
                    'mean_abs_diff': mean_diff,
                    'max_abs_diff': max_diff,
                    'valid_points': len(batch_valid)
                }
                
                status = "✅" if correlation > 0.95 else "⚠️" if correlation > 0.8 else "❌"
                print(f"  {col}: corr={correlation:.3f} {status}, mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f} ({len(batch_valid)} points)")
            
        except Exception as e:
            print(f"  {col}: Comparison failed - {e}")
    
    return comparison_results

def main():
    print("🚀 Batch vs Incremental Feature Generation Comparison")
    print("Testing consistency between processing methods")
    
    # Load EUR_USD data
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    
    if not raw_data_path.exists():
        print(f"❌ Data file not found: {raw_data_path}")
        return
    
    print(f"📊 Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Take 2000-bar slice (middle section to avoid edge effects)
    total_bars = len(df)
    start_idx = total_bars // 3  # Start at 1/3 through the data
    end_idx = start_idx + 2000
    
    df_slice = df.iloc[start_idx:end_idx].copy()
    
    print(f"✅ Loaded {len(df)} total bars")
    print(f"🎯 Using slice: {len(df_slice)} bars from index {start_idx} to {end_idx}")
    print(f"Date range: {df_slice.index[0]} to {df_slice.index[-1]}")
    
    instrument = "EUR_USD"
    
    # Test 1: Batch processing
    batch_results = test_batch_processing(df_slice, instrument)
    
    # Test 2: Incremental processing  
    incremental_results = test_incremental_processing(df_slice, instrument)
    
    # Test 3: Compare results
    comparison = compare_results(batch_results, incremental_results)
    
    # Save results
    save_dir = project_root / "data/test"
    
    if not batch_results.empty:
        batch_path = save_dir / "batch_features_2000bars.csv"
        batch_results.to_csv(batch_path)
        print(f"\n💾 Batch results saved to: {batch_path}")
    
    if not incremental_results.empty:
        incremental_path = save_dir / "incremental_features_2000bars.csv"
        incremental_results.to_csv(incremental_path)
        print(f"💾 Incremental results saved to: {incremental_path}")
    
    if comparison:
        comparison_df = pd.DataFrame(comparison).T
        comparison_path = save_dir / "batch_vs_incremental_comparison.csv"
        comparison_df.to_csv(comparison_path)
        print(f"💾 Comparison results saved to: {comparison_path}")
    
    # Summary
    print(f"\n🎉 Comparison completed!")
    print(f"📊 Batch processing: {'✅ Success' if not batch_results.empty else '❌ Failed'}")
    print(f"📊 Incremental processing: {'✅ Success' if not incremental_results.empty else '❌ Failed'}")
    print(f"📊 Comparison analysis: {'✅ Success' if comparison else '❌ Failed'}")
    
    if comparison:
        print(f"\n📈 Summary Statistics:")
        for col, stats in comparison.items():
            corr = stats.get('correlation', np.nan)
            if not np.isnan(corr):
                status = "✅ Excellent" if corr > 0.95 else "⚠️ Good" if corr > 0.8 else "❌ Poor"
                print(f"  {col}: {corr:.3f} correlation {status}")

if __name__ == "__main__":
    main()