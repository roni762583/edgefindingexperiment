#!/usr/bin/env python3
"""
Check if our Wilder implementation maintains proper alternating HSP/LSP pattern
"""

import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.feature_engineering import FXFeatureGenerator

def check_alternating_pattern():
    """Check if detected swing points properly alternate"""
    
    raw_data_path = project_root / "data/raw/EUR_USD_3years_H1.csv"
    df = pd.read_csv(raw_data_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Use 200-bar range
    test_slice = df.iloc[2000:2200].copy()
    
    generator = FXFeatureGenerator()
    batch_results = generator.generate_features_single_instrument(test_slice, "EUR_USD")
    
    # Extract all detected swing points in sequence
    swing_sequence = []
    
    for i, (asi, hsp, lsp) in enumerate(zip(batch_results['asi'], batch_results['sig_hsp'], batch_results['sig_lsp'])):
        if hsp:
            swing_sequence.append((i, 'HSP', asi))
        if lsp:
            swing_sequence.append((i, 'LSP', asi))
    
    print(f"🔍 Analyzing alternating pattern in detected swings")
    print(f"Total swings detected: {len(swing_sequence)}")
    
    # Check for alternating pattern
    print(f"\n📊 Swing Sequence (should alternate HSP/LSP):")
    print("Bar | Type | ASI   | Alternates?")
    print("----|------|-------|------------")
    
    alternation_violations = []
    
    for i, (bar, type_, asi) in enumerate(swing_sequence):
        alternates = "✓"
        
        if i > 0:
            prev_type = swing_sequence[i-1][1]
            if type_ == prev_type:  # Same type as previous = violation
                alternates = "❌ VIOLATION"
                alternation_violations.append((i, bar, type_, prev_type))
        
        print(f"{bar:3d} | {type_:3s} | {asi:5.1f} | {alternates}")
    
    # Analyze violations
    print(f"\n🚨 Alternation Analysis:")
    if len(alternation_violations) == 0:
        print("✅ Perfect alternation - no violations found!")
    else:
        print(f"❌ Found {len(alternation_violations)} alternation violations:")
        for i, (seq_idx, bar, type_, prev_type) in enumerate(alternation_violations):
            print(f"  Violation {i+1}: Bar {bar} is {type_} but previous was {prev_type}")
    
    # Check for missing patterns in violation areas
    if alternation_violations:
        print(f"\n🔍 Checking for missed 3-bar patterns near violations:")
        
        for seq_idx, bar, type_, prev_type in alternation_violations:
            # Look for missed patterns between this swing and the previous one
            prev_bar = swing_sequence[seq_idx-1][0] if seq_idx > 0 else 0
            
            print(f"\nBetween Bar {prev_bar} ({prev_type}) and Bar {bar} ({type_}):")
            
            # Check all bars in between for missed patterns
            for check_bar in range(prev_bar + 1, bar):
                if check_bar > 0 and check_bar < len(batch_results) - 1:
                    left = batch_results['asi'].iloc[check_bar-1]
                    middle = batch_results['asi'].iloc[check_bar]
                    right = batch_results['asi'].iloc[check_bar+1]
                    
                    should_be_hsp = middle > left and middle > right
                    should_be_lsp = middle < left and middle < right
                    
                    if should_be_hsp and prev_type == 'LSP':
                        print(f"  Bar {check_bar}: MISSED HSP (ASI={middle:.1f}) - would fix alternation!")
                    elif should_be_lsp and prev_type == 'HSP':
                        print(f"  Bar {check_bar}: MISSED LSP (ASI={middle:.1f}) - would fix alternation!")
    
    # Calculate statistics
    hsp_count = sum(1 for _, type_, _ in swing_sequence if type_ == 'HSP')
    lsp_count = sum(1 for _, type_, _ in swing_sequence if type_ == 'LSP')
    
    print(f"\n📈 Statistics:")
    print(f"HSP detected: {hsp_count}")
    print(f"LSP detected: {lsp_count}")
    print(f"Balance: {abs(hsp_count - lsp_count)} difference")
    print(f"Alternation violations: {len(alternation_violations)}")
    print(f"Alternation rate: {(len(swing_sequence) - len(alternation_violations))/len(swing_sequence)*100:.1f}%")
    
    return swing_sequence, alternation_violations

if __name__ == "__main__":
    check_alternating_pattern()