#!/usr/bin/env python3
"""
Regenerate all 24 processed feature files with corrected CSI implementation
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.instruments import FX_INSTRUMENTS

def regenerate_all_features():
    """Regenerate all feature files with corrected CSI"""
    print("🔄 REGENERATING ALL 24 FEATURE FILES WITH CORRECTED CSI")
    print("=" * 60)
    
    data_dir = Path("data/raw")
    processed_count = 0
    failed_count = 0
    
    for i, instrument in enumerate(FX_INSTRUMENTS, 1):
        csv_file = data_dir / f"{instrument}_3years_H1.csv"
        
        if not csv_file.exists():
            print(f"⚠️  [{i:2d}/24] {instrument}: CSV file missing")
            failed_count += 1
            continue
        
        print(f"📊 [{i:2d}/24] Processing {instrument}...", end=" ")
        
        try:
            # Run the processing script
            result = subprocess.run([
                sys.executable, "scripts/process_any_fx_csv.py", str(csv_file)
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Check if output file was created (correct naming pattern)
                processed_file = Path("data/processed") / f"{instrument}_H1_precomputed_features.csv"
                if processed_file.exists():
                    print("✅ Success")
                    processed_count += 1
                else:
                    print("❌ No output file")
                    failed_count += 1
            else:
                print(f"❌ Failed: {result.stderr[:50]}")
                failed_count += 1
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            failed_count += 1
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}")
            failed_count += 1
    
    print(f"\n🎉 REGENERATION COMPLETE!")
    print(f"✅ Successfully processed: {processed_count}/24 instruments")
    print(f"❌ Failed: {failed_count}/24 instruments")
    
    if processed_count == 24:
        print("🚀 All 24 instruments processed with corrected CSI!")
        return True
    else:
        print("⚠️  Some instruments failed - check individual errors above")
        return False

if __name__ == "__main__":
    # Change to project directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    success = regenerate_all_features()
    exit(0 if success else 1)