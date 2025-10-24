# Script Consolidation Plan

## ✅ Created: Unified Feature Generation Script

**Primary Script**: `scripts/generate_features.py`

### Consolidated Functionality:
- Sample data processing (`regenerate_sample_features.py`)
- Single instrument processing (`process_instrument_data.py`)  
- Batch processing capabilities
- Comprehensive feature analysis
- Optional chart generation
- Command-line interface with multiple options

### Usage Examples:
```bash
# Process sample data (most common)
python3 scripts/generate_features.py --sample

# Process specific file
python3 scripts/generate_features.py --file data/raw/EUR_USD.csv

# Process by instrument name
python3 scripts/generate_features.py --instrument EUR_USD

# Process all instruments
python3 scripts/generate_features.py --all data/raw/

# Generate charts too
python3 scripts/generate_features.py --sample --charts
```

## 📋 Scripts to Remove (Redundant)

### Immediate Removal Candidates:
1. **`regenerate_sample_features.py`** - ✅ Replaced by `--sample` option
2. **`process_instrument_data.py`** - ✅ Replaced by `--file` and `--instrument` options

### Keep but Mark as Legacy:
3. **`feature_pipeline.py`** - Generic pipeline framework (keep for reference)
4. **`run_complete_data_pipeline.py`** - Complete pipeline orchestration (keep for production)

## 🔧 Testing/Debug Scripts (Keep):
- `debug_asi_calculation.py` - Direct ASI testing
- `test_*.py` scripts - Unit testing and validation
- `analyze_si_capping.py` - Analysis tool
- `quick_*.py` scripts - Quick analysis tools

## 📊 Chart Generation Scripts (Keep):
- `chart_all_indicators.py` - Main charting
- `chart_raw_angles.py` - Angle analysis
- `generate_asi_chart.py` - ASI visualization

## 📈 Analysis Scripts (Keep):
- `compare_*.py` scripts - Method comparisons
- Volatility comparison scripts
- Direction method analysis

## 🎯 Recommended Actions:

### 1. Remove Redundant Scripts:
```bash
rm scripts/regenerate_sample_features.py
rm scripts/process_instrument_data.py  
```

### 2. Update Documentation:
- Update README.md to reference unified script
- Add usage examples
- Document command-line options

### 3. Update Other Scripts:
- Modify scripts that call removed functions
- Update import statements if needed

### 4. Testing:
- Verify unified script covers all use cases
- Test all command-line options
- Ensure output compatibility

## ✅ Benefits Achieved:

1. **Single Entry Point**: One script for all feature generation
2. **Consistent Interface**: Standardized command-line options
3. **Comprehensive Analysis**: Built-in feature analysis and reporting
4. **Flexible Usage**: Multiple data source options
5. **Maintainability**: One codebase to maintain instead of multiple scripts
6. **Documentation**: Clear usage examples and help text

## 🔄 Migration Path:

### Old Usage → New Usage:
```bash
# Old
python3 scripts/regenerate_sample_features.py
# New  
python3 scripts/generate_features.py --sample

# Old
python3 scripts/process_instrument_data.py data/raw/EUR_USD.csv
# New
python3 scripts/generate_features.py --file data/raw/EUR_USD.csv
# Or
python3 scripts/generate_features.py --instrument EUR_USD
```

The consolidated script provides all functionality of the removed scripts with enhanced features and better usability.