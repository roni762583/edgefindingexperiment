# Edge Finding Experiment - Market Edge Finder 2

Hybrid machine learning system for 1-hour FX return prediction using TCNAE + LightGBM across 20 currency pairs.

## 🚀 Latest Updates - Wilder's ASI Implementation

The Accumulative Swing Index (ASI) has been completely reimplemented according to Wilder's original specification with modern enhancements:

### Key Features:
- **USD Normalization**: OHLC prices converted to USD per 100k standard lot for cross-instrument comparability
- **Dynamic Limit Move**: L = 3 × ATR (Average True Range in USD terms) adapts to market volatility  
- **Wilder's SI Formula**: 50 × (N/R) × (K/L) with no capping for natural range
- **Wilder's Original R Formula**: 3-scenario calculation per original specification
- **Linear Angle Mapping**: degrees (-90°, +90°) mapped to (-1, +1) using linear transformation
- **Significant Swing Points**: 3-bar alternating constraint algorithm for meaningful trend changes

### Implementation Results:
- ✅ **700 EUR_USD H1 bars** processed successfully
- ✅ **140 HSP, 140 LSP** significant swing points detected
- ✅ **ASI range**: [-523, 282] USD per 100k lot (uncapped)
- ✅ **682 HSP, 680 LSP** valid angle calculations with linear mapping
- ✅ **SI range**: [-100, 100] with 50x multiplier (natural bounds)
- ✅ **Cross-instrument compatibility** through USD normalization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate features from sample data (testing)
python3 scripts/generate_features.py --sample

# Process specific instrument
python3 scripts/generate_features.py --instrument EUR_USD

# Generate features with charts
python3 scripts/generate_features.py --sample --charts

# Download all 20 FX pairs historical data
python scripts/download_all_fx_data.py

# Validate downloads
python scripts/download_all_fx_data.py --validate-only
```

## CRITICAL: OANDA API Requirements

**⚠️ MUST USE OFFICIAL OANDA v20 LIBRARY ONLY**

- **Library:** `v20>=3.0.25.0`
- **Documentation:** https://oanda-api-v20.readthedocs.io/
- **NEVER use:** oandapyV20, python-oanda-v20, or any unofficial libraries
- **ONLY use:** Official OANDA v20 library and its official documentation

## Environment Setup

Create `.env` file with OANDA credentials:
```
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=live
IS_DEMO=false
```

## Architecture

- **TCNAE:** Temporal Convolutional Autoencoder for sequence compression
- **LightGBM:** Gradient boosting for final predictions
- **Context Tensor:** Cross-instrument information sharing
- **20 FX Pairs:** EUR_USD, GBP_USD, USD_JPY, etc.
- **Features:** 4 causal indicators per instrument (ASI with USD normalization & uncapped SI, HSP/LSP angle slopes with linear mapping, volatility with ATR zscore, scaled ADX direction)

## Data Pipeline

### 1. Historical Data Download
```bash
# Primary download script (CSV format first)
python download_real_data_v20.py
```

**Data Storage Strategy:**
- **CSV First**: Download complete 3-year datasets as CSV (human readable, debuggable)
- **Verify Completeness**: Ensure all ~26,280 hours downloaded successfully
- **Convert to Parquet**: Only after verification, convert for performance

**Verification:**
```bash
ls -la data/raw/*.csv | wc -l  # Should show 20 files
wc -l data/raw/*.csv           # Should show ~26,280 lines each
```

### 2. Feature Engineering
```bash
# Unified feature generation (recommended)
python3 scripts/generate_features.py --sample

# Complete pipeline for production
python scripts/run_complete_data_pipeline.py
```

### 3. Training & Inference
- **Training:** `training/` (planned)
- **Inference:** `inference/` (planned)

## Current Status

✅ Directory structure created  
✅ OANDA data downloader adapted  
✅ Multi-instrument parallel downloader  
✅ **NEW**: Wilder's ASI implementation with USD normalization
✅ **NEW**: Significant swing point detection algorithm
✅ **NEW**: Linear angle mapping for regression lines between swing points
✅ **NEW**: Unified feature generation script (`generate_features.py`)
✅ **NEW**: Script consolidation and cleanup
✅ Feature engineering pipeline (ASI component)
⏳ TCNAE model implementation  
⏳ LightGBM integration  
⏳ Context tensor system  

## Feature Generation Options

### Unified Script (Recommended)
```bash
# Process sample data for testing
python3 scripts/generate_features.py --sample

# Process specific instrument by name
python3 scripts/generate_features.py --instrument EUR_USD

# Process specific CSV file
python3 scripts/generate_features.py --file data/raw/EUR_USD.csv

# Process all instruments in directory
python3 scripts/generate_features.py --all data/raw/

# Generate with analysis charts
python3 scripts/generate_features.py --sample --charts

# Get help and options
python3 scripts/generate_features.py --help
```

### Key Benefits:
- **Single entry point** for all feature generation needs
- **Comprehensive analysis** with built-in statistics and reporting
- **Flexible options** supporting multiple data sources
- **Optional visualization** with integrated chart generation
- **Consistent output format** across all processing modes

## Dependencies

Core requirements in `requirements.txt` - uses only official libraries and follows strict production standards.