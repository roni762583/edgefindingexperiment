# Edge Finding Experiment - Market Edge Finder 2

Hybrid machine learning system for 1-hour FX return prediction using TCNAE + LightGBM across 20 currency pairs.

## 🚀 Latest Updates - Complete 5-Indicator System + CSI Reference

A comprehensive technical analysis system implementing all 5 core indicators for edge discovery plus CSI for reference:

### 🔧 5-Indicator Edge Finding System:
1. **ASI (Accumulative Swing Index)**: Wilder's formula with USD normalization
2. **HSP Angles**: High swing point regression slopes 
3. **LSP Angles**: Low swing point regression slopes
4. **Direction**: ADX-based directional strength indicator
5. **Volatility**: ATR z-score normalized volatility measure

### 📚 Reference Implementation (NOT used in edge finding):
- **CSI (Commodity Selection Index)**: Wilder's original formula with dynamic OANDA API integration - for market ranking reference only

### ⚙️ Technical Specifications:
- **USD Normalization**: OHLC converted to USD per 100k lot for cross-instrument comparison
- **Wilder's ASI**: 50x multiplier, uncapped SI, dynamic L = 3×ATR calculation
- **Linear Angle Mapping**: Regression slopes mapped from degrees to [-1, +1] range
- **Direction Scaling**: tanh(ADX/25) for bounded [0, 1] directional strength
- **Volatility Transform**: Normalized arctan for bounded [0, 1] volatility levels
- **CSI (Reference Only)**: Wilder's authentic formula ADXR × ATR_pips × [V/√M × 1/(150+C)] × 100 with dynamic OANDA API integration for market ranking reference - NOT used in edge finding system

### 📊 Implementation Results (EUR_USD H1, 700 bars):
- ✅ **ASI**: 700 valid values, range [-523, 282] USD per 100k lot
- ✅ **HSP Angles**: 682 valid (97.4%), range [-0.981, 0.984] 
- ✅ **LSP Angles**: 680 valid (97.1%), range [-0.977, 0.980]
- ✅ **Direction**: 673 valid (96.1%), range [0.558, 0.986]
- ✅ **Volatility**: 188 valid (26.9%), range [0.227, 0.858]
- ✅ **Swing Points**: 140 HSP + 140 LSP detected
- ✅ **Cross-instrument ready** via USD normalization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate features from sample data (testing)
python3 scripts/generate_features.py --sample

# Process specific instrument
python3 scripts/generate_features.py --instrument EUR_USD

# Generate comprehensive 4-indicator system charts
python3 scripts/chart_all_indicators.py

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

# Optional: Enable dynamic CSI parameter fetching (reference only)
CSI_USE_DYNAMIC_API=false
```

## Architecture

- **TCNAE:** Temporal Convolutional Autoencoder for sequence compression (144→120 latent dimensions)
- **LightGBM:** Gradient boosting for final predictions (120 features → 24 predictions)
- **Context Tensor:** Cross-instrument information sharing (24×5 + 24 = 144 inputs)
- **24 FX Pairs:** EUR_USD, GBP_USD, USD_JPY, EUR_CAD, EUR_NZD, GBP_CAD, GBP_NZD, etc.
- **Features:** Complete 5-indicator system per instrument: HSP/LSP angles, Direction, Volatility, Price Change

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

## CSI Reference Implementation

### ⚠️ Important: CSI is Reference Only

The **Commodity Selection Index (CSI)** is implemented for **reference and market ranking purposes only**. It is **NOT used in the edge finding system** for predictions.

### CSI Features:
- **Wilder's Authentic Formula**: ADXR × ATR_pips × [V/√M × 1/(150+C)] × 100
- **99.1% Validation Accuracy**: Against Wilder's original Soybeans example (351.1 vs expected 348)
- **Dynamic OANDA API Integration**: Real-time margin requirements, pip values, commission costs
- **Static Configuration Fallback**: Reliable operation when API unavailable

### CSI Configuration Options:

```bash
# Enable dynamic OANDA API fetching (optional)
export CSI_USE_DYNAMIC_API=true

# Or configure programmatically
from configs.csi_config import set_dynamic_csi
set_dynamic_csi(True)

# Test CSI calculation differences
python3 scripts/compare_static_vs_dynamic_csi.py

# Demo dynamic CSI functionality  
python3 scripts/demo_dynamic_csi.py
```

**Static vs Dynamic Differences:**
- Margins: 16-265% higher with live API (EUR_USD: $2,332 vs $2,000)
- Commissions: 53-217% higher with real spreads (GBP_USD: $57 vs $18)
- CSI Values: ~17% different with dynamic parameters

## Dependencies

Core requirements in `requirements.txt` - uses only official libraries and follows strict production standards.