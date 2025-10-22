# Edge Finding Experiment - Market Edge Finder 2

Hybrid machine learning system for 1-hour FX return prediction using TCNAE + LightGBM across 20 currency pairs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

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
- **Features:** 4 causal indicators per instrument (slope_high, slope_low, volatility, direction)

## Data Pipeline

1. **Historical Download:** `scripts/download_all_fx_data.py`
2. **Feature Engineering:** `features/` (planned)
3. **Training:** `training/` (planned)
4. **Inference:** `inference/` (planned)

## Current Status

✅ Directory structure created  
✅ OANDA data downloader adapted  
✅ Multi-instrument parallel downloader  
⏳ Feature engineering pipeline  
⏳ TCNAE model implementation  
⏳ LightGBM integration  
⏳ Context tensor system  

## Dependencies

Core requirements in `requirements.txt` - uses only official libraries and follows strict production standards.