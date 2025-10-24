# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.


# CLAUDE.md - STRICT PRODUCTION REQUIREMENTS

## 🚨 CRITICAL: MISSION-CRITICAL PRODUCTION SYSTEM

This is a **PRODUCTION-GRADE, MISSION-CRITICAL** trading system. There is **ZERO TOLERANCE** for shortcuts, assumptions, or incomplete implementations.

## 🏦 OANDA LIVE TRADING ENVIRONMENT

**MANDATORY**: This system uses **LIVE OANDA trading accounts** with **REAL MONEY**. 

### OANDA Configuration Requirements:
- **Environment**: `OANDA_ENVIRONMENT=live` (NOT practice)
- **API Endpoints**: Live trading endpoints only (api-fxtrade.oanda.com)
- **Official v20 Library**: Use ONLY the official OANDA v20 Python library (v20>=3.0.25.0)
- **Rate Limits**: Respect production rate limits for live accounts
- **Error Handling**: Fail fast on any API errors - no retries that could cause financial loss

### Live Trading Safety:
- **Data Download Only**: Initially used for historical data retrieval
- **No Automated Trading**: Trading decisions require explicit human authorization
- **Position Monitoring**: All positions must be tracked and logged
- **Risk Management**: Hard stops and position size limits mandatory

## ⚡ ABSOLUTE REQUIREMENTS
### 0. AVOID WRITTING NEW SCRIPTS, FIX/DEBUG EXISTING ONES!
SCRIPT CREATION POLICY - STRICT RULES

    AVOID CREATING NEW SCRIPTS - Always fix/debug/improve existing scripts first
    IF A NEW SCRIPT IS ABSOLUTELY NECESSARY:
        Temporary scripts (testing/debugging/one-off operations):
            MUST be placed in /tmp/ directory (create if needed)
            Name with descriptive prefix: temp_[purpose]_[timestamp].py
        Permanent scripts (essential project functionality):
            MUST inform user and get explicit confirmation BEFORE creation
            Explain WHY existing scripts cannot be modified
            Show exact location and purpose

### 1. FOLLOW INSTRUCTIONS EXACTLY
- **DO NOT** assume what the user wants
- **DO NOT** add features not explicitly requested
- **DO NOT** skip steps or take shortcuts
- Execute **EXACTLY** what is requested - nothing more, nothing less

### 2. VERIFICATION IS MANDATORY
- **NEVER** report completion without verification
- **ALWAYS** test and confirm functionality before claiming completion
- **ALWAYS** show concrete evidence of verification (output, test results, etc.)
- If you cannot verify, explicitly state "UNVERIFIED" in your response

### 3. CODE QUALITY STANDARDS
- **NO MONKEY PATCHES** - ever
- **NO STUB FUNCTIONS** - implement fully or not at all
- **NO PLACEHOLDER CODE** - every line must be production-ready
- **NO APPROXIMATIONS** - exact implementations only
- **NO SILENT FALLBACKS** - fail fast and loud

### 4. PRODUCTION CODE REQUIREMENTS
- **Type hints** on every function
- **Docstrings** with clear specifications
- **Error handling** that fails explicitly (no silent catches)
- **Input validation** on all boundaries
- **Logging** for critical operations
- **No magic numbers** - use named constants
- **DRY principle** - no code duplication
- **Single responsibility** - each function does ONE thing

### 5. FAILURE PHILOSOPHY
```python
# ❌ WRONG - Silent fallback
try:
    result = complex_calculation()
except:
    result = 0  # NO! Never approximate on failure

# ✅ CORRECT - Fail explicitly
try:
    result = complex_calculation()
except SpecificException as e:
    logger.error(f"Calculation failed: {e}")
    raise  # Let it crash - no silent failures
```

### 6. REPORTING STANDARDS
When reporting task completion:
1. State **EXACTLY** what was done
2. Show **VERIFICATION** output
3. List any **ASSUMPTIONS** made (should be zero)
4. Highlight any **DEVIATIONS** from request
5. Provide **TEST RESULTS** when applicable

### 7. IMPLEMENTATION CHECKLIST
Before claiming any task is complete:
- [ ] Code runs without errors
- [ ] All edge cases handled
- [ ] No TODOs or FIXMEs remain
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Error handling is explicit
- [ ] Code follows DRY principle
- [ ] No magic numbers
- [ ] Verified with actual execution

### 8. RED FLAGS - NEVER DO THESE

1. **"This should work"** - Test it or don't claim it
2. **"Approximately correct"** - Exact or nothing
3. **"You can add X later"** - Complete it now
4. **"This is a simple implementation"** - Make it production-ready
5. **"Fallback to default"** - Crash instead
6. **Silent exceptions** - All failures must be loud
7. **Untested code** - If you didn't run it, it's not done

## Always respond with concise, professional clarity.
## Explicitly outline steps or decisions.

## ✅ CORRECT APPROACH EXAMPLE

User: "Calculate the moving average"

❌ WRONG:
```python
def moving_avg(data, window):
    # Simple implementation
    return sum(data[-window:]) / window
```

✅ CORRECT:
```python
from typing import List, Union
import numpy as np

def calculate_moving_average(
    data: np.ndarray, 
    window: int,
    min_periods: int = 1
) -> np.ndarray:
    """
    Calculate simple moving average with proper handling.
    
    Args:
        data: Input array of numerical values
        window: Window size for moving average
        min_periods: Minimum number of observations required
        
    Returns:
        Array of moving averages
        
    Raises:
        ValueError: If window <= 0 or window > len(data)
        TypeError: If data is not numeric
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")
    if window > len(data):
        raise ValueError(f"Window {window} exceeds data length {len(data)}")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Data must be numeric, got {data.dtype}")
        
    # Use pandas for correct edge handling
    import pandas as pd
    return pd.Series(data).rolling(
        window=window, 
        min_periods=min_periods
    ).mean().values
```

## 🔍 VERIFICATION TEMPLATE

After completing any task:

```
TASK COMPLETED: [Exact description]

VERIFICATION:
- Executed: [command/function]
- Output: [actual output]
- Tests passed: [list specific tests]
- Edge cases verified: [list cases]

IMPLEMENTATION DETAILS:
- Files modified: [list files]
- Functions added: [list with signatures]
- Error handling: [describe approach]

DEVIATIONS: None [or list any]
```

## ⚠️ REMEMBER

**This is a PRODUCTION TRADING SYSTEM handling REAL MONEY. Every line of code matters. Every assumption is a potential financial loss. Every shortcut is a system failure waiting to happen.**

**FAIL FAST. FAIL LOUD. NEVER APPROXIMATE.**

---

## 🔒 PROTECTED CONTENT ABOVE - DO NOT MODIFY

**⚠️ CRITICAL**: Everything above this line is IMMUTABLE. Any modifications, additions, or updates MUST be appended below this section.

---
========================================
## 📝 PROJECT-SPECIFIC NOTES AND UPDATES
========================================

## Project Overview

This is a research project for developing a hybrid machine learning system that predicts 1-hour FX returns using a Temporal Convolutional Autoencoder (TCNAE) combined with LightGBM. The system focuses on edge discovery across 20 FX pairs using causal features and cross-instrument context.

## Key Architecture Components

- **TCNAE (Temporal Convolutional Autoencoder)**: Compresses 4-hour sequences of features into 100-dimensional latent representations
- **LightGBM/GBDT**: Maps latent features to 20 instrument predictions 
- **Context Tensor**: Maintains cross-instrument awareness and enables adaptive teacher forcing
- **Feature Engineering**: 4 causal indicators per instrument (slope_high, slope_low, volatility, direction)
- **16-State Market Regime Framework**: Captures volatility×direction and swing structure patterns

## Planned Directory Structure

When implementing this project, follow this structure:

```
market_edge_finder_experiment/
├── data/                    # OANDA historical data (raw/processed)
├── features/                # Feature engineering and normalization
├── models/                  # TCNAE, GBDT, context manager
├── data_pull/               # OANDA v20 API integration
├── training/                # Stage-wise training scripts
├── evaluation/              # Metrics and visualization
├── inference/               # Prediction pipeline
├── scripts/                 # Main execution scripts
├── configs/                 # Hyperparameters and data configs
└── utils/                   # Logger, file IO, timers
```

## Technology Stack

- Python 3.11+
- PyTorch (for TCNAE)
- LightGBM (for gradient boosting)
- **OANDA v20 API - OFFICIAL LIBRARY ONLY** (for historical data)
- Pandas/NumPy (data processing)
- Multiprocessing (parallel feature computation)
- Docker/Docker Compose (containerization)

## CRITICAL: OANDA API Requirements

**MUST USE OFFICIAL OANDA v20 LIBRARY ONLY:**
- Library: `v20>=3.0.25.0` 
- Documentation: https://oanda-api-v20.readthedocs.io/
- **NEVER use oandapyV20, python-oanda-v20, or any unofficial libraries**
- **ONLY use official OANDA v20 library and its official documentation**

## Development Commands

**Data Download & Processing:**
- **Primary Download**: `python download_real_data_v20.py` (saves to CSV first)
- Data preprocessing: `python scripts/run_preprocessing.py`
- Feature engineering: `python scripts/run_complete_data_pipeline.py`

**Training & Inference:**
- Training: `python scripts/run_training.py` 
- Inference: `python scripts/run_inference.py`
- Backtesting: `python scripts/backtest_edges.py`

**Docker:**
- Docker build: `docker-compose build`
- Docker run: `docker-compose up`

## Data Storage Strategy

1. **CSV First**: Download complete 3-year datasets as CSV files (human readable, debuggable)
2. **Verify Completeness**: Ensure all 26,280 hours (~3 years) downloaded successfully
3. **Convert to Parquet**: Only after verification, convert to Parquet for performance

**Verification Commands:**
```bash
ls -la data/raw/*.csv | wc -l  # Should show 20 files
wc -l data/raw/*.csv           # Should show ~26,280 lines each
```

## Training Pipeline

1. **Stage 1**: TCNAE pretraining with reconstruction loss
2. **Stage 2**: TCNAE → LightGBM hybrid training
3. **Stage 3**: Optional cooperative residual learning
4. **Stage 4**: Adaptive teacher forcing with data-driven α_t

### Adaptive Teacher Forcing Details

The system uses correlation-driven adaptive teacher forcing with the formula:

```python
# Data-driven correlation-based blending
α_t = min(1.0, max(0.0, correlation_measure))
C_{t-1}^{input} = (1 - α_t) * C_{t-1}^{true} + α_t * C_{t-1}^{pred}
```

Where:
- **α_t**: Adaptive blending coefficient based on model correlation performance
- **correlation_measure**: Rolling correlation between predictions and targets
- **C_{t-1}^{true}**: True context from actual market data
- **C_{t-1}^{pred}**: Predicted context from model outputs

This ensures stable training progression from teacher-forced (α_t=0) to fully autonomous (α_t=1) based on actual model performance rather than fixed schedules.

## Key Implementation Notes

- All features must be strictly causal (no lookahead bias)
- Use walk-forward validation to prevent data leakage
- Context tensor enables cross-instrument information sharing
- Adaptive teacher forcing blends true vs predicted context based on model correlation
- Multiprocessing parallelizes feature computation across 20 FX pairs
- Optional GPU support via PyTorch Metal (Mac) or CUDA

## Data Handling Requirements

### Storage Optimization
- **Primary Storage**: Parquet files with Snappy compression (~12-35MB for 3 years)
- **Metadata Management**: SQLite for run tracking, checksums, and orchestration
- **Partitioning**: By year/month for efficient read pruning in walk-forward validation
- **Compression Ratio**: ~2-6x with Parquet/Snappy vs raw binary data

### Data Quality & Alignment
- **UTC Alignment**: All timestamps must be converted to UTC for consistency
- **Missing Bar Handling**: Explicit interpolation or gap filling strategies
- **API Rate Limiting**: Exponential backoff and request throttling for OANDA API
- **Data Integrity**: MD5/SHA256 checksums stored in SQLite for each Parquet file
- **Atomic Writes**: Write to temp files then atomic move to prevent partial reads

### Error Handling Requirements
- **API Limits**: Handle OANDA rate limits with exponential backoff
- **Missing Data**: Explicit handling strategies, no silent approximations
- **Network Failures**: Resume capabilities with metadata tracking
- **Data Validation**: Strict validation of all OHLCV data before storage

## Evaluation Metrics

- Sharpe ratio (net of transaction costs)
- Precision@k 
- Maximum drawdown
- Regime persistence analysis
- Cross-instrument correlation decay

## Current Status

The project exists as a detailed specification document (`edge-finding-experiment.txt`) containing the complete technical design. No code has been implemented yet.

---

### Note Management Rules:
1. **ADD ONLY** 
- Never modify existing notes
2. **DATE ALL ENTRIES** - Use format: `[YYYY-MM-DD HH:MM]`
3. **ARCHIVE PERIODICALLY** - Move old notes to `CLAUDE_ARCHIVE_[DATE].md` when this section exceeds 100 lines
4. **MOST RECENT FIRST** - Add new entries at the top of this section
<!-- 🚨 ADD NEW NOTES BELOW THIS LINE 🚨-->

---

### [2025-10-22 23:30] EDGE FINDING EXPERIMENT - PRODUCTION SETUP COMPLETE

#### 🎯 PROJECT STATUS: READY FOR DEVELOPMENT
- **Project Name**: Market Edge Finder Experiment  
- **Architecture**: TCNAE + LightGBM hybrid system for FX edge discovery
- **Scope**: 20 FX pairs, 1-hour return prediction, cross-instrument context
- **Standards**: Production-grade, zero shortcuts, comprehensive testing

#### ✅ INFRASTRUCTURE COMPLETED
- **Directory Structure**: Full project layout with proper separation
- **OANDA Integration**: Official v20 API only, multi-instrument downloader  
- **Testing Framework**: pytest with fixtures, >90% coverage target
- **Docker Environment**: Multi-service architecture (data/training/validation/inference)
- **Development Plan**: 12-week roadmap with detailed phases

#### 🔧 DEVELOPMENT TOOLS READY
- **Quality**: Black, mypy, pytest, pre-commit hooks
- **Containers**: Production Dockerfile with multi-stage builds
- **Monitoring**: Health checks, resource limits, logging
- **Security**: Non-root containers, secret management

#### 📋 NEXT PHASE: DATA INFRASTRUCTURE (Week 1-2)
1. Historical data download for 20 FX pairs
2. Data validation and quality checks  
3. Feature engineering pipeline setup
4. Testing framework expansion

#### 🚨 CRITICAL REMINDERS
- **OANDA API**: Official v20>=3.0.25.0 library ONLY
- **No Shortcuts**: Production-ready code from day one
- **Verification**: All tasks require concrete evidence
- **Transaction Costs**: Sharpe ratio must be net of costs

#### 📚 REFERENCE MATERIALS
REVIEW /Users/shmuelzbaida/Desktop/Aharon2025/new_muzero/README.md , and
/Users/shmuelzbaida/Desktop/Aharon2025/edgefindingexperiment/README.md
PLAN:
    Follow user instructions