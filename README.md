# Market State Matrix – Edge Finding v2

## Overview
A hybrid supervised learning framework for detecting short-term, regime-dependent predictive edges in 20 FX pairs on a 1-hour horizon. Combines a Temporal Convolutional Autoencoder (TCNAE) with LightGBM (GBDT) to capture both temporal patterns and nonlinear decision surfaces.

## Features
- 4 causal indicators per instrument: slope_high, slope_low, volatility, direction
- 16-state market regime framework
- Global context tensor for cross-instrument awareness
- Hybrid TCNAE → LightGBM with optional cooperative residual learning
- Adaptive teacher forcing with data-driven α_t based on model correlation
- Walk-forward evaluation metrics: Sharpe, Precision@k, Max Drawdown, regime persistence

## Tech Stack
- Python 3.11+
- PyTorch (TCNAE)
- LightGBM / XGBoost (GBDT)
- Pandas / NumPy
- Matplotlib / Plotly
- OANDA v20 API
- Multiprocessing for parallel computation
- Docker / Docker Compose
- Optional GPU (Mac M1/M2 via PyTorch Metal or local CUDA)

## Directory Structure
```
market_edge_finder_experiment/
├── data/                    # Raw and processed datasets (~54MB total, ~12-35MB compressed)
├── features/                # Feature engineering and normalization
├── models/                  # TCNAE, GBDT, context manager
├── data_pull/               # OANDA v20 API integration
├── training/                # Stage-wise training scripts
├── evaluation/              # Metrics and visualization
├── inference/               # Real-time prediction pipeline
├── scripts/                 # Main execution scripts
├── configs/                 # Configuration management
├── utils/                   # Logger, file IO, timers
├── tests/                   # Comprehensive test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── USAGE.md                 # Detailed usage guide
```

## Installation

### Local Installation
```bash
# 1. Clone repository
git clone https://github.com/roni762583/edgefindingexperiment.git
cd edgefindingexperiment/market_edge_finder_experiment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure OANDA credentials
cp .env.example .env
# Edit .env with your OANDA API key and account ID
```

### Docker Installation
```bash
# Build Docker image (optional GPU)
docker-compose build
# For GPU support: USE_CUDA=true docker-compose build

# Set environment variables for OANDA
export OANDA_API_KEY=<your_api_key>
export OANDA_ACCOUNT_ID=<your_account_id>
export OANDA_ENVIRONMENT=practice  # or 'live'
```

## Quick Start

### Basic Pipeline
```bash
# 1. Data preprocessing (download + feature engineering)
python scripts/run_preprocessing.py

# 2. Train hybrid TCNAE + LightGBM model
python scripts/run_training.py

# 3. Evaluate model performance
python scripts/run_evaluation.py

# 4. Run real-time inference
python scripts/run_inference.py
```

### Docker Deployment
```bash
# Start complete production pipeline
docker-compose up -d

# Monitor services
docker-compose ps
docker-compose logs -f

# Development environment
docker-compose --profile dev up -d
```

## Configuration

### Environment Variables (.env file)
```bash
# OANDA API Configuration
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice  # or 'live'

# System Configuration
ENVIRONMENT=development  # development, staging, production
DEVICE=auto  # auto, cpu, cuda, mps
LOG_LEVEL=INFO

# Optional Paths
DATA_PATH=./data
MODELS_PATH=./models
LOGS_PATH=./logs
RESULTS_PATH=./results
```

### Configuration Files
The system uses YAML configuration files for detailed parameter control:
- `configs/default_config.yaml` - Default settings
- `configs/production_config.yaml` - Production optimizations
- Custom configurations supported

## Key Components

### 1. Data Processing
- **Storage**: Parquet files with Snappy compression (~12-35MB for 3 years of data)
- **Metadata**: SQLite for run tracking and orchestration
- **Features**: 4 causal indicators per instrument (100 features/hour total)
- **UTC Alignment**: Proper timezone handling and missing bar management

### 2. Hybrid Model Architecture
```
OANDA Data → Feature Engineering → TCNAE (400→100) → LightGBM (100→20) → Predictions
                                     ↑                                         ↓
                              Context Tensor ←────────────────────────────────
```

### 3. Training Pipeline (4 Stages)
1. **Stage 1**: TCNAE pretraining with reconstruction loss
2. **Stage 2**: TCNAE → LightGBM hybrid training
3. **Stage 3**: Cooperative residual learning (optional)
4. **Stage 4**: Adaptive teacher forcing with data-driven α_t

### 4. Adaptive Teacher Forcing
```python
# Data-driven correlation-based blending
α_t = min(1.0, max(0.0, correlation_measure))
context_input = (1 - α_t) * true_context + α_t * predicted_context
```

## Data Specifications

### Dataset Size (3 years, 20 instruments)
- **Total bars**: 374,400 (18,720 per instrument)
- **Raw OHLCV**: ~24MB binary, ~72MB CSV
- **Features**: ~15MB (100 features × 18,720 timesteps)
- **Latents**: ~15MB (100-dim latent space)
- **Total compressed**: ~12-35MB with Parquet/Snappy

### Instruments (20 FX Pairs)
EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD, EUR_GBP, EUR_JPY, GBP_JPY, AUD_JPY, EUR_CHF, GBP_CHF, CHF_JPY, EUR_AUD, GBP_AUD, AUD_CHF, NZD_JPY, CAD_JPY, AUD_NZD

## Usage Examples

### Custom Training
```bash
# Custom configuration
python scripts/run_training.py --config configs/custom_config.yaml

# Override parameters
python scripts/run_training.py --epochs 200 --batch-size 64 --device cuda

# GPU training
python scripts/run_training.py --device cuda
```

### Real-time Inference
```bash
# Health check first
python scripts/run_inference.py --health-check

# Start inference with monitoring
python scripts/run_inference.py --log-level DEBUG

# Production daemon mode
python scripts/run_inference.py --daemon
```

### Evaluation and Analysis
```bash
# Generate comprehensive evaluation report
python scripts/run_evaluation.py --output-dir ./results

# Custom evaluation metrics
python scripts/run_evaluation.py --config configs/evaluation_config.yaml
```

## Production Features

### Error Handling & Recovery
- API rate limit management with exponential backoff
- Missing data interpolation and gap handling
- Graceful shutdown with signal handling
- Comprehensive logging and monitoring

### Performance Optimization
- Multiprocessing for parallel feature computation
- Mixed precision training support
- Memory-efficient data pipelines
- Optional GPU acceleration

### Monitoring & Observability
- Health checks for all services
- Performance metrics tracking
- Error logging and alerting
- System resource monitoring

## Testing

```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_inference.py -v

# Coverage report
python -m pytest tests/ --cov=market_edge_finder_experiment --cov-report=html
```

## Notes

- **Causal Features**: All indicators maintain strict temporal causality (no lookahead bias)
- **Cross-instrument Context**: Context tensor enables information sharing between FX pairs
- **Walk-forward Validation**: Proper temporal splits prevent data leakage
- **Production Ready**: Full Docker orchestration with resource limits and health checks
- **OANDA v20 Only**: Uses official OANDA v20 library exclusively

## Contributions

- All contributions must maintain causal, lagged computations and strictly prevent data leakage
- Incremental updates to LightGBM residuals should follow the cooperative hybrid loop protocol
- Code must pass all tests and include proper type hints and documentation

## License

MIT License - See LICENSE file for details

## Support

For detailed usage instructions, see [USAGE.md](USAGE.md)

For issues and support, please use the GitHub issue tracker.