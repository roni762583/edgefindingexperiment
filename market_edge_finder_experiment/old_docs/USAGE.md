# Market Edge Finder Experiment - Usage Guide

This guide provides comprehensive instructions for using the Market Edge Finder hybrid machine learning system for FX edge discovery.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Real-time Inference](#real-time-inference)
8. [Docker Deployment](#docker-deployment)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.11+
- OANDA v20 API credentials
- Minimum 8GB RAM
- GPU recommended (CUDA or Metal)

### Basic Setup

```bash
# 1. Clone repository
git clone https://github.com/roni762583/edgefindingexperiment.git
cd edgefindingexperiment/market_edge_finder_experiment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure OANDA credentials
cp .env.example .env
# Edit .env with your OANDA API key and account ID

# 4. Run complete pipeline
python scripts/run_preprocessing.py
python scripts/run_training.py
python scripts/run_evaluation.py
python scripts/run_inference.py
```

## Installation

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Docker Installation

```bash
# Build Docker image
docker build -t market-edge-finder .

# Run with Docker Compose
docker-compose up -d
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OANDA API Configuration
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice  # or 'live'

# System Configuration
ENVIRONMENT=development  # development, staging, production
DEVICE=auto  # auto, cpu, cuda, mps
LOG_LEVEL=INFO

# Paths (optional)
DATA_PATH=./data
MODELS_PATH=./models
LOGS_PATH=./logs
RESULTS_PATH=./results
```

### Configuration Files

The system uses YAML configuration files. See `configs/default_config.yaml` for all available options:

```yaml
# Model Configuration
model:
  tcnae_latent_dim: 100
  num_instruments: 20
  device: "auto"

# Training Configuration  
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001

# Data Configuration
data:
  granularity: "H1"
  lookback_days: 365
  instruments:
    - "EUR_USD"
    - "GBP_USD"
    # ... more instruments
```

## Data Preprocessing

### Basic Preprocessing

```bash
# Download and preprocess data for all instruments
python scripts/run_preprocessing.py

# Custom date range
python scripts/run_preprocessing.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01

# Force redownload
python scripts/run_preprocessing.py --force-redownload

# Custom configuration
python scripts/run_preprocessing.py --config configs/custom_config.yaml
```

### Preprocessing Options

- `--start-date`: Start date for data download (YYYY-MM-DD)
- `--end-date`: End date for data download (YYYY-MM-DD)  
- `--force-redownload`: Force redownload even if data exists
- `--config`: Path to custom configuration file
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Output Files

Preprocessing creates the following files in the `data/` directory:

```
data/
├── EUR_USD_raw.parquet          # Raw OHLC data
├── EUR_USD_features.parquet     # Generated features
├── EUR_USD_normalized.parquet   # Normalized features
├── EUR_USD_targets.parquet      # Target returns
├── train/                       # Training splits
│   ├── EUR_USD.parquet
│   └── ...
├── validation/                  # Validation splits
│   ├── EUR_USD.parquet
│   └── ...
└── preprocessing_summary.json   # Summary report
```

## Model Training

### Basic Training

```bash
# Train hybrid TCNAE + LightGBM model
python scripts/run_training.py

# Custom configuration
python scripts/run_training.py --config configs/training_config.yaml

# Override parameters
python scripts/run_training.py \
    --epochs 200 \
    --batch-size 64 \
    --device cuda
```

### Training Stages

The system uses a 4-stage training pipeline:

1. **Stage 1**: TCNAE pretraining with reconstruction loss
2. **Stage 2**: TCNAE → LightGBM hybrid training  
3. **Stage 3**: Cooperative residual learning
4. **Stage 4**: Adaptive teacher forcing with context tensor

### Training Options

- `--config`: Path to configuration file
- `--device`: Override device (cpu, cuda, mps)
- `--epochs`: Override number of epochs
- `--batch-size`: Override batch size
- `--log-level`: Logging verbosity

### Output Files

Training creates these files in the `models/` directory:

```
models/
├── tcnae_best.pth           # TCNAE model weights
├── gbdt_best.pkl            # LightGBM model
├── context_manager.pkl      # Context tensor state
├── normalizer.pkl           # Feature normalizer
└── results/
    └── training_results_*.json  # Training metrics
```

## Model Evaluation

### Basic Evaluation

```bash
# Evaluate trained models
python scripts/run_evaluation.py

# Custom output directory
python scripts/run_evaluation.py --output-dir ./custom_results

# Detailed logging
python scripts/run_evaluation.py --log-level DEBUG
```

### Evaluation Metrics

The system calculates comprehensive metrics:

- **Sharpe Ratios**: Risk-adjusted returns net of transaction costs
- **Precision@k**: Top-k prediction accuracy  
- **Maximum Drawdown**: Worst-case portfolio decline
- **Regime Analysis**: Performance across volatility regimes
- **Cross-instrument Correlation**: Correlation decay patterns

### Evaluation Output

```
results/
├── evaluation_results_*.json    # Metrics data
├── evaluation_*/               # Visualization directory
│   ├── sharpe_ratios.png
│   ├── precision_at_k.png
│   ├── maximum_drawdown.png
│   ├── regime_analysis.png
│   └── correlation_decay.png
└── evaluation_summary.txt      # Text summary
```

## Real-time Inference

### Basic Inference

```bash
# Start real-time inference
python scripts/run_inference.py

# Health check first
python scripts/run_inference.py --health-check

# Run as daemon
python scripts/run_inference.py --daemon
```

### Inference Features

- **Real-time Data**: Live OANDA v20 API streaming
- **Context Management**: Cross-instrument awareness
- **Safety Limits**: Position and exposure constraints
- **Graceful Shutdown**: Signal handling for clean stops
- **Health Monitoring**: System status reporting

### Inference Output

The inference system outputs:

- Live predictions every 5 minutes (configurable)
- Trading signals with confidence scores
- System health metrics
- Performance monitoring logs

Example prediction output:

```json
{
  "predictions": {
    "EUR_USD": 0.0023,
    "GBP_USD": -0.0015,
    "USD_JPY": 0.0008
  },
  "confidence_scores": {
    "EUR_USD": 0.756,
    "GBP_USD": 0.642,
    "USD_JPY": 0.423
  },
  "metadata": {
    "timestamp": "2024-01-15T14:30:00",
    "total_exposure": 3.45,
    "high_confidence_count": 12
  }
}
```

## Docker Deployment

### Development Environment

```bash
# Start development environment
docker-compose --profile dev up -d

# Access development container
docker-compose exec dev-service bash

# Run Jupyter notebook
docker-compose exec dev-service jupyter lab --ip=0.0.0.0 --port=8888
```

### Production Deployment

```bash
# Start full production pipeline
docker-compose up -d

# Monitor services
docker-compose ps
docker-compose logs -f

# Scale specific services
docker-compose up -d --scale inference-service=3
```

### Service Architecture

- **data-service**: Data download and preprocessing
- **training-service**: Model training with GPU support
- **validation-service**: Model evaluation and validation
- **inference-service**: Real-time prediction API
- **monitoring-service**: Grafana dashboards (optional)
- **database**: PostgreSQL for metadata (optional)

### Production Configuration

```yaml
# docker-compose.override.yml for production
version: '3.8'
services:
  inference-service:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    environment:
      - OANDA_ENVIRONMENT=live
```

## API Reference

### Configuration Classes

```python
from configs.config import SystemConfig, ModelConfig, TrainingConfig

# Load configuration
config = load_config('configs/production_config.yaml')

# Access configuration sections
model_config = config.model
training_config = config.training
data_config = config.data
```

### Feature Engineering

```python
from features.feature_engineering import FeatureEngineer
from features.normalization import FeatureNormalizer

# Generate features
engineer = FeatureEngineer()
features = engineer.generate_features_single_instrument(data, 'EUR_USD')

# Normalize features
normalizer = FeatureNormalizer()
normalized = normalizer.normalize_single_instrument(features, 'EUR_USD')
```

### Model Training

```python
from training.train_hybrid import HybridTrainer
from models.tcnae import TCNAEConfig
from models.gbdt_model import GBDTConfig

# Initialize trainer
trainer = HybridTrainer(tcnae_config, gbdt_config, training_config)

# Train models
results = trainer.train_full_pipeline(data_loader)
```

### Inference

```python
from inference.predictor import EdgePredictor, PredictionConfig

# Initialize predictor
predictor = EdgePredictor(config)
predictor.initialize_models()

# Generate predictions
predictions = predictor.generate_predictions()
```

### Evaluation

```python
from evaluation.metrics import TradingMetricsCalculator
from evaluation.visualization import EvaluationVisualizer

# Calculate metrics
calculator = TradingMetricsCalculator()
results = calculator.generate_comprehensive_report(predictions, returns, prices)

# Create visualizations
visualizer = EvaluationVisualizer()
figures = visualizer.create_comprehensive_dashboard(results)
```

## Troubleshooting

### Common Issues

#### OANDA API Connection

```bash
# Test API connection
python -c "
from data_pull.oanda_v20_connector import OANDAConnector
connector = OANDAConnector('your_key', 'your_account', 'practice')
print('Connection successful')
"
```

#### Memory Issues

```bash
# Reduce batch size in config
training:
  batch_size: 16  # Reduce from 32

# Or use environment variable
export BATCH_SIZE=16
python scripts/run_training.py
```

#### GPU Issues

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force CPU mode
python scripts/run_training.py --device cpu
```

#### Missing Dependencies

```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check for conflicts
pip check
```

### Logging and Debugging

#### Enable Debug Logging

```bash
# Global debug logging
python scripts/run_training.py --log-level DEBUG

# Component-specific logging
LOG_LEVEL=DEBUG python scripts/run_training.py
```

#### Log File Locations

```
logs/
├── preprocessing.log    # Data preprocessing logs
├── training.log         # Training logs  
├── evaluation.log       # Evaluation logs
├── inference.log        # Inference logs
├── preprocessing_errors.log  # Error-only logs
├── training_errors.log
├── evaluation_errors.log
└── inference_errors.log
```

#### Health Checks

```bash
# Check model files
python scripts/run_inference.py --health-check

# Check data files
ls -la data/

# Check configuration
python -c "from configs.config import load_config; print(load_config())"
```

### Performance Optimization

#### Training Optimization

```yaml
# Mixed precision training
training:
  mixed_precision: true
  gradient_clip_norm: 1.0

# Larger batch sizes with gradient accumulation
training:
  batch_size: 64
  accumulation_steps: 2
```

#### Inference Optimization

```yaml
# Batch inference
inference:
  batch_inference: true
  confidence_threshold: 0.1

# Reduced update frequency
inference:
  update_interval_seconds: 300  # 5 minutes
```

#### Memory Optimization

```yaml
# Reduce model size
model:
  tcnae_latent_dim: 64      # Reduce from 100
  tcnae_hidden_channels: [16, 32, 64]  # Smaller networks

# Reduce data retention
data:
  buffer_size_hours: 24     # Reduce from 48
```

### Support

For additional support:

1. Check the [GitHub Issues](https://github.com/roni762583/edgefindingexperiment/issues)
2. Review the comprehensive logs in the `logs/` directory
3. Run health checks with `--health-check` flag
4. Validate configuration with debug logging

### Version Information

```bash
# Check system versions
python --version
pip list | grep -E "(torch|lightgbm|pandas|numpy)"

# Check model versions
python -c "
import json
with open('models/training_results_*.json') as f:
    config = json.load(f)
    print(f'Model version: {config.get(\"version\", \"unknown\")}')
"
```