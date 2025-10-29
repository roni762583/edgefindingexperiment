# Market Edge Finder Experiment - Development Plan

## Project Overview

Research experiment to discover statistically significant edges in 1-hour FX movements using TCNAE + LightGBM across 24 currency pairs. Core question: does predictable structure exist that can be systematically exploited, or do markets remain efficient at this timeframe?

## Architecture Components

### 1. Data Pipeline
- **Historical Data Collection**: OANDA v20 API integration for 24 FX pairs
- **Real-time Data Stream**: Live M1 candle updates for feature computation
- **Data Storage**: Efficient storage with DuckDB/Parquet for large datasets
- **Data Validation**: Comprehensive quality checks and anomaly detection

### 2. Feature Engineering Pipeline
- **Causal Features**: 5 indicators per instrument (slope_high, slope_low, volatility, direction, price_change)
- **Cross-Instrument Context**: 24-instrument feature matrix for context tensor
- **Market Regime Detection**: 16-state framework (4 volatility × 4 direction states)
- **Temporal Windows**: 4-hour sequences (240 M1 candles) for TCNAE input
- **Incremental Updates**: Single function for live and historical processing consistency

### 3. Model Architecture
- **TCNAE (Stage 1)**: 4-hour sequence → 120-dim latent representation  
- **LightGBM (Stage 2)**: Latent features → 24 instrument predictions
- **Context Tensor**: Cross-instrument information sharing mechanism
- **Adaptive Teacher Forcing**: Dynamic blend of true vs predicted context

### 4. Training Pipeline
- **Stage 1**: TCNAE pretraining with reconstruction loss
- **Stage 2**: End-to-end TCNAE→LightGBM training
- **Stage 3**: Cooperative residual learning (optional)
- **Stage 4**: Adaptive teacher forcing optimization

### 5. Evaluation Framework
- **Walk-Forward Validation**: Time-series aware validation preventing data leakage
- **Performance Metrics**: Sharpe ratio (net transaction costs), Precision@k, Max drawdown
- **Regime Analysis**: Performance across 16 market states
- **Cross-Instrument Correlation**: Context tensor effectiveness measurement

## Development Phases

### Phase 1: Foundation (Weeks 1-2)
**Objective**: Establish robust data infrastructure and testing framework

#### Data Infrastructure
- [ ] **Historical Data Download System**
  - Parallel downloading for 24 FX pairs
  - Data validation and quality checks
  - Efficient storage format (Parquet/DuckDB)
  - Error handling and retry mechanisms

- [ ] **Real-time Data Pipeline**
  - OANDA streaming integration
  - M1 candle aggregation and buffering
  - Live feature computation
  - Data synchronization across instruments

#### Testing Framework
- [ ] **Unit Tests**: pytest with >90% coverage
- [ ] **Integration Tests**: End-to-end pipeline testing
- [ ] **Data Quality Tests**: Automated data validation
- [ ] **Performance Tests**: Latency and throughput benchmarks

#### Development Infrastructure
- [ ] **Docker Environment**: Multi-container setup (training/validation/live)
- [ ] **CI/CD Pipeline**: GitHub Actions with automated testing
- [ ] **Configuration Management**: Hydra-based config system
- [ ] **Logging System**: Structured logging with correlation IDs

### Phase 2: Feature Engineering (Weeks 3-4)
**Objective**: Implement causal feature extraction and market regime detection

#### Core Features (5 Indicators per Instrument)
- [ ] **Slope Calculation (High/Low Swing Points)**
  - High/low slope over multiple timeframes using regression
  - Robust estimation with outlier handling
  - Causal computation (no lookahead)
  - ASI-based swing point detection

- [ ] **Volatility Estimation (Dollar-Scaled ATR)**
  - USD-normalized ATR for economic comparability
  - Percentile scaling [0,1] with 200-bar rolling window
  - Cross-instrument volatility spillover analysis
  - Production-tested with correlation validation

- [ ] **Direction Detection (ADX-Based)**
  - ADX with percentile scaling [0,1] 
  - Multi-timeframe direction consensus
  - Momentum indicators with trend strength

- [x] **Price Change Indicator (5th Indicator)**
  - **Log returns**: `log(close[t] / close[t-1])` for normalized price changes
  - Real-time compatible with incremental updates
  - Provides scale-invariant measure of price movements
  - Integration with existing 4-indicator framework

#### Incremental Update Architecture
- [ ] **State Management**
  - Prior data structure for rolling calculations (EMA, percentiles, etc.)
  - Memory-efficient state preservation between updates
  - Consistent calculation between historical and live processing

- [x] **Single Update Function**
  - **Input**: New OHLC bar + prior calculation state 
  - **Output**: 5 indicators for current bar + updated state for next bar
  - **Updated State Includes**:
    - Rolling window buffers (200 ATR values, 200 ADX values for percentile scaling)
    - EMA values (ATR EMA, ADX components) for incremental calculation
    - Swing point history (confirmed HSP/LSP points for slope calculation)
    - Previous OHLC values (needed for price change and ASI calculation)
    - Bar count, pip size, and other metadata
  - **Purpose**: Preserve calculation memory between bars to enable live processing
  - **Design**: Prevent transcription errors between training and production

#### Market Regime Framework
- [ ] **16-State Classification**
  - Volatility regime detection (4 states)
  - Direction regime detection (4 states)
  - Combined state transitions
  - Regime persistence analysis

#### Feature Pipeline
- [ ] **Parallel Processing**: Multiprocessing for 24 instruments
- [ ] **Memory Management**: Efficient computation for large datasets
- [ ] **Feature Validation**: Statistical tests for feature quality
- [ ] **Feature Storage**: Optimized storage and retrieval

### Phase 3: TCNAE Implementation (Weeks 5-6)
**Objective**: Build and train temporal convolutional autoencoder

#### Model Architecture
- [ ] **Encoder Design**
  - 1D convolutional layers for temporal processing
  - Residual connections for gradient flow
  - Adaptive pooling for sequence compression
  - 240 M1 candles → 120-dim latent space

- [ ] **Decoder Design**
  - Transposed convolutions for reconstruction
  - Skip connections from encoder
  - Output matching original sequence dimensions

#### Training Infrastructure
- [ ] **Data Loaders**: Efficient batch loading for time series
- [ ] **Loss Functions**: Reconstruction loss with regularization
- [ ] **Optimization**: Adam optimizer with learning rate scheduling
- [ ] **Model Checkpointing**: Best model preservation and restoration

#### Validation
- [ ] **Reconstruction Quality**: Visual and quantitative assessment
- [ ] **Latent Space Analysis**: Dimensionality and clustering validation
- [ ] **Temporal Consistency**: Sequence preservation evaluation

### Phase 4: LightGBM Integration (Weeks 7-8)
**Objective**: Implement gradient boosting for prediction and integrate with TCNAE

#### Model Implementation
- [ ] **Feature Engineering**: Latent features + context tensor
- [x] **Target Engineering**: USD-scaled pip movements for 1-hour forward prediction
  - **Formula**: `(next_close - current_close) / pip_size × pip_value_usd`
  - **Examples**: EUR_USD 10 pips = +$100, GBP_JPY 12 pips = +$117.60
  - **Dynamic pip values**: Real-time calculation via OANDA API for cross-pairs
  - **Target scaling**: Normalize to [-1,1] using percentile scaling for stable ML training
- [ ] **Model Training**: LightGBM with cross-validation
- [ ] **Hyperparameter Optimization**: Optuna-based tuning

#### Hybrid Architecture
- [ ] **End-to-End Training**: TCNAE frozen, LightGBM training
- [ ] **Joint Optimization**: Fine-tuning both components
- [ ] **Loss Integration**: Combined reconstruction + prediction loss

#### Performance Optimization
- [ ] **Feature Importance**: Analysis of most predictive features
- [ ] **Model Compression**: Reducing model size for production
- [ ] **Inference Optimization**: Fast prediction pipeline

### Phase 5: Context Tensor System (Weeks 9-10)
**Objective**: Implement cross-instrument information sharing

#### Context Tensor Design
- [ ] **Architecture**: 24×24 instrument correlation matrix
- [ ] **Update Mechanism**: Real-time context computation
- [ ] **Memory Management**: Efficient tensor operations
- [ ] **Sparsity Handling**: Reduced computation for weak correlations

#### Adaptive Teacher Forcing
- [ ] **Correlation Tracking**: Model performance correlation monitoring
- [ ] **Blending Strategy**: Dynamic true/predicted context mixing
- [ ] **Feedback Loop**: Self-improving context quality

#### Cross-Instrument Learning
- [ ] **Shared Representations**: Common feature encodings
- [ ] **Transfer Learning**: Knowledge sharing between instruments
- [ ] **Ensemble Methods**: Multi-model predictions

### Phase 6: Production Pipeline (Weeks 11-12)
**Objective**: Build production-ready trading system

#### Real-time Inference
- [ ] **Live Feature Computation**: Real-time feature pipeline
- [ ] **Model Serving**: Fast inference with <100ms latency
- [ ] **Prediction Aggregation**: Multi-model ensemble
- [ ] **Risk Management**: Position sizing and risk controls

#### Monitoring and Alerting
- [ ] **Model Performance**: Real-time performance tracking
- [ ] **Data Quality**: Live data validation and alerts
- [ ] **System Health**: Infrastructure monitoring
- [ ] **Trade Execution**: Order management and reporting

#### Deployment
- [ ] **Containerization**: Production Docker containers
- [ ] **Orchestration**: Kubernetes deployment (optional)
- [ ] **Load Balancing**: High availability setup
- [ ] **Backup Systems**: Data and model backup strategies

## Incremental Processing Architecture

### Core Design Philosophy
**Single Source of Truth**: One function handles both historical training data processing and live production feeds to eliminate code divergence and calculation inconsistencies.

### Function Signature
```python
def update_indicators(
    new_ohlc: Dict[str, float],  # {'open': x, 'high': y, 'low': z, 'close': w}
    prior_state: IndicatorState,  # Previous calculation state
    instrument: str              # FX pair identifier
) -> Tuple[Dict[str, float], IndicatorState]:
    """
    Incremental update of 5 indicators with state preservation
    
    Returns:
        indicators: {'slope_high': x, 'slope_low': y, 'volatility': z, 'direction': w, 'price_change': v}
        updated_state: New state for next calculation
    """
```

### State Management Components

#### Combined State Architecture
```python
@dataclass
class InstrumentState:
    """State for individual instrument within combined structure"""
    # ASI/Swing Point State
    asi_value: float
    recent_highs: deque[Tuple[int, float]]  # (index, price) for swing detection
    recent_lows: deque[Tuple[int, float]]   # (index, price) for swing detection
    
    # ATR State (Dollar Scaling)
    atr_ema: float                          # Current ATR EMA value
    prev_close_usd: float                   # Previous close in USD
    atr_history: deque[float]               # 200-bar ATR history for percentiles
    
    # ADX State
    adx_value: float                        # Current ADX value
    di_plus_ema: float                      # +DI EMA
    di_minus_ema: float                     # -DI EMA
    tr_ema: float                           # True Range EMA
    adx_history: deque[float]               # 200-bar ADX history for percentiles
    
    # Price Change State
    prev_close: float                       # Previous close for change calculation
    price_change_history: deque[float]      # History for scaling (if needed)
    
    # General State
    bar_count: int                          # Total bars processed
    pip_size: float                         # Instrument pip size
    pip_value: float                        # USD value per pip

@dataclass
class MultiInstrumentState:
    """Combined state management for all 24 FX instruments"""
    instruments: Dict[str, InstrumentState] # Per-instrument states
    
    # Cross-instrument context tensor
    context_matrix: np.ndarray              # 24×5 feature matrix
    context_timestamp: pd.Timestamp        # Last update time
    
    # Global market regime state
    market_regime: int                      # Current regime (0-15)
    regime_history: deque[int]              # Regime transition history
```

**Advantages of Combined State**:
1. **Single File Persistence**: One state file for all 24 instruments vs 24 separate files
2. **Context Tensor Integration**: Natural fit with cross-instrument feature matrix
3. **Global Market Regime**: Shared regime state across all instruments
4. **Efficient I/O**: Single read/write operation for entire system state
5. **Memory Optimization**: Shared data structures and reduced overhead

### Rolling Calculation Approximations

#### EMA Updates (ATR, ADX components)
```python
# Exact incremental EMA update
alpha = 2.0 / (period + 1)
new_ema = alpha * new_value + (1 - alpha) * prev_ema
```

#### Percentile Scaling (200-bar rolling)
```python
# Efficient percentile with deque
history.append(new_value)
if len(history) > 200:
    history.popleft()
    
# Calculate percentile rank
sorted_values = sorted(history)
rank = bisect.bisect_left(sorted_values, new_value) / len(history)
```

#### Swing Point Detection (Causal)
```python
# Only confirm swing points after sufficient bars
# Maintain recent highs/lows in deque
# Calculate slopes only on confirmed swing points
```

### Processing Modes

#### Historical Batch Processing
```python
def process_historical_data(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Process entire historical dataset using incremental updates"""
    state = IndicatorState.initialize(instrument)
    indicators = []
    
    for _, row in df.iterrows():
        ohlc = row[['open', 'high', 'low', 'close']].to_dict()
        ind, state = update_indicators(ohlc, state, instrument)
        indicators.append(ind)
    
    return pd.DataFrame(indicators, index=df.index)
```

#### Live Production Processing
```python
def process_live_candle(ohlc: Dict, state: IndicatorState, instrument: str):
    """Process single live candle update"""
    indicators, new_state = update_indicators(ohlc, state, instrument)
    
    # Update persistent state storage
    save_state(instrument, new_state)
    
    # Send to ML pipeline
    return indicators
```

### Validation Framework

#### Consistency Testing
```python
def test_historical_vs_live_consistency():
    """Verify identical results between batch and incremental processing"""
    # Process historical data in batch
    batch_results = process_historical_data(df, instrument)
    
    # Process same data incrementally
    state = IndicatorState.initialize(instrument)
    incremental_results = []
    for _, row in df.iterrows():
        ohlc = row[['open', 'high', 'low', 'close']].to_dict()
        ind, state = update_indicators(ohlc, state, instrument)
        incremental_results.append(ind)
    
    # Assert identical results
    assert np.allclose(batch_results.values, incremental_results)
```

#### Monte Carlo Validation
Following Dr. Howard Bandy's methods from new_swt/micro/nano repository:
- Statistical significance testing of edge discovery
- Bootstrap sampling for robustness validation
- Walk-forward analysis with strict time-series splitting
- Out-of-sample testing with unseen market regimes

### Integration Points

#### Training Pipeline Integration
```python
# Historical data processing
for instrument in instruments:
    features_df = process_historical_data(historical_data[instrument], instrument)
    training_features[instrument] = features_df
```

#### Live Trading Integration
```python
# Real-time candle processing with combined state
@stream_handler
def on_new_candle(instrument: str, ohlc: Dict):
    global_state = load_combined_state()  # Single file load
    indicators, updated_state = update_indicators(ohlc, global_state, instrument)
    
    # Context tensor automatically updated within combined state
    # Generate predictions using full 24-instrument context
    predictions = model.predict(updated_state.context_matrix)
    
    # Persist updated state
    save_combined_state(updated_state)  # Single file save
```

## Quality Standards

### Code Quality
- **Type Hints**: All functions fully typed
- **Documentation**: Comprehensive docstrings (Google style)
- **Testing**: >90% code coverage with meaningful tests
- **Linting**: Black, isort, mypy, flake8 compliance
- **Security**: No hardcoded secrets, input validation

### Performance Requirements
- **Data Processing**: Handle 3+ years of M1 data for 24 instruments
- **Feature Computation**: <10 seconds for 240-candle sequence
- **Model Training**: <24 hours for full TCNAE+LightGBM training
- **Inference**: <100ms latency for live predictions
- **Memory Usage**: <16GB RAM for training, <4GB for inference

### Production Standards
- **Error Handling**: Graceful degradation and recovery
- **Logging**: Structured logging with correlation tracking
- **Configuration**: Environment-based config management
- **Monitoring**: Comprehensive metrics and alerting
- **Documentation**: Runbook and operational procedures

## Technology Stack

### Core ML/Data
- **PyTorch 2.0+**: TCNAE implementation
- **LightGBM 4.0+**: Gradient boosting
- **NumPy/Pandas**: Data manipulation
- **DuckDB**: High-performance analytics database
- **Polars**: Fast dataframes for large datasets

### Infrastructure
- **Docker/Docker Compose**: Containerization
- **pytest**: Testing framework
- **Hydra**: Configuration management
- **Loguru**: Structured logging
- **Rich**: Enhanced CLI output

### API/Networking
- **OANDA v20**: Official API library only
- **aiohttp**: Async HTTP client
- **asyncio**: Asynchronous programming

### Development Tools
- **Black**: Code formatting
- **mypy**: Type checking
- **GitHub Actions**: CI/CD pipeline
- **pre-commit**: Git hooks for quality

## Risk Management

### Technical Risks
- **Data Quality**: Comprehensive validation and monitoring
- **Model Overfitting**: Walk-forward validation and regularization
- **System Latency**: Performance optimization and monitoring
- **Infrastructure Failures**: Redundancy and graceful degradation

### Financial Risks
- **Transaction Costs**: Explicit cost modeling in evaluation
- **Market Regime Changes**: Adaptive model retraining
- **Correlation Breakdown**: Context tensor robustness testing
- **Black Swan Events**: Stress testing and position limits

## Success Criteria

### Edge Discovery Metrics
- **Statistical Significance**: p-value <0.05 for prediction accuracy
- **Information Coefficient**: IC >0.05 (correlation between predictions and actual returns)
- **Hit Rate**: >52% directional accuracy (statistically significant above random)
- **Sharpe Ratio**: >1.0 on prediction-based portfolio (no transaction costs, pure edge measurement)
- **Cross-Validation Consistency**: Edge persists across all time-based splits

### Research Validation Metrics
- **Out-of-Sample Performance**: Edge maintained on completely unseen data
- **Regime Robustness**: Predictive power across different market conditions
- **Cross-Instrument Generalization**: Context tensor improves predictions beyond single-instrument models
- **Feature Importance**: Clear attribution of edge sources to specific indicators
- **Monte Carlo Validation**: 6-scenario stress testing per new_swt methodology (bootstrap, adverse selection, early stopping, etc.)
- **Trajectory Analysis**: "Spaghetti plot" validation with confidence bands and percentile analysis
- **Cross-Scenario Robustness**: Edge persistence across all bootstrap stress scenarios

### Technical Implementation Metrics
- **Data Quality**: >99.5% clean data for all 24 instruments
- **Model Convergence**: Stable training across multiple runs
- **Code Quality**: >90% test coverage, production-ready implementation
- **Reproducibility**: Identical results across different environments

**Core Success Definition**: Discovery of measurable, statistically significant predictive edge in 1-hour FX movements that can be systematically captured through ML techniques. If no edge exists, the experiment successfully validates efficient market hypothesis for this timeframe.