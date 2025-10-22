"""
Pytest configuration and shared fixtures for Market Edge Finder Experiment
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
import tempfile
import os

# Add project root to Python path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.instruments import FX_INSTRUMENTS


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root) -> Path:
    """Get test data directory"""
    test_dir = project_root / "tests" / "data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    n_candles = 1000
    
    # Generate realistic FX price data
    base_price = 1.3000  # EUR_USD starting price
    price_changes = np.random.normal(0, 0.0001, n_candles)
    
    # Create OHLCV data
    closes = base_price + np.cumsum(price_changes)
    opens = np.roll(closes, 1)
    opens[0] = base_price
    
    # Generate realistic high/low spreads
    spreads = np.random.uniform(0.0002, 0.0008, n_candles)
    highs = np.maximum(opens, closes) + spreads / 2
    lows = np.minimum(opens, closes) - spreads / 2
    
    # Generate volume
    volumes = np.random.randint(100, 1000, n_candles)
    
    # Create timestamps
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_candles)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


@pytest.fixture
def sample_multiinstrument_data() -> Dict[str, pd.DataFrame]:
    """Generate sample data for multiple instruments"""
    np.random.seed(42)
    data = {}
    
    # Base prices for different instruments
    base_prices = {
        'EUR_USD': 1.3000,
        'GBP_USD': 1.6000,
        'USD_JPY': 110.00,
        'GBP_JPY': 176.00
    }
    
    for instrument in ['EUR_USD', 'GBP_USD', 'USD_JPY', 'GBP_JPY']:
        n_candles = 500
        base_price = base_prices[instrument]
        
        # Adjust volatility for JPY pairs
        vol_scale = 0.01 if 'JPY' in instrument else 0.0001
        price_changes = np.random.normal(0, vol_scale, n_candles)
        
        closes = base_price + np.cumsum(price_changes)
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        spreads = np.random.uniform(vol_scale * 2, vol_scale * 8, n_candles)
        highs = np.maximum(opens, closes) + spreads / 2
        lows = np.minimum(opens, closes) - spreads / 2
        
        volumes = np.random.randint(100, 1000, n_candles)
        
        start_time = datetime(2023, 1, 1)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_candles)]
        
        data[instrument] = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    return data


@pytest.fixture
def sample_features_data() -> pd.DataFrame:
    """Generate sample feature data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate 4 causal features per instrument
    features = {}
    for instrument in FX_INSTRUMENTS[:5]:  # Use first 5 instruments for testing
        features[f'{instrument}_slope_high'] = np.random.normal(0, 0.1, n_samples)
        features[f'{instrument}_slope_low'] = np.random.normal(0, 0.1, n_samples)
        features[f'{instrument}_volatility'] = np.random.exponential(0.5, n_samples)
        features[f'{instrument}_direction'] = np.random.choice([-1, 0, 1], n_samples)
    
    # Add timestamps
    start_time = datetime(2023, 1, 1)
    features['timestamp'] = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    return pd.DataFrame(features)


@pytest.fixture
def mock_oanda_api_response():
    """Mock OANDA API response for testing"""
    return {
        'candles': [
            {
                'time': '2023-01-01T00:00:00.000000000Z',
                'complete': True,
                'volume': 100,
                'mid': {
                    'o': '1.3000',
                    'h': '1.3005',
                    'l': '1.2995',
                    'c': '1.3002'
                }
            },
            {
                'time': '2023-01-01T00:01:00.000000000Z',
                'complete': True,
                'volume': 150,
                'mid': {
                    'o': '1.3002',
                    'h': '1.3008',
                    'l': '1.2998',
                    'c': '1.3005'
                }
            }
        ]
    }


@pytest.fixture
def temp_env_file():
    """Create temporary environment file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("OANDA_API_KEY=test_api_key\n")
        f.write("OANDA_ACCOUNT_ID=test_account_id\n")
        f.write("OANDA_ENVIRONMENT=practice\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing"""
    return {
        'tcnae': {
            'input_dim': 4,  # OHLC
            'latent_dim': 100,
            'sequence_length': 240,  # 4 hours of M1 data
            'num_layers': 3,
            'kernel_size': 3,
            'dropout': 0.1
        },
        'lightgbm': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'early_stopping': 10,
            'validation_split': 0.2
        }
    }


@pytest.fixture
def sample_sequence_data():
    """Generate sample sequence data for TCNAE testing"""
    np.random.seed(42)
    batch_size = 16
    sequence_length = 240
    num_features = 4  # OHLC
    
    # Generate realistic price sequences
    sequences = []
    for _ in range(batch_size):
        base_price = np.random.uniform(1.0, 2.0)
        price_changes = np.random.normal(0, 0.0001, sequence_length)
        closes = base_price + np.cumsum(price_changes)
        
        # Create OHLC from closes
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        spreads = np.random.uniform(0.0001, 0.0005, sequence_length)
        highs = np.maximum(opens, closes) + spreads / 2
        lows = np.minimum(opens, closes) - spreads / 2
        
        sequence = np.stack([opens, highs, lows, closes], axis=-1)
        sequences.append(sequence)
    
    return np.array(sequences)


@pytest.fixture(scope="session")
def test_instruments():
    """Subset of instruments for testing"""
    return ['EUR_USD', 'GBP_USD', 'USD_JPY', 'GBP_JPY']


@pytest.fixture
def market_regime_data():
    """Generate sample market regime data"""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'volatility_regime': np.random.choice([0, 1, 2, 3], n_samples),
        'direction_regime': np.random.choice([0, 1, 2, 3], n_samples),
        'combined_regime': np.random.choice(range(16), n_samples),
        'regime_confidence': np.random.uniform(0.5, 1.0, n_samples)
    })


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )
    config.addinivalue_line(
        "markers", "model: marks tests for model components"
    )
    config.addinivalue_line(
        "markers", "data: marks tests for data processing"
    )


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add slow marker to tests that take >5 seconds
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add api marker to tests that use OANDA API
        if "oanda" in str(item.fspath).lower() or "api" in str(item.fspath).lower():
            item.add_marker(pytest.mark.api)