"""
Configuration management for Market Edge Finder Experiment.

Centralized configuration system with environment-specific settings,
validation, and easy parameter access across all components.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OANDAConfig:
    """OANDA API configuration."""
    api_key: str
    account_id: str
    environment: str = 'practice'  # 'practice' or 'live'
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OANDA API key is required")
        if not self.account_id:
            raise ValueError("OANDA account ID is required")
        if self.environment not in ['practice', 'live']:
            raise ValueError("Environment must be 'practice' or 'live'")


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # TCNAE parameters
    tcnae_input_channels: int = 4
    tcnae_sequence_length: int = 4
    tcnae_latent_dim: int = 100
    tcnae_hidden_channels: List[int] = None
    tcnae_kernel_size: int = 3
    tcnae_dropout: float = 0.2
    
    # LightGBM parameters
    gbdt_n_estimators: int = 1000
    gbdt_max_depth: int = 8
    gbdt_learning_rate: float = 0.05
    gbdt_subsample: float = 0.8
    gbdt_colsample_bytree: float = 0.8
    gbdt_reg_alpha: float = 0.1
    gbdt_reg_lambda: float = 0.1
    
    # Context manager parameters
    context_dim: int = 100
    context_ema_alpha: float = 0.1
    context_correlation_window: int = 100
    
    # General parameters
    num_instruments: int = 20
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    
    def __post_init__(self):
        if self.tcnae_hidden_channels is None:
            self.tcnae_hidden_channels = [32, 64, 128]


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Instruments
    instruments: List[str] = None
    
    # Data parameters
    granularity: str = 'H1'  # Hourly data
    lookback_days: int = 365  # Days of historical data
    
    # Feature engineering
    atr_period: int = 14
    adx_period: int = 14
    volatility_window: int = 24
    direction_window: int = 6
    swing_lookback: int = 12
    
    # Normalization
    normalization_method: str = 'rolling_zscore'
    normalization_window: int = 168  # 1 week
    outlier_method: str = 'iqr'
    outlier_threshold: float = 2.0
    regime_window: int = 720  # 30 days
    
    # Processing
    max_workers: int = 4
    batch_size: int = 32
    
    def __post_init__(self):
        if self.instruments is None:
            self.instruments = [
                'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD',
                'USD_CAD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY',
                'AUD_JPY', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'EUR_AUD',
                'GBP_AUD', 'AUD_CHF', 'NZD_JPY', 'CAD_JPY', 'AUD_NZD'
            ]


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Training stages
    stage1_epochs: int = 50  # TCNAE pretraining
    stage2_epochs: int = 30  # Hybrid training
    stage3_epochs: int = 15  # Cooperative learning
    stage4_epochs: int = 5   # Adaptive teacher forcing
    
    # Validation
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Optimization
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step'
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_n_checkpoints: int = 5
    
    # Walk-forward validation
    train_window_months: int = 12
    test_window_months: int = 1
    step_size_months: int = 1


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    # Model paths
    model_base_path: str = "./models"
    tcnae_model_name: str = "tcnae_best.pth"
    gbdt_model_name: str = "gbdt_best.pkl"
    context_manager_name: str = "context_manager.pkl"
    normalizer_name: str = "normalizer.pkl"
    
    # Prediction parameters
    confidence_threshold: float = 0.1
    max_prediction_age_minutes: int = 60
    update_interval_seconds: int = 300  # 5 minutes
    
    # Safety limits
    max_position_size: float = 1.0
    max_total_exposure: float = 10.0
    
    # Real-time data
    data_buffer_hours: int = 48
    data_update_interval: int = 60  # seconds
    
    # Performance
    batch_inference: bool = True
    use_context: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Component-specific logging
    model_logging: bool = True
    data_logging: bool = True
    training_logging: bool = True
    inference_logging: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""
    
    oanda: OANDAConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig
    logging: LoggingConfig
    
    # Environment
    environment: str = 'development'  # 'development', 'staging', 'production'
    
    # Paths
    data_path: str = "./data"
    models_path: str = "./models"
    logs_path: str = "./logs"
    results_path: str = "./results"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.data_path, self.models_path, self.logs_path, self.results_path]:
            Path(path).mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """
    Configuration manager for loading, validating, and managing configurations.
    
    Supports loading from environment variables, YAML files, and JSON files
    with environment-specific overrides and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        
    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from file and environment variables.
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Complete system configuration
        """
        config_path = config_path or self.config_path
        
        # Start with default configuration
        config_dict = self._get_default_config()
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            file_config = self._load_config_file(config_path)
            config_dict = self._merge_configs(config_dict, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config_dict = self._merge_configs(config_dict, env_config)
        
        # Create and validate configuration
        self.config = self._create_system_config(config_dict)
        self._validate_config(self.config)
        
        logger.info(f"Configuration loaded for {self.config.environment} environment")
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            'oanda': {
                'api_key': '',
                'account_id': '',
                'environment': 'practice'
            },
            'model': asdict(ModelConfig()),
            'data': asdict(DataConfig()),
            'training': asdict(TrainingConfig()),
            'inference': asdict(InferenceConfig()),
            'logging': asdict(LoggingConfig()),
            'environment': 'development',
            'data_path': './data',
            'models_path': './models',
            'logs_path': './logs',
            'results_path': './results'
        }
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {str(e)}")
            return {}
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # OANDA configuration
        if os.getenv('OANDA_API_KEY'):
            env_config.setdefault('oanda', {})['api_key'] = os.getenv('OANDA_API_KEY')
        if os.getenv('OANDA_ACCOUNT_ID'):
            env_config.setdefault('oanda', {})['account_id'] = os.getenv('OANDA_ACCOUNT_ID')
        if os.getenv('OANDA_ENVIRONMENT'):
            env_config.setdefault('oanda', {})['environment'] = os.getenv('OANDA_ENVIRONMENT')
        
        # System environment
        if os.getenv('ENVIRONMENT'):
            env_config['environment'] = os.getenv('ENVIRONMENT')
        
        # Paths
        if os.getenv('DATA_PATH'):
            env_config['data_path'] = os.getenv('DATA_PATH')
        if os.getenv('MODELS_PATH'):
            env_config['models_path'] = os.getenv('MODELS_PATH')
        if os.getenv('LOGS_PATH'):
            env_config['logs_path'] = os.getenv('LOGS_PATH')
        
        # Device
        if os.getenv('DEVICE'):
            env_config.setdefault('model', {})['device'] = os.getenv('DEVICE')
        
        # Logging level
        if os.getenv('LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
        
        return env_config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_system_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Create SystemConfig from configuration dictionary."""
        try:
            return SystemConfig(
                oanda=OANDAConfig(**config_dict.get('oanda', {})),
                model=ModelConfig(**config_dict.get('model', {})),
                data=DataConfig(**config_dict.get('data', {})),
                training=TrainingConfig(**config_dict.get('training', {})),
                inference=InferenceConfig(**config_dict.get('inference', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                environment=config_dict.get('environment', 'development'),
                data_path=config_dict.get('data_path', './data'),
                models_path=config_dict.get('models_path', './models'),
                logs_path=config_dict.get('logs_path', './logs'),
                results_path=config_dict.get('results_path', './results')
            )
        except Exception as e:
            logger.error(f"Error creating system config: {str(e)}")
            raise
    
    def _validate_config(self, config: SystemConfig) -> None:
        """Validate configuration for consistency and requirements."""
        errors = []
        
        # Validate OANDA credentials for production
        if config.environment == 'production':
            if not config.oanda.api_key:
                errors.append("OANDA API key is required for production")
            if not config.oanda.account_id:
                errors.append("OANDA account ID is required for production")
        
        # Validate model dimensions consistency
        if config.model.tcnae_latent_dim != config.model.context_dim:
            errors.append("TCNAE latent dimension must match context dimension")
        
        # Validate training parameters
        if config.training.validation_split <= 0 or config.training.validation_split >= 1:
            errors.append("Validation split must be between 0 and 1")
        
        # Validate data parameters
        if len(config.data.instruments) != config.model.num_instruments:
            errors.append("Number of instruments must match model configuration")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def save_config(self, config_path: str, format: str = 'yaml') -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: Output format ('yaml' or 'json')
        """
        if not self.config:
            raise ValueError("No configuration loaded")
        
        config_dict = asdict(self.config)
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def get_config(self) -> SystemConfig:
        """Get current configuration."""
        if not self.config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        return self.config


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        System configuration
    """
    return config_manager.load_config(config_path)


def get_config() -> SystemConfig:
    """
    Get current configuration.
    
    Returns:
        System configuration
    """
    return config_manager.get_config()


if __name__ == "__main__":
    # Example usage
    print("Market Edge Finder Configuration Management")
    
    # Load default configuration
    config = load_config()
    print(f"Loaded configuration for {config.environment} environment")
    print(f"Model device: {config.model.device}")
    print(f"Number of instruments: {config.model.num_instruments}")
    print(f"TCNAE latent dimension: {config.model.tcnae_latent_dim}")