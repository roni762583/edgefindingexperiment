"""
Target Engineering for Market Edge Finder Experiment

This module generates USD-scaled pip targets for ML training.
Targets represent 1-hour ahead returns normalized for cross-instrument comparability.

Production-ready implementation with percentile scaling and proper error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# Instrument specifications for pip value calculations
INSTRUMENT_SPECS = {
    # Major USD pairs
    'EUR_USD': {'pip_size': 0.0001, 'base': 'EUR', 'quote': 'USD'},
    'GBP_USD': {'pip_size': 0.0001, 'base': 'GBP', 'quote': 'USD'},
    'AUD_USD': {'pip_size': 0.0001, 'base': 'AUD', 'quote': 'USD'},
    'NZD_USD': {'pip_size': 0.0001, 'base': 'NZD', 'quote': 'USD'},
    'USD_JPY': {'pip_size': 0.01, 'base': 'USD', 'quote': 'JPY'},
    'USD_CHF': {'pip_size': 0.0001, 'base': 'USD', 'quote': 'CHF'},
    'USD_CAD': {'pip_size': 0.0001, 'base': 'USD', 'quote': 'CAD'},
    
    # Cross pairs
    'EUR_GBP': {'pip_size': 0.0001, 'base': 'EUR', 'quote': 'GBP'},
    'EUR_JPY': {'pip_size': 0.01, 'base': 'EUR', 'quote': 'JPY'},
    'EUR_CHF': {'pip_size': 0.0001, 'base': 'EUR', 'quote': 'CHF'},
    'EUR_AUD': {'pip_size': 0.0001, 'base': 'EUR', 'quote': 'AUD'},
    'EUR_CAD': {'pip_size': 0.0001, 'base': 'EUR', 'quote': 'CAD'},
    'EUR_NZD': {'pip_size': 0.0001, 'base': 'EUR', 'quote': 'NZD'},
    
    'GBP_JPY': {'pip_size': 0.01, 'base': 'GBP', 'quote': 'JPY'},
    'GBP_CHF': {'pip_size': 0.0001, 'base': 'GBP', 'quote': 'CHF'},
    'GBP_AUD': {'pip_size': 0.0001, 'base': 'GBP', 'quote': 'AUD'},
    'GBP_CAD': {'pip_size': 0.0001, 'base': 'GBP', 'quote': 'CAD'},
    'GBP_NZD': {'pip_size': 0.0001, 'base': 'GBP', 'quote': 'NZD'},
    
    'AUD_JPY': {'pip_size': 0.01, 'base': 'AUD', 'quote': 'JPY'},
    'AUD_CHF': {'pip_size': 0.0001, 'base': 'AUD', 'quote': 'CHF'},
    'AUD_NZD': {'pip_size': 0.0001, 'base': 'AUD', 'quote': 'NZD'},
    
    'CHF_JPY': {'pip_size': 0.01, 'base': 'CHF', 'quote': 'JPY'},
    'CAD_JPY': {'pip_size': 0.01, 'base': 'CAD', 'quote': 'JPY'},
    'NZD_JPY': {'pip_size': 0.01, 'base': 'NZD', 'quote': 'JPY'},
}


class TargetEngineer:
    """
    Target engineering for FX edge discovery experiments.
    
    Generates USD-scaled pip targets for 1-hour ahead prediction.
    Applies percentile scaling for robust ML training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize target engineer with configuration.
        
        Args:
            config: Configuration dictionary with target engineering parameters
        """
        self.config = config
        self.target_config = config.get('targets', {})
        self.position_size_usd = self.target_config.get('position_size_usd', 100000)
        self.scaling_method = self.target_config.get('scaling_method', 'percentile')
        self.scaling_window = self.target_config.get('scaling_window', 200)
        self.target_horizon = self.target_config.get('target_horizon', 1)
        self.clip_percentile = self.target_config.get('clip_percentile', 95)
        
        logger.info(f"Initialized TargetEngineer with scaling_method={self.scaling_method}, "
                   f"window={self.scaling_window}, horizon={self.target_horizon}")
    
    def calculate_pip_value_usd(self, instrument: str, close_price: float, 
                               usd_exchange_rates: Dict[str, float]) -> float:
        """
        Calculate the USD value of one pip for an instrument.
        
        Args:
            instrument: FX pair name (e.g., 'EUR_USD')
            close_price: Current close price
            usd_exchange_rates: Exchange rates to USD for all currencies
            
        Returns:
            USD value of one pip for standard lot (100,000 units)
            
        Raises:
            ValueError: If instrument not supported or missing exchange rates
        """
        if instrument not in INSTRUMENT_SPECS:
            raise ValueError(f"Unsupported instrument: {instrument}")
        
        spec = INSTRUMENT_SPECS[instrument]
        pip_size = spec['pip_size']
        base_currency = spec['base']
        quote_currency = spec['quote']
        
        # Calculate pip value in quote currency
        pip_value_quote = pip_size * self.position_size_usd
        
        # Convert to USD if quote currency is not USD
        if quote_currency == 'USD':
            pip_value_usd = pip_value_quote
        else:
            if quote_currency not in usd_exchange_rates:
                raise ValueError(f"Missing USD exchange rate for {quote_currency}")
            pip_value_usd = pip_value_quote * usd_exchange_rates[quote_currency]
        
        logger.debug(f"{instrument}: pip_value_usd = ${pip_value_usd:.2f}")
        return pip_value_usd
    
    def calculate_price_change_pips(self, price_current: float, price_future: float, 
                                   instrument: str) -> float:
        """
        Calculate price change in pips.
        
        Args:
            price_current: Current price
            price_future: Future price (target_horizon ahead)
            instrument: FX pair name
            
        Returns:
            Price change in pips (positive = appreciation, negative = depreciation)
        """
        if instrument not in INSTRUMENT_SPECS:
            raise ValueError(f"Unsupported instrument: {instrument}")
        
        pip_size = INSTRUMENT_SPECS[instrument]['pip_size']
        price_change = price_future - price_current
        pips_change = price_change / pip_size
        
        return pips_change
    
    def generate_targets(self, ohlc_data: Dict[str, pd.DataFrame], 
                        usd_exchange_rates: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate USD-scaled pip targets for all instruments.
        
        Args:
            ohlc_data: Dictionary of OHLC DataFrames keyed by instrument
            usd_exchange_rates: Dictionary of USD exchange rate DataFrames
            
        Returns:
            Dictionary of target DataFrames with columns:
            - target_pips: Raw pip change
            - target_usd: USD-scaled pip value
            - target_scaled: Percentile-scaled target [0,1]
        """
        targets = {}
        
        logger.info(f"Generating targets for {len(ohlc_data)} instruments")
        
        for instrument, df in ohlc_data.items():
            logger.debug(f"Processing targets for {instrument}")
            
            if len(df) < self.target_horizon + self.scaling_window:
                logger.warning(f"Insufficient data for {instrument}: {len(df)} bars")
                continue
            
            # Calculate future prices
            future_closes = df['close'].shift(-self.target_horizon)
            current_closes = df['close']
            
            # Calculate pip changes
            pip_changes = []
            usd_values = []
            
            for i in range(len(df) - self.target_horizon):
                current_price = current_closes.iloc[i]
                future_price = future_closes.iloc[i]
                
                if pd.isna(current_price) or pd.isna(future_price):
                    pip_changes.append(np.nan)
                    usd_values.append(np.nan)
                    continue
                
                # Calculate pip change
                pip_change = self.calculate_price_change_pips(
                    current_price, future_price, instrument
                )
                
                # Get USD exchange rates for this timestamp
                timestamp = df.index[i]
                rates = {}
                for currency, rate_df in usd_exchange_rates.items():
                    if timestamp in rate_df.index:
                        rates[currency] = rate_df.loc[timestamp, 'close']
                    else:
                        # Forward fill missing rates
                        closest_rates = rate_df[rate_df.index <= timestamp]
                        if not closest_rates.empty:
                            rates[currency] = closest_rates.iloc[-1]['close']
                
                try:
                    # Calculate USD value of pip change
                    pip_value_usd = self.calculate_pip_value_usd(
                        instrument, current_price, rates
                    )
                    usd_value = pip_change * pip_value_usd
                    
                    pip_changes.append(pip_change)
                    usd_values.append(usd_value)
                    
                except ValueError as e:
                    logger.warning(f"Could not calculate USD value for {instrument} at {timestamp}: {e}")
                    pip_changes.append(np.nan)
                    usd_values.append(np.nan)
            
            # Pad with NaN for target_horizon
            pip_changes.extend([np.nan] * self.target_horizon)
            usd_values.extend([np.nan] * self.target_horizon)
            
            # Create target DataFrame
            target_df = pd.DataFrame({
                'target_pips': pip_changes,
                'target_usd': usd_values
            }, index=df.index)
            
            # Apply percentile scaling
            target_df['target_scaled'] = self._apply_percentile_scaling(
                target_df['target_usd']
            )
            
            targets[instrument] = target_df
            
            # Log statistics
            valid_targets = target_df['target_usd'].dropna()
            if len(valid_targets) > 0:
                logger.info(f"{instrument}: {len(valid_targets)} targets, "
                           f"USD range: [{valid_targets.min():.2f}, {valid_targets.max():.2f}], "
                           f"mean: {valid_targets.mean():.2f}")
        
        logger.info(f"Generated targets for {len(targets)} instruments")
        return targets
    
    def _apply_percentile_scaling(self, series: pd.Series) -> pd.Series:
        """
        Apply percentile scaling to target series.
        
        Args:
            series: Input series to scale
            
        Returns:
            Scaled series with values in [0,1] range
        """
        if self.scaling_method != 'percentile':
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
        
        scaled = series.copy()
        
        # Apply rolling percentile scaling
        for i in range(self.scaling_window, len(series)):
            window_data = series.iloc[i-self.scaling_window:i].dropna()
            
            if len(window_data) < 10:  # Need minimum data
                scaled.iloc[i] = np.nan
                continue
            
            current_value = series.iloc[i]
            if pd.isna(current_value):
                scaled.iloc[i] = np.nan
                continue
            
            # Calculate percentile rank
            percentile_rank = (window_data < current_value).sum() / len(window_data)
            
            # Clip extreme values
            clip_lower = (100 - self.clip_percentile) / 200  # e.g., 0.025 for 95th percentile
            clip_upper = 1 - clip_lower  # e.g., 0.975
            
            percentile_rank = np.clip(percentile_rank, clip_lower, clip_upper)
            scaled.iloc[i] = percentile_rank
        
        return scaled
    
    def validate_targets(self, targets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate generated targets and return quality metrics.
        
        Args:
            targets: Dictionary of target DataFrames
            
        Returns:
            Dictionary of validation metrics per instrument
        """
        validation_results = {}
        
        for instrument, target_df in targets.items():
            # Calculate metrics
            valid_targets = target_df['target_scaled'].dropna()
            
            if len(valid_targets) == 0:
                validation_results[instrument] = {
                    'status': 'FAILED',
                    'error': 'No valid targets'
                }
                continue
            
            metrics = {
                'status': 'PASSED',
                'count': len(valid_targets),
                'coverage': len(valid_targets) / len(target_df),
                'min_value': valid_targets.min(),
                'max_value': valid_targets.max(),
                'mean_value': valid_targets.mean(),
                'std_value': valid_targets.std(),
                'in_range': (valid_targets >= 0).all() and (valid_targets <= 1).all()
            }
            
            # Check for issues
            if metrics['coverage'] < 0.5:
                metrics['status'] = 'WARNING'
                metrics['warning'] = 'Low coverage'
            
            if not metrics['in_range']:
                metrics['status'] = 'FAILED'
                metrics['error'] = 'Values outside [0,1] range'
            
            validation_results[instrument] = metrics
        
        return validation_results


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Example usage of TargetEngineer."""
    # Load configuration
    config = load_config('configs/production_config.yaml')
    
    # Initialize target engineer
    target_engineer = TargetEngineer(config)
    
    # Example data loading (replace with actual data loading)
    # ohlc_data = load_ohlc_data()
    # usd_exchange_rates = load_usd_exchange_rates()
    
    # Generate targets
    # targets = target_engineer.generate_targets(ohlc_data, usd_exchange_rates)
    
    # Validate targets
    # validation_results = target_engineer.validate_targets(targets)
    
    print("Target engineering example completed")


if __name__ == "__main__":
    main()