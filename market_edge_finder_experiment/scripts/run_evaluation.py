#!/usr/bin/env python3
"""
Evaluation script for Market Edge Finder Experiment.

Evaluates trained models on validation/test data and generates
comprehensive performance reports with visualizations.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.config import load_config, SystemConfig
from evaluation.metrics import TradingMetricsCalculator, BacktestEvaluator, save_evaluation_results
from evaluation.visualization import EvaluationVisualizer
from inference.predictor import EdgePredictor, PredictionConfig
from utils.logger import setup_logging
import pickle

# Configure logging
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Complete model evaluation system for trained models.
    
    Loads trained models, generates predictions on test data,
    calculates comprehensive metrics, and creates visualization reports.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the model evaluator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.metrics_calculator = TradingMetricsCalculator()
        self.visualizer = EvaluationVisualizer()
        
        # Ensure output directories exist
        Path(config.results_path).mkdir(parents=True, exist_ok=True)
        Path(config.logs_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Model evaluator initialized")
    
    def load_test_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load test data for evaluation.
        
        Returns:
            Dictionary with test data
        """
        logger.info("Loading test data...")
        
        data_path = Path(self.config.data_path)
        test_data = {}
        
        # Check if we have validation data
        val_dir = data_path / "validation"
        if val_dir.exists():
            for instrument in self.config.data.instruments:
                val_file = val_dir / f"{instrument}.parquet"
                if val_file.exists():
                    test_data[instrument] = pd.read_parquet(val_file)
        
        if not test_data:
            logger.warning("No validation data found, using last 20% of training data")
            train_dir = data_path / "train"
            if train_dir.exists():
                for instrument in self.config.data.instruments:
                    train_file = train_dir / f"{instrument}.parquet"
                    if train_file.exists():
                        full_data = pd.read_parquet(train_file)
                        # Use last 20% as test data
                        test_size = int(len(full_data) * 0.2)
                        test_data[instrument] = full_data.tail(test_size).reset_index(drop=True)
        
        logger.info(f"Test data loaded for {len(test_data)} instruments")
        return test_data
    
    def load_target_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load target data for evaluation.
        
        Returns:
            Dictionary with target data
        """
        logger.info("Loading target data...")
        
        data_path = Path(self.config.data_path)
        target_data = {}
        
        for instrument in self.config.data.instruments:
            targets_file = data_path / f"{instrument}_targets.parquet"
            if targets_file.exists():
                target_data[instrument] = pd.read_parquet(targets_file)
        
        logger.info(f"Target data loaded for {len(target_data)} instruments")
        return target_data
    
    def initialize_predictor(self) -> EdgePredictor:
        """
        Initialize predictor with trained models.
        
        Returns:
            Initialized EdgePredictor
        """
        logger.info("Initializing predictor with trained models...")
        
        models_path = Path(self.config.models_path)
        
        # Check if all required model files exist
        required_files = [
            self.config.inference.tcnae_model_name,
            self.config.inference.gbdt_model_name,
            self.config.inference.context_manager_name,
            self.config.inference.normalizer_name
        ]
        
        missing_files = []
        for filename in required_files:
            if not (models_path / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Create prediction config
        prediction_config = PredictionConfig(
            tcnae_model_path=str(models_path / self.config.inference.tcnae_model_name),
            gbdt_model_path=str(models_path / self.config.inference.gbdt_model_name),
            context_manager_path=str(models_path / self.config.inference.context_manager_name),
            normalizer_path=str(models_path / self.config.inference.normalizer_name),
            sequence_length=self.config.model.tcnae_sequence_length,
            latent_dim=self.config.model.tcnae_latent_dim,
            num_instruments=self.config.model.num_instruments,
            device=self.config.model.device
        )
        
        # Initialize and load predictor
        predictor = EdgePredictor(prediction_config)
        predictor.initialize_models()
        
        logger.info("Predictor initialized successfully")
        return predictor
    
    def generate_predictions(self, 
                           predictor: EdgePredictor, 
                           test_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Generate predictions on test data.
        
        Args:
            predictor: Initialized predictor
            test_data: Test data for each instrument
            
        Returns:
            Dictionary with predictions for each instrument
        """
        logger.info("Generating predictions on test data...")
        
        predictions = {}
        
        for instrument, data in test_data.items():
            try:
                # Convert data to the format expected by predictor
                market_data = {instrument: data}
                
                # Generate predictions
                instrument_predictions = predictor.predict_for_instruments([instrument], market_data)
                
                if instrument in instrument_predictions:
                    # For evaluation, we need to generate predictions for each timestep
                    # This is a simplified version - in practice you'd need to run the full sequence
                    pred_values = []
                    
                    # Generate predictions for each timestep (simplified)
                    for i in range(len(data)):
                        # In real implementation, you'd use sliding windows
                        pred_values.append(instrument_predictions[instrument])
                    
                    predictions[instrument] = np.array(pred_values)
                    logger.info(f"Generated {len(pred_values)} predictions for {instrument}")
                else:
                    logger.warning(f"No predictions generated for {instrument}")
                    predictions[instrument] = np.zeros(len(data))
                    
            except Exception as e:
                logger.error(f"Error generating predictions for {instrument}: {str(e)}")
                predictions[instrument] = np.zeros(len(data))
        
        logger.info(f"Predictions generated for {len(predictions)} instruments")
        return predictions
    
    def align_predictions_and_targets(self, 
                                    predictions: Dict[str, np.ndarray],
                                    target_data: Dict[str, pd.DataFrame],
                                    test_data: Dict[str, pd.DataFrame]) -> tuple:
        """
        Align predictions with targets for evaluation.
        
        Args:
            predictions: Model predictions
            target_data: Target returns data
            test_data: Test feature data
            
        Returns:
            Tuple of (aligned_predictions, aligned_targets, timestamps)
        """
        logger.info("Aligning predictions with targets...")
        
        aligned_predictions = []
        aligned_targets = []
        timestamps = []
        
        for instrument in self.config.data.instruments:
            if instrument not in predictions or instrument not in target_data or instrument not in test_data:
                logger.warning(f"Missing data for {instrument}, skipping")
                continue
            
            pred = predictions[instrument]
            targets = target_data[instrument]
            test_features = test_data[instrument]
            
            # Align by timestamp
            test_timestamps = test_features['timestamp'].values
            target_timestamps = targets['timestamp'].values
            
            # Find common timestamps
            common_timestamps = np.intersect1d(test_timestamps, target_timestamps)
            
            if len(common_timestamps) == 0:
                logger.warning(f"No common timestamps for {instrument}")
                continue
            
            # Get indices for common timestamps
            test_indices = np.isin(test_timestamps, common_timestamps)
            target_indices = np.isin(target_timestamps, common_timestamps)
            
            # Align data
            aligned_pred = pred[test_indices] if len(pred) == len(test_timestamps) else pred[:sum(test_indices)]
            aligned_target = targets.loc[target_indices, 'target_return'].values
            
            # Ensure same length
            min_length = min(len(aligned_pred), len(aligned_target))
            aligned_pred = aligned_pred[:min_length]
            aligned_target = aligned_target[:min_length]
            
            if min_length > 0:
                aligned_predictions.append(aligned_pred)
                aligned_targets.append(aligned_target)
                timestamps.extend(common_timestamps[:min_length])
        
        if aligned_predictions:
            # Stack predictions and targets
            final_predictions = np.column_stack(aligned_predictions)
            final_targets = np.column_stack(aligned_targets)
            final_timestamps = np.array(timestamps[:len(final_predictions)])
        else:
            # Empty arrays if no valid data
            final_predictions = np.array([]).reshape(0, len(self.config.data.instruments))
            final_targets = np.array([]).reshape(0, len(self.config.data.instruments))
            final_timestamps = np.array([])
        
        logger.info(f"Aligned data shape: predictions={final_predictions.shape}, targets={final_targets.shape}")
        return final_predictions, final_targets, final_timestamps
    
    def calculate_evaluation_metrics(self, 
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: True targets
            timestamps: Optional timestamps
            
        Returns:
            Evaluation results
        """
        logger.info("Calculating evaluation metrics...")
        
        if len(predictions) == 0 or len(targets) == 0:
            logger.warning("No data available for evaluation")
            return {}
        
        # Generate mock price data for metrics calculation
        mock_prices = np.ones_like(predictions)  # Simplified for metrics
        
        # Generate comprehensive report
        evaluation_results = self.metrics_calculator.generate_comprehensive_report(
            predictions=predictions,
            returns=targets,
            prices=mock_prices
        )
        
        logger.info("Evaluation metrics calculated successfully")
        return evaluation_results
    
    def generate_visualizations(self, 
                              evaluation_results: Dict[str, Any],
                              predictions: Optional[np.ndarray] = None,
                              targets: Optional[np.ndarray] = None,
                              timestamps: Optional[np.ndarray] = None,
                              save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate evaluation visualizations.
        
        Args:
            evaluation_results: Evaluation metrics
            predictions: Model predictions
            targets: True targets
            timestamps: Timestamps
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of generated figures
        """
        logger.info("Generating evaluation visualizations...")
        
        if save_dir is None:
            save_dir = Path(self.config.results_path) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        figures = self.visualizer.create_comprehensive_dashboard(
            evaluation_results=evaluation_results,
            predictions=predictions,
            returns=targets,
            timestamps=timestamps,
            save_dir=save_dir
        )
        
        logger.info(f"Visualizations saved to {save_dir}")
        return figures
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """
        Run complete model evaluation pipeline.
        
        Returns:
            Complete evaluation results
        """
        logger.info("Starting complete model evaluation...")
        
        try:
            # Step 1: Load data
            test_data = self.load_test_data()
            target_data = self.load_target_data()
            
            if not test_data or not target_data:
                raise ValueError("No test or target data available")
            
            # Step 2: Initialize predictor
            predictor = self.initialize_predictor()
            
            # Step 3: Generate predictions
            predictions = self.generate_predictions(predictor, test_data)
            
            # Step 4: Align predictions and targets
            aligned_predictions, aligned_targets, timestamps = self.align_predictions_and_targets(
                predictions, target_data, test_data
            )
            
            # Step 5: Calculate metrics
            evaluation_results = self.calculate_evaluation_metrics(
                aligned_predictions, aligned_targets, timestamps
            )
            
            # Step 6: Generate visualizations
            figures = self.generate_visualizations(
                evaluation_results, aligned_predictions, aligned_targets, timestamps
            )
            
            # Step 7: Save results
            results_file = Path(self.config.results_path) / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_evaluation_results(evaluation_results, results_file)
            
            # Step 8: Create summary
            self._create_evaluation_summary(evaluation_results)
            
            logger.info("Complete model evaluation finished successfully!")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {str(e)}")
            raise
    
    def _create_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Create evaluation summary report."""
        logger.info("Creating evaluation summary...")
        
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"\nOverall Performance:")
            print(f"  Accuracy: {summary.get('overall_accuracy', 0):.4f}")
            print(f"  Correlation: {summary.get('overall_correlation', 0):.4f}")
            print(f"  Hit Rate: {summary.get('hit_rate', 0):.4f}")
            print(f"  Avg Return per Trade: {summary.get('avg_return_per_trade', 0):.6f}")
        
        if 'sharpe_ratios' in results:
            sharpe_ratios = results['sharpe_ratios']
            print(f"\nSharpe Ratios:")
            for instrument, sharpe in sharpe_ratios.items():
                print(f"  {instrument}: {sharpe:.4f}")
        
        if 'maximum_drawdown' in results:
            drawdowns = results['maximum_drawdown']
            print(f"\nMaximum Drawdowns:")
            for instrument, dd in drawdowns.items():
                print(f"  {instrument}: {abs(dd)*100:.2f}%")
        
        print("\n" + "="*80)


def main():
    """Main evaluation execution function."""
    parser = argparse.ArgumentParser(description='Market Edge Finder Model Evaluation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config.results_path = args.output_dir
        
        # Setup logging
        setup_logging(
            level=args.log_level,
            log_file=Path(config.logs_path) / "evaluation.log"
        )
        
        logger.info("Starting Market Edge Finder model evaluation")
        logger.info(f"Configuration: {args.config or 'default'}")
        logger.info(f"Environment: {config.environment}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Run evaluation
        evaluation_results = evaluator.run_complete_evaluation()
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()