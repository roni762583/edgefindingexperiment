#!/usr/bin/env python3
"""
Real-time inference script for Market Edge Finder Experiment.

Runs the production inference pipeline with real-time data streaming,
prediction generation, and monitoring capabilities.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import signal
import time
import json
from typing import Dict, Any, Optional
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.config import load_config, SystemConfig
from inference.predictor import EdgePredictor, PredictionConfig, RealtimePredictor
from inference.data_pipeline import RealtimeDataPipeline, DataConfig
from utils.logger import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """
    Complete inference orchestration for real-time predictions.
    
    Manages data pipeline, prediction engine, monitoring, and
    graceful shutdown handling.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the inference orchestrator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self.data_pipeline: Optional[RealtimeDataPipeline] = None
        self.predictor: Optional[EdgePredictor] = None
        self.realtime_predictor: Optional[RealtimePredictor] = None
        
        # Monitoring
        self.prediction_count = 0
        self.last_prediction_time: Optional[datetime] = None
        self.error_count = 0
        
        # Ensure output directories exist
        Path(config.logs_path).mkdir(parents=True, exist_ok=True)
        Path(config.results_path).mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Inference orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def initialize_data_pipeline(self) -> None:
        """Initialize the real-time data pipeline."""
        logger.info("Initializing real-time data pipeline...")
        
        # Data configuration
        data_config = DataConfig(
            api_key=self.config.oanda.api_key,
            account_id=self.config.oanda.account_id,
            environment=self.config.oanda.environment,
            instruments=self.config.data.instruments,
            granularity=self.config.data.granularity,
            update_interval_seconds=self.config.inference.data_update_interval,
            max_retries=3,
            retry_delay_seconds=5
        )
        
        # Initialize pipeline
        self.data_pipeline = RealtimeDataPipeline(data_config)
        
        logger.info("Data pipeline initialized successfully")
    
    def initialize_predictor(self) -> None:
        """Initialize the prediction engine."""
        logger.info("Initializing prediction engine...")
        
        models_path = Path(self.config.models_path)
        
        # Prediction configuration
        prediction_config = PredictionConfig(
            tcnae_model_path=str(models_path / self.config.inference.tcnae_model_name),
            gbdt_model_path=str(models_path / self.config.inference.gbdt_model_name),
            context_manager_path=str(models_path / self.config.inference.context_manager_name),
            normalizer_path=str(models_path / self.config.inference.normalizer_name),
            sequence_length=self.config.model.tcnae_sequence_length,
            latent_dim=self.config.model.tcnae_latent_dim,
            num_instruments=self.config.model.num_instruments,
            confidence_threshold=self.config.inference.confidence_threshold,
            max_prediction_age_minutes=self.config.inference.max_prediction_age_minutes,
            device=self.config.model.device,
            max_position_size=self.config.inference.max_position_size,
            max_total_exposure=self.config.inference.max_total_exposure
        )
        
        # Initialize predictor
        self.predictor = EdgePredictor(prediction_config)
        self.predictor.initialize_models()
        
        # Initialize real-time wrapper
        self.realtime_predictor = RealtimePredictor(
            prediction_config,
            update_interval_seconds=self.config.inference.update_interval_seconds
        )
        
        logger.info("Prediction engine initialized successfully")
    
    def setup_data_callbacks(self) -> None:
        """Setup callbacks to connect data pipeline to predictor."""
        if not self.data_pipeline or not self.predictor:
            raise ValueError("Data pipeline and predictor must be initialized first")
        
        def prediction_callback(market_data: Dict[str, Any]) -> None:
            """Callback to update predictor with new market data."""
            try:
                self.predictor.update_features(market_data)
                logger.debug(f"Updated predictor with data for {len(market_data)} instruments")
            except Exception as e:
                logger.error(f"Error in prediction callback: {str(e)}")
                self.error_count += 1
        
        self.data_pipeline.add_data_callback(prediction_callback)
        logger.info("Data callbacks configured")
    
    def start_inference_pipeline(self) -> None:
        """Start the complete inference pipeline."""
        logger.info("Starting inference pipeline...")
        
        try:
            # Start data pipeline
            self.data_pipeline.start()
            logger.info("Data pipeline started")
            
            # Start real-time predictor
            self.realtime_predictor.start()
            logger.info("Real-time predictor started")
            
            self.running = True
            logger.info("Inference pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start inference pipeline: {str(e)}")
            self.shutdown()
            raise
    
    def run_monitoring_loop(self) -> None:
        """Run the main monitoring and prediction loop."""
        logger.info("Starting monitoring loop...")
        
        last_status_time = time.time()
        status_interval = 300  # 5 minutes
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get latest predictions
                predictions = self.realtime_predictor.get_current_predictions()
                
                if predictions:
                    self._process_predictions(predictions)
                    self.prediction_count += 1
                    self.last_prediction_time = datetime.now()
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    self._log_status()
                    last_status_time = current_time
                
                # Sleep before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self.error_count += 1
                time.sleep(60)  # Wait longer after error
    
    def _process_predictions(self, predictions: Dict[str, Any]) -> None:
        """
        Process and handle new predictions.
        
        Args:
            predictions: Prediction results from the model
        """
        try:
            # Log predictions (in production, you might send to trading system)
            timestamp = predictions['metadata']['timestamp']
            total_exposure = predictions['metadata']['total_exposure']
            high_confidence_count = predictions['metadata']['high_confidence_count']
            
            logger.info(f"New predictions at {timestamp}")
            logger.info(f"Total exposure: {total_exposure:.4f}")
            logger.info(f"High confidence signals: {high_confidence_count}")
            
            # Log significant predictions
            significant_predictions = []
            for instrument, prediction in predictions['predictions'].items():
                confidence = predictions['confidence_scores'][instrument]
                if abs(prediction) > self.config.inference.confidence_threshold:
                    significant_predictions.append((instrument, prediction, confidence))
            
            if significant_predictions:
                logger.info("Significant predictions:")
                for instrument, pred, conf in significant_predictions:
                    direction = "BUY" if pred > 0 else "SELL"
                    logger.info(f"  {instrument}: {direction} {abs(pred):.4f} (confidence: {conf:.4f})")
            
            # Save predictions to file (optional)
            if self.config.environment == 'production':
                self._save_predictions(predictions)
            
        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
    
    def _save_predictions(self, predictions: Dict[str, Any]) -> None:
        """Save predictions to file for analysis."""
        try:
            results_dir = Path(self.config.results_path)
            timestamp = datetime.now().strftime('%Y%m%d')
            predictions_file = results_dir / f"predictions_{timestamp}.jsonl"
            
            # Append to daily file
            with open(predictions_file, 'a') as f:
                json.dump(predictions, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
    
    def _log_status(self) -> None:
        """Log system status and health metrics."""
        logger.info("System Status Report")
        logger.info("-" * 50)
        
        # Prediction stats
        logger.info(f"Total predictions generated: {self.prediction_count}")
        if self.last_prediction_time:
            time_since_last = (datetime.now() - self.last_prediction_time).total_seconds()
            logger.info(f"Time since last prediction: {time_since_last:.0f}s")
        logger.info(f"Error count: {self.error_count}")
        
        # Data pipeline health
        if self.data_pipeline:
            pipeline_status = self.data_pipeline.get_pipeline_status()
            logger.info(f"Data pipeline running: {pipeline_status['running']}")
            
            api_health = pipeline_status.get('api_health', {})
            logger.info(f"API accessible: {api_health.get('api_accessible', False)}")
        
        # Predictor health
        if self.predictor:
            model_health = self.predictor.get_model_health()
            logger.info(f"Models initialized: {model_health['initialized']}")
            
            cache_status = model_health.get('cache_status', {})
            if cache_status.get('cache_age_minutes') is not None:
                logger.info(f"Prediction cache age: {cache_status['cache_age_minutes']:.1f} minutes")
        
        logger.info("-" * 50)
    
    def shutdown(self) -> None:
        """Shutdown the inference pipeline gracefully."""
        logger.info("Shutting down inference pipeline...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Stop components
        if self.realtime_predictor:
            self.realtime_predictor.stop()
            logger.info("Real-time predictor stopped")
        
        if self.data_pipeline:
            self.data_pipeline.stop()
            logger.info("Data pipeline stopped")
        
        # Save final state
        if self.predictor:
            try:
                state_file = Path(self.config.models_path) / "predictor_state.pkl"
                self.predictor.save_state(str(state_file))
                logger.info("Predictor state saved")
            except Exception as e:
                logger.error(f"Error saving predictor state: {str(e)}")
        
        logger.info("Inference pipeline shutdown completed")
    
    def run_inference(self) -> None:
        """Run the complete inference pipeline."""
        logger.info("Starting Market Edge Finder inference pipeline...")
        
        try:
            # Initialize components
            self.initialize_data_pipeline()
            self.initialize_predictor()
            self.setup_data_callbacks()
            
            # Start pipeline
            self.start_inference_pipeline()
            
            # Run monitoring loop
            self.run_monitoring_loop()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Inference pipeline error: {str(e)}")
        finally:
            self.shutdown()


class InferenceHealthChecker:
    """Health checker for monitoring inference system."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def check_models(self) -> Dict[str, bool]:
        """Check if all required model files exist."""
        models_path = Path(self.config.models_path)
        
        required_files = [
            self.config.inference.tcnae_model_name,
            self.config.inference.gbdt_model_name,
            self.config.inference.context_manager_name,
            self.config.inference.normalizer_name
        ]
        
        status = {}
        for filename in required_files:
            filepath = models_path / filename
            status[filename] = filepath.exists()
        
        return status
    
    def check_api_credentials(self) -> bool:
        """Check if OANDA API credentials are configured."""
        return bool(self.config.oanda.api_key and self.config.oanda.account_id)
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'models': self.check_models(),
            'api_credentials': self.check_api_credentials(),
            'directories': {
                'models_path': Path(self.config.models_path).exists(),
                'data_path': Path(self.config.data_path).exists(),
                'logs_path': Path(self.config.logs_path).exists(),
                'results_path': Path(self.config.results_path).exists()
            }
        }
        
        # Overall health
        models_ok = all(health_status['models'].values())
        health_status['overall_healthy'] = (
            models_ok and 
            health_status['api_credentials'] and
            all(health_status['directories'].values())
        )
        
        return health_status


def main():
    """Main inference execution function."""
    parser = argparse.ArgumentParser(description='Market Edge Finder Real-time Inference')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--health-check', action='store_true', help='Run health check only')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon (background)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        log_file = Path(config.logs_path) / "inference.log"
        setup_logging(
            level=args.log_level,
            log_file=log_file
        )
        
        logger.info("Starting Market Edge Finder inference")
        logger.info(f"Configuration: {args.config or 'default'}")
        logger.info(f"Environment: {config.environment}")
        
        # Health check mode
        if args.health_check:
            health_checker = InferenceHealthChecker(config)
            health_status = health_checker.run_health_check()
            
            print("Health Check Results:")
            print("=" * 50)
            print(f"Overall Healthy: {'✓' if health_status['overall_healthy'] else '✗'}")
            print(f"API Credentials: {'✓' if health_status['api_credentials'] else '✗'}")
            
            print("\nModel Files:")
            for filename, exists in health_status['models'].items():
                print(f"  {filename}: {'✓' if exists else '✗'}")
            
            print("\nDirectories:")
            for dirname, exists in health_status['directories'].items():
                print(f"  {dirname}: {'✓' if exists else '✗'}")
            
            if not health_status['overall_healthy']:
                sys.exit(1)
            else:
                print("\nSystem is healthy and ready for inference!")
                sys.exit(0)
        
        # Normal inference mode
        orchestrator = InferenceOrchestrator(config)
        
        if args.daemon:
            logger.info("Running in daemon mode")
            # In production, you might want to use proper daemon libraries
            
        orchestrator.run_inference()
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()