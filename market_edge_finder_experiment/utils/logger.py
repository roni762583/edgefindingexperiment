"""
Centralized logging configuration for Market Edge Finder Experiment.

Provides consistent logging setup across all components with structured
logging, file rotation, and environment-specific configurations.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging output.
    
    Provides both human-readable and JSON formatted output depending
    on the environment and use case.
    """
    
    def __init__(self, 
                 format_type: str = 'standard',
                 include_extra: bool = True):
        """
        Initialize structured formatter.
        
        Args:
            format_type: Type of formatting ('standard', 'json', 'detailed')
            include_extra: Whether to include extra fields in JSON output
        """
        self.format_type = format_type
        self.include_extra = include_extra
        
        # Standard format string
        if format_type == 'detailed':
            fmt = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        else:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        super().__init__(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record based on format type."""
        
        if self.format_type == 'json':
            return self._format_json(record)
        else:
            return super().format(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'exc_info',
                              'exc_text', 'stack_info', 'lineno', 'funcName',
                              'created', 'msecs', 'relativeCreated', 'thread',
                              'threadName', 'processName', 'process', 'message']:
                    log_data[key] = value
        
        return json.dumps(log_data)


class ComponentFilter(logging.Filter):
    """
    Filter to add component information to log records.
    
    Automatically adds component context based on logger names
    for better traceability in multi-component systems.
    """
    
    def __init__(self, component_name: Optional[str] = None):
        """
        Initialize component filter.
        
        Args:
            component_name: Override component name
        """
        super().__init__()
        self.component_name = component_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add component information to log record."""
        
        # Determine component from logger name if not specified
        if self.component_name:
            component = self.component_name
        else:
            # Extract component from logger name
            logger_parts = record.name.split('.')
            if len(logger_parts) >= 2:
                component = logger_parts[1]  # e.g., 'models', 'features', 'training'
            else:
                component = 'main'
        
        record.component = component
        return True


class LoggerManager:
    """
    Centralized logger management for the entire system.
    
    Provides consistent logging configuration, handles file rotation,
    and manages different logging levels for different components.
    """
    
    def __init__(self):
        self.configured = False
        self.handlers: Dict[str, logging.Handler] = {}
        self.component_levels: Dict[str, str] = {}
    
    def setup_logging(self,
                     level: str = 'INFO',
                     log_file: Optional[Path] = None,
                     console_output: bool = True,
                     json_format: bool = False,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5,
                     component_levels: Optional[Dict[str, str]] = None) -> None:
        """
        Setup comprehensive logging configuration.
        
        Args:
            level: Default logging level
            log_file: Path to log file (optional)
            console_output: Whether to output to console
            json_format: Whether to use JSON formatting
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            component_levels: Per-component logging levels
        """
        
        if self.configured:
            return
        
        # Store component levels
        self.component_levels = component_levels or {}
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            if json_format:
                console_formatter = StructuredFormatter(format_type='json')
            else:
                console_formatter = StructuredFormatter(format_type='standard')
            
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(ComponentFilter())
            
            root_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # File handler with rotation
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            
            # Always use detailed format for file logs
            file_formatter = StructuredFormatter(format_type='detailed')
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(ComponentFilter())
            
            root_logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
        
        # Error file handler (separate file for errors)
        if log_file:
            error_file = log_file.parent / f"{log_file.stem}_errors{log_file.suffix}"
            error_handler = logging.handlers.RotatingFileHandler(
                filename=error_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            error_formatter = StructuredFormatter(format_type='detailed')
            error_handler.setFormatter(error_formatter)
            error_handler.addFilter(ComponentFilter())
            
            root_logger.addHandler(error_handler)
            self.handlers['error'] = error_handler
        
        # Configure component-specific levels
        self._configure_component_levels()
        
        self.configured = True
        
        # Log configuration
        logger = logging.getLogger(__name__)
        logger.info("Logging system configured")
        logger.info(f"Default level: {level}")
        if log_file:
            logger.info(f"Log file: {log_file}")
        if component_levels:
            logger.info(f"Component levels: {component_levels}")
    
    def _configure_component_levels(self) -> None:
        """Configure logging levels for specific components."""
        
        for component, level in self.component_levels.items():
            logger = logging.getLogger(component)
            logger.setLevel(getattr(logging, level.upper()))
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    def add_performance_handler(self, log_file: Path) -> None:
        """
        Add a dedicated handler for performance metrics.
        
        Args:
            log_file: Path to performance log file
        """
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        
        # JSON format for performance metrics
        perf_formatter = StructuredFormatter(format_type='json')
        perf_handler.setFormatter(perf_formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        
        self.handlers['performance'] = perf_handler
    
    def add_audit_handler(self, log_file: Path) -> None:
        """
        Add a dedicated handler for audit logs.
        
        Args:
            log_file: Path to audit log file
        """
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,  # Keep more audit logs
            encoding='utf-8'
        )
        audit_handler.setLevel(logging.INFO)
        
        # JSON format for audit logs
        audit_formatter = StructuredFormatter(format_type='json')
        audit_handler.setFormatter(audit_formatter)
        
        # Create audit logger
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        
        self.handlers['audit'] = audit_handler
    
    def log_performance(self, 
                       component: str,
                       operation: str,
                       duration: float,
                       **kwargs) -> None:
        """
        Log performance metrics.
        
        Args:
            component: Component name
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        perf_logger = logging.getLogger('performance')
        
        extra_data = {
            'component': component,
            'operation': operation,
            'duration_seconds': duration,
            **kwargs
        }
        
        perf_logger.info("Performance metric", extra=extra_data)
    
    def log_audit(self,
                  user: str,
                  action: str,
                  resource: str,
                  result: str,
                  **kwargs) -> None:
        """
        Log audit events.
        
        Args:
            user: User or system performing action
            action: Action performed
            resource: Resource accessed
            result: Result of action
            **kwargs: Additional context
        """
        audit_logger = logging.getLogger('audit')
        
        extra_data = {
            'user': user,
            'action': action,
            'resource': resource,
            'result': result,
            **kwargs
        }
        
        audit_logger.info("Audit event", extra=extra_data)


# Global logger manager instance
logger_manager = LoggerManager()


def setup_logging(level: str = 'INFO',
                 log_file: Optional[Path] = None,
                 console_output: bool = True,
                 json_format: bool = False,
                 component_levels: Optional[Dict[str, str]] = None) -> None:
    """
    Convenience function to setup logging.
    
    Args:
        level: Default logging level
        log_file: Path to log file
        console_output: Whether to output to console
        json_format: Whether to use JSON formatting
        component_levels: Per-component logging levels
    """
    logger_manager.setup_logging(
        level=level,
        log_file=log_file,
        console_output=console_output,
        json_format=json_format,
        component_levels=component_levels
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logger_manager.get_logger(name)


def log_performance(component: str, operation: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics.
    
    Args:
        component: Component name
        operation: Operation name
        duration: Duration in seconds
        **kwargs: Additional metrics
    """
    logger_manager.log_performance(component, operation, duration, **kwargs)


def log_audit(user: str, action: str, resource: str, result: str, **kwargs) -> None:
    """
    Log audit events.
    
    Args:
        user: User or system performing action
        action: Action performed
        resource: Resource accessed
        result: Result of action
        **kwargs: Additional context
    """
    logger_manager.log_audit(user, action, resource, result, **kwargs)


class LoggingContext:
    """
    Context manager for adding context to log messages.
    
    Automatically adds context information to all log messages
    within the context block.
    """
    
    def __init__(self, **context):
        """
        Initialize logging context.
        
        Args:
            **context: Context key-value pairs
        """
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context manager."""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        logging.setLogRecordFactory(self.old_factory)


# Convenience decorators
def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger to use (defaults to function's module logger)
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            
            start_time = time.time()
            func_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.debug(f"{func.__name__} completed in {duration:.4f}s")
                
                # Log performance metrics
                log_performance(
                    component=func.__module__,
                    operation=func.__name__,
                    duration=duration,
                    success=True
                )
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(f"{func.__name__} failed after {duration:.4f}s: {str(e)}")
                
                # Log performance metrics for failed calls
                log_performance(
                    component=func.__module__,
                    operation=func.__name__,
                    duration=duration,
                    success=False,
                    error=str(e)
                )
                
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    setup_logging(
        level='DEBUG',
        log_file=Path('test.log'),
        component_levels={
            'models': 'INFO',
            'features': 'DEBUG',
            'training': 'INFO'
        }
    )
    
    logger = get_logger(__name__)
    logger.info("Logging system test")
    
    # Test performance logging
    log_performance('test', 'example_operation', 0.123, items_processed=100)
    
    # Test audit logging
    log_audit('system', 'model_training', 'tcnae_model', 'success', epoch=10)
    
    # Test context manager
    with LoggingContext(session_id='test-123', user='testuser'):
        logger.info("This message will include context")
    
    print("Logging test completed")