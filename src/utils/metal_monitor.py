import logging
import time
import psutil
import torch
from functools import wraps
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager

from src.config.settings import get_settings, parse_memory_size
from src.utils.metal_error_handler import (
    MetalError,
    MetalErrorCategory,
    log_metal_error,
    with_metal_error_handling
)

logger = logging.getLogger(__name__)
settings = get_settings()

class MetalMonitor:
    """Monitor Metal Performance Shaders usage and system resources."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.metrics_history: list[Dict[str, Any]] = []
        self.device = self._initialize_device()
    
    @with_metal_error_handling()
    def _initialize_device(self) -> str:
        """Initialize the device with error handling."""
        if torch.backends.mps.is_available() and not settings.metal.force_cpu:
            # Test device functionality
            device = torch.device("mps")
            test_tensor = torch.randn(1, 1, device=device)
            del test_tensor
            torch.mps.empty_cache()
            return "mps"
        return "cpu"
    
    @with_metal_error_handling()
    def start(self) -> None:
        """Start monitoring with error handling."""
        if self.start_time is not None:
            raise MetalError(
                "Monitoring already started",
                MetalErrorCategory.VALIDATION_ERROR
            )
        
        self.start_time = time.time()
        self.peak_memory = 0
        self.metrics_history.clear()
        logger.info(f"Starting Metal monitoring on device: {self.device}")
    
    @with_metal_error_handling()
    def stop(self) -> dict:
        """Stop monitoring and return metrics with error handling."""
        if self.start_time is None:
            raise MetalError(
                "Monitoring not started",
                MetalErrorCategory.VALIDATION_ERROR
            )
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        metrics = {
            "duration": duration,
            "peak_memory": self.peak_memory,
            "device": self.device,
            "metrics_history": self.metrics_history
        }
        
        logger.info(
            f"Metal monitoring stopped. Duration: {duration:.2f}s, "
            f"Peak Memory: {self.peak_memory:.2f}MB"
        )
        
        # Reset state
        self.start_time = None
        self.end_time = None
        
        return metrics
    
    @with_metal_error_handling()
    def update_metrics(self) -> None:
        """Update monitoring metrics with error handling."""
        if self.start_time is None:
            raise MetalError(
                "Cannot update metrics: monitoring not started",
                MetalErrorCategory.VALIDATION_ERROR
            )
        
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        metrics = {
            "timestamp": time.time(),
            "current_memory": current_memory,
            "peak_memory": self.peak_memory
        }
        
        self.metrics_history.append(metrics)
        
        # Check for memory pressure
        if self._check_memory_pressure(current_memory):
            self._handle_memory_pressure()
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            if self.device == "mps":
                if hasattr(torch.mps, "current_allocated_memory"):
                    return torch.mps.current_allocated_memory() / (1024 * 1024)
                raise MetalError(
                    "Cannot get MPS memory usage",
                    MetalErrorCategory.DEVICE_UNAVAILABLE
                )
            else:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            error_str = str(e).lower()
            category = MetalErrorCategory.PERFORMANCE_ERROR
            
            if "memory" in error_str or "allocation" in error_str:
                category = MetalErrorCategory.MEMORY_PRESSURE
            elif "temporary" in error_str or "transient" in error_str:
                category = MetalErrorCategory.TRANSIENT_ERROR
            elif "device" in error_str or "unavailable" in error_str:
                category = MetalErrorCategory.DEVICE_UNAVAILABLE
                
            raise MetalError(
                f"Error getting memory usage: {str(e)}",
                category,
                e
            )
    
    def _check_memory_pressure(self, current_memory: float) -> bool:
        """Check if system is under memory pressure."""
        try:
            if self.device == "mps":
                max_memory = parse_memory_size(settings.metal.gpu_memory)
                return current_memory > max_memory * 0.9
            else:
                memory = psutil.virtual_memory()
                return memory.percent > 90
        except Exception as e:
            logger.error(f"Error checking memory pressure: {str(e)}", exc_info=True)
            return True
    
    def _handle_memory_pressure(self) -> None:
        """Handle memory pressure situation."""
        try:
            if self.device == "mps":
                logger.warning("Memory pressure detected, clearing MPS cache")
                torch.mps.empty_cache()
            
            if settings.metal.mps_fallback_to_cpu and self.device == "mps":
                logger.warning("Switching to CPU due to memory pressure")
                self.device = "cpu"
        except Exception as e:
            logger.error(f"Error handling memory pressure: {str(e)}", exc_info=True)

@contextmanager
def metal_performance_context():
    """Context manager for monitoring Metal performance with error handling."""
    monitor = MetalMonitor()
    try:
        monitor.start()
        yield monitor
    except Exception as e:
        if not isinstance(e, MetalError):
            e = MetalError(
                str(e),
                MetalErrorCategory.UNKNOWN,
                e
            )
        log_metal_error(e)
        raise
    finally:
        if monitor.start_time is not None:
            try:
                metrics = monitor.stop()
                if settings.metal.enable_performance_logging:
                    logger.info("Performance metrics", extra={"metrics": metrics})
            except Exception as e:
                logger.error(f"Error stopping monitor: {str(e)}", exc_info=True)

def monitor_metal_performance(
    retries: Optional[int] = None,
    retry_delay: Optional[float] = None
) -> Callable:
    """Decorator to monitor Metal performance with error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @with_metal_error_handling(
            max_retries=retries or settings.metal.max_retries,
            retry_delay=retry_delay or settings.metal.retry_delay
        )
        def wrapper(*args, **kwargs):
            with metal_performance_context() as monitor:
                return func(*args, **kwargs)
        return wrapper
    return decorator
