"""Metal error handling and recovery utilities."""
import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, Type, Union
from functools import wraps
import torch

logger = logging.getLogger(__name__)

class MetalErrorCategory(Enum):
    """Categories of Metal-related errors."""
    DEVICE_UNAVAILABLE = "device_unavailable"
    MEMORY_PRESSURE = "memory_pressure"
    INITIALIZATION_ERROR = "initialization_error"
    VALIDATION_ERROR = "validation_error"
    PERFORMANCE_ERROR = "performance_error"
    TRANSIENT_ERROR = "transient_error"
    INGESTION_ERROR = "ingestion_error"
    PROCESSING_ERROR = "processing_error"
    LOAD_TEST_ERROR = "load_test_error"
    EXPLANATION_ERROR = "explanation_error"
    UNKNOWN = "unknown"

class MetalError(Exception):
    """Base class for Metal-related errors."""
    def __init__(self, message: str, category: MetalErrorCategory, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.category = category
        self.original_error = original_error

def categorize_error(error: Exception) -> MetalErrorCategory:
    """Categorize an error based on its type and message."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # First check error type
    if isinstance(error, torch.cuda.OutOfMemoryError) or "memoryerror" in error_type:
        return MetalErrorCategory.MEMORY_PRESSURE
    elif "devicenotfounderror" in error_type or "device" in error_type:
        return MetalErrorCategory.DEVICE_UNAVAILABLE
    elif "valueerror" in error_type or "validationerror" in error_type:
        return MetalErrorCategory.VALIDATION_ERROR
    elif "timeouterror" in error_type or "performanceerror" in error_type:
        return MetalErrorCategory.PERFORMANCE_ERROR
    
    # Then check error message
    if "memory" in error_str or "allocation" in error_str or "out of memory" in error_str:
        return MetalErrorCategory.MEMORY_PRESSURE
    elif "temporary" in error_str or "transient" in error_str or "retry" in error_str:
        return MetalErrorCategory.TRANSIENT_ERROR
    elif "device" in error_str or "mps" in error_str:
        return MetalErrorCategory.DEVICE_UNAVAILABLE
    elif "initialize" in error_str or "setup" in error_str:
        return MetalErrorCategory.INITIALIZATION_ERROR
    elif "validate" in error_str or "invalid" in error_str:
        return MetalErrorCategory.VALIDATION_ERROR
    elif "performance" in error_str or "timeout" in error_str:
        return MetalErrorCategory.PERFORMANCE_ERROR
    elif "temporary" in error_str or "transient" in error_str or "retry" in error_str:
        return MetalErrorCategory.TRANSIENT_ERROR
    
    return MetalErrorCategory.UNKNOWN

def with_metal_error_handling(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retryable_categories: Optional[set[MetalErrorCategory]] = None
) -> Callable:
    """Decorator for handling Metal-related errors with retry logic."""
    if retryable_categories is None:
        retryable_categories = {MetalErrorCategory.TRANSIENT_ERROR, MetalErrorCategory.MEMORY_PRESSURE}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    category = categorize_error(e)
                    last_error = MetalError(str(e), category, e)
                    
                    logger.error(
                        f"Metal error in {func.__name__} (attempt {attempt + 1}/{max_retries}): "
                        f"[{category.value}] {str(e)}",
                        exc_info=True
                    )
                    
                    if category not in retryable_categories:
                        logger.error(f"Non-retryable error category: {category.value}")
                        break
                    
                    if attempt < max_retries - 1:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                        
                        # Try to recover before retry
                        try:
                            if category == MetalErrorCategory.MEMORY_PRESSURE:
                                torch.mps.empty_cache()
                                logger.info("Cleared MPS cache before retry")
                        except Exception as recovery_error:
                            logger.warning(f"Error during recovery: {str(recovery_error)}")
            
            if last_error:
                raise last_error

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    category = categorize_error(e)
                    last_error = MetalError(str(e), category, e)
                    
                    logger.error(
                        f"Metal error in {func.__name__} (attempt {attempt + 1}/{max_retries}): "
                        f"[{category.value}] {str(e)}",
                        exc_info=True
                    )
                    
                    if category not in retryable_categories:
                        logger.error(f"Non-retryable error category: {category.value}")
                        break
                    
                    if attempt < max_retries - 1:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        
                        # Try to recover before retry
                        try:
                            if category == MetalErrorCategory.MEMORY_PRESSURE:
                                torch.mps.empty_cache()
                                logger.info("Cleared MPS cache before retry")
                        except Exception as recovery_error:
                            logger.warning(f"Error during recovery: {str(recovery_error)}")
            
            if last_error:
                raise last_error

        # Return appropriate wrapper based on whether the function is async
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def log_metal_error(
    error: Union[Exception, MetalError],
    context: Optional[dict] = None
) -> None:
    """Log a Metal-related error with detailed context."""
    if isinstance(error, MetalError):
        category = error.category
        original_error = error.original_error
    else:
        category = categorize_error(error)
        original_error = error
    
    error_context = {
        "error_category": category.value,
        "error_type": type(original_error).__name__,
        "error_message": str(original_error),
        "device_available": torch.backends.mps.is_available(),
        **(context or {})
    }
    
    try:
        if torch.backends.mps.is_available():
            error_context.update({
                "allocated_memory": torch.mps.current_allocated_memory() / (1024 * 1024),  # MB
                "device_name": torch.mps.get_device_name()
            })
    except Exception as e:
        logger.warning(f"Could not get device metrics: {str(e)}")
    
    logger.error(
        f"Metal error occurred: [{category.value}] {str(error)}",
        extra={"error_context": error_context},
        exc_info=True
    )
