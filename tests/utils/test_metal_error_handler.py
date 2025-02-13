"""Tests for Metal error handling utilities."""
import pytest
import time
from unittest.mock import Mock, patch

from src.utils.metal_error_handler import (
    MetalError,
    MetalErrorCategory,
    categorize_error,
    with_metal_error_handling,
    log_metal_error
)

def test_error_categorization():
    """Test error categorization logic."""
    # Memory errors
    error = Exception("CUDA out of memory")
    assert categorize_error(error) == MetalErrorCategory.MEMORY_PRESSURE
    
    # Device errors
    error = Exception("MPS device not found")
    assert categorize_error(error) == MetalErrorCategory.DEVICE_UNAVAILABLE
    
    # Initialization errors
    error = Exception("Failed to initialize Metal context")
    assert categorize_error(error) == MetalErrorCategory.INITIALIZATION_ERROR
    
    # Validation errors
    error = Exception("Invalid configuration value")
    assert categorize_error(error) == MetalErrorCategory.VALIDATION_ERROR
    
    # Performance errors
    error = Exception("Operation timed out")
    assert categorize_error(error) == MetalErrorCategory.PERFORMANCE_ERROR
    
    # Transient errors
    error = Exception("Temporary failure, retry later")
    assert categorize_error(error) == MetalErrorCategory.TRANSIENT_ERROR
    
    # Unknown errors
    error = Exception("Some random error")
    assert categorize_error(error) == MetalErrorCategory.UNKNOWN

def test_metal_error_creation():
    """Test MetalError creation and properties."""
    original_error = ValueError("Test error")
    error = MetalError(
        "Error message",
        MetalErrorCategory.VALIDATION_ERROR,
        original_error
    )
    
    assert str(error) == "Error message"
    assert error.category == MetalErrorCategory.VALIDATION_ERROR
    assert error.original_error == original_error

@patch("utils.metal_error_handler.logger")
def test_log_metal_error(mock_logger):
    """Test error logging functionality."""
    error = MetalError(
        "Test error",
        MetalErrorCategory.PERFORMANCE_ERROR,
        ValueError("Original error")
    )
    
    context = {"test_key": "test_value"}
    log_metal_error(error, context)
    
    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[0]
    assert "Test error" in call_args[0]
    assert "error_context" in mock_logger.error.call_args[1]["extra"]

def test_retry_decorator():
    """Test retry logic in decorator."""
    mock_func = Mock(side_effect=[
        MetalError("Retry me", MetalErrorCategory.TRANSIENT_ERROR),
        MetalError("Retry me again", MetalErrorCategory.TRANSIENT_ERROR),
        "success"
    ])
    
    @with_metal_error_handling(max_retries=3, retry_delay=0.1)
    def test_func():
        return mock_func()
    
    start_time = time.time()
    result = test_func()
    duration = time.time() - start_time
    
    assert result == "success"
    assert mock_func.call_count == 3
    assert duration >= 0.2  # Account for exponential backoff

def test_retry_decorator_max_retries():
    """Test retry decorator respects max retries."""
    mock_func = Mock(side_effect=MetalError("Always fail", MetalErrorCategory.TRANSIENT_ERROR))
    
    @with_metal_error_handling(max_retries=2, retry_delay=0.1)
    def test_func():
        return mock_func()
    
    with pytest.raises(MetalError) as exc_info:
        test_func()
    
    assert mock_func.call_count == 2
    assert exc_info.value.category == MetalErrorCategory.TRANSIENT_ERROR

def test_retry_decorator_non_retryable():
    """Test retry decorator doesn't retry non-retryable errors."""
    mock_func = Mock(side_effect=MetalError("Don't retry me", MetalErrorCategory.VALIDATION_ERROR))
    
    @with_metal_error_handling(max_retries=3, retry_delay=0.1)
    def test_func():
        return mock_func()
    
    with pytest.raises(MetalError) as exc_info:
        test_func()
    
    assert mock_func.call_count == 1
    assert exc_info.value.category == MetalErrorCategory.VALIDATION_ERROR

@pytest.mark.parametrize("error_msg,expected_category", [
    ("Out of memory", MetalErrorCategory.MEMORY_PRESSURE),
    ("Device not found", MetalErrorCategory.DEVICE_UNAVAILABLE),
    ("Invalid config", MetalErrorCategory.VALIDATION_ERROR),
    ("Operation timeout", MetalErrorCategory.PERFORMANCE_ERROR),
    ("Temporary failure", MetalErrorCategory.TRANSIENT_ERROR),
    ("Random error", MetalErrorCategory.UNKNOWN),
])
def test_error_categorization_parametrized(error_msg, expected_category):
    """Test error categorization with different error messages."""
    error = Exception(error_msg)
    assert categorize_error(error) == expected_category
