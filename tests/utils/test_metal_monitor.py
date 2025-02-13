"""Tests for Metal Performance Monitoring."""
import pytest
import time
from unittest.mock import patch, Mock, MagicMock
import torch
import psutil

from src.utils.metal_monitor import (
    MetalMonitor,
    metal_performance_context,
    monitor_metal_performance
)
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@pytest.fixture
def mock_torch_mps():
    """Fixture for mocking torch.mps functionality."""
    with patch("torch.backends.mps.is_available") as mock_available, \
         patch("torch.mps.current_allocated_memory") as mock_memory, \
         patch("torch.mps.empty_cache") as mock_empty_cache, \
         patch("torch.device") as mock_device, \
         patch("torch.randn") as mock_randn:
        
        mock_available.return_value = True
        mock_memory.return_value = 1024 * 1024 * 1024  # 1GB
        mock_device.return_value = "mps"
        mock_randn.return_value = torch.tensor([1.0])
        
        yield {
            "is_available": mock_available,
            "current_memory": mock_memory,
            "empty_cache": mock_empty_cache,
            "device": mock_device,
            "randn": mock_randn
        }

def test_metal_monitor_initialization(mock_torch_mps):
    """Test MetalMonitor initialization."""
    monitor = MetalMonitor()
    
    assert monitor.start_time is None
    assert monitor.end_time is None
    assert monitor.peak_memory == 0
    assert isinstance(monitor.metrics_history, list)
    assert monitor.device == "mps"
    
    # Test CPU fallback when MPS is not available
    mock_torch_mps["is_available"].return_value = False
    monitor = MetalMonitor()
    assert monitor.device == "cpu"

def test_metal_monitor_start_stop(mock_torch_mps):
    """Test monitoring start and stop functionality."""
    monitor = MetalMonitor()
    
    # Test start
    monitor.start()
    assert monitor.start_time is not None
    assert len(monitor.metrics_history) == 0
    
    # Test double start
    with pytest.raises(MetalError) as exc_info:
        monitor.start()
    assert exc_info.value.category == MetalErrorCategory.VALIDATION_ERROR
    
    # Test stop
    time.sleep(0.1)  # Ensure measurable duration
    metrics = monitor.stop()
    
    assert monitor.start_time is None
    assert metrics["duration"] >= 0.1
    assert metrics["peak_memory"] == 0
    assert metrics["device"] == "mps"
    assert isinstance(metrics["metrics_history"], list)
    
    # Test stop without start
    with pytest.raises(MetalError) as exc_info:
        monitor.stop()
    assert exc_info.value.category == MetalErrorCategory.VALIDATION_ERROR

def test_metal_monitor_update_metrics(mock_torch_mps):
    """Test metrics update functionality."""
    monitor = MetalMonitor()
    
    # Test update without start
    with pytest.raises(MetalError) as exc_info:
        monitor.update_metrics()
    assert exc_info.value.category == MetalErrorCategory.VALIDATION_ERROR
    
    # Test normal update
    monitor.start()
    mock_torch_mps["current_memory"].return_value = 2 * 1024 * 1024 * 1024  # 2GB
    monitor.update_metrics()
    
    assert monitor.peak_memory == 2 * 1024  # Should be in MB
    assert len(monitor.metrics_history) == 1
    assert "current_memory" in monitor.metrics_history[0]
    assert "peak_memory" in monitor.metrics_history[0]
    assert "timestamp" in monitor.metrics_history[0]

@patch("psutil.Process")
def test_metal_monitor_cpu_metrics(mock_process, mock_torch_mps):
    """Test CPU metrics monitoring."""
    mock_torch_mps["is_available"].return_value = False
    mock_process.return_value.memory_info.return_value = MagicMock(rss=1024 * 1024 * 1024)
    
    monitor = MetalMonitor()
    monitor.start()
    monitor.update_metrics()
    
    assert monitor.device == "cpu"
    assert monitor.peak_memory == 1024  # 1GB in MB
    assert len(monitor.metrics_history) == 1

def test_metal_monitor_memory_pressure(mock_torch_mps):
    """Test memory pressure detection and handling."""
    monitor = MetalMonitor()
    monitor.start()
    
    # Simulate high memory usage
    mock_torch_mps["current_memory"].return_value = 19 * 1024 * 1024 * 1024  # 19GB
    monitor.update_metrics()
    
    # Should trigger memory pressure handling
    assert mock_torch_mps["empty_cache"].called
    metrics = monitor.stop()
    assert metrics["peak_memory"] == 19 * 1024  # Should be in MB

@pytest.mark.asyncio
async def test_metal_performance_context():
    """Test performance monitoring context manager."""
    with metal_performance_context() as monitor:
        assert monitor.start_time is not None
        assert isinstance(monitor, MetalMonitor)
        
    # Context should be cleaned up
    assert monitor.start_time is None

def test_monitor_metal_performance_decorator():
    """Test performance monitoring decorator."""
    call_count = 0
    
    @monitor_metal_performance(retries=2, retry_delay=0.1)
    def test_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise MetalError("Retry me", MetalErrorCategory.TRANSIENT_ERROR)
        return "success"
    
    result = test_function()
    assert result == "success"
    assert call_count == 2

def test_metal_monitor_error_recovery(mock_torch_mps):
    """Test error recovery mechanisms."""
    monitor = MetalMonitor()
    
    # Test device initialization recovery
    mock_torch_mps["randn"].side_effect = [
        RuntimeError("First attempt failed"),
        torch.tensor([1.0])
    ]
    
    monitor = MetalMonitor()  # Should retry and succeed
    assert monitor.device == "mps"
    
    # Test memory pressure recovery
    monitor.start()
    mock_torch_mps["current_memory"].side_effect = [
        RuntimeError("Memory error"),
        1024 * 1024 * 1024
    ]
    
    monitor.update_metrics()  # Should handle error and retry
    assert monitor.peak_memory > 0

@pytest.mark.parametrize("error_type,error_msg,expected_category", [
    (RuntimeError, "out of memory", MetalErrorCategory.MEMORY_PRESSURE),
    (RuntimeError, "device not found", MetalErrorCategory.DEVICE_UNAVAILABLE),
    (ValueError, "invalid operation", MetalErrorCategory.VALIDATION_ERROR),
    (TimeoutError, "operation timeout", MetalErrorCategory.PERFORMANCE_ERROR),
    (RuntimeError, "temporary failure", MetalErrorCategory.TRANSIENT_ERROR),
])
def test_metal_monitor_error_handling(
    error_type,
    error_msg,
    expected_category,
    mock_torch_mps
):
    """Test error handling for different error types."""
    monitor = MetalMonitor()
    monitor.start()
    
    mock_torch_mps["current_memory"].side_effect = error_type(error_msg)
    
    with pytest.raises(MetalError) as exc_info:
        monitor.update_metrics()
    
    assert exc_info.value.category == expected_category
    assert error_msg in str(exc_info.value)
