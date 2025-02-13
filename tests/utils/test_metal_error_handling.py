import pytest
import torch
import logging
from unittest.mock import Mock, patch
from src.utils.metal_monitor import MetalMonitor, MetalError, MetalErrorCategory
from src.utils.metal_error_handler import categorize_error

logger = logging.getLogger(__name__)

@pytest.fixture
def metal_monitor():
    return MetalMonitor()

@pytest.mark.asyncio
async def test_memory_pressure_detection(metal_monitor):
    """Test detection of memory pressure conditions."""
    with patch('torch.mps.current_allocated_memory') as mock_memory:
        # Simulate memory error
        mock_memory.side_effect = RuntimeError("Out of memory on MPS device")
        
        with pytest.raises(MetalError) as exc_info:
            metal_monitor._get_current_memory()
        
        assert exc_info.value.category == MetalErrorCategory.MEMORY_PRESSURE
        assert "memory" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_device_unavailable(metal_monitor):
    """Test handling of device unavailability."""
    with patch('torch.mps.current_allocated_memory') as mock_memory:
        # Simulate device unavailable
        mock_memory.side_effect = RuntimeError("MPS device unavailable")
        
        with pytest.raises(MetalError) as exc_info:
            metal_monitor._get_current_memory()
        
        assert exc_info.value.category == MetalErrorCategory.DEVICE_UNAVAILABLE
        assert "device" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_transient_error_handling(metal_monitor):
    """Test handling of temporary/transient errors."""
    with patch('torch.mps.current_allocated_memory') as mock_memory:
        # Simulate transient error
        mock_memory.side_effect = RuntimeError("Temporary device error")
        
        with pytest.raises(MetalError) as exc_info:
            metal_monitor._get_current_memory()
        
        assert exc_info.value.category == MetalErrorCategory.TRANSIENT_ERROR
        assert "temporary" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_performance_error_fallback(metal_monitor):
    """Test fallback to performance error for uncategorized errors."""
    with patch('torch.mps.current_allocated_memory') as mock_memory:
        # Simulate generic error
        mock_memory.side_effect = Exception("Unknown error occurred")
        
        with pytest.raises(MetalError) as exc_info:
            metal_monitor._get_current_memory()
        
        assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR

@pytest.mark.asyncio
async def test_memory_pressure_handling(metal_monitor):
    """Test the memory pressure handling mechanism."""
    with patch('torch.mps.empty_cache') as mock_cache:
        with patch('torch.mps.current_allocated_memory') as mock_memory:
            mock_memory.return_value = 15 * 1024 * 1024 * 1024  # 15GB
            
            # Should trigger memory pressure handling
            metal_monitor._handle_memory_pressure()
            
            # Verify cache was cleared
            mock_cache.assert_called_once()

@pytest.mark.asyncio
async def test_error_categorization():
    """Test error categorization logic."""
    # Test memory pressure error
    error = RuntimeError("Out of memory on MPS device")
    assert categorize_error(error) == MetalErrorCategory.MEMORY_PRESSURE
    
    # Test transient error
    error = RuntimeError("Temporary failure in MPS operation")
    assert categorize_error(error) == MetalErrorCategory.TRANSIENT_ERROR
    
    # Test device unavailable
    error = RuntimeError("MPS device unavailable")
    assert categorize_error(error) == MetalErrorCategory.DEVICE_UNAVAILABLE
    
    # Test unknown error fallback
    error = RuntimeError("Unknown error")
    assert categorize_error(error) == MetalErrorCategory.UNKNOWN
