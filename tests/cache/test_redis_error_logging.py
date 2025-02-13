"""Tests for Redis error logging and categorization."""
import pytest
import logging
from unittest.mock import Mock, patch
from src.cache.redis_memory_monitor import RedisMemoryMonitor, MemoryMetrics
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@pytest.fixture
def caplog(caplog):
    """Configure logging capture."""
    caplog.set_level(logging.WARNING)
    return caplog

@pytest.mark.asyncio
async def test_connection_error_logging(caplog):
    """Test logging of connection errors."""
    redis_mock = Mock()
    redis_mock.info.side_effect = ConnectionError("Redis connection failed")
    monitor = RedisMemoryMonitor(redis_mock)
    
    with pytest.raises(MetalError) as exc_info:
        await monitor.get_memory_metrics()
    
    assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR
    assert "Redis connection failed" in str(exc_info.value)
    assert any("Failed to get Redis memory metrics" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_configuration_error_logging(caplog):
    """Test logging of configuration errors."""
    redis_mock = Mock()
    redis_mock.config_set.side_effect = RuntimeError("Invalid configuration")
    monitor = RedisMemoryMonitor(redis_mock)
    
    alerts = {
        "high_memory_usage": True,
        "high_fragmentation": True,
        "high_eviction_rate": False
    }
    
    with pytest.raises(MetalError) as exc_info:
        await monitor.handle_memory_pressure(alerts)
    
    assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR
    assert "Invalid configuration" in str(exc_info.value)
    assert any("Failed to handle Redis memory pressure" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_memory_pressure_warning_logs(caplog):
    """Test warning logs for memory pressure conditions."""
    redis_mock = Mock()
    monitor = RedisMemoryMonitor(redis_mock)
    
    metrics = MemoryMetrics(
        used_memory=41000000000,  # High memory usage
        used_memory_peak=42000000000,
        used_memory_lua=100000,
        used_memory_scripts=200000,
        maxmemory=47244640256,
        fragmentation_ratio=1.6,  # High fragmentation
        evicted_keys=2000,
        expired_keys=500
    )
    
    alerts = monitor.check_memory_pressure(metrics)
    
    # Check warning logs
    assert any("High Redis memory usage" in record.message for record in caplog.records)
    assert any("High Redis fragmentation" in record.message for record in caplog.records)
    
    # Verify log levels
    memory_warnings = [r for r in caplog.records if "memory usage" in r.message]
    frag_warnings = [r for r in caplog.records if "fragmentation" in r.message]
    
    assert all(r.levelno == logging.WARNING for r in memory_warnings)
    assert all(r.levelno == logging.WARNING for r in frag_warnings)

@pytest.mark.asyncio
async def test_error_categorization():
    """Test proper categorization of different error types."""
    redis_mock = Mock()
    monitor = RedisMemoryMonitor(redis_mock)
    
    # Test connection errors
    redis_mock.info.side_effect = ConnectionError("Connection failed")
    with pytest.raises(MetalError) as exc_info:
        await monitor.get_memory_metrics()
    assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR
    
    # Test configuration errors
    redis_mock.info.side_effect = ValueError("Invalid configuration")
    with pytest.raises(MetalError) as exc_info:
        await monitor.get_memory_metrics()
    assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR
    
    # Test memory allocation errors
    redis_mock.info.side_effect = MemoryError("Out of memory")
    with pytest.raises(MetalError) as exc_info:
        await monitor.get_memory_metrics()
    assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR
