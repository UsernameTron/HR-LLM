"""Tests for Redis memory monitoring."""
import pytest
from unittest.mock import Mock, patch
from src.cache.redis_memory_monitor import RedisMemoryMonitor, MemoryMetrics
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@pytest.fixture
def redis_mock():
    """Mock Redis client."""
    redis = Mock()
    
    # Mock info method
    info_data = {
        "used_memory": 1000000000,  # ~1GB
        "used_memory_peak": 1200000000,
        "used_memory_lua": 100000,
        "used_memory_scripts": 200000,
        "maxmemory": 47244640256,  # 44GB
        "mem_fragmentation_ratio": 1.2,
        "evicted_keys": 100,
        "expired_keys": 200
    }
    redis.info = Mock(return_value=info_data)
    
    # Mock config_set method
    redis.config_set = Mock(return_value=True)
    return redis

@pytest.fixture
def memory_monitor(redis_mock):
    """Initialize memory monitor with mock Redis."""
    return RedisMemoryMonitor(redis_mock)

@pytest.mark.asyncio
async def test_get_memory_metrics(memory_monitor, redis_mock):
    """Test getting memory metrics."""
    metrics = await memory_monitor.get_memory_metrics()
    
    assert isinstance(metrics, MemoryMetrics)
    assert metrics.used_memory == 1000000000
    assert metrics.maxmemory == 47244640256
    assert metrics.fragmentation_ratio == 1.2
    assert metrics.evicted_keys == 100

@pytest.mark.asyncio
async def test_memory_pressure_detection(memory_monitor):
    """Test memory pressure detection."""
    # Simulate high memory usage
    metrics = MemoryMetrics(
        used_memory=41000000000,  # ~41GB
        used_memory_peak=42000000000,
        used_memory_lua=100000,
        used_memory_scripts=200000,
        maxmemory=47244640256,  # 44GB
        fragmentation_ratio=1.6,
        evicted_keys=2000,
        expired_keys=500
    )
    
    alerts = memory_monitor.check_memory_pressure(metrics)
    
    assert alerts["high_memory_usage"]  # Should alert at >85% usage
    assert alerts["high_fragmentation"]  # Should alert at >1.5 ratio

@pytest.mark.asyncio
async def test_handle_memory_pressure(memory_monitor, redis_mock):
    """Test memory pressure handling."""
    alerts = {
        "high_memory_usage": True,
        "high_fragmentation": True,
        "high_eviction_rate": True
    }
    
    await memory_monitor.handle_memory_pressure(alerts)
    
    # Verify configuration changes
    redis_mock.config_set.assert_any_call("activedefrag", "yes")
    redis_mock.config_set.assert_any_call("maxmemory-policy", "volatile-lru")
    redis_mock.config_set.assert_any_call("maxmemory-samples", "10")

@pytest.mark.asyncio
async def test_monitoring_lifecycle(memory_monitor):
    """Test starting and stopping monitoring."""
    # Start monitoring
    await memory_monitor.start_monitoring(interval_seconds=1)
    assert memory_monitor._monitoring
    assert memory_monitor._monitor_task is not None
    
    # Stop monitoring
    await memory_monitor.stop_monitoring()
    assert not memory_monitor._monitoring
    assert memory_monitor._monitor_task is None or memory_monitor._monitor_task.done()

@pytest.mark.asyncio
async def test_error_handling(memory_monitor, redis_mock):
    """Test error handling during monitoring."""
    # Set up error condition
    error_msg = "Redis connection error"
    redis_mock.info.side_effect = RuntimeError(error_msg)
    
    with pytest.raises(MetalError) as exc_info:
        await memory_monitor.get_memory_metrics()
    
    assert exc_info.value.category == MetalErrorCategory.PERFORMANCE_ERROR
    assert error_msg in str(exc_info.value)
    
    # Verify error was logged
    redis_mock.info.assert_called_once_with(section="memory")
