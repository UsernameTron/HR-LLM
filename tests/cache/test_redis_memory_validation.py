"""Validation tests for Redis memory pressure handling."""
import pytest
import logging
from unittest.mock import Mock, patch
from src.cache.redis_memory_monitor import RedisMemoryMonitor, MemoryMetrics
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@pytest.mark.asyncio
async def test_gradual_memory_pressure():
    """Test memory pressure handling with gradually increasing memory usage."""
    redis_mock = Mock()
    monitor = RedisMemoryMonitor(redis_mock)
    
    # Test with different memory levels
    memory_levels = [
        (30000000000, False),  # ~30GB, should not trigger
        (35000000000, False),  # ~35GB, should not trigger
        (41000000000, True),   # ~41GB, should trigger
        (43000000000, True),   # ~43GB, should trigger
    ]
    
    for memory, should_trigger in memory_levels:
        metrics = MemoryMetrics(
            used_memory=memory,
            used_memory_peak=memory + 1000000000,
            used_memory_lua=100000,
            used_memory_scripts=200000,
            maxmemory=47244640256,  # 44GB
            fragmentation_ratio=1.3,
            evicted_keys=1000,
            expired_keys=500
        )
        
        alerts = monitor.check_memory_pressure(metrics)
        assert alerts["high_memory_usage"] == should_trigger

@pytest.mark.asyncio
async def test_fragmentation_handling():
    """Test handling of different fragmentation levels."""
    redis_mock = Mock()
    redis_mock.config_set = Mock(return_value=True)
    monitor = RedisMemoryMonitor(redis_mock)
    
    frag_levels = [
        (1.3, False),  # Normal fragmentation
        (1.5, True),   # Threshold
        (1.7, True),   # High fragmentation
        (2.0, True),   # Very high fragmentation
    ]
    
    for frag_ratio, should_trigger in frag_levels:
        metrics = MemoryMetrics(
            used_memory=35000000000,
            used_memory_peak=36000000000,
            used_memory_lua=100000,
            used_memory_scripts=200000,
            maxmemory=47244640256,
            fragmentation_ratio=frag_ratio,
            evicted_keys=1000,
            expired_keys=500
        )
        
        alerts = monitor.check_memory_pressure(metrics)
        assert alerts["high_fragmentation"] == should_trigger
        
        if should_trigger:
            await monitor.handle_memory_pressure(alerts)
            redis_mock.config_set.assert_any_call("activedefrag", "yes")

@pytest.mark.asyncio
async def test_eviction_rate_monitoring():
    """Test monitoring of eviction rates."""
    redis_mock = Mock()
    monitor = RedisMemoryMonitor(redis_mock)
    
    # Simulate increasing eviction rates
    eviction_scenarios = [
        (100, 200),    # Low rate
        (200, 800),    # Medium rate
        (800, 2000),   # High rate
        (2000, 5000),  # Very high rate
    ]
    
    for prev_evictions, curr_evictions in eviction_scenarios:
        # Set previous metrics
        prev_metrics = MemoryMetrics(
            used_memory=35000000000,
            used_memory_peak=36000000000,
            used_memory_lua=100000,
            used_memory_scripts=200000,
            maxmemory=47244640256,
            fragmentation_ratio=1.3,
            evicted_keys=prev_evictions,
            expired_keys=500
        )
        monitor.check_memory_pressure(prev_metrics)
        
        # Check current metrics
        curr_metrics = MemoryMetrics(
            used_memory=35000000000,
            used_memory_peak=36000000000,
            used_memory_lua=100000,
            used_memory_scripts=200000,
            maxmemory=47244640256,
            fragmentation_ratio=1.3,
            evicted_keys=curr_evictions,
            expired_keys=500
        )
        
        alerts = monitor.check_memory_pressure(curr_metrics)
        expected_alert = (curr_evictions - prev_evictions) >= monitor.alert_thresholds["eviction_rate"]
        assert alerts["high_eviction_rate"] == expected_alert
