"""Benchmark tests for Redis memory management."""
import pytest
import asyncio
from unittest.mock import Mock
from src.cache.redis_memory_monitor import RedisMemoryMonitor, MemoryMetrics

@pytest.mark.benchmark(
    group="redis-memory",
    min_rounds=100,
    warmup=True
)
def test_memory_pressure_detection_benchmark(benchmark):
    """Benchmark memory pressure detection performance."""
    redis_mock = Mock()
    monitor = RedisMemoryMonitor(redis_mock)
    
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
    
    def run_check():
        return monitor.check_memory_pressure(metrics)
    
    result = benchmark(run_check)
    assert result is not None

@pytest.mark.benchmark(
    group="redis-memory",
    min_rounds=100,
    warmup=True
)
@pytest.mark.asyncio
async def test_memory_metrics_retrieval_benchmark(benchmark):
    """Benchmark memory metrics retrieval performance."""
    redis_mock = Mock()
    info_data = {
        "used_memory": 41000000000,
        "used_memory_peak": 42000000000,
        "used_memory_lua": 100000,
        "used_memory_scripts": 200000,
        "maxmemory": 47244640256,
        "mem_fragmentation_ratio": 1.6,
        "evicted_keys": 2000,
        "expired_keys": 500
    }
    redis_mock.info = Mock(return_value=info_data)
    monitor = RedisMemoryMonitor(redis_mock)
    
    def run_metrics():
        metrics = monitor.get_memory_metrics()
        return metrics
    
    result = benchmark(run_metrics)
    assert result is not None

@pytest.mark.benchmark(
    group="redis-memory",
    min_rounds=100,
    warmup=True
)
@pytest.mark.asyncio
async def test_pressure_handling_benchmark(benchmark):
    """Benchmark memory pressure handling performance."""
    redis_mock = Mock()
    redis_mock.config_set = Mock(return_value=True)
    monitor = RedisMemoryMonitor(redis_mock)
    
    alerts = {
        "high_memory_usage": True,
        "high_fragmentation": True,
        "high_eviction_rate": True
    }
    
    def run_handler():
        monitor.handle_memory_pressure(alerts)
        return True
    
    result = benchmark(run_handler)
    assert result is True
