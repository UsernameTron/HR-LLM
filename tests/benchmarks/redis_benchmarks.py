"""
Redis cache benchmarking suite with Prometheus metrics integration.
Measures performance across various load patterns and cache scenarios.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pytest
from prometheus_client import Counter, Gauge, Histogram
from pytest_benchmark.fixture import BenchmarkFixture

from config.cache_config import PERFORMANCE_CONFIG
from src.cache.middleware import CacheMiddleware
from src.cache.redis_manager import RedisManager
from tests.cache.utils import TestDataGenerator

logger = logging.getLogger(__name__)

# Prometheus metrics
LATENCY_HISTOGRAM = Histogram(
    'redis_operation_latency_seconds',
    'Redis operation latency in seconds',
    ['operation_type']
)

THROUGHPUT_GAUGE = Gauge(
    'redis_operations_per_second',
    'Current operations per second',
    ['operation_type']
)

MEMORY_USAGE_GAUGE = Gauge(
    'redis_memory_bytes',
    'Redis memory usage in bytes',
    ['type']
)

CACHE_HIT_COUNTER = Counter(
    'redis_cache_hits_total',
    'Total number of cache hits',
    ['operation_type']
)

CACHE_MISS_COUNTER = Counter(
    'redis_cache_misses_total',
    'Total number of cache misses',
    ['operation_type']
)

CONNECTION_GAUGE = Gauge(
    'redis_connections_active',
    'Number of active Redis connections'
)

@pytest.fixture
async def benchmark_manager():
    """Initialize Redis manager for benchmarking."""
    manager = RedisManager()
    try:
        await manager.redis.flushall()
        yield manager
    finally:
        await manager.cleanup()

class RedisBenchmarkSuite:
    """Comprehensive Redis benchmarking suite."""
    
    def __init__(self, manager: RedisManager):
        self.manager = manager
        self.data_gen = TestDataGenerator()
    
    async def measure_operation(
        self,
        operation_type: str,
        func,
        *args,
        **kwargs
    ) -> float:
        """Measure operation latency with Prometheus metrics."""
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            if result is not None:
                CACHE_HIT_COUNTER.labels(operation_type).inc()
            else:
                CACHE_MISS_COUNTER.labels(operation_type).inc()
        finally:
            duration = time.time() - start_time
            LATENCY_HISTOGRAM.labels(operation_type).observe(duration)
        return duration
    
    async def benchmark_write_throughput(
        self,
        num_operations: int = 10000,
        value_size: int = 1024
    ) -> Dict:
        """Benchmark write throughput with varying payload sizes."""
        durations = []
        
        async def write_operation(i: int):
            data = self.data_gen.generate_api_response(value_size)
            duration = await self.measure_operation(
                'write',
                self.manager.cache_api_response,
                'benchmark',
                f'key_{i}',
                {},
                data
            )
            durations.append(duration)
        
        start_time = time.time()
        tasks = [write_operation(i) for i in range(num_operations)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        ops_per_second = num_operations / total_time
        THROUGHPUT_GAUGE.labels('write').set(ops_per_second)
        
        return {
            'operations': num_operations,
            'total_time': total_time,
            'ops_per_second': ops_per_second,
            'avg_latency': np.mean(durations),
            'p95_latency': np.percentile(durations, 95),
            'p99_latency': np.percentile(durations, 99)
        }
    
    async def benchmark_read_throughput(
        self,
        num_operations: int = 10000,
        hit_ratio: float = 0.8
    ) -> Dict:
        """Benchmark read throughput with controlled hit ratio."""
        # Prepare data with specified hit ratio
        num_cached = int(num_operations * hit_ratio)
        for i in range(num_cached):
            await self.manager.cache_api_response(
                'benchmark',
                f'key_{i}',
                {},
                {'data': f'value_{i}'}
            )
        
        durations = []
        hits = 0
        
        async def read_operation(i: int):
            nonlocal hits
            duration = await self.measure_operation(
                'read',
                self.manager.get_api_response,
                'benchmark',
                f'key_{i}',
                {}
            )
            durations.append(duration)
            if duration is not None:
                hits += 1
        
        start_time = time.time()
        tasks = [read_operation(i) for i in range(num_operations)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        ops_per_second = num_operations / total_time
        THROUGHPUT_GAUGE.labels('read').set(ops_per_second)
        
        return {
            'operations': num_operations,
            'total_time': total_time,
            'ops_per_second': ops_per_second,
            'hit_ratio': hits / num_operations,
            'avg_latency': np.mean(durations),
            'p95_latency': np.percentile(durations, 95),
            'p99_latency': np.percentile(durations, 99)
        }
    
    async def benchmark_pipeline_throughput(
        self,
        num_batches: int = 100,
        batch_size: int = 1000
    ) -> Dict:
        """Benchmark pipeline operation throughput."""
        durations = []
        
        async def pipeline_operation(batch_id: int):
            pipeline = self.manager.redis.pipeline()
            for i in range(batch_size):
                key = f"batch_{batch_id}_key_{i}"
                value = f"value_{i}" * 100
                pipeline.set(key, value)
            
            start_time = time.time()
            await pipeline.execute()
            duration = time.time() - start_time
            durations.append(duration)
            return duration
        
        start_time = time.time()
        tasks = [pipeline_operation(i) for i in range(num_batches)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        total_operations = num_batches * batch_size
        ops_per_second = total_operations / total_time
        THROUGHPUT_GAUGE.labels('pipeline').set(ops_per_second)
        
        return {
            'operations': total_operations,
            'total_time': total_time,
            'ops_per_second': ops_per_second,
            'avg_batch_latency': np.mean(durations),
            'p95_batch_latency': np.percentile(durations, 95),
            'p99_batch_latency': np.percentile(durations, 99)
        }
    
    async def benchmark_memory_efficiency(
        self,
        start_size: int = 1024,
        max_size: int = 1024 * 1024,
        step_factor: int = 2
    ) -> List[Dict]:
        """Benchmark memory efficiency with increasing value sizes."""
        results = []
        current_size = start_size
        
        while current_size <= max_size:
            # Clear previous data
            await self.manager.redis.flushall()
            
            # Generate and store data
            data = self.data_gen.generate_large_document(
                current_size // 1024
            )
            num_items = max(1, 1024 * 1024 // current_size)
            
            start_time = time.time()
            for i in range(num_items):
                await self.manager.cache_api_response(
                    'benchmark',
                    f'key_{i}',
                    {},
                    {'data': data}
                )
            
            # Collect memory metrics
            info = await self.manager.redis.info('memory')
            used_memory = int(info['used_memory'])
            MEMORY_USAGE_GAUGE.labels('used').set(used_memory)
            
            results.append({
                'value_size': current_size,
                'num_items': num_items,
                'total_size': current_size * num_items,
                'used_memory': used_memory,
                'memory_ratio': used_memory / (current_size * num_items),
                'time_taken': time.time() - start_time
            })
            
            current_size *= step_factor
        
        return results
    
    async def benchmark_connection_pool(
        self,
        num_clients: int = 100,
        operations_per_client: int = 1000
    ) -> Dict:
        """Benchmark connection pool performance."""
        CONNECTION_GAUGE.set(0)
        durations = []
        
        async def client_operation(client_id: int):
            """Simulate client operations."""
            client_durations = []
            CONNECTION_GAUGE.inc()
            
            try:
                for i in range(operations_per_client):
                    start_time = time.time()
                    await self.manager.get_api_response(
                        'benchmark',
                        f'client_{client_id}_op_{i}',
                        {}
                    )
                    duration = time.time() - start_time
                    client_durations.append(duration)
                    
                    if i % 100 == 0:
                        await asyncio.sleep(0.01)  # Simulate think time
            finally:
                CONNECTION_GAUGE.dec()
            
            return client_durations
        
        start_time = time.time()
        tasks = [client_operation(i) for i in range(num_clients)]
        all_durations = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Flatten durations
        durations = [d for client_durations in all_durations
                    for d in client_durations]
        
        total_operations = num_clients * operations_per_client
        return {
            'num_clients': num_clients,
            'operations_per_client': operations_per_client,
            'total_operations': total_operations,
            'total_time': total_time,
            'ops_per_second': total_operations / total_time,
            'avg_latency': np.mean(durations),
            'p95_latency': np.percentile(durations, 95),
            'p99_latency': np.percentile(durations, 99)
        }

@pytest.mark.asyncio
async def test_write_throughput(
    benchmark_manager: RedisManager,
    benchmark: BenchmarkFixture
):
    """Benchmark write throughput."""
    suite = RedisBenchmarkSuite(benchmark_manager)
    result = await suite.benchmark_write_throughput()
    
    # Verify performance meets requirements
    assert result['ops_per_second'] > 5000, \
        "Write throughput below threshold"
    assert result['p95_latency'] < 0.01, \
        "Write latency too high"

@pytest.mark.asyncio
async def test_read_throughput(
    benchmark_manager: RedisManager,
    benchmark: BenchmarkFixture
):
    """Benchmark read throughput."""
    suite = RedisBenchmarkSuite(benchmark_manager)
    result = await suite.benchmark_read_throughput()
    
    # Verify performance meets requirements
    assert result['ops_per_second'] > 10000, \
        "Read throughput below threshold"
    assert result['hit_ratio'] > 0.75, \
        "Cache hit ratio too low"

@pytest.mark.asyncio
async def test_pipeline_throughput(
    benchmark_manager: RedisManager,
    benchmark: BenchmarkFixture
):
    """Benchmark pipeline throughput."""
    suite = RedisBenchmarkSuite(benchmark_manager)
    result = await suite.benchmark_pipeline_throughput()
    
    # Verify performance meets requirements
    assert result['ops_per_second'] > 50000, \
        "Pipeline throughput below threshold"
    assert result['p95_batch_latency'] < 0.1, \
        "Pipeline batch latency too high"

@pytest.mark.asyncio
async def test_memory_efficiency(
    benchmark_manager: RedisManager,
    benchmark: BenchmarkFixture
):
    """Benchmark memory efficiency."""
    suite = RedisBenchmarkSuite(benchmark_manager)
    results = await suite.benchmark_memory_efficiency()
    
    # Verify memory efficiency
    for result in results:
        assert result['memory_ratio'] < 1.5, \
            f"Memory overhead too high for size {result['value_size']}"

@pytest.mark.asyncio
async def test_connection_pool(
    benchmark_manager: RedisManager,
    benchmark: BenchmarkFixture
):
    """Benchmark connection pool."""
    suite = RedisBenchmarkSuite(benchmark_manager)
    result = await suite.benchmark_connection_pool()
    
    # Verify connection pool performance
    assert result['ops_per_second'] > 1000, \
        "Connection pool throughput too low"
    assert result['p95_latency'] < 0.05, \
        "Connection pool latency too high"
