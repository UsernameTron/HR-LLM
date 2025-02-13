"""
Stress tests for Redis caching layer.
Tests system behavior under high load and adverse conditions.
"""
import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import pytest
from redis.exceptions import RedisError

from config.cache_config import (CACHE_TTL, MEMORY_LIMITS,
                               PERFORMANCE_CONFIG)
from src.cache.middleware import CacheMiddleware
from src.cache.redis_manager import RedisManager
from tests.cache.utils import TestDataGenerator

logger = logging.getLogger(__name__)

@pytest.fixture
async def redis_manager():
    """Initialize Redis manager for stress testing."""
    manager = RedisManager()
    try:
        await manager.redis.flushall()
        yield manager
    finally:
        await manager.cleanup()

class TestConcurrentAccess:
    """Test cache behavior under heavy concurrent load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, redis_manager):
        """Test concurrent read/write operations."""
        data_gen = TestDataGenerator()
        num_operations = 10000
        
        async def write_operation(i: int):
            """Simulate write operation."""
            try:
                data = data_gen.generate_api_response(size=100)
                return await redis_manager.cache_api_response(
                    'stress_test',
                    f'endpoint_{i}',
                    {'id': i},
                    data
                )
            except Exception as e:
                logger.error(f"Write operation failed: {str(e)}")
                return False
        
        async def read_operation(i: int):
            """Simulate read operation."""
            try:
                return await redis_manager.get_api_response(
                    'stress_test',
                    f'endpoint_{i}',
                    {'id': i}
                )
            except Exception as e:
                logger.error(f"Read operation failed: {str(e)}")
                return None
        
        # Execute concurrent writes
        write_tasks = [
            write_operation(i)
            for i in range(num_operations)
        ]
        write_results = await asyncio.gather(*write_tasks)
        successful_writes = sum(1 for r in write_results if r)
        
        assert successful_writes > 0.95 * num_operations, \
            "Too many write operations failed"
        
        # Execute concurrent reads
        read_tasks = [
            read_operation(i)
            for i in range(num_operations)
        ]
        read_results = await asyncio.gather(*read_tasks)
        successful_reads = sum(1 for r in read_results if r is not None)
        
        assert successful_reads > 0.95 * num_operations, \
            "Too many read operations failed"
    
    @pytest.mark.asyncio
    async def test_pipeline_batching(self, redis_manager):
        """Test pipeline performance under load."""
        batch_size = PERFORMANCE_CONFIG['max_pipeline_size']
        num_batches = 100
        
        async def execute_batch(batch_id: int):
            """Execute a batch of pipeline operations."""
            pipeline = redis_manager.redis.pipeline()
            for i in range(batch_size):
                key = f"batch_{batch_id}_key_{i}"
                value = f"value_{i}" * 100
                pipeline.set(key, value)
            
            try:
                return await pipeline.execute()
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                return None
        
        # Execute concurrent batches
        batch_tasks = [
            execute_batch(i)
            for i in range(num_batches)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*batch_tasks)
        execution_time = time.time() - start_time
        
        successful_batches = sum(1 for r in results if r is not None)
        operations_per_second = (
            successful_batches * batch_size
        ) / execution_time
        
        assert operations_per_second > 10000, \
            "Pipeline throughput too low"
    
    @pytest.mark.asyncio
    async def test_connection_pool_saturation(self, redis_manager):
        """Test behavior under connection pool saturation."""
        pool_size = PERFORMANCE_CONFIG['connection_pool_size']
        num_clients = pool_size * 2  # Intentionally exceed pool size
        
        async def client_operation():
            """Simulate client operations."""
            for _ in range(100):
                try:
                    await redis_manager.get_api_response(
                        'stress_test',
                        'endpoint',
                        {'random': random.random()}
                    )
                    await asyncio.sleep(0.01)  # Simulate processing
                except Exception as e:
                    logger.error(f"Client operation failed: {str(e)}")
        
        # Execute concurrent clients
        client_tasks = [
            client_operation()
            for _ in range(num_clients)
        ]
        
        start_time = time.time()
        await asyncio.gather(*client_tasks)
        execution_time = time.time() - start_time
        
        # Verify reasonable execution time despite pool saturation
        assert execution_time < 15.0, \
            "Connection pool handling too slow"
    
    @pytest.mark.asyncio
    async def test_cache_stampede(self, redis_manager):
        """Test cache stampede prevention."""
        num_clients = 100
        test_key = "stampede_test_key"
        
        async def get_or_compute():
            """Simulate expensive computation with cache."""
            try:
                # Try to get from cache
                result = await redis_manager.get_api_response(
                    'stress_test',
                    test_key,
                    {}
                )
                
                if result is None:
                    # Simulate expensive computation
                    await asyncio.sleep(0.5)
                    result = {'computed': time.time()}
                    await redis_manager.cache_api_response(
                        'stress_test',
                        test_key,
                        {},
                        result
                    )
                
                return result
            except Exception as e:
                logger.error(f"Cache operation failed: {str(e)}")
                return None
        
        # Simulate concurrent cache misses
        tasks = [get_or_compute() for _ in range(num_clients)]
        results = await asyncio.gather(*tasks)
        
        # Verify that only one computation occurred
        unique_timestamps = {
            r['computed'] for r in results if r is not None
        }
        assert len(unique_timestamps) == 1, \
            "Cache stampede prevention failed"

class TestMemoryPressure:
    """Test cache behavior under memory pressure."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_ingestion(self, redis_manager):
        """Test handling of large dataset ingestion."""
        data_size = 100 * 1024  # 100KB per item
        num_items = 1000  # Total ~100MB
        
        async def ingest_batch(batch_id: int):
            """Ingest a batch of large items."""
            try:
                data = TestDataGenerator.generate_large_document(
                    data_size // 1024
                )
                return await redis_manager.cache_api_response(
                    'stress_test',
                    f'large_data_{batch_id}',
                    {},
                    {'data': data}
                )
            except Exception as e:
                logger.error(f"Ingestion failed: {str(e)}")
                return False
        
        # Monitor memory during ingestion
        initial_memory = int(
            (await redis_manager.redis.info('memory'))['used_memory']
        )
        
        tasks = [ingest_batch(i) for i in range(num_items)]
        results = await asyncio.gather(*tasks)
        
        final_memory = int(
            (await redis_manager.redis.info('memory'))['used_memory']
        )
        memory_growth = final_memory - initial_memory
        
        # Verify reasonable memory growth
        assert memory_growth < data_size * num_items * 1.5, \
            "Excessive memory growth"
    
    @pytest.mark.asyncio
    async def test_eviction_policy(self, redis_manager):
        """Test LRU eviction policy effectiveness."""
        # Fill cache to near capacity
        data_gen = TestDataGenerator()
        
        async def fill_cache(start_id: int, count: int):
            """Fill cache with test data."""
            for i in range(start_id, start_id + count):
                data = data_gen.generate_api_response(1024)  # 1KB
                await redis_manager.cache_api_response(
                    'stress_test',
                    f'eviction_test_{i}',
                    {},
                    data
                )
                if i % 100 == 0:
                    await asyncio.sleep(0.1)  # Prevent overwhelming
        
        # Fill cache and verify eviction
        await fill_cache(0, 1000)
        initial_keys = await redis_manager.redis.keys('*')
        
        # Add more data to trigger eviction
        await fill_cache(1000, 1000)
        final_keys = await redis_manager.redis.keys('*')
        
        # Verify that eviction occurred
        assert len(final_keys) < len(initial_keys) + 1000, \
            "Eviction policy not working"
    
    @pytest.mark.asyncio
    async def test_memory_fragmentation(self, redis_manager):
        """Test memory fragmentation patterns."""
        # Create varying size objects
        sizes = [100, 1000, 10000, 100000]  # Bytes
        
        async def write_varying_sizes():
            """Write objects of varying sizes."""
            for size in sizes:
                data = TestDataGenerator.generate_large_document(
                    size // 1024
                )
                await redis_manager.cache_api_response(
                    'stress_test',
                    f'frag_test_{size}',
                    {},
                    {'data': data}
                )
        
        # Monitor fragmentation ratio
        initial_info = await redis_manager.redis.info('memory')
        
        for _ in range(100):  # Create many objects
            await write_varying_sizes()
        
        final_info = await redis_manager.redis.info('memory')
        
        # Check fragmentation ratio
        frag_ratio = float(final_info['mem_fragmentation_ratio'])
        assert frag_ratio < 1.5, \
            "Excessive memory fragmentation"

class TestNetworkResilience:
    """Test cache behavior under network issues."""
    
    @pytest.mark.asyncio
    async def test_connection_drops(self, redis_manager):
        """Test handling of connection drops."""
        async def operation_with_drops():
            """Simulate operations with connection drops."""
            try:
                # Simulate connection drop
                redis_manager.redis.connection_pool.disconnect()
                
                # Attempt operation
                return await redis_manager.cache_api_response(
                    'stress_test',
                    'reconnect_test',
                    {},
                    {'data': 'test'}
                )
            except Exception as e:
                logger.error(f"Operation failed: {str(e)}")
                return False
        
        results = await asyncio.gather(
            *[operation_with_drops() for _ in range(100)]
        )
        
        # Verify some operations succeeded after reconnection
        assert any(results), \
            "Failed to handle connection drops"
    
    @pytest.mark.asyncio
    async def test_latency_spikes(self, redis_manager):
        """Test handling of latency spikes."""
        async def delayed_operation():
            """Simulate operation with latency."""
            try:
                await asyncio.sleep(random.random())  # Random delay
                return await redis_manager.get_api_response(
                    'stress_test',
                    'latency_test',
                    {}
                )
            except Exception as e:
                logger.error(f"Operation failed: {str(e)}")
                return None
        
        start_time = time.time()
        results = await asyncio.gather(
            *[delayed_operation() for _ in range(100)]
        )
        execution_time = time.time() - start_time
        
        # Verify operations completed despite latency
        assert execution_time < 10.0, \
            "Poor latency handling"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, redis_manager):
        """Test circuit breaker effectiveness."""
        failure_threshold = 5
        failures = 0
        
        async def operation_with_circuit():
            """Simulate operation with circuit breaker."""
            nonlocal failures
            try:
                if failures >= failure_threshold:
                    # Simulate recovery
                    failures = 0
                    await asyncio.sleep(1)
                
                result = await redis_manager.cache_api_response(
                    'stress_test',
                    'circuit_test',
                    {},
                    {'data': 'test'}
                )
                
                if not result:
                    failures += 1
                
                return result
            except Exception:
                failures += 1
                return False
        
        results = []
        for _ in range(100):
            result = await operation_with_circuit()
            results.append(result)
            await asyncio.sleep(0.1)
        
        # Verify circuit breaker prevented cascade failure
        assert sum(results) > 50, \
            "Circuit breaker ineffective"
