"""
Integration tests for Redis caching layer.
Tests cache operations, error handling, and performance under load.
"""
import asyncio
import json
import logging
import time
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from redis.exceptions import RedisError

from config.cache_config import CACHE_TTL, MEMORY_LIMITS
from src.cache.middleware import CacheMiddleware
from src.cache.redis_manager import RedisManager
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

@pytest.fixture
async def redis_manager(event_loop):
    """Initialize Redis manager with test configuration."""
    manager = await RedisManager.create()
    try:
        await manager.redis.flushall()  # Clean state
        return manager
    except Exception as e:
        logger.error(f"Failed to initialize redis manager: {e}")
        await manager.close()
        raise
    finally:
        event_loop.call_later(0, manager.close)

@pytest.fixture
async def cache_middleware():
    """Initialize cache middleware for testing."""
    middleware = CacheMiddleware()
    try:
        return middleware
    except Exception as e:
        logger.error(f"Failed to initialize cache middleware: {e}")
        await middleware.cleanup()
        raise

class TestCacheOperations:
    """Test basic cache operations and consistency."""
    
    @pytest.mark.asyncio
    async def test_api_response_cache(self, redis_manager):
        """Test API response caching with various data types."""
        test_data = {
            'string': 'test value',
            'number': 42,
            'nested': {'key': 'value'},
            'list': [1, 2, 3]
        }
        
        # Test write
        success = await redis_manager.cache_api_response(
            'test_source',
            'test_endpoint',
            {'param': 'value'},
            test_data
        )
        assert success, "Failed to cache API response"
        
        # Test read
        cached_data = await redis_manager.get_api_response(
            'test_source',
            'test_endpoint',
            {'param': 'value'}
        )
        assert cached_data == test_data, "Cache read/write inconsistency"
        
        # Test TTL
        ttl = await redis_manager.redis.ttl(
            redis_manager._generate_api_cache_key(
                'test_source',
                'test_endpoint',
                {'param': 'value'}
            )
        )
        assert ttl > 0, "TTL not set correctly"
    
    @pytest.mark.asyncio
    async def test_embedding_cache(self, redis_manager):
        """Test embedding serialization and retrieval."""
        # Create test embedding
        test_embedding = np.random.rand(768).astype(np.float32)
        test_text = "test embedding text"
        
        # Test write
        success = await redis_manager.cache_embedding(test_text, test_embedding)
        assert success, "Failed to cache embedding"
        
        # Test read
        cached_embedding = await redis_manager.get_embedding(test_text)
        assert np.allclose(cached_embedding, test_embedding), "Embedding mismatch"
        
        # Verify memory usage
        info = await redis_manager.redis.memory_usage(
            f"embedding:{hash(test_text)}"
        )
        assert info < 4096, "Embedding storage too large"
    
    @pytest.mark.asyncio
    async def test_prediction_cache(self, redis_manager):
        """Test ML prediction caching with confidence scores."""
        test_prediction = {
            'prediction': {'label': 'HIRING', 'score': 0.95},
            'confidence': 0.95
        }
        
        # Test high-confidence caching
        success = await redis_manager.cache_prediction(
            'test_model',
            'test_input_hash',
            test_prediction['prediction'],
            test_prediction['confidence']
        )
        assert success, "Failed to cache prediction"
        
        # Verify cached prediction
        cached = await redis_manager.get_prediction(
            'test_model',
            'test_input_hash'
        )
        assert cached['prediction'] == test_prediction['prediction']
        
        # Test low-confidence prediction
        low_conf_prediction = {
            'prediction': {'label': 'UNSURE', 'score': 0.3},
            'confidence': 0.3
        }
        await redis_manager.cache_prediction(
            'test_model',
            'low_conf_hash',
            low_conf_prediction['prediction'],
            low_conf_prediction['confidence']
        )
        
        # Verify TTL differences
        high_conf_ttl = await redis_manager.redis.ttl(
            f"prediction:test_model:test_input_hash"
        )
        low_conf_ttl = await redis_manager.redis.ttl(
            f"prediction:test_model:low_conf_hash"
        )
        assert high_conf_ttl > low_conf_ttl, "Confidence-based TTL not working"

class TestErrorScenarios:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_network_interruption(self, redis_manager):
        """Test behavior during Redis network issues."""
        with patch('redis.Redis.set', side_effect=RedisError("Network error")):
            success = await redis_manager.cache_api_response(
                'test_source',
                'test_endpoint',
                {'param': 'value'},
                {'data': 'test'}
            )
            assert not success, "Should fail gracefully on network error"
    
    @pytest.mark.asyncio
    async def test_memory_pressure(self, redis_manager):
        """Test behavior under memory pressure."""
        # Fill cache to trigger memory pressure
        large_data = "x" * 1024 * 1024  # 1MB
        tasks = []
        
        for i in range(100):  # Try to exceed memory limit
            tasks.append(
                redis_manager.cache_api_response(
                    'test_source',
                    f'endpoint_{i}',
                    {},
                    {'data': large_data}
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify LRU eviction
        info = await redis_manager.redis.info('memory')
        assert int(info['used_memory']) < MEMORY_LIMITS['max_memory'], \
            "Memory limit not enforced"
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, redis_manager):
        """Test concurrent cache access patterns."""
        async def concurrent_write(i: int):
            return await redis_manager.cache_api_response(
                'test_source',
                'test_endpoint',
                {'id': i},
                {'data': f'test_{i}'}
            )
        
        # Perform concurrent writes
        tasks = [concurrent_write(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        assert all(results), "Concurrent writes failed"
        
        # Verify data integrity
        for i in range(100):
            data = await redis_manager.get_api_response(
                'test_source',
                'test_endpoint',
                {'id': i}
            )
            assert data['data'] == f'test_{i}', f"Data corruption at index {i}"
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, redis_manager):
        """Test cache invalidation edge cases."""
        # Test model version update
        await redis_manager.cache_prediction(
            'test_model',
            'test_hash',
            {'label': 'OLD'},
            0.9
        )
        
        await redis_manager.update_model_version('test_model', '2.0.0')
        
        # Verify old prediction is invalidated
        cached = await redis_manager.get_prediction('test_model', 'test_hash')
        assert cached is None, "Cache invalidation failed"

class TestPerformanceValidation:
    """Test cache performance under load."""
    
    @pytest.mark.asyncio
    async def test_response_times(self, redis_manager):
        """Test cache response times under load."""
        # Prepare test data
        test_data = [
            {'id': i, 'data': f'test_{i}'}
            for i in range(1000)
        ]
        
        # Measure write performance
        start_time = time.time()
        write_tasks = [
            redis_manager.cache_api_response(
                'test_source',
                'test_endpoint',
                {'id': item['id']},
                item
            )
            for item in test_data
        ]
        await asyncio.gather(*write_tasks)
        write_time = time.time() - start_time
        
        assert write_time < 5.0, "Write performance too slow"
        
        # Measure read performance
        start_time = time.time()
        read_tasks = [
            redis_manager.get_api_response(
                'test_source',
                'test_endpoint',
                {'id': i}
            )
            for i in range(1000)
        ]
        await asyncio.gather(*read_tasks)
        read_time = time.time() - start_time
        
        assert read_time < 2.0, "Read performance too slow"
    
    @pytest.mark.asyncio
    async def test_memory_growth(self, redis_manager):
        """Test memory growth patterns."""
        initial_memory = int((await redis_manager.redis.info('memory'))['used_memory'])
        
        # Add test data
        for i in range(1000):
            await redis_manager.cache_api_response(
                'test_source',
                f'endpoint_{i}',
                {},
                {'data': f'test_{i}' * 100}
            )
        
        final_memory = int((await redis_manager.redis.info('memory'))['used_memory'])
        memory_growth = final_memory - initial_memory
        
        # Check memory growth is reasonable
        assert memory_growth < 50 * 1024 * 1024, "Excessive memory growth"
    
    @pytest.mark.asyncio
    async def test_pipeline_efficiency(self, redis_manager):
        """Test pipeline operation efficiency."""
        # Prepare pipeline operations
        pipeline = redis_manager.redis.pipeline()
        for i in range(1000):
            pipeline.set(f"test_key_{i}", f"test_value_{i}")
        
        # Measure pipeline execution time
        start_time = time.time()
        await pipeline.execute()
        pipeline_time = time.time() - start_time
        
        # Compare with individual operations
        start_time = time.time()
        for i in range(1000):
            await redis_manager.redis.set(f"test_key2_{i}", f"test_value_{i}")
        individual_time = time.time() - start_time
        
        assert pipeline_time < individual_time / 10, "Pipeline not efficient enough"
    
    @pytest.mark.asyncio
    async def test_connection_pool(self, redis_manager):
        """Test connection pool behavior."""
        async def make_requests():
            for _ in range(100):
                await redis_manager.get_api_response(
                    'test_source',
                    'test_endpoint',
                    {'random': np.random.rand()}
                )
        
        # Simulate multiple clients
        clients = [make_requests() for _ in range(10)]
        start_time = time.time()
        await asyncio.gather(*clients)
        total_time = time.time() - start_time
        
        assert total_time < 5.0, "Connection pool performance inadequate"
