"""
Cache middleware for integrating Redis caching with data processors.
Optimized for M4 Pro hardware with efficient memory usage.
"""
import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

import numpy as np

from src.config.settings import get_settings
from src.cache.redis_manager import RedisManager
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)
T = TypeVar('T')

class CacheMiddleware:
    def __init__(self):
        self.redis_manager = RedisManager()
        self.metrics = MetricsTracker()
        self._cache_warmed = False
        
    async def init(self):
        """Initialize the cache middleware asynchronously"""
        if not self._cache_warmed:
            await self._warm_cache()
            self._cache_warmed = True
    
    def cache_api_response(self, source: str, endpoint: str):
        """
        Decorator for caching API responses.
        Implements intelligent caching with automatic invalidation.
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # Generate cache key from function arguments
                params = {**kwargs}
                
                try:
                    # Check cache first
                    cached_response = await self.redis_manager.get_api_response(
                        source, endpoint, params
                    )
                    
                    if cached_response is not None:
                        self.metrics.record_success('cache_hit', f"{source}_{endpoint}")
                        return cached_response
                    
                    # Cache miss - call original function
                    response = await func(*args, **kwargs)
                    
                    # Cache the response
                    if response:
                        await self.redis_manager.cache_api_response(
                            source, endpoint, params, response
                        )
                        self.metrics.record_success('cache_store', f"{source}_{endpoint}")
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Cache middleware error: {str(e)}")
                    self.metrics.record_failure('cache', f"{source}_{endpoint}", str(e))
                    # Fallback to original function
                    return await func(*args, **kwargs)
                
            return wrapper
        return decorator
    
    def cache_embedding(self):
        """
        Decorator for caching embeddings.
        Optimized for M4 Pro's unified memory architecture.
        """
        def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
            @wraps(func)
            async def wrapper(text: str, *args, **kwargs) -> np.ndarray:
                try:
                    # Check embedding cache
                    cached_embedding = await self.redis_manager.get_embedding(text)
                    
                    if cached_embedding is not None:
                        self.metrics.record_success('cache_hit', 'embedding')
                        return cached_embedding
                    
                    # Cache miss - generate embedding
                    embedding = await func(text, *args, **kwargs)
                    
                    # Cache the embedding
                    if embedding is not None:
                        await self.redis_manager.cache_embedding(text, embedding)
                        self.metrics.record_success('cache_store', 'embedding')
                    
                    return embedding
                    
                except Exception as e:
                    logger.error(f"Embedding cache error: {str(e)}")
                    self.metrics.record_failure('cache', 'embedding', str(e))
                    return await func(text, *args, **kwargs)
                
            return wrapper
        return decorator
    
    def cache_prediction(self, model_name: str):
        """
        Decorator for caching ML predictions.
        Implements confidence-based caching strategy.
        """
        def decorator(func: Callable[..., Dict]) -> Callable[..., Dict]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Dict:
                # Generate deterministic input hash
                input_hash = str(hash(str(args) + str(sorted(kwargs.items()))))
                
                try:
                    # Check prediction cache
                    cached_prediction = await self.redis_manager.get_prediction(
                        model_name, input_hash
                    )
                    
                    if cached_prediction is not None:
                        self.metrics.record_success('cache_hit', f"prediction_{model_name}")
                        return cached_prediction['prediction']
                    
                    # Cache miss - generate prediction
                    result = await func(*args, **kwargs)
                    
                    # Cache prediction if confidence meets threshold
                    confidence = result.get('confidence', 0.0)
                    if confidence >= settings.CONFIDENCE_THRESHOLD:
                        await self.redis_manager.cache_prediction(
                            model_name,
                            input_hash,
                            result['prediction'],
                            confidence
                        )
                        self.metrics.record_success('cache_store', f"prediction_{model_name}")
                    
                    return result['prediction']
                    
                except Exception as e:
                    logger.error(f"Prediction cache error: {str(e)}")
                    self.metrics.record_failure('cache', f"prediction_{model_name}", str(e))
                    result = await func(*args, **kwargs)
                    return result['prediction']
                
            return wrapper
        return decorator
    
    async def _warm_cache(self) -> None:
        """
        Warm up cache with frequently accessed data.
        Implements intelligent cache warming based on access patterns.
        """
        try:
            # Warm up company data cache
            common_companies = await self._get_common_companies()
            for company in common_companies:
                await self._warm_company_cache(company)
            
            # Warm up embedding cache for common phrases
            common_phrases = await self._get_common_phrases()
            for phrase in common_phrases:
                await self._warm_embedding_cache(phrase)
            
            logger.info("Cache warming completed successfully")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {str(e)}")
    
    async def _get_common_companies(self) -> list:
        """Get list of commonly accessed companies."""
        # Implement logic to identify frequently accessed companies
        return []
    
    async def _get_common_phrases(self) -> list:
        """Get list of commonly used phrases for embedding cache."""
        # Implement logic to identify frequent phrases
        return []
    
    async def _warm_company_cache(self, company: str) -> None:
        """Pre-cache company data."""
        # Implement company data caching
        pass
    
    async def _warm_embedding_cache(self, phrase: str) -> None:
        """Pre-cache embeddings for common phrases."""
        # Implement embedding caching
        pass
    
    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        try:
            await self.redis_manager.cleanup_expired_keys()
            self.redis_manager.close()
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
            raise
