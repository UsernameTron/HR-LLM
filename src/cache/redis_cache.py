"""
Redis cache implementation for sentiment analysis results.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['operation']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['operation']
)

CACHE_LATENCY = Histogram(
    'cache_operation_duration_seconds',
    'Cache operation latency in seconds',
    ['operation']
)

class RedisCache:
    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        """Initialize Redis cache connection."""
        self.redis_url = f"redis://{host}:{port}/{db}"
        self._redis: Optional[redis.Redis] = None
        
    async def connect(self) -> None:
        """Create Redis connection."""
        if not self._redis:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        try:
            if not self._redis:
                await self.connect()
            
            with CACHE_LATENCY.labels('get').time():
                result = await self._redis.get(key)
                
            if result:
                CACHE_HITS.labels('get').inc()
                return json.loads(result)
            
            CACHE_MISSES.labels('get').inc()
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    async def set(self, key: str, value: Dict[str, Any], expire: int = 3600) -> bool:
        """Set value in cache with expiration."""
        try:
            if not self._redis:
                await self.connect()
            
            with CACHE_LATENCY.labels('set').time():
                result = await self._redis.set(
                    key,
                    json.dumps(value),
                    ex=expire
                )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False
    
    async def get_many(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Get multiple values from cache."""
        try:
            if not self._redis:
                await self.connect()
            
            with CACHE_LATENCY.labels('get_many').time():
                pipe = self._redis.pipeline()
                for key in keys:
                    pipe.get(key)
                results = await pipe.execute()
            
            # Process results
            processed = []
            for result in results:
                if result:
                    CACHE_HITS.labels('get_many').inc()
                    processed.append(json.loads(result))
                else:
                    CACHE_MISSES.labels('get_many').inc()
                    processed.append(None)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error getting multiple from cache: {str(e)}")
            return [None] * len(keys)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if not self._redis:
                await self.connect()
            
            with CACHE_LATENCY.labels('delete').time():
                result = await self._redis.delete(key)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False
