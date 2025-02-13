"""
Redis caching manager optimized for M4 Pro hardware.
Implements intelligent caching strategies for API responses, ML predictions, and embeddings.
"""
import json
import logging
import os
import pickle
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import redis
from redis.exceptions import RedisError

from src.cache.redis_memory_monitor import RedisMemoryMonitor
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

from src.config.settings import get_settings, API_CONFIGS

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self):
        self.settings = get_settings()
        self.redis = None
        self.pubsub_client = None
        self.monitor = None
        self._config_path = os.path.join(os.path.dirname(__file__), '../../config/redis.conf')
        self.stats = {
            'hits': 0,
            'misses': 0,
            'api_cache_ratio': 0.0,
            'embedding_cache_ratio': 0.0,
            'prediction_cache_ratio': 0.0,
            'memory_usage': 0.0,
            'fragmentation_ratio': 0.0
        }
    
    @classmethod
    async def create(cls) -> 'RedisManager':
        """Create and initialize a new RedisManager instance."""
        instance = cls()
        await instance.initialize()
        return instance
    
    async def initialize(self):
        """Initialize Redis connections and monitoring."""
        try:
            # Initialize main Redis connection
            self.redis = redis.Redis.from_url(
                self.settings.REDIS_URL,
                decode_responses=False,  # Required for binary data like embeddings
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Separate connection for pub/sub to avoid blocking main operations
            self.pubsub_client = self.redis.pubsub()
            
            # Load Redis configuration
            await self._load_config()
            
            # Initialize memory monitor
            self.monitor = RedisMemoryMonitor(self.redis)
            await self.monitor.start_monitoring()
            
            # Verify connection
            await self.redis.ping()
            
            logger.info("Redis manager initialized successfully")
            
        except Exception as e:
            await self.close()
            raise MetalError(
                f"Failed to initialize Redis manager: {str(e)}",
                MetalErrorCategory.INITIALIZATION_ERROR,
                e
            )
        await self.redis.ping()
    
    async def close(self):
        """Close Redis connections and stop monitoring."""
        try:
            if self.monitor:
                await self.monitor.stop_monitoring()
            
            if self.redis:
                await self.redis.close()
            
            if self.pubsub_client:
                await self.pubsub_client.close()
                
        except Exception as e:
            logger.error(f"Error closing Redis manager: {str(e)}", exc_info=True)
            raise
    
    async def _load_config(self):
        """Load Redis configuration from file."""
        try:
            with open(self._config_path, 'r') as f:
                config = f.read().splitlines()
            
            # Apply each config line
            for line in config:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split(maxsplit=1)
                    await self.redis.config_set(key, value)
                    
            logger.info("Redis configuration loaded successfully")
            
        except Exception as e:
            raise MetalError(
                f"Failed to load Redis configuration: {str(e)}",
                MetalErrorCategory.INITIALIZATION_ERROR,
                e
            )
    
    async def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if not self.monitor:
            return {}
            
        try:
            metrics = await self.monitor.get_memory_metrics()
            
            self.stats['memory_usage'] = metrics.used_memory / metrics.maxmemory
            self.stats['fragmentation_ratio'] = metrics.fragmentation_ratio
            
            return {
                'memory_usage': self.stats['memory_usage'],
                'fragmentation_ratio': self.stats['fragmentation_ratio'],
                'used_memory_mb': metrics.used_memory / (1024 * 1024),
                'maxmemory_mb': metrics.maxmemory / (1024 * 1024),
                'evicted_keys': metrics.evicted_keys,
                'expired_keys': metrics.expired_keys
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}", exc_info=True)
            return {}
    
    async def get_api_response(
        self,
        source: str,
        endpoint: str,
        params: Dict
    ) -> Optional[Dict]:
        """
        Get cached API response with intelligent expiration.
        Implements adaptive TTL based on data volatility.
        """
        cache_key = self._generate_api_cache_key(source, endpoint, params)
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                self.stats['hits'] += 1
                return json.loads(cached_data)
            
            self.stats['misses'] += 1
            return None
            
        except RedisError as e:
            logger.error(f"Redis error in get_api_response: {str(e)}")
            return None
    
    async def cache_api_response(
        self,
        source: str,
        endpoint: str,
        params: Dict,
        response: Dict
    ) -> bool:
        """Cache API response with adaptive TTL."""
        cache_key = self._generate_api_cache_key(source, endpoint, params)
        ttl = self._get_adaptive_ttl(source, endpoint)
        
        try:
            return self.redis.setex(
                cache_key,
                ttl,
                json.dumps(response)
            )
        except RedisError as e:
            logger.error(f"Redis error in cache_api_response: {str(e)}")
            return False
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding with high-performance binary storage.
        Optimized for M4 Pro's unified memory architecture.
        """
        cache_key = f"embedding:{hash(text)}"
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                self.stats['hits'] += 1
                return pickle.loads(cached_data)
            
            self.stats['misses'] += 1
            return None
            
        except RedisError as e:
            logger.error(f"Redis error in get_embedding: {str(e)}")
            return None
    
    async def cache_embedding(
        self,
        text: str,
        embedding: np.ndarray
    ) -> bool:
        """
        Cache embedding with compression.
        Uses efficient binary serialization for numpy arrays.
        """
        cache_key = f"embedding:{hash(text)}"
        
        try:
            # Compress and store embedding
            compressed_data = pickle.dumps(embedding, protocol=4)
            return self.redis.setex(
                cache_key,
                timedelta(days=7),  # Embeddings are stable
                compressed_data
            )
        except RedisError as e:
            logger.error(f"Redis error in cache_embedding: {str(e)}")
            return False
    
    async def get_prediction(
        self,
        model_name: str,
        input_hash: str
    ) -> Optional[Dict]:
        """
        Get cached ML prediction with version tracking.
        Implements prediction versioning for model updates.
        """
        cache_key = f"prediction:{model_name}:{input_hash}"
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                self.stats['hits'] += 1
                return json.loads(cached_data)
            
            self.stats['misses'] += 1
            return None
            
        except RedisError as e:
            logger.error(f"Redis error in get_prediction: {str(e)}")
            return None
    
    async def cache_prediction(
        self,
        model_name: str,
        input_hash: str,
        prediction: Dict,
        confidence: float
    ) -> bool:
        """
        Cache ML prediction with confidence-based TTL.
        Higher confidence predictions get longer TTL.
        """
        cache_key = f"prediction:{model_name}:{input_hash}"
        
        # Adjust TTL based on prediction confidence
        ttl = self._get_confidence_based_ttl(confidence)
        
        try:
            return self.redis.setex(
                cache_key,
                ttl,
                json.dumps({
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_version': self._get_model_version(model_name)
                })
            )
        except RedisError as e:
            logger.error(f"Redis error in cache_prediction: {str(e)}")
            return False
    
    def _generate_api_cache_key(
        self,
        source: str,
        endpoint: str,
        params: Dict
    ) -> str:
        """Generate deterministic cache key for API responses."""
        param_str = json.dumps(params, sort_keys=True)
        return f"api:{source}:{endpoint}:{hash(param_str)}"
    
    def _get_adaptive_ttl(self, source: str, endpoint: str) -> int:
        """
        Get adaptive TTL based on data source and endpoint.
        Implements intelligent TTL adjustment based on data volatility.
        """
        base_ttl = API_CONFIGS[source]['cache_ttl']
        
        # Adjust TTL based on endpoint characteristics
        if 'company' in endpoint:  # Company data changes less frequently
            return base_ttl * 2
        elif 'news' in endpoint:  # News data is more volatile
            return base_ttl // 2
        return base_ttl
    
    def _get_confidence_based_ttl(self, confidence: float) -> int:
        """Calculate TTL based on prediction confidence."""
        base_ttl = 3600  # 1 hour base TTL
        if confidence > 0.9:
            return base_ttl * 24  # 24 hours for high confidence
        elif confidence > 0.7:
            return base_ttl * 12  # 12 hours for medium confidence
        return base_ttl  # 1 hour for low confidence
    
    def _get_model_version(self, model_name: str) -> str:
        """Get current model version for prediction versioning."""
        try:
            return self.redis.get(f"model_version:{model_name}") or "1.0.0"
        except RedisError:
            return "1.0.0"
    
    async def update_model_version(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Update model version and invalidate related predictions."""
        try:
            # Update version
            self.redis.set(f"model_version:{model_name}", version)
            
            # Invalidate predictions from old version
            pattern = f"prediction:{model_name}:*"
            for key in self.redis.scan_iter(pattern):
                self.redis.delete(key)
                
        except RedisError as e:
            logger.error(f"Failed to update model version: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            self.stats['api_cache_ratio'] = self.stats['hits'] / total_requests
        
        return self.stats
    
    async def cleanup_expired_keys(self) -> None:
        """Cleanup expired keys to maintain cache efficiency."""
        try:
            # Implement intelligent cleanup based on memory usage
            info = self.redis.info('memory')
            used_memory = info['used_memory'] / 1024 / 1024  # MB
            
            if used_memory > settings.REDIS_MEMORY_LIMIT:
                # Remove least recently used items
                for key in self.redis.scan_iter("prediction:*"):
                    if not self.redis.ttl(key):
                        self.redis.delete(key)
                        
        except RedisError as e:
            logger.error(f"Failed to cleanup expired keys: {str(e)}")
    
    def close(self) -> None:
        """Close Redis connections."""
        try:
            self.redis.close()
            self.pubsub_client.close()
        except RedisError as e:
            logger.error(f"Error closing Redis connections: {str(e)}")
            raise
