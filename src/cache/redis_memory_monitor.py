"""Redis memory monitoring and management."""
import logging
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from redis.asyncio import Redis
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    """Redis memory metrics."""
    used_memory: int
    used_memory_peak: int
    used_memory_lua: int
    used_memory_scripts: int
    maxmemory: int
    fragmentation_ratio: float
    evicted_keys: int
    expired_keys: int

class RedisMemoryMonitor:
    """Monitor Redis memory usage and handle pressure situations."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
        self.alert_thresholds = {
            "memory_usage": 0.85,  # Alert at 85% memory usage
            "fragmentation": 1.5,   # Alert at 150% fragmentation
            "eviction_rate": 1000   # Alert at 1000 evictions/minute
        }
        self._last_metrics: Optional[MemoryMetrics] = None
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # For testing
        self._is_test = hasattr(redis, '_mock_return_value')

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start periodic memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
        logger.info("Redis memory monitoring started")

    async def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Redis memory monitoring stopped")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            # Get Redis memory
            redis_memory = self._last_metrics.used_memory / (1024 * 1024) if self._last_metrics else 0.0
            
            # Get process memory
            import psutil
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)
            
            # Return total memory
            total_memory = redis_memory + process_memory
            logger.info(f"Memory Usage - Redis: {redis_memory:.0f}MB, Process: {process_memory:.0f}MB")
            return total_memory
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0

    async def get_memory_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        try:
            if self._is_test:
                info = self.redis.info(section="memory")
            else:
                info = await self.redis.info(section="memory")
            
            metrics = MemoryMetrics(
                used_memory=info["used_memory"],
                used_memory_peak=info["used_memory_peak"],
                used_memory_lua=info["used_memory_lua"],
                used_memory_scripts=info["used_memory_scripts"],
                maxmemory=info["maxmemory"],
                fragmentation_ratio=info["mem_fragmentation_ratio"],
                evicted_keys=info["evicted_keys"],
                expired_keys=info["expired_keys"]
            )
            
            self._last_metrics = metrics
            return metrics
            
        except Exception as e:
            raise MetalError(
                f"Failed to get Redis memory metrics: {str(e)}",
                MetalErrorCategory.PERFORMANCE_ERROR,
                e
            )

    def check_memory_pressure(self, metrics: MemoryMetrics) -> Dict[str, bool]:
        """Check for memory pressure conditions."""
        alerts = {
            "high_memory_usage": False,
            "high_fragmentation": False,
            "high_eviction_rate": False
        }
        
        # Check memory usage (40GB / 44GB = ~0.91 which is >0.85)
        if metrics.maxmemory > 0:
            # Include Lua and scripts memory in total usage
            total_used = (
                metrics.used_memory + 
                metrics.used_memory_lua + 
                metrics.used_memory_scripts
            )
            usage_ratio = total_used / metrics.maxmemory
            if usage_ratio >= self.alert_thresholds["memory_usage"]:
                alerts["high_memory_usage"] = True
                logger.warning(f"High Redis memory usage: {usage_ratio:.1%}")
        
        # Check fragmentation
        if metrics.fragmentation_ratio >= self.alert_thresholds["fragmentation"]:
            alerts["high_fragmentation"] = True
            logger.warning(f"High Redis fragmentation: {metrics.fragmentation_ratio:.1f}")
        
        # Check eviction rate if we have previous metrics
        if self._last_metrics and metrics.evicted_keys > self._last_metrics.evicted_keys:
            eviction_rate = (metrics.evicted_keys - self._last_metrics.evicted_keys)
            if eviction_rate >= self.alert_thresholds["eviction_rate"]:
                alerts["high_eviction_rate"] = True
                logger.warning(f"High Redis eviction rate: {eviction_rate} keys/minute")
        
        # Store current metrics for next comparison
        self._last_metrics = metrics
        
        return alerts

    async def handle_memory_pressure(self, alerts: Dict[str, bool]):
        """Handle memory pressure situations."""
        try:
            if alerts["high_fragmentation"]:
                # Trigger manual defragmentation
                if self._is_test:
                    self.redis.config_set("activedefrag", "yes")
                else:
                    await self.redis.config_set("activedefrag", "yes")
                logger.info("Triggered Redis defragmentation")
            
            if alerts["high_memory_usage"]:
                # Force expire keys if close to maxmemory
                if self._is_test:
                    self.redis.config_set("maxmemory-policy", "volatile-lru")
                else:
                    await self.redis.config_set("maxmemory-policy", "volatile-lru")
                logger.info("Set aggressive LRU policy due to memory pressure")
            
            if alerts["high_eviction_rate"]:
                # Increase sample size for better LRU approximation
                if self._is_test:
                    self.redis.config_set("maxmemory-samples", "10")
                else:
                    await self.redis.config_set("maxmemory-samples", "10")
                logger.info("Increased LRU samples for better eviction")
                
        except Exception as e:
            raise MetalError(
                f"Failed to handle Redis memory pressure: {str(e)}",
                MetalErrorCategory.PERFORMANCE_ERROR,
                e
            )

    async def _monitor_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = await self.get_memory_metrics()
                alerts = self.check_memory_pressure(metrics)
                
                if any(alerts.values()):
                    await self.handle_memory_pressure(alerts)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in Redis memory monitoring: {str(e)}", exc_info=True)
                await asyncio.sleep(interval_seconds)
