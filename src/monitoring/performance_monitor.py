"""Performance monitoring for the sentiment analysis pipeline."""
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    processing_time: float
    memory_usage: float
    cache_hits: int
    cache_misses: int
    batch_size: int
    error_count: int

class PerformanceMonitor:
    """Monitors and exports performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metrics
        self.processing_time = Histogram(
            'sentiment_processing_time_seconds',
            'Time spent processing messages',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        
        self.memory_usage = Gauge(
            'sentiment_memory_usage_bytes',
            'Current memory usage'
        )
        
        self.cache_hits = Counter(
            'sentiment_cache_hits_total',
            'Number of cache hits'
        )
        
        self.cache_misses = Counter(
            'sentiment_cache_misses_total',
            'Number of cache misses'
        )
        
        self.error_count = Counter(
            'sentiment_errors_total',
            'Number of processing errors',
            ['error_type']
        )
        
        self.batch_size = Gauge(
            'sentiment_batch_size',
            'Current batch size'
        )
    
    async def record_metrics(
        self,
        metrics: PerformanceMetrics
    ) -> None:
        """Record performance metrics."""
        try:
            # Update Prometheus metrics
            self.processing_time.observe(metrics.processing_time)
            self.memory_usage.set(metrics.memory_usage)
            self.cache_hits.inc(metrics.cache_hits)
            self.cache_misses.inc(metrics.cache_misses)
            self.batch_size.set(metrics.batch_size)
            
            # Log metrics
            self.logger.info(
                f"Performance Metrics - "
                f"Processing time: {metrics.processing_time:.2f}s, "
                f"Memory usage: {metrics.memory_usage / 1024 / 1024:.2f}MB, "
                f"Cache hits/misses: {metrics.cache_hits}/{metrics.cache_misses}, "
                f"Batch size: {metrics.batch_size}, "
                f"Errors: {metrics.error_count}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record metrics: {str(e)}")
    
    def record_error(
        self,
        error_type: str,
        error: Exception
    ) -> None:
        """Record processing errors."""
        try:
            self.error_count.labels(error_type=error_type).inc()
            self.logger.error(f"Error ({error_type}): {str(error)}")
        except Exception as e:
            self.logger.error(f"Failed to record error: {str(e)}")
    
    async def monitor_system_health(
        self,
        redis_client: Any,
        interval: int = 60
    ) -> None:
        """Continuous system health monitoring."""
        while True:
            try:
                # Get Redis metrics
                redis_info = await redis_client.info()
                
                # Record system metrics
                await self.record_metrics(
                    PerformanceMetrics(
                        processing_time=0.0,  # Not applicable for system health
                        memory_usage=redis_info["used_memory"],
                        cache_hits=redis_info["keyspace_hits"],
                        cache_misses=redis_info["keyspace_misses"],
                        batch_size=0,  # Not applicable for system health
                        error_count=0  # Reset for each interval
                    )
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(5)  # Retry after 5 seconds
    
    @staticmethod
    def calculate_processing_rate(
        batch_size: int,
        processing_time: float
    ) -> float:
        """Calculate messages processed per second."""
        return batch_size / processing_time if processing_time > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of current performance metrics."""
        return {
            "processing_time": self.processing_time.describe(),
            "memory_usage": self.memory_usage._value.get(),
            "cache_hits": self.cache_hits._value.get(),
            "cache_misses": self.cache_misses._value.get(),
            "error_count": sum(
                self.error_count.labels(error_type=et)._value.get()
                for et in self.error_count._labelnames
            )
        }
