"""Health check implementation for the sentiment analysis pipeline."""
import asyncio
import logging
from typing import Dict, List, Optional
import psutil
import time

from .config import metrics, alert_thresholds, health_config

logger = logging.getLogger(__name__)

class HealthChecker:
    """Implements system health checks and monitoring."""
    
    def __init__(self):
        self.last_check_times: Dict[str, float] = {}
        self.check_results: Dict[str, bool] = {}
        
    async def check_memory(self) -> bool:
        """Check memory usage against thresholds."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage = memory_info.rss
            
            metrics.memory_usage.set(memory_usage)
            
            if memory_usage > alert_thresholds.memory_rollback:
                logger.critical(f"Memory usage ({memory_usage/1e9:.1f}GB) exceeded rollback threshold")
                return False
            elif memory_usage > alert_thresholds.memory_critical:
                logger.error(f"Memory usage ({memory_usage/1e9:.1f}GB) exceeded critical threshold")
                return False
            elif memory_usage > alert_thresholds.memory_warning:
                logger.warning(f"Memory usage ({memory_usage/1e9:.1f}GB) exceeded warning threshold")
                
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return False
    
    async def check_latency(self) -> bool:
        """Check latency against thresholds."""
        try:
            p95_latency = metrics.request_latency.observe()._sum / metrics.request_latency.observe()._count
            baseline_latency = 0.055  # 55ms from our baseline tests
            
            degradation = ((p95_latency - baseline_latency) / baseline_latency) * 100
            
            if degradation > alert_thresholds.perf_rollback:
                logger.critical(f"Latency degradation ({degradation:.1f}%) exceeded rollback threshold")
                return False
            elif degradation > alert_thresholds.perf_critical:
                logger.error(f"Latency degradation ({degradation:.1f}%) exceeded critical threshold")
                return False
            elif degradation > alert_thresholds.perf_warning:
                logger.warning(f"Latency degradation ({degradation:.1f}%) exceeded warning threshold")
                
            return True
        except Exception as e:
            logger.error(f"Latency check failed: {str(e)}")
            return False
    
    async def check_quality(self) -> bool:
        """Check attribution quality metrics."""
        try:
            entropy = metrics.attribution_entropy._value.get()
            sparsity = metrics.attribution_sparsity._value.get()
            
            if not alert_thresholds.min_entropy <= entropy <= alert_thresholds.max_entropy:
                logger.error(f"Attribution entropy ({entropy:.2f}) outside acceptable range")
                return False
                
            if sparsity < alert_thresholds.min_sparsity:
                logger.error(f"Attribution sparsity ({sparsity:.2f}) below minimum threshold")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Quality check failed: {str(e)}")
            return False
    
    async def check_cache(self) -> bool:
        """Check cache performance."""
        try:
            hits = metrics.cache_hits._value.get()
            misses = metrics.cache_misses._value.get()
            total = hits + misses
            
            if total > 0:
                hit_rate = (hits / total) * 100
                if hit_rate < 50:
                    logger.warning(f"Cache hit rate ({hit_rate:.1f}%) below 50%")
                    
            return True
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            return False
    
    async def run_health_checks(self):
        """Run all health checks at configured intervals."""
        while True:
            current_time = time.time()
            
            # Memory Check
            if current_time - self.last_check_times.get("memory", 0) >= health_config.memory_interval:
                self.check_results["memory"] = await self.check_memory()
                self.last_check_times["memory"] = current_time
            
            # Latency Check
            if current_time - self.last_check_times.get("latency", 0) >= health_config.latency_interval:
                self.check_results["latency"] = await self.check_latency()
                self.last_check_times["latency"] = current_time
            
            # Quality Check
            if current_time - self.last_check_times.get("quality", 0) >= health_config.quality_interval:
                self.check_results["quality"] = await self.check_quality()
                self.last_check_times["quality"] = current_time
            
            # Cache Check
            if current_time - self.last_check_times.get("cache", 0) >= health_config.cache_interval:
                self.check_results["cache"] = await self.check_cache()
                self.last_check_times["cache"] = current_time
            
            # Overall system health
            system_healthy = all(self.check_results.values())
            metrics_summary = [f"{k}: {'✓' if v else '✗'}" for k, v in self.check_results.items()]
            
            if system_healthy:
                logger.info(f"System healthy | {' | '.join(metrics_summary)}")
            else:
                logger.error(f"System unhealthy | {' | '.join(metrics_summary)}")
            
            await asyncio.sleep(min(
                health_config.memory_interval,
                health_config.latency_interval,
                health_config.quality_interval,
                health_config.cache_interval
            ))
