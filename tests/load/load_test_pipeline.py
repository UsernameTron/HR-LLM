"""Load testing suite for the sentiment analysis pipeline."""
import asyncio
import numpy as np
import logging
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import aiokafka
from prometheus_client import Counter, Histogram, Gauge, multiprocess, values
from src.monitoring.config import metrics, registry

from src.ingestion.pipeline_manager import DataPipelineManager
from tests.mocks.mock_kafka import MockKafkaConsumer
from src.classification.model import SentimentClassifier
from src.monitoring.drift_detector import DriftDetector
from src.classification.explainer import RealTimeExplainer
from src.cache.redis_memory_monitor import RedisMemoryMonitor
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@dataclass
class LoadTestMetrics:
    """Container for load test metrics."""
    throughput: float  # messages/second
    latency_p95: float  # ms
    memory_usage: float  # MB
    cache_hit_rate: float  # percentage
    error_rate: float  # percentage
    drift_detection_rate: float  # percentage
    explanation_time_p95: float  # ms
    attribution_entropy: float  # bits
    attribution_sparsity: float  # percentage
    attribution_gradient_norm: float  # magnitude

class LoadTestRunner:
    """Manages load testing of the pipeline."""
    
    async def __aenter__(self):
        """Initialize resources."""
        try:
            await self.initialize()
            return self
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            await self.cleanup()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        try:
            await self.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise
    
    async def initialize(self):
        """Initialize resources."""
        # Mark this process for metrics collection
        multiprocess.mark_process_dead(os.getpid())
        self.start_time = time.time()
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
            
            # Clean up metrics for this process
            multiprocess.mark_process_dead(os.getpid())
            
            # Record end time
            self.end_time = time.time()
            
            self.logger.info("Load test resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise
    
    def __init__(
        self,
        pipeline: DataPipelineManager,
        test_duration: int = 3600,  # 1 hour
        message_rate: int = 100,  # messages/second
        batch_size: int = 32
    ):
        self.pipeline = pipeline
        self.test_duration = test_duration
        self.message_rate = message_rate
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Get metrics collector from pipeline
        from src.monitoring.config import metrics
        self.metrics = metrics
        self.metrics_collector = self.pipeline.explainer.metrics_collector
        
        # Initialize metrics tracking
        self.start_time = None
        self.end_time = None
        self.messages_processed = 0
        self.errors = []
        
        # Mark this process for metrics collection
        multiprocess.mark_process_dead(os.getpid())
        
        # Use shared metrics registry
        self.metrics = metrics
        self.latencies = []
        self.explanation_times = []
        self.drift_detections = 0
        
        # Initialize Prometheus metrics
        self.latency_histogram = Histogram(
            'pipeline_latency_ms',
            'Message processing latency in milliseconds',
            buckets=np.logspace(0, 4, 50)
        )
        self.throughput_gauge = Gauge(
            'pipeline_throughput',
            'Messages processed per second'
        )
        self.memory_gauge = Gauge(
            'pipeline_memory_mb',
            'Memory usage in MB'
        )
        self.error_counter = Counter(
            'pipeline_errors',
            'Number of processing errors'
        )
        self.cache_hits = Counter(
            'cache_hits',
            'Number of cache hits'
        )
        self.drift_detections = Counter(
            'drift_detections',
            'Number of drift detections'
        )
        
        # Success criteria
        self.thresholds = {
            "max_latency": 1000.0,  # ms
            "max_error_rate": 0.01,  # 1%
            "max_memory": 40960.0,  # MB (85% of 48GB)
            "min_entropy": 0.5,
            "max_entropy": 2.5,
            "min_sparsity": 0.1
        }
        
        # Test data generation
        self.test_texts = [
            "Great opportunity for experienced developers",
            "Seeking talented engineers for immediate roles",
            "Competitive salary and benefits package",
            "Remote work options available",
            "Must have 5+ years of experience",
            "Exciting startup environment",
            "Join our growing team",
            "Leadership position available",
            "Entry level position with training",
            "Fast-paced work environment"
        ]
        
    async def _generate_test_message(self) -> Dict[str, Any]:
        """Generate a test message with realistic data."""
        text = np.random.choice(self.test_texts)
        return {
            "text": text,
            "source": "load_test",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "test_id": np.random.randint(1000000),
                "priority": np.random.choice(["high", "medium", "low"])
            }
        }
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> LoadTestMetrics:
        """Process a batch and collect metrics."""
        start_time = time.time()
        
        try:
            # Process messages
            results = await self.pipeline._process_batch(batch)
            
            # Collect metrics
            latency = (time.time() - start_time) * 1000
            self.latency_histogram.observe(latency)
            
            # Check for drift detections
            drift_detections = sum(
                1 for r in results
                if r["classification"].get("drift_metrics") is not None
            )
            self.drift_detections.inc(drift_detections)
            
            # Check cache hits
            cache_hits = sum(
                1 for r in results
                if "cache_hit" in r and r["cache_hit"]
            )
            self.cache_hits.inc(cache_hits)
            
            # Get histogram values
            latency_values = [latency]  # Use current batch latency
            
            # Get attribution metrics
            attribution_stats = self.metrics_collector.get_summary_statistics()
            
            return LoadTestMetrics(
                throughput=len(batch) / (time.time() - start_time),
                latency_p95=float(np.percentile(latency_values, 95)),
                memory_usage=self.pipeline.redis_monitor.get_memory_usage(),
                cache_hit_rate=cache_hits / len(batch) * 100,
                error_rate=0,
                drift_detection_rate=drift_detections / len(batch) * 100,
                explanation_time_p95=latency,
                attribution_entropy=attribution_stats.get('entropy', {}).get('mean', 0.0),
                attribution_sparsity=attribution_stats.get('sparsity', {}).get('mean', 0.0),
                attribution_gradient_norm=attribution_stats.get('gradient_norm', {}).get('mean', 0.0)
            )
            
        except Exception as e:
            self.error_counter.inc()
            self.logger.error(f"Batch processing error: {str(e)}")
            raise
    
    async def run_load_test(self) -> List[LoadTestMetrics]:
        """Run the load test and collect metrics."""
        self.logger.info(
            f"Starting load test: {self.message_rate} msg/s for {self.test_duration}s"
        )
        
        start_time = time.time()
        metrics_history = []
        
        try:
            while time.time() - start_time < self.test_duration:
                # Generate batch
                batch = [
                    await self._generate_test_message()
                    for _ in range(self.batch_size)
                ]
                
                # Process batch and collect metrics
                metrics = await self._process_batch(batch)
                metrics_history.append(metrics)
                
                # Update Prometheus metrics with proper multiprocess handling
                values.ValueClass.set(self.metrics.request_latency, metrics.latency_p95 / 1000)  # Convert ms to seconds
                values.ValueClass.set(self.metrics.memory_usage, metrics.memory_usage * 1024 * 1024)  # Convert MB to bytes
                values.ValueClass.set(self.metrics.attribution_entropy, metrics.attribution_entropy)
                values.ValueClass.set(self.metrics.attribution_sparsity, metrics.attribution_sparsity)
                values.ValueClass.set(self.metrics.batch_size, self.batch_size)
                values.ValueClass.inc(self.metrics.message_throughput, len(batch))
                
                # Log real-time metrics
                self.logger.info(
                    f"Metrics Update | "
                    f"P95 Latency: {metrics.latency_p95:.2f}ms | "
                    f"Memory: {metrics.memory_usage:.0f}MB | "
                    f"Error Rate: {metrics.error_rate:.2%} | "
                    f"Attribution Entropy: {metrics.attribution_entropy:.2f} | "
                    f"Sparsity: {metrics.attribution_sparsity:.2f}"
                )
                
                # Rate limiting
                await asyncio.sleep(self.batch_size / self.message_rate)
                
            return metrics_history
            
        except Exception as e:
            self.logger.error(f"Load test error: {str(e)}")
            raise MetalError(
                f"Load test failed: {str(e)}",
                category=MetalErrorCategory.LOAD_TEST_ERROR
            )
        
    def generate_report(
        self,
        metrics_history: List[LoadTestMetrics]
    ) -> Dict[str, Any]:
        """Generate a summary report of the load test."""
        return {
            "test_duration": self.test_duration,
            "message_rate": self.message_rate,
            "total_messages": self.message_rate * self.test_duration,
            "metrics": {
                "avg_throughput": np.mean([m.throughput for m in metrics_history]),
                "p95_latency": np.percentile(
                    [m.latency_p95 for m in metrics_history],
                    95
                ),
                "max_memory": max(m.memory_usage for m in metrics_history),
                "avg_attribution_entropy": np.mean([m.attribution_entropy for m in metrics_history]),
                "avg_attribution_sparsity": np.mean([m.attribution_sparsity for m in metrics_history]),
                "avg_gradient_norm": np.mean([m.attribution_gradient_norm for m in metrics_history]),
                "avg_cache_hit_rate": np.mean(
                    [m.cache_hit_rate for m in metrics_history]
                ),
                "error_rate": np.mean([m.error_rate for m in metrics_history]),
                "drift_detection_rate": np.mean(
                    [m.drift_detection_rate for m in metrics_history]
                )
            },
            "thresholds_met": {
                "throughput": np.mean([m.throughput for m in metrics_history])
                    >= self.message_rate,
                "latency": np.percentile(
                    [m.latency_p95 for m in metrics_history],
                    95
                ) <= 1000,  # 1s max latency
                "memory": max(m.memory_usage for m in metrics_history)
                    <= 1024 * 48,  # 48GB limit
                "error_rate": np.mean([m.error_rate for m in metrics_history])
                    <= 0.1  # 0.1% max error rate
            }
        }
