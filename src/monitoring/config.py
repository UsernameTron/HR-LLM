"""Configuration for monitoring and alerting."""
import os
import atexit
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import prometheus_client as prom
from prometheus_client import CollectorRegistry, multiprocess

# Configure logging
logger = logging.getLogger(__name__)

# Core Metrics
LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]

# Set up metrics directory in project
PROJECT_ROOT = Path(__file__).parent.parent.parent
METRICS_DIR = PROJECT_ROOT / 'metrics_tmp'

def setup_metrics_dir() -> Path:
    """Set up metrics directory with proper cleanup handling."""
    try:
        # Clean up any existing metrics files
        if METRICS_DIR.exists():
            shutil.rmtree(METRICS_DIR)
        
        # Create fresh directory
        METRICS_DIR.mkdir(exist_ok=True)
        
        # Register cleanup handler
        atexit.register(cleanup_metrics)
        
        # Set environment variable for Prometheus
        os.environ['prometheus_multiproc_dir'] = str(METRICS_DIR)
        
        return METRICS_DIR
    except Exception as e:
        logger.error(f"Failed to set up metrics directory: {e}")
        raise

def cleanup_metrics():
    """Clean up metrics files on exit."""
    try:
        if METRICS_DIR.exists():
            shutil.rmtree(METRICS_DIR)
            logger.info(f"Cleaned up metrics directory: {METRICS_DIR}")
    except Exception as e:
        logger.error(f"Failed to clean up metrics directory: {e}")

# Initialize metrics directory
METRICS_DIR = setup_metrics_dir()

# Initialize registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class MetricsRegistry:
    """Core metrics for the sentiment analysis pipeline."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        # Performance Metrics
        self.request_latency = prom.Histogram(
            "sentiment_request_latency_seconds",
            "Request latency in seconds",
            buckets=LATENCY_BUCKETS
        )
        
        self.memory_usage = prom.Gauge(
            "sentiment_memory_usage_bytes",
            "Memory usage in bytes"
        )
        
        self.error_rate = prom.Counter(
            "sentiment_errors_total",
            "Total number of errors",
            ["type"]
        )
        
        # Attribution Quality Metrics
        self.attribution_entropy = prom.Gauge(
            "sentiment_attribution_entropy",
            "Token attribution entropy"
        )
        
        self.attribution_sparsity = prom.Gauge(
            "sentiment_attribution_sparsity",
            "Token attribution sparsity"
        )
        
        # System Health Metrics
        self.message_throughput = prom.Counter(
            "sentiment_messages_processed_total",
            "Total number of messages processed"
        )
        
        self.batch_size = prom.Gauge(
            "sentiment_batch_size",
            "Current batch size"
        )
        
        self.cache_hits = prom.Counter(
            "sentiment_cache_hits_total",
            "Total number of cache hits"
        )
        
        self.cache_misses = prom.Counter(
            "sentiment_cache_misses_total",
            "Total number of cache misses"
        )

# Alert Thresholds
@dataclass
class AlertThresholds:
    """Alert thresholds for monitoring."""
    
    # Memory Thresholds (bytes)
    memory_warning: int = int(33.6e9)  # 70% of 48GB
    memory_critical: int = int(40.8e9)  # 85% of 48GB
    memory_rollback: int = int(43.2e9)  # 90% of 48GB
    
    # Error Rate Thresholds (percentage)
    error_warning: float = 5.0
    error_critical: float = 10.0
    error_intervention: float = 15.0
    
    # Performance Degradation (percentage from baseline)
    perf_warning: float = 25.0
    perf_critical: float = 40.0
    perf_rollback: float = 50.0
    
    # Attribution Quality Thresholds
    min_entropy: float = 0.5
    max_entropy: float = 2.5
    min_sparsity: float = 0.1

# Health Check Configuration
@dataclass
class HealthCheckConfig:
    """Configuration for system health checks."""
    
    # Check Intervals (seconds)
    memory_interval: int = 60
    latency_interval: int = 30
    quality_interval: int = 300
    cache_interval: int = 120
    
    # Check Endpoints
    endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        self.endpoints = {
            "memory": "/health/memory",
            "latency": "/health/latency",
            "quality": "/health/quality",
            "cache": "/health/cache",
            "overall": "/health"
        }

# Initialize Global Instances
metrics = MetricsRegistry()
alert_thresholds = AlertThresholds()
health_config = HealthCheckConfig()
