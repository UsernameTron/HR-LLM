"""Metrics tracking utility focused on hiring signals and source reliability."""
import logging
from datetime import datetime
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self):
        # Hiring signal metrics
        self.hiring_signal = Gauge(
            'hiring_signal_strength',
            'Strength of hiring signals detected',
            ['source', 'company']
        )
        
        # Source reliability metrics
        self.source_reliability = Gauge(
            'source_reliability',
            'Reliability score of data sources',
            ['source_type']
        )
        
        # Performance metrics
        self.processing_time = Histogram(
            'processing_duration_seconds',
            'Time spent processing content',
            buckets=[.005, .01, .025, .05, .075, .1, .25, .5, 1]
        )
        
        # Error tracking
        self.errors = Counter(
            'processing_errors_total',
            'Total number of processing errors',
            ['source', 'error_type']
        )
    
    def record_hiring_signal(self, strength: float, source: str, company: str) -> None:
        """Record strength of a hiring signal."""
        self.hiring_signal.labels(source=source, company=company).set(strength)
        logger.info(f"Recorded hiring signal: {strength} from {company} via {source}")
    
    def update_source_reliability(self, source_type: str, score: float) -> None:
        """Update reliability score for a data source."""
        self.source_reliability.labels(source_type=source_type).set(score)
        logger.info(f"Updated reliability score for {source_type}: {score}")
    
    def record_processing_time(self, duration: float) -> None:
        """Record time taken to process content."""
        self.processing_time.observe(duration)
    
    def record_error(self, source: str, error_type: str) -> None:
        """Record processing error."""
        self.errors.labels(source=source, error_type=error_type).inc()
        logger.error(f"Error processing {source}: {error_type}")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of current metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'hiring_signals': self._collect_metric(self.hiring_signal),
                'source_reliability': self._collect_metric(self.source_reliability),
                'error_counts': self._collect_metric(self.errors)
            }
        }
    
    def _collect_metric(self, metric) -> Dict:
        """Helper to collect all values for a metric."""
        return {
            sample.labels['source']: sample.value 
            for sample in metric.collect()[0].samples
        }
