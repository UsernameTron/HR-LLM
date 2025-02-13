"""
Metrics tracking utility for monitoring data pipeline performance.
"""
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'processing_times': [],
            'last_error': None,
            'last_success': None
        })
        
        self.batch_metrics = defaultdict(lambda: {
            'total_processed': 0,
            'success_rate': 0.0,
            'avg_processing_time': 0.0,
            'last_batch_size': 0
        })
    
    def get_current_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def record_success(self, operation: str, source: str) -> None:
        """Record successful operation."""
        key = f"{operation}_{source}"
        self.metrics[key]['success_count'] += 1
        self.metrics[key]['last_success'] = self.get_current_timestamp()
        
        logger.info(f"Success: {operation} from {source}")
    
    def record_failure(self, operation: str, source: str, error: str) -> None:
        """Record failed operation."""
        key = f"{operation}_{source}"
        self.metrics[key]['failure_count'] += 1
        self.metrics[key]['last_error'] = {
            'timestamp': self.get_current_timestamp(),
            'error': error
        }
        
        logger.error(f"Failure: {operation} from {source}: {error}")
    
    def record_processing_time(self, operation: str, source: str, duration_ms: float) -> None:
        """Record processing time for operation."""
        key = f"{operation}_{source}"
        self.metrics[key]['processing_times'].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.metrics[key]['processing_times']) > 1000:
            self.metrics[key]['processing_times'] = self.metrics[key]['processing_times'][-1000:]
    
    def record_batch_success(self, operation: str, batch_size: int) -> None:
        """Record successful batch processing."""
        self.batch_metrics[operation]['total_processed'] += batch_size
        self.batch_metrics[operation]['last_batch_size'] = batch_size
        self._update_batch_metrics(operation)
    
    def record_batch_failure(self, operation: str, error: str) -> None:
        """Record batch processing failure."""
        logger.error(f"Batch failure in {operation}: {error}")
        self._update_batch_metrics(operation)
    
    def _update_batch_metrics(self, operation: str) -> None:
        """Update batch processing metrics."""
        metrics = self.batch_metrics[operation]
        success_count = self.metrics[f"{operation}_success"]['success_count']
        failure_count = self.metrics[f"{operation}_failure"]['failure_count']
        
        total_attempts = success_count + failure_count
        if total_attempts > 0:
            metrics['success_rate'] = (success_count / total_attempts) * 100
        
        processing_times = self.metrics[f"{operation}_success"]['processing_times']
        if processing_times:
            metrics['avg_processing_time'] = sum(processing_times) / len(processing_times)
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return {
            'operations': dict(self.metrics),
            'batch_processing': dict(self.batch_metrics),
            'timestamp': self.get_current_timestamp()
        }
    
    def get_health_status(self) -> Dict:
        """Get system health status based on metrics."""
        status = {
            'healthy': True,
            'issues': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for operation, metrics in self.metrics.items():
            success_rate = 0
            if metrics['success_count'] + metrics['failure_count'] > 0:
                success_rate = (
                    metrics['success_count'] /
                    (metrics['success_count'] + metrics['failure_count'])
                    * 100
                )
            
            if success_rate < 95:  # Alert if success rate drops below 95%
                status['healthy'] = False
                status['issues'].append({
                    'operation': operation,
                    'success_rate': success_rate,
                    'last_error': metrics['last_error']
                })
        
        return status
