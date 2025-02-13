"""
Metrics exporter for performance benchmarks.
Collects and exports metrics to Prometheus for visualization in Grafana.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.utils import INF

logger = logging.getLogger(__name__)

# Performance metrics
OPERATION_LATENCY = Histogram(
    'redis_operation_latency_seconds',
    'Redis operation latency in seconds',
    ['operation_type'],
    buckets=(
        .001, .005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, INF
    )
)

OPERATION_COUNTER = Counter(
    'redis_operations_total',
    'Total number of Redis operations',
    ['operation_type', 'status']
)

MEMORY_USAGE = Gauge(
    'redis_memory_usage_bytes',
    'Redis memory usage in bytes'
)

CACHE_HIT_RATIO = Gauge(
    'redis_cache_hit_ratio',
    'Redis cache hit ratio'
)

CONNECTION_POOL_SIZE = Gauge(
    'redis_connection_pool_size',
    'Redis connection pool size'
)

# Benchmark metrics
REGRESSION_SCORE = Gauge(
    'performance_regression_score',
    'Performance regression score (-1 to 1)',
    ['metric_name']
)

BASELINE_DEVIATION = Gauge(
    'baseline_deviation_percent',
    'Deviation from baseline performance',
    ['metric_name']
)

class MetricsExporter:
    """Exports performance metrics to Prometheus."""
    
    def __init__(
        self,
        port: int = 9090,
        benchmark_dir: str = "tests/benchmarks",
        history_retention_days: int = 90
    ):
        self.port = port
        self.benchmark_dir = Path(benchmark_dir)
        self.history_retention_days = history_retention_days
        self.start_time = datetime.now()
    
    def start(self):
        """Start the metrics exporter server."""
        start_http_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")
    
    def record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool = True
    ):
        """Record a Redis operation metric."""
        OPERATION_LATENCY.labels(operation_type=operation_type).observe(duration)
        status = "success" if success else "failure"
        OPERATION_COUNTER.labels(
            operation_type=operation_type,
            status=status
        ).inc()
    
    def update_memory_usage(self, bytes_used: int):
        """Update Redis memory usage metric."""
        MEMORY_USAGE.set(bytes_used)
    
    def update_cache_stats(self, hit_ratio: float):
        """Update cache statistics."""
        CACHE_HIT_RATIO.set(hit_ratio)
    
    def update_pool_size(self, size: int):
        """Update connection pool size metric."""
        CONNECTION_POOL_SIZE.set(size)
    
    def process_benchmark_results(
        self,
        results_file: Path,
        baseline_file: Optional[Path] = None
    ):
        """Process benchmark results and update metrics."""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            baseline = None
            if baseline_file and baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline = json.load(f)
            
            self._update_benchmark_metrics(results, baseline)
            
        except Exception as e:
            logger.error(f"Error processing benchmark results: {str(e)}", exc_info=True)
    
    def _update_benchmark_metrics(
        self,
        results: Dict,
        baseline: Optional[Dict] = None
    ):
        """Update benchmark-related metrics."""
        for metric_name, data in results.items():
            if isinstance(data, dict) and 'current_value' in data:
                # Update regression score
                z_score = data.get('z_score', 0)
                REGRESSION_SCORE.labels(metric_name=metric_name).set(z_score)
                
                # Update baseline deviation
                percent_change = data.get('percent_change', 0)
                BASELINE_DEVIATION.labels(metric_name=metric_name).set(percent_change)
    
    def cleanup_old_results(self):
        """Clean up old benchmark results."""
        cutoff_date = datetime.now().timestamp() - (
            self.history_retention_days * 24 * 60 * 60
        )
        
        try:
            history_dir = self.benchmark_dir / "history"
            if not history_dir.exists():
                return
            
            for results_file in history_dir.glob("regression_test_*.json"):
                try:
                    if results_file.stat().st_mtime < cutoff_date:
                        results_file.unlink()
                        logger.info(f"Cleaned up old results file: {results_file}")
                except Exception as e:
                    logger.error(
                        f"Error cleaning up {results_file}: {str(e)}",
                        exc_info=True
                    )
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
    
    def export_metrics_snapshot(self, output_file: Path):
        """Export current metrics snapshot to JSON."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'metrics': {
                'operations': {
                    'latency': OPERATION_LATENCY._samples(),
                    'count': OPERATION_COUNTER._samples()
                },
                'resources': {
                    'memory_bytes': MEMORY_USAGE._value,
                    'connection_pool_size': CONNECTION_POOL_SIZE._value
                },
                'cache': {
                    'hit_ratio': CACHE_HIT_RATIO._value
                },
                'regression': {
                    'score': REGRESSION_SCORE._samples(),
                    'baseline_deviation': BASELINE_DEVIATION._samples()
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        logger.info(f"Metrics snapshot exported to {output_file}")

def setup_grafana_dashboard(
    dashboard_file: Path,
    grafana_url: str,
    api_key: str
):
    """Set up Grafana dashboard from JSON definition."""
    import requests
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        with open(dashboard_file, 'r') as f:
            dashboard_json = json.load(f)
        
        # Ensure dashboard has required fields
        dashboard_json['id'] = None  # Let Grafana assign an ID
        dashboard_json['version'] = 1
        
        payload = {
            'dashboard': dashboard_json,
            'overwrite': True
        }
        
        response = requests.post(
            f'{grafana_url}/api/dashboards/db',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            logger.info("Grafana dashboard created/updated successfully")
            return response.json()
        else:
            logger.error(
                f"Failed to create dashboard: {response.status_code} - {response.text}"
            )
            return None
            
    except Exception as e:
        logger.error(f"Error setting up Grafana dashboard: {str(e)}", exc_info=True)
        return None

def main():
    """Main entry point for metrics exporter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance metrics exporter")
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Port to serve metrics on"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="tests/benchmarks",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--grafana-url",
        type=str,
        help="Grafana URL for dashboard setup"
    )
    parser.add_argument(
        "--grafana-key",
        type=str,
        help="Grafana API key for dashboard setup"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and start metrics exporter
    exporter = MetricsExporter(
        port=args.port,
        benchmark_dir=args.benchmark_dir
    )
    exporter.start()
    
    # Set up Grafana dashboard if configured
    if args.grafana_url and args.grafana_key:
        dashboard_file = Path(__file__).parent / "grafana/dashboards/performance_benchmarks.json"
        if dashboard_file.exists():
            setup_grafana_dashboard(
                dashboard_file,
                args.grafana_url,
                args.grafana_key
            )
    
    # Keep the server running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Metrics exporter stopped")

if __name__ == "__main__":
    main()
