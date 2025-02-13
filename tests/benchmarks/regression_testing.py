"""
Automated performance regression testing framework.
Tracks and analyzes benchmark results over time to detect performance degradation.
"""
import asyncio
import json
import logging
import os
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prometheus_client import Gauge
from scipy import stats

from src.utils.metrics import MetricsTracker
from tests.benchmarks.redis_benchmarks import RedisBenchmarkSuite

def load_environment_config(config_file: str = 'config/benchmark_environments.yml') -> dict:
    """Load environment-specific configuration."""
    env_name = os.getenv('BENCHMARK_ENV', 'staging')
    with open(config_file) as f:
        configs = yaml.safe_load(f)
    return configs.get(env_name, configs['default'])

logger = logging.getLogger(__name__)

# Prometheus metrics for regression tracking
REGRESSION_GAUGE = Gauge(
    'performance_regression_score',
    'Performance regression score (-1 to 1)',
    ['metric_name']
)

BASELINE_DEVIATION_GAUGE = Gauge(
    'baseline_deviation_percent',
    'Deviation from baseline performance',
    ['metric_name']
)

class RegressionAnalyzer:
    """Analyzes benchmark results for performance regressions."""
    
    def __init__(
        self,
        baseline_path: str = "tests/benchmarks/baselines",
        history_path: str = "tests/benchmarks/history",
        config_file: str = "config/benchmark_environments.yml"
    ):
        # Load environment-specific configuration
        self.config = load_environment_config(config_file)
        self.thresholds = self.config['thresholds']
        
        # Set confidence level and regression threshold from config
        self.confidence_level = self.thresholds['regression_confidence']
        self.regression_threshold = self.thresholds['performance_degradation']
        self.baseline_path = Path(baseline_path)
        self.history_path = Path(history_path)
        self.confidence_level = confidence_level
        self.regression_threshold = regression_threshold
        self.metrics_tracker = MetricsTracker()
        
        # Create directories if they don't exist
        self.baseline_path.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)
    
    async def capture_baseline(
        self,
        benchmark_suite: RedisBenchmarkSuite,
        force: bool = False
    ) -> Dict:
        """Capture baseline performance metrics using environment config."""
        # Get environment-specific test parameters
        test_config = self.config['pipeline']
        num_operations = self.config['latency']['num_operations']
        num_batches = test_config['num_batches']
        """
        Capture baseline performance metrics.
        Only updates if force=True or no baseline exists.
        """
        baseline_file = self.baseline_path / "redis_baseline.json"
        
        if baseline_file.exists() and not force:
            logger.info("Using existing baseline")
            with open(baseline_file, 'r') as f:
                return json.load(f)
        
        logger.info("Capturing new baseline metrics")
        baseline_metrics = {}
        
        # Run each benchmark type multiple times based on environment config
        for _ in range(self.config['test_iterations']):
            # Write throughput
            write_result = await benchmark_suite.benchmark_write_throughput(
                num_operations=num_operations
            )
            baseline_metrics.setdefault('write', []).append({
                'ops_per_second': write_result['ops_per_second'],
                'p95_latency': write_result['p95_latency']
            })
            
            # Read throughput
            read_result = await benchmark_suite.benchmark_read_throughput(
                num_operations=num_operations
            )
            baseline_metrics.setdefault('read', []).append({
                'ops_per_second': read_result['ops_per_second'],
                'hit_ratio': read_result['hit_ratio']
            })
            
            # Pipeline throughput
            pipeline_result = await benchmark_suite.benchmark_pipeline_throughput(
                num_batches=num_batches,
                batch_size=test_config['batch_size'],
                timeout=test_config['timeout']
            )
            baseline_metrics.setdefault('pipeline', []).append({
                'ops_per_second': pipeline_result['ops_per_second'],
                'p95_batch_latency': pipeline_result['p95_batch_latency']
            })
            
            # Memory efficiency
            memory_result = await benchmark_suite.benchmark_memory_efficiency(
                max_size=1024 * 100  # Smaller size for baseline
            )
            baseline_metrics.setdefault('memory', []).append({
                'memory_ratio': np.mean([r['memory_ratio'] for r in memory_result])
            })
        
        # Calculate statistical measures
        baseline = {
            metric: {
                key: {
                    'mean': np.mean([run[key] for run in runs]),
                    'std': np.std([run[key] for run in runs]),
                    'samples': len(runs)
                }
                for key in runs[0].keys()
            }
            for metric, runs in baseline_metrics.items()
        }
        
        # Save baseline
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info("Baseline metrics captured and saved")
        return baseline
    
    async def run_regression_test(
        self,
        benchmark_suite: RedisBenchmarkSuite
    ) -> Tuple[bool, Dict]:
        """
        Run regression test against baseline.
        Returns (passed, detailed_results).
        """
        # Load baseline
        baseline_file = self.baseline_path / "redis_baseline.json"
        if not baseline_file.exists():
            raise ValueError("No baseline found. Run capture_baseline first.")
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        # Run current benchmarks
        current_metrics = {}
        regression_detected = False
        detailed_results = {}
        
        # Write throughput test
        write_result = await benchmark_suite.benchmark_write_throughput(
            num_operations=1000
        )
        current_metrics['write'] = {
            'ops_per_second': write_result['ops_per_second'],
            'p95_latency': write_result['p95_latency']
        }
        
        # Read throughput test
        read_result = await benchmark_suite.benchmark_read_throughput(
            num_operations=1000
        )
        current_metrics['read'] = {
            'ops_per_second': read_result['ops_per_second'],
            'hit_ratio': read_result['hit_ratio']
        }
        
        # Pipeline throughput test
        pipeline_result = await benchmark_suite.benchmark_pipeline_throughput(
            num_batches=10
        )
        current_metrics['pipeline'] = {
            'ops_per_second': pipeline_result['ops_per_second'],
            'p95_batch_latency': pipeline_result['p95_batch_latency']
        }
        
        # Memory efficiency test
        memory_result = await benchmark_suite.benchmark_memory_efficiency(
            max_size=1024 * 100
        )
        current_metrics['memory'] = {
            'memory_ratio': np.mean([r['memory_ratio'] for r in memory_result])
        }
        
        # Analyze each metric
        for metric_name, metric_data in current_metrics.items():
            baseline_data = baseline[metric_name]
            metric_results = {}
            
            for key, value in metric_data.items():
                baseline_mean = baseline_data[key]['mean']
                baseline_std = baseline_data[key]['std']
                
                # Calculate z-score
                z_score = (value - baseline_mean) / baseline_std
                
                # Calculate p-value
                p_value = 1 - stats.norm.cdf(abs(z_score))
                
                # Calculate percent change
                percent_change = ((value - baseline_mean) / baseline_mean) * 100
                
                # Determine if this is a regression
                is_regression = (
                    p_value < (1 - self.confidence_level) and
                    abs(percent_change) > (self.regression_threshold * 100)
                )
                
                if is_regression and percent_change < 0:
                    regression_detected = True
                
                # Update Prometheus metrics
                metric_key = f"{metric_name}_{key}"
                REGRESSION_GAUGE.labels(metric_key).set(z_score)
                BASELINE_DEVIATION_GAUGE.labels(metric_key).set(percent_change)
                
                metric_results[key] = {
                    'current_value': value,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'z_score': z_score,
                    'p_value': p_value,
                    'percent_change': percent_change,
                    'is_regression': is_regression
                }
            
            detailed_results[metric_name] = metric_results
        
        # Save results to history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.history_path / f"regression_test_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': detailed_results,
                'regression_detected': regression_detected
            }, f, indent=2)
        
        # Log results
        self._log_regression_results(detailed_results, regression_detected)
        
        return not regression_detected, detailed_results
    
    def _log_regression_results(
        self,
        results: Dict,
        regression_detected: bool
    ) -> None:
        """Log detailed regression test results."""
        if regression_detected:
            logger.error("Performance regression detected!")
        else:
            logger.info("No performance regression detected")
        
        for metric_name, metric_results in results.items():
            logger.info(f"\nResults for {metric_name}:")
            for key, data in metric_results.items():
                status = "REGRESSION" if data['is_regression'] else "OK"
                logger.info(
                    f"{key}: {status}\n"
                    f"  Current: {data['current_value']:.2f}\n"
                    f"  Baseline: {data['baseline_mean']:.2f} ± {data['baseline_std']:.2f}\n"
                    f"  Change: {data['percent_change']:.1f}%\n"
                    f"  Confidence: {(1 - data['p_value']) * 100:.1f}%"
                )
    
    async def analyze_trends(
        self,
        days: int = 30
    ) -> Dict:
        """
        Analyze performance trends over time.
        Returns trend analysis results.
        """
        history_files = sorted(self.history_path.glob("regression_test_*.json"))
        if not history_files:
            return {}
        
        # Load historical data
        history_data = []
        for file in history_files:
            with open(file, 'r') as f:
                data = json.load(f)
                history_data.append({
                    'timestamp': datetime.strptime(
                        data['timestamp'],
                        "%Y%m%d_%H%M%S"
                    ),
                    **data['results']
                })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(history_data)
        
        # Calculate trends
        trends = {}
        for metric_name in df.columns:
            if metric_name == 'timestamp':
                continue
            
            # Calculate rolling statistics
            rolling_mean = df[metric_name].rolling(window=5).mean()
            rolling_std = df[metric_name].rolling(window=5).std()
            
            # Calculate trend
            x = np.arange(len(df))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x,
                df[metric_name]
            )
            
            trends[metric_name] = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'current_mean': rolling_mean.iloc[-1],
                'current_std': rolling_std.iloc[-1],
                'significant_trend': p_value < (1 - self.confidence_level)
            }
        
        return trends

def create_regression_report(
    results: Dict,
    trends: Dict,
    output_path: str = "tests/benchmarks/reports"
) -> str:
    """Create detailed regression test report."""
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / f"regression_report_{int(time.time())}.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# Performance Regression Test Report\n\n")
        f.write(f"Generated: {report_time}\n\n")
        
        # Overall Status
        regression_detected = any(
            any(m['is_regression'] for m in metric.values())
            for metric in results.values()
        )
        status = "❌ FAIL" if regression_detected else "✅ PASS"
        f.write(f"## Overall Status: {status}\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        for metric_name, metric_results in results.items():
            f.write(f"### {metric_name}\n\n")
            f.write("| Metric | Status | Current | Baseline | Change | Confidence |\n")
            f.write("|--------|--------|---------|-----------|--------|------------|\n")
            
            for key, data in metric_results.items():
                status = "❌" if data['is_regression'] else "✅"
                f.write(
                    f"| {key} | {status} | "
                    f"{data['current_value']:.2f} | "
                    f"{data['baseline_mean']:.2f} ± {data['baseline_std']:.2f} | "
                    f"{data['percent_change']:+.1f}% | "
                    f"{(1 - data['p_value']) * 100:.1f}% |\n"
                )
            f.write("\n")
        
        # Trend Analysis
        f.write("## Performance Trends\n\n")
        f.write("| Metric | Trend | Significance | Current State |\n")
        f.write("|--------|-------|--------------|---------------|\n")
        
        for metric_name, trend_data in trends.items():
            trend_direction = "↑" if trend_data['slope'] > 0 else "↓"
            significance = "Significant" if trend_data['significant_trend'] else "Not significant"
            f.write(
                f"| {metric_name} | {trend_direction} | {significance} | "
                f"{trend_data['current_mean']:.2f} ± {trend_data['current_std']:.2f} |\n"
            )
    
    return str(report_file)

async def main():
    """Run regression tests and generate report."""
    from redis_benchmarks import redis_manager
    
    # Initialize components
    benchmark_suite = RedisBenchmarkSuite(redis_manager)
    analyzer = RegressionAnalyzer()
    
    try:
        # Ensure baseline exists
        baseline = await analyzer.capture_baseline(benchmark_suite)
        logger.info("Baseline loaded/captured")
        
        # Run regression tests
        passed, results = await analyzer.run_regression_test(benchmark_suite)
        logger.info(f"Regression test {'passed' if passed else 'failed'}")
        
        # Analyze trends
        trends = await analyzer.analyze_trends()
        logger.info("Trend analysis completed")
        
        # Generate report
        report_path = create_regression_report(results, trends)
        logger.info(f"Report generated: {report_path}")
        
        return passed
        
    except Exception as e:
        logger.error(f"Regression testing failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(main())
