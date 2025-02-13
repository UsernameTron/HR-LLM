"""Execute load tests with various configurations."""
import asyncio
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.ingestion.pipeline_manager import DataPipelineManager, PipelineConfig
from src.ingestion.kafka_consumer import ConsumerConfig
from src.classification.model import SentimentClassifier
from src.monitoring.drift_detector import DriftDetector
from src.classification.explainer import RealTimeExplainer
from src.cache.redis_memory_monitor import RedisMemoryMonitor
from tests.load.load_test_pipeline import LoadTestRunner, LoadTestMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configurations
TEST_DURATION = 900  # 15 minutes
MESSAGE_RATES = [100, 250, 500]  # messages per second
BATCH_SIZES = [32, 64, 128]

# Success criteria
SUCCESS_CRITERIA = {
    "latency": 1000.0,  # P95 latency in ms
    "error_rate": 0.01,  # 1% maximum
    "memory": 40960.0,  # 85% of 48GB in MB
    "min_entropy": 0.5,
    "max_entropy": 2.5,
    "min_sparsity": 0.1
}

class MockRedis:
    """Mock Redis client for testing."""
    def __init__(self):
        self.data = {}
        self.stream_data = {}
    
    async def get(self, key):
        return self.data.get(key)
    
    async def set(self, key, value, ex=None):
        self.data[key] = value
    
    async def xadd(self, stream, fields, id='*', maxlen=None, approximate=True):
        if stream not in self.stream_data:
            self.stream_data[stream] = []
        self.stream_data[stream].append(fields)
        return b'1-0'
    
    async def info(self, section=None):
        return {
            'used_memory': 1000000,
            'used_memory_rss': 2000000,
            'used_memory_peak': 3000000
        }

class LoadTestOrchestrator:
    """Manages progressive load testing with success criteria validation."""
    
    def __init__(self):
        self.pipeline = None
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.redis_client = MockRedis()
        self.results = []
    
    def validate_metrics(self, metrics: LoadTestMetrics) -> Dict[str, bool]:
        """Validate metrics against success criteria."""
        return {
            "latency": metrics.latency_p95 < SUCCESS_CRITERIA["latency"],
            "error_rate": metrics.error_rate < SUCCESS_CRITERIA["error_rate"],
            "memory": metrics.memory_usage < SUCCESS_CRITERIA["memory"],
            "entropy": (SUCCESS_CRITERIA["min_entropy"] <= metrics.attribution_entropy <= 
                       SUCCESS_CRITERIA["max_entropy"]),
            "sparsity": metrics.attribution_sparsity > SUCCESS_CRITERIA["min_sparsity"]
        }
    
    async def setup_pipeline(self):
        """Initialize the pipeline components."""
        config = PipelineConfig(
            kafka_config=ConsumerConfig(
                topic="test_topic",
                group_id="load_test_group",
                bootstrap_servers=["localhost:9092"],
                max_poll_records=500,
                max_poll_interval_ms=300000,
                retry_backoff_ms=500,
                max_retries=3
            ),
            redis_stream_key="test_stream",
            batch_size=32,
            processing_timeout=30,
            use_mock_consumer=True
        )
        
        # Initialize components with mock Redis
        classifier = SentimentClassifier()
        redis_monitor = RedisMemoryMonitor(self.redis_client)
        drift_detector = DriftDetector()
        explainer = RealTimeExplainer(tokenizer=classifier.tokenizer, device=classifier.device)
        
        self.pipeline = DataPipelineManager(
            config=config,
            redis_client=self.redis_client,
            classifier=classifier,
            redis_monitor=redis_monitor,
            drift_detector=drift_detector,
            explainer=explainer
        )
        await self.pipeline.initialize()
    
    def save_results(self, scenario_name: str, metrics_history: List[LoadTestMetrics]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"load_test_{scenario_name}_{timestamp}.json"
        
        # Convert metrics to dictionary
        results = {
            "scenario": scenario_name,
            "timestamp": timestamp,
            "metrics": [
                {
                    "throughput": m.throughput,
                    "latency_p95": m.latency_p95,
                    "memory_usage": m.memory_usage,
                    "cache_hit_rate": m.cache_hit_rate,
                    "error_rate": m.error_rate,
                    "drift_detection_rate": m.drift_detection_rate,
                    "explanation_time_p95": m.explanation_time_p95
                }
                for m in metrics_history
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def analyze_results(self, metrics_history: List[LoadTestMetrics]) -> Dict:
        """Analyze test results and generate summary statistics."""
        metrics_array = np.array([
            [m.throughput, m.latency_p95, m.memory_usage, m.cache_hit_rate,
             m.error_rate, m.drift_detection_rate, m.explanation_time_p95]
            for m in metrics_history
        ])
        
        return {
            "throughput": {
                "mean": float(np.mean(metrics_array[:, 0])),
                "std": float(np.std(metrics_array[:, 0])),
                "p95": float(np.percentile(metrics_array[:, 0], 95))
            },
            "latency": {
                "mean": float(np.mean(metrics_array[:, 1])),
                "std": float(np.std(metrics_array[:, 1])),
                "p95": float(np.percentile(metrics_array[:, 1], 95))
            },
            "memory": {
                "mean": float(np.mean(metrics_array[:, 2])),
                "max": float(np.max(metrics_array[:, 2])),
                "growth_rate": float(np.polyfit(range(len(metrics_array)), metrics_array[:, 2], 1)[0])
            },
            "cache": {
                "hit_rate_mean": float(np.mean(metrics_array[:, 3])),
                "hit_rate_std": float(np.std(metrics_array[:, 3]))
            },
            "errors": {
                "rate_mean": float(np.mean(metrics_array[:, 4])),
                "total": float(np.sum(metrics_array[:, 4]))
            }
        }
    
    async def run_configuration(self, message_rate: int, batch_size: int) -> bool:
        """Run a single test configuration and validate results."""
        logger.info(f"\nStarting configuration: {message_rate} msg/s, batch={batch_size}")
        
        try:
            # Initialize and run test with proper cleanup
            async with LoadTestRunner(
                pipeline=self.pipeline,
                test_duration=TEST_DURATION,
                message_rate=message_rate,
                batch_size=batch_size
            ) as runner:
                # Run test and collect metrics
                metrics_history = await runner.run_load_test()
                if not metrics_history:
                    return False
            
            # Get final metrics
            final_metrics = metrics_history[-1]
            validation = self.validate_metrics(final_metrics)
            
            # Save results
            result = {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "message_rate": message_rate,
                    "batch_size": batch_size,
                    "duration": TEST_DURATION
                },
                "metrics": {
                    "throughput": final_metrics.throughput,
                    "latency_p95": final_metrics.latency_p95,
                    "memory_usage": final_metrics.memory_usage,
                    "cache_hit_rate": final_metrics.cache_hit_rate,
                    "error_rate": final_metrics.error_rate,
                    "drift_detection_rate": final_metrics.drift_detection_rate,
                    "explanation_time_p95": final_metrics.explanation_time_p95,
                    "attribution_entropy": final_metrics.attribution_entropy,
                    "attribution_sparsity": final_metrics.attribution_sparsity,
                    "attribution_gradient_norm": final_metrics.attribution_gradient_norm
                },
                "validation": validation,
                "success": all(validation.values())
            }
            self.results.append(result)
            
            # Log results
            logger.info("Results:")
            for criterion, passed in validation.items():
                status = "✓" if passed else "✗"
                logger.info(f"{criterion}: {status}")
            
            return all(validation.values())
            
        except Exception as e:
            logger.error(f"Configuration failed: {str(e)}")
            return False

async def main():
    import argparse, sys, time
    parser = argparse.ArgumentParser(description='Run load tests for sentiment analysis pipeline')
    parser.add_argument('--message-rate', type=int, default=100, help='Messages per second')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--duration', type=int, default=900, help='Test duration in seconds')
    args = parser.parse_args()
    
    orchestrator = LoadTestOrchestrator()
    await orchestrator.setup_pipeline()
    
    try:
        logger.info(f"Starting load test with configuration:")
        logger.info(f"Message Rate: {args.message_rate} msg/s")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Duration: {args.duration}s")
        
        success = await orchestrator.run_configuration(args.message_rate, args.batch_size)
        
        # Save final results
        results_file = orchestrator.results_dir / f"load_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(orchestrator.results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        if not success:
            logger.error("Load test failed to meet success criteria")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Load test execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
