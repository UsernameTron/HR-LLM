"""Pre-production validation suite."""
import asyncio
import logging
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

from src.ingestion.pipeline_manager import PipelineManager
from src.classification.model import SentimentClassifier
from src.monitoring.drift_detector import DriftDetector
from src.classification.explainer import RealTimeExplainer
from src.cache.redis_memory_monitor import RedisMemoryMonitor
from tests.load.load_test_pipeline import LoadTestRunner
from tests.integration.api_compatibility_test import APICompatibilityTester
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

class PreProdValidator:
    """Manages pre-production validation."""
    
    def __init__(
        self,
        pipeline: PipelineManager,
        output_dir: str = "validation_results"
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize test suites
        self.load_tester = LoadTestRunner(
            pipeline,
            test_duration=3600,  # 1 hour
            message_rate=100,
            batch_size=32
        )
        self.api_tester = APICompatibilityTester(
            pipeline.api_aggregator,
            test_duration=3600,
            request_rate=1.0
        )
    
    async def validate_memory_usage(self) -> Dict[str, Any]:
        """Validate memory usage patterns."""
        memory_metrics = []
        start_time = datetime.utcnow()
        
        # Monitor memory for 5 minutes
        for _ in range(30):  # 30 samples, 10 seconds apart
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_memory": self.pipeline.redis_monitor.get_memory_usage(),
                "cache_memory": self.pipeline.redis_monitor.get_cache_memory_usage(),
                "fragmentation": self.pipeline.redis_monitor.get_fragmentation_ratio()
            }
            memory_metrics.append(metrics)
            await asyncio.sleep(10)
        
        return {
            "duration": (datetime.utcnow() - start_time).total_seconds(),
            "samples": memory_metrics,
            "summary": {
                "max_memory": max(m["total_memory"] for m in memory_metrics),
                "avg_memory": sum(m["total_memory"] for m in memory_metrics) / len(memory_metrics),
                "max_fragmentation": max(m["fragmentation"] for m in memory_metrics),
                "cache_efficiency": sum(m["cache_memory"] for m in memory_metrics) / 
                                  sum(m["total_memory"] for m in memory_metrics)
            }
        }
    
    async def validate_cache_performance(self) -> Dict[str, Any]:
        """Validate cache performance."""
        # Generate test data
        test_texts = [
            "Senior software engineer position",
            "Entry level developer role",
            "Technical lead opportunity",
            "Software architect position",
            "Full stack developer needed"
        ] * 20  # 100 samples with duplicates
        
        cache_metrics = {
            "hits": 0,
            "misses": 0,
            "latencies": []
        }
        
        # Process texts and measure cache performance
        for text in test_texts:
            start_time = datetime.utcnow()
            cache_key = f"test:{text[:50]}"
            
            # Check cache
            cached = await self.pipeline._get_cached_response(cache_key)
            if cached:
                cache_metrics["hits"] += 1
            else:
                cache_metrics["misses"] += 1
                # Process and cache
                result = await self.pipeline._process_batch([{"text": text}])
                await self.pipeline._cache_llm_response(cache_key, result[0])
            
            latency = (datetime.utcnow() - start_time).total_seconds()
            cache_metrics["latencies"].append(latency)
        
        return {
            "samples": len(test_texts),
            "hit_rate": cache_metrics["hits"] / len(test_texts),
            "avg_latency": sum(cache_metrics["latencies"]) / len(cache_metrics["latencies"]),
            "cache_size": self.pipeline.redis_monitor.get_cache_memory_usage()
        }
    
    async def validate_data_flow(self) -> Dict[str, Any]:
        """Validate end-to-end data flow."""
        flow_metrics = {
            "ingestion": [],
            "processing": [],
            "output": []
        }
        
        # Test messages
        messages = [
            {
                "text": "Senior software engineer position",
                "source": "validation",
                "timestamp": datetime.utcnow().isoformat()
            }
            for _ in range(10)
        ]
        
        # Track each stage
        for msg in messages:
            try:
                # Ingestion
                ingest_start = datetime.utcnow()
                validated = await self.pipeline._validate_data(msg)
                flow_metrics["ingestion"].append({
                    "success": True,
                    "latency": (datetime.utcnow() - ingest_start).total_seconds()
                })
                
                # Processing
                process_start = datetime.utcnow()
                result = await self.pipeline._process_batch([validated])
                flow_metrics["processing"].append({
                    "success": True,
                    "latency": (datetime.utcnow() - process_start).total_seconds(),
                    "drift_detected": result[0]["classification"].get("drift_metrics") is not None
                })
                
                # Output (Redis Stream)
                output_start = datetime.utcnow()
                await self.pipeline._stream_results(result)
                flow_metrics["output"].append({
                    "success": True,
                    "latency": (datetime.utcnow() - output_start).total_seconds()
                })
                
            except Exception as e:
                self.logger.error(f"Data flow validation error: {str(e)}")
                for stage in flow_metrics:
                    if len(flow_metrics[stage]) < len(messages):
                        flow_metrics[stage].append({
                            "success": False,
                            "error": str(e)
                        })
        
        return {
            "samples": len(messages),
            "success_rate": {
                stage: sum(1 for m in metrics if m["success"]) / len(metrics)
                for stage, metrics in flow_metrics.items()
            },
            "avg_latency": {
                stage: sum(m["latency"] for m in metrics if m.get("latency")) / 
                       sum(1 for m in metrics if m.get("latency"))
                for stage, metrics in flow_metrics.items()
            },
            "drift_detection_rate": sum(
                1 for m in flow_metrics["processing"]
                if m.get("drift_detected", False)
            ) / len(flow_metrics["processing"])
        }
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete pre-production validation."""
        self.logger.info("Starting pre-production validation")
        
        try:
            # 1. Load Testing
            load_test_metrics = await self.load_tester.run_load_test()
            load_test_report = self.load_tester.generate_report(load_test_metrics)
            
            # 2. API Compatibility
            api_report = await self.api_tester.run_compatibility_test()
            rate_limit_violations = self.api_tester.validate_rate_limits(api_report)
            
            # 3. Memory Usage
            memory_report = await self.validate_memory_usage()
            
            # 4. Cache Performance
            cache_report = await self.validate_cache_performance()
            
            # 5. Data Flow
            flow_report = await self.validate_data_flow()
            
            # Compile results
            validation_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "load_test": load_test_report,
                "api_compatibility": api_report,
                "memory_usage": memory_report,
                "cache_performance": cache_report,
                "data_flow": flow_report,
                "validation_passed": all([
                    load_test_report["thresholds_met"]["throughput"],
                    load_test_report["thresholds_met"]["latency"],
                    load_test_report["thresholds_met"]["memory"],
                    api_report["thresholds_met"]["success_rate"],
                    not rate_limit_violations,
                    memory_report["summary"]["max_memory"] <= 1024 * 48,  # 48GB limit
                    cache_report["hit_rate"] >= 0.8,  # 80% hit rate
                    all(rate >= 0.99 for rate in flow_report["success_rate"].values())
                ])
            }
            
            # Save report
            report_path = self.output_dir / f"validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            return validation_report
            
        except Exception as e:
            raise MetalError(
                f"Pre-production validation failed: {str(e)}",
                category=MetalErrorCategory.VALIDATION_ERROR
            )
