"""Main entry point for running the sentiment analysis pipeline."""
import asyncio
import logging
from typing import Dict, Any

import redis.asyncio as redis
from src.ingestion.pipeline_manager import DataPipelineManager, PipelineConfig
from src.ingestion.kafka_consumer import ConsumerConfig
from src.classification.model import SentimentClassifier
from src.monitoring.performance_monitor import PerformanceMonitor
from src.cache.redis_memory_monitor import RedisMemoryMonitor
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

async def main():
    """Initialize and run the pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize Redis
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Initialize Redis monitor
        redis_monitor = RedisMemoryMonitor(redis_client)
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Initialize classifier
        classifier = SentimentClassifier(
            model_name="distilbert-base-uncased",
            num_labels=3  # Positive, Negative, Neutral
        )
        
        # Configure pipeline
        pipeline_config = PipelineConfig(
            kafka_config=ConsumerConfig(
                topic="hiring_sentiment",
                group_id="sentiment_processor",
                bootstrap_servers=["localhost:9092"]
            ),
            redis_stream_key="sentiment:stream",
            batch_size=32
        )
        
        # Initialize pipeline
        pipeline = DataPipelineManager(
            config=pipeline_config,
            redis_client=redis_client,
            classifier=classifier,
            redis_monitor=redis_monitor
        )
        
        # Initialize components
        await pipeline.initialize()
        
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(
            performance_monitor.monitor_system_health(redis_client)
        )
        pipeline_monitoring_task = asyncio.create_task(
            pipeline.monitor_performance()
        )
        
        # Run pipeline
        logger.info("Starting sentiment analysis pipeline...")
        await pipeline.run()
        
    except Exception as e:
        raise MetalError(
            f"Pipeline execution error: {str(e)}",
            category=MetalErrorCategory.PIPELINE_ERROR
        )
    finally:
        # Cleanup
        await redis_client.close()
        
if __name__ == "__main__":
    asyncio.run(main())
