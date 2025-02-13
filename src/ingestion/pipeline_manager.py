"""Data pipeline manager integrating Kafka, Redis Streams, and classification."""
import asyncio
import json
import logging
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from src.ingestion.kafka_consumer import KafkaDataConsumer, ConsumerConfig
from src.classification.model import SentimentClassifier
from src.utils.metal_error_handler import MetalError, MetalErrorCategory
from src.cache.redis_memory_monitor import RedisMemoryMonitor
from src.monitoring.drift_detector import DriftDetector
from src.classification.explainer import RealTimeExplainer

@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    kafka_config: ConsumerConfig
    redis_stream_key: str
    batch_size: int = 32
    processing_timeout: int = 30
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    use_mock_consumer: bool = False

class DataPipelineManager:
    """Manages data flow from Kafka to Redis Streams with classification."""
    
    def __init__(
        self,
        config: PipelineConfig,
        redis_client: Any,
        classifier: SentimentClassifier,
        redis_monitor: RedisMemoryMonitor,
        drift_detector: Optional[DriftDetector] = None,
        explainer: Optional[RealTimeExplainer] = None
    ):
        self.config = config
        self.redis = redis_client
        self.classifier = classifier
        self.redis_monitor = redis_monitor
        self.logger = logging.getLogger(__name__)
        
        # Initialize drift detector if not provided
        self.drift_detector = drift_detector or DriftDetector()
        
        # Initialize explainer if not provided
        self.explainer = explainer or RealTimeExplainer(
            tokenizer=classifier.tokenizer,
            device=classifier.device
        )
        
        # Initialize Kafka consumer
        if hasattr(config, 'use_mock_consumer') and config.use_mock_consumer:
            from tests.mocks.mock_kafka import MockKafkaConsumer
            self.kafka_consumer = MockKafkaConsumer(
                config=config.kafka_config,
                processor=self._validate_data
            )
        else:
            self.kafka_consumer = KafkaDataConsumer(
                config=config.kafka_config,
                redis_monitor=redis_monitor,
                processor=self._validate_data
            )
        
    async def initialize(self):
        """Initialize the pipeline components."""
        try:
            await self.kafka_consumer.initialize()
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            raise MetalError(
                f"Failed to initialize pipeline: {str(e)}",
                category=MetalErrorCategory.INGESTION_ERROR
            )
    
    def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean incoming data."""
        required_fields = {"text", "source", "timestamp"}
        
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")
            
            if not all(field in data for field in required_fields):
                missing = required_fields - set(data.keys())
                raise ValueError(f"Missing required fields: {missing}")
            
            # Clean and validate text
            if not isinstance(data["text"], str) or not data["text"].strip():
                raise ValueError("Invalid or empty text field")
            data["text"] = data["text"].strip()
            
            # Validate timestamp
            try:
                datetime.fromisoformat(data["timestamp"])
            except ValueError:
                data["timestamp"] = datetime.utcnow().isoformat()
            
            return data
            
        except Exception as e:
            raise MetalError(
                f"Data validation error: {str(e)}",
                category=MetalErrorCategory.VALIDATION_ERROR
            )
    
    async def _cache_llm_response(
        self,
        cache_key: str,
        response: Dict[str, Any]
    ) -> None:
        """Cache LLM API response in Redis."""
        try:
            await self.redis.set(
                f"llm:response:{cache_key}",
                json.dumps(response),
                ex=self.config.cache_ttl
            )
        except Exception as e:
            self.logger.error(f"Failed to cache LLM response: {str(e)}")
    
    async def _get_cached_response(
        self,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached LLM response."""
        try:
            cached = await self.redis.get(f"llm:response:{cache_key}")
            return json.loads(cached) if cached else None
        except Exception as e:
            self.logger.error(f"Failed to get cached response: {str(e)}")
            return None
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of messages with classification and drift detection."""
        try:
            # Extract texts for classification
            texts = [item["text"] for item in batch]
            
            # Get predictions and explanations
            predictions = self.classifier.predict(texts, return_probabilities=True)
            embeddings = self.classifier.get_embeddings(texts)
            
            # Process results
            results = []
            for i, item in enumerate(batch):
                # Generate cache key
                cache_key = f"{item['source']}:{item['text'][:50]}"
                
                # Check cache first
                cached_result = await self._get_cached_response(cache_key)
                if cached_result:
                    results.append(cached_result)
                    continue
                
                # Get prediction confidence
                confidence = float(torch.max(predictions["probabilities"][i]))
                
                # Generate explanation
                explanation = self.explainer.explain_prediction(
                    item["text"],
                    predictions["predictions"][i],
                    self.classifier.model,
                    confidence
                )
                
                # Check for drift
                drift_metrics = self.drift_detector.detect_drift(
                    item["text"],
                    embeddings[i],
                    predictions["probabilities"][i]
                )
                
                # Combine results
                pred_result = {
                    "predictions": predictions["predictions"][i].tolist(),
                    "probabilities": predictions["probabilities"][i].tolist(),
                    "explanation": {
                        "feature_importance": explanation.feature_importance,
                        "token_attributions": explanation.token_attributions,
                        "confidence": explanation.explanation_confidence
                    },
                    "drift_metrics": drift_metrics._asdict() if drift_metrics else None
                }
                
                # Combine with original data
                result = {
                    **item,
                    "classification": pred_result,
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                # Cache result
                await self._cache_llm_response(cache_key, result)
                results.append(result)
                
                # Check if reference distribution should be updated
                if self.drift_detector.should_update_reference():
                    self.logger.warning("Updating reference distribution due to drift")
                    self.drift_detector.update_reference(
                        texts,
                        embeddings,
                        predictions["probabilities"]
                    )
                    # Update explainer background
                    self.explainer.update_background(texts)
            
            return results
            
        except Exception as e:
            raise MetalError(
                f"Batch processing error: {str(e)}",
                category=MetalErrorCategory.PROCESSING_ERROR
            )
    
    async def _stream_to_redis(
        self,
        processed_data: List[Dict[str, Any]]
    ) -> None:
        """Stream processed data to Redis Streams."""
        try:
            for item in processed_data:
                # Convert to Redis stream format
                stream_data = {
                    "data": json.dumps(item)
                }
                
                # Add to Redis stream
                await self.redis.xadd(
                    self.config.redis_stream_key,
                    stream_data,
                    maxlen=10000  # Limit stream length
                )
        except Exception as e:
            raise MetalError(
                f"Redis streaming error: {str(e)}",
                category=MetalErrorCategory.STREAMING_ERROR
            )
    
    async def run(self):
        """Run the pipeline."""
        batch = []
        try:
            async for message in self.kafka_consumer.consume():
                batch.append(message)
                
                if len(batch) >= self.config.batch_size:
                    # Process batch
                    processed_data = await self._process_batch(batch)
                    
                    # Stream to Redis
                    await self._stream_to_redis(processed_data)
                    
                    # Clear batch
                    batch = []
                    
                # Check memory pressure
                metrics = await self.redis_monitor.get_memory_metrics()
                alerts = self.redis_monitor.check_memory_pressure(metrics)
                if any(alerts.values()):
                    await self.redis_monitor.handle_memory_pressure(alerts)
                    
        except Exception as e:
            raise MetalError(
                f"Pipeline error: {str(e)}",
                category=MetalErrorCategory.PIPELINE_ERROR
            )
        finally:
            # Process remaining messages
            if batch:
                processed_data = await self._process_batch(batch)
                await self._stream_to_redis(processed_data)
            
            await self.kafka_consumer.close()
    
    async def monitor_performance(self):
        """Monitor pipeline performance metrics."""
        while True:
            try:
                # Get memory metrics
                metrics = await self.redis_monitor.get_memory_metrics()
                
                # Log performance metrics
                self.logger.info(
                    "Pipeline metrics - "
                    f"Memory usage: {metrics.used_memory / 1024 / 1024:.2f}MB, "
                    f"Cache hits: {await self.redis.info('stats')['keyspace_hits']}, "
                    f"Processing rate: {self.config.batch_size}/batch"
                )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(5)  # Retry after 5 seconds
