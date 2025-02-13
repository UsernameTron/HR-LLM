"""Kafka consumer implementation with rate limiting and backoff."""
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from aiokafka import AIOKafkaConsumer
from src.utils.metal_error_handler import MetalError, MetalErrorCategory
from src.cache.redis_memory_monitor import RedisMemoryMonitor

@dataclass
class ConsumerConfig:
    """Configuration for Kafka consumer."""
    topic: str
    group_id: str
    bootstrap_servers: list[str]
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    retry_backoff_ms: int = 500
    max_retries: int = 3

class KafkaDataConsumer:
    """Manages Kafka consumption with rate limiting and backoff."""
    
    def __init__(
        self,
        config: ConsumerConfig,
        redis_monitor: RedisMemoryMonitor,
        processor: Optional[Callable[[Dict[str, Any]], Any]] = None
    ):
        self.config = config
        self.redis_monitor = redis_monitor
        self.processor = processor or self._default_processor
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.logger = logging.getLogger(__name__)
        self._backoff_count = 0
        self._rate_limit_tokens = asyncio.Queue(maxsize=1000)
        
    async def initialize(self):
        """Initialize the Kafka consumer and rate limiter."""
        try:
            self.consumer = AIOKafkaConsumer(
                self.config.topic,
                group_id=self.config.group_id,
                bootstrap_servers=self.config.bootstrap_servers,
                max_poll_records=self.config.max_poll_records,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                retry_backoff_ms=self.config.retry_backoff_ms
            )
            await self.consumer.start()
            self.logger.info(f"Kafka consumer initialized for topic: {self.config.topic}")
            
            # Start rate limiter
            asyncio.create_task(self._rate_limiter())
            
        except Exception as e:
            raise MetalError(
                f"Failed to initialize Kafka consumer: {str(e)}",
                category=MetalErrorCategory.INGESTION_ERROR
            )
    
    async def _rate_limiter(self, tokens_per_second: int = 100):
        """Implements token bucket rate limiting."""
        while True:
            try:
                if self._rate_limit_tokens.qsize() < tokens_per_second:
                    await self._rate_limit_tokens.put(1)
                await asyncio.sleep(1/tokens_per_second)
            except Exception as e:
                self.logger.error(f"Rate limiter error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _get_rate_limit_token(self):
        """Get a token from the rate limiter."""
        await self._rate_limit_tokens.get()
    
    async def _check_memory_pressure(self):
        """Check Redis memory pressure and apply backoff if needed."""
        try:
            metrics = await self.redis_monitor.get_memory_metrics()
            alerts = self.redis_monitor.check_memory_pressure(metrics)
            
            if any(alerts.values()):
                await self.redis_monitor.handle_memory_pressure(alerts)
                backoff_time = min(
                    self.config.retry_backoff_ms * (2 ** self._backoff_count),
                    5000  # Max 5 second backoff
                )
                self._backoff_count += 1
                self.logger.warning(f"Memory pressure detected, backing off for {backoff_time}ms")
                await asyncio.sleep(backoff_time / 1000)
            else:
                self._backoff_count = 0
                
        except Exception as e:
            self.logger.error(f"Error checking memory pressure: {str(e)}")
    
    def _default_processor(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Default message processor that validates and cleans data."""
        # Implement basic validation and cleaning
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        
        # Remove null values and strip strings
        cleaned = {
            k: v.strip() if isinstance(v, str) else v
            for k, v in message.items()
            if v is not None
        }
        
        return cleaned
    
    async def consume(self):
        """Main consumption loop with rate limiting and backoff."""
        if not self.consumer:
            raise MetalError(
                "Consumer not initialized",
                category=MetalErrorCategory.INGESTION_ERROR
            )
        
        try:
            async for message in self.consumer:
                # Apply rate limiting
                await self._get_rate_limit_token()
                
                # Check memory pressure
                await self._check_memory_pressure()
                
                try:
                    # Process message
                    data = message.value
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    
                    processed_data = self.processor(data)
                    self.logger.debug(f"Processed message from partition {message.partition}")
                    
                    yield processed_data
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    if self._backoff_count < self.config.max_retries:
                        continue
                    else:
                        raise MetalError(
                            f"Max retries exceeded processing message: {str(e)}",
                            category=MetalErrorCategory.INGESTION_ERROR
                        )
                        
        except Exception as e:
            raise MetalError(
                f"Error in Kafka consumer: {str(e)}",
                category=MetalErrorCategory.INGESTION_ERROR
            )
        
    async def close(self):
        """Clean up consumer resources."""
        if self.consumer:
            await self.consumer.stop()
            self.logger.info("Kafka consumer closed")
