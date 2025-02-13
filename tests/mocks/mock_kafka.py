"""Mock Kafka consumer for testing."""
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import random

from src.ingestion.kafka_consumer import ConsumerConfig
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

logger = logging.getLogger(__name__)

class MockKafkaConsumer:
    """Mock Kafka consumer for testing."""
    
    def __init__(
        self,
        config: ConsumerConfig,
        processor: Optional[Callable[[Dict[str, Any]], Any]] = None
    ):
        self.config = config
        self.processor = processor
        self.running = False
        self.messages = []
        self._rate_limit_tokens = asyncio.Queue(maxsize=1000)
        
    async def initialize(self):
        """Initialize the mock consumer."""
        self.running = True
        logger.info(f"Mock Kafka consumer initialized for topic: {self.config.topic}")
    
    async def stop(self):
        """Stop the mock consumer."""
        self.running = False
        logger.info("Mock Kafka consumer stopped")
    
    async def generate_test_message(self) -> Dict[str, Any]:
        """Generate a test message."""
        texts = [
            "Great team environment and work-life balance",
            "Challenging projects but good learning opportunities",
            "Management needs improvement",
            "Competitive salary and benefits",
            "Limited career growth potential",
            "Excellent remote work policy",
            "High workload and stress",
            "Supportive colleagues and mentorship",
            "Poor communication from leadership",
            "Innovative technology stack"
        ]
        
        return {
            "text": random.choice(texts),
            "timestamp": asyncio.get_event_loop().time(),
            "metadata": {
                "source": "load_test",
                "confidence": random.uniform(0.7, 1.0)
            }
        }
    
    async def process_messages(self, batch_size: int) -> List[Dict[str, Any]]:
        """Process a batch of messages."""
        if not self.running:
            return []
        
        batch = []
        for _ in range(batch_size):
            message = await self.generate_test_message()
            if self.processor:
                try:
                    processed = await self.processor(message)
                    batch.append(processed)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
            else:
                batch.append(message)
        
        return batch
    
    async def get_metrics(self) -> Dict[str, float]:
        """Get consumer metrics."""
        return {
            "messages_processed": len(self.messages),
            "processing_rate": len(self.messages) / max(1, asyncio.get_event_loop().time()),
            "error_rate": 0.0,
            "avg_batch_size": self.config.max_poll_records
        }
