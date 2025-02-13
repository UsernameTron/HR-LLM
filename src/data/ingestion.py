"""
Data ingestion pipeline with Kafka integration for real-time hiring signals.
Optimized for high-throughput processing on M4 Pro hardware.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from config.config import settings
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Ensure durability
            compression_type='gzip',  # Optimize network bandwidth
            batch_size=32 * 1024,  # 32KB batches
            linger_ms=50,  # Wait for batch completion
        )
        
        self.consumer = KafkaConsumer(
            settings.KAFKA_TOPIC,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            group_id='hiring_signals_processor',
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_poll_records=100,  # Optimize for M4 Pro memory
        )
        
        self.metrics = MetricsTracker()
    
    async def ingest_data(self, source: str, data: Dict[str, Any]) -> bool:
        """
        Ingest data from various sources into Kafka.
        Implements backpressure handling and metrics tracking.
        """
        try:
            message = {
                'source': source,
                'timestamp': self.metrics.get_current_timestamp(),
                'data': data,
                'metadata': {
                    'version': '1.0',
                    'processing_stage': 'raw_ingestion'
                }
            }
            
            # Asynchronously send to Kafka
            future = self.producer.send(
                settings.KAFKA_TOPIC,
                value=message,
                key=source.encode('utf-8')  # Ensure related messages go to same partition
            )
            
            # Handle backpressure
            try:
                record_metadata = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=2.0
                )
                self.metrics.record_success('ingestion', source)
                logger.info(f"Successfully ingested data from {source} to partition {record_metadata.partition}")
                return True
                
            except asyncio.TimeoutError:
                self.metrics.record_failure('ingestion', source, 'timeout')
                logger.error(f"Timeout while ingesting data from {source}")
                return False
                
        except KafkaError as e:
            self.metrics.record_failure('ingestion', source, str(e))
            logger.error(f"Failed to ingest data from {source}: {str(e)}")
            return False
    
    async def process_messages(self, batch_handler: callable) -> None:
        """
        Process messages in batches for optimal performance on M4 Pro.
        Implements concurrent processing with backpressure handling.
        """
        try:
            while True:
                message_batch = self.consumer.poll(
                    timeout_ms=1000,
                    max_records=settings.BATCH_SIZE
                )
                
                if not message_batch:
                    continue
                
                # Process batches concurrently
                tasks = []
                for partition_batch in message_batch.values():
                    task = asyncio.create_task(
                        self._process_partition_batch(partition_batch, batch_handler)
                    )
                    tasks.append(task)
                
                # Wait for all batch processing to complete
                await asyncio.gather(*tasks)
                
                # Commit offsets only after successful processing
                self.consumer.commit()
                
        except Exception as e:
            logger.error(f"Error in message processing: {str(e)}")
            raise
    
    async def _process_partition_batch(
        self,
        partition_batch: List[Any],
        batch_handler: callable
    ) -> None:
        """Process a batch of messages from a single partition."""
        try:
            messages = [msg.value for msg in partition_batch]
            await batch_handler(messages)
            self.metrics.record_batch_success('processing', len(messages))
            
        except Exception as e:
            self.metrics.record_batch_failure('processing', str(e))
            logger.error(f"Failed to process batch: {str(e)}")
            raise
    
    def close(self) -> None:
        """Gracefully close Kafka connections."""
        try:
            self.producer.close()
            self.consumer.close()
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {str(e)}")
            raise
