"""API aggregator for hiring data sources with rate limiting and validation."""
import asyncio
import aiohttp
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from aiokafka import AIOKafkaProducer
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@dataclass
class APIConfig:
    """Configuration for API endpoints."""
    endpoint: str
    api_key: str
    rate_limit: int  # requests per minute
    batch_size: int = 100
    timeout: int = 30

@dataclass
class SourceMetrics:
    """Metrics for data source quality."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    last_timestamp: Optional[str] = None
    schema_violations: Dict[str, int] = None

    def __post_init__(self):
        self.schema_violations = {}

class APIAggregator:
    """Manages API data collection with rate limiting and validation."""
    
    def __init__(
        self,
        kafka_producer: AIOKafkaProducer,
        topic: str,
        apis: Dict[str, APIConfig]
    ):
        self.producer = kafka_producer
        self.topic = topic
        self.apis = apis
        self.logger = logging.getLogger(__name__)
        self.source_metrics = {
            source: SourceMetrics() for source in apis.keys()
        }
        self._rate_limiters = {}
        self._initialize_rate_limiters()
    
    def _initialize_rate_limiters(self):
        """Initialize token bucket rate limiters for each API."""
        for source, config in self.apis.items():
            self._rate_limiters[source] = asyncio.Queue(maxsize=config.rate_limit)
            asyncio.create_task(self._rate_limiter_task(source))
    
    async def _rate_limiter_task(self, source: str):
        """Replenish rate limit tokens."""
        config = self.apis[source]
        while True:
            try:
                if self._rate_limiters[source].qsize() < config.rate_limit:
                    await self._rate_limiters[source].put(1)
                await asyncio.sleep(60 / config.rate_limit)  # Distribute over minute
            except Exception as e:
                self.logger.error(f"Rate limiter error for {source}: {str(e)}")
                await asyncio.sleep(1)
    
    async def _get_rate_limit_token(self, source: str):
        """Get a token from the rate limiter."""
        await self._rate_limiters[source].get()
    
    def _validate_record(
        self,
        record: Dict[str, Any],
        source: str
    ) -> Optional[Dict[str, Any]]:
        """Validate individual record schema and content."""
        required_fields = {
            "text": str,
            "timestamp": str,
            "source_id": str,
            "metadata": dict
        }
        
        try:
            # Check required fields and types
            for field, field_type in required_fields.items():
                if field not in record:
                    self.source_metrics[source].schema_violations[field] = \
                        self.source_metrics[source].schema_violations.get(field, 0) + 1
                    return None
                
                if not isinstance(record[field], field_type):
                    self.source_metrics[source].schema_violations[f"{field}_type"] = \
                        self.source_metrics[source].schema_violations.get(f"{field}_type", 0) + 1
                    return None
            
            # Validate timestamp
            try:
                timestamp = datetime.fromisoformat(record["timestamp"])
                if timestamp > datetime.utcnow() + timedelta(hours=1):
                    self.source_metrics[source].schema_violations["future_timestamp"] = \
                        self.source_metrics[source].schema_violations.get("future_timestamp", 0) + 1
                    return None
            except ValueError:
                self.source_metrics[source].schema_violations["invalid_timestamp"] = \
                    self.source_metrics[source].schema_violations.get("invalid_timestamp", 0) + 1
                return None
            
            # Validate text content
            if not record["text"].strip():
                self.source_metrics[source].schema_violations["empty_text"] = \
                    self.source_metrics[source].schema_violations.get("empty_text", 0) + 1
                return None
            
            # Update metrics
            self.source_metrics[source].valid_records += 1
            self.source_metrics[source].last_timestamp = record["timestamp"]
            
            return record
            
        except Exception as e:
            self.logger.error(f"Validation error for {source}: {str(e)}")
            return None
    
    async def _fetch_data(
        self,
        session: aiohttp.ClientSession,
        source: str,
        config: APIConfig
    ) -> List[Dict[str, Any]]:
        """Fetch data from API with rate limiting."""
        try:
            # Get rate limit token
            await self._get_rate_limit_token(source)
            
            async with session.get(
                config.endpoint,
                headers={"Authorization": f"Bearer {config.api_key}"},
                timeout=config.timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Update metrics
                self.source_metrics[source].total_records += len(data)
                
                # Validate records
                valid_records = []
                for record in data:
                    validated = self._validate_record(record, source)
                    if validated:
                        valid_records.append(validated)
                    else:
                        self.source_metrics[source].invalid_records += 1
                
                return valid_records
                
        except aiohttp.ClientError as e:
            raise MetalError(
                f"API request failed for {source}: {str(e)}",
                category=MetalErrorCategory.API_ERROR
            )
    
    async def _produce_to_kafka(
        self,
        records: List[Dict[str, Any]],
        source: str
    ):
        """Produce validated records to Kafka."""
        try:
            for record in records:
                await self.producer.send_and_wait(
                    self.topic,
                    json.dumps(record).encode()
                )
        except Exception as e:
            raise MetalError(
                f"Failed to produce to Kafka for {source}: {str(e)}",
                category=MetalErrorCategory.KAFKA_ERROR
            )
    
    def get_data_quality_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get data quality metrics for all sources."""
        return {
            source: {
                "total_records": metrics.total_records,
                "valid_records": metrics.valid_records,
                "invalid_records": metrics.invalid_records,
                "last_timestamp": metrics.last_timestamp,
                "schema_violations": metrics.schema_violations,
                "validity_rate": (
                    metrics.valid_records / metrics.total_records
                    if metrics.total_records > 0 else 0
                )
            }
            for source, metrics in self.source_metrics.items()
        }
    
    async def run(self):
        """Run the API aggregator."""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    tasks = []
                    for source, config in self.apis.items():
                        task = asyncio.create_task(
                            self._fetch_data(session, source, config)
                        )
                        tasks.append(task)
                    
                    # Gather results
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for source, result in zip(self.apis.keys(), results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error fetching from {source}: {str(result)}")
                            continue
                        
                        if result:
                            await self._produce_to_kafka(result, source)
                    
                    # Log metrics periodically
                    self.logger.info("Data quality metrics: %s", self.get_data_quality_metrics())
                    
                    await asyncio.sleep(60)  # Poll every minute
                    
                except Exception as e:
                    self.logger.error(f"Aggregator error: {str(e)}")
                    await asyncio.sleep(5)  # Retry after 5 seconds
