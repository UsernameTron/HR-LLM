"""
Data ingestion pipeline using RSS feeds and web scraping for hiring signals.
Optimized for reliability and simplicity.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self):
        self.metrics = MetricsTracker()
        self.session = None
        
        # Default RSS feeds for job postings
        self.rss_feeds = [
            'https://www.indeed.com/rss?q=software',
            'https://stackoverflow.com/jobs/feed',
            'https://www.dice.com/rss'
        ]
        
        # Web sources to scrape
        self.web_sources = [
            'https://www.linkedin.com/jobs',
            'https://www.glassdoor.com/Job'
        ]
        
    async def init_session(self):
        """Initialize aiohttp session for async requests"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def ingest_data(self) -> List[Dict]:
        """
        Collect job posting data from RSS feeds and web sources.
        Returns a list of job postings with metadata.
        """
        try:
            await self.init_session()
            
            # Collect data from both sources
            rss_data = await self.process_rss_feeds()
            web_data = await self.process_web_sources()
            
            # Combine and deduplicate
            all_data = rss_data + web_data
            seen_urls = set()
            unique_data = []
            
            for item in all_data:
                if item['url'] not in seen_urls:
                    seen_urls.add(item['url'])
                    unique_data.append(item)
            
            return unique_data
            
        except Exception as e:
            logger.error(f"Error ingesting data: {str(e)}")
            raise
            
    async def process_rss_feeds(self) -> List[Dict]:
        """Process all RSS feeds for job postings"""
        results = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    results.append({
                        'title': entry.title,
                        'description': entry.description,
                        'url': entry.link,
                        'source': feed_url,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'rss'
                    })
                    
            except Exception as e:
                logger.error(f"Error processing RSS feed {feed_url}: {str(e)}")
                continue
                
        return results
    
    async def process_web_sources(self) -> List[Dict]:
        """Scrape job postings from web sources"""
        results = []
        
        for source_url in self.web_sources:
            try:
                async with self.session.get(source_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract job postings (customize selectors per site)
                        job_elements = soup.find_all('div', class_='job-posting')
                        
                        for job in job_elements:
                            results.append({
                                'title': job.find('h2').text.strip(),
                                'description': job.find('div', class_='description').text.strip(),
                                'url': job.find('a')['href'],
                                'source': source_url,
                                'timestamp': datetime.now().isoformat(),
                                'type': 'web'
                            })
                            
            except Exception as e:
                logger.error(f"Error scraping {source_url}: {str(e)}")
                continue
                
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
    async def send_to_kafka(self, source: str, data: dict) -> bool:
        """
        Send ingested data to Kafka topic with proper error handling and metrics tracking.
        
        Args:
            source: The data source identifier
            data: The data to be sent
            
        Returns:
            bool: True if successful, False otherwise
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
