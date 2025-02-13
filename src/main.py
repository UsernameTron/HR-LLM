"""
Main entry point for the Hiring Sentiment Tracker.
Coordinates data ingestion, processing, and monitoring.
"""
import asyncio
import logging
from typing import Dict, List

from config.config import settings
from src.data.ingestion import DataIngestionPipeline
from src.data.processors import (GDELTProcessor, LinkedInProcessor,
                               NewsAPIProcessor)
from src.utils.metrics import MetricsTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HiringSentinelPipeline:
    def __init__(self):
        self.ingestion = DataIngestionPipeline()
        self.metrics = MetricsTracker()
        
        # Initialize processors
        self.processors = {
            'newsapi': NewsAPIProcessor(),
            'linkedin': LinkedInProcessor(),
            'gdelt': GDELTProcessor()
        }
    
    async def start(self):
        """Start the pipeline with all processors."""
        try:
            logger.info("Starting Hiring Sentinel Pipeline...")
            
            # Start message processing
            await self.ingestion.process_messages(self.handle_batch)
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            raise
        finally:
            await self.cleanup()
    
    async def handle_batch(self, messages: List[Dict]) -> None:
        """Handle a batch of messages with appropriate processor."""
        tasks = []
        
        for message in messages:
            source = message.get('source')
            if source in self.processors:
                processor = self.processors[source]
                tasks.append(
                    self._process_message(message, processor)
                )
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _process_message(self, message: Dict, processor: any) -> None:
        """Process a single message with error handling."""
        try:
            start_time = self.metrics.get_current_timestamp()
            
            # Process message
            processed_data = await processor.process_batch([message])
            
            # Record processing time
            duration = self.metrics.get_current_timestamp() - start_time
            self.metrics.record_processing_time(
                'processing',
                message['source'],
                duration
            )
            
            if processed_data:
                # Re-ingest processed data
                for item in processed_data:
                    await self.ingestion.ingest_data(
                        f"{message['source']}_processed",
                        item
                    )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self.metrics.record_failure(
                'processing',
                message['source'],
                str(e)
            )
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.ingestion.close()
            for processor in self.processors.values():
                await processor.close()
                
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            raise

async def main():
    """Main entry point."""
    pipeline = HiringSentinelPipeline()
    try:
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
