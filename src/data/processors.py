"""
Source-specific data processors for hiring signals.
Optimized for parallel processing on M4 Pro hardware.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import aiohttp
from transformers import pipeline

from config.config import settings, API_CONFIGS
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class BaseDataProcessor:
    def __init__(self):
        self.metrics = MetricsTracker()
        self.session = aiohttp.ClientSession()
        
        # Initialize sentiment analyzer with MPS optimization
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased",
            device=settings.DEVICE
        )
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of messages with MPS acceleration."""
        raise NotImplementedError
    
    async def close(self):
        await self.session.close()

class NewsAPIProcessor(BaseDataProcessor):
    """Processor for NewsAPI data with hiring signal detection."""
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        processed_data = []
        
        # Process in optimal batch sizes for M4 Pro
        for mini_batch in self._create_mini_batches(batch):
            texts = [item['data']['content'] for item in mini_batch]
            
            # Parallel sentiment analysis with MPS
            sentiments = await self._analyze_sentiments(texts)
            
            for item, sentiment in zip(mini_batch, sentiments):
                if self._is_hiring_signal(sentiment, item['data']):
                    processed_item = self._enrich_item(item, sentiment)
                    processed_data.append(processed_item)
        
        return processed_data
    
    def _create_mini_batches(self, batch: List[Dict], size: int = 32) -> List[List[Dict]]:
        """Create optimally sized mini-batches for MPS processing."""
        return [batch[i:i + size] for i in range(0, len(batch), size)]
    
    async def _analyze_sentiments(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiments with MPS acceleration."""
        try:
            return self.sentiment_analyzer(texts)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            self.metrics.record_failure('sentiment_analysis', 'newsapi', str(e))
            return [{'label': 'NEUTRAL', 'score': 0.5} for _ in texts]
    
    def _is_hiring_signal(self, sentiment: Dict, data: Dict) -> bool:
        """Detect hiring signals using sentiment and content analysis."""
        # Implement hiring signal detection logic
        keywords = ['hiring', 'recruitment', 'job opening', 'position available']
        return (
            sentiment['label'] == 'POSITIVE' and
            sentiment['score'] > settings.CONFIDENCE_THRESHOLD and
            any(keyword in data['content'].lower() for keyword in keywords)
        )
    
    def _enrich_item(self, item: Dict, sentiment: Dict) -> Dict:
        """Enrich item with sentiment and metadata."""
        return {
            **item,
            'sentiment': sentiment,
            'metadata': {
                **item.get('metadata', {}),
                'processing_stage': 'enriched',
                'confidence_score': sentiment['score']
            }
        }

class LinkedInProcessor(BaseDataProcessor):
    """Processor for LinkedIn data with company insights."""
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        processed_data = []
        
        for item in batch:
            try:
                # Enrich with company insights
                company_data = await self._fetch_company_insights(item['data'])
                if company_data:
                    enriched_item = self._enrich_with_company_data(item, company_data)
                    processed_data.append(enriched_item)
                
            except Exception as e:
                logger.error(f"LinkedIn processing failed: {str(e)}")
                self.metrics.record_failure('processing', 'linkedin', str(e))
        
        return processed_data
    
    async def _fetch_company_insights(self, data: Dict) -> Optional[Dict]:
        """Fetch company insights from LinkedIn API."""
        if 'company_id' not in data:
            return None
            
        api_config = API_CONFIGS['linkedin']
        url = f"{api_config['base_url']}/organizations/{data['company_id']}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch company insights: {str(e)}")
            return None
    
    def _enrich_with_company_data(self, item: Dict, company_data: Dict) -> Dict:
        """Enrich item with company insights."""
        return {
            **item,
            'company_insights': company_data,
            'metadata': {
                **item.get('metadata', {}),
                'processing_stage': 'enriched_with_company_data'
            }
        }

class GDELTProcessor(BaseDataProcessor):
    """Processor for GDELT global event database."""
    
    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        processed_data = []
        
        for mini_batch in self._create_mini_batches(batch):
            # Parallel processing of global events
            event_data = await asyncio.gather(*[
                self._process_global_event(item)
                for item in mini_batch
            ])
            
            processed_data.extend([
                item for item in event_data if item is not None
            ])
        
        return processed_data
    
    async def _process_global_event(self, item: Dict) -> Optional[Dict]:
        """Process global event data with hiring signal detection."""
        try:
            # Analyze event tone and context
            tone_scores = await self._analyze_event_tone(item['data'])
            if self._is_relevant_event(tone_scores):
                return self._enrich_with_global_context(item, tone_scores)
            return None
            
        except Exception as e:
            logger.error(f"GDELT processing failed: {str(e)}")
            self.metrics.record_failure('processing', 'gdelt', str(e))
            return None
    
    async def _analyze_event_tone(self, data: Dict) -> Dict:
        """Analyze event tone using sentiment analysis."""
        text = data.get('eventDescription', '')
        try:
            sentiment = self.sentiment_analyzer(text)[0]
            return {
                'tone': sentiment['label'],
                'score': sentiment['score'],
                'confidence': sentiment['score']
            }
        except Exception as e:
            logger.error(f"Event tone analysis failed: {str(e)}")
            return {'tone': 'NEUTRAL', 'score': 0.5, 'confidence': 0.5}
    
    def _is_relevant_event(self, tone_scores: Dict) -> bool:
        """Determine if event is relevant for hiring signals."""
        return (
            tone_scores['tone'] == 'POSITIVE' and
            tone_scores['confidence'] > settings.CONFIDENCE_THRESHOLD
        )
    
    def _enrich_with_global_context(self, item: Dict, tone_scores: Dict) -> Dict:
        """Enrich item with global context and tone analysis."""
        return {
            **item,
            'tone_analysis': tone_scores,
            'metadata': {
                **item.get('metadata', {}),
                'processing_stage': 'enriched_with_global_context'
            }
        }
