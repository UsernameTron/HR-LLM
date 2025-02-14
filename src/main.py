"""
Main entry point for the Hiring Sentiment Tracker.
Coordinates data ingestion, processing, and monitoring.
"""
import asyncio
import logging
from typing import Dict, List, Optional

from src.models.job_analyzer import JobAnalyzer
from src.data.ingestion import DataIngestionPipeline
from src.utils.metrics_new import MetricsTracker

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.metrics import MetricsManager

# Initialize components
metrics = MetricsTracker()
job_analyzer = JobAnalyzer()
data_pipeline = DataIngestionPipeline()

# Request/Response models
class JobAnalysisRequest(BaseModel):
    text: str
    source: str
    company: Optional[str] = None

class JobAnalysisResponse(BaseModel):
    hiring_signal: float
    sentiment: Dict[str, float]
    confidence: float
    source: str
    company: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "hiring_signal": 0.85,
                "sentiment": {
                    "positive": 0.7,
                    "neutral": 0.2,
                    "negative": 0.1
                },
                "confidence": 0.92,
                "source": "indeed.com",
                "company": "Tech Corp"
            }
        }
    }

class SourceReliabilityResponse(BaseModel):
    source_type: str
    reliability_score: float
    total_processed: int
    error_rate: float





# Initialize FastAPI app
app = FastAPI(
    title="Hiring Signal Analyzer",
    description="Analyze job postings for hiring signals and market trends",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")





@app.post("/analyze", response_model=JobAnalysisResponse)
async def analyze_job_posting(request: JobAnalysisRequest):
    """Analyze a job posting for hiring signals."""
    try:
        start_time = time.time()
        
        # Analyze the job posting
        result = await job_analyzer.analyze_job_post(request.text)
        
        # Record metrics
        metrics.record_hiring_signal(
            strength=result['hiring_signal'],
            source=request.source,
            company=request.company or 'unknown'
        )
        metrics.record_processing_time(time.time() - start_time)
        
        return JobAnalysisResponse(
            hiring_signal=result['hiring_signal'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            source=request.source,
            company=request.company or 'unknown'
        )
        
    except Exception as e:
        metrics.record_error(request.source, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/{source_type}", response_model=SourceReliabilityResponse)
async def get_source_reliability(source_type: str):
    """Get reliability metrics for a data source."""
    try:
        # Get metrics for the source
        metrics_summary = metrics.get_metrics_summary()
        source_metrics = metrics_summary['metrics']['source_reliability'].get(source_type, {})
        
        return SourceReliabilityResponse(
            source_type=source_type,
            reliability_score=source_metrics.get('reliability', 0.0),
            total_processed=source_metrics.get('total_processed', 0),
            error_rate=source_metrics.get('error_rate', 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check health of all components."""
    try:
        # Check if analyzer is loaded
        if not job_analyzer.model or not job_analyzer.tokenizer:
            raise Exception("Job analyzer not properly initialized")
            
        # Test data pipeline
        await data_pipeline.init_session()
        
        return {"status": "healthy"}
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get all current metrics."""
    try:
        return metrics.get_metrics_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HiringSentinelPipeline:
    def __init__(self):
        logger.info("Initializing Hiring Sentinel Pipeline")
        # We'll add these back once the basic API is working
        # self.ingestion = DataIngestionPipeline()
        # self.metrics = MetricsTracker()
        # self.processors = {
        #     'newsapi': NewsAPIProcessor(),
        #     'linkedin': LinkedInProcessor(),
        #     'gdelt': GDELTProcessor()
        # }
    
    async def start(self):
        """Start the pipeline with all processors."""
        try:
            logger.info("Starting Hiring Sentinel Pipeline...")
            # We'll add message processing back once the basic API is working
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            raise
    
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

@app.get("/")
async def health_check():
    REQUEST_COUNT.labels('GET', '/', '200').inc()
    return {"status": "healthy", "service": "hiring-sentiment-tracker"}

@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(metrics_manager.registry),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of a single text and extract business intelligence."""
    start_time = time.time()
    request_id = hashlib.md5(f"{time.time()}-{request.text[:50]}".encode()).hexdigest()[:8]
    
    try:
        logger.info(f"[{request_id}] Starting sentiment analysis for text of length: {len(request.text)}")
        
        # Technical Performance Monitoring
        import psutil
        process = psutil.Process()
        
        # Cache Operations
        text_hash = hashlib.md5(request.text.encode()).hexdigest()
        cache_key = f"sentiment:{text_hash}"
        logger.debug(f"[{request_id}] Generated cache key: {cache_key}")
        
        cached_result = await cache.get(cache_key)
        logger.info(f"[{request_id}] Cache {'hit' if cached_result else 'miss'}")
        
        if cached_result:
            logger.debug(f"[{request_id}] Cached result: {cached_result}")
            try:
                if not isinstance(cached_result, dict) or 'compound' not in cached_result:
                    raise ValueError(f"Invalid cached result structure: {cached_result}")
                
                metrics_manager.update_cache_metrics(hit=True)
                return await process_analysis_result(cached_result, request.text, request_id, True)
                
            except Exception as cache_error:
                logger.error(f"[{request_id}] Cache processing error: {str(cache_error)}", exc_info=True)
                metrics_manager.error_rate.labels(error_type='cache').inc()
        else:
            # Record cache miss once
            metrics_manager.update_cache_metrics(hit=False)
        
        # Fresh Analysis with Performance Monitoring
        logger.info(f"[{request_id}] Starting fresh analysis")
        
        inference_start = time.time()
        with metrics_manager.model_inference_time.time():
            result = await analyzer.analyze_text(request.text)
        result['inference_time'] = time.time() - inference_start
        
        # Validate and Cache Result
        if not isinstance(result, dict) or 'compound' not in result:
            raise ValueError(f"Invalid analysis result structure: {result}")
        
        await cache.set(cache_key, result)
        
        # Process Result and Update Business Intelligence
        response = await process_analysis_result(result, request.text, request_id, False)
        
        # Performance Metrics
        duration = time.time() - start_time
        metrics_manager.response_time.observe(duration)
        logger.info(f"[{request_id}] Analysis completed in {duration:.2f}s")
        
        # Update error rate metric (success case)
        metrics_manager.error_rate.labels(error_type='analysis').set(0)
        
        return response
        
    except Exception as e:
        metrics_manager.error_rate.labels(error_type='analysis').inc()
        logger.error(f"[{request_id}] Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cache metrics are already updated in the success paths
        pass

async def process_analysis_result(result: Dict, text: str, request_id: str, cached: bool) -> SentimentResponse:
    """Process analysis results and update business intelligence metrics."""
    try:
        # Extract Business Intelligence
        department = extract_department(text)
        seniority = extract_seniority(text)
        skills = extract_skills(text)
        sentiment_score = result['compound']
        
        # Update Department Intelligence
        if department:
            timeframe = 'current'  # You might want to make this configurable
            metrics_manager.sentiment_by_department.labels(
                department=department,
                timeframe=timeframe
            ).set(sentiment_score)
            
            # Market Intelligence
            market_score = analyze_market_competitiveness(text)
            metrics_manager.market_competitiveness.labels(
                department=department,
                region='global'
            ).set(market_score)
        
        # Skill Trends Analysis
        for skill in skills:
            metrics_manager.skill_demand_trend.labels(
                skill=skill,
                timeframe='current'
            ).set(sentiment_score)
        
        # Model Performance Tracking
        confidence = result.get('confidence', 0.8)
        metrics_manager.prediction_accuracy.labels(
            prediction_type='sentiment'
        ).set(confidence)
        
        # Record confidence distribution
        metrics_manager.confidence_distribution.labels(
            analysis_type='sentiment'
        ).observe(confidence)
        
        # Update response time if not cached
        if not cached and 'inference_time' in result:
            metrics_manager.response_time.observe(result['inference_time'])
        
        # Update error rate metric
        metrics_manager.error_rate.labels(error_type='analysis').set(0)
        
        logger.info(f"[{request_id}] Updated business intelligence metrics")
        return SentimentResponse(sentiment=result, cached=cached)
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to process analysis result: {str(e)}", exc_info=True)
        metrics_manager.error_rate.labels(error_type='processing').inc()
        raise HTTPException(status_code=500, detail=f"Failed to process analysis result: {str(e)}")

@app.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch(request: BatchSentimentRequest):
    """Analyze sentiment of multiple texts."""
    try:
        # Generate cache keys
        text_hashes = [hashlib.md5(text.encode()).hexdigest() for text in request.texts]
        cache_keys = [f"sentiment:{text_hash}" for text_hash in text_hashes]
        
        # Check cache for all texts
        cached_results = await cache.get_many(cache_keys)
        
        # Find texts that need analysis
        uncached_indices = []
        results = [None] * len(request.texts)
        
        for i, (text, cached) in enumerate(zip(request.texts, cached_results)):
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)
        
        # Analyze uncached texts
        if uncached_indices:
            uncached_texts = [request.texts[i] for i in uncached_indices]
            with SENTIMENT_LATENCY.labels('batch').time():
                uncached_results = await analyzer.analyze_batch(uncached_texts)
            
            # Update results and cache
            for idx, result in zip(uncached_indices, uncached_results):
                results[idx] = result
                await cache.set(cache_keys[idx], result)
        
        REQUEST_COUNT.labels('POST', '/analyze/batch', '200').inc()
        return BatchSentimentResponse(
            sentiments=results,
            cached=len(uncached_indices) == 0
        )
        
    except Exception as e:
        REQUEST_COUNT.labels('POST', '/analyze/batch', '500').inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
