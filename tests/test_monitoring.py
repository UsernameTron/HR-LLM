import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock, AsyncMock
from prometheus_client import CollectorRegistry, Counter

from src.main import app, process_analysis_result
from src.cache.redis_cache import RedisCache
from src.models.sentiment import SentimentAnalyzer
from src.metrics import MetricsManager

@pytest.fixture(scope="function")
def metrics_registry():
    """Provide a clean registry for each test"""
    registry = CollectorRegistry()
    # Initialize the registry with a zero value for each counter
    Counter('hiring_cache_hits_total', 'Total number of cache hits', registry=registry).inc(0)
    Counter('hiring_cache_misses_total', 'Total number of cache misses', registry=registry).inc(0)
    yield registry
    # Clear registry after each test
    for collector in list(registry._collector_to_names.keys()):
        registry.unregister(collector)

@pytest.fixture(scope="function")
def metrics_manager(metrics_registry):
    """Initialize metrics manager with test registry"""
    return MetricsManager(registry=metrics_registry)

@pytest.fixture
def test_client(metrics_manager):
    """Configure test client with metrics manager"""
    with patch('src.main.metrics_manager', metrics_manager):
        return TestClient(app)

@pytest.fixture
def mock_redis():
    with patch('src.main.cache') as mock:
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        yield mock

@pytest.fixture
def mock_analyzer():
    with patch('src.main.analyzer') as mock:
        mock.analyze_text = AsyncMock(return_value={
            'compound': 0.8,
            'pos': 0.6,
            'neg': 0.1,
            'neu': 0.3,
            'confidence': 0.9,
            'inference_time': 0.05
        })
        yield mock

@pytest.fixture
def sample_job_posting():
    return {
        "text": """
        Senior Software Engineer - Cloud Infrastructure
        Department: Engineering
        Location: San Francisco
        
        We're seeking an experienced Software Engineer to join our Cloud Infrastructure team.
        Required skills: Python, Kubernetes, AWS
        Compensation: Competitive salary with excellent benefits
        """
    }

@pytest.mark.asyncio
async def test_metrics_initialization(metrics_manager):
    """Test that all required metrics are properly initialized"""
    assert metrics_manager.sentiment_by_department is not None
    assert metrics_manager.skill_demand_trend is not None
    assert metrics_manager.market_competitiveness is not None
    assert metrics_manager.prediction_accuracy is not None
    assert metrics_manager.response_time is not None
    assert metrics_manager.cache_hits is not None
    assert metrics_manager.cache_misses is not None
    assert metrics_manager.cache_effectiveness is not None
    assert metrics_manager.error_rate is not None

@pytest.mark.asyncio
async def test_analyze_endpoint_metrics(test_client, mock_redis, mock_analyzer, sample_job_posting, metrics_registry):
    """Test that the analyze endpoint updates all required metrics"""
    # Configure mock responses
    mock_redis.get.return_value = None
    
    # Make request
    response = test_client.post("/analyze", json=sample_job_posting)
    assert response.status_code == 200
    
    # Verify metrics were updated
    metrics = {m.name: m for m in metrics_registry.collect()}
    
    # Check department sentiment
    assert 'hiring_sentiment_by_department' in metrics
    dept_metric = metrics['hiring_sentiment_by_department']
    samples = list(dept_metric.samples)
    assert any(s.labels.get('department') == 'Engineering' for s in samples)
    
    # Check response time
    assert 'hiring_response_time_seconds' in metrics
    
    # Check department sentiment
    dept_metric = metrics.get('hiring_sentiment_by_department')
    assert dept_metric is not None
    assert any(s.labels['department'] == 'Engineering' for s in dept_metric.samples)
    
    # Check skill demand
    skill_metric = metrics.get('hiring_skill_demand_trend')
    assert skill_metric is not None
    for skill in ['Python', 'Kubernetes', 'AWS']:
        assert any(s.labels['skill'] == skill for s in skill_metric.samples)
    
    # Check performance metrics
    assert metrics.get('hiring_response_time_seconds') is not None
    assert metrics.get('hiring_cache_effectiveness_ratio') is not None

@pytest.mark.asyncio
async def test_error_handling(test_client, mock_redis, mock_analyzer, metrics_registry):
    """Test error handling and error rate metrics"""
    # Configure mock to raise error
    mock_analyzer.analyze_text = AsyncMock(side_effect=Exception("Test error"))
    
    # Make request that should trigger error
    response = test_client.post("/analyze", json={"text": "Test text"})
    assert response.status_code == 500
    
    # Verify error metrics were updated
    metrics = {m.name: m for m in metrics_registry.collect()}
    assert 'hiring_error_rate_percentage' in metrics
    error_metric = metrics['hiring_error_rate_percentage']
    samples = list(error_metric.samples)
    assert any(s.labels.get('error_type') == 'analysis' for s in samples)
    assert any(s.labels['error_type'] == 'analysis' for s in error_metric.samples)

@pytest.mark.asyncio
async def test_cache_metrics(test_client, mock_redis, mock_analyzer, sample_job_posting, metrics_registry, metrics_manager):
    """Test cache effectiveness metrics"""
    # Debug: Print all registered metrics before starting
    metrics = {m.name: m for m in metrics_registry.collect()}
    print("\nInitial registered metrics:", list(metrics.keys()))
    
    # Test cache miss
    mock_redis.get.return_value = None
    
    response = test_client.post("/analyze", json=sample_job_posting)
    assert response.status_code == 200
    
    # Verify miss was recorded
    snapshot = metrics_manager.get_cache_metrics_snapshot()
    assert snapshot['misses'] == 1, f"Cache miss not recorded correctly. Metrics: {snapshot}"
    assert snapshot['hits'] == 0, f"Unexpected cache hits recorded. Metrics: {snapshot}"
    
    # Test cache hit
    mock_redis.get.return_value = {
        'compound': 0.5,
        'pos': 0.6,
        'neg': 0.1,
        'neu': 0.3
    }
    
    response = test_client.post("/analyze", json=sample_job_posting)
    assert response.status_code == 200
    
    # Verify first hit was recorded
    snapshot = metrics_manager.get_cache_metrics_snapshot()
    assert snapshot['hits'] == 1, f"First cache hit not recorded correctly. Metrics: {snapshot}"
    assert snapshot['misses'] == 1, f"Previous cache miss count changed. Metrics: {snapshot}"
    
    # Test second cache hit
    mock_redis.get.return_value = {'compound': 0.5}
    response = test_client.post("/analyze", json=sample_job_posting)
    assert response.status_code == 200
    
    # Verify second hit was recorded
    snapshot = metrics_manager.get_cache_metrics_snapshot()
    assert snapshot['hits'] == 2, f"Second cache hit not recorded correctly. Metrics: {snapshot}"
    assert snapshot['misses'] == 1, f"Cache miss count unexpectedly changed. Metrics: {snapshot}"
    assert snapshot['total_requests'] == 3, f"Total requests count incorrect. Metrics: {snapshot}"
    
    # Now verify effectiveness ratio
    assert snapshot['effectiveness'] > 0, f"Cache effectiveness should be > 0. Metrics: {snapshot}"
    assert abs(snapshot['effectiveness'] - 66.67) < 1, f"Cache effectiveness should be ~66.67%. Metrics: {snapshot}"
    
    # Verify internal counters match Prometheus metrics
    assert snapshot['internal_hits'] == snapshot['hits'], f"Internal hits counter mismatch. Metrics: {snapshot}"
    assert snapshot['internal_misses'] == snapshot['misses'], f"Internal misses counter mismatch. Metrics: {snapshot}"

@pytest.mark.asyncio
async def test_business_intelligence_metrics(test_client, mock_redis, mock_analyzer, sample_job_posting, metrics_registry):
    """Test business intelligence metrics are properly updated"""
    mock_analyzer.analyze_text.return_value = {
        'compound': 0.8,
        'confidence': 0.9
    }
    mock_redis.get.return_value = None
    
    response = test_client.post("/analyze", json=sample_job_posting)
    assert response.status_code == 200
    
    metrics = {m.name: m for m in metrics_registry.collect()}
    
    # Check market competitiveness
    market_metric = metrics.get('hiring_market_competitiveness')
    assert market_metric is not None
    assert any(s.labels['department'] == 'Engineering' for s in market_metric.samples)
    
    # Check hiring wave indicator
    wave_metric = metrics.get('hiring_wave_indicator')
    assert wave_metric is not None
    assert any(s.labels['department'] == 'Engineering' for s in wave_metric.samples)
    
    # Check skill premium
    premium_metric = metrics.get('hiring_skill_premium')
    assert premium_metric is not None
    for skill in ['Python', 'Kubernetes', 'AWS']:
        assert any(s.labels['skill'] == skill for s in premium_metric.samples)

@pytest.mark.asyncio
async def test_model_performance_metrics(test_client, mock_redis, mock_analyzer, metrics_registry):
    """Test model performance metrics"""
    mock_analyzer.analyze_text.return_value = {
        'compound': 0.8,
        'confidence': 0.95
    }
    mock_redis.get.return_value = None
    
    response = test_client.post("/analyze", json={"text": "Test text"})
    assert response.status_code == 200
    
    metrics = {m.name: m for m in metrics_registry.collect()}
    
    # Check prediction accuracy
    accuracy_metric = metrics.get('hiring_prediction_accuracy')
    assert accuracy_metric is not None
    assert any(s.labels['prediction_type'] == 'sentiment' for s in accuracy_metric.samples)
    
    # Check confidence distribution
    conf_metric = metrics.get('hiring_confidence_distribution')
    assert conf_metric is not None
    assert any(s.labels['analysis_type'] == 'sentiment' for s in conf_metric.samples)
