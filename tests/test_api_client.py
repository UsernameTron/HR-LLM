"""
Tests for the unified API client
"""
"""Tests for the unified API client"""
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from src.clients.api_client import APIClient, get_api_client

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Setup mock environment variables"""
    env_vars = {
        'OPENAI_API_KEY': 'sk-proj-Lgm74GAr-z6xmprUTkY4HkSeLGAaP',
        'GEMINI_API_KEY': 'AIzaSyCnc7azEL3mS31VD7Nk3Ab8No5HsEenBLE',
        'GOOGLE_SENTIMENT_API_KEY': 'AIzaSyBF81zdYjqXCojuvDNYnHx2nD8yKhjL3XI',
        'PERPLEXITY_API_KEY': 'mock-perplexity-key'
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars

@pytest.fixture
def mock_clients():
    """Mock external API clients"""
    with patch('google.generativeai.GenerativeModel'), \
         patch('google.cloud.language.LanguageServiceClient'), \
         patch('openai.OpenAI'), \
         patch('httpx.Client'):
        yield

@pytest.fixture
def api_client(mock_env_vars, mock_clients):
    """Create API client instance"""
    return APIClient()

@pytest.mark.asyncio
async def test_api_client_initialization(api_client):
    """Test API client initialization"""
    assert api_client.openai_key == 'sk-proj-Lgm74GAr-z6xmprUTkY4HkSeLGAaP'
    assert api_client.gemini_key == 'AIzaSyCnc7azEL3mS31VD7Nk3Ab8No5HsEenBLE'
    assert api_client.google_sentiment_key == 'AIzaSyBF81zdYjqXCojuvDNYnHx2nD8yKhjL3XI'
    assert api_client.perplexity_key == 'mock-perplexity-key'

@pytest.mark.asyncio
async def test_analyze_hiring_signals(api_client):
    """Test the main hiring signals analysis function"""
    # Mock async methods
    api_client._get_perplexity_signals = AsyncMock(return_value=["Signal 1", "Signal 2"])
    api_client._analyze_sentiment = AsyncMock(return_value={"score": 0.8, "magnitude": 0.9})
    api_client._get_gpt4_insights = AsyncMock(return_value="Detailed insights about hiring")
    api_client._validate_with_gemini = AsyncMock(return_value={"confidence": 0.85})
    
    # Test analysis
    result = await api_client.analyze_hiring_signals("TestCompany")
    
    # Verify result structure
    assert "company" in result
    assert "signals" in result
    assert "sentiment" in result
    assert "insights" in result
    assert "confidence" in result
    
    # Verify values
    assert result["company"] == "TestCompany"
    assert len(result["signals"]) == 2
    assert result["sentiment"]["score"] == 0.8
    assert result["insights"] == "Detailed insights about hiring"
    assert result["confidence"] == 0.85

@pytest.mark.asyncio
async def test_api_client_singleton(mock_env_vars, mock_clients):
    """Test that get_api_client returns a singleton instance"""
    client1 = get_api_client()
    client2 = get_api_client()
    assert client1 is client2

@pytest.mark.asyncio
async def test_error_handling(api_client):
    """Test error handling in the analysis pipeline"""
    api_client._get_perplexity_signals = AsyncMock(side_effect=Exception("API Error"))
    
    result = await api_client.analyze_hiring_signals("TestCompany")
    
    # Verify default values on error
    assert result["company"] == "TestCompany"
    assert result["signals"] == []
    assert result["sentiment"] is None
    assert result["insights"] is None
    assert result["confidence"] == 0.0
