"""
Tests for SEC EDGAR integration functionality.
"""
import pytest
import asyncio
import time
from aioresponses import aioresponses
from src.data.sources.edgar import EDGARIntegration
from src.data import init_cache

from tests.mock_data import MOCK_SUBMISSION_DATA, MOCK_FILING_DATA

# Set up the event loop for all tests
@pytest.fixture(scope='session')
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
    
@pytest.fixture(autouse=True)
async def setup_cache():
    """Initialize cache before running tests"""
    await init_cache()
    
@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m

@pytest.mark.asyncio
async def test_edgar_integration(mock_aioresponse):
    """Test basic SEC EDGAR integration functionality"""
    edgar = EDGARIntegration()
    
    # Mock the submission endpoint
    submission_url = "https://data.sec.gov/submissions/CIK0000320193.json"
    mock_aioresponse.get(submission_url, payload=MOCK_SUBMISSION_DATA)
    
    # Mock the filing endpoint
    filing_url = f"{edgar.base_url}/0000320193/0000320193-23-000077"
    mock_aioresponse.get(filing_url, body=MOCK_FILING_DATA)
    
    try:
        # Test with Apple Inc.'s CIK
        filing = await edgar.fetch_10k("0000320193", 2023)
        
        # Basic validation
        assert filing is not None
        assert isinstance(filing, str)
        assert len(filing) > 0
        assert "APPLE INC" in filing
        
        # Test section extraction
        sections = await edgar.extract_hiring_sections(filing)
        assert isinstance(sections, dict)
        assert len(sections) > 0
        assert "growing its workforce" in sections['employees']
        
        # Test sentiment analysis
        sentiment = await edgar.analyze_hiring_sentiment(sections)
        assert isinstance(sentiment, float)
        assert sentiment > 0  # Should be positive due to "growing workforce"
        
    finally:
        await edgar.cleanup()
        
@pytest.mark.asyncio
async def test_edgar_error_handling(mock_aioresponse):
    """Test SEC EDGAR error handling"""
    edgar = EDGARIntegration()
    
    # Mock error responses
    mock_aioresponse.get(
        "https://data.sec.gov/submissions/CIKinvalid_cik.json",
        status=404
    )
    mock_aioresponse.get(
        "https://data.sec.gov/submissions/CIK0000320193.json",
        payload={"filings": {"recent": []}}
    )
    
    try:
        # Test with invalid CIK
        filing = await edgar.fetch_10k("invalid_cik", 2023)
        assert filing is None
        
        # Test with future year
        filing = await edgar.fetch_10k("0000320193", 2030)
        assert filing is None
        
        # Test section extraction with invalid text
        sections = await edgar.extract_hiring_sections("")
        assert isinstance(sections, dict)
        assert all(value == '' for value in sections.values())
        
        # Test sentiment analysis with empty sections
        sentiment = await edgar.analyze_hiring_sentiment({})
        assert sentiment == 0.0
        
    finally:
        await edgar.cleanup()

@pytest.mark.asyncio
async def test_rate_limiting(mock_aioresponse):
    """Test SEC EDGAR rate limiting"""
    edgar = EDGARIntegration()
    
    # Mock responses for multiple CIKs
    ciks = ["0000320193", "0001018724", "0001326801"]  # Apple, AMZN, META
    
    for cik in ciks:
        submission_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        mock_aioresponse.get(submission_url, payload=MOCK_SUBMISSION_DATA)
        
        filing_url = f"{edgar.base_url}/{cik}/0000320193-23-000077"
        mock_aioresponse.get(filing_url, body=MOCK_FILING_DATA)
    
    try:
        start_time = time.time()
        
        results = await asyncio.gather(*[
            edgar.fetch_10k(cik, 2023) for cik in ciks
        ])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should take at least 0.3 seconds due to rate limiting
        assert duration >= 0.3
        assert all(result is not None for result in results)
        assert len(results) == 3
        
    finally:
        await edgar.cleanup()
