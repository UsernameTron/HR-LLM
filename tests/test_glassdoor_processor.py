"""Tests for the Glassdoor data processor."""
import pytest
import pandas as pd
from pathlib import Path
from src.data.processors.glassdoor_processor import GlassdoorProcessor

@pytest.fixture
def processor():
    """Create a GlassdoorProcessor instance."""
    return GlassdoorProcessor()

def test_data_loading(processor):
    """Test that all data sources are loaded correctly."""
    # Check main CSV files
    assert processor.jobs_df is not None
    assert processor.reviews_df is not None
    assert processor.company_df is not None
    
    # Check industry Excel files
    assert processor.industry_data is not None
    assert len(processor.industry_data) > 0
    
    # Check specific industries
    expected_industries = {
        'companies_overview',
        'accounting',
        'automotive',
        'computer_software',
        'construction',
        'education_management',
        'financial_services',
        'higher_ed',
        'hospital_and_health_care',
        'it_services',
        'retail'
    }
    
    loaded_industries = set(processor.industry_data.keys())
    assert expected_industries.issubset(loaded_industries)
    
    # Check that each industry has multiple sheets
    for industry, data in processor.industry_data.items():
        assert isinstance(data, dict), f"Industry {industry} data should be a dict of sheets"
        assert len(data) > 0, f"Industry {industry} should have at least one sheet"

def test_clean_salary(processor):
    """Test salary cleaning functionality."""
    test_cases = [
        ('$100,000', 100000.0),
        ('100K', 100000.0),  # K suffix should multiply by 1000
        ('invalid', 0.0),
        (None, 0.0),
        ('75,000-125,000', 75000.0),  # Takes first number
        ('$80.5K', 80500.0),  # K suffix with decimal
        ('₹3,25,236', 325236.0),  # Indian currency format
        ('€50k', 50000.0),  # Euro with K
        ('100k-150k', 100000.0),  # Range with K
        ('', 0.0),  # Empty string
        (0, 0.0),  # Zero
        ('N/A', 0.0)  # Invalid string
    ]
    
    for input_str, expected in test_cases:
        result = processor._clean_salary(input_str)
        assert result == expected, f"Failed for {input_str}: got {result}, expected {expected}"

def test_company_profile_generation(processor):
    """Test company profile generation with the enhanced dataset."""
    # Test with exact and case-insensitive matches
    sample_company = processor.jobs_df['company'].iloc[0]
    
    # Test exact match
    profile = processor.get_company_profile(sample_company)
    assert profile is not None
    assert profile.name == sample_company
    
    # Test case-insensitive match
    profile_lower = processor.get_company_profile(sample_company.lower())
    assert profile_lower is not None
    assert profile_lower.name == sample_company
    
    # Check profile attributes and types
    assert isinstance(profile.growth_score, float) and 0 <= profile.growth_score <= 1
    assert isinstance(profile.sentiment_score, float) and 0 <= profile.sentiment_score <= 1
    assert isinstance(profile.recent_job_count, int) and profile.recent_job_count >= 0
    assert isinstance(profile.avg_rating, float) and (1 <= profile.avg_rating <= 5 or profile.avg_rating == 0)
    assert isinstance(profile.review_count, int) and profile.review_count >= 0
    assert isinstance(profile.hiring_probability, float) and 0 <= profile.hiring_probability <= 1
    assert isinstance(profile.salary_percentile, float)
    
    # Test with non-existent company
    assert processor.get_company_profile('NonExistentCompany123') is None

def test_industry_benchmarks(processor):
    """Test industry benchmark generation."""
    benchmarks = processor.get_industry_benchmarks()
    
    assert benchmarks is not None
    assert len(benchmarks) > 0
    
    # Check benchmark metrics for each industry
    for industry, metrics in benchmarks.items():
        # Check that industry name is valid
        assert industry and not pd.isna(industry)
        
        # Check required metrics exist
        required_metrics = {
            'avg_rating', 'avg_salary', 'company_count', 'job_count',
            'avg_growth_score', 'avg_sentiment_score'
        }
        assert set(metrics.keys()) == required_metrics
        
        # Validate metric types and ranges
        assert isinstance(metrics['avg_rating'], float)
        assert isinstance(metrics['avg_salary'], float)
        assert isinstance(metrics['company_count'], int)
        assert isinstance(metrics['job_count'], int)
        assert isinstance(metrics['avg_growth_score'], float)
        assert isinstance(metrics['avg_sentiment_score'], float)
        
        # Validate metric ranges
        assert 0 <= metrics['avg_growth_score'] <= 1
        assert 0 <= metrics['avg_sentiment_score'] <= 1
        assert metrics['avg_rating'] >= 0
        assert metrics['avg_salary'] >= 0
        assert metrics['company_count'] >= 0
        assert metrics['job_count'] >= 0

def test_hiring_trends(processor):
    """Test hiring trends analysis."""
    # Get a sample company
    sample_company = processor.jobs_df['company'].iloc[0]
    
    # Test with different time periods
    for days in [30, 90, 180, 365]:
        trends = processor.get_hiring_trends(sample_company, days=days)
        
        assert trends is not None
        assert isinstance(trends, dict)
        
        # Check required fields
        required_fields = {'dates', 'job_counts', 'sentiment_scores', 'growth_scores'}
        assert set(trends.keys()) == required_fields
        
        # Check data types and lengths
        assert all(isinstance(d, str) for d in trends['dates'])
        assert all(isinstance(c, int) for c in trends['job_counts'])
        assert all(isinstance(s, (float, type(None))) for s in trends['sentiment_scores'])
        assert all(isinstance(g, float) for g in trends['growth_scores'])
        
        # Check list lengths match
        list_length = len(trends['dates'])
        assert len(trends['job_counts']) == list_length
        assert len(trends['sentiment_scores']) == list_length
        assert len(trends['growth_scores']) == list_length
        
        # Validate score ranges
        assert all(0 <= score <= 1 for score in trends['growth_scores'])
        assert all(1 <= score <= 5 for score in trends['sentiment_scores'] if score is not None)
        assert all(count >= 0 for count in trends['job_counts'])
    
    # Test with non-existent company
    empty_trends = processor.get_hiring_trends('NonExistentCompany123')
    assert empty_trends == {}
    assert len(trends['dates']) == len(trends['growth_scores'])
