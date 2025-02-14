"""Tests for the enhanced JobAnalyzer."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.job_analyzer import JobAnalyzer

@pytest.fixture
def analyzer():
    """Create a JobAnalyzer instance for testing."""
    analyzer = JobAnalyzer()
    
    # Add some test historical data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    companies = ['TestCo', 'OtherCo']
    industries = ['Tech', 'Finance']
    
    data = []
    for date in dates:
        for company in companies:
            for industry in industries:
                # Generate some test hiring rates with trends
                base_rate = 0.5
                time_factor = (date - dates[0]).days / 90  # 0 to 1
                company_factor = 0.2 if company == 'TestCo' else -0.1
                industry_factor = 0.1 if industry == 'Tech' else -0.2
                
                hiring_rate = base_rate + (time_factor * company_factor) + (time_factor * industry_factor)
                hiring_rate = max(0, min(1, hiring_rate))  # Clamp to 0-1
                
                data.append({
                    'date': date,
                    'company': company,
                    'industry': industry,
                    'hiring_rate': hiring_rate
                })
    
    analyzer.historical_trends = pd.DataFrame(data)
    return analyzer

@pytest.mark.asyncio
async def test_pattern_detection(analyzer):
    """Test pattern detection in job posts with fuzzy matching and new categories."""
    # Test growth and tech patterns
    tech_growth_text = "We're scaling rapidly & expanding our engineering teams. Building cutting-edge AI/ML solutions."
    tech_scores = analyzer._detect_patterns(tech_growth_text)
    assert tech_scores['growth'] > 0.3  # Should detect 'scaling' ~ 'expanding'
    assert tech_scores['tech_stack'] > 0.2  # Should detect 'AI/ML' and 'cutting-edge'
    
    # Test benefits and culture patterns
    culture_text = "Competitive compensation including equity, stock options, and comprehensive health benefits. Strong focus on work-life balance."
    culture_scores = analyzer._detect_patterns(culture_text)
    assert culture_scores['benefits'] > 0.3  # Should detect benefits mentions
    assert culture_scores['culture'] > 0.2  # Should detect culture aspects
    
    # Test urgency and stability with fuzzy matches
    mixed_text = "Time-sensitive position at well-funded Series B company. Industry-leading tech firm."
    mixed_scores = analyzer._detect_patterns(mixed_text)
    assert mixed_scores['urgency'] > 0.2  # Should detect 'time-sensitive'
    assert mixed_scores['stability'] > 0.2  # Should detect 'well-funded'
    
    # Test fuzzy matching thresholds
    fuzzy_text = "Expanding team (spelled wrong) in cloud-based systems"
    fuzzy_scores = analyzer._detect_patterns(fuzzy_text)
    assert fuzzy_scores['growth'] > 0  # Should detect 'Expanding' despite typo
    assert fuzzy_scores['tech_stack'] > 0  # Should detect 'cloud-based' ~ 'cloud native'

@pytest.mark.asyncio
async def test_trend_analysis(analyzer):
    """Test historical trend analysis."""
    company_trend, industry_trend = analyzer._analyze_historical_trend('TestCo', 'Tech')
    
    # TestCo in Tech should show positive trends based on our test data
    assert company_trend > 0.5
    assert industry_trend > 0.5
    
    # Test with unknown company/industry
    unknown_trend = analyzer._analyze_historical_trend('UnknownCo', 'UnknownIndustry')
    assert unknown_trend == (0.5, 0.5)  # Should return neutral scores

@pytest.mark.asyncio
async def test_job_post_analysis(analyzer):
    """Test complete job post analysis."""
    test_post = """
    IMMEDIATE OPENING at TestCo!
    
    We are a rapidly growing technology company looking to expand our team.
    As an established market leader, we offer stability and growth opportunities.
    Multiple positions available for immediate start.
    
    Join our profitable, fast-growing team!
    """
    
    result = await analyzer.analyze_job_post(
        text=test_post,
        company='TestCo',
        industry='Tech'
    )
    
    # Check all components are present
    assert 'hiring_signal' in result
    assert 'sentiment' in result
    assert 'confidence' in result
    assert 'patterns' in result
    assert 'trends' in result
    
    # Check value ranges
    assert 0 <= result['hiring_signal'] <= 1
    assert 0 <= result['confidence'] <= 1
    assert all(0 <= v <= 1 for v in result['sentiment'].values())
    assert all(0 <= v <= 1 for v in result['patterns'].values())
    assert all(0 <= v <= 1 for v in result['trends'].values())
    
    # This post should have some hiring signals
    assert result['hiring_signal'] > 0.35  # Some hiring signal
    assert result['patterns']['growth'] > 0.3  # Some growth pattern
    assert result['patterns']['urgency'] > 0.15  # Some urgency
    assert result['patterns']['stability'] > 0.15  # Some stability
    
    # Test without company/industry info
    basic_result = await analyzer.analyze_job_post(text=test_post)
    assert basic_result['trends']['company'] == 0.5  # Neutral when no history
    assert basic_result['trends']['industry'] == 0.5  # Neutral when no history
