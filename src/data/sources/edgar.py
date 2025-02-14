"""SEC EDGAR integration for fetching and analyzing company filings.
Optimized for hiring sentiment analysis.
"""
import logging
from typing import Optional, Dict, List
import aiohttp
import asyncio
from datetime import datetime
from functools import wraps

from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries=3, delay=1):
    """Decorator for retrying failed API calls"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

class EDGARIntegration:
    """SEC EDGAR integration for fetching company filings."""
    
    def __init__(self):
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            "User-Agent": "HiringSentimentTracker 1.0 hiring.sentiment@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
            "Accept": "application/json, text/html"
        }
        self.session = None
        self.metrics = MetricsTracker()
        
    async def init_session(self):
        """Initialize aiohttp session with proper headers"""
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    @retry_on_failure()
    async def _fetch_json(self, url: str) -> Optional[dict]:
        """Fetch JSON data from URL with retry logic"""
        await self.init_session()
        await asyncio.sleep(0.15)  # Rate limiting - SEC requires max 10 requests per second
        
        async with self.session.get(url) as response:
            if response.status != 200:
                self.metrics.record_failure('edgar_fetch', url, f"Status: {response.status}")
                logger.error(f"Failed to fetch data: {response.status}")
                return None
            
            try:
                data = await response.json()
                self.metrics.record_success('edgar_fetch', url)
                return data
            except Exception as e:
                self.metrics.record_failure('edgar_fetch', url, f"Parse error: {str(e)}")
                logger.error(f"Failed to parse JSON: {str(e)}")
                return None
    
    @retry_on_failure()
    async def _fetch_text(self, url: str) -> Optional[str]:
        """Fetch text data from URL with retry logic"""
        await self.init_session()
        await asyncio.sleep(0.15)  # Rate limiting - SEC requires max 10 requests per second
        
        async with self.session.get(url) as response:
            if response.status != 200:
                self.metrics.record_failure('edgar_fetch', url, f"Status: {response.status}")
                logger.error(f"Failed to fetch text: {response.status}")
                return None
            
            try:
                text = await response.text()
                self.metrics.record_success('edgar_fetch', url)
                return text
            except Exception as e:
                self.metrics.record_failure('edgar_fetch', url, f"Parse error: {str(e)}")
                logger.error(f"Failed to parse text: {str(e)}")
                return None
            
    async def fetch_10k(self, cik: str, year: int) -> Optional[str]:
        """
        Fetch 10-K filing for a given company and year.
        
        Args:
            cik: Company CIK number
            year: Filing year
            
        Returns:
            Filing text if successful, None otherwise
        """
        try:
            # SEC requires 10 digit CIK
            padded_cik = str(cik).zfill(10)
            
            # First get the submission file
            submission_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
            data = await self._fetch_json(submission_url)
            
            if not data:
                return None
                
            # Find the 10-K filing for the requested year
            for filing in data.get('filings', {}).get('recent', []):
                if filing['form'] == '10-K' and filing['filingDate'].startswith(str(year)):
                    accession_number = filing['accessionNumber']
                    filing_url = f"{self.base_url}/{padded_cik}/{accession_number}"
                    return await self._fetch_text(filing_url)
                    
            logger.error(f"No 10-K found for {cik} in {year}")
            return None
                
        except Exception as e:
            self.metrics.record_failure('edgar_fetch', 'unknown', str(e))
            logger.error(f"Error fetching 10-K: {str(e)}")
            return None
            
    async def extract_hiring_sections(self, filing_text: str) -> Dict[str, str]:
        """
        Extract sections from 10-K that are relevant to hiring analysis.
        
        Args:
            filing_text: Full text of the SEC filing
            
        Returns:
            Dictionary of section name to section text
        """
        relevant_sections = {
            'employees': '',
            'human_capital': '',
            'business': '',
            'risk_factors': ''
        }
        
        try:
            # Basic section extraction - this can be enhanced with better parsing
            lower_text = filing_text.lower()
            
            # Find Employee section
            emp_start = lower_text.find('employees')
            if emp_start != -1:
                emp_end = lower_text.find('\n\n', emp_start)
                if emp_end != -1:
                    relevant_sections['employees'] = filing_text[emp_start:emp_end].strip()
                    
            # Find Human Capital section (newer 10-Ks)
            hc_start = lower_text.find('human capital')
            if hc_start != -1:
                hc_end = lower_text.find('\n\n', hc_start)
                if hc_end != -1:
                    relevant_sections['human_capital'] = filing_text[hc_start:hc_end].strip()
                    
            self.metrics.record_success('section_extraction', 'filing')
            return relevant_sections
            
        except Exception as e:
            self.metrics.record_failure('section_extraction', 'filing', str(e))
            logger.error(f"Error extracting sections: {str(e)}")
            return relevant_sections
            
    async def analyze_hiring_sentiment(self, sections: Dict[str, str]) -> float:
        """
        Analyze hiring sentiment in extracted sections.
        
        Args:
            sections: Dictionary of section name to section text
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            # Simple keyword-based analysis - can be enhanced with proper NLP
            positive_indicators = [
                'growing', 'expansion', 'hiring',
                'increased headcount', 'new positions',
                'talent acquisition', 'workforce growth'
            ]
            
            negative_indicators = [
                'layoff', 'reduction', 'restructuring',
                'decreased headcount', 'attrition',
                'workforce reduction', 'downsizing'
            ]
            
            # Combine all text
            all_text = ' '.join(sections.values()).lower()
            
            # Count indicators with weights
            positive_count = sum(2 if indicator in ['hiring', 'expansion'] else 1
                               for indicator in positive_indicators
                               if indicator in all_text)
                               
            negative_count = sum(2 if indicator in ['layoff', 'downsizing'] else 1
                                for indicator in negative_indicators
                                if indicator in all_text)
            
            # Calculate weighted sentiment score
            total = positive_count + negative_count
            if total == 0:
                return 0.0
                
            sentiment = (positive_count - negative_count) / total
            self.metrics.record_success('sentiment_analysis', 'filing')
            return max(min(sentiment, 1.0), -1.0)
            
        except Exception as e:
            self.metrics.record_failure('sentiment_analysis', 'filing', str(e))
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0
            
    async def cleanup(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
