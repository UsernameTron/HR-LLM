import os
from typing import List, Dict, Any
import logging
from openai import OpenAI
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class PerplexityClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
    def search_hiring_data(self, query: str, domains: List[str] = None, recency: str = "month") -> Dict[str, Any]:
        """
        Search for hiring-related data using Perplexity API.
        
        Args:
            query: Search query
            domains: Optional list of domains to filter results
            recency: Time filter ('month', 'week', 'day', 'hour')
            
        Returns:
            Dict containing search results and citations
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a hiring and employment data analyst. "
                        "Focus on extracting specific, factual information about "
                        "hiring trends, layoffs, and workforce changes. "
                        "Provide concrete numbers and dates when available."
                    )
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            response = self.client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                temperature=0.2,
                search_domain_filter=domains,
                search_recency_filter=recency
            )
            
            result = {
                "content": response.choices[0].message.content,
                "citations": response.citations if hasattr(response, 'citations') else []
            }
            
            logger.info(f"Successfully retrieved data for query: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Error searching Perplexity API: {str(e)}")
            return {"content": "", "citations": []}
            
    def batch_search(self, queries: List[str], domains: List[str] = None) -> List[Dict[str, Any]]:
        """
        Perform multiple searches in batch.
        
        Args:
            queries: List of search queries
            domains: Optional list of domains to filter results
            
        Returns:
            List of search results
        """
        results = []
        for query in queries:
            result = self.search_hiring_data(query, domains)
            results.append(result)
        return results
