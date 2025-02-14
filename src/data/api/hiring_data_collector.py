import os
from typing import List, Dict, Any
import pandas as pd
import logging
from rich.logging import RichHandler
from .perplexity_client import PerplexityClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class HiringDataCollector:
    def __init__(self, api_key: str):
        self.client = PerplexityClient(api_key)
        self.domains = [
            "linkedin.com",
            "glassdoor.com",
            "sec.gov"
        ]
        
    def collect_company_data(self, company_name: str) -> Dict[str, Any]:
        """
        Collect hiring-related data for a specific company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dict containing collected data
        """
        queries = [
            f"{company_name} recent hiring announcements 2024",
            f"{company_name} layoffs workforce changes 2024",
            f"{company_name} job market expansion plans",
            f"{company_name} quarterly earnings hiring discussion"
        ]
        
        results = self.client.batch_search(queries, self.domains)
        return {
            "company": company_name,
            "hiring_data": results
        }
        
    def collect_industry_data(self, industry: str) -> Dict[str, Any]:
        """
        Collect hiring trends for a specific industry.
        
        Args:
            industry: Industry name
            
        Returns:
            Dict containing industry trends
        """
        queries = [
            f"{industry} industry hiring trends 2024",
            f"{industry} sector job market outlook",
            f"{industry} workforce expansion patterns",
            f"{industry} employment growth forecast"
        ]
        
        results = self.client.batch_search(queries, self.domains)
        return {
            "industry": industry,
            "trend_data": results
        }
        
    def save_to_parquet(self, data: List[Dict[str, Any]], filename: str):
        """
        Save collected data to parquet format.
        
        Args:
            data: List of collected data
            filename: Output filename
        """
        df = pd.DataFrame(data)
        output_path = os.path.join("data", "processed", filename)
        df.to_parquet(output_path)
        logger.info(f"Saved collected data to {output_path}")

def main():
    # Initialize collector with API key
    api_key = "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"
    collector = HiringDataCollector(api_key)
    
    # Example usage
    industries = [
        "Technology",
        "Healthcare",
        "Finance",
        "Manufacturing",
        "Retail"
    ]
    
    industry_data = []
    for industry in industries:
        data = collector.collect_industry_data(industry)
        industry_data.append(data)
    
    collector.save_to_parquet(industry_data, "industry_hiring_trends.parquet")

if __name__ == "__main__":
    main()
