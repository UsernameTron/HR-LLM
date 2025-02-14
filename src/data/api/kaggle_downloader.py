import os
import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class KaggleDownloader:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.raw_data_dir = Path("data/raw")
        
    def search_datasets(self, keyword: str, max_size_mb: int = 1000) -> list:
        """
        Search for relevant datasets on Kaggle.
        
        Args:
            keyword: Search term
            max_size_mb: Maximum dataset size in MB
            
        Returns:
            List of relevant dataset references
        """
        try:
            datasets = self.api.dataset_list(search=keyword)
            filtered_datasets = [
                d for d in datasets 
                if d.size / (1024 * 1024) <= max_size_mb  # Convert bytes to MB
            ]
            logger.info(f"Found {len(filtered_datasets)} datasets matching '{keyword}'")
            return filtered_datasets
        except Exception as e:
            logger.error(f"Error searching Kaggle datasets: {str(e)}")
            return []
            
    def download_dataset(self, dataset_ref: str, target_dir: str = None) -> bool:
        """
        Download a specific dataset.
        
        Args:
            dataset_ref: Dataset reference (owner/dataset-name)
            target_dir: Optional specific target directory
            
        Returns:
            bool indicating success
        """
        try:
            target = target_dir or self.raw_data_dir / dataset_ref.split('/')[-1]
            os.makedirs(target, exist_ok=True)
            
            logger.info(f"Downloading dataset {dataset_ref}")
            self.api.dataset_download_files(
                dataset_ref,
                path=target,
                unzip=True
            )
            logger.info(f"Successfully downloaded to {target}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_ref}: {str(e)}")
            return False
            
    def download_sentiment_datasets(self):
        """Download relevant sentiment analysis datasets."""
        search_terms = [
            "sentiment analysis financial",
            "company reviews sentiment",
            "employee reviews sentiment",
            "job market sentiment",
            "hiring sentiment"
        ]
        
        for term in search_terms:
            datasets = self.search_datasets(term)
            for dataset in datasets[:3]:  # Top 3 most relevant
                self.download_dataset(dataset.ref)

def main():
    downloader = KaggleDownloader()
    downloader.download_sentiment_datasets()

if __name__ == "__main__":
    main()
