"""
Extended Kaggle dataset downloader for financial sentiment analysis.
"""
import logging
from pathlib import Path
from typing import List, Optional

from kaggle.api.kaggle_api_extended import KaggleApi
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class ExtendedKaggleDownloader:
    """Enhanced helper class to search and download relevant Kaggle datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.api = KaggleApi()
        self.api.authenticate()
        
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ExtendedKaggleDownloader with data directory: {self.data_dir}")
        
    def search_financial_sentiment_datasets(self) -> None:
        """Search and download financial sentiment datasets."""
        search_queries = [
            {
                "query": "financial sentiment analysis pretrained",
                "file_types": ["csv", "json", "txt", "parquet"]
            },
            {
                "query": "stock market sentiment labeled dataset",
                "file_types": ["csv", "json", "txt", "parquet"]
            },
            {
                "query": "finance news sentiment corpus",
                "file_types": ["csv", "json", "txt", "parquet"]
            },
            {
                "query": "company earnings sentiment dataset",
                "file_types": ["csv", "json", "txt", "parquet"]
            }
        ]
        
        existing_datasets = {p.name for p in self.raw_dir.iterdir() if p.is_dir()}
        
        for search in search_queries:
            logger.info(f"\nSearching for {search['query']}...")
            datasets = self.api.dataset_list(search=search["query"])
            
            for dataset in datasets:
                try:
                    # Skip if already downloaded
                    dataset_name = dataset.ref.split("/")[1]
                    if dataset_name in existing_datasets:
                        logger.info(f"Skipping {dataset.ref} - already downloaded")
                        continue
                    
                    # Check file types
                    files = self.api.dataset_list_files(dataset.ref).files
                    if not any(f.name.lower().endswith(tuple(f".{ft.lower()}" for ft in search["file_types"])) for f in files):
                        continue
                    
                    # Download dataset
                    logger.info(f"Downloading dataset: {dataset.ref}")
                    dataset_dir = self.raw_dir / dataset_name
                    dataset_dir.mkdir(exist_ok=True)
                    
                    self.api.dataset_download_files(
                        dataset.ref,
                        path=str(dataset_dir),
                        unzip=True
                    )
                    logger.info(f"Successfully downloaded dataset to {dataset_dir}")
                    
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset.ref}: {str(e)}")
                    continue

def main():
    """Download additional financial sentiment datasets."""
    downloader = ExtendedKaggleDownloader()
    downloader.search_financial_sentiment_datasets()

if __name__ == "__main__":
    main()
