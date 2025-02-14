"""
Kaggle dataset downloader for SEC filings and sentiment analysis datasets.
"""
import logging
import os
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

class KaggleDownloader:
    """Helper class to search and download relevant Kaggle datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Base directory for downloading datasets
        """
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Set up data directories
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized KaggleDownloader with data directory: {self.data_dir}")
        
    def search_datasets(
        self,
        query: str,
        max_size_mb: int = 1000,
        file_types: Optional[List[str]] = None
    ) -> List[dict]:
        """
        Search for relevant datasets on Kaggle.
        
        Args:
            query: Search query
            max_size_mb: Maximum dataset size in MB
            file_types: List of acceptable file extensions (e.g., ['csv', 'json'])
            
        Returns:
            List of dataset metadata
        """
        logger.info(f"Searching for datasets matching query: {query}")
        
        datasets = self.api.dataset_list(search=query)
        filtered_datasets = []
        
        for dataset in datasets:
            # Skip datasets that are too large
            try:
                size_bytes = int(dataset.size)
                if size_bytes > max_size_mb * 1024 * 1024:
                    continue
            except (ValueError, TypeError):
                # If size can't be determined, include the dataset
                pass
                
            # Skip datasets with wrong file types if specified
            if file_types:
                files = self.api.dataset_list_files(dataset.ref).files
                if not any(f.name.lower().endswith(tuple(f".{ft.lower()}" for ft in file_types)) for f in files):
                    continue
                    
            try:
                size_mb = round(float(dataset.size) / (1024 * 1024), 2)
            except (ValueError, TypeError):
                size_mb = 0
                
            filtered_datasets.append({
                "ref": dataset.ref,
                "title": dataset.title,
                "size_mb": size_mb,
                "last_updated": dataset.lastUpdated,
                "download_count": dataset.downloadCount
            })
            
        logger.info(f"Found {len(filtered_datasets)} matching datasets")
        return filtered_datasets
        
    def download_dataset(self, dataset_ref: str, extract: bool = True) -> Path:
        """
        Download a specific dataset.
        
        Args:
            dataset_ref: Dataset reference (e.g., 'username/dataset-name')
            extract: Whether to extract the downloaded files
            
        Returns:
            Path to the downloaded dataset
        """
        logger.info(f"Downloading dataset: {dataset_ref}")
        
        # Create dataset directory
        dataset_dir = self.raw_dir / dataset_ref.split("/")[1]
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            self.api.dataset_download_files(
                dataset_ref,
                path=str(dataset_dir),
                unzip=extract
            )
            logger.info(f"Successfully downloaded dataset to {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_ref}: {str(e)}")
            raise

def main():
    """Main function to download relevant datasets."""
    downloader = KaggleDownloader()
    
    # Search queries for different types of datasets
    searches = [
        {
            "query": "sec filings sentiment analysis",
            "file_types": ["csv", "json", "txt"]
        },
        {
            "query": "financial sentiment analysis labeled",
            "file_types": ["csv", "json", "txt"]
        },
        {
            "query": "company hiring announcements",
            "file_types": ["csv", "json", "txt"]
        }
    ]
    
    for search in searches:
        logger.info(f"\nSearching for {search['query']}...")
        datasets = downloader.search_datasets(
            query=search["query"],
            file_types=search["file_types"]
        )
        
        # Sort by download count as a proxy for quality
        datasets.sort(key=lambda x: x["download_count"], reverse=True)
        
        # Download top datasets
        for dataset in datasets[:2]:  # Download top 2 from each search
            try:
                downloader.download_dataset(dataset["ref"])
                logger.info(f"Successfully downloaded {dataset['ref']}")
            except Exception as e:
                logger.error(f"Failed to download {dataset['ref']}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
