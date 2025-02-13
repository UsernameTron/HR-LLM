"""
Configuration settings for the Hiring Sentiment Tracker.
Optimized for Apple M4 Pro hardware with MPS acceleration.
"""
from pathlib import Path
from typing import Dict, Optional

import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Hardware Optimization
    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
    BATCH_SIZE: int = 32  # Optimized for 48GB unified memory
    NUM_WORKERS: int = 8  # Matched to performance cores
    
    # Cache Configuration
    CACHE_DIR: Path = Path("cache")
    EMBEDDING_CACHE_SIZE: int = 10000  # Number of cached embeddings
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Configuration
    API_RATE_LIMIT: int = 100  # Requests per minute
    API_TIMEOUT: int = 30  # Seconds
    
    # Model Configuration
    MODEL_NAME: str = "microsoft/deberta-v3-large"  # Base model for fine-tuning
    MAX_SEQ_LENGTH: int = 512
    LEARNING_RATE: float = 2e-5
    
    # Data Processing
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC: str = "hiring-signals"
    
    # Monitoring
    DRIFT_THRESHOLD: float = 0.1
    CONFIDENCE_THRESHOLD: float = 0.8
    
    class Config:
        env_file = ".env"

settings = Settings()

# MPS Specific Optimizations
MPS_CONFIG = {
    "enable_graph_mode": True,
    "enable_async": True,
    "benchmark_mode": True,
}

# API Provider Configurations
API_CONFIGS: Dict[str, Dict] = {
    "newsapi": {
        "base_url": "https://newsapi.org/v2",
        "cache_ttl": 3600,  # 1 hour
    },
    "linkedin": {
        "base_url": "https://api.linkedin.com/v2",
        "cache_ttl": 7200,  # 2 hours
    },
    "gdelt": {
        "base_url": "https://api.gdeltproject.org/api/v2",
        "cache_ttl": 1800,  # 30 minutes
    }
}
