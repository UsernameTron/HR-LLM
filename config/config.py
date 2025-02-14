"""
Configuration settings for the hiring sentiment tracker.
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    """Application settings."""
    PROJECT_NAME: str = "Hiring Sentiment Tracker"
    DEBUG: bool = True
    DATA_DIR: str = "data/raw/glassdoor"
    MODEL_DIR: str = "models/baseline"
    
    class Config:
        env_file = ".env"

# API configurations
API_CONFIGS: Dict[str, Any] = {
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "model": "sonar-pro",
        "timeout": 30,
    },
    "openai": {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.7,
        "max_tokens": 1000,
    },
    "gemini": {
        "model": "gemini-pro",
        "temperature": 0.7,
        "max_output_tokens": 1000,
    }
}

settings = Settings()
