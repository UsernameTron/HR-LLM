"""
API Configuration settings for the hiring sentiment tracker.
"""
import os
from typing import Dict, Any

# API Settings
PERPLEXITY_CONFIG = {
    "base_url": "https://api.perplexity.ai",
    "model": "sonar-pro",
    "temperature": 0.2,
    "max_retries": 3,
    "timeout": 30,
    "batch_size": 5
}

# Rate Limiting
RATE_LIMITS = {
    "perplexity": {
        "requests_per_minute": 50,
        "burst_limit": 10,
        "cooldown_period": 60
    }
}

# Cache Settings
CACHE_CONFIG = {
    "ttl": 3600,  # 1 hour
    "max_size": 1000,
    "refresh_pattern": "adaptive"
}

# Monitoring Thresholds
MONITORING = {
    "error_threshold": 0.05,  # 5% error rate threshold
    "latency_threshold": 2000,  # 2 seconds
    "memory_warning": 0.7,  # 70% memory usage warning
    "batch_timeout": 30
}

def get_api_key(service: str) -> str:
    """Get API key from environment variables."""
    env_var = f"{service.upper()}_API_KEY"
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Missing API key for {service}. Set {env_var} environment variable.")
    return api_key
