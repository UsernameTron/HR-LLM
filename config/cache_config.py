"""
Redis cache configuration optimized for M4 Pro hardware.
"""
from typing import Dict

# Redis Configuration
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'max_connections': 10,
    'socket_timeout': 5,
    'retry_on_timeout': True
}

# Cache TTL Configuration (in seconds)
CACHE_TTL = {
    'api_response': {
        'default': 3600,  # 1 hour
        'company_data': 86400,  # 24 hours
        'news': 1800,  # 30 minutes
        'market_data': 300  # 5 minutes
    },
    'embedding': {
        'default': 604800,  # 7 days
        'common_phrases': 2592000  # 30 days
    },
    'prediction': {
        'high_confidence': 86400,  # 24 hours
        'medium_confidence': 43200,  # 12 hours
        'low_confidence': 3600  # 1 hour
    }
}

# Memory Management
MEMORY_LIMITS = {
    'max_memory': '2gb',  # Maximum memory usage
    'max_memory_policy': 'allkeys-lru',  # Least Recently Used eviction
    'max_memory_samples': 10  # Number of samples for LRU algorithm
}

# Cache Warming Configuration
CACHE_WARMING = {
    'enabled': True,
    'company_limit': 1000,  # Number of companies to pre-cache
    'phrase_limit': 5000,  # Number of common phrases to pre-cache
    'batch_size': 100  # Batch size for cache warming operations
}

# Performance Optimization
PERFORMANCE_CONFIG = {
    'compression_enabled': True,
    'compression_threshold': 1024,  # Compress values larger than 1KB
    'max_pipeline_size': 1000,  # Maximum number of commands in pipeline
    'connection_pool_size': 10  # Size of connection pool
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'stats_enabled': True,
    'stats_interval': 60,  # Collect stats every 60 seconds
    'alert_threshold': {
        'memory_usage': 80,  # Alert at 80% memory usage
        'hit_rate': 50,  # Alert if hit rate drops below 50%
        'error_rate': 5  # Alert if error rate exceeds 5%
    }
}

# Source-specific Cache Configuration
SOURCE_CONFIG: Dict[str, Dict] = {
    'newsapi': {
        'cache_enabled': True,
        'ttl': CACHE_TTL['api_response']['news'],
        'rate_limit': 100,  # Requests per minute
        'batch_size': 50
    },
    'linkedin': {
        'cache_enabled': True,
        'ttl': CACHE_TTL['api_response']['company_data'],
        'rate_limit': 50,
        'batch_size': 25
    },
    'gdelt': {
        'cache_enabled': True,
        'ttl': CACHE_TTL['api_response']['news'],
        'rate_limit': 200,
        'batch_size': 100
    }
}
