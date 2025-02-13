"""
Data processing package initialization.
Sets up cache middleware for all processors.
"""
from src.cache.middleware import CacheMiddleware

# Initialize cache middleware
cache = CacheMiddleware()
