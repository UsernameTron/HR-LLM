"""
Data processing package initialization.
Sets up cache middleware for all processors.
"""
from ..cache.middleware import CacheMiddleware

# Create cache middleware instance
cache = CacheMiddleware()

async def init_cache():
    """Initialize cache middleware asynchronously"""
    await cache.init()
