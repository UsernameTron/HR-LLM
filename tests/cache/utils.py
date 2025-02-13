"""
Test utilities for cache integration testing.
Provides helper functions and mock data generators.
"""
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

class TestDataGenerator:
    """Generate test data for cache integration tests."""
    
    @staticmethod
    def generate_api_response(size: int = 1000) -> Dict:
        """Generate mock API response data."""
        return {
            'id': str(random.randint(1000, 9999)),
            'timestamp': random.randint(1600000000, 1700000000),
            'data': {
                'text': 'x' * size,
                'metadata': {
                    'source': 'test',
                    'confidence': random.random(),
                    'tags': [f'tag_{i}' for i in range(5)]
                }
            }
        }
    
    @staticmethod
    def generate_embedding(dim: int = 768) -> np.ndarray:
        """Generate mock embedding vector."""
        return np.random.randn(dim).astype(np.float32)
    
    @staticmethod
    def generate_prediction(
        confidence: Optional[float] = None
    ) -> Tuple[Dict, float]:
        """Generate mock ML prediction with confidence."""
        if confidence is None:
            confidence = random.random()
        
        return {
            'label': random.choice(['HIRING', 'LAYOFF', 'NEUTRAL']),
            'scores': {
                'HIRING': random.random(),
                'LAYOFF': random.random(),
                'NEUTRAL': random.random()
            },
            'metadata': {
                'model_version': '1.0.0',
                'timestamp': random.randint(1600000000, 1700000000)
            }
        }, confidence
    
    @staticmethod
    def generate_batch_predictions(
        size: int = 100
    ) -> List[Tuple[Dict, float]]:
        """Generate batch of predictions for performance testing."""
        return [
            TestDataGenerator.generate_prediction()
            for _ in range(size)
        ]
    
    @staticmethod
    def generate_large_document(kb_size: int = 1024) -> str:
        """Generate large document of specified size in KB."""
        words = [
            'hiring', 'growth', 'expansion', 'talent',
            'recruitment', 'position', 'opportunity'
        ]
        
        document = []
        while len(' '.join(document).encode('utf-8')) < kb_size * 1024:
            document.append(random.choice(words))
        
        return ' '.join(document)

class MockRedisClient:
    """Mock Redis client for testing without actual Redis server."""
    
    def __init__(self):
        self.data = {}
        self.ttls = {}
    
    async def get(self, key: str) -> Optional[bytes]:
        """Mock get operation."""
        if key in self.data:
            return self.data[key]
        return None
    
    async def set(
        self,
        key: str,
        value: bytes,
        ex: Optional[int] = None
    ) -> bool:
        """Mock set operation."""
        try:
            self.data[key] = value
            if ex:
                self.ttls[key] = ex
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> int:
        """Mock delete operation."""
        if key in self.data:
            del self.data[key]
            if key in self.ttls:
                del self.ttls[key]
            return 1
        return 0
    
    async def ttl(self, key: str) -> int:
        """Mock TTL operation."""
        return self.ttls.get(key, -2)
    
    async def info(self, section: str = None) -> Dict:
        """Mock info operation."""
        return {
            'used_memory': len(str(self.data).encode('utf-8')),
            'maxmemory': 1024 * 1024 * 1024  # 1GB
        }
    
    def pipeline(self):
        """Mock pipeline operation."""
        return MockRedisPipeline(self)

class MockRedisPipeline:
    """Mock Redis pipeline for testing."""
    
    def __init__(self, client: MockRedisClient):
        self.client = client
        self.commands = []
    
    def set(self, key: str, value: bytes, ex: Optional[int] = None):
        """Mock pipeline set operation."""
        self.commands.append(('set', key, value, ex))
        return self
    
    def get(self, key: str):
        """Mock pipeline get operation."""
        self.commands.append(('get', key))
        return self
    
    def delete(self, key: str):
        """Mock pipeline delete operation."""
        self.commands.append(('delete', key))
        return self
    
    async def execute(self) -> List:
        """Execute mock pipeline commands."""
        results = []
        for cmd in self.commands:
            if cmd[0] == 'set':
                results.append(
                    await self.client.set(cmd[1], cmd[2], cmd[3])
                )
            elif cmd[0] == 'get':
                results.append(
                    await self.client.get(cmd[1])
                )
            elif cmd[0] == 'delete':
                results.append(
                    await self.client.delete(cmd[1])
                )
        return results
