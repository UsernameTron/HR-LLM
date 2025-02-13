"""
Load testing script using Locust.
Simulates realistic user patterns for cache operations.
"""
import json
import random
import time
from typing import Dict, Optional

import numpy as np
from locust import FastHttpUser, between, events, task

from tests.cache.utils import TestDataGenerator

class CacheLoadTest(FastHttpUser):
    """Simulate realistic cache usage patterns."""
    
    wait_time = between(0.1, 1.0)  # Simulate think time
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_gen = TestDataGenerator()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def on_start(self):
        """Initialize test data on startup."""
        self.common_keys = [
            f"common_key_{i}"
            for i in range(1000)
        ]
        self.rare_keys = [
            f"rare_key_{i}"
            for i in range(10000)
        ]
    
    @task(40)
    def read_common_key(self):
        """Simulate frequent reads of popular keys."""
        key = random.choice(self.common_keys)
        start_time = time.time()
        try:
            response = self.client.get(
                f"/api/cache/{key}",
                name="Read Common Key"
            )
            if response.status_code == 200:
                self.cache_hits += 1
            elif response.status_code == 404:
                self.cache_misses += 1
        except Exception as e:
            events.request_failure.fire(
                request_type="GET",
                name="Read Common Key",
                response_time=int((time.time() - start_time) * 1000),
                exception=e
            )
    
    @task(20)
    def read_rare_key(self):
        """Simulate occasional reads of rare keys."""
        key = random.choice(self.rare_keys)
        start_time = time.time()
        try:
            response = self.client.get(
                f"/api/cache/{key}",
                name="Read Rare Key"
            )
            if response.status_code == 200:
                self.cache_hits += 1
            elif response.status_code == 404:
                self.cache_misses += 1
        except Exception as e:
            events.request_failure.fire(
                request_type="GET",
                name="Read Rare Key",
                response_time=int((time.time() - start_time) * 1000),
                exception=e
            )
    
    @task(30)
    def write_data(self):
        """Simulate write operations."""
        key = f"write_key_{random.randint(1, 1000000)}"
        data = self.data_gen.generate_api_response(
            random.randint(100, 10000)
        )
        
        start_time = time.time()
        try:
            response = self.client.post(
                f"/api/cache/{key}",
                json=data,
                name="Write Data"
            )
            if response.status_code != 200:
                events.request_failure.fire(
                    request_type="POST",
                    name="Write Data",
                    response_time=int((time.time() - start_time) * 1000),
                    exception=f"Status {response.status_code}"
                )
        except Exception as e:
            events.request_failure.fire(
                request_type="POST",
                name="Write Data",
                response_time=int((time.time() - start_time) * 1000),
                exception=e
            )
    
    @task(10)
    def pipeline_operation(self):
        """Simulate pipeline operations."""
        batch_size = random.randint(10, 100)
        operations = []
        
        for _ in range(batch_size):
            key = f"pipeline_key_{random.randint(1, 1000000)}"
            data = self.data_gen.generate_api_response(1000)
            operations.append({
                'key': key,
                'data': data
            })
        
        start_time = time.time()
        try:
            response = self.client.post(
                "/api/cache/pipeline",
                json={'operations': operations},
                name="Pipeline Operation"
            )
            if response.status_code != 200:
                events.request_failure.fire(
                    request_type="POST",
                    name="Pipeline Operation",
                    response_time=int((time.time() - start_time) * 1000),
                    exception=f"Status {response.status_code}"
                )
        except Exception as e:
            events.request_failure.fire(
                request_type="POST",
                name="Pipeline Operation",
                response_time=int((time.time() - start_time) * 1000),
                exception=e
            )

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("Starting load test...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report test results."""
    stats = environment.stats
    
    # Calculate cache hit ratio
    total_hits = sum(
        user.cache_hits
        for user in environment.runner.user_instances
    )
    total_misses = sum(
        user.cache_misses
        for user in environment.runner.user_instances
    )
    hit_ratio = total_hits / (total_hits + total_misses) \
        if (total_hits + total_misses) > 0 else 0
    
    print("\nTest Results:")
    print(f"Cache Hit Ratio: {hit_ratio:.2%}")
    print("\nRequest Statistics:")
    for name, stats in environment.stats.entries.items():
        print(f"\n{name}:")
        print(f"  Requests: {stats.num_requests}")
        print(f"  Failures: {stats.num_failures}")
        print(f"  Median Response Time: {stats.median_response_time}ms")
        print(f"  95th Percentile: {stats.get_response_time_percentile(0.95)}ms")
        print(f"  99th Percentile: {stats.get_response_time_percentile(0.99)}ms")
