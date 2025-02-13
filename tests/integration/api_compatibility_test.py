"""API compatibility and rate limit testing."""
import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.ingestion.api_aggregator import APIAggregator
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@dataclass
class APITestResult:
    """Results from API compatibility testing."""
    endpoint: str
    success_rate: float
    avg_latency: float
    rate_limit_hits: int
    errors: List[str]
    samples: int

class APICompatibilityTester:
    """Tests API compatibility and rate limiting."""
    
    def __init__(
        self,
        api_aggregator: APIAggregator,
        test_duration: int = 3600,  # 1 hour
        request_rate: float = 1.0,  # requests/second
        endpoints: Optional[List[str]] = None
    ):
        self.api_aggregator = api_aggregator
        self.test_duration = test_duration
        self.request_rate = request_rate
        self.endpoints = endpoints or [
            "/api/v1/jobs",
            "/api/v1/companies",
            "/api/v1/skills"
        ]
        self.logger = logging.getLogger(__name__)
        
    async def _test_endpoint(
        self,
        endpoint: str,
        session: aiohttp.ClientSession
    ) -> APITestResult:
        """Test a single API endpoint."""
        start_time = time.time()
        samples = 0
        errors = []
        latencies = []
        rate_limit_hits = 0
        
        while time.time() - start_time < self.test_duration:
            try:
                # Make request
                request_start = time.time()
                async with session.get(endpoint) as response:
                    latency = time.time() - request_start
                    latencies.append(latency)
                    
                    if response.status == 429:  # Rate limit
                        rate_limit_hits += 1
                        # Honor rate limit retry-after
                        retry_after = int(response.headers.get('Retry-After', '60'))
                        await asyncio.sleep(retry_after)
                    elif response.status != 200:
                        errors.append(
                            f"HTTP {response.status}: {await response.text()}"
                        )
                    
                    samples += 1
                    
            except Exception as e:
                errors.append(str(e))
            
            # Rate limiting
            await asyncio.sleep(1 / self.request_rate)
        
        return APITestResult(
            endpoint=endpoint,
            success_rate=(samples - len(errors)) / samples if samples > 0 else 0,
            avg_latency=sum(latencies) / len(latencies) if latencies else 0,
            rate_limit_hits=rate_limit_hits,
            errors=errors[:10],  # Limit error list
            samples=samples
        )
    
    async def run_compatibility_test(self) -> Dict[str, Any]:
        """Run compatibility tests for all endpoints."""
        self.logger.info(
            f"Starting API compatibility test for {len(self.endpoints)} endpoints"
        )
        
        async with aiohttp.ClientSession() as session:
            # Test all endpoints concurrently
            tasks = [
                self._test_endpoint(endpoint, session)
                for endpoint in self.endpoints
            ]
            results = await asyncio.gather(*tasks)
            
        # Generate report
        report = {
            "test_duration": self.test_duration,
            "request_rate": self.request_rate,
            "endpoints_tested": len(self.endpoints),
            "results": {
                result.endpoint: {
                    "success_rate": result.success_rate,
                    "avg_latency": result.avg_latency,
                    "rate_limit_hits": result.rate_limit_hits,
                    "errors": result.errors,
                    "samples": result.samples
                }
                for result in results
            },
            "summary": {
                "overall_success_rate": sum(
                    r.success_rate for r in results
                ) / len(results),
                "avg_latency_across_endpoints": sum(
                    r.avg_latency for r in results
                ) / len(results),
                "total_rate_limit_hits": sum(
                    r.rate_limit_hits for r in results
                ),
                "total_samples": sum(r.samples for r in results)
            }
        }
        
        # Add pass/fail status
        report["thresholds_met"] = {
            "success_rate": report["summary"]["overall_success_rate"] >= 0.99,
            "latency": report["summary"]["avg_latency_across_endpoints"] <= 1.0,
            "rate_limits": report["summary"]["total_rate_limit_hits"] == 0
        }
        
        return report
    
    def validate_rate_limits(
        self,
        report: Dict[str, Any]
    ) -> List[str]:
        """Validate rate limit compliance."""
        violations = []
        
        for endpoint, results in report["results"].items():
            rate_limit_percentage = (
                results["rate_limit_hits"] / results["samples"]
                if results["samples"] > 0 else 0
            )
            
            if rate_limit_percentage > 0.01:  # More than 1% rate limit hits
                violations.append(
                    f"Endpoint {endpoint} exceeded rate limit threshold: "
                    f"{rate_limit_percentage:.2%} of requests"
                )
            
            if results["avg_latency"] > 1.0:  # More than 1s average latency
                violations.append(
                    f"Endpoint {endpoint} exceeded latency threshold: "
                    f"{results['avg_latency']:.2f}s average"
                )
        
        return violations
