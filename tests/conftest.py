"""Pytest configuration and fixtures."""
import pytest
import logging
from rich.logging import RichHandler

# Configure logging for tests
@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

# Mock hardware configuration for M4 Pro
@pytest.fixture
def mock_m4_pro_config():
    """Mock M4 Pro hardware configuration."""
    return {
        "cpu_cores": 12,
        "performance_cores": 8,
        "efficiency_cores": 4,
        "gpu_cores": 18,
        "neural_engine_cores": 16,
        "unified_memory": 48 * 1024,  # MB
        "memory_bandwidth": 300,  # GB/s
        "metal_version": 3
    }

@pytest.fixture
def mock_settings():
    """Mock settings configuration."""
    from config.settings import MetalSettings, BenchmarkSettings
    
    metal_settings = MetalSettings(
        max_memory="42G",
        cpu_threads=12,
        gpu_memory="20G",
        mps_batch_size=256,
        memory_pressure_check_interval=30,
        force_cpu=False,
        mps_fallback_to_cpu=True,
        max_retries=3,
        retry_delay=1.0,
        enable_performance_logging=True,
        log_memory_usage=True
    )
    
    return BenchmarkSettings(
        environment="test",
        config_file="config/test_environments.yml",
        python_version="3.11",
        metal=metal_settings
    )

@pytest.fixture
def mock_error_context():
    """Mock error context for testing."""
    return {
        "device_info": {
            "name": "Apple M4 Pro",
            "metal_version": 3,
            "unified_memory": "48GB"
        },
        "performance_metrics": {
            "memory_usage": 1024,
            "cpu_usage": 50,
            "gpu_usage": 75
        },
        "system_state": {
            "available_memory": 40960,
            "memory_pressure": False,
            "thermal_pressure": False
        }
    }
