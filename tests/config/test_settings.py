"""Tests for Metal configuration settings."""
import pytest
from unittest.mock import patch, Mock
import torch

from src.config.settings import (
    parse_memory_size,
    check_metal_device_capabilities,
    validate_metal_config,
    MetalSettings,
    get_settings
)
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

def test_parse_memory_size():
    """Test memory size parsing."""
    assert parse_memory_size("1G") == 1024
    assert parse_memory_size("512M") == 512
    assert parse_memory_size("2048K") == 2
    
    with pytest.raises(ValueError):
        parse_memory_size("invalid")
    
    with pytest.raises(ValueError):
        parse_memory_size("1H")  # Invalid unit

def test_metal_settings_validation():
    """Test MetalSettings validation."""
    settings = MetalSettings(
        max_memory="42G",
        cpu_threads=12,
        gpu_memory="20G",
        mps_batch_size=256
    )
    
    assert settings.max_memory == "42G"
    assert settings.cpu_threads == 12
    assert settings.gpu_memory == "20G"
    assert settings.mps_batch_size == 256

@patch("torch.backends.mps.is_available")
@patch("torch.mps.get_device_name")
@patch("torch.mps.current_allocated_memory")
def test_check_metal_device_capabilities(
    mock_allocated_memory,
    mock_device_name,
    mock_is_available
):
    """Test device capability checking."""
    # Mock MPS availability
    mock_is_available.return_value = True
    mock_device_name.return_value = "Apple M4 Pro"
    mock_allocated_memory.return_value = 1024 * 1024 * 1024  # 1GB
    
    capabilities = check_metal_device_capabilities()
    
    assert capabilities["mps_available"] is True
    assert capabilities["device_name"] == "Apple M4 Pro"
    assert capabilities["memory_available"] == 1024 * 1024 * 1024
    assert "basic_operations" in capabilities["supported_features"]
    
    # Test when MPS is not available
    mock_is_available.return_value = False
    capabilities = check_metal_device_capabilities()
    
    assert capabilities["mps_available"] is False
    assert capabilities["device_name"] == "Unknown"
    assert capabilities["memory_available"] == 0

@patch("config.settings.check_metal_device_capabilities")
def test_validate_metal_config_success(mock_check_capabilities):
    """Test successful configuration validation."""
    mock_check_capabilities.return_value = {
        "mps_available": True,
        "device_name": "Apple M4 Pro",
        "memory_available": 1024 * 1024 * 1024,
        "supported_features": {"basic_operations", "convolution"}
    }
    
    settings = get_settings()
    assert validate_metal_config() is True

@patch("config.settings.check_metal_device_capabilities")
def test_validate_metal_config_failure(mock_check_capabilities):
    """Test configuration validation failure cases."""
    # Test when MPS is not available and fallback is disabled
    mock_check_capabilities.return_value = {
        "mps_available": False,
        "device_name": "Unknown",
        "memory_available": 0,
        "supported_features": set()
    }
    
    settings = get_settings()
    settings.metal.mps_fallback_to_cpu = False
    
    with pytest.raises(MetalError) as exc_info:
        validate_metal_config()
    
    assert exc_info.value.category == MetalErrorCategory.DEVICE_UNAVAILABLE

@patch("config.settings.check_metal_device_capabilities")
def test_validate_metal_config_memory_adjustment(mock_check_capabilities):
    """Test memory settings adjustment based on capabilities."""
    mock_check_capabilities.return_value = {
        "mps_available": True,
        "device_name": "Apple M4 Pro",
        "memory_available": 5 * 1024 * 1024 * 1024,  # 5GB
        "supported_features": {"basic_operations"}
    }
    
    settings = get_settings()
    settings.metal.gpu_memory = "10G"  # Request more than available
    
    validate_metal_config()
    
    # Should be adjusted to 90% of available memory
    assert settings.metal.gpu_memory.endswith("M")
    adjusted_memory = parse_memory_size(settings.metal.gpu_memory)
    assert adjusted_memory <= (5 * 1024 * 0.9)  # 90% of 5GB

@patch("config.settings.check_metal_device_capabilities")
def test_validate_metal_config_warnings(mock_check_capabilities, caplog):
    """Test warning generation during validation."""
    mock_check_capabilities.return_value = {
        "mps_available": True,
        "device_name": "Apple M4 Pro",
        "memory_available": 1024 * 1024 * 1024,
        "supported_features": {"basic_operations"}
    }
    
    settings = get_settings()
    settings.metal.mps_batch_size = 1024  # Large batch size
    settings.metal.cpu_threads = 24  # More than available cores
    
    validate_metal_config()
    
    # Check for warning messages
    assert any("Large batch size" in record.message for record in caplog.records)
    assert any("CPU thread count" in record.message for record in caplog.records)
