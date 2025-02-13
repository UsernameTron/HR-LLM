from pydantic_settings import BaseSettings
from typing import Optional
import logging
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

class MetalSettings(BaseSettings):
    """Settings for Metal Performance Shaders configuration."""
    max_memory: str = "42G"
    cpu_threads: int = 12
    gpu_memory: str = "20G"
    mps_batch_size: int = 256
    memory_pressure_check_interval: int = 30
    
    # Device settings
    force_cpu: bool = False
    mps_fallback_to_cpu: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Monitoring
    enable_performance_logging: bool = True
    log_memory_usage: bool = True
    
    class Config:
        env_prefix = "METAL_"
        case_sensitive = False

class BenchmarkSettings(BaseSettings):
    """Settings for benchmark configuration."""
    environment: str = "production"
    config_file: str = "config/benchmark_environments.yml"
    python_version: str = "3.11"
    
    # Metal settings
    metal: MetalSettings = MetalSettings()
    
    class Config:
        env_prefix = ""
        case_sensitive = False

# API configurations
API_CONFIGS = {
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'max_connections': 10,
        'socket_timeout': 5,
        'retry_on_timeout': True,
        'health_check_interval': 30
    }
}

# Global settings instance
settings = BenchmarkSettings()

def get_settings() -> BenchmarkSettings:
    """Get the global settings instance."""
    return settings

def parse_memory_size(size_str: str) -> float:
    """Parse memory size string to megabytes."""
    try:
        unit = size_str[-1].upper()
        value = float(size_str[:-1])
        if unit == 'G':
            return value * 1024
        elif unit == 'M':
            return value
        elif unit == 'K':
            return value / 1024
        raise ValueError(f"Invalid memory unit: {unit}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid memory format: {size_str}") from e

def check_metal_device_capabilities() -> dict:
    """Check Metal device capabilities and limitations."""
    import torch
    capabilities = {
        "mps_available": torch.backends.mps.is_available(),
        "device_name": "Unknown",
        "memory_available": 0,
        "supported_features": set()
    }
    
    try:
        if capabilities["mps_available"]:
            capabilities["device_name"] = torch.mps.get_device_name()
            capabilities["memory_available"] = torch.mps.current_allocated_memory()
            
            # Test basic operations
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100, device=device)
            capabilities["supported_features"].update([
                "basic_operations",
                "tensor_creation",
                "device_transfer"
            ])
            
            # Test advanced operations
            try:
                conv = torch.nn.Conv2d(1, 1, 3).to(device)
                capabilities["supported_features"].add("convolution")
            except Exception:
                logger.warning("Convolution operations not fully supported")
            
            # Cleanup
            del test_tensor
            torch.mps.empty_cache()
    except Exception as e:
        logger.error(f"Error checking device capabilities: {str(e)}", exc_info=True)
    
    return capabilities

def validate_metal_config() -> bool:
    """Validate Metal configuration settings with enhanced error handling."""
    from utils.metal_error_handler import MetalError, MetalErrorCategory, log_metal_error
    
    try:
        import torch
        
        # Check device availability and capabilities
        capabilities = check_metal_device_capabilities()
        if not capabilities["mps_available"]:
            if settings.metal.mps_fallback_to_cpu:
                logger.warning("MPS not available, falling back to CPU")
                settings.metal.force_cpu = True
                return True
            raise MetalError(
                "MPS not available and fallback to CPU is disabled",
                MetalErrorCategory.DEVICE_UNAVAILABLE
            )
        
        # Validate memory settings
        try:
            gpu_memory_mb = parse_memory_size(settings.metal.gpu_memory)
            max_memory_mb = parse_memory_size(settings.metal.max_memory)
            
            # Check against device capabilities
            if capabilities["memory_available"] > 0:
                available_memory_mb = capabilities["memory_available"] / (1024 * 1024)
                if gpu_memory_mb > available_memory_mb:
                    logger.warning(
                        f"Requested GPU memory ({gpu_memory_mb}MB) exceeds available memory "
                        f"({available_memory_mb}MB). Adjusting..."
                    )
                    settings.metal.gpu_memory = f"{int(available_memory_mb * 0.9)}M"
        except ValueError as e:
            raise MetalError(str(e), MetalErrorCategory.VALIDATION_ERROR, e)
        
        # Validate performance settings
        if settings.metal.mps_batch_size > 512:
            logger.warning("Large batch size may impact performance")
        
        # Validate thread settings
        if settings.metal.cpu_threads > 12:  # M4 Pro has 12 cores
            logger.warning(
                f"CPU thread count {settings.metal.cpu_threads} exceeds available cores (12). "
                "This may impact performance."
            )
        
        # Log validation success with capabilities
        logger.info(
            "Metal configuration validated successfully",
            extra={"capabilities": capabilities}
        )
        return True
        
    except Exception as e:
        if not isinstance(e, MetalError):
            e = MetalError(
                f"Error validating Metal configuration: {str(e)}",
                MetalErrorCategory.VALIDATION_ERROR,
                e
            )
        log_metal_error(e, {"settings": settings.metal.dict()})
        return False
