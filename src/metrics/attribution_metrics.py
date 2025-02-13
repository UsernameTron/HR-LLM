"""Attribution metrics collector for monitoring token attribution quality."""
import numpy as np
import torch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AttributionMetricsCollector:
    """Collects and analyzes metrics related to token attribution quality."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._reset_metrics()
    
    def _reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            "entropy": [],
            "sparsity": [],
            "gradient_norm": [],
            "latency": [],
            "error_count": 0,
            "total_count": 0
        }
    
    def record_attribution(
        self,
        attributions: torch.Tensor,
        latency_ms: float,
        error: Optional[str] = None
    ):
        """Record metrics for a single attribution computation."""
        try:
            # Update counters
            self.metrics["total_count"] += 1
            if error:
                self.metrics["error_count"] += 1
                logger.warning(f"Attribution error: {error}")
                return
            
            # Record latency
            self.metrics["latency"].append(latency_ms)
            
            # Compute attribution metrics
            with torch.no_grad():
                # Normalize attributions
                norm_attr = torch.nn.functional.softmax(attributions, dim=-1)
                
                # Compute entropy
                entropy = -torch.sum(norm_attr * torch.log(norm_attr + 1e-10))
                self.metrics["entropy"].append(entropy.item())
                
                # Compute sparsity (percentage of attributions above threshold)
                threshold = 0.05  # 5% threshold
                sparsity = torch.mean((norm_attr > threshold).float())
                self.metrics["sparsity"].append(sparsity.item())
                
                # Compute gradient norm
                grad_norm = torch.norm(attributions)
                self.metrics["gradient_norm"].append(grad_norm.item())
            
            # Trim metrics if they exceed window size
            for key in ["entropy", "sparsity", "gradient_norm", "latency"]:
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key] = self.metrics[key][-self.window_size:]
        
        except Exception as e:
            logger.error(f"Error recording attribution metrics: {str(e)}")
            self.metrics["error_count"] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics summary."""
        try:
            return {
                "entropy": np.mean(self.metrics["entropy"]) if self.metrics["entropy"] else 0.0,
                "entropy_std": np.std(self.metrics["entropy"]) if self.metrics["entropy"] else 0.0,
                "sparsity": np.mean(self.metrics["sparsity"]) if self.metrics["sparsity"] else 0.0,
                "gradient_norm": np.mean(self.metrics["gradient_norm"]) if self.metrics["gradient_norm"] else 0.0,
                "latency_p95": np.percentile(self.metrics["latency"], 95) if self.metrics["latency"] else 0.0,
                "error_rate": self.metrics["error_count"] / max(1, self.metrics["total_count"])
            }
        except Exception as e:
            logger.error(f"Error computing metrics summary: {str(e)}")
            return {
                "entropy": 0.0,
                "entropy_std": 0.0,
                "sparsity": 0.0,
                "gradient_norm": 0.0,
                "latency_p95": 0.0,
                "error_rate": 1.0
            }
    
    def check_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Check if metrics are within specified thresholds."""
        metrics = self.get_metrics()
        return {
            "entropy": thresholds["min_entropy"] <= metrics["entropy"] <= thresholds["max_entropy"],
            "sparsity": metrics["sparsity"] >= thresholds["min_sparsity"],
            "latency": metrics["latency_p95"] <= thresholds["max_latency"],
            "error_rate": metrics["error_rate"] <= thresholds["max_error_rate"]
        }
