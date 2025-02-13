import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import defaultdict
import torch
import logging
from rich.logging import RichHandler

@dataclass
class AttributionMetrics:
    entropy: float
    sparsity: float
    gradient_norm: float
    latency_ms: float
    memory_mb: float
    input_length: int
    error: Optional[str] = None

class AttributionMetricsCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[AttributionMetrics] = []
        self.error_counts = defaultdict(int)
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
    
    def compute_entropy(self, attributions: List[Dict[str, float]]) -> float:
        importances = np.array([abs(attr['importance']) for attr in attributions])
        total = importances.sum()
        if total == 0:
            return 0.0
        probs = importances / total
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    def compute_sparsity(self, attributions: List[Dict[str, float]], threshold: float = 0.01) -> float:
        importances = np.array([abs(attr['importance']) for attr in attributions])
        total = importances.sum()
        if total == 0:
            return 0.0
        normalized = importances / total
        return float(np.mean(normalized > threshold))
    
    def record_attribution(
        self,
        attributions: List[Dict[str, float]],
        gradient_norm: float,
        latency_ms: float,
        memory_mb: float,
        input_length: int,
        error: Optional[str] = None
    ) -> AttributionMetrics:
        try:
            metrics = AttributionMetrics(
                entropy=self.compute_entropy(attributions),
                sparsity=self.compute_sparsity(attributions),
                gradient_norm=gradient_norm,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                input_length=input_length,
                error=error
            )
            self.metrics_history.append(metrics)
            if error:
                self.error_counts[error] += 1
                self.logger.warning(f"Attribution error: {error}")
            else:
                self.logger.debug(
                    f"Recorded metrics - Entropy: {metrics.entropy:.3f}, "
                    f"Sparsity: {metrics.sparsity:.3f}, "
                    f"Latency: {metrics.latency_ms:.2f}ms"
                )
            return metrics
        except Exception as e:
            self.logger.error(f"Error recording metrics: {e}")
            return None

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        if not self.metrics_history:
            return {}
        
        metrics_array = np.array([
            [m.entropy, m.sparsity, m.gradient_norm, m.latency_ms, m.memory_mb]
            for m in self.metrics_history
        ])
        
        metrics_names = ['entropy', 'sparsity', 'gradient_norm', 'latency_ms', 'memory_mb']
        stats = {}
        
        for i, name in enumerate(metrics_names):
            values = metrics_array[:, i]
            stats[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'p50': float(np.percentile(values, 50)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
        
        return stats
