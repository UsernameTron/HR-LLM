"""Drift detection and anomaly tracking for sentiment classification."""
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, UTC, timedelta
from collections import deque
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

@dataclass
class DriftMetrics:
    """Metrics for tracking distribution drift."""
    ks_statistic: float
    p_value: float
    feature_importance_shift: float
    confidence_distribution_shift: float
    timestamp: str

class DriftDetector:
    """Detects and monitors concept drift in sentiment classification."""
    
    def __init__(
        self,
        window_size: int = 1000,
        reference_size: int = 5000,
        drift_threshold: float = 0.05,
        feature_drift_threshold: float = 0.3
    ):
        self.window_size = window_size
        self.reference_size = reference_size
        self.drift_threshold = drift_threshold
        self.feature_drift_threshold = feature_drift_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self.reference_embeddings = None
        self.reference_confidences = None
        self.reference_features = None
        self.current_window = deque(maxlen=window_size)
        self.drift_history = deque(maxlen=100)
        
        # Initialize feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.pca = IncrementalPCA(
            n_components=50,
            batch_size=100
        )
        
    def _compute_feature_importance_shift(
        self,
        current_features: np.ndarray
    ) -> float:
        """Compute shift in feature importance distributions."""
        if self.reference_features is None:
            return 0.0
            
        try:
            # Compare feature distributions
            reference_mean = np.mean(self.reference_features, axis=0)
            current_mean = np.mean(current_features, axis=0)
            
            # Compute relative importance shift
            importance_shift = np.abs(
                (current_mean - reference_mean) / (reference_mean + 1e-10)
            ).mean()
            
            return float(importance_shift)
            
        except Exception as e:
            self.logger.error(f"Feature importance shift computation error: {str(e)}")
            return 0.0
    
    def _compute_confidence_distribution_shift(
        self,
        current_confidences: np.ndarray
    ) -> float:
        """Compute shift in model confidence distributions."""
        if self.reference_confidences is None:
            return 0.0
            
        try:
            # Perform KS test on confidence distributions
            ks_statistic, _ = stats.ks_2samp(
                self.reference_confidences.flatten(),
                current_confidences.flatten()
            )
            return float(ks_statistic)
            
        except Exception as e:
            self.logger.error(f"Confidence distribution shift computation error: {str(e)}")
            return 0.0
    
    def update_reference(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        confidences: np.ndarray
    ):
        """Update reference distribution."""
        try:
            # Update feature reference
            features = self.vectorizer.fit_transform(texts).toarray()
            self.reference_features = features
            
            # Update embedding reference
            self.reference_embeddings = embeddings
            
            # Update confidence reference
            self.reference_confidences = confidences
            
            self.logger.info(
                f"Updated reference distribution with {len(texts)} samples"
            )
            
        except Exception as e:
            raise MetalError(
                f"Failed to update reference distribution: {str(e)}",
                category=MetalErrorCategory.MONITORING_ERROR
            )
    
    def detect_drift(
        self,
        text: str,
        embedding: np.ndarray,
        confidence: np.ndarray
    ) -> Optional[DriftMetrics]:
        """Detect drift in a single sample."""
        try:
            # Add to current window
            self.current_window.append({
                "text": text,
                "embedding": embedding,
                "confidence": confidence
            })
            
            # Only perform detection with sufficient samples
            if len(self.current_window) < self.window_size:
                return None
            
            # Extract features
            current_texts = [item["text"] for item in self.current_window]
            current_features = self.vectorizer.transform(current_texts).toarray()
            
            # Compute embedding distribution shift
            current_embeddings = np.stack(
                [item["embedding"] for item in self.current_window]
            )
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_embeddings.flatten(),
                current_embeddings.flatten()
            )
            
            # Compute feature importance shift
            feature_shift = self._compute_feature_importance_shift(current_features)
            
            # Compute confidence distribution shift
            current_confidences = np.stack(
                [item["confidence"] for item in self.current_window]
            )
            confidence_shift = self._compute_confidence_distribution_shift(
                current_confidences
            )
            
            # Create metrics
            metrics = DriftMetrics(
                ks_statistic=float(ks_statistic),
                p_value=float(p_value),
                feature_importance_shift=feature_shift,
                confidence_distribution_shift=confidence_shift,
                timestamp=datetime.now(UTC).isoformat()
            )
            
            # Store in history
            self.drift_history.append(metrics)
            
            # Log if significant drift detected
            if (p_value < self.drift_threshold or 
                feature_shift > self.feature_drift_threshold):
                self.logger.warning(
                    f"Significant drift detected: "
                    f"KS p-value={p_value:.4f}, "
                    f"Feature shift={feature_shift:.4f}"
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Drift detection error: {str(e)}")
            return None
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of recent drift metrics."""
        if not self.drift_history:
            return {}
            
        recent_metrics = list(self.drift_history)
        return {
            "recent_ks_statistic": np.mean([m.ks_statistic for m in recent_metrics]),
            "recent_p_value": np.mean([m.p_value for m in recent_metrics]),
            "recent_feature_shift": np.mean([m.feature_importance_shift for m in recent_metrics]),
            "recent_confidence_shift": np.mean([m.confidence_distribution_shift for m in recent_metrics]),
            "drift_detected": any(
                m.p_value < self.drift_threshold or
                m.feature_importance_shift > self.feature_drift_threshold
                for m in recent_metrics
            )
        }
    
    def should_update_reference(self) -> bool:
        """Determine if reference distribution should be updated."""
        summary = self.get_drift_summary()
        if not summary:
            return False
            
        # Update if consistent drift is detected
        return (
            summary["drift_detected"] and
            summary["recent_p_value"] < self.drift_threshold and
            summary["recent_feature_shift"] > self.feature_drift_threshold
        )
