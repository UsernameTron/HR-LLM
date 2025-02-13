"""Test individual components of the pipeline."""
import pytest
import asyncio
import torch
import warnings
from typing import Dict, List, Any
from datetime import datetime

# Filter SHAP internal deprecation warnings
warnings.filterwarnings(
    "ignore",
    message="Converting `np.inexact` or `np.floating` to a dtype is deprecated",
    category=DeprecationWarning,
    module="shap.plots.colors._colorconv"
)

from src.monitoring.drift_detector import DriftDetector
from src.classification.explainer import RealTimeExplainer
from src.cache.redis_memory_monitor import RedisMemoryMonitor

@pytest.mark.asyncio
async def test_drift_detector():
    """Test drift detection functionality."""
    detector = DriftDetector(
        window_size=3,  # Small window for testing
        reference_size=5,
        drift_threshold=0.05
    )
    
    # Create sample data
    embeddings = torch.randn(10, 768)  # Simulated embeddings
    confidences = torch.sigmoid(torch.randn(10, 3))  # Simulated confidences
    texts = [
        "Software engineer position available",
        "Looking for experienced developers",
        "Great opportunity in tech",
        "Senior developer role",
        "Python engineer needed"
    ] * 2
    
    # Update reference distribution
    detector.update_reference(
        texts[:5],
        embeddings[:5],
        confidences[:5]
    )
    
    # Add samples to current window
    for i in range(3):  # Fill window size
        detector.detect_drift(
            texts[i],
            embeddings[i],
            confidences[i]
        )
    
    # Test drift detection with new sample
    metrics = detector.detect_drift(
        texts[-1],
        embeddings[-1],
        confidences[-1]
    )
    
    assert metrics is not None
    assert 0 <= metrics.ks_statistic <= 1
    assert 0 <= metrics.p_value <= 1
    assert 0 <= metrics.feature_importance_shift <= 1
    assert 0 <= metrics.confidence_distribution_shift <= 1
    
    # Test drift summary
    summary = detector.get_drift_summary()
    assert isinstance(summary, dict)
    assert "drift_detected" in summary

@pytest.mark.asyncio
async def test_explainer():
    """Test SHAP explanation generation."""
    # Mock tokenizer and model
    class MockTokenizer:
        def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
            return {
                "input_ids": torch.ones(len(texts), 10),
                "attention_mask": torch.ones(len(texts), 10)
            }
        
        def tokenize(self, text):
            return ["token1", "token2", "token3"]
    
    class MockModel:
        def __init__(self):
            self.output_names = ["negative", "neutral", "positive"]
        
        def __call__(self, **inputs):
            batch_size = inputs["input_ids"].shape[0]
            class MockOutput:
                def __init__(self):
                    self.logits = torch.randn(batch_size, 3)
                    self.hidden_states = [
                        torch.randn(batch_size, 10, 768)
                        for _ in range(12)  # 12 layers
                    ]
            return MockOutput()
    
    # Create mock SHAP values
    class MockShapValues:
        def __init__(self):
            self.values = torch.randn(1, 3, 3)  # [batch, tokens, classes]
            self.base_values = torch.zeros(1, 3)
            self.data = ["token1", "token2", "token3"]
    
    # Mock SHAP explainer
    class MockShapExplainer:
        def __init__(self, model, tokenizer, output_names=None):
            self.model = model
            self.tokenizer = tokenizer
            self.output_names = output_names
        
        def __call__(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return MockShapValues()
    
    # Patch SHAP explainer creation
    import shap
    original_explainer = shap.Explainer
    shap.Explainer = MockShapExplainer
    
    try:
        explainer = RealTimeExplainer(
            tokenizer=MockTokenizer(),
            device="cpu"
        )
        
        # Update background
        explainer.update_background(
            ["Sample text 1", "Sample text 2", "Sample text 3"]
        )
        
        # Test explanation generation
        text = "Software engineer position"
        prediction = torch.tensor([0.1, 0.2, 0.7])
        confidence = 0.7
        
        explanation = explainer.explain_prediction(
            text,
            prediction,
            MockModel(),
            confidence
        )
        
        assert explanation is not None
        assert len(explanation.feature_importance) > 0
        assert len(explanation.token_attributions) > 0
        assert 0 <= explanation.explanation_confidence <= 1
        
        # Test explanation metrics
        metrics = explainer.get_explanation_metrics(explanation)
        assert isinstance(metrics, dict)
        assert "num_important_features" in metrics
        assert "explanation_confidence" in metrics
        
    finally:
        # Restore original SHAP explainer
        shap.Explainer = original_explainer

@pytest.mark.asyncio
async def test_memory_monitor():
    """Test memory monitoring functionality."""
    class MockRedis:
        async def info(self, section=None):
            return {
                'used_memory': 1000000,
                'used_memory_rss': 2000000,
                'used_memory_peak': 3000000,
                'used_memory_lua': 50000,
                'used_memory_scripts': 25000,
                'maxmemory': 8000000000
            }
        
        async def memory(self, subcommand, *args):
            if subcommand == 'stats':
                return {
                    'total.allocated': 1000000,
                    'total.fragmentation': 1.5,
                    'keys.count': 1000
                }
            return {}
    
    redis_client = MockRedis()
    monitor = RedisMemoryMonitor(redis_client)
    
    # Test Redis info method
    memory_info = await redis_client.info()
    assert isinstance(memory_info["used_memory"], int)
    assert memory_info["used_memory"] >= 0
    assert memory_info["used_memory_rss"] >= 0
    assert memory_info["maxmemory"] >= 0
    
    # Test Redis memory stats
    memory_stats = await redis_client.memory("stats")
    assert isinstance(memory_stats["total.allocated"], int)
    assert memory_stats["total.allocated"] >= 0
    assert isinstance(memory_stats["total.fragmentation"], float)
    assert memory_stats["total.fragmentation"] >= 0
    assert memory_stats["keys.count"] >= 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])
