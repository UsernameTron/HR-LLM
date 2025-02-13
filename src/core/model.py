"""
Core model implementation with MPS optimization for M4 Pro hardware.
"""
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config.config import settings, MPS_CONFIG

logger = logging.getLogger(__name__)

class HiringSentimentModel(nn.Module):
    def __init__(
        self,
        model_name: str = settings.MODEL_NAME,
        num_labels: int = 4,  # hiring, layoffs, restructuring, funding
        device: Optional[str] = None
    ):
        super().__init__()
        self.device = device or settings.DEVICE
        self.setup_mps_optimization()
        
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Multi-label classification head
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
        self.to(self.device)
        
    def setup_mps_optimization(self) -> None:
        """Configure MPS-specific optimizations for M4 Pro."""
        if self.device == "mps":
            logger.info("Configuring MPS optimizations for M4 Pro")
            for key, value in MPS_CONFIG.items():
                setattr(torch.backends.mps, key, value)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with MPS optimization."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return torch.sigmoid(logits)  # Multi-label output
    
    def predict(self, texts: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Predict hiring sentiments with confidence scores.
        Optimized for batch processing on MPS.
        """
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=settings.MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.forward(**inputs)
            confidences = {
                "hiring": outputs[:, 0].mean().item(),
                "layoffs": outputs[:, 1].mean().item(),
                "restructuring": outputs[:, 2].mean().item(),
                "funding": outputs[:, 3].mean().item(),
            }
            
            return outputs, confidences
    
    @staticmethod
    def create_batches(texts: List[str], batch_size: int = settings.BATCH_SIZE):
        """Create optimized batches for MPS processing."""
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]
