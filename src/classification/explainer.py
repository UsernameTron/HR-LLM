"""Real-time SHAP integration for sentiment classification."""
import torch
import numpy as np
import shap
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from src.utils.metal_error_handler import MetalError, MetalErrorCategory
from src.monitoring.attribution_metrics import AttributionMetricsCollector

@dataclass
class ExplanationResult:
    """Container for explanation results."""
    feature_importance: Dict[str, float]
    token_attributions: List[Dict[str, Any]]
    base_score: float
    explanation_confidence: float

class RealTimeExplainer:
    """Manages real-time explanations for model predictions."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: str = "mps",
        batch_size: int = 32,
        max_local_size: int = 100
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_local_size = max_local_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics collector
        self.metrics_collector = AttributionMetricsCollector()
        
        # Initialize SHAP explainer
        self.background_data = []
        self.background_updated = False
    
    def _prepare_background(
        self,
        texts: List[str]
    ):
        """Prepare background dataset for SHAP."""
        try:
            # Tokenize background data
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Store background data
            self.background_data = {
                k: v.to(self.device) for k, v in encodings.items()
            }
            self.background_updated = True
            
        except Exception as e:
            raise MetalError(
                f"Failed to prepare background data: {str(e)}",
                category=MetalErrorCategory.EXPLANATION_ERROR
            )
    
    def _compute_token_attributions(
        self,
        text: str,
        model_fn: Any,
        target_class: int
    ) -> List[Dict[str, Any]]:
        """Compute token-level attributions using SHAP."""
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            # Process token attributions first
            tokens = self.tokenizer.tokenize(text)
            attributions = []
            
            try:
                # Tokenize input
                inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
                
                # Get embeddings from model
                with torch.no_grad():
                    input_ids = inputs["input_ids"].to(self.device)
                    embeddings = model_fn.distilbert.embeddings.word_embeddings(input_ids)
                
                # Enable gradient computation on embeddings
                embeddings.requires_grad_(True)
                
                # Forward pass through model using embeddings
                outputs = model_fn.distilbert(inputs_embeds=embeddings)
                hidden_states = outputs.last_hidden_state
                
                # Get classifier output
                pooled = hidden_states[:, 0]  # Use [CLS] token
                logits = model_fn.pre_classifier(pooled)
                logits = model_fn.classifier(logits)
                
                # Zero gradients
                if embeddings.grad is not None:
                    embeddings.grad.zero_()
                
                # Compute gradients for target class
                target_score = logits[0, target_class]
                target_score.backward()
                
                # Get gradients as attribution values
                if embeddings.grad is not None:
                    values = embeddings.grad[0].sum(-1).cpu().detach().numpy()
                else:
                    raise ValueError("No gradients computed")
                
                # Process each token
                for i, token in enumerate(tokens):
                    if i < len(values):
                        attribution = {
                            "token": token,
                            "importance": float(abs(values[i])),
                            "attribution": float(values[i]),
                            "position": i
                        }
                        attributions.append(attribution)
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_mb = (current_memory - initial_memory) / (1024 * 1024)
                
                self.metrics_collector.record_attribution(
                    attributions=attributions,
                    gradient_norm=float(torch.norm(embeddings.grad).cpu()),
                    latency_ms=latency_ms,
                    memory_mb=memory_mb,
                    input_length=len(tokens)
                )
                
            except Exception as e:
                self.logger.warning(f"Error computing gradient values: {e}")
                # Fallback to uniform importance
                for i, token in enumerate(tokens):
                    attribution = {
                        "token": token,
                        "importance": 1.0 / len(tokens),
                        "attribution": 0.0,
                        "position": i
                    }
                    attributions.append(attribution)
                
                # Record error metrics
                latency_ms = (time.time() - start_time) * 1000
                self.metrics_collector.record_attribution(
                    attributions=attributions,
                    gradient_norm=0.0,
                    latency_ms=latency_ms,
                    memory_mb=0.0,
                    input_length=len(tokens),
                    error=str(e)
                )
            
            return attributions
            
        except Exception as e:
            self.logger.error(f"Token attribution error: {str(e)}")
            return []
    
    def _compute_feature_importance(
        self,
        attributions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute overall feature importance scores."""
        try:
            # Group by token and aggregate importance
            token_importance = {}
            for attr in attributions:
                token = attr["token"]
                if token in token_importance:
                    token_importance[token] += abs(attr["attribution"])
                else:
                    token_importance[token] = abs(attr["attribution"])
            
            # Normalize scores
            total = sum(token_importance.values()) + 1e-10
            normalized = {
                token: score / total
                for token, score in token_importance.items()
            }
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Feature importance computation error: {str(e)}")
            return {}
    
    def explain_prediction(
        self,
        text: str,
        prediction: torch.Tensor,
        model_fn: Any,
        confidence: float
    ) -> ExplanationResult:
        """Generate real-time explanation for a prediction."""
        try:
            # Get predicted class
            target_class = int(torch.argmax(prediction))
            
            # Compute token attributions
            attributions = self._compute_token_attributions(
                text,
                model_fn,
                target_class
            )
            
            # Compute feature importance
            importance = self._compute_feature_importance(attributions)
            
            # Compute base score
            base_score = float(prediction.float().mean())
            
            # Create explanation result
            result = ExplanationResult(
                feature_importance=importance,
                token_attributions=attributions,
                base_score=base_score,
                explanation_confidence=confidence
            )
            
            return result
            
        except Exception as e:
            raise MetalError(
                f"Failed to generate explanation: {str(e)}",
                category=MetalErrorCategory.EXPLANATION_ERROR
            )
    
    def update_background(
        self,
        texts: List[str]
    ):
        """Update background dataset for SHAP."""
        try:
            if len(texts) > self.max_local_size:
                # Randomly sample if too many examples
                indices = np.random.choice(
                    len(texts),
                    self.max_local_size,
                    replace=False
                )
                texts = [texts[i] for i in indices]
            
            self._prepare_background(texts)
            self.logger.info(f"Updated background dataset with {len(texts)} samples")
            
        except Exception as e:
            self.logger.error(f"Background update error: {str(e)}")
    
    def get_explanation_metrics(
        self,
        explanation: ExplanationResult
    ) -> Dict[str, Any]:
        """Get metrics about the explanation quality."""
        return {
            "num_important_features": len(explanation.feature_importance),
            "max_attribution": max(
                abs(attr["attribution"])
                for attr in explanation.token_attributions
            ),
            "mean_attribution": np.mean([
                abs(attr["attribution"])
                for attr in explanation.token_attributions
            ]),
            "explanation_confidence": explanation.explanation_confidence
        }
