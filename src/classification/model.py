"""Transformer-based classification system with domain adaptation."""
import torch
import logging
import shap
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.metal_error_handler import MetalError, MetalErrorCategory

class SentimentClassifier:
    """Multi-label sentiment classifier with domain adaptation."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        device: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Set device with Metal optimization priority
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            self.model.to(self.device)
            self.logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            raise MetalError(
                f"Failed to initialize model: {str(e)}",
                category=MetalErrorCategory.MODEL_ERROR
            )
    
    def _preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess texts for model input."""
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            raise MetalError(
                f"Preprocessing error: {str(e)}",
                category=MetalErrorCategory.MODEL_ERROR
            )
    
    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Get model embeddings for texts."""
        try:
            inputs = self._preprocess(texts)
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Use last hidden state as embedding
            embeddings = outputs.hidden_states[-1][:, 0, :]  # [CLS] token
            return embeddings.cpu()
            
        except Exception as e:
            raise MetalError(
                f"Embedding extraction error: {str(e)}",
                category=MetalErrorCategory.MODEL_ERROR
            )
    
    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Perform multi-label classification."""
        try:
            inputs = self._preprocess(texts)
            outputs = self.model(**inputs)
            
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int()
            
            result = {
                "predictions": predictions.cpu(),
                "probabilities": probs.cpu() if return_probabilities else None
            }
            
            return result
            
        except Exception as e:
            raise MetalError(
                f"Prediction error: {str(e)}",
                category=MetalErrorCategory.MODEL_ERROR
            )
    
    def explain(
        self,
        text: str,
        method: str = "shap",
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Generate explanations for model predictions."""
        try:
            if method.lower() == "shap":
                # Create explainer
                explainer = shap.Explainer(
                    lambda x: self.predict([x])["probabilities"].numpy(),
                    shap.maskers.Text(self.tokenizer)
                )
                
                # Generate explanation
                shap_values = explainer([text], max_evals=num_samples)
                
                return {
                    "method": "shap",
                    "values": shap_values.values,
                    "base_values": shap_values.base_values,
                    "data": shap_values.data
                }
            else:
                raise ValueError(f"Unsupported explanation method: {method}")
                
        except Exception as e:
            raise MetalError(
                f"Explanation error: {str(e)}",
                category=MetalErrorCategory.MODEL_ERROR
            )
    
    def adapt_domain(
        self,
        texts: List[str],
        labels: torch.Tensor,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """Perform domain adaptation fine-tuning."""
        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
            
            self.model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_labels = labels[i:i + batch_size].to(self.device)
                    
                    inputs = self._preprocess(batch_texts)
                    outputs = self.model(**inputs, labels=batch_labels)
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                avg_loss = total_loss / (len(texts) / batch_size)
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            self.model.eval()
            
        except Exception as e:
            raise MetalError(
                f"Domain adaptation error: {str(e)}",
                category=MetalErrorCategory.MODEL_ERROR
            )
