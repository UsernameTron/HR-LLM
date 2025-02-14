"""
Sentiment analysis model implementation using transformers.
"""
import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU for now
        logger.info(f"Using device: {self.device}")
        
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.labels = ["negative", "neutral", "positive"]
        
        # Try to load the model with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} to load model")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    use_auth_token=None,
                    revision="main",
                    timeout=60
                )
                logger.info("Successfully loaded tokenizer")
                
                # Load model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    use_auth_token=None,
                    revision="main"
                )
                self.model.to(self.device)
                logger.info("Successfully loaded model")
                
                # If we get here, both loaded successfully
                break
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to load model after {max_retries} attempts: {str(e)}")
                import time
                time.sleep(5)  # Wait before retrying
    
    async def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of a given text."""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                
            # Convert to dict with compound score
            scores_dict = {
                label: score.item()
                for label, score in zip(self.labels, scores[0])
            }
            
            # Calculate compound score as weighted sum
            compound = (scores_dict['positive'] - scores_dict['negative']) / \
                      (scores_dict['positive'] + scores_dict['negative'] + scores_dict['neutral'])
            
            scores_dict['compound'] = compound
            return scores_dict
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for a batch of texts."""
        try:
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Convert to list of dicts with compound scores
            results = []
            for i in range(len(texts)):
                scores_dict = {label: score.item() for label, score in zip(self.labels, scores[i])}
                # Calculate compound score
                compound = (scores_dict['positive'] - scores_dict['negative']) / \
                          (scores_dict['positive'] + scores_dict['negative'] + scores_dict['neutral'])
                scores_dict['compound'] = compound
                results.append(scores_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {str(e)}")
            raise
