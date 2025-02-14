"""
Job content analyzer using MiniLM model specifically tuned for job content.
Includes pattern recognition, trend analysis, and confidence scoring.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class JobAnalyzer:
    def __init__(self):
        # Setup cache directory first
        self.cache_dir = Path(os.path.expanduser('~/.cache/hiring-sentiment-tracker'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device and model settings
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model_name = "microsoft/MiniLM-L12-H384-uncased"
        self.model_cache_file = self.cache_dir / 'model.pt'
        self.tokenizer_cache_dir = self.cache_dir / 'tokenizer'
        self.tokenizer = None
        self.model = None
        
        # Initialize pattern recognition components
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.pattern_weights = {}
        self.historical_trends = pd.DataFrame()
        
        # Signal patterns to look for with categories and weights
        self.signal_patterns = {
            'growth': {
                'patterns': [
                    'expanding', 'growing', 'scaling', 'new office',
                    'multiple positions', 'rapid growth', 'series',
                    'hiring spree', 'expansion', 'opening new',
                    'additional teams', 'building out', 'growth phase'
                ],
                'weight': 1.0,
                'fuzzy_threshold': 70  # Lower threshold for better matching
            },
            'urgency': {
                'patterns': [
                    'immediate', 'urgent', 'asap', 'fast-track',
                    'quick start', 'start immediately', 'urgent need',
                    'critical role', 'key position', 'time sensitive',
                    'priority hire', 'actively interviewing', 'time-sensitive'
                ],
                'weight': 1.2,  # Increased weight for urgency
                'fuzzy_threshold': 70  # Lower threshold
            },
            'stability': {
                'patterns': [
                    'established', 'funded', 'stable', 'profitable',
                    'market leader', 'industry leader', 'well funded',
                    'series a', 'series b', 'series c', 'public company',
                    'fortune 500', 'industry expert', 'well-funded'
                ],
                'weight': 1.0,
                'fuzzy_threshold': 70
            },
            'benefits': {
                'patterns': [
                    'competitive salary', 'equity', 'stock options',
                    'health insurance', '401k', 'flexible hours',
                    'remote work', 'work from home', 'bonus',
                    'comprehensive benefits', 'paid time off',
                    'compensation', 'health benefits'
                ],
                'weight': 0.9,
                'fuzzy_threshold': 70
            },
            'tech_stack': {
                'patterns': [
                    'modern stack', 'cutting edge', 'latest technology',
                    'cloud native', 'microservices', 'containerized',
                    'ai/ml', 'machine learning', 'deep learning',
                    'blockchain', 'serverless', 'kubernetes',
                    'cutting-edge', 'AI', 'ML'
                ],
                'weight': 1.0,
                'fuzzy_threshold': 70
            },
            'culture': {
                'patterns': [
                    'inclusive', 'diverse', 'collaborative',
                    'innovative', 'fast-paced', 'startup culture',
                    'work-life balance', 'mentorship', 'career growth',
                    'learning opportunities', 'team player',
                    'work/life balance', 'strong focus'
                ],
                'weight': 0.8,
                'fuzzy_threshold': 70
            }
        }
        
        # Cache for fuzzy pattern matches
        self._pattern_cache = {}
        
        # Initialize pattern cache
        self._pattern_cache = {}
        
        # Try to load the model with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} to load model")
                
                # Try to load from cache first
                if self.model_cache_file.exists() and self.tokenizer_cache_dir.exists():
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.tokenizer_cache_dir,
                            local_files_only=True
                        )
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            self.model_name
                        )
                        self.model.load_state_dict(torch.load(self.model_cache_file))
                        self.model.to(self.device)
                        logger.info("Successfully loaded model and tokenizer from cache")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load from cache: {str(e)}")
                
                # Download and cache if not found
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    use_auth_token=None,
                    revision="main",
                    timeout=60
                )
                self.tokenizer.save_pretrained(self.tokenizer_cache_dir)
                logger.info("Successfully loaded and cached tokenizer")
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    use_auth_token=None,
                    revision="main"
                )
                self.model.to(self.device)
                torch.save(self.model.state_dict(), self.model_cache_file)
                logger.info("Successfully loaded and cached model")
                break
                
            except Exception as e:
                logger.error(f"Failed to load model on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
    
    @lru_cache(maxsize=1000)
    def _fuzzy_match_pattern(self, text: str, pattern: str, threshold: int) -> float:
        """
        Perform fuzzy matching between text and pattern.
        Results are cached for performance.
        
        Args:
            text: Text to search in
            pattern: Pattern to match
            threshold: Minimum similarity score (0-100)
            
        Returns:
            Match score between 0 and 1
        """
        # Check for exact match first
        if pattern in text:
            return 1.0
            
        # Use multiple fuzzy matching algorithms for better accuracy
        token_sort_score = fuzz.token_sort_ratio(text, pattern)
        token_set_score = fuzz.token_set_ratio(text, pattern)
        partial_score = fuzz.partial_ratio(text, pattern)
        
        # Weight the different scores
        weighted_score = (
            0.4 * token_sort_score +  # Good for word order variations
            0.4 * token_set_score +   # Good for subset matching
            0.2 * partial_score       # Good for partial matches
        ) / 100.0  # Normalize to 0-1
        
        # Apply threshold as a soft cutoff
        return weighted_score if weighted_score >= (threshold / 100.0) else 0.0
    
    def _detect_patterns(self, text: str) -> Dict[str, float]:
        """
        Detect predefined patterns in the job posting text using fuzzy matching.
        
        Args:
            text: The job posting text to analyze
            
        Returns:
            Dict containing weighted pattern scores for each category
        """
        pattern_scores = {}
        text_lower = text.lower()
        
        # Process each category
        for category, config in self.signal_patterns.items():
            patterns = config['patterns']
            threshold = config.get('fuzzy_threshold', 75)  # Lower default threshold
            weight = config['weight']
            
            # Get match scores for each pattern
            pattern_matches = [
                self._fuzzy_match_pattern(text_lower, pattern.lower(), threshold)
                for pattern in patterns
            ]
            
            # Take top 3 matches to avoid dilution from many weak matches
            top_matches = sorted(pattern_matches, reverse=True)[:3]
            
            if top_matches:
                # Calculate weighted average with diminishing returns
                weighted_sum = sum(score * (0.7 ** i) for i, score in enumerate(top_matches))
                denominator = sum(0.7 ** i for i in range(len(top_matches)))
                raw_score = weighted_sum / denominator
                
                # Apply sigmoid normalization for smoother scoring
                normalized_score = 1 / (1 + np.exp(-5 * (raw_score - 0.5)))
                pattern_scores[category] = normalized_score * weight
            else:
                pattern_scores[category] = 0.0
            
        return pattern_scores
    
    def _analyze_historical_trend(self, company: str, industry: str) -> Tuple[float, float]:
        """
        Analyze historical hiring trends for the company and industry.
        
        Args:
            company: Company name
            industry: Industry name
            
        Returns:
            Tuple of (company_trend_score, industry_trend_score)
        """
        try:
            # Get last 90 days of data
            cutoff_date = datetime.now() - timedelta(days=90)
            recent_data = self.historical_trends[self.historical_trends['date'] >= cutoff_date]
            
            # Calculate company trend
            company_data = recent_data[recent_data['company'] == company]
            company_trend = self._calculate_trend_score(company_data['hiring_rate'])
            
            # Calculate industry trend
            industry_data = recent_data[recent_data['industry'] == industry]
            industry_trend = self._calculate_trend_score(industry_data['hiring_rate'])
            
            return company_trend, industry_trend
            
        except Exception as e:
            logger.warning(f"Error calculating trends: {str(e)}")
            return 0.5, 0.5
    
    def _calculate_trend_score(self, series: pd.Series) -> float:
        """
        Calculate trend score from a time series.
        
        Args:
            series: Pandas series of hiring rates
            
        Returns:
            Trend score between 0 and 1
        """
        if len(series) < 2:
            return 0.5
            
        # Calculate slope of trend line
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        # Normalize slope to 0-1 range
        return 1 / (1 + np.exp(-10 * slope))  # Sigmoid function
    
    async def analyze_job_post(self, text: str, company: str = None, industry: str = None) -> Dict:
        """
        Analyze job posting text for hiring signals and sentiment.
        
        Args:
            text: The job posting text to analyze
            company: Optional company name for trend analysis
            industry: Optional industry name for trend analysis
            
        Returns:
            Dict containing:
                - hiring_signal: float between 0-1 indicating hiring signal strength
                - sentiment: Dict with positive/neutral/negative scores
                - confidence: Model confidence score
                - patterns: Dict of detected pattern scores
                - trends: Dict of historical trend scores
        """
        try:
            # Get base sentiment analysis
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get pattern scores
            pattern_scores = self._detect_patterns(text)
            
            # Get trend scores if company and industry provided
            company_trend, industry_trend = (0.5, 0.5)
            if company and industry:
                company_trend, industry_trend = self._analyze_historical_trend(company, industry)
            
            # The model outputs 2 classes: [negative, positive]
            base_signal = float(probs[0][1])  # Positive class probability
            
            # Calculate weighted pattern signal
            pattern_values = list(pattern_scores.values())
            pattern_signal = np.mean(pattern_values) if pattern_values else 0.0
            
            # Calculate trend signal
            trend_signal = (company_trend + industry_trend) / 2
            
            # Combine signals with adjusted weights
            # Increased weight for patterns since we have improved fuzzy matching
            # Reduced weight for base model since it's not fine-tuned
            hiring_signal = (
                0.5 * pattern_signal +  # Increased pattern weight
                0.3 * base_signal +    # Reduced model weight
                0.2 * trend_signal     # Maintained trend weight
            )
            
            # Calculate confidence based on signal agreement and pattern diversity
            pattern_diversity = len([v for v in pattern_values if v > 0.3]) / len(self.signal_patterns)
            signals = [base_signal, pattern_signal, trend_signal]
            signal_agreement = 1 - np.std(signals)  # Higher agreement = lower std dev
            confidence = 0.7 * signal_agreement + 0.3 * pattern_diversity  # Combine both factors
            
            # Map model's binary output to sentiment scores
            sentiment_scores = {
                "positive": float(probs[0][1]),  # Positive class
                "neutral": 0.0,  # Model doesn't have neutral class
                "negative": float(probs[0][0])  # Negative class
            }
            
            return {
                "hiring_signal": float(hiring_signal),
                "sentiment": sentiment_scores,
                "confidence": float(confidence),
                "patterns": pattern_scores,
                "trends": {
                    "company": float(company_trend),
                    "industry": float(industry_trend)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing job post: {str(e)}")
            raise
