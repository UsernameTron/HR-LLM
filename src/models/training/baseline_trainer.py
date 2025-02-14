"""
Training pipeline for baseline hiring prediction models using Glassdoor data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json

from src.data.processors.glassdoor_processor import GlassdoorProcessor, CompanyProfile

# Configure logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Trains baseline models for hiring prediction using Glassdoor data."""
    
    def __init__(self, 
                 model_dir: str = "models/baseline",
                 config_dir: str = "config"):
        """Initialize the trainer."""
        self.model_dir = Path(model_dir)
        self.config_dir = Path(config_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.glassdoor = GlassdoorProcessor()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Load or initialize signal weights
        self.signal_weights = self._load_signal_weights()
    
    def _load_signal_weights(self) -> Dict[str, float]:
        """Load or initialize signal weights configuration."""
        weights_path = self.config_dir / "signal_weights.json"
        
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                return json.load(f)
        
        # Initialize default weights
        default_weights = {
            "growth_score": 0.3,
            "sentiment_score": 0.2,
            "recent_job_count": 0.15,
            "company_rating": 0.15,
            "salary_percentile": 0.1,
            "review_count": 0.1
        }
        
        # Save default weights
        with open(weights_path, 'w') as f:
            json.dump(default_weights, f, indent=4)
        
        return default_weights
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from Glassdoor profiles."""
        features = []
        labels = []
        
        # Get unique companies
        companies = self.glassdoor.jobs_df['company'].unique()
        
        for company in companies:
            profile = self.glassdoor.get_company_profile(company)
            if profile is None:
                continue
            
            # Extract features
            feature_vector = [
                profile.growth_score,
                profile.sentiment_score,
                profile.recent_job_count / 100,  # Normalize
                profile.avg_rating / 5,  # Normalize
                profile.salary_percentile / 100000,  # Normalize
                min(profile.review_count / 1000, 1.0)  # Normalize with cap
            ]
            
            # Define positive class (high hiring probability) threshold
            is_hiring = profile.hiring_probability >= 0.7
            
            features.append(feature_vector)
            labels.append(1 if is_hiring else 0)
        
        return np.array(features), np.array(labels)
    
    def train_baseline_model(self) -> Dict[str, float]:
        """Train the baseline hiring prediction model."""
        logger.info("Preparing training data...")
        X, y = self._prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("Training baseline model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Get feature importance
        feature_names = [
            "growth_score",
            "sentiment_score",
            "recent_job_count",
            "company_rating",
            "salary_percentile",
            "review_count"
        ]
        
        importance_dict = dict(zip(
            feature_names,
            self.model.feature_importances_
        ))
        
        # Update signal weights based on feature importance
        total_importance = sum(importance_dict.values())
        self.signal_weights = {
            k: v / total_importance
            for k, v in importance_dict.items()
        }
        
        # Save updated weights
        with open(self.config_dir / "signal_weights.json", 'w') as f:
            json.dump(self.signal_weights, f, indent=4)
        
        # Save model and scaler
        joblib.dump(self.model, self.model_dir / "baseline_model.joblib")
        joblib.dump(self.scaler, self.model_dir / "feature_scaler.joblib")
        
        # Generate metrics
        metrics = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info("Model training completed!")
        logger.info(f"Model accuracy: {metrics['accuracy']:.3f}")
        logger.info("Updated feature importance weights saved")
        
        return metrics
    
    def predict_hiring_probability(self, company: str) -> Optional[float]:
        """Predict hiring probability for a company using the trained model."""
        profile = self.glassdoor.get_company_profile(company)
        if profile is None:
            return None
        
        # Extract features
        features = np.array([[
            profile.growth_score,
            profile.sentiment_score,
            profile.recent_job_count / 100,
            profile.avg_rating / 5,
            profile.salary_percentile / 100000,
            min(profile.review_count / 1000, 1.0)
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probability of positive class
        prob = self.model.predict_proba(features_scaled)[0][1]
        
        return prob
    
    def get_industry_insights(self) -> Dict[str, Dict]:
        """Get hiring insights by industry."""
        insights = {}
        benchmarks = self.glassdoor.get_industry_benchmarks()
        
        for industry, metrics in benchmarks.items():
            # Skip industries with insufficient data
            if metrics['company_count'] < 5:
                continue
                
            # Calculate average hiring probability for the industry
            probabilities = []
            companies = self.glassdoor.jobs_df[
                self.glassdoor.jobs_df['industry'] == industry
            ]['company'].unique()
            
            for company in companies[:50]:  # Limit to top 50 companies per industry
                prob = self.predict_hiring_probability(company)
                if prob is not None:
                    probabilities.append(prob)
            
            if probabilities:
                insights[industry] = {
                    'avg_hiring_probability': np.mean(probabilities),
                    'hiring_probability_std': np.std(probabilities),
                    'company_count': metrics['company_count'],
                    'avg_growth_score': metrics['avg_growth_score'],
                    'avg_sentiment_score': metrics['avg_sentiment_score']
                }
        
        return insights
