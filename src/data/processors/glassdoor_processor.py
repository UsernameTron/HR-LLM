"""
Glassdoor data processor for hiring signals analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CompanyProfile:
    """Company profile with aggregated metrics from Glassdoor data."""
    name: str
    industry: str
    sector: Optional[str]
    size: Optional[str]
    founded_year: Optional[int]
    avg_rating: float
    review_count: int
    recent_job_count: int
    salary_percentile: float
    growth_score: float
    sentiment_score: float
    hiring_probability: float

class GlassdoorProcessor:
    """Process Glassdoor data to extract hiring signals and patterns."""
    
    def __init__(self, data_dir: str = settings.DATA_DIR):
        """Initialize the processor with data directory."""
        self.data_dir = Path(data_dir)
        self.jobs_df = None
        self.reviews_df = None
        self._load_data()
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by removing BOM markers and standardizing format."""
        df.columns = df.columns.str.replace('ï»¿', '')
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df
    
    def _load_data(self):
        """Load Glassdoor datasets including both CSV and Excel files."""
        try:
            # Load main CSV files with different encodings
            encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Load job postings
                    self.jobs_df = pd.read_csv(
                        self.data_dir / "Glassdoor_Job_Postings.csv",
                        encoding=encoding,
                        on_bad_lines='skip'
                    )
                    self.jobs_df = self._clean_column_names(self.jobs_df)
                    
                    # Load reviews
                    self.reviews_df = pd.read_csv(
                        self.data_dir / "glassdoor_reviews.csv",
                        encoding=encoding,
                        on_bad_lines='skip'
                    )
                    self.reviews_df = self._clean_column_names(self.reviews_df)
                    
                    # Load company data
                    self.company_df = pd.read_csv(
                        self.data_dir / "glassdoor_comany.csv",
                        encoding=encoding,
                        on_bad_lines='skip'
                    )
                    self.company_df = self._clean_column_names(self.company_df)
                    
                    break  # If successful, exit the encoding loop
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Failed to load CSV files with any encoding")
            
            # Load industry-specific Excel files
            self.industry_data = {}
            excel_files = list(self.data_dir.glob("*.xlsx"))
            
            for excel_file in excel_files:
                try:
                    # Convert "Companies Overview" to "companies_overview"
                    industry_name = excel_file.stem.lower()
                    industry_name = industry_name.replace(' ', '_')
                    industry_name = industry_name.replace('-', '_')
                    
                    # Read all sheets in the Excel file
                    excel_data = pd.read_excel(excel_file, sheet_name=None)
                    
                    # Clean column names in each sheet
                    for sheet_name in excel_data:
                        excel_data[sheet_name] = self._clean_column_names(excel_data[sheet_name])
                    
                    self.industry_data[industry_name] = excel_data
                    
                except Exception as e:
                    logger.warning(f"Failed to load {excel_file.name}: {str(e)}")
                    continue
            
            # Convert date columns
            try:
                self.reviews_df['date_review'] = pd.to_datetime(
                    self.reviews_df['date_review'],
                    errors='coerce'
                )
            except Exception as e:
                logger.warning(f"Error converting dates: {str(e)}")
            
            # Log loading status
            logger.info(f"Loaded {len(self.jobs_df)} job postings and {len(self.reviews_df)} reviews")
            logger.info(f"Loaded {len(self.industry_data)} industry-specific Excel files")
            for industry, data in self.industry_data.items():
                logger.info(f"  - {industry}: {len(data)} sheets")
                
        except Exception as e:
            logger.error(f"Error loading Glassdoor data: {str(e)}")
            raise
    
    def _clean_salary(self, salary_str: str) -> float:
        """Clean salary string and convert to float."""
        try:
            if pd.isna(salary_str) or salary_str == 0:
                return 0.0
                
            # Convert to string and handle ranges
            clean_str = str(salary_str)
            if '-' in clean_str:
                clean_str = clean_str.split('-')[0]  # Take the lower bound
                
            # Handle K/k suffix
            multiplier = 1000.0 if any(x in clean_str.lower() for x in ['k', 'к']) else 1.0
            
            # Remove all non-digit characters except decimal point
            clean_str = ''.join(c for c in clean_str if c.isdigit() or c == '.')
            
            if not clean_str:
                return 0.0
                
            return float(clean_str) * multiplier
            
        except Exception as e:
            logger.warning(f"Error cleaning salary {salary_str}: {str(e)}")
            return 0.0

    def _calculate_growth_score(self, company: str) -> float:
        """Calculate company growth score based on job postings and reviews."""
        try:
            company_jobs = self.jobs_df[self.jobs_df['company'].str.lower() == company.lower()]
            company_reviews = self.reviews_df[self.reviews_df['firm'].str.lower() == company.lower()]
            
            if len(company_jobs) == 0 and len(company_reviews) == 0:
                return 0.0
            
            # Calculate job posting growth
            job_volume_score = min(len(company_jobs) / 100, 1.0)  # Normalize to max 1.0
            
            # Calculate rating score
            avg_rating = company_jobs['company_rating'].mean() if 'company_rating' in company_jobs.columns else 0
            rating_score = (avg_rating - 1) / 4 if avg_rating > 0 else 0  # Normalize 1-5 to 0-1
            
            # Recent review sentiment
            if len(company_reviews) > 0:
                recent_reviews = company_reviews.sort_values('date_review', ascending=False).head(10)
                recent_sentiment = recent_reviews['overall_rating'].mean() if 'overall_rating' in recent_reviews.columns else 0
                sentiment_score = (recent_sentiment - 1) / 4 if recent_sentiment > 0 else 0
            else:
                sentiment_score = 0.0
            
            # Combine scores with weights
            growth_score = (
                0.4 * job_volume_score +
                0.3 * rating_score +
                0.3 * sentiment_score
            )
            
            return min(max(growth_score, 0.0), 1.0)  # Ensure score is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating growth score for {company}: {str(e)}")
            return 0.0
    
    def _calculate_sentiment_score(self, company: str) -> float:
        """Calculate sentiment score from reviews."""
        company_reviews = self.reviews_df[self.reviews_df['firm'] == company]
        
        if len(company_reviews) == 0:
            return 0.0
        
        # Get recent reviews (last 6 months)
        recent_mask = company_reviews['date_review'] >= (
            company_reviews['date_review'].max() - pd.Timedelta(days=180)
        )
        recent_reviews = company_reviews[recent_mask]
        
        if len(recent_reviews) == 0:
            recent_reviews = company_reviews
        
        # Calculate weighted average of different rating components
        weights = {
            'overall_rating': 0.3,
            'culture_values': 0.15,
            'career_opp': 0.2,
            'comp_benefits': 0.15,
            'work_life_balance': 0.1,
            'senior_mgmt': 0.1
        }
        
        weighted_score = 0
        valid_weights_sum = 0
        
        for metric, weight in weights.items():
            if metric in recent_reviews.columns:
                avg_score = recent_reviews[metric].mean()
                if not pd.isna(avg_score):
                    weighted_score += (avg_score - 1) / 4 * weight  # Normalize to 0-1
                    valid_weights_sum += weight
        
        if valid_weights_sum == 0:
            return 0.0
            
        return weighted_score / valid_weights_sum
    
    def _calculate_hiring_probability(self, 
                                   growth_score: float, 
                                   sentiment_score: float,
                                   recent_job_count: int) -> float:
        """Calculate the probability of active hiring."""
        # Normalize job count to 0-1 scale
        job_score = min(recent_job_count / 50, 1.0)
        
        # Weighted combination of factors
        probability = (
            0.4 * growth_score +
            0.3 * sentiment_score +
            0.3 * job_score
        )
        
        return probability
    
    def get_company_profile(self, company: str) -> Optional[CompanyProfile]:
        """Generate a company profile with hiring signals."""
        try:
            # Case-insensitive company name matching
            company_jobs = self.jobs_df[self.jobs_df['company'].str.lower() == company.lower()]
            company_reviews = self.reviews_df[self.reviews_df['firm'].str.lower() == company.lower()]
            
            if len(company_jobs) == 0:
                logger.warning(f"No job data found for company: {company}")
                return None
            
            # Get the original company name from the data
            original_company_name = company_jobs['company'].iloc[0]
            
            # Calculate metrics
            growth_score = self._calculate_growth_score(original_company_name)
            sentiment_score = self._calculate_sentiment_score(original_company_name)
            recent_job_count = len(company_jobs)
            
            # Get company metadata from the most recent job posting
            latest_job = company_jobs.iloc[0]
            
            # Clean and process salary data
            salaries = company_jobs['salary_avg_estimate'].apply(self._clean_salary)
            salary_percentile = np.percentile(salaries[salaries > 0], 50) if len(salaries[salaries > 0]) > 0 else 0
            
            # Get average rating with fallback
            avg_rating = (
                company_jobs['company_rating'].mean() 
                if 'company_rating' in company_jobs.columns 
                else company_reviews['overall_rating'].mean() 
                if len(company_reviews) > 0 and 'overall_rating' in company_reviews.columns
                else 0.0
            )
            
            profile = CompanyProfile(
                name=original_company_name,  # Use original case from data
                industry=str(latest_job.get('industry', '')),
                sector=str(latest_job.get('sector', '')) if pd.notna(latest_job.get('sector')) else None,
                size=str(latest_job.get('company_size', '')) if pd.notna(latest_job.get('company_size')) else None,
                founded_year=int(latest_job['company_founded']) if pd.notna(latest_job.get('company_founded')) else None,
                avg_rating=float(avg_rating),
                review_count=len(company_reviews),
                recent_job_count=recent_job_count,
                salary_percentile=float(salary_percentile),
                growth_score=float(growth_score),
                sentiment_score=float(sentiment_score),
                hiring_probability=float(self._calculate_hiring_probability(
                    growth_score, 
                    sentiment_score,
                    recent_job_count
                ))
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating profile for {company}: {str(e)}")
            return None
    
    def get_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Calculate industry benchmarks for comparison."""
        benchmarks = {}
        
        for industry in self.jobs_df['industry'].unique():
            if pd.isna(industry) or not industry:
                continue
                
            industry_jobs = self.jobs_df[self.jobs_df['industry'] == industry]
            industry_companies = industry_jobs['company'].unique()
            
            # Clean and process salary data
            salaries = industry_jobs['salary_avg_estimate'].apply(self._clean_salary)
            
            # Calculate average rating with NaN handling
            avg_rating = 0.0
            if 'company_rating' in industry_jobs.columns:
                ratings = industry_jobs['company_rating'].dropna()
                avg_rating = float(ratings.mean()) if len(ratings) > 0 else 0.0
            
            # Calculate average salary with NaN handling
            avg_salary = float(salaries[salaries > 0].mean()) if len(salaries[salaries > 0]) > 0 else 0.0
            
            industry_metrics = {
                'avg_rating': avg_rating,
                'avg_salary': avg_salary,
                'company_count': int(len(industry_companies)),
                'job_count': int(len(industry_jobs))
            }
            
            # Calculate average growth and sentiment scores
            growth_scores = []
            sentiment_scores = []
            
            # Limit to top 100 companies per industry for performance
            for company in industry_companies[:100]:
                try:
                    growth_score = self._calculate_growth_score(company)
                    sentiment_score = self._calculate_sentiment_score(company)
                    
                    if not pd.isna(growth_score):
                        growth_scores.append(growth_score)
                    if not pd.isna(sentiment_score):
                        sentiment_scores.append(sentiment_score)
                        
                except Exception as e:
                    logger.warning(f"Error calculating scores for {company}: {str(e)}")
                    continue
            
            industry_metrics['avg_growth_score'] = float(np.mean(growth_scores)) if growth_scores else 0.0
            industry_metrics['avg_sentiment_score'] = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            
            benchmarks[industry] = industry_metrics
        
        return benchmarks
    
    def get_hiring_trends(self, company: str, days: int = 180) -> Dict[str, List]:
        """Get hiring trends over time for a company."""
        try:
            # Case-insensitive company name matching
            company_jobs = self.jobs_df[self.jobs_df['company'].str.lower() == company.lower()]
            company_reviews = self.reviews_df[self.reviews_df['firm'].str.lower() == company.lower()]
            
            if len(company_jobs) == 0 and len(company_reviews) == 0:
                logger.warning(f"No data found for company: {company}")
                return {}
            
            # Get date range
            if len(company_reviews) > 0:
                end_date = company_reviews['date_review'].max()
            else:
                end_date = pd.Timestamp.now()
                
            start_date = end_date - pd.Timedelta(days=days)
            
            # Create weekly date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='W')
            
            trends = {
                'dates': date_range.strftime('%Y-%m-%d').tolist(),
                'job_counts': [],
                'sentiment_scores': [],
                'growth_scores': []
            }
            
            # Calculate metrics for each week
            for date in date_range:
                week_end = date + pd.Timedelta(days=7)
                
                # Get jobs posted in this week
                week_jobs = company_jobs[
                    (company_jobs['date_posted'] >= date) &
                    (company_jobs['date_posted'] < week_end)
                ] if 'date_posted' in company_jobs.columns else pd.DataFrame()
                
                # Get reviews from this week
                week_reviews = company_reviews[
                    (company_reviews['date_review'] >= date) &
                    (company_reviews['date_review'] < week_end)
                ]
                
                # Calculate metrics
                job_count = len(week_jobs)
                
                sentiment_score = (
                    float(week_reviews['overall_rating'].mean())
                    if len(week_reviews) > 0 and 'overall_rating' in week_reviews.columns
                    else None
                )
                
                growth_score = self._calculate_growth_score(company)
                
                trends['job_counts'].append(job_count)
                trends['sentiment_scores'].append(sentiment_score)
                trends['growth_scores'].append(float(growth_score))
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting hiring trends for {company}: {str(e)}")
            return {}
