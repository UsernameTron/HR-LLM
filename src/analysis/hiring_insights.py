"""
Hiring insights analysis module for extracting detailed metrics from job postings.
"""
import re
from typing import Dict, List, Optional, Tuple

import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

# Common departments in tech companies
DEPARTMENTS = {
    'engineering': ['software', 'developer', 'engineer', 'technical', 'devops'],
    'data': ['data scientist', 'machine learning', 'ai', 'analytics'],
    'product': ['product manager', 'product owner', 'program manager'],
    'design': ['designer', 'ux', 'ui', 'user experience'],
    'marketing': ['marketing', 'growth', 'seo', 'content'],
    'sales': ['sales', 'account executive', 'business development'],
    'hr': ['hr', 'human resources', 'recruiter', 'talent']
}

# Seniority levels
SENIORITY_LEVELS = {
    'entry': ['junior', 'entry level', 'associate', 'intern'],
    'mid': ['mid level', 'intermediate', 'experienced'],
    'senior': ['senior', 'lead', 'principal', 'staff'],
    'management': ['manager', 'director', 'head of', 'vp', 'chief']
}

# Common tech skills
TECH_SKILLS = {
    'languages': ['python', 'java', 'javascript', 'typescript', 'go', 'rust'],
    'frontend': ['react', 'angular', 'vue', 'html', 'css'],
    'backend': ['django', 'flask', 'spring', 'node.js', 'express'],
    'data': ['sql', 'pandas', 'tensorflow', 'pytorch', 'scikit-learn'],
    'devops': ['docker', 'kubernetes', 'aws', 'gcp', 'azure'],
    'tools': ['git', 'jira', 'confluence', 'slack', 'notion']
}

def extract_department(text: str) -> Optional[str]:
    """Extract department from job posting text."""
    text_lower = text.lower()
    for dept, keywords in DEPARTMENTS.items():
        if any(keyword in text_lower for keyword in keywords):
            return dept
    return None

def extract_seniority(text: str) -> Optional[str]:
    """Extract seniority level from job posting text."""
    text_lower = text.lower()
    for level, keywords in SENIORITY_LEVELS.items():
        if any(keyword in text_lower for keyword in keywords):
            return level
    return None

def extract_skills(text: str) -> List[str]:
    """Extract required skills from job posting text."""
    text_lower = text.lower()
    skills = []
    for category, skill_list in TECH_SKILLS.items():
        for skill in skill_list:
            if skill in text_lower:
                skills.append(skill)
    return skills

def calculate_complexity(text: str) -> float:
    """Calculate readability score of the job posting."""
    return textstat.flesch_reading_ease(text) / 100.0

def predict_response_rate(text: str) -> float:
    """Predict candidate response rate based on posting content."""
    # Simplified scoring based on key factors
    score = 0.5  # Base score
    
    # Length factor (prefer 300-800 words)
    words = len(text.split())
    if 300 <= words <= 800:
        score += 0.2
    elif words > 1000:
        score -= 0.1
    
    # Readability factor
    readability = calculate_complexity(text)
    if readability > 0.7:
        score += 0.1
    
    # Keyword density
    keyword_density = len(extract_skills(text)) / words
    if 0.02 <= keyword_density <= 0.05:
        score += 0.2
    
    return min(max(score, 0.0), 1.0)

def detect_bias(text: str) -> Dict[str, float]:
    """Detect potential biases in job posting."""
    text_lower = text.lower()
    
    biases = {
        'gender': 0.0,
        'age': 0.0,
        'culture': 0.0
    }
    
    # Gender bias keywords
    gender_terms = ['he', 'she', 'his', 'her', 'himself', 'herself']
    biases['gender'] = sum(text_lower.count(term) for term in gender_terms) / len(text.split())
    
    # Age bias keywords
    age_terms = ['young', 'energetic', 'fresh', 'mature', 'experienced']
    biases['age'] = sum(text_lower.count(term) for term in age_terms) / len(text.split())
    
    # Cultural bias keywords
    culture_terms = ['cultural fit', 'culture fit', 'like-minded']
    biases['culture'] = sum(text_lower.count(term) for term in culture_terms) / len(text.split())
    
    return biases

def analyze_keywords(text: str) -> Dict[str, float]:
    """Analyze effectiveness of keywords in the posting."""
    words = text.lower().split()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Get top keywords by TF-IDF score
    keyword_scores = {}
    for keyword, score in zip(feature_names, scores):
        if len(keyword) > 3:  # Filter out short words
            keyword_scores[keyword] = float(score)
    
    # Return top 10 keywords
    return dict(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:10])

def analyze_market_competitiveness(text: str) -> float:
    """Analyze market competitiveness of the job posting."""
    score = 0.5  # Base score
    
    # Skills factor
    skills = extract_skills(text)
    if len(skills) >= 5:
        score += 0.2
    
    # Seniority factor
    seniority = extract_seniority(text)
    if seniority in ['senior', 'management']:
        score += 0.1
    
    # Complexity factor
    complexity = calculate_complexity(text)
    if complexity > 0.6:
        score += 0.1
    
    # Department factor
    department = extract_department(text)
    if department in ['engineering', 'data']:
        score += 0.1
    
    return min(max(score, 0.0), 1.0)
