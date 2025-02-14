"""
Text analysis utilities for extracting structured information from job postings.
"""
import re
from typing import List, Optional

def extract_department(text: str) -> Optional[str]:
    """Extract department information from job posting text."""
    # Common department indicators
    dept_patterns = [
        r"Department:\s*([A-Za-z\s&]+)",
        r"Team:\s*([A-Za-z\s&]+)\s+Department",
        r"([A-Za-z\s&]+)\s+Department"
    ]
    
    # Common department names
    departments = {
        'Engineering': ['Engineering', 'Software', 'Development', 'Tech'],
        'Sales': ['Sales', 'Business Development', 'Revenue'],
        'Marketing': ['Marketing', 'Growth', 'Brand'],
        'HR': ['HR', 'Human Resources', 'People', 'Talent'],
        'Finance': ['Finance', 'Accounting', 'Treasury'],
        'Operations': ['Operations', 'Ops', 'Supply Chain'],
        'Product': ['Product', 'Product Management'],
        'Design': ['Design', 'UX', 'UI', 'User Experience'],
        'Research': ['Research', 'R&D', 'Data Science'],
        'Legal': ['Legal', 'Compliance', 'Regulatory']
    }
    
    # Try explicit department patterns first
    for pattern in dept_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dept = match.group(1).strip()
            # Map to standardized department name
            for std_dept, variants in departments.items():
                if any(variant.lower() in dept.lower() for variant in variants):
                    return std_dept
            return dept
    
    # Fallback to scanning for department names in text
    text_lower = text.lower()
    for std_dept, variants in departments.items():
        if any(variant.lower() in text_lower for variant in variants):
            return std_dept
    
    return None

def extract_seniority(text: str) -> Optional[str]:
    """Extract seniority level from job posting text."""
    seniority_levels = {
        'Entry': ['entry level', 'junior', 'associate', 'apprentice'],
        'Mid': ['mid level', 'intermediate', 'experienced'],
        'Senior': ['senior', 'sr', 'lead', 'principal'],
        'Manager': ['manager', 'head of', 'director', 'vp', 'chief']
    }
    
    text_lower = text.lower()
    for level, indicators in seniority_levels.items():
        if any(indicator in text_lower for indicator in indicators):
            return level
    
    return None

def extract_skills(text: str) -> List[str]:
    """Extract technical skills and requirements from job posting text."""
    # Common skill patterns
    skill_patterns = [
        r"Required skills:\s*([^.]+)",
        r"Requirements:\s*([^.]+)",
        r"Skills:\s*([^.]+)",
        r"Technologies:\s*([^.]+)"
    ]
    
    # Common technical skills to look for
    common_skills = {
        'Languages': ['Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'Go', 'Rust'],
        'Web': ['React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask'],
        'Data': ['SQL', 'MongoDB', 'PostgreSQL', 'Redis', 'Elasticsearch'],
        'Cloud': ['AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker'],
        'ML/AI': ['TensorFlow', 'PyTorch', 'scikit-learn', 'NLP', 'Computer Vision']
    }
    
    found_skills = set()
    
    # Try explicit skill sections first
    for pattern in skill_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills_text = match.group(1)
            # Look for common skills in the skills section
            for category, skills in common_skills.items():
                for skill in skills:
                    if re.search(rf'\b{skill}\b', skills_text, re.IGNORECASE):
                        found_skills.add(skill)
    
    # Also scan entire text for skills
    for category, skills in common_skills.items():
        for skill in skills:
            if re.search(rf'\b{skill}\b', text, re.IGNORECASE):
                found_skills.add(skill)
    
    return sorted(list(found_skills))

def analyze_market_competitiveness(text: str) -> float:
    """Analyze market competitiveness based on job posting text.
    Returns a score from 0 to 1, where higher values indicate more competitive positions.
    """
    competitiveness_factors = {
        'skills_required': 0.3,  # Weight for number of required skills
        'experience_level': 0.3,  # Weight for seniority/experience requirements
        'benefits': 0.2,         # Weight for benefits/perks mentioned
        'company_profile': 0.2   # Weight for company size/status indicators
    }
    
    score = 0.0
    
    # Analyze required skills
    skills = extract_skills(text)
    skills_score = min(len(skills) / 5, 1.0)  # Normalize to max of 5 skills
    score += skills_score * competitiveness_factors['skills_required']
    
    # Analyze experience level
    seniority = extract_seniority(text)
    experience_scores = {
        'Entry': 0.3,
        'Mid': 0.6,
        'Senior': 0.9,
        'Manager': 1.0
    }
    if seniority:
        score += experience_scores.get(seniority, 0.5) * competitiveness_factors['experience_level']
    
    # Analyze benefits
    benefit_keywords = ['competitive salary', 'equity', 'stock', 'bonus', 'benefits',
                       'insurance', '401k', 'flexible', 'remote']
    benefits_mentioned = sum(1 for keyword in benefit_keywords 
                           if keyword in text.lower())
    benefits_score = min(benefits_mentioned / len(benefit_keywords), 1.0)
    score += benefits_score * competitiveness_factors['benefits']
    
    # Analyze company profile
    company_keywords = ['leader', 'best', 'top', 'growing', 'innovative',
                       'funded', 'startup', 'enterprise', 'global']
    company_mentions = sum(1 for keyword in company_keywords 
                         if keyword in text.lower())
    company_score = min(company_mentions / len(company_keywords), 1.0)
    score += company_score * competitiveness_factors['company_profile']
    
    return score
