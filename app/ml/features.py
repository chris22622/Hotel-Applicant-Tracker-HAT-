"""
Feature extraction for ML-based resume ranking.
"""
import logging
import re
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from rapidfuzz import fuzz
from app.ml.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features for resume-job matching."""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.skill_keywords = self._load_skill_keywords()
        self.education_levels = {
            'phd': 5, 'doctorate': 5, 'doctoral': 5,
            'masters': 4, 'master': 4, 'mba': 4, 'ms': 4, 'ma': 4,
            'bachelor': 3, 'bachelors': 3, 'bs': 3, 'ba': 3, 'bsc': 3,
            'associate': 2, 'associates': 2, 'aa': 2, 'as': 2,
            'diploma': 1, 'certificate': 1, 'certification': 1,
            'high school': 0, 'hs': 0, 'secondary': 0
        }
    
    def extract_features(self, resume_text: str, job_description: str, 
                        resume_metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Extract feature vector from resume and job description.
        
        Args:
            resume_text: Full text of the resume
            job_description: Full text of the job description
            resume_metadata: Optional metadata (experience years, title, etc.)
            
        Returns:
            Fixed-length feature vector as numpy array
        """
        if resume_metadata is None:
            resume_metadata = {}
        
        features = []
        
        # 1. Semantic similarity (embedding cosine similarity)
        semantic_sim = self._compute_semantic_similarity(resume_text, job_description)
        features.append(semantic_sim)
        
        # 2. Skill overlap (Jaccard similarity)
        skill_overlap = self._compute_skill_overlap(resume_text, job_description)
        features.append(skill_overlap)
        
        # 3. Title similarity (fuzzy matching)
        title_sim = self._compute_title_similarity(resume_metadata, job_description)
        features.append(title_sim)
        
        # 4. Experience years gap penalty
        exp_penalty = self._compute_experience_penalty(resume_metadata, job_description)
        features.append(exp_penalty)
        
        # 5. Education level score
        edu_score = self._compute_education_score(resume_text)
        features.append(edu_score)
        
        # 6. Recency score (how recent is the experience)
        recency_score = self._compute_recency_score(resume_text, resume_metadata)
        features.append(recency_score)
        
        # 7. Resume length feature (normalized)
        length_feature = min(len(resume_text) / 5000, 1.0)  # Normalize to [0,1]
        features.append(length_feature)
        
        # 8. Keyword density features
        keyword_density = self._compute_keyword_density(resume_text, job_description)
        features.append(keyword_density)
        
        # 9. Experience diversity (different companies/roles)
        exp_diversity = self._compute_experience_diversity(resume_text)
        features.append(exp_diversity)
        
        # 10. Contact information completeness
        contact_completeness = self._compute_contact_completeness(resume_metadata)
        features.append(contact_completeness)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_semantic_similarity(self, resume_text: str, job_description: str) -> float:
        """Compute cosine similarity between resume and job embeddings."""
        try:
            resume_embedding = self.embedding_service.embed_single(resume_text)
            job_embedding = self.embedding_service.embed_single(job_description)
            
            if resume_embedding.size == 0 or job_embedding.size == 0:
                return 0.0
                
            similarity = self.embedding_service.cosine_similarity(resume_embedding, job_embedding)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _compute_skill_overlap(self, resume_text: str, job_description: str) -> float:
        """Compute Jaccard similarity of skills mentioned."""
        resume_skills = self._extract_skills(resume_text.lower())
        job_skills = self._extract_skills(job_description.lower())
        
        if not resume_skills and not job_skills:
            return 0.0
        
        intersection = len(resume_skills.intersection(job_skills))
        union = len(resume_skills.union(job_skills))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_title_similarity(self, resume_metadata: Dict, job_description: str) -> float:
        """Compute similarity between resume title and job title."""
        resume_title = resume_metadata.get('title', '')
        if not resume_title:
            # Try to extract title from text
            resume_title = resume_metadata.get('candidate_name', '')
        
        # Extract job title from description (first line or prominent text)
        job_lines = job_description.strip().split('\n')
        job_title = job_lines[0] if job_lines else job_description[:100]
        
        if not resume_title or not job_title:
            return 0.0
        
        # Use fuzzy string matching
        similarity = fuzz.ratio(resume_title.lower(), job_title.lower()) / 100.0
        return similarity
    
    def _compute_experience_penalty(self, resume_metadata: Dict, job_description: str) -> float:
        """Compute penalty based on experience gap."""
        resume_years = resume_metadata.get('experience_years', 0)
        if isinstance(resume_years, str):
            resume_years = self._extract_years_from_text(resume_years)
        
        # Extract required years from job description
        required_years = self._extract_required_experience(job_description)
        
        if required_years == 0:
            return 1.0  # No penalty if no requirement specified
        
        if resume_years >= required_years:
            return 1.0  # No penalty if meets requirement
        
        # Apply penalty for missing experience
        gap = required_years - resume_years
        penalty = max(0.0, 1.0 - (gap / required_years))
        return penalty
    
    def _compute_education_score(self, resume_text: str) -> float:
        """Compute education level score."""
        text_lower = resume_text.lower()
        max_score = 0
        
        for education, score in self.education_levels.items():
            if education in text_lower:
                max_score = max(max_score, score)
        
        # Normalize to [0,1]
        return max_score / 5.0
    
    def _compute_recency_score(self, resume_text: str, resume_metadata: Dict) -> float:
        """Compute how recent the experience is."""
        current_year = datetime.now().year
        
        # Extract years from resume
        years = re.findall(r'\b(19|20)\d{2}\b', resume_text)
        if not years:
            return 0.5  # Default if no years found
        
        # Find most recent year
        recent_year = max(int(year) for year in years)
        
        # Compute recency score (more recent = higher score)
        age = current_year - recent_year
        if age == 0:
            return 1.0
        elif age <= 2:
            return 0.9
        elif age <= 5:
            return 0.7
        elif age <= 10:
            return 0.5
        else:
            return 0.2
    
    def _compute_keyword_density(self, resume_text: str, job_description: str) -> float:
        """Compute density of job keywords in resume."""
        job_keywords = self._extract_important_keywords(job_description)
        if not job_keywords:
            return 0.0
        
        resume_lower = resume_text.lower()
        matches = sum(1 for keyword in job_keywords if keyword in resume_lower)
        
        return matches / len(job_keywords)
    
    def _compute_experience_diversity(self, resume_text: str) -> float:
        """Compute diversity of experience (different companies/roles)."""
        # Simple heuristic: count unique company indicators
        company_indicators = re.findall(r'\b(?:inc|corp|llc|ltd|company|corporation)\b', 
                                       resume_text.lower())
        
        # Count role transitions (years with different contexts)
        years = re.findall(r'\b(19|20)\d{2}\b', resume_text)
        unique_years = len(set(years))
        
        # Combine indicators
        diversity = (len(set(company_indicators)) * 0.5 + unique_years * 0.1)
        return min(diversity, 1.0)  # Cap at 1.0
    
    def _compute_contact_completeness(self, resume_metadata: Dict) -> float:
        """Compute completeness of contact information."""
        fields = ['email', 'phone', 'location', 'candidate_name']
        present = sum(1 for field in fields if resume_metadata.get(field))
        return present / len(fields)
    
    def _extract_skills(self, text: str) -> set:
        """Extract skills from text."""
        found_skills = set()
        
        for skill in self.skill_keywords:
            if skill in text:
                found_skills.add(skill)
        
        return found_skills
    
    def _extract_years_from_text(self, text: str) -> int:
        """Extract years of experience from text."""
        if isinstance(text, (int, float)):
            return int(text)
        
        # Look for patterns like "5 years", "3+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?',
            r'(\d+)\s*years?\s*of\s*experience'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                return int(match.group(1))
        
        return 0
    
    def _extract_required_experience(self, job_description: str) -> int:
        """Extract required years of experience from job description."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
            r'(\d+)\+\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, job_description.lower())
            if match:
                return int(match.group(1))
        
        return 0
    
    def _extract_important_keywords(self, job_description: str) -> List[str]:
        """Extract important keywords from job description."""
        # Simple keyword extraction (could be enhanced with NLP)
        text = job_description.lower()
        
        # Remove common words
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract potential keywords (nouns, skills, technologies)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Get skill keywords that appear in the job description
        job_skills = self._extract_skills(text)
        
        # Combine and deduplicate
        all_keywords = list(set(keywords[:20] + list(job_skills)))  # Limit to top keywords
        return all_keywords
    
    def _load_skill_keywords(self) -> List[str]:
        """Load predefined skill keywords."""
        # Comprehensive skill dictionary
        skills = [
            # Technical skills
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'node.js', 'django', 'flask', 'fastapi', 'spring', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'mysql', 'postgresql', 'mongodb', 'redis',
            'git', 'github', 'gitlab', 'jenkins', 'ci/cd', 'devops', 'linux', 'windows',
            'html', 'css', 'sql', 'nosql', 'rest', 'api', 'microservices',
            'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'statistics', 'analytics',
            
            # Business skills
            'project management', 'agile', 'scrum', 'kanban', 'leadership', 'communication',
            'teamwork', 'problem solving', 'analytical', 'strategic', 'planning',
            'budgeting', 'forecasting', 'reporting', 'presentation', 'negotiation',
            
            # Industry-specific (hospitality)
            'customer service', 'guest relations', 'front desk', 'housekeeping',
            'food service', 'restaurant', 'hotel management', 'hospitality',
            'concierge', 'reservations', 'pms', 'opera', 'micros', 'pos',
            'food safety', 'haccp', 'beverage', 'culinary', 'kitchen',
            'banquet', 'events', 'catering', 'sales', 'marketing',
            
            # Soft skills
            'attention to detail', 'multitasking', 'time management', 'organization',
            'flexibility', 'adaptability', 'creativity', 'innovation', 'reliability',
            'professionalism', 'integrity', 'collaboration', 'mentoring', 'training'
        ]
        
        return skills
    
    @property
    def feature_names(self) -> List[str]:
        """Get names of features in the order they appear in the feature vector."""
        return [
            'semantic_similarity',
            'skill_overlap',
            'title_similarity', 
            'experience_penalty',
            'education_score',
            'recency_score',
            'resume_length',
            'keyword_density',
            'experience_diversity',
            'contact_completeness'
        ]
    
    @property
    def feature_count(self) -> int:
        """Get the number of features in the feature vector."""
        return len(self.feature_names)


# Global instance for reuse
_feature_extractor: Optional[FeatureExtractor] = None


def get_feature_extractor() -> FeatureExtractor:
    """Get the global feature extractor instance."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor


def extract_features(resume_text: str, job_description: str, 
                    resume_metadata: Optional[Dict] = None) -> np.ndarray:
    """
    Convenience function to extract features using the global extractor.
    
    Args:
        resume_text: Full text of the resume
        job_description: Full text of the job description
        resume_metadata: Optional metadata
        
    Returns:
        Feature vector as numpy array
    """
    extractor = get_feature_extractor()
    return extractor.extract_features(resume_text, job_description, resume_metadata)
