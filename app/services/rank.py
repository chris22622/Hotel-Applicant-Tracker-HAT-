"""Ranking service for candidates - simplified without LightGBM."""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))  # Both should be normalized


def calculate_must_have_coverage(candidate_skills: List[str], must_have_skills: List[str]) -> float:
    """Calculate coverage of must-have skills."""
    if not must_have_skills:
        return 1.0
    
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    must_have_lower = [skill.lower() for skill in must_have_skills]
    
    matched = sum(1 for skill in must_have_lower if skill in candidate_skills_lower)
    return matched / len(must_have_skills)


def calculate_nice_to_have_coverage(candidate_skills: List[str], nice_to_have_skills: List[str]) -> float:
    """Calculate coverage of nice-to-have skills."""
    if not nice_to_have_skills:
        return 0.0
    
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    nice_to_have_lower = [skill.lower() for skill in nice_to_have_skills]
    
    matched = sum(1 for skill in nice_to_have_lower if skill in candidate_skills_lower)
    return min(matched / len(nice_to_have_skills), 1.0)


def calculate_recency_score(last_relevant_role_year: Optional[int]) -> float:
    """Calculate recency score based on last relevant role."""
    if not last_relevant_role_year:
        return 0.0
    
    current_year = datetime.now().year
    months_since = (current_year - last_relevant_role_year) * 12
    
    return 1.0 if months_since <= 24 else 0.0


def calculate_tenure_score(avg_tenure_months: Optional[float]) -> float:
    """Calculate tenure stability score."""
    if not avg_tenure_months:
        return 0.0
    
    return min(avg_tenure_months / 24, 1.0)  # Normalize to 24 months


def calculate_seniority_alignment(candidate_years: float, role_min_years: Optional[int]) -> float:
    """Calculate seniority alignment score."""
    if not role_min_years:
        return 1.0
    
    # Simple alignment: penalize if too far from required years
    diff = abs(candidate_years - role_min_years)
    return max(0.0, 1.0 - (diff / 3.0))  # Penalty increases with difference


def hybrid_score(
    resume_vec: np.ndarray,
    role_vec: np.ndarray,
    must_coverage: float,
    nice_coverage: float,
    recency: float,
    tenure: float,
    seniority_align: float
) -> float:
    """Calculate hybrid ranking score."""
    semantic = cosine_similarity(resume_vec, role_vec)
    
    final_score = (
        0.45 * semantic +
        0.20 * must_coverage +
        0.10 * nice_coverage +
        0.10 * recency +
        0.10 * tenure +
        0.05 * seniority_align
    )
    
    return final_score


def generate_explanation(
    matched_must: List[str],
    matched_nice: List[str],
    years_total: float,
    last_role_year: Optional[int],
    avg_tenure_months: Optional[float]
) -> str:
    """Generate human-readable explanation for ranking."""
    parts = []
    
    if matched_must:
        parts.append(f"Must matches: {', '.join(matched_must[:5])}")
    
    if matched_nice:
        parts.append(f"Nice matches: {', '.join(matched_nice[:5])}")
    
    parts.append(f"{years_total:.1f} yrs total")
    
    if last_role_year:
        parts.append(f"last relevant role {last_role_year}")
    
    if avg_tenure_months:
        parts.append(f"avg tenure {avg_tenure_months:.0f} mo")
    
    return " | ".join(parts)


def check_hard_disqualifiers(
    candidate_data: Dict[str, Any],
    role_data: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Check if candidate meets hard requirements."""
    # Check minimum years
    min_years = role_data.get('min_years')
    if min_years and candidate_data.get('years_total', 0) < min_years:
        return False, f"Insufficient experience: {candidate_data.get('years_total', 0)} < {min_years} years required"
    
    # Check must-have skills coverage
    must_have = role_data.get('must_have', [])
    candidate_skills = candidate_data.get('skills', [])
    must_coverage = calculate_must_have_coverage(candidate_skills, must_have)
    
    if must_have and must_coverage < 0.7:  # Require 70% coverage
        return False, f"Insufficient must-have skills: {must_coverage:.1%} coverage"
    
    # TODO: Add other hard filters (work_auth, location, etc.)
    
    return True, None


def rank_candidate(
    candidate_data: Dict[str, Any],
    role_data: Dict[str, Any],
    candidate_embedding: np.ndarray,
    role_embedding: np.ndarray
) -> Tuple[float, Dict[str, Any], str]:
    """Rank a single candidate for a role."""
    # Check hard disqualifiers first
    qualified, disqualifier_reason = check_hard_disqualifiers(candidate_data, role_data)
    if not qualified:
        return 0.0, {"disqualified": True, "reason": disqualifier_reason}, disqualifier_reason
    
    # Calculate components
    candidate_skills = candidate_data.get('skills', [])
    must_have = role_data.get('must_have', [])
    nice_to_have = role_data.get('nice_to_have', [])
    
    must_coverage = calculate_must_have_coverage(candidate_skills, must_have)
    nice_coverage = calculate_nice_to_have_coverage(candidate_skills, nice_to_have)
    
    # Get matched skills for explanation
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    matched_must = [skill for skill in must_have if skill.lower() in candidate_skills_lower]
    matched_nice = [skill for skill in nice_to_have if skill.lower() in candidate_skills_lower]
    
    # Calculate other scores
    last_role_year = None
    avg_tenure = None
    experiences = candidate_data.get('experiences', [])
    if experiences:
        # Find most recent role year
        for exp in experiences:
            end_date = exp.get('end_date')
            if end_date:
                last_role_year = max(last_role_year or 0, end_date.year)
            else:  # Current role
                last_role_year = datetime.now().year
                break
        
        # Calculate average tenure
        tenures = []
        for exp in experiences:
            start_date = exp.get('start_date')
            end_date = exp.get('end_date') or datetime.now().date()
            if start_date and end_date:
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                tenures.append(max(1, months))  # At least 1 month
        avg_tenure = sum(tenures) / len(tenures) if tenures else None
    
    recency = calculate_recency_score(last_role_year)
    tenure = calculate_tenure_score(avg_tenure)
    seniority_align = calculate_seniority_alignment(
        candidate_data.get('years_total', 0),
        role_data.get('min_years')
    )
    
    # Calculate final score
    score = hybrid_score(
        candidate_embedding, role_embedding,
        must_coverage, nice_coverage,
        recency, tenure, seniority_align
    )
    
    # Generate explanation
    explanation = generate_explanation(
        matched_must, matched_nice,
        candidate_data.get('years_total', 0),
        last_role_year, avg_tenure
    )
    
    # Score breakdown for debugging
    breakdown = {
        'semantic': cosine_similarity(candidate_embedding, role_embedding),
        'must_coverage': must_coverage,
        'nice_coverage': nice_coverage,
        'recency': recency,
        'tenure': tenure,
        'seniority_align': seniority_align,
        'final': score
    }
    
    return score, breakdown, explanation


class LightGBMReranker:
    """Optional LightGBM model for reranking."""
    
    def __init__(self, model_path: str = "reranker_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model if exists."""
        if LIGHTGBM_AVAILABLE and os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Loaded LightGBM reranker from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
    
    def train(self, features: List[List[float]], labels: List[float]) -> bool:
        """Train the reranking model."""
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM not available, skipping training")
            return False
        
        if len(features) < 10:  # Need minimum data
            print("Insufficient training data")
            return False
        
        try:
            train_data = lgb.Dataset(features, label=labels)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            print(f"Trained and saved LightGBM reranker to {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict(self, features: List[float]) -> Optional[float]:
        """Predict reranked score."""
        if self.model is None:
            return None
        
        try:
            return float(self.model.predict([features])[0])
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None


# Global reranker instance
reranker = LightGBMReranker()
