"""
Bias detection and mitigation for ML ranking system.
"""
import logging
from typing import Dict, List, Any, Optional, Set
import numpy as np
import pandas as pd
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class BiasDetector:
    """Detect potential bias in ranking results and training data."""
    
    def __init__(self):
        """Initialize bias detector."""
        # Sensitive fields that should not influence ranking
        self.sensitive_fields = {
            'names': [
                # Common names that might indicate gender or ethnicity
                'maria', 'jose', 'mohammed', 'ahmed', 'kumar', 'patel', 'kim', 'wang',
                'jennifer', 'michael', 'david', 'sarah', 'john', 'lisa', 'robert', 'mary'
            ],
            'gender_indicators': [
                'he', 'she', 'his', 'her', 'him', 'himself', 'herself',
                'mr.', 'mrs.', 'ms.', 'miss'
            ],
            'age_indicators': [
                'young', 'old', 'senior', 'recent graduate', 'experienced professional',
                'early career', 'mid-career', 'late career', 'retirement'
            ],
            'location_bias': [
                'urban', 'rural', 'city', 'suburb', 'downtown', 'uptown'
            ],
            'education_bias': [
                'ivy league', 'community college', 'state university', 'private school',
                'public school', 'prestigious', 'elite'
            ]
        }
        
        # Protected attributes patterns
        self.protected_patterns = {
            'age': [
                r'\b\d{2}\s*years?\s*old\b',
                r'\bage\s*:?\s*\d{2}\b',
                r'\bborn\s+in\s+\d{4}\b'
            ],
            'gender': [
                r'\b(he|she|his|her|him)\b',
                r'\b(male|female|man|woman)\b',
                r'\b(mr\.|mrs\.|ms\.|miss)\b'
            ],
            'ethnicity': [
                r'\b(african.american|hispanic|latino|asian|caucasian|white|black)\b',
                r'\b(native.american|indigenous|pacific.islander)\b'
            ],
            'religion': [
                r'\b(christian|muslim|jewish|hindu|buddhist|atheist|agnostic)\b',
                r'\b(church|mosque|synagogue|temple|religious)\b'
            ]
        }
    
    def detect_text_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect potential bias indicators in text.
        
        Args:
            text: Text to analyze for bias
            
        Returns:
            Dictionary with bias detection results
        """
        text_lower = text.lower()
        bias_indicators = {
            'protected_attributes': {},
            'sensitive_keywords': [],
            'bias_score': 0.0,
            'recommendations': []
        }
        
        # Check for protected attribute patterns
        for attribute, patterns in self.protected_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                bias_indicators['protected_attributes'][attribute] = matches
                bias_indicators['bias_score'] += len(matches) * 0.1
        
        # Check for sensitive keywords
        for category, keywords in self.sensitive_fields.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                bias_indicators['sensitive_keywords'].extend(found_keywords)
                bias_indicators['bias_score'] += len(found_keywords) * 0.05
        
        # Generate recommendations
        if bias_indicators['bias_score'] > 0.2:
            bias_indicators['recommendations'].append(
                "High bias risk detected. Consider reviewing for protected attributes."
            )
        
        if bias_indicators['protected_attributes']:
            bias_indicators['recommendations'].append(
                "Protected attributes detected. Ensure these don't influence ranking."
            )
        
        return bias_indicators
    
    def analyze_ranking_fairness(self, rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze fairness of ranking results.
        
        Args:
            rankings: List of ranking results
            
        Returns:
            Fairness analysis results
        """
        fairness_metrics = {
            'total_candidates': len(rankings),
            'bias_indicators_per_candidate': [],
            'aggregate_bias_score': 0.0,
            'fairness_recommendations': [],
            'demographic_distribution': defaultdict(int)
        }
        
        total_bias_score = 0.0
        
        for i, ranking in enumerate(rankings):
            candidate_bias = {
                'candidate_index': i,
                'score': ranking.get('score', 0.0),
                'bias_analysis': {}
            }
            
            # Analyze resume text if available
            resume_data = ranking.get('resume_data', {})
            if resume_data:
                raw_text = resume_data.get('raw_text', '')
                if raw_text:
                    bias_analysis = self.detect_text_bias(raw_text)
                    candidate_bias['bias_analysis'] = bias_analysis
                    total_bias_score += bias_analysis['bias_score']
                
                # Analyze contact info for potential bias
                contact_info = resume_data.get('contact_info', {})
                name = contact_info.get('name', '').lower()
                
                # Check for potentially biased name patterns
                for sensitive_name in self.sensitive_fields['names']:
                    if sensitive_name in name:
                        candidate_bias['bias_analysis']['name_bias'] = True
                        break
            
            fairness_metrics['bias_indicators_per_candidate'].append(candidate_bias)
        
        # Calculate aggregate metrics
        if len(rankings) > 0:
            fairness_metrics['aggregate_bias_score'] = total_bias_score / len(rankings)
        
        # Generate fairness recommendations
        if fairness_metrics['aggregate_bias_score'] > 0.3:
            fairness_metrics['fairness_recommendations'].append(
                "High overall bias score detected. Review ranking algorithm for fairness."
            )
        
        # Check for score distribution patterns
        scores = [r.get('score', 0.0) for r in rankings]
        if len(scores) > 5:
            score_std = np.std(scores)
            if score_std < 0.1:
                fairness_metrics['fairness_recommendations'].append(
                    "Low score variance detected. Check if ranking is discriminating effectively."
                )
        
        return fairness_metrics
    
    def sanitize_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Remove or modify features that might contain bias.
        
        Args:
            features: Feature array
            feature_names: Names of features
            
        Returns:
            Sanitized feature array
        """
        # Create copy to avoid modifying original
        sanitized_features = features.copy()
        
        # List of potentially biased features to handle carefully
        biased_feature_names = ['name_similarity', 'location_match', 'school_prestige']
        
        for i, name in enumerate(feature_names):
            if name in biased_feature_names:
                logger.warning(f"Potentially biased feature detected: {name}")
                # Option 1: Zero out the feature
                # sanitized_features[i] = 0.0
                
                # Option 2: Normalize to reduce impact
                if len(sanitized_features.shape) > 1:
                    sanitized_features[:, i] = (sanitized_features[:, i] - 
                                               np.mean(sanitized_features[:, i])) / np.std(sanitized_features[:, i])
                else:
                    sanitized_features[i] = 0.0
        
        return sanitized_features
    
    def audit_training_data(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Audit training data for bias.
        
        Args:
            training_data: Training data DataFrame
            
        Returns:
            Audit results
        """
        audit_results = {
            'total_samples': len(training_data),
            'label_distribution': {},
            'bias_indicators': [],
            'recommendations': [],
            'fairness_score': 0.0
        }
        
        # Analyze label distribution
        if 'label' in training_data.columns:
            label_counts = training_data['label'].value_counts()
            audit_results['label_distribution'] = label_counts.to_dict()
            
            # Check for imbalanced labels
            if len(label_counts) > 1:
                max_count = label_counts.max()
                min_count = label_counts.min()
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > 10:
                    audit_results['recommendations'].append(
                        f"Severe label imbalance detected (ratio: {imbalance_ratio:.1f}). "
                        "Consider balancing training data."
                    )
                elif imbalance_ratio > 3:
                    audit_results['recommendations'].append(
                        f"Moderate label imbalance detected (ratio: {imbalance_ratio:.1f}). "
                        "Monitor for bias in predictions."
                    )
        
        # Analyze features for bias patterns
        if 'features' in training_data.columns:
            features_analysis = []
            for idx, features in enumerate(training_data['features']):
                if isinstance(features, (list, np.ndarray)):
                    feature_array = np.array(features)
                    # Simple bias check: look for suspicious patterns
                    if len(feature_array) > 5 and np.std(feature_array) < 0.01:
                        features_analysis.append({
                            'sample_idx': idx,
                            'issue': 'low_variance_features',
                            'std': float(np.std(feature_array))
                        })
            
            if features_analysis:
                audit_results['bias_indicators'] = features_analysis
        
        # Calculate overall fairness score
        fairness_score = 1.0
        
        # Penalty for label imbalance
        if 'label' in training_data.columns and len(training_data['label'].unique()) > 1:
            max_count = training_data['label'].value_counts().max()
            min_count = training_data['label'].value_counts().min()
            imbalance_penalty = min(max_count / min_count / 10, 0.5)
            fairness_score -= imbalance_penalty
        
        # Penalty for bias indicators
        bias_penalty = min(len(audit_results['bias_indicators']) * 0.1, 0.3)
        fairness_score -= bias_penalty
        
        audit_results['fairness_score'] = max(0.0, fairness_score)
        
        # Overall recommendations
        if audit_results['fairness_score'] < 0.7:
            audit_results['recommendations'].append(
                "Low fairness score. Review training data and consider bias mitigation strategies."
            )
        
        return audit_results


class FairRanker:
    """Wrapper for ranking with fairness constraints."""
    
    def __init__(self, base_ranker, bias_detector: Optional[BiasDetector] = None):
        """
        Initialize fair ranker.
        
        Args:
            base_ranker: Base ranking model
            bias_detector: Bias detection instance
        """
        self.base_ranker = base_ranker
        self.bias_detector = bias_detector or BiasDetector()
        self.fairness_threshold = 0.3
    
    def rank_with_fairness(self, features: np.ndarray, 
                          resume_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Rank candidates with fairness considerations.
        
        Args:
            features: Feature matrix
            resume_data: Optional resume data for bias analysis
            
        Returns:
            Fair ranking results
        """
        # Sanitize features
        sanitized_features = self.bias_detector.sanitize_features(
            features, 
            ['semantic_similarity', 'skill_overlap', 'title_similarity',
             'experience_years', 'education_score', 'certification_score',
             'language_score', 'keyword_density', 'text_quality', 'completeness']
        )
        
        # Get base rankings
        base_scores = self.base_ranker.predict(sanitized_features)
        
        # Create ranking results
        rankings = []
        for i, score in enumerate(base_scores):
            ranking = {
                'candidate_index': i,
                'score': float(score),
                'features': sanitized_features[i].tolist() if len(sanitized_features.shape) > 1 else sanitized_features.tolist()
            }
            
            if resume_data and i < len(resume_data):
                ranking['resume_data'] = resume_data[i]
            
            rankings.append(ranking)
        
        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Analyze fairness
        fairness_analysis = self.bias_detector.analyze_ranking_fairness(rankings)
        
        # Apply fairness adjustments if needed
        if fairness_analysis['aggregate_bias_score'] > self.fairness_threshold:
            rankings = self._apply_fairness_adjustments(rankings, fairness_analysis)
            fairness_analysis['adjustments_applied'] = True
        else:
            fairness_analysis['adjustments_applied'] = False
        
        return {
            'rankings': rankings,
            'fairness_analysis': fairness_analysis,
            'bias_mitigation': 'applied' if fairness_analysis.get('adjustments_applied') else 'not_needed'
        }
    
    def _apply_fairness_adjustments(self, rankings: List[Dict[str, Any]], 
                                   fairness_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply fairness adjustments to rankings."""
        # Simple fairness adjustment: slightly randomize rankings with high bias
        adjusted_rankings = rankings.copy()
        
        for i, ranking in enumerate(adjusted_rankings):
            bias_indicators = fairness_analysis['bias_indicators_per_candidate'][i]
            bias_score = bias_indicators.get('bias_analysis', {}).get('bias_score', 0.0)
            
            if bias_score > self.fairness_threshold:
                # Small score adjustment to reduce bias impact
                adjustment = -bias_score * 0.1
                ranking['score'] = max(0.0, ranking['score'] + adjustment)
                ranking['fairness_adjusted'] = True
            else:
                ranking['fairness_adjusted'] = False
        
        # Re-sort after adjustments
        adjusted_rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return adjusted_rankings


# Global bias detector instance
_bias_detector: Optional[BiasDetector] = None


def get_bias_detector() -> BiasDetector:
    """Get the global bias detector instance."""
    global _bias_detector
    if _bias_detector is None:
        _bias_detector = BiasDetector()
    return _bias_detector
