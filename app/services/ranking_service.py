"""
ML-powered ranking service for candidate evaluation.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..ml.embeddings import get_embedding_service
from ..ml.features import get_feature_extractor
from ..ml.ranker import get_ranking_pipeline
from ..ml.store import get_model_store
from .parsing import get_resume_parser

logger = logging.getLogger(__name__)

class RankingService:
    """ML-powered candidate ranking service."""
    
    def __init__(self):
        """Initialize the ranking service."""
        self.embedding_service = get_embedding_service()
        self.feature_extractor = get_feature_extractor()
        self.ranking_pipeline = get_ranking_pipeline()
        self.model_store = get_model_store()
        self.resume_parser = get_resume_parser()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def rank_candidates(
        self, 
        job_description: str,
        candidate_files: List[str],
        role_id: str,
        use_ml: bool = True,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Rank candidates for a job position.
        
        Args:
            job_description: Job description text
            candidate_files: List of candidate resume file paths
            role_id: Role identifier for model selection
            use_ml: Whether to use ML ranking (fallback to rule-based)
            top_k: Number of top candidates to return
            
        Returns:
            Dictionary containing ranked candidates and metadata
        """
        try:
            logger.info(f"Starting ranking for {len(candidate_files)} candidates, role: {role_id}")
            
            # Parse all resumes
            parsed_resumes = await self._parse_resumes_batch(candidate_files)
            
            if not parsed_resumes:
                return self._empty_ranking_result()
            
            # Extract features for all candidates
            features_batch = await self._extract_features_batch(
                job_description, parsed_resumes
            )
            
            # Rank candidates
            if use_ml and self.model_store.model_exists(role_id):
                rankings = await self._rank_with_ml(
                    features_batch, role_id, top_k
                )
                ranking_method = "ml"
            else:
                rankings = await self._rank_with_rules(
                    features_batch, top_k
                )
                ranking_method = "rule-based"
            
            # Add resume data to rankings
            for ranking in rankings:
                candidate_idx = ranking['candidate_index']
                if candidate_idx < len(parsed_resumes):
                    ranking['resume_data'] = parsed_resumes[candidate_idx]
            
            # Calculate statistics
            stats = self._calculate_ranking_stats(rankings, features_batch)
            
            result = {
                'rankings': rankings,
                'job_description': job_description,
                'total_candidates': len(candidate_files),
                'ranked_candidates': len(rankings),
                'ranking_method': ranking_method,
                'role_id': role_id,
                'statistics': stats,
                'timestamp': self._get_timestamp()
            }
            
            logger.info(f"Ranking completed: {len(rankings)} candidates ranked using {ranking_method}")
            return result
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            raise
    
    async def _parse_resumes_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple resumes in parallel."""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for file_path in file_paths:
            task = loop.run_in_executor(
                self.executor, 
                self.resume_parser.parse_resume, 
                file_path
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return valid results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to parse {file_paths[i]}: {result}")
                    # Add empty result with error info
                    empty_data = self.resume_parser._empty_resume_data()
                    empty_data['file_path'] = file_paths[i]
                    empty_data['error'] = str(result)
                    valid_results.append(empty_data)
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Batch resume parsing failed: {e}")
            return []
    
    async def _extract_features_batch(
        self, 
        job_description: str, 
        parsed_resumes: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Extract features for all candidates."""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for resume_data in parsed_resumes:
            task = loop.run_in_executor(
                self.executor,
                self.feature_extractor.extract_features,
                job_description,
                resume_data
            )
            tasks.append(task)
        
        try:
            features_batch = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            valid_features = []
            for i, features in enumerate(features_batch):
                if isinstance(features, Exception):
                    logger.error(f"Feature extraction failed for candidate {i}: {features}")
                    # Use zero features as fallback
                    valid_features.append(np.zeros(10))  # 10 features
                else:
                    valid_features.append(features)
            
            return valid_features
            
        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e}")
            return [np.zeros(10) for _ in parsed_resumes]
    
    async def _rank_with_ml(
        self, 
        features_batch: List[np.ndarray], 
        role_id: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank candidates using ML model."""
        try:
            # Load trained model
            model = self.model_store.load_model(role_id)
            
            # Prepare features matrix
            X = np.array(features_batch)
            
            # Predict scores
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                self.executor,
                model.predict,
                X
            )
            
            # Get explanations
            explanations = await loop.run_in_executor(
                self.executor,
                model.explain_predictions,
                X
            )
            
            # Create rankings
            rankings = []
            for i, (score, explanation) in enumerate(zip(scores, explanations)):
                rankings.append({
                    'candidate_index': i,
                    'score': float(score),
                    'explanation': explanation,
                    'features': features_batch[i].tolist(),
                    'ranking_method': 'ml'
                })
            
            # Sort by score (descending) and return top-k
            rankings.sort(key=lambda x: x['score'], reverse=True)
            return rankings[:top_k]
            
        except Exception as e:
            logger.error(f"ML ranking failed: {e}")
            # Fallback to rule-based ranking
            return await self._rank_with_rules(features_batch, top_k)
    
    async def _rank_with_rules(
        self, 
        features_batch: List[np.ndarray], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank candidates using rule-based scoring."""
        rankings = []
        
        for i, features in enumerate(features_batch):
            # Rule-based scoring (weighted sum of features)
            weights = np.array([
                0.25,  # semantic_similarity
                0.20,  # skill_overlap
                0.15,  # title_similarity
                0.15,  # experience_years
                0.10,  # education_score
                0.05,  # certification_score
                0.05,  # language_score
                0.02,  # keyword_density
                0.02,  # text_quality
                0.01   # completeness
            ])
            
            score = np.dot(features, weights)
            
            # Create explanation
            feature_names = [
                'semantic_similarity', 'skill_overlap', 'title_similarity',
                'experience_years', 'education_score', 'certification_score',
                'language_score', 'keyword_density', 'text_quality', 'completeness'
            ]
            
            explanation = {
                'score_breakdown': {
                    name: float(features[j] * weights[j])
                    for j, name in enumerate(feature_names)
                },
                'top_factors': []
            }
            
            # Find top contributing factors
            contributions = features * weights
            top_indices = np.argsort(contributions)[-3:][::-1]
            
            for idx in top_indices:
                explanation['top_factors'].append({
                    'factor': feature_names[idx],
                    'contribution': float(contributions[idx]),
                    'raw_value': float(features[idx])
                })
            
            rankings.append({
                'candidate_index': i,
                'score': float(score),
                'explanation': explanation,
                'features': features.tolist(),
                'ranking_method': 'rule-based'
            })
        
        # Sort by score (descending) and return top-k
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings[:top_k]
    
    def _calculate_ranking_stats(
        self, 
        rankings: List[Dict[str, Any]], 
        features_batch: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate ranking statistics."""
        if not rankings:
            return {}
        
        scores = [r['score'] for r in rankings]
        
        stats = {
            'score_stats': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'feature_stats': {},
            'distribution': self._get_score_distribution(scores)
        }
        
        # Feature statistics
        if features_batch:
            features_matrix = np.array(features_batch)
            feature_names = [
                'semantic_similarity', 'skill_overlap', 'title_similarity',
                'experience_years', 'education_score', 'certification_score',
                'language_score', 'keyword_density', 'text_quality', 'completeness'
            ]
            
            for i, name in enumerate(feature_names):
                feature_values = features_matrix[:, i]
                stats['feature_stats'][name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values))
                }
        
        return stats
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get score distribution in bins."""
        if not scores:
            return {}
        
        bins = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        distribution = {bin_name: 0 for bin_name in bins}
        
        for score in scores:
            if score < 0.2:
                distribution['0.0-0.2'] += 1
            elif score < 0.4:
                distribution['0.2-0.4'] += 1
            elif score < 0.6:
                distribution['0.4-0.6'] += 1
            elif score < 0.8:
                distribution['0.6-0.8'] += 1
            else:
                distribution['0.8-1.0'] += 1
        
        return distribution
    
    def _empty_ranking_result(self) -> Dict[str, Any]:
        """Return empty ranking result."""
        return {
            'rankings': [],
            'job_description': '',
            'total_candidates': 0,
            'ranked_candidates': 0,
            'ranking_method': 'none',
            'role_id': '',
            'statistics': {},
            'timestamp': self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def train_model(
        self, 
        role_id: str, 
        training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Train ML model for a specific role.
        
        Args:
            role_id: Role identifier
            training_data: List of training examples with features and labels
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info(f"Starting model training for role: {role_id}")
            
            # Prepare training data
            X = np.array([example['features'] for example in training_data])
            y = np.array([example['label'] for example in training_data])
            
            # Train model
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                self.executor,
                self.ranking_pipeline.fit,
                X, y
            )
            
            # Save trained model
            self.model_store.save_model(role_id, self.ranking_pipeline.model)
            
            # Save training metrics
            self.model_store.save_metrics(role_id, metrics)
            
            logger.info(f"Model training completed for role: {role_id}")
            
            return {
                'role_id': role_id,
                'training_samples': len(training_data),
                'metrics': metrics,
                'status': 'success',
                'timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Model training failed for role {role_id}: {e}")
            raise
    
    async def get_model_info(self, role_id: str) -> Dict[str, Any]:
        """Get information about a trained model."""
        try:
            model_exists = self.model_store.model_exists(role_id)
            
            if not model_exists:
                return {
                    'role_id': role_id,
                    'model_exists': False,
                    'metrics': None,
                    'timestamp': None
                }
            
            # Load metrics
            try:
                metrics = self.model_store.load_metrics(role_id)
            except FileNotFoundError:
                metrics = None
            
            return {
                'role_id': role_id,
                'model_exists': True,
                'metrics': metrics,
                'model_info': self.model_store.get_role_info(role_id),
                'timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for role {role_id}: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Global service instance
_ranking_service: Optional[RankingService] = None


def get_ranking_service() -> RankingService:
    """Get the global ranking service instance."""
    global _ranking_service
    if _ranking_service is None:
        _ranking_service = RankingService()
    return _ranking_service
