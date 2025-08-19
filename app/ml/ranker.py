"""
ML-based ranking system for resume candidates.
"""
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import ndcg_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime

logger = logging.getLogger(__name__)

class MLRanker:
    """Machine learning ranker for resume scoring."""
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        Initialize the ML ranker.
        
        Args:
            model_type: Type of model to use ("lightgbm", "xgboost", or "sklearn")
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.training_metrics: Dict[str, float] = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train the ranking model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Relevance scores (n_samples,)
            groups: Group identifiers for ranking (n_samples,)
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} ranker with {len(X)} samples")
        
        self.feature_names = feature_names
        
        try:
            if self.model_type == "lightgbm":
                metrics = self._fit_lightgbm(X, y, groups)
            elif self.model_type == "xgboost":
                metrics = self._fit_xgboost(X, y, groups)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.is_trained = True
            self.training_metrics = metrics
            
            logger.info(f"Training completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict scores for features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted scores (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        try:
            if self.model_type == "lightgbm":
                scores = self.model.predict(X)
            elif self.model_type == "xgboost":
                dtest = xgb.DMatrix(X, feature_names=self.feature_names)
                scores = self.model.predict(dtest)
            else:
                scores = self.model.predict(X)
            
            return np.array(scores)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_with_explanation(self, X: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict scores with feature explanations.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            top_k: Number of top features to include in explanation
            
        Returns:
            List of prediction results with explanations
        """
        scores = self.predict(X)
        explanations = []
        
        for i, score in enumerate(scores):
            explanation = {
                'score': float(score),
                'top_features': self._get_top_features(X[i], top_k)
            }
            explanations.append(explanation)
        
        return explanations
    
    def save(self, file_path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'trained_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load(self, file_path: str) -> None:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = True
            
            logger.info(f"Model loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
        """Train LightGBM LambdaMART model."""
        # Prepare data for LightGBM ranking
        train_data = lgb.Dataset(X, label=y, group=self._get_group_sizes(groups))
        
        # LightGBM parameters for ranking
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Train with early stopping
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Calculate metrics
        predictions = self.model.predict(X)
        metrics = self._calculate_metrics(y, predictions, groups)
        
        return metrics
    
    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
        """Train XGBoost ranking model."""
        # Prepare data for XGBoost ranking
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        dtrain.set_group(self._get_group_sizes(groups))
        
        # XGBoost parameters for ranking
        params = {
            'objective': 'rank:pairwise',
            'eta': 0.1,
            'gamma': 1.0,
            'min_child_weight': 0.1,
            'max_depth': 6,
            'eval_metric': 'ndcg@10',
            'random_state': 42
        }
        
        # Train the model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=50,
            verbose_eval=0
        )
        
        # Calculate metrics
        predictions = self.model.predict(dtrain)
        metrics = self._calculate_metrics(y, predictions, groups)
        
        return metrics
    
    def _get_group_sizes(self, groups: np.ndarray) -> List[int]:
        """Get group sizes for ranking."""
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        return group_counts.tolist()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          groups: np.ndarray) -> Dict[str, float]:
        """Calculate ranking metrics."""
        metrics = {}
        
        try:
            # NDCG@10 (group-wise)
            ndcg_scores = []
            unique_groups = np.unique(groups)
            
            for group in unique_groups:
                group_mask = groups == group
                group_true = y_true[group_mask]
                group_pred = y_pred[group_mask]
                
                if len(group_true) > 1:  # Need at least 2 items for NDCG
                    # Reshape for ndcg_score function
                    ndcg = ndcg_score([group_true], [group_pred], k=10)
                    ndcg_scores.append(ndcg)
            
            if ndcg_scores:
                metrics['ndcg@10'] = np.mean(ndcg_scores)
                metrics['ndcg@5'] = np.mean(ndcg_scores)  # Simplified
            
            # AUC (if binary relevance)
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
            
            # Precision@K metrics
            metrics.update(self._calculate_precision_at_k(y_true, y_pred, groups))
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 groups: np.ndarray, k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """Calculate Precision@K metrics."""
        precision_metrics = {}
        
        for k in k_values:
            precisions = []
            unique_groups = np.unique(groups)
            
            for group in unique_groups:
                group_mask = groups == group
                group_true = y_true[group_mask]
                group_pred = y_pred[group_mask]
                
                if len(group_true) >= k:
                    # Get top-k predictions
                    top_k_indices = np.argsort(group_pred)[-k:]
                    top_k_true = group_true[top_k_indices]
                    
                    # Calculate precision (assuming relevance threshold)
                    relevant_threshold = np.median(group_true)
                    precision = np.sum(top_k_true > relevant_threshold) / k
                    precisions.append(precision)
            
            if precisions:
                precision_metrics[f'precision@{k}'] = np.mean(precisions)
        
        return precision_metrics
    
    def _get_top_features(self, feature_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top contributing features for explanation."""
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_vector))]
        else:
            feature_names = self.feature_names
        
        # Simple feature importance based on values (could be enhanced with SHAP)
        feature_importance = np.abs(feature_vector)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'name': feature_names[idx],
                'value': float(feature_vector[idx]),
                'importance': float(feature_importance[idx])
            })
        
        return top_features
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_metrics': self.training_metrics
        }


class RankingPipeline:
    """Complete pipeline for training and using ranking models."""
    
    def __init__(self, model_type: str = "lightgbm"):
        self.ranker = MLRanker(model_type)
        self.feature_extractor = None
    
    def train_from_data(self, training_data: pd.DataFrame, 
                       feature_columns: List[str], 
                       target_column: str,
                       group_column: str) -> Dict[str, float]:
        """
        Train the ranker from a pandas DataFrame.
        
        Args:
            training_data: DataFrame with features, targets, and groups
            feature_columns: List of feature column names
            target_column: Name of the target/relevance column
            group_column: Name of the group identifier column
            
        Returns:
            Training metrics
        """
        X = training_data[feature_columns].values
        y = training_data[target_column].values
        groups = training_data[group_column].values
        
        return self.ranker.fit(X, y, groups, feature_columns)
    
    def predict_candidates(self, candidates_data: pd.DataFrame,
                          feature_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Predict scores for candidates.
        
        Args:
            candidates_data: DataFrame with candidate features
            feature_columns: List of feature column names
            
        Returns:
            List of candidates with scores and explanations
        """
        X = candidates_data[feature_columns].values
        results = self.ranker.predict_with_explanation(X)
        
        # Combine with candidate data
        for i, result in enumerate(results):
            result.update(candidates_data.iloc[i].to_dict())
        
        return results


# Utility functions for training data preparation
def prepare_training_data(applications_df: pd.DataFrame,
                         labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training data from applications and labels.
    
    Args:
        applications_df: DataFrame with application data
        labels_df: DataFrame with feedback labels
        
    Returns:
        Combined training DataFrame
    """
    # Merge applications with labels
    training_data = applications_df.merge(labels_df, on=['candidate_id', 'role_id'], how='inner')
    
    # Convert labels to numeric relevance scores
    label_mapping = {
        'hire': 3,
        'interview': 2,
        'shortlist': 1,
        'reject': 0
    }
    
    training_data['relevance'] = training_data['label'].map(label_mapping)
    
    # Add thumbs up/down if available
    if 'thumbs_up' in training_data.columns:
        training_data['relevance'] += training_data['thumbs_up'].fillna(0) * 0.5
    
    return training_data


def create_ml_ranker(model_type: str = "lightgbm") -> MLRanker:
    """Factory function to create a configured ML ranker."""
    return MLRanker(model_type=model_type)


# Global ranking pipeline instance
_ranking_pipeline: Optional[RankingPipeline] = None


def get_ranking_pipeline() -> RankingPipeline:
    """Get the global ranking pipeline instance."""
    global _ranking_pipeline
    if _ranking_pipeline is None:
        _ranking_pipeline = RankingPipeline()
    return _ranking_pipeline
