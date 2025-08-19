"""
Training pipeline for ML ranking models.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import sys
import json

# Add app to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.ml.features import get_feature_extractor
from app.ml.ranker import get_ranking_pipeline
from app.ml.store import get_model_store
from app.services.parsing import get_resume_parser

logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Complete training pipeline for ML ranking models."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.feature_extractor = get_feature_extractor()
        self.ranking_pipeline = get_ranking_pipeline()
        self.model_store = get_model_store()
        self.resume_parser = get_resume_parser()
    
    def create_synthetic_training_data(self, role_id: str, num_samples: int = 100) -> pd.DataFrame:
        """
        Create synthetic training data for initial model training.
        
        Args:
            role_id: Role identifier
            num_samples: Number of synthetic samples to create
            
        Returns:
            DataFrame with training data
        """
        logger.info(f"Creating {num_samples} synthetic training samples for role: {role_id}")
        
        # Hotel role descriptions for different positions
        hotel_roles = {
            'front_desk_agent': {
                'description': 'Front desk agent responsible for guest check-in/out, reservations, and customer service',
                'key_skills': ['customer service', 'hotel management', 'reservations', 'front desk', 'hospitality']
            },
            'housekeeping_supervisor': {
                'description': 'Housekeeping supervisor managing cleaning staff and room maintenance',
                'key_skills': ['housekeeping', 'cleaning', 'supervision', 'hospitality', 'team management']
            },
            'food_service_manager': {
                'description': 'Food service manager overseeing restaurant operations and staff',
                'key_skills': ['food service', 'restaurant management', 'hospitality', 'team leadership', 'customer service']
            },
            'concierge': {
                'description': 'Concierge providing guest services and local recommendations',
                'key_skills': ['customer service', 'hospitality', 'concierge', 'guest services', 'communication']
            },
            'maintenance_technician': {
                'description': 'Maintenance technician handling hotel facility repairs and upkeep',
                'key_skills': ['maintenance', 'repair', 'facility management', 'troubleshooting', 'technical skills']
            }
        }
        
        # Use role description or default
        role_info = hotel_roles.get(role_id, hotel_roles['front_desk_agent'])
        job_description = role_info['description']
        key_skills = role_info['key_skills']
        
        training_data = []
        
        for i in range(num_samples):
            # Create synthetic candidate data
            candidate_data = self._create_synthetic_candidate(key_skills, i)
            
            # Extract features (correct order: resume_text, job_description, resume_metadata)
            resume_text = candidate_data.get('raw_text', '')
            features = self.feature_extractor.extract_features(resume_text, job_description, candidate_data)
            
            # Assign synthetic label based on feature quality
            label = self._calculate_synthetic_label(features, candidate_data, key_skills)
            
            training_data.append({
                'candidate_id': f"synthetic_{role_id}_{i}",
                'features': features.tolist(),
                'label': label,
                'role_id': role_id
            })
        
        df = pd.DataFrame(training_data)
        logger.info(f"Created synthetic training data: {len(df)} samples")
        
        return df
    
    def _create_synthetic_candidate(self, key_skills: List[str], index: int) -> Dict[str, Any]:
        """Create synthetic candidate data."""
        # Random experience and skills
        np.random.seed(index)  # For reproducible results
        
        experience_years = np.random.uniform(0, 15)
        num_skills = np.random.randint(2, min(8, len(key_skills) + 3))
        
        # Mix of relevant and irrelevant skills
        relevant_skills = np.random.choice(key_skills, 
                                         size=min(num_skills//2, len(key_skills)), 
                                         replace=False).tolist()
        
        irrelevant_skills = ['python', 'java', 'marketing', 'accounting', 'sales', 'engineering']
        additional_skills = np.random.choice(irrelevant_skills, 
                                           size=num_skills - len(relevant_skills), 
                                           replace=False).tolist()
        
        all_skills = relevant_skills + additional_skills
        
        # Create candidate data
        candidate_data = {
            'contact_info': {
                'name': f'Candidate {index}',
                'email': f'candidate{index}@example.com',
                'phone': f'555-{index:04d}',
                'location': 'City, State'
            },
            'skills': all_skills,
            'experience': [
                {
                    'title': 'Previous Position',
                    'company': 'Previous Company',
                    'duration_months': int(experience_years * 12),
                    'description': f'Experience with {", ".join(relevant_skills[:2])}'
                }
            ],
            'education': [
                {
                    'degree': 'Bachelor',
                    'field': 'Hospitality Management' if np.random.random() > 0.5 else 'General Studies',
                    'institution': 'University'
                }
            ],
            'total_experience_years': experience_years,
            'summary': f'Professional with {experience_years:.1f} years of experience in {relevant_skills[0] if relevant_skills else "various fields"}',
            'raw_text': f'Candidate with skills in {", ".join(all_skills[:3])} and {experience_years:.1f} years experience'
        }
        
        return candidate_data
    
    def _calculate_synthetic_label(self, features: np.ndarray, candidate_data: Dict[str, Any], key_skills: List[str]) -> float:
        """Calculate synthetic label based on candidate quality."""
        # Base score from features
        score = 0.0
        
        # Weight important features
        if len(features) >= 4:
            score += features[0] * 0.3  # semantic_similarity
            score += features[1] * 0.3  # skill_overlap  
            score += features[3] * 0.2  # experience_years (normalized)
            score += features[4] * 0.2  # education_score
        
        # Bonus for relevant skills
        candidate_skills = set(skill.lower() for skill in candidate_data.get('skills', []))
        key_skills_lower = set(skill.lower() for skill in key_skills)
        skill_match = len(candidate_skills.intersection(key_skills_lower)) / len(key_skills_lower)
        score += skill_match * 0.3
        
        # Bonus for experience
        experience_years = candidate_data.get('total_experience_years', 0)
        if experience_years >= 3:
            score += 0.2
        elif experience_years >= 1:
            score += 0.1
        
        # Bonus for hospitality education
        education = candidate_data.get('education', [])
        for edu in education:
            if 'hospitality' in edu.get('field', '').lower():
                score += 0.1
                break
        
        # Add some noise and clamp to [0, 1]
        score += np.random.normal(0, 0.1)
        score = max(0.0, min(1.0, score))
        
        return score
    
    def train_model_for_role(self, role_id: str, num_synthetic_samples: int = 200) -> Dict[str, Any]:
        """
        Train a model for a specific role.
        
        Args:
            role_id: Role identifier
            num_synthetic_samples: Number of synthetic samples to generate
            
        Returns:
            Training results
        """
        logger.info(f"Starting model training for role: {role_id}")
        
        try:
            # Create synthetic training data
            training_df = self.create_synthetic_training_data(role_id, num_synthetic_samples)
            
            # Prepare data for training
            X = np.array([features for features in training_df['features']])
            y_float = np.array(training_df['label'].values)
            
            # Convert to integer labels for ranking (0-4 scale)
            y = (y_float * 4).astype(int)
            
            # Create groups (each candidate is a separate group for ranking)
            groups = np.arange(len(X))
            
            # Train model
            feature_names = [
                'semantic_similarity', 'skill_overlap', 'title_similarity',
                'experience_years', 'education_score', 'certification_score',
                'language_score', 'keyword_density', 'text_quality', 'completeness'
            ]
            
            metrics = self.ranking_pipeline.ranker.fit(X, y, groups, feature_names)
            
            # Save model
            self.model_store.save_model(role_id, self.ranking_pipeline.ranker)
            self.model_store.save_metrics(role_id, metrics)
            
            logger.info(f"Model training completed for role: {role_id}")
            
            return {
                'role_id': role_id,
                'training_samples': len(training_df),
                'metrics': metrics,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Model training failed for role {role_id}: {e}")
            raise
    
    def train_all_hotel_roles(self) -> Dict[str, Any]:
        """Train models for all hotel roles."""
        hotel_roles = [
            'front_desk_agent',
            'housekeeping_supervisor', 
            'food_service_manager',
            'concierge',
            'maintenance_technician',
            'guest_services_representative',
            'night_auditor',
            'valet_parking_attendant',
            'hotel_manager',
            'sales_coordinator'
        ]
        
        results = {}
        
        for role in hotel_roles:
            try:
                logger.info(f"Training model for role: {role}")
                result = self.train_model_for_role(role, num_synthetic_samples=150)
                results[role] = result
                logger.info(f"Successfully trained model for role: {role}")
            except Exception as e:
                logger.error(f"Failed to train model for role {role}: {e}")
                results[role] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Summary
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        total = len(results)
        
        logger.info(f"Training completed: {successful}/{total} models trained successfully")
        
        return {
            'total_roles': total,
            'successful_training': successful,
            'failed_training': total - successful,
            'results': results
        }
    
    def evaluate_model(self, role_id: str, test_samples: int = 50) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            role_id: Role identifier
            test_samples: Number of test samples to generate
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model for role: {role_id}")
        
        try:
            # Check if model exists
            if not self.model_store.model_exists(role_id):
                raise ValueError(f"No trained model found for role: {role_id}")
            
            # Load model
            model = self.model_store.load_model(role_id)
            
            # Create test data
            test_df = self.create_synthetic_training_data(role_id, test_samples)
            
            # Prepare test data
            X_test = np.array([features for features in test_df['features']])
            y_true = test_df['label'].values
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr, spearmanr
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            pearson_r, _ = pearsonr(y_true, y_pred)
            spearman_r, _ = spearmanr(y_true, y_pred)
            
            evaluation_results = {
                'role_id': role_id,
                'test_samples': test_samples,
                'mse': float(mse),
                'mae': float(mae),
                'pearson_correlation': float(pearson_r),
                'spearman_correlation': float(spearman_r),
                'predictions_summary': {
                    'mean_pred': float(np.mean(y_pred)),
                    'std_pred': float(np.std(y_pred)),
                    'min_pred': float(np.min(y_pred)),
                    'max_pred': float(np.max(y_pred))
                },
                'labels_summary': {
                    'mean_true': float(np.mean(y_true)),
                    'std_true': float(np.std(y_true)),
                    'min_true': float(np.min(y_true)),
                    'max_true': float(np.max(y_true))
                }
            }
            
            logger.info(f"Model evaluation completed for role: {role_id}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed for role {role_id}: {e}")
            raise


def main():
    """Main training script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Train ML ranking models')
    parser.add_argument('--role', type=str, help='Specific role to train')
    parser.add_argument('--all', action='store_true', help='Train all hotel roles')
    parser.add_argument('--evaluate', type=str, help='Evaluate specific role model')
    parser.add_argument('--samples', type=int, default=200, help='Number of training samples')
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    trainer = MLTrainingPipeline()
    
    if args.all:
        # Train all roles
        logger.info("Training models for all hotel roles...")
        results = trainer.train_all_hotel_roles()
        
        # Save results
        results_file = Path("var/training_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training completed. Results saved to {results_file}")
        print(f"Successfully trained {results['successful_training']}/{results['total_roles']} models")
        
    elif args.role:
        # Train specific role
        logger.info(f"Training model for role: {args.role}")
        result = trainer.train_model_for_role(args.role, args.samples)
        print(f"Training completed for {args.role}: {result}")
        
    elif args.evaluate:
        # Evaluate model
        logger.info(f"Evaluating model for role: {args.evaluate}")
        result = trainer.evaluate_model(args.evaluate)
        print(f"Evaluation results for {args.evaluate}:")
        print(json.dumps(result, indent=2))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
