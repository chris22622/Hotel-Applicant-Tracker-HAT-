"""
Complete setup and testing script for ML ranking system.
"""
import logging
import sys
import os
from pathlib import Path
import asyncio
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ml.embeddings import get_embedding_service
from app.ml.features import get_feature_extractor
from app.ml.ranker import get_ranking_pipeline
from app.ml.store import get_model_store
from app.ml.bias_safety import get_bias_detector
from app.services.parsing import get_resume_parser
from app.services.ranking_service import get_ranking_service

logger = logging.getLogger(__name__)

class MLSystemSetup:
    """Complete ML system setup and testing."""
    
    def __init__(self):
        """Initialize setup manager."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.components = {}
    
    def test_embeddings(self) -> bool:
        """Test embedding service."""
        try:
            logger.info("Testing embedding service...")
            embedding_service = get_embedding_service()
            
            # Test text embedding
            test_texts = [
                "Frontend developer with React experience",
                "Hotel manager with 5 years hospitality experience",
                "Python developer with machine learning skills"
            ]
            
            embeddings = embedding_service.embed_texts(test_texts)
            logger.info(f"✓ Embeddings generated: {embeddings.shape}")
            
            # Test similarity
            similarity = embedding_service.cosine_similarity(embeddings[0], embeddings[1])
            logger.info(f"✓ Similarity calculation: {similarity:.3f}")
            
            self.components['embeddings'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Embedding test failed: {e}")
            self.components['embeddings'] = False
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test feature extraction."""
        try:
            logger.info("Testing feature extraction...")
            feature_extractor = get_feature_extractor()
            
            # Sample job description and resume
            job_description = """
            Front Desk Agent - Hotel XYZ
            We are seeking an experienced front desk agent for our luxury hotel.
            Requirements: customer service experience, hotel management skills,
            fluency in English, computer skills, professional appearance.
            """
            
            resume_data = {
                'contact_info': {
                    'name': 'John Smith',
                    'email': 'john@example.com',
                    'phone': '555-1234'
                },
                'skills': ['customer service', 'hotel management', 'front desk', 'reservations'],
                'experience': [
                    {
                        'title': 'Front Desk Associate',
                        'company': 'Hotel ABC',
                        'duration_months': 24,
                        'description': 'Handled guest check-in and reservations'
                    }
                ],
                'education': [
                    {
                        'degree': 'Bachelor',
                        'field': 'Hospitality Management'
                    }
                ],
                'total_experience_years': 2.0,
                'summary': 'Experienced front desk professional with hospitality background',
                'raw_text': 'John Smith. Customer service and hotel management experience.'
            }
            
            # Get resume text from resume_data
            resume_text = resume_data.get('raw_text', '')
            
            features = feature_extractor.extract_features(resume_text, job_description, resume_data)
            logger.info(f"✓ Features extracted: {features.shape}, values: {features[:5]}")
            
            self.components['features'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Feature extraction test failed: {e}")
            self.components['features'] = False
            return False
    
    def test_ml_ranker(self) -> bool:
        """Test ML ranking model."""
        try:
            logger.info("Testing ML ranker...")
            ranking_pipeline = get_ranking_pipeline()
            
            # Create sample training data
            import numpy as np
            
            n_samples = 50
            n_features = 10
            
            # Generate synthetic training data
            X = np.random.rand(n_samples, n_features)
            y_float = np.random.rand(n_samples)  # Relevance scores
            # Convert to integer labels for ranking (0-4 scale)
            y = (y_float * 4).astype(int)
            groups = np.arange(n_samples)  # Each sample is its own group for ranking
            
            feature_names = [
                'semantic_similarity', 'skill_overlap', 'title_similarity',
                'experience_years', 'education_score', 'certification_score',
                'language_score', 'keyword_density', 'text_quality', 'completeness'
            ]
            
            # Train model
            metrics = ranking_pipeline.ranker.fit(X, y, groups, feature_names)
            logger.info(f"✓ Model training completed: {metrics}")
            
            # Test prediction
            predictions = ranking_pipeline.ranker.predict(X[:5])
            logger.info(f"✓ Predictions: {predictions}")
            
            self.components['ranker'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ ML ranker test failed: {e}")
            self.components['ranker'] = False
            return False
    
    def test_resume_parsing(self) -> bool:
        """Test resume parsing."""
        try:
            logger.info("Testing resume parsing...")
            resume_parser = get_resume_parser()
            
            # Test with sample text file
            sample_text = """
            John Smith
            john.smith@email.com
            (555) 123-4567
            
            SUMMARY
            Experienced hotel professional with 3 years in front desk operations
            
            EXPERIENCE
            Front Desk Agent - Hotel ABC (2021-2024)
            - Handled guest check-in and check-out
            - Managed reservations system
            - Provided customer service
            
            EDUCATION
            Bachelor of Science in Hospitality Management
            State University (2021)
            
            SKILLS
            Customer service, hotel management, front desk operations, reservations
            """
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_text)
                temp_file = f.name
            
            try:
                # Parse resume
                parsed_data = resume_parser.parse_resume(temp_file)
                logger.info(f"✓ Resume parsed successfully")
                logger.info(f"  - Name: {parsed_data['contact_info']['name']}")
                logger.info(f"  - Skills: {len(parsed_data['skills'])} found")
                logger.info(f"  - Experience: {parsed_data['total_experience_years']} years")
                
                self.components['parsing'] = True
                return True
                
            finally:
                # Clean up
                os.unlink(temp_file)
            
        except Exception as e:
            logger.error(f"✗ Resume parsing test failed: {e}")
            self.components['parsing'] = False
            return False
    
    def test_model_storage(self) -> bool:
        """Test model storage."""
        try:
            logger.info("Testing model storage...")
            model_store = get_model_store()
            
            # Test role creation
            test_role = "test_role"
            
            # Create dummy model (just a dict for testing)
            dummy_model = {"type": "test", "params": [1, 2, 3]}
            
            # Save model
            model_path = model_store.save_model(test_role, dummy_model)
            logger.info(f"✓ Model saved: {model_path}")
            
            # Load model
            loaded_model = model_store.load_model(test_role)
            logger.info(f"✓ Model loaded: {loaded_model}")
            
            # Test metrics
            test_metrics = {"accuracy": 0.85, "precision": 0.78}
            metrics_path = model_store.save_metrics(test_role, test_metrics)
            logger.info(f"✓ Metrics saved: {metrics_path}")
            
            loaded_metrics = model_store.load_metrics(test_role)
            logger.info(f"✓ Metrics loaded: {loaded_metrics}")
            
            # Clean up
            model_store.delete_role_data(test_role)
            logger.info("✓ Test data cleaned up")
            
            self.components['storage'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Model storage test failed: {e}")
            self.components['storage'] = False
            return False
    
    def test_bias_detection(self) -> bool:
        """Test bias detection."""
        try:
            logger.info("Testing bias detection...")
            bias_detector = get_bias_detector()
            
            # Test text with potential bias
            biased_text = """
            John is a 25-year-old male candidate with experience.
            He graduated from Harvard University.
            """
            
            bias_analysis = bias_detector.detect_text_bias(biased_text)
            logger.info(f"✓ Bias analysis completed: score={bias_analysis['bias_score']:.3f}")
            
            if bias_analysis['protected_attributes']:
                logger.info(f"  - Protected attributes detected: {list(bias_analysis['protected_attributes'].keys())}")
            
            self.components['bias_detection'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Bias detection test failed: {e}")
            self.components['bias_detection'] = False
            return False
    
    async def test_full_ranking_pipeline(self) -> bool:
        """Test complete ranking pipeline."""
        try:
            logger.info("Testing full ranking pipeline...")
            ranking_service = get_ranking_service()
            
            # Create test resume files
            import tempfile
            import os
            
            temp_dir = tempfile.mkdtemp()
            test_files = []
            
            # Sample resumes
            resumes = [
                {
                    'filename': 'candidate1.txt',
                    'content': """
                    Jane Doe
                    jane@email.com
                    (555) 111-2222
                    
                    Experienced front desk agent with 4 years in luxury hotels.
                    Skills: customer service, hotel management, front desk operations, reservations
                    
                    EXPERIENCE
                    Senior Front Desk Agent - Luxury Hotel (2020-2024)
                    """
                },
                {
                    'filename': 'candidate2.txt', 
                    'content': """
                    Bob Wilson
                    bob@email.com
                    (555) 333-4444
                    
                    Recent hospitality graduate seeking front desk position.
                    Skills: customer service, computer skills, communication
                    
                    EDUCATION
                    Bachelor in Hospitality Management (2024)
                    """
                }
            ]
            
            # Create test files
            for resume in resumes:
                file_path = os.path.join(temp_dir, resume['filename'])
                with open(file_path, 'w') as f:
                    f.write(resume['content'])
                test_files.append(file_path)
            
            try:
                # Test ranking
                job_description = """
                Front Desk Agent - Luxury Hotel
                Seeking experienced front desk professional for luxury hotel.
                Requirements: 2+ years hotel experience, excellent customer service,
                professional appearance, hotel software knowledge.
                """
                
                results = await ranking_service.rank_candidates(
                    job_description=job_description,
                    candidate_files=test_files,
                    role_id="front_desk_agent",
                    use_ml=False,  # Use rule-based for testing
                    top_k=5
                )
                
                logger.info("✓ Full ranking pipeline completed")
                logger.info(f"  - Candidates processed: {results['total_candidates']}")
                logger.info(f"  - Candidates ranked: {results['ranked_candidates']}")
                logger.info(f"  - Ranking method: {results['ranking_method']}")
                
                # Show top candidate
                if results['rankings']:
                    top_candidate = results['rankings'][0]
                    logger.info(f"  - Top candidate score: {top_candidate['score']:.3f}")
                
                self.components['full_pipeline'] = True
                return True
                
            finally:
                # Clean up
                import shutil
                shutil.rmtree(temp_dir)
            
        except Exception as e:
            logger.error(f"✗ Full pipeline test failed: {e}")
            self.components['full_pipeline'] = False
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all system tests."""
        logger.info("=" * 50)
        logger.info("STARTING ML RANKING SYSTEM TESTS")
        logger.info("=" * 50)
        
        # Run tests in order
        tests = [
            ("Embeddings", self.test_embeddings),
            ("Feature Extraction", self.test_feature_extraction),
            ("ML Ranker", self.test_ml_ranker),
            ("Resume Parsing", self.test_resume_parsing),
            ("Model Storage", self.test_model_storage),
            ("Bias Detection", self.test_bias_detection),
            ("Full Pipeline", self.test_full_ranking_pipeline)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Testing {test_name} ---")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name:20} : {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! ML ranking system is ready.")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Check logs for details.")
        
        return results


async def main():
    """Main setup function."""
    setup = MLSystemSetup()
    results = await setup.run_all_tests()
    
    # Create status file
    status_file = Path("var/ml_system_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(status_file, 'w') as f:
        json.dump({
            'setup_complete': all(results.values()),
            'test_results': results,
            'timestamp': str(pd.Timestamp.now())
        }, f, indent=2)
    
    logger.info(f"Setup status saved to: {status_file}")


if __name__ == "__main__":
    import pandas as pd
    asyncio.run(main())
