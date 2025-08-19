"""
ML-powered candidate ranking routes.
"""
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, Field
import tempfile
import os
from pathlib import Path

from ..services.ranking_service import get_ranking_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response models
class RankingRequest(BaseModel):
    """Request model for candidate ranking."""
    job_description: str = Field(..., description="Job description text")
    role_id: str = Field(..., description="Role identifier")
    use_ml: bool = Field(True, description="Whether to use ML ranking")
    top_k: int = Field(10, description="Number of top candidates to return")

class TrainingExample(BaseModel):
    """Training example for ML model."""
    features: List[float] = Field(..., description="Feature vector")
    label: float = Field(..., description="Relevance label (0-1)")
    role_id: Optional[str] = Field(None, description="Role identifier")

class TrainingRequest(BaseModel):
    """Request model for model training."""
    role_id: str = Field(..., description="Role identifier")
    training_data: List[TrainingExample] = Field(..., description="Training examples")

class LabelRequest(BaseModel):
    """Request model for labeling candidates."""
    candidate_id: str = Field(..., description="Candidate identifier")
    role_id: str = Field(..., description="Role identifier")
    label: str = Field(..., description="Label (hire/interview/shortlist/reject)")
    score: Optional[float] = Field(None, description="Optional numeric score")

# Main ranking endpoints
@router.post("/ml")
async def rank_candidates_ml(request: RankingRequest, files: List[UploadFile] = File(...)):
    """
    Rank candidates using ML-powered ranking.
    
    This endpoint accepts job description and resume files, then returns
    ranked candidates with scores and explanations.
    """
    ranking_service = get_ranking_service()
    
    try:
        # Save uploaded files temporarily
        temp_files = []
        temp_dir = tempfile.mkdtemp()
        
        for file in files:
            # Validate file type
            if not file.filename or not file.filename.lower().endswith(('.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg')):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            
            # Save file
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            temp_files.append(temp_path)
        
        # Rank candidates
        results = await ranking_service.rank_candidates(
            job_description=request.job_description,
            candidate_files=temp_files,
            role_id=request.role_id,
            use_ml=request.use_ml,
            top_k=request.top_k
        )
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        return {
            "status": "success",
            "data": results,
            "message": f"Successfully ranked {results['ranked_candidates']} candidates"
        }
        
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        # Clean up on error
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

@router.post("/batch")
async def rank_candidates_batch(request: RankingRequest):
    """
    Rank candidates from the input_resumes folder.
    
    This endpoint processes all resumes in the input_resumes directory.
    """
    ranking_service = get_ranking_service()
    
    try:
        # Get all resume files from input directory
        input_dir = Path("input_resumes")
        if not input_dir.exists():
            raise HTTPException(status_code=404, detail="input_resumes directory not found")
        
        # Find all resume files
        resume_files = []
        for ext in ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg']:
            resume_files.extend(list(input_dir.glob(f"*{ext}")))
        
        if not resume_files:
            raise HTTPException(status_code=404, detail="No resume files found in input_resumes")
        
        # Convert to string paths
        file_paths = [str(f) for f in resume_files]
        
        # Rank candidates
        results = await ranking_service.rank_candidates(
            job_description=request.job_description,
            candidate_files=file_paths,
            role_id=request.role_id,
            use_ml=request.use_ml,
            top_k=request.top_k
        )
        
        return {
            "status": "success",
            "data": results,
            "message": f"Successfully ranked {results['ranked_candidates']} candidates from {len(file_paths)} files"
        }
        
    except Exception as e:
        logger.error(f"Batch ranking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch ranking failed: {str(e)}")

# Training and labeling endpoints
@router.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train ML model for a specific role.
    
    This endpoint trains a ranking model using labeled training data.
    Training is performed in the background.
    """
    ranking_service = get_ranking_service()
    
    try:
        # Validate training data
        if len(request.training_data) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 training examples")
        
        # Convert to format expected by service
        training_data = []
        for example in request.training_data:
            training_data.append({
                'features': example.features,
                'label': example.label
            })
        
        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            ranking_service,
            request.role_id,
            training_data
        )
        
        return {
            "status": "training_started",
            "role_id": request.role_id,
            "training_samples": len(training_data),
            "message": f"Model training started for role {request.role_id}"
        }
        
    except Exception as e:
        logger.error(f"Training request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/labels")
async def add_label(request: LabelRequest):
    """
    Add a label for a candidate to improve ML model.
    
    Labels are used for training and model improvement.
    """
    try:
        # TODO: Store label in database for future training
        # For now, just return success
        
        return {
            "status": "success",
            "candidate_id": request.candidate_id,
            "role_id": request.role_id,
            "label": request.label,
            "message": "Label added successfully"
        }
        
    except Exception as e:
        logger.error(f"Label addition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Label addition failed: {str(e)}")

# Model management endpoints
@router.get("/models/{role_id}")
async def get_model_info(role_id: str):
    """Get information about a trained model for a role."""
    ranking_service = get_ranking_service()
    
    try:
        model_info = await ranking_service.get_model_info(role_id)
        
        return {
            "status": "success",
            "data": model_info
        }
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

@router.get("/models")
async def list_models():
    """List all available trained models."""
    ranking_service = get_ranking_service()
    
    try:
        # Get all available roles with models
        roles = ranking_service.model_store.list_roles()
        
        models = []
        for role_id in roles:
            model_info = await ranking_service.get_model_info(role_id)
            models.append(model_info)
        
        return {
            "status": "success",
            "data": {
                "models": models,
                "total_models": len(models)
            }
        }
        
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model listing failed: {str(e)}")

# Statistics and monitoring endpoints
@router.get("/stats/{role_id}")
async def get_ranking_stats(role_id: str):
    """Get ranking statistics for a role."""
    try:
        ranking_service = get_ranking_service()
        
        # Load model metrics if available
        try:
            metrics = ranking_service.model_store.load_metrics(role_id)
        except FileNotFoundError:
            metrics = None
        
        role_info = ranking_service.model_store.get_role_info(role_id)
        
        return {
            "status": "success",
            "data": {
                "role_id": role_id,
                "metrics": metrics,
                "role_info": role_info
            }
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.get("/health")
async def ranking_health():
    """Check ranking service health."""
    try:
        ranking_service = get_ranking_service()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "embedding_service": hasattr(ranking_service, 'embedding_service'),
            "feature_extractor": hasattr(ranking_service, 'feature_extractor'),
            "model_store": hasattr(ranking_service, 'model_store'),
            "resume_parser": hasattr(ranking_service, 'resume_parser')
        }
        
        return {
            "status": "success",
            "data": health_status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }

# Legacy endpoints for compatibility
@router.post("/")
async def rank_candidates():
    """Legacy ranking endpoint."""
    return {
        "message": "Use /ml or /batch endpoints for ML-powered ranking",
        "endpoints": {
            "ml": "POST /rank/ml - Upload files and rank",
            "batch": "POST /rank/batch - Rank from input_resumes folder",
            "train": "POST /rank/train - Train ML model",
            "models": "GET /rank/models - List trained models"
        }
    }

@router.get("/scores")
async def get_candidate_scores():
    """Legacy scores endpoint."""
    return {
        "message": "Use /models/{role_id} for model-specific information",
        "endpoints": {
            "models": "GET /rank/models - List all trained models",
            "model_info": "GET /rank/models/{role_id} - Get model information",
            "stats": "GET /rank/stats/{role_id} - Get ranking statistics"
        }
    }

# Background task functions
async def _train_model_background(ranking_service, role_id: str, training_data: List[Dict[str, Any]]):
    """Background task for model training."""
    try:
        logger.info(f"Starting background training for role: {role_id}")
        result = await ranking_service.train_model(role_id, training_data)
        logger.info(f"Training completed for role {role_id}: {result}")
    except Exception as e:
        logger.error(f"Background training failed for role {role_id}: {e}")
