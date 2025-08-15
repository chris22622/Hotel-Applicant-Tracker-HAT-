"""Celery tasks for background processing."""
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.workers.celery_app import celery
from app.services import storage, parse, extract, embed, rank
from app.models import candidate as M_candidate, application as M_application, role as M_role
from app.deps import db_session


@celery.task(name="ingest_file", bind=True)
def ingest_file(self, file_key: str, filename: str, source: str = "upload") -> Dict[str, Any]:
    """Process uploaded resume file."""
    try:
        # Parse text from file
        text, parse_confidence = parse.extract_text(file_key)
        
        if not text or parse_confidence < 0.1:
            return {"error": "Failed to extract text from file", "file_key": file_key}
        
        # Extract structured data
        parsed_data = extract.extract_structured(text)
        
        # Generate embedding
        embedding = embed.embed(text)
        
        # Save to database
        with db_session() as db:
            # Check if candidate already exists (by email)
            existing = None
            email = parsed_data.get('email')
            if email:
                existing = db.query(M_candidate.Candidate).filter(
                    M_candidate.Candidate.email == email
                ).first()
            
            if existing:
                # Update existing candidate
                candidate = existing
                candidate.resume_file_key = file_key
                candidate.resume_text = text
                candidate.parsed_json = parsed_data
                candidate.embedding = embedding.tolist()
                candidate.years_total = parsed_data.get('years_total')
                candidate.current_title = parsed_data.get('current_title')
                candidate.current_company = parsed_data.get('current_company')
            else:
                # Create new candidate
                candidate = M_candidate.Candidate(
                    full_name=parsed_data.get('name') or 'Unknown',
                    email=parsed_data.get('email'),
                    phone=parsed_data.get('phone'),
                    source=source,
                    resume_file_key=file_key,
                    resume_text=text,
                    parsed_json=parsed_data,
                    years_total=parsed_data.get('years_total'),
                    current_title=parsed_data.get('current_title'),
                    current_company=parsed_data.get('current_company'),
                    embedding=embedding.tolist()
                )
                db.add(candidate)
            
            db.commit()
            db.refresh(candidate)
            
            # Create experience records
            experiences = parsed_data.get('experiences', [])
            for exp_data in experiences:
                experience = M_candidate.Experience(
                    candidate_id=candidate.id,
                    title=exp_data.get('title', ''),
                    company=exp_data.get('company', ''),
                    start_date=exp_data.get('start_date'),
                    end_date=exp_data.get('end_date'),
                    responsibilities=exp_data.get('responsibilities', ''),
                    skills=exp_data.get('skills', [])
                )
                db.add(experience)
            
            db.commit()
        
        return {
            "success": True,
            "candidate_id": candidate.id,
            "file_key": file_key,
            "parse_confidence": parse_confidence
        }
        
    except Exception as e:
        return {"error": str(e), "file_key": file_key}


@celery.task(name="rank_role", bind=True)
def rank_role(self, role_id: int) -> Dict[str, Any]:
    """Rank all candidates for a specific role."""
    try:
        with db_session() as db:
            # Get role
            role = db.query(M_role.Role).filter(M_role.Role.id == role_id).first()
            if not role:
                return {"error": f"Role {role_id} not found"}
            
            # Generate role embedding
            role_embedding = embed.embed_role(
                role.title,
                role.description or "",
                role.must_have or [],
                role.nice_to_have or []
            )
            
            # Get all active candidates with embeddings
            candidates = db.query(M_candidate.Candidate).filter(
                M_candidate.Candidate.embedding.isnot(None),
                M_candidate.Candidate.soft_deleted_at.is_(None)
            ).all()
            
            ranked_results = []
            
            for candidate in candidates:
                # Prepare candidate data for ranking
                candidate_data = {
                    'years_total': candidate.years_total,
                    'skills': candidate.parsed_json.get('skills', []) if candidate.parsed_json else [],
                    'experiences': candidate.parsed_json.get('experiences', []) if candidate.parsed_json else []
                }
                
                # Prepare role data
                role_data = {
                    'min_years': role.min_years,
                    'must_have': role.must_have or [],
                    'nice_to_have': role.nice_to_have or [],
                    'knock_outs': role.knock_outs or []
                }
                
                # Convert embedding back to numpy array
                import numpy as np
                candidate_embedding = np.array(candidate.embedding, dtype=np.float32)
                
                # Rank candidate
                score, breakdown, explanation = rank.rank_candidate(
                    candidate_data, role_data,
                    candidate_embedding, role_embedding
                )
                
                # Check if application already exists
                existing_app = db.query(M_application.Application).filter(
                    M_application.Application.candidate_id == candidate.id,
                    M_application.Application.role_id == role_id
                ).first()
                
                if existing_app:
                    # Update existing application
                    existing_app.score_numeric = score
                    existing_app.score_breakdown = breakdown
                    existing_app.explanation = explanation
                else:
                    # Create new application
                    application = M_application.Application(
                        candidate_id=candidate.id,
                        role_id=role_id,
                        score_numeric=score,
                        score_breakdown=breakdown,
                        explanation=explanation
                    )
                    db.add(application)
                
                ranked_results.append({
                    'candidate_id': candidate.id,
                    'score': score,
                    'explanation': explanation
                })
            
            db.commit()
            
            # Sort by score
            ranked_results.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                "success": True,
                "role_id": role_id,
                "candidates_ranked": len(ranked_results),
                "top_candidates": ranked_results[:10]
            }
            
    except Exception as e:
        return {"error": str(e), "role_id": role_id}


@celery.task(name="train_reranker", bind=True)
def train_reranker(self) -> Dict[str, Any]:
    """Train the ML reranker model weekly."""
    try:
        with db_session() as db:
            # Get labeled applications
            labeled_apps = db.query(M_application.Application).filter(
                M_application.Application.decision.isnot(None),
                M_application.Application.score_breakdown.isnot(None)
            ).all()
            
            if len(labeled_apps) < 10:
                return {"message": "Insufficient labeled data for training", "count": len(labeled_apps)}
            
            # Prepare training data
            features = []
            labels = []
            
            for app in labeled_apps:
                breakdown = app.score_breakdown
                if not breakdown:
                    continue
                
                # Feature vector: [semantic, must_coverage, nice_coverage, recency, tenure, seniority_align]
                feature_vector = [
                    breakdown.get('semantic', 0),
                    breakdown.get('must_coverage', 0),
                    breakdown.get('nice_coverage', 0),
                    breakdown.get('recency', 0),
                    breakdown.get('tenure', 0),
                    breakdown.get('seniority_align', 0)
                ]
                
                # Label: 1 for advance, 0 for reject, 0.5 for pool
                label = 1.0 if app.decision == M_application.ApplicationDecision.ADVANCE else 0.0
                if app.decision == M_application.ApplicationDecision.POOL:
                    label = 0.5
                
                features.append(feature_vector)
                labels.append(label)
            
            # Train reranker
            success = rank.reranker.train(features, labels)
            
            return {
                "success": success,
                "training_samples": len(features),
                "message": "Model trained successfully" if success else "Training failed"
            }
            
    except Exception as e:
        return {"error": str(e)}


@celery.task(name="cleanup_old_files", bind=True)
def cleanup_old_files(self, days_old: int = 30) -> Dict[str, Any]:
    """Clean up old resume files from storage."""
    try:
        # TODO: Implement cleanup logic
        return {"message": "Cleanup task not yet implemented"}
    except Exception as e:
        return {"error": str(e)}
