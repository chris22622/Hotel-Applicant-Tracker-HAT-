"""Candidate management routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, Candidate

router = APIRouter()

@router.get("/")
async def list_candidates(db: Session = Depends(get_db)):
    """List all candidates."""
    candidates = db.query(Candidate).all()
    return {
        "candidates": [
            {
                "id": candidate.id,
                "name": candidate.name,
                "email": candidate.email,
                "phone": candidate.phone,
                "experience_years": candidate.experience_years,
                "score": candidate.score
            } for candidate in candidates
        ]
    }

@router.get("/{candidate_id}")
async def get_candidate(candidate_id: int, db: Session = Depends(get_db)):
    """Get candidate details."""
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        return {"error": "Candidate not found"}
    
    return {
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "experience_years": candidate.experience_years,
        "score": candidate.score,
        "skills": candidate.skills,
        "resume_text": candidate.resume_text[:500] + "..." if candidate.resume_text else ""
    }

@router.post("/")
async def create_candidate(candidate_data: dict, db: Session = Depends(get_db)):
    """Create a new candidate."""
    candidate = Candidate(
        name=candidate_data.get("name"),
        email=candidate_data.get("email"),
        phone=candidate_data.get("phone", ""),
        resume_text=candidate_data.get("resume_text", ""),
        skills=candidate_data.get("skills", ""),
        experience_years=candidate_data.get("experience_years", 0),
        score=candidate_data.get("score", 0.0)
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return {"message": "Candidate created", "candidate_id": candidate.id}
