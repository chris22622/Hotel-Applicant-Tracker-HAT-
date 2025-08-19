"""Role management routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, Role
from typing import List

router = APIRouter()

@router.get("/")
async def list_roles(db: Session = Depends(get_db)):
    """List all roles."""
    roles = db.query(Role).all()
    return {
        "roles": [
            {
                "id": role.id,
                "title": role.title,
                "department": role.department,
                "description": role.description,
                "requirements": role.requirements
            } for role in roles
        ]
    }

@router.get("/{role_id}")
async def get_role(role_id: int, db: Session = Depends(get_db)):
    """Get role by ID."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        return {"error": "Role not found"}
    
    return {
        "id": role.id,
        "title": role.title,
        "department": role.department,
        "description": role.description,
        "requirements": role.requirements,
        "created_at": role.created_at
    }

@router.post("/")
async def create_role(role_data: dict, db: Session = Depends(get_db)):
    """Create a new role."""
    role = Role(
        title=role_data.get("title"),
        department=role_data.get("department", "General"),
        description=role_data.get("description", ""),
        requirements=role_data.get("requirements", "")
    )
    db.add(role)
    db.commit()
    db.refresh(role)
    return {"message": "Role created", "role_id": role.id}
