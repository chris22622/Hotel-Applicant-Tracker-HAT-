"""Application management routes."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_applications():
    """List all applications."""
    return {"applications": [], "message": "Applications endpoint ready"}

@router.post("/")
async def create_application():
    """Create a new application."""
    return {"message": "Application creation ready"}
