"""Authentication routes."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/login")
async def login():
    """Login endpoint (placeholder)."""
    return JSONResponse(content={
        "message": "Auth system coming soon",
        "status": "development"
    })

@router.post("/logout") 
async def logout():
    """Logout endpoint (placeholder)."""
    return JSONResponse(content={
        "message": "Logged out successfully"
    })

@router.get("/me")
async def get_current_user():
    """Get current user (placeholder)."""
    return JSONResponse(content={
        "user": "demo_user",
        "role": "admin",
        "status": "development"
    })
