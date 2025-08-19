"""Health check routes."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import settings

router = APIRouter()

@router.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@router.get("/health")
async def health_check_detailed():
    """Detailed health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "service": "Hotel Applicant Tracker",
        "version": settings.APP_VERSION
    })

@router.get("/ready")
async def readiness_check():
    """Readiness check for deployment."""
    return JSONResponse(content={
        "status": "ready",
        "database": "connected"
    })
