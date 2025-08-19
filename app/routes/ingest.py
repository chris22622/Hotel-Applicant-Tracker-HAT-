"""Resume ingestion routes."""
from fastapi import APIRouter

router = APIRouter()

@router.post("/upload")
async def upload_resume():
    """Upload and process resume."""
    return {"message": "Resume upload endpoint ready"}

@router.post("/parse")
async def parse_resume():
    """Parse resume text and extract information."""
    return {"message": "Resume parsing endpoint ready"}
