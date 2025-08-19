"""Export routes."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/excel")
async def export_excel():
    """Export candidates to Excel format."""
    return {"message": "Excel export endpoint ready"}

@router.get("/pdf")
async def export_pdf():
    """Export candidate report to PDF."""
    return {"message": "PDF export endpoint ready"}
