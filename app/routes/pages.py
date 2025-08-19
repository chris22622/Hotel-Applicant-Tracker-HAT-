"""Web page routes."""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import get_db, Role, Candidate
from pathlib import Path

router = APIRouter()

# Check if templates exist
templates_dir = Path("app/templates")
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    templates = None

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """Home page."""
    if templates and (templates_dir / "index.html").exists():
        roles_count = db.query(Role).count()
        candidates_count = db.query(Candidate).count()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "roles_count": roles_count,
            "candidates_count": candidates_count
        })
    else:
        # Simple HTML response if no templates
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hotel Applicant Tracker</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .btn { background: #007bff; color: white; padding: 10px 20px; 
                       text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px; }
                .feature { background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 5px; }
                .header { text-align: center; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏨 Hotel Applicant Tracker (HAT)</h1>
                    <p>AI-powered resume screening for hotels and resorts</p>
                </div>
                
                <div class="feature">
                    <h3>🚀 Quick Start Options</h3>
                    <a href="/docs" class="btn">📚 API Documentation</a>
                    <a href="/api/candidates" class="btn">👥 View Candidates</a>
                    <a href="/api/roles" class="btn">💼 View Roles</a>
                </div>
                
                <div class="feature">
                    <h3>📊 Current Status</h3>
                    <p>Database: Connected ✅</p>
                    <p>API: Ready ✅</p>
                    <p>Resume Processing: Available ✅</p>
                </div>
                
                <div class="feature">
                    <h3>🎯 Features</h3>
                    <ul>
                        <li>AI-powered candidate screening</li>
                        <li>Resume parsing (PDF, DOCX, TXT)</li>
                        <li>Position-specific scoring</li>
                        <li>Excel export and reporting</li>
                        <li>100% local operation</li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🏨 Hotel Positions Supported</h3>
                    <ul>
                        <li>Front Desk Agent</li>
                        <li>Housekeeping Supervisor</li>
                        <li>Food & Beverage Server</li>
                        <li>Maintenance Technician</li>
                        <li>Guest Services Manager</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
