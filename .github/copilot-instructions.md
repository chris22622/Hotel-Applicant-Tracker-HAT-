# HR Applicant Tracker (HAT) - Project Instructions

## Project Overview
HR Applicant Tracker (HAT) is a comprehensive, FREE, local-first Applicant Tracking System built with FastAPI and SQLite. The system has been customized specifically for **Royalton Resort** hiring needs.

## Completed Features
- [x] Full FastAPI application with SQLite database
- [x] Resume parsing (PDF, DOCX, TXT)
- [x] Semantic search and candidate ranking
- [x] Standalone resume screening tool for Royalton Resort
- [x] Excel export with detailed candidate analysis
- [x] Automated file organization and copying
- [x] Pre-configured resort positions and requirements
- [x] Batch processing capabilities
- [x] 100% local operation (no cloud dependencies)

## Key Components
1. **Web Application** (`app/`): Full-featured ATS with web interface
2. **Standalone Screener** (`royalton_screener.py`): Batch resume processing tool
3. **Setup Scripts**: Easy installation and launch batch files
4. **Database**: SQLite with candidate, role, and application models
5. **Templates**: Bootstrap-based responsive UI

## Usage Instructions
### Standalone Screener (Recommended for Quick Hiring):
1. Run `Setup_Royalton_Screener.bat` (one-time setup)
2. Put resumes in `input_resumes/` folder
3. Run `Start_Royalton_Screener.bat`
4. Enter position and get organized results

### Full Web Application:
1. Install dependencies: `pip install -e .`
2. Initialize database: `python setup_local.py`
3. Start server: `uvicorn app.main:app --reload`
4. Access at `http://localhost:8000`

## Development Notes
- Converted from Docker/PostgreSQL to local SQLite for cost-free operation
- Removed complex dependencies (Celery, Redis, MinIO)
- Optimized for Royalton Resort positions and workflows
- Supports PDF, DOCX, TXT resume formats
- Includes comprehensive scoring algorithms for hospitality roles
