"""Main FastAPI application."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import settings
from app.routes import (
    pages, auth, roles, candidates, applications, 
    ingest, rank, export, health
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="HR Applicant Tracking System",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(pages.router, tags=["pages"])
app.include_router(roles.router, prefix="/api/roles", tags=["roles"])
app.include_router(candidates.router, prefix="/api/candidates", tags=["candidates"])
app.include_router(applications.router, prefix="/api/applications", tags=["applications"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(rank.router, prefix="/api/rank", tags=["ranking"])
app.include_router(export.router, prefix="/api/export", tags=["export"])


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("HR ATS application starting up", version=settings.APP_VERSION)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("HR ATS application shutting down")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    if request.url.path.startswith("/api/"):
        return {"error": "Not found"}
    return templates.TemplateResponse(
        "404.html", 
        {"request": request}, 
        status_code=404
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error("Internal server error", exc_info=exc)
    if request.url.path.startswith("/api/"):
        return {"error": "Internal server error"}
    return templates.TemplateResponse(
        "500.html", 
        {"request": request}, 
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
