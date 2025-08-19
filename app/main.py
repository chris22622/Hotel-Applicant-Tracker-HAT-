"""Main FastAPI application (hardened)."""
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import uuid
import time

from app.config import settings
from app.routes import (
    pages, auth, roles, candidates, applications,
    ingest, rank, export, health
)

# ---------- Logging (structured, robust) ----------
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# ---------- Request ID + timing middleware ----------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "request.complete",
                method=request.method,
                path=request.url.path,
                status=getattr(response, "status_code", None),
                request_id=req_id,
                elapsed_ms=elapsed_ms,
            )
        response.headers["x-request-id"] = req_id
        return response

# ---------- Lifespan (replaces on_event) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app.startup", version=settings.APP_VERSION)
    yield
    logger.info("app.shutdown")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="HR Applicant Tracking System",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------- Middlewares ----------
# Allow only trusted hosts in prod; "*" by default for dev.
trusted_hosts = ["*"] if settings.ENV != "production" else ["example.com", "*.example.com", "localhost", "127.0.0.1"]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENV != "production" else settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(RequestContextMiddleware)

# ---------- Static & templates (don't crash if missing) ----------
static_dir = Path("app/static")
templates_dir = Path("app/templates")

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning("static.missing", directory=str(static_dir.resolve()))

templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.exists() else None
if templates is None:
    logger.warning("templates.missing", directory=str(templates_dir.resolve()))

# ---------- Routers ----------
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(roles.router, prefix="/api/roles", tags=["roles"])
app.include_router(candidates.router, prefix="/api/candidates", tags=["candidates"])
app.include_router(applications.router, prefix="/api/applications", tags=["applications"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(rank.router, prefix="/api/rank", tags=["ranking"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(pages.router, tags=["pages"])

# ---------- Root convenience ----------
@app.get("/", include_in_schema=False)
async def root():
    # If you have a landing page template, render it. Otherwise send to /docs.
    if templates and (templates_dir / "index.html").exists():
        from fastapi import Request
        return templates.TemplateResponse("index.html", {"request": Request})
    return RedirectResponse(url="/docs")

# ---------- Error handlers ----------
def _json_error(status: int, code: str, message: str, extra: Dict[str, Any] | None = None):
    payload = {"error": {"code": code, "message": message}}
    if extra:
        payload["error"].update(extra)
    return JSONResponse(status_code=status, content=payload)

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return _json_error(404, "not_found", "The requested resource was not found.")
    if templates and (templates_dir / "404.html").exists():
        return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
    return _json_error(404, "not_found", "Page not found.")

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error("server.error", path=request.url.path, exc_info=exc)
    if request.url.path.startswith("/api/"):
        return _json_error(500, "internal_error", "An internal server error occurred.")
    if templates and (templates_dir / "500.html").exists():
        return templates.TemplateResponse("500.html", {"request": request}, status_code=500)
    return _json_error(500, "internal_error", "Something went wrong.")

# ---------- Local run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
