"""Common error handling utilities."""
from typing import Dict, Any, Optional
from fastapi import HTTPException
from fastapi.responses import JSONResponse


def api_error(
    status_code: int,
    code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create a standardized API error response.
    
    Args:
        status_code: HTTP status code
        code: Error code (e.g., "validation_error", "not_found")
        message: Human-readable error message
        extra: Additional error details
        
    Returns:
        JSONResponse with standardized error format
    """
    payload = {
        "error": {
            "code": code,
            "message": message
        }
    }
    
    if extra:
        payload["error"].update(extra)
    
    return JSONResponse(status_code=status_code, content=payload)


def raise_api_error(
    status_code: int,
    code: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Raise an HTTPException with standardized error format.
    
    Args:
        status_code: HTTP status code
        code: Error code (e.g., "validation_error", "not_found")
        message: Human-readable error message
        extra: Additional error details
        
    Raises:
        HTTPException with standardized detail format
    """
    detail = {
        "code": code,
        "message": message
    }
    
    if extra:
        detail.update(extra)
    
    raise HTTPException(status_code=status_code, detail=detail)


# Common error responses
def not_found_error(resource: str, resource_id: str) -> JSONResponse:
    """Standard 404 error for missing resources."""
    return api_error(
        status_code=404,
        code="not_found",
        message=f"{resource} not found",
        extra={"resource_id": resource_id}
    )


def validation_error(message: str, field: Optional[str] = None) -> JSONResponse:
    """Standard 422 error for validation failures."""
    extra = {"field": field} if field else None
    return api_error(
        status_code=422,
        code="validation_error",
        message=message,
        extra=extra
    )


def unauthorized_error(message: str = "Authentication required") -> JSONResponse:
    """Standard 401 error for authentication failures."""
    return api_error(
        status_code=401,
        code="unauthorized",
        message=message
    )


def forbidden_error(message: str = "Insufficient permissions") -> JSONResponse:
    """Standard 403 error for authorization failures."""
    return api_error(
        status_code=403,
        code="forbidden",
        message=message
    )


def internal_error(message: str = "An internal server error occurred") -> JSONResponse:
    """Standard 500 error for server errors."""
    return api_error(
        status_code=500,
        code="internal_error",
        message=message
    )
