"""Application dependencies."""
from contextlib import contextmanager
from typing import Generator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

# Database setup
engine = create_engine(settings.DB_DSN, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Security
security = HTTPBearer(auto_error=False)


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Get database session context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user_id(token: Optional[str] = Depends(security)) -> Optional[int]:
    """Get current user ID from JWT token."""
    if not token:
        return None
    
    try:
        payload = jwt.decode(
            token.credentials, settings.SECRET_KEY, algorithms=["HS256"]
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None


def require_auth(current_user_id: Optional[int] = Depends(get_current_user_id)) -> int:
    """Require authentication."""
    if current_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return current_user_id


def require_admin(current_user_id: int = Depends(require_auth)) -> int:
    """Require admin role."""
    # TODO: Add role checking
    return current_user_id
