"""Application configuration."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Security
    SECRET_KEY: str = "change-me-to-a-secure-random-key"
    JWT_EXPIRE_MIN: int = 1440
    
    # Database - SQLite instead of PostgreSQL
    DB_DSN: str = "sqlite+aiosqlite:///./hr_ats.db"
    
    # Local file storage instead of MinIO/S3
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Email
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASS: Optional[str] = None
    SMTP_TLS: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # App
    APP_NAME: str = "HR ATS"
    APP_VERSION: str = "1.0.0"
    
    # NLP Models (free)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
