"""User model."""
from sqlalchemy import Column, Integer, String, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from app.models.base import Base, TimestampMixin


class UserRole(str, enum.Enum):
    """User role enumeration."""
    ADMIN = "admin"
    HR = "hr"
    MANAGER = "manager"


class User(Base, TimestampMixin):
    """User model."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.HR)
    
    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user")
    applications_decided = relationship("Application", back_populates="decided_by_user")
