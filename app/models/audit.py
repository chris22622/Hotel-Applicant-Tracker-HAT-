"""Audit log model."""
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base


class AuditLog(Base):
    """Audit log model for tracking changes."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    entity = Column(String(100), nullable=False, index=True)  # Table name
    entity_id = Column(String(100), nullable=False, index=True)  # Record ID
    action = Column(String(50), nullable=False, index=True)  # CREATE, UPDATE, DELETE
    diff = Column(JSON, nullable=True)  # Changes made
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, entity='{self.entity}', action='{self.action}')>"
