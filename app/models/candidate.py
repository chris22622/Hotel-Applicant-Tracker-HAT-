"""Candidate model."""
from sqlalchemy import Column, Integer, String, Text, JSON, Float, DateTime
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin


class Candidate(Base, TimestampMixin):
    """Candidate model."""
    
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    work_auth = Column(String(100), nullable=True)
    source = Column(String(100), nullable=True, index=True)  # upload, email, etc.
    status = Column(String(50), nullable=False, default="new", index=True)
    
    # Resume data
    resume_file_path = Column(String(255), nullable=True)  # Local file path
    resume_text = Column(Text, nullable=True)
    parsed_json = Column(JSON, nullable=True)  # Structured extracted data
    
    # Computed fields
    years_total = Column(Float, nullable=True)
    current_title = Column(String(255), nullable=True)
    current_company = Column(String(255), nullable=True)
    education_level = Column(String(100), nullable=True)
    
    # ML embedding - store as JSON text instead of vector
    embedding = Column(Text, nullable=True)  # JSON string of float array
    
    # Soft delete
    soft_deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    experiences = relationship("Experience", back_populates="candidate", cascade="all, delete-orphan")
    applications = relationship("Application", back_populates="candidate")
    
    def __repr__(self) -> str:
        return f"<Candidate(id={self.id}, name='{self.full_name}')>"
