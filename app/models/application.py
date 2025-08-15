"""Application model."""
from sqlalchemy import Column, Integer, String, Text, JSON, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from app.models.base import Base, TimestampMixin


class ApplicationStage(str, enum.Enum):
    """Application stage enumeration."""
    NEW = "new"
    SCREENED = "screened"
    INTERVIEW = "interview"
    OFFER = "offer"
    HIRED = "hired"
    REJECTED = "rejected"


class ApplicationDecision(str, enum.Enum):
    """Application decision enumeration."""
    ADVANCE = "advance"
    REJECT = "reject"
    POOL = "pool"  # Keep in talent pool


class Application(Base, TimestampMixin):
    """Application model - candidate applied to role."""
    
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False, index=True)
    stage = Column(SQLEnum(ApplicationStage), nullable=False, default=ApplicationStage.NEW, index=True)
    
    # Scoring
    score_numeric = Column(Float, nullable=True)
    score_breakdown = Column(JSON, nullable=True)  # Detailed score components
    explanation = Column(Text, nullable=True)  # Human-readable explanation
    
    # Decision tracking
    decision = Column(SQLEnum(ApplicationDecision), nullable=True)
    decided_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    decided_at = Column(DateTime(timezone=True), nullable=True)
    labels = Column(JSON, nullable=True)  # ML training labels
    
    # Relationships
    candidate = relationship("Candidate", back_populates="applications")
    role = relationship("Role", back_populates="applications")
    decided_by_user = relationship("User", back_populates="applications_decided")
    
    def __repr__(self) -> str:
        return f"<Application(id={self.id}, candidate_id={self.candidate_id}, role_id={self.role_id})>"
