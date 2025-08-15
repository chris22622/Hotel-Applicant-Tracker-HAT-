"""Experience model."""
from sqlalchemy import Column, Integer, String, Text, JSON, Date, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import Base


class Experience(Base):
    """Work experience model."""
    
    __tablename__ = "experiences"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)  # NULL means current
    responsibilities = Column(Text, nullable=True)
    skills = Column(JSON, nullable=True)  # List of skills used
    
    # Relationships
    candidate = relationship("Candidate", back_populates="experiences")
    
    def __repr__(self) -> str:
        return f"<Experience(id={self.id}, title='{self.title}', company='{self.company}')>"
