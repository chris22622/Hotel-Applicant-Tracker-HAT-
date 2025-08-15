"""Role model."""
from sqlalchemy import Column, Integer, String, Text, JSON, Float
from sqlalchemy.orm import relationship

from app.models.base import Base, TimestampMixin


class Role(Base, TimestampMixin):
    """Job role model."""
    
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    department = Column(String(100), nullable=True, index=True)
    location = Column(String(255), nullable=True)
    min_years = Column(Integer, nullable=True)
    salary_band_min = Column(Float, nullable=True)
    salary_band_max = Column(Float, nullable=True)
    must_have = Column(JSON, nullable=True)  # List of required skills
    nice_to_have = Column(JSON, nullable=True)  # List of preferred skills
    knock_outs = Column(JSON, nullable=True)  # Hard disqualifiers
    version = Column(Integer, nullable=False, default=1)
    
    # Relationships
    applications = relationship("Application", back_populates="role")
    
    def __repr__(self) -> str:
        return f"<Role(id={self.id}, title='{self.title}')>"
