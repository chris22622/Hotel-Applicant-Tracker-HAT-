#!/usr/bin/env python3
"""Setup script for local development without Docker."""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.models.base import Base
from app.deps import get_db


async def create_database():
    """Create SQLite database and tables."""
    settings = get_settings()
    
    print("🗄️ Creating SQLite database...")
    
    # Create async engine
    engine = create_async_engine(settings.DB_DSN)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database created successfully!")
    
    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    print(f"✅ Upload directory created: {upload_dir}")


def seed_initial_data():
    """Seed initial user and roles."""
    print("🌱 Seeding initial data...")
    
    # Import here to avoid circular imports
    from app.models.user import User
    from app.models.role import Role
    from passlib.context import CryptContext
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    
    settings = get_settings()
    
    # Create sync engine for seeding
    sync_db_url = settings.DB_DSN.replace("+aiosqlite", "")
    engine = create_engine(sync_db_url)
    Session = sessionmaker(bind=engine)
    
    with Session() as session:
        # Create admin user if not exists
        admin = session.query(User).filter(User.email == "admin@company.com").first()
        if not admin:
            pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
            hashed_password = pwd_context.hash("admin123")
            
            admin = User(
                email="admin@company.com",
                password_hash=hashed_password,
                role="admin"
            )
            session.add(admin)
            print("✅ Created admin user: admin@company.com / admin123")
        
        # Create sample role if not exists
        role = session.query(Role).filter(Role.title == "Software Engineer").first()
        if not role:
            role = Role(
                title="Software Engineer",
                description="Develop and maintain software applications",
                department="Engineering",
                location="Remote",
                min_years=2,
                salary_band_min=70000,
                salary_band_max=120000,
                must_have=["Python", "JavaScript", "SQL"],
                nice_to_have=["React", "Docker", "AWS"],
                knock_outs=["No work authorization"],
                version=1
            )
            session.add(role)
            print("✅ Created sample role: Software Engineer")
        
        session.commit()
    
    print("✅ Initial data seeded!")


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        print("⚠️  Warning: Not in a virtual environment!")
        print("   Recommended: python -m venv .venv && .venv\\Scripts\\activate")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Install packages
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", "."
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Dependencies installed!")
        return True
    else:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False


def main():
    """Main setup function."""
    print("🏢 HR Applicant Tracker (HAT) - Local Setup")
    print("=" * 45)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create database
    asyncio.run(create_database())
    
    # Seed data
    seed_initial_data()
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To start the application:")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("\n🌐 Then visit: http://localhost:8000")
    print("📝 Login: admin@company.com / admin123")


if __name__ == "__main__":
    main()
