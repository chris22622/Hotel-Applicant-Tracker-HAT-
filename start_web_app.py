"""Start the web application with database initialization."""
import uvicorn
from app.database import init_db
from app.config import settings
import os
from pathlib import Path

def main():
    print("🏨 Starting Hotel Applicant Tracker (HAT)")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize database
    print("📊 Initializing database...")
    init_db()
    print("✅ Database ready")
    
    # Add some sample data if empty
    try:
        from app.database import SessionLocal, Role
        db = SessionLocal()
        if db.query(Role).count() == 0:
            print("📝 Adding sample roles...")
            sample_roles = [
                Role(title="Front Desk Agent", department="Front Office", 
                     description="Guest check-in/out, reservations, customer service",
                     requirements="High school diploma, customer service experience, computer skills"),
                Role(title="Housekeeping Supervisor", department="Housekeeping", 
                     description="Oversee room cleaning, quality control, staff management",
                     requirements="Experience in housekeeping, leadership skills, attention to detail"),
                Role(title="Food & Beverage Server", department="F&B", 
                     description="Restaurant service, guest interaction, order taking",
                     requirements="Food service experience, communication skills, teamwork"),
                Role(title="Maintenance Technician", department="Engineering", 
                     description="Property maintenance, repairs, preventive care",
                     requirements="Technical skills, problem-solving, maintenance experience"),
                Role(title="Guest Services Manager", department="Guest Services", 
                     description="Guest relations, complaint resolution, VIP services",
                     requirements="Management experience, customer service, problem-solving")
            ]
            for role in sample_roles:
                db.add(role)
            db.commit()
            print("✅ Sample roles added")
        db.close()
    except Exception as e:
        print(f"⚠️  Sample data error: {e}")
    
    print("\n🚀 Starting web server...")
    print(f"🌐 Open your browser to: http://localhost:8000")
    print(f"📚 API docs available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
