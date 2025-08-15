#!/usr/bin/env python3
"""
Demo script for Hotel AI Resume Screener
Shows the key capabilities and sample workflow
"""

from typing import List, Dict, Any

def demo_hotel_positions():
    """Display available hotel positions"""
    print("🏨 HOTEL AI RESUME SCREENER - DEMO")
    print("=" * 50)
    print("\n📋 Available Hotel Positions:")
    
    positions = [
        "Front Desk Agent", "Guest Services", "Concierge",
        "Hotel Manager", "Assistant Manager", "Night Auditor",
        "Executive Chef", "Sous Chef", "Line Cook",
        "Restaurant Manager", "Server", "Bartender",
        "Housekeeping Manager", "Housekeeper", "Maintenance",
        "Security Officer", "Sales Manager", "Event Coordinator"
    ]
    
    for i, position in enumerate(positions, 1):
        print(f"  {i:2d}. {position}")
    
    return positions

def demo_screening_process():
    """Demonstrate the AI screening workflow"""
    print("\n🤖 AI SCREENING PROCESS:")
    print("-" * 30)
    
    steps = [
        "📄 Parse resumes (PDF, Word, Text)",
        "🔍 Extract key information",
        "⚡ Apply position-specific intelligence",
        "📊 Calculate compatibility scores", 
        "🎯 Rank candidates by fit",
        "📧 Extract contact information",
        "📁 Organize results for hiring team"
    ]
    
    for step in steps:
        print(f"  ✅ {step}")

def demo_sample_analysis():
    """Show sample candidate analysis"""
    print("\n📊 SAMPLE CANDIDATE ANALYSIS:")
    print("-" * 35)
    
    sample_candidates: List[Dict[str, Any]] = [
        {
            "name": "Sarah Mitchell",
            "score": 94.2,
            "email": "sarah.m@email.com",
            "phone": "(555) 123-4567",
            "strengths": ["5+ years hotel experience", "PMS systems expert", "Trilingual"]
        },
        {
            "name": "David Rodriguez", 
            "score": 88.7,
            "email": "d.rodriguez@email.com",
            "phone": "(555) 987-6543",
            "strengths": ["Customer service excellence", "Team leadership", "Hospitality degree"]
        },
        {
            "name": "Emma Thompson",
            "score": 82.3, 
            "email": "emma.t@email.com",
            "phone": "(555) 456-7890",
            "strengths": ["Strong communication", "Quick learner", "Flexible schedule"]
        }
    ]
    
    for i, candidate in enumerate(sample_candidates, 1):
        status = "🏆 HIGHLY RECOMMENDED" if candidate["score"] >= 90 else "✅ RECOMMENDED"
        print(f"\n  {i}. {candidate['name']} - {candidate['score']:.1f}% {status}")
        print(f"     📧 {candidate['email']} | 📞 {candidate['phone']}")
        strengths_list = candidate['strengths']
        if isinstance(strengths_list, list):
            print(f"     💪 Strengths: {', '.join(strengths_list)}")

def demo_output_formats():
    """Show available output formats"""
    print("\n📁 OUTPUT FORMATS:")
    print("-" * 20)
    
    outputs = [
        "📊 Excel Workbook with candidate contact sheets",
        "📋 CSV files for easy data import", 
        "📂 Organized folders with top candidate resumes",
        "📈 Detailed scoring breakdown reports",
        "📧 Ready-to-use contact lists"
    ]
    
    for output in outputs:
        print(f"  ✅ {output}")

def main():
    """Run the complete demo"""
    try:
        # Show available positions
        demo_hotel_positions()
        
        # Demonstrate screening process
        demo_screening_process()
        
        # Show sample analysis
        demo_sample_analysis()
        
        # Display output formats
        demo_output_formats()
        
        print("\n" + "=" * 50)
        print("🚀 Ready to try it yourself?")
        print("   1. Add resumes to 'input_resumes/' folder")
        print("   2. Run: python hotel_ai_screener.py")
        print("   3. Or use web interface: streamlit run streamlit_app.py")
        print("\n💡 Perfect for hotels looking to streamline hiring! 🏨✨")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for watching!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
