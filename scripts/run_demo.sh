#!/bin/bash
# Quick demo script for Hotel Applicant Tracker (HOT)

set -e

echo "🏨 Hotel Applicant Tracker - Quick Demo"
echo "======================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Check if sample resumes exist
if [ ! -d "input_resumes" ] || [ -z "$(ls -A input_resumes 2>/dev/null)" ]; then
    echo "❌ No sample resumes found in input_resumes/ folder"
    echo "💡 Add some resume files (PDF, DOCX, TXT) to try the demo"
    exit 1
fi

resume_count=$(ls input_resumes/*.{pdf,docx,txt,PDF,DOCX,TXT} 2>/dev/null | wc -l)
echo "📄 Found $resume_count sample resumes"

# Run quick CLI demo
echo ""
echo "🚀 Running quick CLI demo..."
python cli.py --input input_resumes/ --position front_desk_agent --output demo_results.json --quiet

if [ -f "demo_results.json" ]; then
    echo "✅ Demo completed successfully!"
    echo "📊 Results saved to demo_results.json"
else
    echo "⚠️  CLI demo had issues, but web interface should work"
fi

echo ""
echo "🌐 Starting Streamlit web interface..."
echo "📝 Instructions:"
echo "  1. Upload resumes using the file uploader"
echo "  2. Select a hotel position from dropdown"
echo "  3. Click 'Run Screening' to analyze candidates"
echo "  4. Download Excel report when complete"
echo ""
echo "🔗 Opening http://localhost:8501..."

# Launch Streamlit
streamlit run streamlit_app.py
