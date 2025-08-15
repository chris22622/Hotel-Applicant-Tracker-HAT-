#!/usr/bin/env python3
"""
Royalton Resort AI Screener - Web Interface
Simple GUI for non-technical HR staff
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
import zipfile
import io
from royalton_ai_screener import RoyaltonAIScreener

# Page configuration
st.set_page_config(
    page_title="Royalton Resort AI Screener",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #4a90e2);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a90e2;
        margin: 0.5rem 0;
    }
    .candidate-card {
        background: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏨 Royalton Resort AI Screener</h1>
        <p>Intelligent candidate selection powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Screening Setup")
    
    # Position selection
    positions = [
        "Front Desk Agent", "Chef", "Bartender", "Housekeeping Supervisor",
        "Security Officer", "Spa Therapist", "Restaurant Server", "Concierge",
        "Guest Services Manager", "Maintenance Technician", "Sales Manager"
    ]
    
    position = st.sidebar.selectbox(
        "🎯 Select Position to Fill:",
        positions,
        help="Choose the position you're hiring for"
    )
    
    # Number of candidates
    num_candidates = st.sidebar.number_input(
        "👥 Number of Candidates Needed:",
        min_value=1,
        max_value=20,
        value=5,
        help="How many top candidates do you want to interview?"
    )
    
    # File upload
    st.sidebar.markdown("### 📁 Upload Resumes")
    uploaded_files = st.sidebar.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'txt', 'doc'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT resume files"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤖 AI Screening Results")
        
        if uploaded_files and st.sidebar.button("🚀 Start AI Screening", type="primary"):
            run_screening(uploaded_files, position, num_candidates)
        elif not uploaded_files:
            st.info("👆 Please upload resume files in the sidebar to begin screening")
        else:
            st.info("👆 Click 'Start AI Screening' when ready")
    
    with col2:
        st.header("ℹ️ How it Works")
        st.markdown("""
        **🎯 Smart Matching**
        - AI analyzes each resume
        - Matches skills to job requirements
        - Evaluates experience relevance
        
        **📊 Intelligent Scoring**
        - Experience weight: 30-40%
        - Skills match: 25-30%
        - Cultural fit: 15-25%
        - Hospitality background: 15-20%
        
        **📈 Results**
        - Top candidates ranked
        - Detailed explanations
        - Excel & CSV exports
        - Organized file copies
        """)

def run_screening(uploaded_files, position, num_candidates):
    """Run the AI screening process."""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input_resumes"
            input_dir.mkdir(exist_ok=True)
            
            # Save uploaded files
            status_text.text("📁 Saving uploaded files...")
            progress_bar.progress(10)
            
            for uploaded_file in uploaded_files:
                file_path = input_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Initialize screener
            status_text.text("🤖 Initializing AI screener...")
            progress_bar.progress(20)
            
            # Mock the screener initialization for this temp directory
            screener = RoyaltonAIScreener()
            screener.input_dir = input_dir
            screener.output_dir = temp_path / "screening_results"
            screener.output_dir.mkdir(exist_ok=True)
            
            # Get resume files
            status_text.text("📄 Analyzing resume files...")
            progress_bar.progress(30)
            
            resume_files = screener.get_resume_files()
            if not resume_files:
                st.error("❌ No valid resume files found")
                return
            
            # Run screening
            status_text.text(f"🧠 AI analyzing {len(resume_files)} candidates...")
            progress_bar.progress(50)
            
            candidates = []
            for i, file_path in enumerate(resume_files):
                candidate = screener.process_single_resume(file_path, position.lower().replace(' ', '_'))
                if candidate:
                    candidates.append(candidate)
                progress_bar.progress(50 + (40 * (i + 1) // len(resume_files)))
            
            # Remove duplicates and sort
            status_text.text("🔄 Processing results...")
            progress_bar.progress(90)
            
            candidates = screener.remove_duplicates(candidates)
            candidates = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
            
            # Display results
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            display_results(candidates, position, num_candidates, screener, temp_path)
            
    except Exception as e:
        st.error(f"❌ Error during screening: {str(e)}")

def display_results(candidates, position, num_candidates, screener, temp_path):
    """Display the screening results."""
    
    # Summary metrics
    st.markdown("### 📊 Screening Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_candidates = len(candidates)
    excellent_candidates = len([c for c in candidates if c.get('score', 0) >= 80])
    qualified_candidates = len([c for c in candidates if c.get('score', 0) >= 60])
    avg_score = sum(c.get('score', 0) for c in candidates) / max(len(candidates), 1)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_candidates}</h3>
            <p>Total Candidates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{excellent_candidates}</h3>
            <p>Excellent (80+)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{qualified_candidates}</h3>
            <p>Qualified (60+)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_score:.1f}</h3>
            <p>Average Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top candidates
    st.markdown(f"### 🏆 Top {num_candidates} Recommendations")
    
    for i, candidate in enumerate(candidates[:num_candidates]):
        score = candidate.get('score', 0)
        name = candidate.get('name', 'Unknown')
        email = candidate.get('email', 'No email')
        phone = candidate.get('phone', 'No phone')
        reasons = candidate.get('reasons', [])
        
        # Score color
        if score >= 80:
            score_class = "score-high"
            recommendation = "HIRE IMMEDIATELY"
            recommendation_color = "#28a745"
        elif score >= 60:
            score_class = "score-medium"
            recommendation = "INTERVIEW"
            recommendation_color = "#ffc107"
        else:
            score_class = "score-low"
            recommendation = "CONSIDER"
            recommendation_color = "#dc3545"
        
        st.markdown(f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>#{i+1} {name}</h4>
                <div>
                    <span class="{score_class}">Score: {score:.1f}</span>
                    <span style="color: {recommendation_color}; margin-left: 1rem; font-weight: bold;">{recommendation}</span>
                </div>
            </div>
            <p><strong>📧</strong> {email} | <strong>📞</strong> {phone}</p>
            <p><strong>🤖 AI Analysis:</strong> {' | '.join(reasons[:3])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Download options
    st.markdown("### 📥 Download Results")
    
    # Create Excel file
    try:
        excel_file = screener.create_ai_excel_report(candidates, temp_path, position, num_candidates)
        
        with open(excel_file, 'rb') as f:
            excel_data = f.read()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"AI_Screening_{position.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Create CSV
            df_data = []
            for candidate in candidates:
                df_data.append({
                    'Rank': candidates.index(candidate) + 1,
                    'Name': candidate.get('name', 'Unknown'),
                    'Email': candidate.get('email', ''),
                    'Phone': candidate.get('phone', ''),
                    'Score': candidate.get('score', 0),
                    'Recommendation': 'HIRE' if candidate.get('score', 0) >= 80 else 'INTERVIEW' if candidate.get('score', 0) >= 60 else 'CONSIDER',
                    'AI_Analysis': ' | '.join(candidate.get('reasons', [])[:3])
                })
            
            df = pd.DataFrame(df_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name=f"AI_Results_{position.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error creating download files: {str(e)}")

if __name__ == "__main__":
    main()
