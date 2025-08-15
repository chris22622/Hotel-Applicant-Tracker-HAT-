#!/usr/bin/env python3
"""
Hotel AI Resume Screener - Web Interface
Simple GUI for non-technical HR staff
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import os
import zipfile
import io
from hotel_ai_screener import HotelAIScreener

# Page configuration
st.set_page_config(
    page_title="Hotel AI Resume Screener",
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
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏨 Hotel AI Resume Screener</h1>
        <p>Intelligent candidate selection for hotels and resorts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize screener
    if 'screener' not in st.session_state:
        st.session_state.screener = HotelAIScreener()
    
    screener = st.session_state.screener
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🎯 Screening Configuration")
        
        # Position selection
        available_positions = list(screener.get_hotel_job_intelligence().keys())
        position = st.selectbox(
            "Select Position",
            available_positions + ["Custom Position"],
            help="Choose from pre-configured hotel positions or enter a custom role"
        )
        
        if position == "Custom Position":
            position = st.text_input("Enter Custom Position", placeholder="e.g., Night Auditor")
        
        # Number of candidates
        num_candidates = st.number_input(
            "Number of Top Candidates",
            min_value=1,
            max_value=20,
            value=5,
            help="How many top candidates to select"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            enable_ocr = st.checkbox("Enable OCR for Scanned Documents", value=True)
            min_score = st.slider("Minimum AI Score (%)", 0, 100, 40)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Resume Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Resume Files",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT resume files for screening"
        )
        
        if uploaded_files and position:
            st.success(f"✅ {len(uploaded_files)} files uploaded for {position} screening")
            
            if st.button("🚀 Start AI Screening", type="primary"):
                with st.spinner("🤖 AI is analyzing resumes..."):
                    # Save uploaded files temporarily
                    temp_dir = Path(tempfile.mkdtemp())
                    temp_files = []
                    
                    for uploaded_file in uploaded_files:
                        temp_file = temp_dir / uploaded_file.name
                        with open(temp_file, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        temp_files.append(temp_file)
                    
                    # Process resumes
                    all_candidates = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file_path in enumerate(temp_files):
                        status_text.text(f"Processing {file_path.name}...")
                        candidate = screener.process_resume_ai(file_path)
                        all_candidates.append(candidate)
                        progress_bar.progress((i + 1) / len(temp_files))
                    
                    # AI screening
                    status_text.text("Running AI analysis...")
                    top_candidates = screener.ai_screen_candidates(position, all_candidates, num_candidates)
                    
                    # Filter by minimum score
                    qualified_candidates = [c for c in top_candidates if c.get('ai_score', 0) >= min_score]
                    
                    # Store results in session state
                    st.session_state.results = {
                        'candidates': qualified_candidates,
                        'position': position,
                        'total_processed': len(all_candidates)
                    }
                    
                    status_text.text("✅ Screening complete!")
                    progress_bar.progress(1.0)
                    
                    # Clean up temp files
                    import shutil
                    shutil.rmtree(temp_dir)
    
    with col2:
        st.header("📊 Quick Stats")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            st.metric("Position", results['position'])
            st.metric("Candidates Processed", results['total_processed'])
            st.metric("Qualified Candidates", len(results['candidates']))
            
            if results['candidates']:
                avg_score = sum(c.get('ai_score', 0) for c in results['candidates']) / len(results['candidates'])
                st.metric("Average Score", f"{avg_score:.1f}%")
        else:
            st.info("Upload resumes and run screening to see statistics")
    
    # Results section
    if 'results' in st.session_state and st.session_state.results['candidates']:
        st.header("🏆 Screening Results")
        
        candidates = st.session_state.results['candidates']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            highly_recommended = len([c for c in candidates if c.get('ai_score', 0) >= 80])
            st.metric("Highly Recommended", highly_recommended, delta=None)
        with col2:
            recommended = len([c for c in candidates if 60 <= c.get('ai_score', 0) < 80])
            st.metric("Recommended", recommended, delta=None)
        with col3:
            consider = len([c for c in candidates if 40 <= c.get('ai_score', 0) < 60])
            st.metric("Consider", consider, delta=None)
        with col4:
            avg_experience = sum(c.get('experience_years', 0) for c in candidates) / len(candidates) if candidates else 0
            st.metric("Avg Experience", f"{avg_experience:.1f} years")
        
        # Candidate cards
        st.subheader("🎯 Top Candidates")
        
        for i, candidate in enumerate(candidates, 1):
            score = candidate.get('ai_score', 0)
            score_class = "score-high" if score >= 80 else "score-medium" if score >= 60 else "score-low"
            
            with st.expander(f"#{i} {candidate.get('name', 'Unknown')} - {score:.1f}%"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📧 Email:** {candidate.get('email', 'Not found')}")
                    st.write(f"**📞 Phone:** {candidate.get('phone', 'Not found')}")
                    st.write(f"**📍 Location:** {candidate.get('location', 'Not specified')}")
                    st.write(f"**💼 Experience:** {candidate.get('experience_years', 0)} years")
                    
                    if candidate.get('strengths'):
                        st.write("**💪 Strengths:**")
                        for strength in candidate['strengths']:
                            st.write(f"• {strength}")
                    
                    if candidate.get('weaknesses'):
                        st.write("**⚠️ Areas for Development:**")
                        for weakness in candidate['weaknesses']:
                            st.write(f"• {weakness}")
                
                with col2:
                    st.markdown(f"<p class='{score_class}'>AI Score: {score:.1f}%</p>", unsafe_allow_html=True)
                    st.write(f"**Recommendation:** {candidate.get('recommendation', 'No recommendation')}")
                    
                    if candidate.get('skills'):
                        st.write("**Skills:**")
                        skills_text = ", ".join(candidate['skills'][:5])  # Show first 5 skills
                        st.write(skills_text)
                    
                    if candidate.get('role_specific_skills'):
                        st.write("**Role-Specific Skills:**")
                        role_skills = ", ".join(candidate['role_specific_skills'][:3])
                        st.write(role_skills)
        
        # Export options
        st.header("📄 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Excel Report"):
                # Create temporary output folder
                temp_output = Path(tempfile.mkdtemp())
                results_file = screener.export_ai_results(
                    candidates, temp_output, st.session_state.results['position'], len(candidates)
                )
                
                # Read the file and provide download
                with open(results_file, 'rb') as f:
                    st.download_button(
                        label="📥 Download Excel File",
                        data=f.read(),
                        file_name=f"Hotel_Screening_Results_{st.session_state.results['position'].replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            if st.button("📋 Download Contact List"):
                # Create contact list CSV
                contact_data = []
                for i, candidate in enumerate(candidates, 1):
                    contact_data.append({
                        'Rank': i,
                        'Name': candidate.get('name', 'Not found'),
                        'Phone': candidate.get('phone', 'Not found'),
                        'Email': candidate.get('email', 'Not found'),
                        'AI Score (%)': candidate.get('ai_score', 0),
                        'Recommendation': candidate.get('recommendation', 'No recommendation')
                    })
                
                contact_df = pd.DataFrame(contact_data)
                csv = contact_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Contact CSV",
                    data=csv,
                    file_name=f"Top_Candidates_Contact_List.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📊 View Detailed Analysis"):
                st.subheader("📈 Detailed Screening Analysis")
                
                # Create analysis dataframe
                analysis_data = []
                for candidate in candidates:
                    analysis_data.append({
                        'Name': candidate.get('name', 'Unknown'),
                        'AI Score': candidate.get('ai_score', 0),
                        'Experience': f"{candidate.get('experience_years', 0)} years",
                        'Skills Count': len(candidate.get('skills', [])),
                        'Role Skills': len(candidate.get('role_specific_skills', [])),
                        'Recommendation': candidate.get('recommendation', 'No recommendation')
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
                
                # Score distribution chart
                st.subheader("📊 Score Distribution")
                score_data = [candidate.get('ai_score', 0) for candidate in candidates]
                st.bar_chart(pd.DataFrame({'AI Scores': score_data}))
    
    # Help section
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        ### Getting Started
        1. **Select Position**: Choose from pre-configured hotel roles or enter a custom position
        2. **Upload Resumes**: Add PDF, DOCX, or TXT resume files
        3. **Configure Settings**: Set number of candidates and minimum score threshold
        4. **Run Screening**: Click "Start AI Screening" to analyze resumes
        5. **Review Results**: Examine top candidates with AI scores and recommendations
        6. **Export Data**: Download Excel reports or contact lists for hiring decisions
        
        ### AI Scoring
        - **80-100%**: Highly Recommended - Excellent match for the position
        - **60-79%**: Recommended - Good candidate with minor gaps  
        - **40-59%**: Consider - Meets basic requirements
        - **Below 40%**: Not Recommended - Significant gaps in requirements
        
        ### Features
        - ✅ **Smart Text Extraction**: Handles PDF, DOCX, and text files
        - ✅ **OCR Support**: Processes scanned documents and images
        - ✅ **Role Intelligence**: Pre-configured requirements for hotel positions
        - ✅ **AI Analysis**: Comprehensive candidate evaluation and scoring
        - ✅ **Export Options**: Excel reports and CSV contact lists
        - ✅ **Privacy First**: All processing happens locally, no data sent externally
        """)

if __name__ == "__main__":
    main()
