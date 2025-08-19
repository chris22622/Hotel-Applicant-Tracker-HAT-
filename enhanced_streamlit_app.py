#!/usr/bin/env python3
"""
Enhanced Hotel AI Resume Screener - Streamlit Web Interface
Advanced web application with improved UI, real-time processing, and comprehensive analytics
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

# Try to import plotting libraries with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    plotly_available = True
    st.success("✅ Advanced charting available")
except ImportError:
    plotly_available = False
    st.warning("⚠️ Plotly not installed - using basic charts. Run Quick_Setup.bat to get advanced charts.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

# Try to import the enhanced screener
try:
    from enhanced_ai_screener import EnhancedHotelAIScreener
    enhanced_available = True
except ImportError:
    # Fallback to original screener
    try:
        from hotel_ai_screener import HotelAIScreener
        enhanced_available = False
        st.warning("⚠️ Enhanced AI features not available - using basic screener")
    except ImportError:
        st.error("❌ No screener module found!")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Hotel AI Resume Screener",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    
    .candidate-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .candidate-card.recommended {
        border-left-color: #28a745;
    }
    
    .candidate-card.highly-recommended {
        border-left-color: #007bff;
        background: #e7f3ff;
    }
    
    .candidate-card.consider {
        border-left-color: #ffc107;
        background: #fff9e6;
    }
    
    .candidate-card.not-recommended {
        border-left-color: #dc3545;
        background: #ffeaea;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'selected_position' not in st.session_state:
        st.session_state.selected_position = None

def get_available_positions():
    """Get list of available positions from the screener."""
    if enhanced_available:
        screener = EnhancedHotelAIScreener()
        return list(screener.position_intelligence.keys())
    else:
        screener = HotelAIScreener()
        return list(screener.get_hotel_job_intelligence().keys())

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp())
    saved_files = []
    
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    
    return temp_dir, saved_files

def create_candidate_card(candidate, rank):
    """Create a styled candidate card."""
    recommendation = candidate.get('recommendation', 'Unknown')
    score = candidate.get('total_score', 0)
    
    # Determine card class based on recommendation
    card_class = "candidate-card"
    if recommendation == "Highly Recommended":
        card_class += " highly-recommended"
        icon = "🌟"
    elif recommendation == "Recommended":
        card_class += " recommended"
        icon = "✅"
    elif recommendation == "Consider with Interview":
        card_class += " consider"
        icon = "⚠️"
    else:
        card_class += " not-recommended"
        icon = "❌"
    
    # Create expandable candidate card
    with st.expander(f"{icon} #{rank} {candidate.get('candidate_name', 'Unknown')} - {score*100:.1f} percent ({recommendation})"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Contact Information:**")
            st.write(f"📧 Email: {candidate.get('email', 'Not found')}")
            st.write(f"📞 Phone: {candidate.get('phone', 'Not found')}")
            st.write(f"📍 Location: {candidate.get('location', 'Not specified')}")
            st.write(f"📄 File: {candidate.get('file_name', 'Unknown')}")
            
            # Gender information
            gender = candidate.get('gender', 'Unknown')
            gender_confidence = candidate.get('gender_confidence', 0.0)
            if gender != 'Unknown':
                confidence_emoji = "🔥" if gender_confidence > 0.7 else "✅" if gender_confidence > 0.4 else "❓"
                st.write(f"👤 Gender: {gender} {confidence_emoji} ({gender_confidence:.1%} confidence)")
            else:
                st.write(f"👤 Gender: {gender}")
        
        with col2:
            st.write("**Experience & Skills:**")
            st.write(f"💼 Experience: {candidate.get('experience_years', 0)} years")
            st.write(f"⭐ Quality: {candidate.get('experience_quality', 'Unknown')}")
            skills = candidate.get('skills_found', [])
            if skills:
                st.write(f"🎯 Key Skills: {', '.join(skills[:5])}")
                if len(skills) > 5:
                    st.write(f"+ {len(skills) - 5} more skills")
        
        # Score breakdown chart
        if 'category_scores' in candidate:
            scores_data = candidate['category_scores']
            
            if plotly_available:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(scores_data.keys()),
                        y=[scores_data[key] * 100 for key in scores_data.keys()],
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    )
                ])
                fig.update_layout(
                    title="Score Breakdown",
                    xaxis_title="Category",
                    yaxis_title="Score (Percentage)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"score_breakdown_chart_{rank}")
            else:
                # Fallback to simple bar chart
                chart_data = pd.DataFrame({
                    'Category': list(scores_data.keys()),
                    'Score (Percentage)': [scores_data[key] * 100 for key in scores_data.keys()]
                })
                st.bar_chart(chart_data.set_index('Category'))

def create_analytics_dashboard(results):
    """Create comprehensive analytics dashboard."""
    if not results:
        return
    
    st.subheader("📊 Screening Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_candidates = len(results)
    avg_score = sum(r['total_score'] for r in results) / total_candidates if total_candidates > 0 else 0
    highly_recommended = sum(1 for r in results if r.get('recommendation') == 'Highly Recommended')
    recommended = sum(1 for r in results if r.get('recommendation') == 'Recommended')
    
    with col1:
        st.metric("Total Candidates", total_candidates)
    
    with col2:
        st.metric("Average Score", f"{avg_score*100:.1f}%")
    
    with col3:
        st.metric("Highly Recommended", highly_recommended)
    
    with col4:
        st.metric("Recommended", recommended)
    
    # Gender distribution metrics
    st.markdown("---")
    st.subheader("👥 Gender Analytics")
    
    gender_counts = {}
    for r in results:
        gender = r.get('gender', 'Unknown')
        gender_counts[gender] = gender_counts.get(gender, 0) + 1
    
    if gender_counts:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            male_count = gender_counts.get('Male', 0)
            st.metric("Male Candidates", male_count, f"{male_count/total_candidates*100:.1f}%" if total_candidates > 0 else "0%")
        
        with col2:
            female_count = gender_counts.get('Female', 0)
            st.metric("Female Candidates", female_count, f"{female_count/total_candidates*100:.1f}%" if total_candidates > 0 else "0%")
        
        with col3:
            unknown_count = gender_counts.get('Unknown', 0)
            st.metric("Unknown Gender", unknown_count, f"{unknown_count/total_candidates*100:.1f}%" if total_candidates > 0 else "0%")
        
        with col4:
            # Show average confidence for detected genders
            detected_candidates = [r for r in results if r.get('gender') in ['Male', 'Female']]
            if detected_candidates:
                avg_confidence = sum(r.get('gender_confidence', 0) for r in detected_candidates) / len(detected_candidates)
                st.metric("Avg. Detection Confidence", f"{avg_confidence*100:.1f}%")
            else:
                st.metric("Detection Confidence", "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        scores = [r['total_score'] * 100 for r in results]
        
        if plotly_available:
            fig = px.histogram(
                x=scores,
                nbins=10,
                title="Score Distribution",
                labels={'x': 'Score (Percentage)', 'y': 'Number of Candidates'},
                color_discrete_sequence=['#2a5298']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="score_distribution_chart")
        else:
            # Fallback to simple histogram
            st.subheader("Score Distribution")
            chart_data = pd.DataFrame({'Scores': scores})
            st.bar_chart(chart_data)
    
    with col2:
        # Recommendation breakdown
        recommendations = [r.get('recommendation', 'Unknown') for r in results]
        rec_counts = pd.Series(recommendations).value_counts()
        
        if plotly_available:
            colors = {
                'Highly Recommended': '#007bff',
                'Recommended': '#28a745',
                'Consider with Interview': '#ffc107',
                'Not Recommended': '#dc3545'
            }
            
            fig = px.pie(
                values=rec_counts.values,
                names=rec_counts.index,
                title="Recommendation Distribution",
                color=rec_counts.index,
                color_discrete_map=colors
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="recommendation_distribution_chart")
        else:
            # Fallback to simple chart
            st.subheader("Recommendation Distribution")
            st.bar_chart(rec_counts)
    
    # Category scores heatmap
    if results and 'category_scores' in results[0]:
        st.subheader("📈 Category Performance Analysis")
        
        if plotly_available:
            # Prepare data for heatmap
            candidates = [r.get('candidate_name', f"Candidate {i+1}") for i, r in enumerate(results[:10])]
            categories = list(results[0]['category_scores'].keys())
            
            heatmap_data = []
            for candidate in results[:10]:
                row = [candidate['category_scores'][cat] * 100 for cat in categories]
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=categories,
                y=candidates,
                colorscale='RdYlBu_r',
                textfont={"size": 10},
                showscale=True
            ))
            
            fig.update_layout(
                title="Top 10 Candidates - Category Scores Heatmap",
                xaxis_title="Score Categories",
                yaxis_title="Candidates",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True, key="category_heatmap_chart")
        else:
            # Fallback to simple table
            candidates_data = []
            for i, candidate in enumerate(results[:10], 1):
                row = {'Rank': i, 'Candidate': candidate.get('candidate_name', f"Candidate {i}")}
                if 'category_scores' in candidate:
                    for category, score in candidate['category_scores'].items():
                        row[f"{category.title()} Score"] = f"{score * 100:.1f}"
                candidates_data.append(row)
            
            df_categories = pd.DataFrame(candidates_data)
            st.dataframe(df_categories, use_container_width=True)

def export_results_to_excel(results, position):
    """Export results to Excel and provide download link."""
    if not results:
        return None
    
    # Create Excel data
    excel_data = []
    for i, candidate in enumerate(results, 1):
        excel_data.append({
            'Rank': i,
            'Name': candidate.get('candidate_name', 'Unknown'),
            'Email': candidate.get('email', 'Not found'),
            'Phone': candidate.get('phone', 'Not found'),
            'Location': candidate.get('location', 'Not specified'),
            'Total Score': f"{candidate.get('total_score', 0)*100:.1f}%",
            'Recommendation': candidate.get('recommendation', 'Unknown'),
            'Experience Years': candidate.get('experience_years', 0),
            'Experience Quality': candidate.get('experience_quality', 'Unknown'),
            'Skills Count': len(candidate.get('skills_found', [])),
            'Key Skills': ', '.join(candidate.get('skills_found', [])[:5]),
            'File Name': candidate.get('file_name', 'Unknown')
        })
    
    df = pd.DataFrame(excel_data)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Screening Results', index=False)
        
        # Add category scores if available
        if results and 'category_scores' in results[0]:
            category_data = []
            for candidate in results:
                row = {'Name': candidate.get('candidate_name', 'Unknown')}
                row.update({f"{k.title()} Score": f"{v*100:.1f} percent" for k, v in candidate.get('category_scores', {}).items()})
                category_data.append(row)
            
            df_categories = pd.DataFrame(category_data)
            df_categories.to_excel(writer, sheet_name='Category Scores', index=False)
    
    output.seek(0)
    return output

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏨 Hotel AI Resume Screener</h1>
        <p>Advanced AI-powered candidate screening for hospitality positions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Screening Configuration")
    
    # Position selection
    available_positions = get_available_positions()
    selected_position = st.sidebar.selectbox(
        "Select Position to Fill:",
        available_positions,
        help="Choose the hotel position you're hiring for"
    )
    st.session_state.selected_position = selected_position
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        max_candidates = st.number_input(
            "Maximum Candidates to Display:",
            min_value=5,
            max_value=100,
            value=20,
            help="Limit the number of candidates shown in results"
        )
        
        score_threshold = st.slider(
            "Minimum Score Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Only show candidates above this score"
        )
        
        # Gender filter
        gender_filter = st.selectbox(
            "Gender Filter:",
            options=["All", "Male", "Female", "Unknown"],
            index=0,
            help="Filter candidates by detected gender"
        )
        
        auto_export = st.checkbox(
            "Auto-export to Excel",
            value=True,
            help="Automatically generate Excel report"
        )
    
    # File upload section
    st.subheader("📂 Upload Resume Files")
    
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload resumes in PDF, Word, text, or image format"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
        
        # Show uploaded files
        with st.expander("📋 Uploaded Files"):
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size:,} bytes)")
    
    # Processing section
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if st.button("🚀 Start Screening", disabled=not uploaded_files or not selected_position):
            if uploaded_files and selected_position:
                with st.spinner("🔍 Processing resumes... This may take a few minutes."):
                    try:
                        # Save files temporarily
                        temp_dir, saved_files = save_uploaded_files(uploaded_files)
                        
                        # Initialize screener
                        if enhanced_available:
                            screener = EnhancedHotelAIScreener(str(temp_dir))
                        else:
                            screener = HotelAIScreener(str(temp_dir))
                        
                        # Process candidates
                        results = screener.screen_candidates(selected_position, max_candidates)
                        
                        # Filter by score threshold
                        results = [r for r in results if r['total_score'] >= score_threshold]
                        
                        # Filter by gender if specified
                        if gender_filter != "All":
                            results = [r for r in results if r.get('gender', 'Unknown') == gender_filter]
                        
                        st.session_state.screening_results = results
                        
                        # Clean up temporary files
                        import shutil
                        shutil.rmtree(temp_dir)
                        
                        st.success(f"✅ Screening complete! Found {len(results)} qualified candidates.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.error("Please check your files and try again.")
    
    # Results section
    if st.session_state.screening_results:
        results = st.session_state.screening_results
        
        st.markdown("---")
        st.subheader(f"🏆 Screening Results for {selected_position}")
        
        # Analytics dashboard
        create_analytics_dashboard(results)
        
        st.markdown("---")
        st.subheader("👥 Candidate Rankings")
        
        # Display candidates
        for i, candidate in enumerate(results, 1):
            create_candidate_card(candidate, i)
        
        # Export section
        st.markdown("---")
        st.subheader("📊 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Excel Report"):
                excel_data = export_results_to_excel(results, selected_position)
                if excel_data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screening_results_{selected_position}_{timestamp}.xlsx"
                    
                    st.download_button(
                        label="💾 Download Excel File",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            if st.button("📋 Generate Text Report"):
                if enhanced_available:
                    screener = EnhancedHotelAIScreener()
                    report = screener.generate_report(results, selected_position)
                else:
                    # Generate basic report
                    report = f"""
Hotel AI Resume Screener Report
Position: {selected_position}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Top Candidates:
"""
                    for i, candidate in enumerate(results[:10], 1):
                        report += f"\n{i}. {candidate.get('candidate_name', 'Unknown')} - {candidate.get('total_score', 0)*100:.1f} percent"
                        report += f"\n   Email: {candidate.get('email', 'Not found')}"
                        report += f"\n   Phone: {candidate.get('phone', 'Not found')}"
                        report += f"\n   Recommendation: {candidate.get('recommendation', 'Unknown')}\n"
                
                st.download_button(
                    label="📄 Download Text Report",
                    data=report,
                    file_name=f"screening_report_{selected_position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🔄 Clear Results"):
                st.session_state.screening_results = None
                st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏨 Hotel AI Resume Screener - Enhanced Version 2.0</p>
        <p>Built with ❤️ for hospitality professionals</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
