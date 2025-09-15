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
from typing import Any, Dict, List, Optional, Tuple, Type

# Try to import plotting libraries with fallbacks
# Predefine names for type checkers
px: Any
go: Any
make_subplots: Any
try:
    import plotly.express as px  # type: ignore[import-not-found]
    import plotly.graph_objects as go  # type: ignore[import-not-found]
    from plotly.subplots import make_subplots  # type: ignore[import-not-found]
    plotly_available = True
    st.success("✅ Advanced charting available")
except ImportError:
    plotly_available = False
    st.warning("⚠️ Plotly not installed - using basic charts. Run Quick_Setup.bat to get advanced charts.")
    # Assign dummies for runtime safety
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

# Try to import the enhanced screener
# Type-checker friendly optional imports
EnhancedHotelAIScreener: Optional[Type[Any]] = None
HotelAIScreener: Optional[Type[Any]] = None
enhanced_available: bool = False
try:
    from enhanced_ai_screener import EnhancedHotelAIScreener as _EnhancedHotelAIScreener
    EnhancedHotelAIScreener = _EnhancedHotelAIScreener
    enhanced_available = True
except ImportError:
    # Fallback to original screener
    try:
        from hotel_ai_screener import HotelAIScreener as _HotelAIScreener
        HotelAIScreener = _HotelAIScreener
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
    # Pagination and change-tracking defaults
    if 'page' not in st.session_state:
        st.session_state.page = 1
    for key in (
        'prev_sort_by', 'prev_search_query', 'prev_page_size',
        'prev_score_threshold', 'prev_gender_filter',
        'prev_selected_position', 'prev_include_unknown_gender'
    ):
        if key not in st.session_state:
            st.session_state[key] = None

@st.cache_data(show_spinner=False, ttl=600)
def get_available_positions() -> List[str]:
    """Get list of available positions from the screener (cached)."""
    if enhanced_available and EnhancedHotelAIScreener is not None:
        screener = EnhancedHotelAIScreener()
        return list(screener.position_intelligence.keys())
    else:
        if HotelAIScreener is None:
            return []
        screener = HotelAIScreener()
        return list(screener.get_hotel_job_intelligence().keys())

def save_uploaded_files(uploaded_files: List[Any]) -> Tuple[Path, List[Path]]:
    """Save uploaded files to temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp())
    saved_files: List[Path] = []
    
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    
    return temp_dir, saved_files

def create_candidate_card(candidate: Dict[str, Any], rank: int) -> None:
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
            # Role evidence
            if candidate.get('explicit_role_evidence'):
                details = candidate.get('role_evidence_details', {}) or {}
                matched_titles = ", ".join(details.get('matched_titles', [])[:4])
                extras = []
                if details.get('exact_title_match'):
                    extras.append('exact title')
                if details.get('training_hits'):
                    extras.append('training')
                if details.get('certification_hits'):
                    extras.append('certification')
                suffix = f" ({', '.join(extras)})" if extras else ""
                st.write(f"🧭 Role Evidence: Yes{suffix}")
                if matched_titles:
                    st.caption(f"Matches: {matched_titles}")
            else:
                st.write("🧭 Role Evidence: No")
            
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
            # Benchmarks (if available)
            bench = (candidate.get('breakdown') or {}).get('benchmark') or {}
            gp = bench.get('percentile_global')
            rp = bench.get('percentile_role')
            if gp is not None or rp is not None:
                st.write("**Benchmarks:**")
                st.write(f"📈 Percentile (Role): {rp if rp is not None else '—'}% | Global: {gp if gp is not None else '—'}%")
            # Plugin effects (if available)
            plugins = (candidate.get('breakdown') or {}).get('plugins') or []
            if plugins:
                ids = [str(p.get('id')) for p in plugins if p.get('id')]
                if ids:
                    st.write(f"🔌 Plugins: {', '.join(ids[:4])}{'…' if len(ids) > 4 else ''}")
        
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

def create_analytics_dashboard(results: List[Dict[str, Any]]) -> None:
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
    
    gender_counts: Dict[str, int] = {}
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

def export_results_to_excel(results: List[Dict[str, Any]], position: str) -> Optional[BytesIO]:
    """Export results to Excel and provide download link."""
    if not results:
        return None
    
    # Create Excel data
    excel_data = []
    for i, candidate in enumerate(results, 1):
        bench = (candidate.get('breakdown') or {}).get('benchmark') or {}
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
            'File Name': candidate.get('file_name', 'Unknown'),
            'Percentile (Role)': bench.get('percentile_role'),
            'Percentile (Global)': bench.get('percentile_global'),
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

def export_results_to_csv(results: List[Dict[str, Any]], position: str) -> Optional[bytes]:
    """Export results to CSV bytes (same fields as Excel export)."""
    if not results:
        return None
    rows = []
    for i, candidate in enumerate(results, 1):
        bench = (candidate.get('breakdown') or {}).get('benchmark') or {}
        rows.append({
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
            'File Name': candidate.get('file_name', 'Unknown'),
            'Percentile (Role)': bench.get('percentile_role'),
            'Percentile (Global)': bench.get('percentile_global'),
        })
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return csv_bytes

def main() -> None:
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
    with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
        strict_role_match = st.checkbox(
            "Require explicit role/title/training evidence",
            value=True,
            help=(
                "Only show candidates whose resume explicitly mentions this role (or clear aliases) "
                "as a title, or shows training/certification tied to the role."
            ),
        )
        st.session_state["strict_role_match"] = strict_role_match
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

        # Include Unknown when filtering by Male/Female
        include_unknown_gender = False
        if gender_filter in ("Male", "Female"):
            include_unknown_gender = st.checkbox(
                "Include 'Unknown' gender in results",
                value=True,
                help="When enabled, candidates with unknown gender will not be excluded"
            )
        st.session_state["include_unknown_gender"] = include_unknown_gender
        
        # Sorting options
        sort_by = st.selectbox(
            "Sort Results By:",
            options=[
                "Score (desc)",
                "Score (asc)",
                "Recommendation",
                "Name (A-Z)",
                "Name (Z-A)"
            ],
            index=0,
            help="Choose how to sort the candidate list"
        )
        st.session_state["sort_by"] = sort_by

        # Search within results
        search_query = st.text_input(
            "Search Candidates:",
            value="",
            help="Filter results by name, file name, or skill keyword"
        )
        st.session_state["search_query"] = search_query

        # Pagination size
        page_size = st.number_input(
            "Results per page:",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="How many candidates to show per page"
        )
        st.session_state["page_size"] = page_size
        
        auto_export = st.checkbox(
            "Auto-export to Excel",
            value=True,
            help="Automatically generate Excel report"
        )

    # Reset filters
    if st.sidebar.button("↩️ Reset Filters"):
        # Clear related session state keys
        for key in ("include_unknown_gender", "sort_by", "search_query", "page_size", "page"):
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Auto-reset page to 1 if filters/sort/search/page size/position change
    try:
        _changed = False
        if st.session_state.get('prev_selected_position') != selected_position:
            _changed = True
        if st.session_state.get('prev_score_threshold') != score_threshold:
            _changed = True
        if st.session_state.get('prev_gender_filter') != gender_filter:
            _changed = True
        if st.session_state.get('prev_include_unknown_gender') != st.session_state.get('include_unknown_gender'):
            _changed = True
        if st.session_state.get('prev_sort_by') != st.session_state.get('sort_by'):
            _changed = True
        if st.session_state.get('prev_search_query') != (st.session_state.get('search_query') or ""):
            _changed = True
        if st.session_state.get('prev_page_size') != st.session_state.get('page_size'):
            _changed = True

        # Update prev_* snapshots
        st.session_state['prev_selected_position'] = selected_position
        st.session_state['prev_score_threshold'] = score_threshold
        st.session_state['prev_gender_filter'] = gender_filter
        st.session_state['prev_include_unknown_gender'] = st.session_state.get('include_unknown_gender')
        st.session_state['prev_sort_by'] = st.session_state.get('sort_by')
        st.session_state['prev_search_query'] = st.session_state.get('search_query') or ""
        st.session_state['prev_page_size'] = st.session_state.get('page_size')

        if _changed:
            st.session_state['page'] = 1
    except Exception:
        # Non-fatal: if any state not available yet, ignore
        pass
    
    # Source selection: Upload vs Folder
    st.subheader("📂 Source of Resume Files")
    source_mode = st.radio("Select source:", ["Upload", "Folder"], horizontal=True)

    uploaded_files = []
    folder_path = Path("input_resumes")
    if source_mode == "Upload":
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload resumes in PDF, Word, text, or image format"
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
            with st.expander("📋 Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
    else:
        st.caption("Using files directly from the input folder.")
        colf1, colf2, colf3 = st.columns([3,1,2])
        with colf1:
            folder_path = Path(st.text_input("Folder path", value=str(folder_path)))
        with colf2:
            if st.button("🔄 Refresh"):
                st.rerun()
        with colf3:
            auto_refresh = st.checkbox("Auto-refresh UI", value=False, help="Auto-refresh this page to pick up new files")
            interval = st.number_input("Every (sec)", min_value=5, max_value=120, value=15, step=5)
        only_new = st.checkbox("Only new since last run", value=False, help="Process only files not seen before (by modified time)")
        try:
            files_in_folder = [p for p in folder_path.glob("*.*") if p.suffix.lower() in (".pdf",".doc",".docx",".txt",".jpg",".jpeg",".png")]
            if only_new:
                cache_key = f"seen_mtimes::{str(folder_path)}"
                seen = st.session_state.get(cache_key, {})
                new_files = []
                for p in files_in_folder:
                    mt = p.stat().st_mtime
                    if seen.get(p.name) != mt:
                        new_files.append(p)
                        seen[p.name] = mt
                st.session_state[cache_key] = seen
                files_in_folder = new_files
            st.info(f"📂 {len(files_in_folder)} files in {folder_path}")
            st.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if auto_refresh and interval:
                # Inject meta refresh to trigger rerun on an interval
                st.markdown(f"<meta http-equiv='refresh' content='{int(interval)}'>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Folder not accessible: {e}")

    # Processing section
    col1, col2, col3 = st.columns(3)
    with col2:
        trigger = st.button("🚀 Start Screening")
    
    if trigger:
        if source_mode == "Upload":
            if not (uploaded_files and selected_position):
                st.warning("Please upload files and choose a position.")
            else:
                with st.spinner("🔍 Processing resumes... This may take a few minutes."):
                    try:
                        # Save files temporarily
                        temp_dir, saved_files = save_uploaded_files(uploaded_files)
                        # Initialize screener
                        if enhanced_available and EnhancedHotelAIScreener is not None:
                            screener = EnhancedHotelAIScreener(str(temp_dir))
                        elif HotelAIScreener is not None:
                            screener = HotelAIScreener(str(temp_dir))
                        else:
                            raise RuntimeError("No screener class available")
                        # Process candidates
                        results = screener.screen_candidates(
                            selected_position,
                            max_candidates,
                            require_explicit_role=st.session_state.get("strict_role_match", True),
                        )
                        # Enforce strict role evidence if enabled and available
                        if st.session_state.get("strict_role_match", True):
                            if results and all("explicit_role_evidence" in r for r in results):
                                before_ct = len(results)
                                results = [r for r in results if r.get("explicit_role_evidence", False)]
                                removed_ct = before_ct - len(results)
                                if removed_ct > 0:
                                    st.info(f"Filtered out {removed_ct} candidate(s) without explicit role/title/training evidence.")
                            else:
                                st.warning("Strict role match is enabled, but this engine didn't return role-evidence flags. Results are shown without strict filtering.")
                        # Filter by score threshold
                        results = [r for r in results if r['total_score'] >= score_threshold]
                        # Filter by gender if specified
                        if gender_filter != "All":
                            include_unknown = st.session_state.get("include_unknown_gender", False)
                            results = [
                                r for r in results
                                if r.get('gender', 'Unknown') == gender_filter or (include_unknown and r.get('gender', 'Unknown') == 'Unknown')
                            ]
                        st.session_state.screening_results = results
                        st.session_state['page'] = 1
                        # Clean up temporary files
                        import shutil
                        shutil.rmtree(temp_dir)
                        st.success(f"✅ Screening complete! Found {len(results)} qualified candidates.")
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.error("Please check your files and try again.")
        else:
            if not selected_position:
                st.warning("Please select a position.")
            else:
                with st.spinner("🔍 Scanning folder and processing resumes..."):
                    try:
                        if enhanced_available and EnhancedHotelAIScreener is not None:
                            screener = EnhancedHotelAIScreener(str(folder_path))
                        elif HotelAIScreener is not None:
                            screener = HotelAIScreener(str(folder_path))
                        else:
                            raise RuntimeError("No screener class available")
                        results = screener.screen_candidates(
                            selected_position,
                            max_candidates,
                            require_explicit_role=st.session_state.get("strict_role_match", True),
                        )
                        if st.session_state.get("strict_role_match", True):
                            if results and all("explicit_role_evidence" in r for r in results):
                                before_ct = len(results)
                                results = [r for r in results if r.get("explicit_role_evidence", False)]
                                removed_ct = before_ct - len(results)
                                if removed_ct > 0:
                                    st.info(f"Filtered out {removed_ct} candidate(s) without explicit role/title/training evidence.")
                            else:
                                st.warning("Strict role match is enabled, but this engine didn't return role-evidence flags.")
                        results = [r for r in results if r['total_score'] >= score_threshold]
                        if gender_filter != "All":
                            include_unknown = st.session_state.get("include_unknown_gender", False)
                            results = [
                                r for r in results
                                if r.get('gender', 'Unknown') == gender_filter or (include_unknown and r.get('gender', 'Unknown') == 'Unknown')
                            ]
                        st.session_state.screening_results = results
                        st.session_state['page'] = 1
                        st.success(f"✅ Screening complete! Found {len(results)} qualified candidates from folder.")
                    except Exception as e:
                        st.error(f"❌ Error during folder processing: {str(e)}")
    
    # Results section
    if st.session_state.screening_results:
        results = st.session_state.screening_results
        
        st.markdown("---")
        st.subheader(f"🏆 Screening Results for {selected_position}")
        
        # Analytics dashboard
        create_analytics_dashboard(results)
        
        st.markdown("---")
        st.subheader("👥 Candidate Rankings")

        # Apply search filter
        search_query = (st.session_state.get("search_query") or "").strip().lower()
        if search_query:
            def _match(r):
                name = str(r.get('candidate_name', '')).lower()
                fname = str(r.get('file_name', '')).lower()
                skills = ",".join(r.get('skills_found', [])).lower() if isinstance(r.get('skills_found'), list) else str(r.get('skills_found', ''))
                return (search_query in name) or (search_query in fname) or (search_query in skills)
            results = [r for r in results if _match(r)]

        # Apply sorting
        sort_by = st.session_state.get("sort_by", "Score (desc)")
        if sort_by == "Score (desc)":
            results = sorted(results, key=lambda r: r.get('total_score', 0), reverse=True)
        elif sort_by == "Score (asc)":
            results = sorted(results, key=lambda r: r.get('total_score', 0))
        elif sort_by == "Recommendation":
            results = sorted(results, key=lambda r: str(r.get('recommendation', '')))
        elif sort_by == "Name (A-Z)":
            results = sorted(results, key=lambda r: str(r.get('candidate_name', '')))
        elif sort_by == "Name (Z-A)":
            results = sorted(results, key=lambda r: str(r.get('candidate_name', '')), reverse=True)

        # Pagination
        page_size = st.session_state.get("page_size", 10)
        total = len(results)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = st.session_state.get("page", 1)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        page_results = results[start:end]

        # Display candidates for current page
        for i, candidate in enumerate(page_results, start + 1):
            create_candidate_card(candidate, i)

        # Pagination controls
        col_prev, col_info, col_next = st.columns([1,2,1])
        with col_prev:
            if st.button("⬅️ Previous", disabled=page <= 1):
                st.session_state["page"] = page - 1
                st.rerun()
        with col_info:
            st.markdown(f"Page {page} of {total_pages} • Showing {len(page_results)} of {total}")
        with col_next:
            if st.button("Next ➡️", disabled=page >= total_pages):
                st.session_state["page"] = page + 1
                st.rerun()
        
    # Export section
    if st.session_state.screening_results:
        results = st.session_state.screening_results
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
            # JSON Export
            safe_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="🧾 Download JSON",
                data=safe_json,
                file_name=f"screening_results_{selected_position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

            # CSV Export
            csv_bytes = export_results_to_csv(results, selected_position)
            st.download_button(
                label="🧾 Download CSV",
                data=csv_bytes or b"",
                file_name=f"screening_results_{selected_position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                disabled=not bool(csv_bytes)
            )

            # Clear Results
            if st.button("🔄 Clear Results"):
                st.session_state.screening_results = None
                st.rerun()
    
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
