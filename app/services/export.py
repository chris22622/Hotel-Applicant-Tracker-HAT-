"""Export service for generating Excel files."""
import io
from typing import List, Dict, Any
import pandas as pd


def export_shortlist_to_excel(applications: List[Dict[str, Any]]) -> bytes:
    """Export application shortlist to Excel format."""
    # Prepare data for DataFrame
    data = []
    
    for app in applications:
        candidate = app.get('candidate', {})
        role = app.get('role', {})
        
        row = {
            'Rank': app.get('rank', ''),
            'Name': candidate.get('full_name', ''),
            'Email': candidate.get('email', ''),
            'Phone': candidate.get('phone', ''),
            'Location': candidate.get('location', ''),
            'Current Title': candidate.get('current_title', ''),
            'Current Company': candidate.get('current_company', ''),
            'Years Experience': candidate.get('years_total', ''),
            'Role': role.get('title', ''),
            'Score': app.get('score_numeric', ''),
            'Stage': app.get('stage', ''),
            'Explanation': app.get('explanation', ''),
            'Applied Date': app.get('created_at', ''),
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format columns
    if 'Score' in df.columns:
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').round(3)
    
    if 'Years Experience' in df.columns:
        df['Years Experience'] = pd.to_numeric(df['Years Experience'], errors='coerce').round(1)
    
    # Create Excel file in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Shortlist', index=False)
        
        # Get worksheet for formatting
        worksheet = writer.sheets['Shortlist']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 chars
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    buffer.seek(0)
    return buffer.getvalue()


def export_candidates_to_excel(candidates: List[Dict[str, Any]]) -> bytes:
    """Export candidates list to Excel format."""
    data = []
    
    for candidate in candidates:
        row = {
            'ID': candidate.get('id', ''),
            'Name': candidate.get('full_name', ''),
            'Email': candidate.get('email', ''),
            'Phone': candidate.get('phone', ''),
            'Location': candidate.get('location', ''),
            'Current Title': candidate.get('current_title', ''),
            'Current Company': candidate.get('current_company', ''),
            'Years Experience': candidate.get('years_total', ''),
            'Status': candidate.get('status', ''),
            'Source': candidate.get('source', ''),
            'Created Date': candidate.get('created_at', ''),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Format numeric columns
    if 'Years Experience' in df.columns:
        df['Years Experience'] = pd.to_numeric(df['Years Experience'], errors='coerce').round(1)
    
    # Create Excel file
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Candidates', index=False)
        
        worksheet = writer.sheets['Candidates']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    buffer.seek(0)
    return buffer.getvalue()
