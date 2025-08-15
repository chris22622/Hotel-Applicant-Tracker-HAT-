# Assets Folder

This folder contains visual assets for the Hotel Applicant Tracker (HOT) project.

## Required Screenshots/Assets

To complete the README, please add:

1. **hot-ui.png** - Screenshot of the Streamlit web interface showing:
   - File upload area with sample resumes
   - Position selection dropdown
   - "Run Screening" button
   - Recommended width: 900px

2. **excel-output.png** - Screenshot of the Excel output showing:
   - Candidate ranking table
   - Contact information
   - Scoring breakdown
   - Recommended width: 700px

## How to Create Screenshots

### For Streamlit UI (hot-ui.png):
1. Run: `streamlit run streamlit_app.py`
2. Upload 2-3 sample resumes from `input_resumes/`
3. Select a position (e.g., "Front Desk Agent")
4. Take screenshot before clicking "Run Screening"
5. Save as `assets/hot-ui.png`

### For Excel Output (excel-output.png):
1. Run screening on sample resumes
2. Open the generated Excel file from `screening_results/`
3. Take screenshot of the candidate summary sheet
4. Save as `assets/excel-output.png`

## Usage in README

These images will be referenced in the README.md as:
```markdown
<img src="assets/hot-ui.png" alt="HOT UI" width="900"/>
<img src="assets/excel-output.png" alt="Excel Output" width="700"/>
```
