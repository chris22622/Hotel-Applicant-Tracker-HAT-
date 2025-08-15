# Quick Start Guide - Hotel AI Resume Screener

## 🚀 Get Started in 3 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Resumes
Place resume files in the `input_resumes/` folder:
- PDF files (including scanned documents)
- Microsoft Word (.docx, .doc) 
- Text files (.txt)

### 3. Run the Screener

**Option A: Command Line**
```bash
python hotel_ai_screener.py
```

**Option B: Web Interface**
```bash
streamlit run streamlit_app.py
```

## 🎯 Example Workflow

1. **Select Position**: Choose "Front Desk Agent" from the menu
2. **Set Requirements**: System automatically loads hotel-specific criteria
3. **AI Analysis**: Processes all resumes with intelligent scoring
4. **Get Results**: Ranked candidates with contact info and recommendations

## 📊 Sample Output

```
🏆 TOP CANDIDATES:
  1. Sarah Johnson - 92.5% (Highly Recommended)
     📧 sarah.j@email.com | 📞 (555) 123-4567
     💪 Strengths: 5+ years hotel experience, PMS systems, multilingual
  
  2. Mike Chen - 87.3% (Highly Recommended)  
     📧 mchen@email.com | 📞 (555) 987-6543
     💪 Strengths: Customer service excellence, team leadership
```

## 🔧 Customization

Edit `hotel_config.yaml` to customize requirements for your property:

```yaml
positions:
  front_desk_agent:
    must_have_skills:
      - customer service
      - computer skills
    nice_to_have_skills:
      - hotel experience
      - multilingual
```

## 📁 Results

The system creates organized output folders with:
- ✅ **Excel reports** with candidate contact sheets
- ✅ **Resume copies** of top candidates  
- ✅ **CSV exports** for easy importing
- ✅ **Detailed analysis** with scoring breakdown

Perfect for streamlining your hotel hiring process! 🏨✨
