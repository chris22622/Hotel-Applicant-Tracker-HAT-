# Royalton AI Screener - Enhanced Features Guide

## 🚀 New Enhanced Features

### 1. 📱 **OCR Support for Scanned Documents**
- **Automatic fallback to OCR** when PDF text extraction fails
- **Supports scanned resumes** and image-based documents
- **Uses pytesseract + pdf2image** for high-quality text extraction
- **Configurable DPI settings** for optimal OCR accuracy

```yaml
# In royalton_config.yaml
ocr:
  enabled: true
  tesseract_path: null  # Auto-detect
  languages: "eng"
  dpi: 300
```

### 2. 🧠 **Semantic Skill Matching with spaCy**
- **Intelligent synonym detection** beyond keyword matching
- **Context-aware skill recognition** using NLP models
- **Similarity scoring** for related skills and experience
- **Fallback to keyword matching** if spaCy unavailable

### 3. ⚙️ **Configuration File System**
- **YAML-based job requirements** - edit without touching code
- **Customizable scoring weights** per position
- **Skill synonym definitions** for better matching
- **Easy position management** for HR staff

```yaml
# Example: Edit royalton_config.yaml
positions:
  chef:
    must_have_skills:
      - cooking
      - kitchen management
      - food safety
    experience_weight: 0.4
    skills_weight: 0.3
```

### 4. 📊 **Enhanced Export Options**
- **CSV exports** alongside Excel reports
- **Organized folder structure** with separate CSV directory
- **Configurable output formats** via config file
- **Batch processing ready** for large resume volumes

### 5. 🌐 **Streamlit Web Interface**
- **User-friendly GUI** for non-technical HR staff
- **Drag-and-drop file uploads** (PDF, DOCX, TXT)
- **Real-time progress tracking** during analysis
- **Interactive results display** with download options
- **No command line required** - pure web interface

## 📋 Installation Guide

### Quick Setup (Enhanced Features)
```bash
# Install enhanced dependencies
pip install -r enhanced_requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Optional: Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Or Use Batch Files
```bash
# Windows - automatic installation
Install_Enhanced_Features.bat

# Start web interface
Royalton_AI_Web.bat

# Traditional command line
Royalton_AI_Screener.bat
```

## 🎯 Usage Options

### Option 1: Web Interface (Recommended for HR Staff)
1. Run `Royalton_AI_Web.bat`
2. Open browser to http://localhost:8501
3. Select position and upload resumes
4. View results and download reports

### Option 2: Command Line (Power Users)
1. Run `Royalton_AI_Screener.bat`
2. Follow prompts as before
3. Enhanced features work automatically

### Option 3: Configuration Management
1. Edit `royalton_config.yaml` for custom job requirements
2. Add new positions without code changes
3. Adjust scoring weights per role
4. Define skill synonyms for better matching

## 🔧 Configuration Examples

### Adding New Position
```yaml
positions:
  spa_therapist:
    must_have_skills:
      - massage therapy
      - customer service
      - wellness knowledge
    nice_to_have_skills:
      - certification
      - aromatherapy
      - reflexology
    cultural_fit_keywords:
      - calming
      - healing
      - wellness focused
    experience_weight: 0.35
    skills_weight: 0.30
    cultural_fit_weight: 0.20
    hospitality_weight: 0.15
```

### Custom Skill Synonyms
```yaml
skill_synonyms:
  massage_therapy:
    - therapeutic massage
    - spa massage
    - deep tissue
    - Swedish massage
    - bodywork
```

## 📈 Benefits Over Basic Version

| Feature | Basic Version | Enhanced Version |
|---------|---------------|------------------|
| **Scanned PDFs** | ❌ Cannot read | ✅ OCR extraction |
| **Skill Matching** | ❌ Keywords only | ✅ Semantic + synonyms |
| **Job Requirements** | ❌ Hard-coded | ✅ YAML configurable |
| **Export Formats** | ❌ Excel only | ✅ Excel + CSV |
| **User Interface** | ❌ Command line | ✅ Web + command line |
| **Batch Processing** | ❌ Limited | ✅ Optimized for scale |

## 🧠 Smart AI Enhancements Still Included

All the intelligent matching features from the base version:
- ✅ **Role-specific experience analysis**
- ✅ **Smart skill detection with synonyms**
- ✅ **Context-aware scoring algorithms**
- ✅ **Hospitality industry intelligence**
- ✅ **Cultural fit assessment**
- ✅ **Duplicate candidate detection**

## 🔍 Technical Architecture

```
royalton_ai_screener.py     # Core AI engine (enhanced)
├── OCR Integration         # pytesseract + pdf2image
├── spaCy NLP              # Semantic matching
├── YAML Config            # Job requirements
└── CSV Export             # Additional formats

streamlit_gui.py           # Web interface
├── File Upload            # Drag-and-drop
├── Progress Tracking      # Real-time updates
└── Interactive Results    # Visual dashboard

royalton_config.yaml       # Configuration
├── Position Definitions   # Job requirements
├── Skill Synonyms        # Better matching
└── Output Settings       # Export options
```

## 🎉 Ready for Enterprise Use

The enhanced AI screener is now ready for:
- **Large-scale hiring** with batch processing
- **Non-technical users** via web interface
- **Custom job roles** via configuration
- **Scanned document processing** with OCR
- **Advanced skill matching** with NLP

Perfect for Royalton Resort's professional HR operations! 🏨✨
