# Hotel Applicant Tracker (HOT)
AI-powered hotel resume screener with drag-and-drop UI, OCR for scanned PDFs, role-aware scoring, and instant Excel/CSV reports.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Code Quality](https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-/workflows/Code%20Quality/badge.svg)](https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-/actions)

> **📸 Screenshot Coming Soon:** Upload your screenshots to `assets/hot-ui.png` to complete the README!

## 60-second run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Install Tesseract OCR (Windows: Install_Tesseract_OCR.bat | Mac: brew install tesseract)
streamlit run streamlit_app.py
```

**Try it:** drag PDFs/DOCX/TXT → pick a role → click **Run Screening** → download the Excel.

## How to Evaluate (30 seconds)
1. **🌐 Launch UI**: `streamlit run streamlit_app.py` 
2. **📁 Drop samples**: Use resumes from `input_resumes/` folder
3. **📊 Download Excel**: Get ranked candidates with contact info

## Quick Evaluation Script
```bash
python evaluate.py  # Interactive 30-second demo
```

---

## 🌟 Features

- **🤖 AI-Powered Analysis**: Advanced algorithms evaluate candidates based on skills, experience, and cultural fit
- **🏨 Hotel-Specific Intelligence**: Pre-configured requirements for 15+ common hotel positions  
- **📄 Multi-Format Support**: Handles PDF, DOCX, TXT, and even scanned documents with OCR
- **📊 Detailed Reporting**: Excel exports with contact sheets and comprehensive candidate analysis
- **🖥️ Dual Interface**: Both command-line and web-based (Streamlit) interfaces
- **🔒 Privacy First**: 100% local processing - no data sent to external services
- **⚡ Fast & Efficient**: Process dozens of resumes in minutes

## 🚀 Quick Start

### Web Interface (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Install Tesseract OCR
# Windows: Run Install_Tesseract_OCR.bat
# Mac: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr

# 3. Launch web interface
streamlit run streamlit_app.py
```

### Command Line Interface
```bash
# Install dependencies (same as above)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the screener
python hotel_ai_screener.py
```

### Demo Mode
```bash
# See capabilities without setup
python demo.py
# or
Run_Demo.bat
```

## 📋 Supported Hotel Positions

**Front of House:**
- Front Desk Agent, Guest Services Agent, Concierge, Night Auditor

**Food & Beverage:**
- Executive Chef, Sous Chef, Line Cook, Bartender, Server, Restaurant Manager

**Operations:**
- Hotel Manager, Assistant Manager, Housekeeping Manager, Housekeeper, Maintenance

**Specialized:**
- Security Officer, Sales Manager, Event Coordinator

## 📊 Sample Output

**Console Results:**
```
🏆 TOP CANDIDATES FOR FRONT DESK AGENT:
  1. Sarah Johnson - 94.2% (Highly Recommended)
     📧 sarah.j@email.com | 📞 (555) 123-4567
     💪 Strengths: 5+ years hotel experience, PMS systems, multilingual
  
  2. Mike Chen - 87.3% (Recommended)
     📧 mchen@email.com | 📞 (555) 987-6543  
     💪 Strengths: Customer service excellence, team leadership
```

**Excel Export Includes:**
- 📊 Ranked candidate summary with scores
- 📧 Contact information extraction
- 💼 Skills analysis and recommendations
- 📁 Organized resume copies for top candidates

## 🛠️ Customization

Edit `hotel_config.yaml` to customize for your property:

```yaml
positions:
  front_desk_agent:
    must_have_skills:
      - customer service
      - computer skills
    nice_to_have_skills:
      - hotel experience
      - multilingual
    cultural_fit_keywords:
      - team player
      - positive attitude
```

## 📦 Installation Details

### Requirements
- **Python 3.10+**
- **Tesseract OCR** (for scanned documents)
- **spaCy English model** (`en_core_web_sm`)

### Dependencies
- `streamlit` - Web interface
- `pandas` - Data processing  
- `openpyxl` - Excel export
- `python-docx` - Word document parsing
- `PyPDF2` - PDF parsing
- `pytesseract` - OCR functionality
- `spacy` - Natural language processing

## 🔧 Troubleshooting

**Tesseract OCR Issues:**
- Windows: Run `Install_Tesseract_OCR.bat` as administrator
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

**spaCy Model Missing:**
```bash
python -m spacy download en_core_web_sm
```

**Excel Export Errors:**
The system automatically falls back to CSV if Excel libraries are unavailable.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the hospitality industry
- Powered by spaCy and modern NLP techniques
- Designed for privacy and local operation
- Optimized for hotel hiring workflows

---

**Perfect for hotels looking to streamline hiring and find the best candidates faster! 🏨✨**