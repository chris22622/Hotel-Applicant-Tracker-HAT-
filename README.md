# Hotel Applicant Tracker (HOT)
**AI-powered resume screener for hotels.** Drag-and-drop resumes, OCR for scanned PDFs, role-aware scoring from a YAML profile, instant shortlist export to CSV/Excel, and transparent scoring breakdowns.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Code Quality](https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-/workflows/Code%20Quality/badge.svg)](https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-/actions)

> **📸 Screenshot Coming Soon:** Upload your screenshots to `assets/hot-ui.png` to complete the README!

## 30-second Quickstart
```bash
python -m venv .venv && .venv\Scripts\activate  # Windows (.venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run streamlit_app.py
```

**Try the demo data:** Use sample resumes in `input_resumes/` → pick "Front Desk Agent" → click **Run Screening** → download Excel.

## Features

- **📄 OCR for scans:** Reads text from scanned PDFs and images with Tesseract
- **⚙️ Role presets:** Configure requirements via `hotel_config.yaml` (15+ hotel positions included)
- **� Skill/tenure scoring:** Weighted algorithms for experience, skills, and cultural fit
- **📊 CSV/Excel export:** Instant shortlists with contact info and scoring breakdowns
- **� Duplicate detection:** Identifies potential duplicate candidates
- **⚖️ Bias guardrails:** Ignores name/age/photo fields during scoring, focuses on qualifications
- **🖥️ Dual interface:** Web UI for HR teams, CLI for power users
- **🔒 Privacy-first:** 100% local processing, no data sent to external services

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

## Config

Role requirements are defined in `hotel_config.yaml`:

```yaml
positions:
  front_desk_agent:
    must_have_skills: [customer service, computer skills, communication]
    nice_to_have_skills: [hotel experience, multilingual, PMS systems]
    cultural_fit_keywords: [team player, positive attitude, professional]
    experience_weight: 0.3
    skills_weight: 0.4
    cultural_fit_weight: 0.3
```

## Demo Data

**Try with included samples:** The `input_resumes/` folder contains 15+ sample resumes for testing different hotel positions. Perfect for evaluating the system before adding your own candidate data.

**Preset roles available:** Front Desk Agent, Executive Chef, Housekeeping Manager, Security Officer, Sales Manager, and 10+ more hotel positions.

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

## 🗺️ Roadmap

- [ ] **Dockerfile** for containerized deployment
- [ ] **Multi-role batch scoring** for processing large candidate pools
- [ ] **Job description parser** to auto-generate role requirements
- [ ] **SQLite persistence** for candidate history and analytics
- [ ] **Simple authentication** for multi-user hotel chains
- [ ] **Streamlit Community Cloud** one-click deploy
- [ ] **REST API** for integration with existing HR systems
- [ ] **Bias detection reports** and fairness metrics

## 🚀 One-Click Deploy

Deploy to Streamlit Community Cloud in 60 seconds:
1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and `streamlit_app.py`
5. Click Deploy!

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