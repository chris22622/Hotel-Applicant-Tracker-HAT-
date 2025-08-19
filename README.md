# Hotel AI Resume Screener

🚀 **NEW: Enhanced Version 2.0 Available!** → [See Enhanced Features](README_ENHANCED.md)

Smart, AI-powered resume screening for **hotels and resorts** with **OCR**, **role-aware scoring**, and **instant Excel reports**. Clean `hotel_ai_screener.py` core, CI-ready, and easy for hiring managers to evaluate.

[![CI](https://github.com/chris22622/Hotel-Applicant-Tracker-HAT-/workflows/CI/badge.svg)](https://github.com/chris22622/Hotel-Applicant-Tracker-HAT-/actions)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Enhanced Version](https://img.shields.io/badge/Version-2.0%20Available-brightgreen)](README_ENHANCED.md)

> _"Screen honestly. Rank fairly. Hire the best candidates faster."_

## 🎯 Quick Start Options

### 🚀 **Enhanced Version (Recommended)**
**1-Click Setup & Launch with Advanced AI**
```bash
# 1. Double-click Quick_Setup.bat
# 2. Double-click Quick_Start_Screener.bat
# 3. Upload resumes & get enhanced results!
```
[📖 Full Enhanced Documentation](README_ENHANCED.md)

### 📁 **Original Version**
**Basic but reliable core functionality**

![Screenshot](assets/hat-ui.png)
*(Add a screenshot to `assets/hat-ui.png` and it will render above.)*

---

## ✨ Features
- **Dual interface:** Web UI for HR teams, CLI for power users.
- **OCR-powered:** Reads scanned PDFs and image-based documents.
- **Role-aware:** 15+ hotel positions pre-configured (Front Desk, Chef, Manager, etc.).
- **Bias guardrails:** Ignores demographics, focuses on qualifications.
- **Smart scoring:** Experience + Skills + Cultural fit with transparent weights.
- **Instant exports:** Excel/CSV with contact info and ranking breakdowns.
- **Privacy-first:** 100% local processing, no external data sharing.

---

## 🚀 Quickstart (30 sec)

### 1) Setup
```bash
python -m venv .venv && .venv\Scripts\activate    # Windows (.venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Install Tesseract OCR: Windows (Install_Tesseract_OCR.bat) | Mac (brew install tesseract)
```

### 2) Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

### 3) Try the Demo Data
- Upload resumes from `input_resumes/` folder (15+ samples included)
- Select "Front Desk Agent" position
- Click **Run Screening**
- Download Excel report with ranked candidates

### 4) Command Line (Power Users)
```bash
python hotel_ai_screener.py
# or headless mode:
python cli.py --input input_resumes/ --position front_desk_agent --output results.json
```

---

## 🔧 Configuration (YAML)

`hotel_config.yaml` example:

```yaml
positions:
  front_desk_agent:
    must_have_skills: [customer service, computer skills, communication]
    nice_to_have_skills: [hotel experience, multilingual, PMS systems]
    cultural_fit_keywords: [team player, positive attitude, professional]
    weights:
      experience: 0.3
      skills: 0.4
      cultural_fit: 0.3
  executive_chef:
    must_have_skills: [culinary arts, food safety, team leadership]
    nice_to_have_skills: [fine dining, menu development, cost control]
    cultural_fit_keywords: [creative, detail oriented, passionate]
    weights:
      experience: 0.4
      skills: 0.4
      cultural_fit: 0.2
```

---

## 🔑 Environment (.env)

Optional configuration:

```dotenv
# OCR Configuration
TESSERACT_CMD=/usr/local/bin/tesseract  # Custom Tesseract path if needed

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Application Settings
MAX_FILE_SIZE_MB=10
SUPPORTED_FORMATS=pdf,docx,txt
DEFAULT_POSITION=front_desk_agent
```

---

## 📊 Sample Output

**Console Results:**
```
🏆 TOP CANDIDATES FOR FRONT DESK AGENT:
  1. Sarah Johnson - 92.5% (Highly Recommended)
     📧 sarah.j@email.com | 📞 (555) 123-4567
     💪 Strengths: 5+ years hotel experience, PMS systems, trilingual
  
  2. Mike Chen - 87.3% (Recommended)
     📧 mchen@email.com | 📞 (555) 987-6543
     💪 Strengths: Customer service excellence, team leadership
```

**Excel Export Includes:**
- 📊 Ranked candidate summary with scores
- 📧 Contact information extraction
- 💼 Skills analysis and recommendations
- 📁 Organized resume copies for top candidates

---

## 🧱 Project Structure

```
Hotel-Applicant-Tracker-HAT-/
├─ hotel_ai_screener.py   # Core AI screening engine
├─ streamlit_app.py       # Web interface
├─ cli.py                 # Headless command-line tool
├─ hotel_config.yaml      # Position requirements & scoring
├─ input_resumes/         # Sample resumes for testing
├─ screening_results/     # Output reports and organized files
├─ tests/                 # pytest suite with smoke tests
├─ assets/                # Screenshots and visual assets
├─ .streamlit/            # Streamlit theming and config
├─ requirements.txt       # Python dependencies
└─ README.md
```

---

## � Docker (optional)

```bash
docker build -t hotel-ats .
docker run -p 8501:8501 hotel-ats
# Access at http://localhost:8501
```

---

## 🚀 Deploy Options

### Streamlit Community Cloud (Free)
1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub and deploy `streamlit_app.py`

### Local Installation
```bash
git clone https://github.com/chris22622/Hotel-Applicant-Tracker-HAT-.git
cd Hotel-Applicant-Tracker-HAT-
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run streamlit_app.py
```

---

## 📈 CI & Quality

* CI runs lint + tests on push/PR (badge above).
* Smoke tests validate repo structure and imports.
* Bias detection tests ensure fair candidate evaluation.

---

## 🤝 Contributing

We welcome contributions! Perfect for:
- Adding new hotel position templates
- Improving OCR accuracy for scanned documents
- Expanding bias detection and fairness metrics
- UI/UX enhancements for the Streamlit interface

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and good first issues.

---

## ⚠️ Fair Hiring Notice

This tool is designed to **assist**, not replace, human judgment in hiring. Always review AI recommendations and ensure compliance with local employment laws and fair hiring practices.

---

## 📝 License

MIT © **Chrissano Leslie**  
See [LICENSE](LICENSE).

---

**Perfect for hotels looking to streamline hiring and find the best candidates faster! 🏨✨**

## 📊 Sample Output

```
🏆 TOP CANDIDATES FOR FRONT DESK AGENT:
  1. Sarah Johnson - 92.5% (Highly Recommended)
     📧 sarah.j@email.com | 📞 (555) 123-4567
     💪 Strengths: 5+ years hotel experience, PMS systems, trilingual
  
  2. Mike Chen - 87.3% (Recommended)
     📧 mchen@email.com | 📞 (555) 987-6543
     💪 Strengths: Customer service excellence, team leadership
```

## 🛠️ Installation

### Requirements
- **Python 3.10+**
- **Tesseract OCR** (for scanned documents)
- **spaCy model**: `python -m spacy download en_core_web_sm`

### Setup
```bash
# Clone and setup
git clone https://github.com/chris22622/Hotel-Applicant-Tracker-HAT-.git
cd Hotel-Applicant-Tracker-HAT-
python -m venv .venv && .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment (optional)
cp .env.example .env

# Install OCR (Windows)
Install_Tesseract_OCR.bat

# Launch
streamlit run streamlit_app.py
```

### Headless Mode
```bash
python cli.py --input input_resumes/ --position front_desk_agent --output results.json
```

## 🚀 Deploy Options

### Streamlit Community Cloud (Free)
1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub and deploy `streamlit_app.py`

### Docker
```bash
docker build -t hotel-ats .
docker run -p 8501:8501 hotel-ats
```

## 🗺️ Roadmap

- [ ] **Multi-role batch processing** for large candidate pools
- [ ] **Job description parser** to auto-generate requirements
- [ ] **SQLite persistence** for candidate history
- [ ] **REST API** for HR system integration
- [ ] **Advanced bias detection** and fairness metrics
- [ ] **Multi-language support** beyond English

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Good first issues:**
- Add Dockerfile optimization
- Implement CSV export tests  
- Add more hotel position templates
- Improve OCR accuracy for low-quality scans

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built for the hospitality industry with ❤️
- Powered by spaCy NLP and Streamlit
- Designed for privacy and local operation
- Optimized for hotel hiring workflows

---

**Perfect for hotels looking to streamline hiring and find the best candidates faster! 🏨✨**
