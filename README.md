# Hotel Applicant Tracker (HOT)

**AI-powered resume screener for hotels.** Drag-and-drop resumes, OCR for scanned PDFs, role-aware scoring from YAML config, instant Excel/CSV export, and transparent scoring breakdowns.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF6B6B.svg)](https://streamlit.io)
[![Code Quality](https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-/workflows/CI/badge.svg)](https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-/actions)

## 🚀 30-Second Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate  # Windows (.venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run streamlit_app.py
```

**Try the demo:** Use sample resumes in `input_resumes/` → select "Front Desk Agent" → click **Run Screening** → download Excel.

## ✨ Features

- **📄 OCR for scans** → Reads text from scanned PDFs and images
- **⚙️ Role presets** → Configure via `hotel_config.yaml` (15+ positions included)
- **🧮 Smart scoring** → Weighted algorithms for experience, skills, cultural fit
- **📊 Excel/CSV export** → Instant shortlists with contact info and breakdowns
- **🔍 Duplicate detection** → Identifies potential duplicate candidates
- **⚖️ Bias guardrails** → Ignores demographic fields, focuses on qualifications
- **🖥️ Dual interface** → Web UI for HR teams, CLI for power users
- **🔒 Privacy-first** → 100% local processing, no external data transfer

## 🎯 How to Evaluate (60 seconds)

1. **🌐 Launch UI**: `streamlit run streamlit_app.py`
2. **📁 Upload resumes**: Drag PDF/DOCX/TXT files or use samples from `input_resumes/`
3. **🎯 Select position**: Choose from 15+ hotel roles (Front Desk, Chef, Manager, etc.)
4. **▶️ Run screening**: Click "Run Screening" and wait for AI analysis
5. **📊 Download results**: Get Excel with ranked candidates and contact info

## 📋 Config Example

Role requirements in `hotel_config.yaml`:

```yaml
positions:
  front_desk_agent:
    must_have_skills: [customer service, computer skills, communication]
    nice_to_have_skills: [hotel experience, multilingual, PMS systems]
    cultural_fit_keywords: [team player, positive attitude, professional]
    weights: { experience: 0.3, skills: 0.4, cultural_fit: 0.3 }
```

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
git clone https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-.git
cd Hotel-Applicant-Tracker-HOT-
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
