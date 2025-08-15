# Hotel AI Resume Screener

**Intelligent candidate selection for hotels and resorts**

A powerful, AI-driven resume screening system designed specifically for the hospitality industry. This tool automatically analyzes resumes, scores candidates based on role-specific requirements, and provides detailed recommendations to streamline your hiring process.

## 🌟 Features

- **🤖 AI-Powered Analysis**: Advanced algorithms evaluate candidates based on skills, experience, and cultural fit
- **🏨 Hotel-Specific Intelligence**: Pre-configured requirements for 15+ common hotel positions
- **📄 Multi-Format Support**: Handles PDF, DOCX, TXT, and even scanned documents with OCR
- **📊 Detailed Reporting**: Excel exports with contact sheets and comprehensive candidate analysis
- **🖥️ Dual Interface**: Both command-line and web-based (Streamlit) interfaces
- **🔒 Privacy First**: 100% local processing - no data sent to external services
- **⚡ Fast & Efficient**: Process dozens of resumes in minutes

## 🚀 Quick Start

### Option 1: Command Line Interface
```bash
# Install dependencies
pip install -r requirements.txt

# Run the screener
python hotel_ai_screener.py
```

### Option 2: Web Interface
```bash
# Install dependencies
pip install -r requirements.txt

# Launch web interface
streamlit run streamlit_app.py
```

## 📋 Supported Positions

**Front of House:**
- Front Desk Agent
- Guest Services Agent
- Concierge

**Food & Beverage:**
- Executive Chef
- Sous Chef
- Line Cook
- Bartender
- Server

**Operations:**
- Housekeeping Supervisor
- Room Attendant
- Security Officer
- Maintenance Technician

**Sales & Events:**
- Sales Manager
- Event Coordinator

*Plus support for custom positions*

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- For OCR support: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

### Step-by-Step Setup

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/yourusername/hotel-ai-screener.git
   cd hotel-ai-screener
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install Tesseract for OCR**
   - Windows: Download from [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

4. **Test the installation**
   ```bash
   python hotel_ai_screener.py
   ```

## 📖 How It Works

### 1. Upload Resumes
Place resume files (PDF, DOCX, TXT) in the `input_resumes/` folder or upload via web interface.

### 2. Select Position
Choose from pre-configured hotel positions or enter a custom role.

### 3. AI Analysis
The system analyzes each resume for:
- **Skills Match**: Required and preferred skills for the position
- **Experience Relevance**: Years of experience and industry background
- **Cultural Fit**: Personality traits and work style indicators
- **Hospitality Background**: Hotel, resort, and service industry experience

### 4. Smart Scoring
Each candidate receives an AI score (0-100%) based on:
- **Experience Weight** (30%): Years and relevance of experience
- **Skills Weight** (25%): Technical and soft skills match
- **Cultural Fit Weight** (25%): Personality and work style alignment
- **Hospitality Weight** (20%): Industry-specific experience

### 5. Results & Export
Get ranked candidates with:
- ✅ **Contact Information**: Name, phone, email, location
- ✅ **AI Score & Recommendation**: Detailed scoring breakdown
- ✅ **Strengths & Weaknesses**: Specific feedback for each candidate
- ✅ **Excel Reports**: HR-ready contact sheets and detailed analysis

## 📊 Sample Output

```
🏆 TOP CANDIDATES:
  1. Sarah Johnson - 92.5% (Highly Recommended - Excellent match for the position)
  2. Mike Chen - 87.3% (Highly Recommended - Excellent match for the position)  
  3. Lisa Rodriguez - 78.9% (Recommended - Good candidate with minor gaps)
  4. Tom Williams - 71.2% (Recommended - Good candidate with minor gaps)
  5. Amy Taylor - 68.4% (Recommended - Good candidate with minor gaps)
```

## ⚙️ Configuration

Customize the screening criteria by editing `hotel_config.yaml`:

```yaml
positions:
  front_desk_agent:
    must_have_skills:
      - customer service
      - computer skills
      - communication
    nice_to_have_skills:
      - hotel experience
      - PMS systems
    cultural_fit_keywords:
      - team player
      - friendly
      - professional
    experience_weight: 0.3
    skills_weight: 0.25
    cultural_fit_weight: 0.25
    hospitality_weight: 0.2
```

## 🛠️ Advanced Features

### OCR Support
Automatically processes scanned PDFs and image-based resumes using Tesseract OCR.

### Skill Synonyms
Smart matching recognizes equivalent terms:
- "customer service" = "guest service" = "client service"
- "hotel experience" = "hospitality" = "resort"

### Export Options
- **Excel Reports**: Detailed analysis with multiple sheets
- **CSV Exports**: Contact lists and qualified candidates
- **Resume Copies**: Organized folders with top candidates' resumes

## 🔒 Privacy & Security

- **100% Local Processing**: No data sent to external services
- **No Internet Required**: Works completely offline after installation  
- **Your Data Stays Private**: All resume data remains on your computer
- **No Accounts or Logins**: Simple file-based operation

## 📱 System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for dependencies
- **Optional**: Tesseract OCR for scanned document support

## 🆘 Troubleshooting

### Common Issues

**"No module named 'streamlit'"**
```bash
pip install streamlit
```

**"Tesseract not found"**
- Install Tesseract OCR from the official repository
- Ensure it's added to your system PATH

**"No text extracted from PDF"**
- Try installing additional dependencies: `pip install PyPDF2 pdfminer.six`
- For scanned PDFs, install Tesseract OCR

**Import errors**
```bash
pip install --upgrade -r requirements.txt
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add New Positions**: Expand the job intelligence database
2. **Improve Scoring**: Enhance the AI scoring algorithms  
3. **Add Features**: New export formats, integrations, or UI improvements
4. **Fix Bugs**: Report issues and submit fixes
5. **Documentation**: Improve setup guides and usage examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Created by **Chris** - A comprehensive solution for modern hotel recruitment.

## 🙏 Acknowledgments

- Built with Python, Pandas, and Streamlit
- OCR support powered by Tesseract
- PDF processing via pdfminer and PyPDF2
- Document parsing with python-docx

---

**Transform your hiring process with AI-powered candidate screening for the hospitality industry!** 🏨✨
