# Hotel AI Resume Screener - Enhanced Version 2.0

🚀 **Advanced AI-powered candidate screening with 1-click setup and intelligent matching**

[![Enhanced AI](https://img.shields.io/badge/AI-Enhanced-blue)](enhanced_ai_screener.py) [![1-Click Setup](https://img.shields.io/badge/Setup-1%20Click-green)](Quick_Setup.bat) [![Web Interface](https://img.shields.io/badge/Interface-Streamlit-red)](enhanced_streamlit_app.py)

## ✨ What's New in Version 2.0

### 🧠 **Enhanced AI Intelligence**
- **Semantic Matching**: Advanced NLP understanding of skills and experience
- **Position-Specific Intelligence**: Deep knowledge of 15+ hotel positions
- **Bias Detection**: Fair and transparent candidate evaluation
- **Experience Quality Assessment**: Not just years, but relevance and depth

### 🎯 **Improved Accuracy**
- **Smart Skill Extraction**: Contextual understanding of candidate abilities
- **Advanced Scoring Algorithm**: Multi-dimensional candidate evaluation
- **Cultural Fit Analysis**: Personality and work style matching
- **Technical Skills Recognition**: Industry-specific technology proficiency

### 🖥️ **Better User Experience**
- **1-Click Setup**: `Quick_Setup.bat` installs everything automatically
- **1-Click Launch**: `Quick_Start_Screener.bat` opens web interface instantly
- **Beautiful Web Dashboard**: Interactive charts and candidate analytics
- **Drag & Drop Interface**: Simple file upload with real-time processing

## 🚀 Quick Start (30 Seconds!)

### Option 1: Web Interface (Recommended)
```bash
# 1. Double-click Quick_Setup.bat (first time only)
# 2. Double-click Quick_Start_Screener.bat
# 3. Upload resumes in your browser
# 4. Get results instantly!
```

### Option 2: Command Line
```bash
# 1. Double-click Quick_Setup.bat (first time only)
# 2. Double-click Enhanced_CLI_Screener.bat
# 3. Select position and get results
```

## 📊 Enhanced Features

### 🎯 **Advanced Position Intelligence**
- **Front Office**: Front Desk Agent, Guest Services Manager, Concierge
- **Food & Beverage**: Executive Chef, Sous Chef, Bartender, Server
- **Operations**: Housekeeping Manager, Maintenance Supervisor
- **Each position has**: Detailed requirements, scoring weights, experience levels

### 🔍 **Smart Document Processing**
- **Multi-format Support**: PDF, Word, images, text files
- **OCR Technology**: Scanned documents and images
- **Semantic Extraction**: Contact info, skills, experience
- **Quality Assessment**: Document completeness and clarity

### 📈 **Comprehensive Analytics**
- **Score Distribution**: Visual candidate performance analysis
- **Category Breakdown**: Experience, skills, cultural fit, hospitality
- **Recommendation Engine**: Highly Recommended, Recommended, Consider, Not Recommended
- **Comparative Analysis**: Heatmaps and performance charts

### 📋 **Professional Reports**
- **Excel Export**: Multi-sheet reports with detailed analysis
- **Interactive Dashboard**: Real-time charts and candidate cards
- **Text Reports**: Detailed written summaries
- **Ranking System**: Candidates sorted by overall fit

## 🛠️ System Requirements

### Minimum Requirements
- Windows 10/11
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended for Best Performance
- Windows 11
- Python 3.10+
- 8GB RAM
- 5GB disk space
- SSD storage

## 📁 File Structure

```
Hotel-AI-Resume-Screener/
├── 🚀 Quick_Setup.bat              # 1-click installer
├── 🌐 Quick_Start_Screener.bat     # Launch web interface
├── 💻 Enhanced_CLI_Screener.bat    # Command-line launcher
├── 🧠 enhanced_ai_screener.py      # Advanced AI engine
├── 🎨 enhanced_streamlit_app.py    # Beautiful web interface
├── 📦 enhanced_requirements.txt    # Complete dependency list
├── 📂 input_resumes/               # Place resume files here
├── 📊 screening_results/           # Results and reports
└── 📝 enhanced_config.yaml        # Configuration settings
```

## 🎯 Supported Positions

### Front Office & Guest Services
- **Front Desk Agent**: Customer service, PMS systems, communication
- **Guest Services Manager**: Leadership, guest relations, team management
- **Concierge**: Local knowledge, luxury service, networking

### Food & Beverage
- **Executive Chef**: Culinary leadership, kitchen management, cost control
- **Sous Chef**: Cooking skills, team support, menu development
- **Bartender**: Mixology, customer service, product knowledge
- **Server**: Guest interaction, food knowledge, multitasking

### Operations & Maintenance
- **Housekeeping Manager**: Quality control, team leadership, efficiency
- **Maintenance Supervisor**: Technical skills, safety, problem-solving

## 🔧 Advanced Configuration

### Custom Position Setup
```yaml
# enhanced_config.yaml
positions:
  Custom_Manager:
    description: "Your custom position description"
    priority_skills: ["skill1", "skill2", "skill3"]
    min_experience: 2
    preferred_experience: 5
    scoring_weights:
      experience: 0.3
      skills: 0.3
      cultural_fit: 0.2
      hospitality: 0.2
```

### Scoring Customization
- **Experience Weight**: How much experience matters (0.1-0.5)
- **Skills Weight**: Importance of technical abilities (0.2-0.4)
- **Cultural Fit**: Personality and work style (0.1-0.3)
- **Hospitality**: Industry-specific knowledge (0.1-0.3)

## 📊 Sample Results

```
🏆 TOP CANDIDATES FOR FRONT DESK AGENT:

1. Sarah Johnson - 94.2% (Highly Recommended)
   📧 sarah.j@email.com | 📞 (555) 123-4567
   💪 Strengths: 5+ years hotel experience, Opera PMS expert, trilingual
   🎯 Skills: customer service, PMS systems, multilingual, guest relations
   
2. Mike Chen - 87.8% (Recommended)
   📧 mchen@email.com | 📞 (555) 987-6543
   💪 Strengths: Customer service excellence, team leadership experience
   🎯 Skills: communication, problem solving, hospitality, computer skills
```

## 🎨 Web Interface Features

### Dashboard Analytics
- **Real-time Processing**: Watch candidates get scored live
- **Interactive Charts**: Click and explore candidate data
- **Visual Score Breakdown**: Category performance heatmaps
- **Recommendation Distribution**: Pie charts and bar graphs

### Candidate Cards
- **Expandable Details**: Click to see full candidate analysis
- **Color-coded Ratings**: Visual recommendation indicators
- **Skill Highlighting**: Key abilities prominently displayed
- **Contact Information**: Easy-to-copy contact details

### Export Options
- **Excel Reports**: Multi-sheet workbooks with charts
- **PDF Summaries**: Professional candidate reports
- **CSV Data**: Raw data for further analysis
- **Text Reports**: Human-readable summaries

## 🚨 Troubleshooting

### Common Issues

**"Python not found"**
- Install Python 3.8+ from python.org
- Make sure "Add to PATH" is checked during installation

**"OCR not working"**
- Download Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
- Add to system PATH or use text-based resumes

**"Web interface won't start"**
- Run `Quick_Setup.bat` first
- Check if port 8501 is available
- Try restarting your computer

**"No candidates found"**
- Check resume files are in `input_resumes/` folder
- Verify file formats (PDF, Word, images, text)
- Try with different position selection

### Getting Help
1. Check the troubleshooting section above
2. Review the setup logs in the terminal
3. Ensure all dependencies are installed
4. Try with sample resume files first

## 🔮 Future Enhancements

- **Multi-language Support**: Spanish, French, German candidate processing
- **Video Resume Analysis**: AI-powered video interview screening
- **Integration APIs**: Connect with ATS and HRIS systems
- **Mobile App**: Screen candidates on tablets and phones
- **AI Interview Questions**: Generate position-specific interview questions

## 📄 License

MIT License - Feel free to use and modify for your hotel's needs!

---

**🏨 Built specifically for hospitality professionals who want to find the best candidates faster!**
