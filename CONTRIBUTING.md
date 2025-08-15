# Contributing to Hotel Applicant Tracker (HOT)

Thank you for your interest in contributing! This project helps hotels streamline their hiring process with AI-powered resume screening.

## 🚀 Quick Start for Contributors

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/Hotel-Applicant-Tracker-HOT-.git`
3. **Create** a virtual environment: `python -m venv .venv && .venv\Scripts\activate`
4. **Install** dependencies: `pip install -r requirements.txt`
5. **Download** spaCy model: `python -m spacy download en_core_web_sm`
6. **Run** tests: `pytest tests/`

## 📋 Good First Issues

Perfect for new contributors:

- **Add Dockerfile optimization** - Reduce image size and improve caching
- **Implement CSV export tests** - Test the fallback export functionality
- **Add more hotel position templates** - Expand `hotel_config.yaml` with new roles
- **Improve OCR accuracy** - Enhance text extraction for low-quality scans
- **Add multi-language support** - Extend beyond English resume processing
- **Create position requirement wizard** - UI for easy config generation

## 🎯 Areas We Need Help

### 🏨 Hotel Industry Expertise
- **Position templates**: More role configurations (spa, golf, events, etc.)
- **Scoring algorithms**: Industry-specific candidate evaluation criteria
- **Bias detection**: Ensure fair hiring practices

### 🔧 Technical Improvements
- **Performance**: Faster resume processing for large batches
- **Testing**: More comprehensive test coverage
- **Documentation**: Better setup guides and API docs
- **Integrations**: Connect with popular HR systems

### 🎨 User Experience
- **UI/UX**: Streamlit interface improvements
- **Accessibility**: Make the app usable for everyone
- **Mobile**: Responsive design for mobile HR teams

## 📐 Development Guidelines

### Code Style
We use **Ruff** for linting and formatting:
```bash
ruff check .          # Check for issues
ruff format .         # Format code
```

### Testing
- Write tests for new features in `tests/`
- Run the test suite: `pytest tests/`
- Use the smoke test: `python tests/test_repo_smoke.py`

### Commits
Use clear, descriptive commit messages:
```
✨ Add spa therapist position template
🐛 Fix CSV export encoding issue  
📚 Update installation docs for Windows
🧪 Add bias detection tests
```

## 🔍 Code Review Process

1. **Submit PR** with clear description and tests
2. **Automated checks** must pass (CI, linting, tests)
3. **Maintainer review** - we'll provide feedback
4. **Merge** after approval and passing checks

## 🏗️ Development Setup

### Local Development
```bash
# Setup development environment
git clone https://github.com/chris22622/Hotel-Applicant-Tracker-HOT-.git
cd Hotel-Applicant-Tracker-HOT-
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run in development mode
streamlit run streamlit_app.py

# Run tests
pytest tests/ -v
```

### Testing Your Changes
```bash
# Run smoke tests
python tests/test_repo_smoke.py

# Test CLI functionality  
python cli.py --input input_resumes/ --position front_desk_agent --output test.json

# Test with sample data
python demo.py
```

## 📝 Documentation

When adding features:
- Update `README.md` if user-facing
- Add docstrings to new functions
- Update `hotel_config.yaml` for new positions
- Add examples to help users understand

## 🐛 Bug Reports

**Before reporting:**
1. Check existing issues
2. Run `python tests/test_repo_smoke.py`
3. Include your Python version and OS

**Include in your report:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Error messages (full stack trace)
- Your environment details

## 💡 Feature Requests

We love new ideas! Before suggesting:
1. Check existing issues and discussions
2. Consider if it fits the hotel hiring use case
3. Think about maintenance burden

**Include in your request:**
- Clear problem description
- Proposed solution
- Use cases and benefits
- Implementation complexity estimate

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Community

- **Be respectful** - This is an inclusive, welcoming project
- **Be patient** - Maintainers are volunteers with day jobs
- **Be helpful** - Help other contributors and users

## 📞 Contact

- **Issues**: Use GitHub Issues for bugs and features
- **Questions**: Start a GitHub Discussion
- **Security**: Report security issues privately to maintainers

---

**Thank you for making hotel hiring better! 🏨✨**
