# 🛠️ Hotel AI Resume Screener - Troubleshooting Guide

## 🚨 Common Issues & Solutions

### 🌐 Port Already in Use Error

**Problem:** `Port 8501 is already in use` or `Port 8502 is already in use`

**Solution:**
1. **Quick Fix:** Double-click `Stop_Screener.bat` to stop all running instances
2. **Manual Fix:** Open Task Manager → Find `streamlit.exe` or `python.exe` → End Task
3. **Command Line:** Run `taskkill /f /im streamlit.exe` in PowerShell

### 📦 Missing Dependencies Error

**Problem:** `ModuleNotFoundError: No module named 'streamlit'` or similar

**Solution:**
1. **Automatic Fix:** Double-click `Fix_Missing_Packages.bat`
2. **Full Reinstall:** Double-click `Quick_Setup.bat`
3. **Manual Fix:** Run these commands in PowerShell:
   ```powershell
   cd "C:\Users\Chris\HR Applicant Tracker (HAT)"
   .venv\Scripts\activate
   pip install -r enhanced_requirements.txt
   ```

### 🔍 SpaCy/OCR Warnings

**Problem:** `⚠️ SpaCy not available - using basic text processing`

**Solution:**
- This is **normal** and the system works fine with basic processing
- For advanced features, run: `pip install spacy` (optional)
- The AI screening still works excellently without SpaCy

### 📁 No Resumes Found

**Problem:** "No resumes found in input folder"

**Solution:**
1. Put resume files in `input_resumes/` folder
2. Supported formats: PDF, DOCX, TXT
3. Make sure files have proper extensions (.pdf, .docx, .txt)

### 🌐 Browser Doesn't Open Automatically

**Problem:** Web interface doesn't open in browser

**Solution:**
1. **Manual Open:** Go to `http://localhost:8502` in your browser
2. **Check Port:** Look at the console output for the correct port number
3. **Firewall:** Make sure Windows Firewall isn't blocking the application

### 💻 Virtual Environment Issues

**Problem:** "venv is not recognized" or activation fails

**Solution:**
1. **Recreate Environment:** Delete `.venv` folder and run `Quick_Setup.bat`
2. **Python Path:** Make sure Python is installed and in PATH
3. **Permissions:** Run as Administrator if needed

### 📊 Charts Not Displaying

**Problem:** Empty charts or visualization errors, or `SyntaxError: [sprintf] unexpected placeholder`

**Solution:**
1. **JavaScript Error Fix:** The sprintf error has been resolved in the latest version
2. **Dependencies:** Run `Fix_Missing_Packages.bat`
3. **Browser Compatibility:** Try Chrome, Firefox, or Edge
4. **JavaScript:** Make sure JavaScript is enabled in browser
5. **Refresh:** Hard refresh the browser (Ctrl+F5) to clear any cached JavaScript

### 🔧 JavaScript sprintf Error

**Problem:** `SyntaxError: [sprintf] unexpected placeholder` in browser console

**Solution:**
- This issue has been **fixed** in the enhanced version
- The error was caused by percentage formatting in plotly charts
- **Restart the application** to get the fix: Run `Stop_Screener.bat` then `Quick_Start_Screener.bat`

### 🐌 Slow Performance

**Problem:** Application runs slowly

**Solution:**
1. **File Count:** Process fewer resumes at once (under 50 recommended)
2. **File Size:** Compress large PDF files
3. **RAM:** Close other applications to free memory

## 🔧 Advanced Troubleshooting

### Complete Reset
If nothing works, do a complete reset:
1. Delete `.venv` folder
2. Run `Quick_Setup.bat`
3. Test with `Quick_Start_Screener.bat`

### Check System Requirements
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 2GB free space
- **Internet:** Required for initial setup only

### Log Files
Check these files for detailed error information:
- Console output in the PowerShell window
- Streamlit logs in the application

## 📞 Quick Fixes Checklist

Before asking for help, try these steps:

1. ✅ **Stop all instances:** Run `Stop_Screener.bat`
2. ✅ **Fix dependencies:** Run `Fix_Missing_Packages.bat`
3. ✅ **Check resumes:** Ensure files are in `input_resumes/` folder
4. ✅ **Try different port:** Use `http://localhost:8502` instead of 8501
5. ✅ **Restart completely:** Close all windows and run `Quick_Start_Screener.bat`

## 🎯 Performance Tips

### For Best Results:
- Use **PDF or DOCX** files for best text extraction
- Keep resume files **under 5MB** each
- Process **20-50 resumes** at a time for optimal speed
- Use **descriptive filenames** for easier identification

### Browser Tips:
- Use **Chrome or Firefox** for best compatibility
- **Refresh the page** if charts don't load
- **Clear browser cache** if having display issues

---

## 🆘 Still Having Issues?

If you're still experiencing problems:

1. **Run Full Diagnostic:**
   ```powershell
   cd "C:\Users\Chris\HR Applicant Tracker (HAT)"
   python -c "import sys; print('Python:', sys.version)"
   pip list | findstr streamlit
   ```

2. **Check Error Messages:** Look at the exact error message in the console

3. **Try Safe Mode:** Run with minimal features to isolate the issue

**Remember:** The system includes multiple fallbacks and should work even if some advanced features aren't available! 🚀
