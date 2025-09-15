# Job Descriptions Folder

Place one file per role in this folder. Files are used to enrich scoring with JD keywords and semantic similarity.

Accepted formats:
- .txt (preferred)
- .pdf (text-based PDFs preferred; scanned PDFs need OCR via Tesseract)

File naming rules:
- Name the file exactly like the position (case-insensitive).
- Spaces or underscores are fine; avoid hyphens.
- Examples:
  - `Front Office Manager.txt`
  - `Head Butler.pdf`
  - `Housekeeping Supervisor.txt`
  - `Diamond Club Manager.pdf`

Tips:
- Keep each file focused on the responsibilities, must-have skills, and nice-to-haves.
- Shorter, clean JD text improves signal.

Advanced:
- You can pass a custom path when creating the screener:
  ```python
  from enhanced_ai_screener import EnhancedHotelAIScreener
  screener = EnhancedHotelAIScreener(job_desc_dir="C:/path/to/JDs")
  ```
