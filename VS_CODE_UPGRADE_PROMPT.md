# 🧠 VS Code AI Pair-Programming Prompt

**Project: Hotel Applicant Tracker (HAT) — Production-ready upgrade**

**Goals:**

1. Make the ATS robust and "better than anything close to it".
2. Zero runtime crashes; great DX; clean UX.
3. Keep current API/routers working; add tests.

**Do now (step by step, with commits):**

## 1. **App core**

* ✅ Use the provided `main.py` exactly. Keep routers under `app/routes/*` intact.
* ✅ Add a `settings` schema in `app/config.py` with `ENV`, `APP_NAME`, `APP_VERSION`, `CORS_ALLOW_ORIGINS` (list).

## 2. **Error model & responses**

* Create a shared `app/common/errors.py` with `api_error(code, message, extra=None)` returning FastAPI `JSONResponse`.
* Ensure every router uses `HTTPException` or this helper with consistent payloads.

## 3. **Validation & schemas**

* Add Pydantic models for **Role, Candidate, Application, RankRequest/Response** in `app/schemas/*`.
* Enforce request/response models on all endpoints.

## 4. **Auth**

* Implement JWT auth (access + refresh) with dependency guards on `/api/*`.
* Add rate-limit (slowapi) on auth endpoints.

## 5. **Ingestion & ranking**

* Create a background task queue (e.g., `arq` or `RQ`) to parse resumes (PDF/DOCX) and extract: name, email, phone, skills, years, last roles.
* Store extracted fields plus raw text.
* Implement a simple **skill match score** (0–100) per role (TF-IDF or rapidfuzz), cached.
* Endpoint: `/api/rank?role_id=...` returns sorted candidates with score and top matching skills.

## 6. **Search**

* Add `/api/candidates/search?q=...&skill=...&min_years=...` using SQLite FTS5 or PostgreSQL `tsvector` if available.

## 7. **Export**

* `/api/export/shortlist.xlsx` builds an Excel with top N candidates (name, email, phone, score, notes).

## 8. **Frontend pages** (server-rendered minimal)

* `templates/` pages: roles list, candidates list, candidate detail, shortlist view.
* Add basic Bootstrap 5; keep it clean and fast.

## 9. **Ops**

* Add `uvicorn[standard]`, `python-multipart`, `python-docx`, `pypdf`, `rapidfuzz`, `pandas`, `openpyxl`, `slowapi`, `structlog`.
* Add `Dockerfile` (multi-stage) + `docker-compose.yml` for app + worker + Redis (queue).

## 10. **Quality**

* Add `ruff` + `mypy` configs; fix lint/type issues.
* Add unit tests for ranking, parsing, and one router.

**Deliverables:**

* Updated code, configs, and templates.
* A `README.md` with **Run locally**, **Docker**, **API examples**, **Screenshots**.
* Seed script: `scripts/seed_demo.py` to create demo roles and fake candidates.
* Everything should run with:

  * `uvicorn main:app --reload` (dev)
  * `docker compose up --build` (prod-like)

---

## Current Project Structure:

```
c:\Users\Chris\HR Applicant Tracker (HAT)\
├── app/
│   ├── __init__.py
│   ├── config.py ✅ (updated with ENV, CORS settings)
│   ├── main.py ✅ (hardened FastAPI with structured logging, middleware)
│   ├── deps.py
│   ├── models/ (SQLAlchemy models)
│   ├── routes/ (existing routers)
│   ├── services/
│   ├── static/ (may not exist)
│   └── templates/ (may not exist)
├── enhanced_ai_screener.py (standalone AI screener)
├── enhanced_streamlit_app.py (Streamlit web interface)
├── Quick_Start_Screener.bat (1-click launcher)
└── input_resumes/ (resume processing folder)
```

## Key Integration Points:

1. **Keep existing enhanced AI screener**: The standalone `enhanced_ai_screener.py` and Streamlit interface should remain functional
2. **Upgrade FastAPI app**: Make the main web application production-ready
3. **Bridge both systems**: Allow users to choose between standalone screening and full ATS
4. **Hotel-specific**: Maintain focus on hospitality industry requirements

## Implementation Priority:

1. **Core stability** (error handling, validation, auth)
2. **Resume processing** (integrate existing AI capabilities)
3. **Search & ranking** (leverage existing algorithms)
4. **Professional UI** (clean templates)
5. **Production readiness** (Docker, tests, docs)

---

**Start with step 2 (error handling) and work through each step systematically. Focus on making the FastAPI application as robust and professional as the enhanced Streamlit interface.**
