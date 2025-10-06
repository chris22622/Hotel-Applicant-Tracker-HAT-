#!/usr/bin/env python3
"""
Enhanced Hotel AI Resume Screener - Version 2.0
Advanced AI-powered candidate selection with semantic matching, bias detection, and intelligent scoring
"""

import os
import sys
import json
import hashlib
import yaml
import yaml
import logging
import time
import threading
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import pandas as pd
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Enhanced imports with fallbacks
try:
    import spacy
    from spacy.matcher import Matcher
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
    logger.info("✅ SpaCy loaded successfully")
except (ImportError, OSError):
    spacy_available = False
    nlp = None
    logger.warning("⚠️ SpaCy not available - using basic text processing")

try:
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    import pdf2image
    ocr_available = True
    logger.info("✅ OCR capabilities loaded")
except ImportError:
    ocr_available = False
    logger.warning("⚠️ OCR not available - text files only")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
    logger.info("✅ Advanced text analysis available")
except ImportError:
    sklearn_available = False
    logger.warning("⚠️ Advanced text analysis not available")

# Optional LLM client (OpenAI) for full resume review
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_OK = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_OK = False


SCORING_VERSION = "4.0.0"

# Role ontology for title normalization
ROLE_ONTOLOGY: Dict[str, List[str]] = {
    "front desk agent": ["front desk", "receptionist", "guest services agent", "front office associate"],
    "front desk manager": ["front office manager", "guest services manager", "front desk supervisor"],
    "housekeeper": ["room attendant", "housekeeping attendant", "cleaning staff"],
    "housekeeping supervisor": ["housekeeping lead", "housekeeping manager"],
    "server": ["waiter", "waitress", "food server", "restaurant server"],
    "host": ["hostess", "restaurant host"],
    "bartender": ["bar staff", "mixologist"],
    "chef": ["cook", "line cook", "sous chef", "head chef", "executive chef"],
    "butler": ["head butler", "personal butler", "butler supervisor"],
    "concierge": ["guest concierge"],
    "spa therapist": ["massage therapist", "spa specialist"],
    "security officer": ["security guard", "loss prevention"],
    "sales manager": ["hospitality sales manager", "hotel sales manager"],
}

# Skill alias graph (maps alias -> canonical skill)
SKILL_ALIASES: Dict[str, str] = {
    "pos": "point of sale",
    "pms": "property management system",
    "ms office": "microsoft office",
    "excel": "microsoft excel",
    "word": "microsoft word",
    "outlook": "microsoft outlook",
    "cust service": "customer service",
    "cs": "customer service",
    "food prep": "food preparation",
    "mixology": "bartending",
    "line cook": "cook",
    "sous chef": "chef",
    "exec chef": "executive chef",
}

# Negative domain signals (if frequently present, candidate likely outside hospitality focus)
NEGATIVE_DOMAIN_TERMS = [
    "warehouse", "forklift", "logistics", "assembly line", "manufacturing",
    "construction", "oilfield", "mining", "truck driving", "long haul",
]

class EnhancedHotelAIScreener:
    """Enhanced AI-powered hotel resume screener with advanced matching algorithms."""

    def __init__(self, input_dir: str = "input_resumes", output_dir: str = "screening_results", job_desc_dir: Optional[str] = "job_descriptions", config_dir: str = "config"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Initialize enhanced position intelligence
        self.position_intelligence = self._load_enhanced_position_intelligence()

        # Attempt to load external scoring overrides
        self._config = {}
        scoring_cfg = self.config_dir / "scoring.yaml"
        if scoring_cfg.exists():
            try:
                with open(scoring_cfg, "r", encoding="utf-8") as fh:
                    self._config = yaml.safe_load(fh) or {}
                logger.info("🛠 Loaded scoring config overrides from scoring.yaml")
            except Exception as e:
                logger.warning(f"Failed to load scoring config: {e}")

        # Initialize skill taxonomy
        self.skill_taxonomy = self._build_skill_taxonomy()

        # Optional job description directory (PDF / TXT) enrichment
        self.job_desc_dir = Path(job_desc_dir) if job_desc_dir else None
        if self.job_desc_dir and self.job_desc_dir.exists():
            self._augment_with_job_descriptions(self.job_desc_dir)
        else:
            logger.info("ℹ️ No external job descriptions folder found or provided.")

        # Track duplicate resume hashes
        self._seen_hashes = set()

        # Initialize semantic matcher if spacy available
        if spacy_available and nlp:
            self.matcher = Matcher(nlp.vocab)
            self._setup_patterns()
        # Embedding cache & model placeholders
        self._embedding_model = None
        self._embedding_dim = None
        self._embed_cache: Dict[str, List[float]] = {}
        # Try load persisted embeddings
        try:
            self._load_embedding_cache()
        except Exception:
            pass

        # Explain / debug modes
        self._explain_mode = False
        self._debug_emb = False

        # Plugin loader cache (lazy-loaded list of callables)
        self._plugin_hooks = None

        logger.info("🏨 Enhanced Hotel AI Screener initialized")
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
        if self.job_desc_dir:
            logger.info(f"📄 Job Descriptions: {self.job_desc_dir} (loaded)")
        logger.info(f"🧪 Scoring Version: {SCORING_VERSION}")
        # Performance tuning flags (via environment variables)
        self.fast_mode: bool = str(os.getenv('HAT_FAST_MODE', '0')).strip().lower() in ('1','true','yes','on')
        self.disable_ocr: bool = str(os.getenv('HAT_DISABLE_OCR', '0')).strip().lower() in ('1','true','yes','on')
        try:
            self.ocr_max_pages: int = int(os.getenv('HAT_OCR_MAX_PAGES', '2'))
        except Exception:
            self.ocr_max_pages = 2
        try:
            self.max_files: int = int(os.getenv('HAT_MAX_FILES', '0'))
        except Exception:
            self.max_files = 0
        if self.fast_mode:
            logger.info("⚡ Fast mode enabled")
        if self.disable_ocr:
            logger.info("🛑 OCR disabled by config")
        if self.max_files > 0:
            logger.info(f"📦 File cap per run: {self.max_files}")

        # Configure Tesseract path on Windows if available
        try:
            if ocr_available:
                import platform
                if platform.system().lower().startswith('win'):
                    default_tess = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                    try:
                        import pytesseract as _pt  # type: ignore
                        if os.path.exists(default_tess):
                            _pt.pytesseract.tesseract_cmd = default_tess  # type: ignore
                    except Exception:
                        pass
        except Exception:
            pass

        # Default Tesseract runtime config (good general-mode)
        self._tess_config = "--oem 3 --psm 6"

        # Optional: LLM full-text review settings
        self._llm_cfg = (self._config.get("llm") or {}) if hasattr(self, "_config") else {}
        env_flag = str(os.getenv('HAT_LLM_FULL_REVIEW', self._llm_cfg.get('full_review_enabled', '0'))).strip().lower()
        self._llm_full_review_enabled: bool = env_flag in ('1','true','yes','on')
        self._llm_model: str = str(self._llm_cfg.get('model', 'gpt-4o-mini'))
        try:
            self._llm_temperature: float = float(self._llm_cfg.get('temperature', 0.1))
        except Exception:
            self._llm_temperature = 0.1
        try:
            self._llm_timeout: int = int(self._llm_cfg.get('timeout', 45))
        except Exception:
            self._llm_timeout = 45
        try:
            self._llm_chunk_chars: int = int(self._llm_cfg.get('chunk_chars', 6000))
        except Exception:
            self._llm_chunk_chars = 6000

        # Resilience controls to avoid long retry storms
        try:
            self._llm_max_retries: int = int(self._llm_cfg.get('max_retries', 0))
        except Exception:
            self._llm_max_retries = 0
        # Do NOT disable on rate-limit by default; only on hard errors when enabled
        self._llm_disable_on_error: bool = str(self._llm_cfg.get('disable_on_error', 'false')).strip().lower() in ('1','true','yes','on')
        fb = self._llm_cfg.get('fallback_model')
        self._llm_fallback_models: List[str] = []
        if isinstance(fb, str) and fb:
            self._llm_fallback_models.append(fb)
        for m in ('gpt-4o', 'gpt-4.1-mini'):
            if m != self._llm_model and m not in self._llm_fallback_models:
                self._llm_fallback_models.append(m)

        # Rate limit and concurrency controls
        try:
            self._llm_requests_per_minute: float = float(self._llm_cfg.get('requests_per_minute', 8))
        except Exception:
            self._llm_requests_per_minute = 8.0
        try:
            self._llm_cooldown_on_429: int = int(self._llm_cfg.get('cooldown_on_429_secs', 30))
        except Exception:
            self._llm_cooldown_on_429 = 30
        try:
            self._llm_max_concurrent: int = int(self._llm_cfg.get('max_concurrent', 1))
        except Exception:
            self._llm_max_concurrent = 1
        try:
            self._llm_max_chunks_per_resume: int = int(self._llm_cfg.get('max_chunks_per_resume', 2))
        except Exception:
            self._llm_max_chunks_per_resume = 2
        self._llm_show_cooldown_progress: bool = str(self._llm_cfg.get('show_cooldown_progress', 'true')).strip().lower() in ('1','true','yes','on')

        self._llm_min_interval = 60.0 / self._llm_requests_per_minute if self._llm_requests_per_minute > 0 else 0.0
        self._llm_next_allowed_ts: float = 0.0
        self._llm_sema = threading.Semaphore(max(1, self._llm_max_concurrent))
        self._llm_last_status: Optional[int] = None

        # LLM metrics
        self._llm_metrics: Dict[str, Any] = {
            "calls_total": 0,
            "success_total": 0,
            "fail_total": 0,
            "rate_limit_events": 0,
            "cooldown_seconds": 0,
            "candidates_scored": 0,
            "candidates_skipped_429": 0,
        }

        # Thoroughness mode overrides (env: HAT_THOROUGHNESS=fast|balanced|full)
        try:
            mode = str(os.getenv('HAT_THOROUGHNESS', 'balanced')).strip().lower()
            if mode in ("fast", "balanced", "full"):
                if mode == "fast":
                    self._llm_requests_per_minute = min(self._llm_requests_per_minute, 4)
                    self._llm_max_chunks_per_resume = 1 if self._llm_max_chunks_per_resume != 0 else 1
                    self._llm_min_interval = 60.0 / self._llm_requests_per_minute
                elif mode == "balanced":
                    # keep config defaults
                    pass
                elif mode == "full":
                    # be more thorough (may trigger more 429s if RPM is too high)
                    if self._llm_max_chunks_per_resume != 0:
                        self._llm_max_chunks_per_resume = 0  # no cap
                    # allow a bit higher RPM if user configured low
                    self._llm_requests_per_minute = max(self._llm_requests_per_minute, 12)
                    self._llm_min_interval = 60.0 / self._llm_requests_per_minute
                logger.info(f"🧭 Thoroughness mode: {mode} (rpm={self._llm_requests_per_minute}, chunks_cap={self._llm_max_chunks_per_resume})")
        except Exception:
            pass

        # LLM cache and client
        self._llm_cache_path = Path("var") / "llm_full_cache.json"
        self._llm_cache: Dict[str, Any] = {}
        try:
            self._llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self._llm_cache_path.exists():
                self._llm_cache = json.loads(self._llm_cache_path.read_text(encoding='utf-8'))
        except Exception:
            self._llm_cache = {}
        self._llm_client = self._init_llm_client() if self._llm_full_review_enabled else None
        if self._llm_full_review_enabled and not self._llm_client:
            logger.warning("⚠️ LLM full review enabled but OpenAI client not available. Set OPENAI_API_KEY or disable the feature.")

    # ---------------- Metrics & Benchmarks -----------------
    def _metrics_dir(self) -> Path:
        d = Path("var/metrics")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _scores_log_path(self) -> Path:
        return self._metrics_dir() / "scores.jsonl"

    def _read_score_samples(self) -> List[Dict[str, Any]]:
        p = self._scores_log_path()
        samples: List[Dict[str, Any]] = []
        if not p.exists():
            return samples
        try:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "score" in rec and "position" in rec:
                            samples.append(rec)
                    except Exception:
                        continue
        except Exception:
            pass
        return samples

    def _append_score_sample(self, position: str, score: float) -> None:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "position": position,
            "score": float(score),
            "version": SCORING_VERSION,
        }
        try:
            with open(self._scores_log_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.debug(f"Scores log append failed: {e}")

    def _percentile(self, values: List[float], x: float) -> Optional[float]:
        if not values:
            return None
        try:
            # percent of values <= x
            n = len(values)
            rank = sum(1 for v in values if v <= x)
            return round(100.0 * rank / n, 2)
        except Exception:
            return None

    # ---------------- Plugin System -----------------
    def _plugins_dir(self) -> Path:
        d = Path("plugins")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_plugins(self) -> List[Any]:
        if self._plugin_hooks is not None:
            return self._plugin_hooks
        hooks: List[Any] = []
        try:
            import importlib.util
            for py in self._plugins_dir().glob("*.py"):
                if py.name.startswith("__"):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(py.stem, str(py))
                    if spec and spec.loader:  # type: ignore
                        mod = importlib.util.module_from_spec(spec)  # type: ignore
                        spec.loader.exec_module(mod)  # type: ignore
                        fn = getattr(mod, "augment_score", None)
                        if callable(fn):
                            hooks.append({"id": py.stem, "fn": fn})
                except Exception as pe:
                    logger.debug(f"Plugin load failed for {py.name}: {pe}")
        except Exception as e:
            logger.debug(f"Plugin discovery error: {e}")
        self._plugin_hooks = hooks
        if hooks:
            logger.info(f"🔌 Loaded {len(hooks)} plugin(s)")
        return hooks

    def _apply_scoring_overrides(self, weights: Dict[str, float]) -> Dict[str, float]:
        try:
            override = (self._config.get("weights") or {}) if hasattr(self, "_config") else {}
            if not override:
                return weights
            merged = {**weights}
            for k,v in override.items():
                if isinstance(v, (int,float)) and k in merged:
                    merged[k] = float(v)
            total = sum(merged.values())
            if total > 0:
                # normalize back to 1.0
                merged = {k: v/total for k,v in merged.items()}
            return merged
        except Exception:
            return weights

    # ---------------- Explain Mode Controls -----------------
    def set_explain_mode(self, enabled: bool = True, debug_embeddings: bool = False) -> None:
        """Toggle global explainability output.

        enabled: if True, scoring returns detailed 'explanation' dict
        debug_embeddings: if True, include embedding similarity raw values
        """
        self._explain_mode = bool(enabled)
        self._debug_emb = bool(debug_embeddings)
        logger.info(f"🔍 Explain mode {'ENABLED' if self._explain_mode else 'DISABLED'} (embeddings debug={'on' if self._debug_emb else 'off'})")

    # ---------------- Embedding Utilities -----------------
    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_name = (self._config.get("embeddings", {}) or {}).get("model", "all-MiniLM-L6-v2")
            self._embedding_model = SentenceTransformer(model_name)
            self._embedding_dim = len(self._embedding_model.encode(["test"], show_progress_bar=False)[0])  # type: ignore
            logger.info(f"🧠 Loaded embedding model: {model_name}")
        except Exception as e:
            logger.info(f"ℹ️ Embedding model unavailable ({e}); proceeding without embeddings")
            self._embedding_model = None
        return self._embedding_model

    def _embed_text(self, text: str) -> Optional[List[float]]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in self._embed_cache:
            return self._embed_cache[key]
        model = self._get_embedding_model()
        if not model:
            return None
        try:
            vec = model.encode([text], show_progress_bar=False)[0]  # type: ignore
            self._embed_cache[key] = list(map(float, vec))
            # Persist occasionally
            if len(self._embed_cache) % 25 == 0:
                try:
                    self._save_embedding_cache()
                except Exception:
                    pass
            return self._embed_cache[key]
        except Exception:
            return None

    def _cache_dir(self) -> Path:
        d = Path("var/cache")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _embedding_cache_path(self) -> Path:
        return self._cache_dir() / "embeddings.json"

    def _load_embedding_cache(self) -> None:
        p = self._embedding_cache_path()
        if p.exists():
            try:
                import json
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    # ensure list[float]
                    cleaned = {}
                    for k,v in data.items():
                        if isinstance(v, list):
                            try:
                                cleaned[k] = [float(x) for x in v]
                            except Exception:
                                continue
                    self._embed_cache.update(cleaned)
                    logger.info(f"🗂 Loaded {len(cleaned)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed loading embedding cache: {e}")

    def _save_embedding_cache(self) -> None:
        try:
            import json
            p = self._embedding_cache_path()
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self._embed_cache, fh)
        except Exception as e:
            logger.debug(f"Embed cache save skipped: {e}")

    def _embedding_cosine(self, a: List[float], b: List[float]) -> float:
        try:
            import math
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na*nb)
        except Exception:
            return 0.0

    # ------------------------ Section Segmentation ------------------------
    def _segment_resume_sections(self, text: str) -> Dict[str, str]:
        """Rudimentary section segmentation using regex headers.
        Returns dict {section_name: joined_text}.
        Safe fallback: returns single 'full_text' if no headers.
        """
        try:
            lines = [ln.strip() for ln in text.splitlines()]
            header_pattern = re.compile(r"^(experience|work experience|professional experience|education|skills|certifications|summary|profile|objective|projects|training)[:\-]?$", re.IGNORECASE)
            sections: Dict[str, List[str]] = {}
            current = None
            for ln in lines:
                if not ln:
                    continue
                if header_pattern.match(ln.lower()):
                    current = ln.lower().rstrip(':')
                    sections.setdefault(current, [])
                else:
                    if current is None:
                        current = 'preamble'
                        sections.setdefault(current, [])
                    sections[current].append(ln)
            if not sections:
                return {"full_text": text}
            return {k: "\n".join(v) for k,v in sections.items()}
        except Exception:
            return {"full_text": text}

    # ------------------------ Career Timeline Analysis ------------------------
    def _analyze_career_timeline(self, text: str) -> Dict[str, Any]:
        """Extract years and detect gaps. Simple heuristic: find year ranges like 2019-2022.
        Returns {roles: [...], total_years: float, gaps: [...]}.
        """
        try:
            year_range_pattern = re.compile(r"(19\d{2}|20\d{2})\s*[\-–to]{1,3}\s*(19\d{2}|20\d{2}|present|current)", re.IGNORECASE)
            single_year_pattern = re.compile(r"(19\d{2}|20\d{2})")
            ranges = []
            for m in year_range_pattern.finditer(text):
                start_raw, end_raw = m.group(1), m.group(2)
                try:
                    start = int(start_raw)
                    end = datetime.now().year if end_raw.lower() in ("present","current") else int(end_raw)
                    if end >= start and 1900 <= start <= datetime.now().year:
                        ranges.append((start,end))
                except Exception:
                    continue
            # If no ranges, attempt approximate from individual years
            if not ranges:
                years = sorted({int(y) for y in single_year_pattern.findall(text) if 1900 <= int(y) <= datetime.now().year})
                if len(years) >= 2:
                    ranges.append((years[0], years[-1]))
            # Merge overlaps
            ranges = sorted(ranges)
            merged = []
            for r in ranges:
                if not merged or r[0] > merged[-1][1] + 1:
                    merged.append(list(r))
                else:
                    merged[-1][1] = max(merged[-1][1], r[1])
            total = sum(e - s + 1 for s,e in merged)
            # Detect gaps between merged intervals > 6 months (approx by year diff >1)
            gaps = []
            for i in range(1, len(merged)):
                prev_end = merged[i-1][1]
                cur_start = merged[i][0]
                if cur_start - prev_end > 1:
                    gaps.append({"from": prev_end+1, "to": cur_start-1})
            return {
                "intervals": merged,
                "total_year_span": total,
                "gaps": gaps,
                "gap_count": len(gaps)
            }
        except Exception:
            return {"intervals": [], "total_year_span": 0, "gaps": [], "gap_count": 0}

    def _temporal_experience_weight(self, timeline: Dict[str, Any]) -> float:
        """Compute a temporal weighting factor favoring recent continuous experience.

        Heuristic:
        - Determine coverage in last 5 years, last 2 years.
        - Reward if at least one interval touches current year (recency).
        Returns multiplier in [0.9, 1.15].
        """
        try:
            intervals = timeline.get("intervals") or []
            if not intervals:
                return 1.0
            current_year = datetime.now().year
            last5_start = current_year - 5
            last2_start = current_year - 2
            years_last5 = set()
            years_last2 = set()
            for s,e in intervals:
                for y in range(s, e+1):
                    if y >= last5_start:
                        years_last5.add(y)
                    if y >= last2_start:
                        years_last2.add(y)
            coverage5 = len(years_last5) / 6.0  # 0..1 approx (6 year span inclusive)
            coverage2 = len(years_last2) / 3.0
            recency = any(e >= current_year-1 for _,e in intervals)
            base = 1.0 + 0.08*min(1.0, coverage5) + 0.05*min(1.0, coverage2)
            if recency:
                base += 0.02
            return float(max(0.9, min(1.15, base)))
        except Exception:
            return 1.0

    def _augment_with_job_descriptions(self, folder: Path) -> None:
        """Enrich position intelligence with keywords extracted from job description PDF/TXT files.

        Strategy:
        - Match filename (stem) to existing position key (case-insensitive) or create new entry.
        - Extract text from .txt directly; for .pdf try PyPDF2; fall back to OCR if available.
        - Tokenize, remove stopwords / boilerplate; keep top-N frequent unique tokens not already present.
        - Store in position_data['job_description_keywords'] (max 50) for scoring bonus later.
        - Maintain raw text map for potential TF-IDF similarity (self._jd_raw_texts).
        """
        self._jd_raw_texts: Dict[str, str] = {}
        files = list(folder.glob("*.txt")) + list(folder.glob("*.pdf"))
        if not files:
            logger.info(f"ℹ️ No job description files found in {folder}")
            return

        try:
            import PyPDF2  # type: ignore
            pdf_ok = True
        except Exception:
            pdf_ok = False
            logger.warning("⚠️ PyPDF2 not installed - PDF JD parsing limited (install PyPDF2 for better extraction)")

        stop_words = {
            "the","and","for","with","that","this","from","into","your","you","our","are","was","were",
            "will","shall","but","not","any","all","per","job","role","responsibilities","duties","ability",
            "skills","other","such","etc","more","have","has","one","two","three","may","their","them","they",
            "her","his","she","him","who","prior","work","years","year","requirements","qualifications","position",
            "hotel","hospitality","guest","service","experience","candidate","team","perform","ensure","provide"
        }

        def extract_text(path: Path) -> str:
            if path.suffix.lower() == ".txt":
                return path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".pdf":
                text = ""
                # Try PyPDF2
                if pdf_ok:
                    try:
                        with open(path, "rb") as fh:
                            reader = PyPDF2.PdfReader(fh)  # type: ignore
                            for page in reader.pages:
                                page_text = page.extract_text() or ""
                                text += page_text + "\n"
                    except Exception as e:
                        logger.debug(f"PDF parse issue ({path.name}) via PyPDF2: {e}")
                # Fallback OCR if empty and OCR available
                if not text.strip() and ocr_available:
                    try:
                        images = pdf2image.convert_from_path(path)  # type: ignore
                        ocr_chunks = []
                        for img in images[:10]:  # safety cap
                            try:
                                ocr_chunks.append(pytesseract.image_to_string(img))  # type: ignore
                            except Exception:
                                pass
                        text = "\n".join(ocr_chunks)
                    except Exception as e:
                        logger.debug(f"OCR fallback failed for {path.name}: {e}")
                return text
            return ""

        def extract_keywords(text: str) -> List[str]:
            tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
            freq = Counter(t for t in tokens if t not in stop_words)
            return [w for w, _ in freq.most_common(120)]

        enriched = 0
        for f in files:
            try:
                raw = extract_text(f)
                if len(raw.strip()) < 40:
                    continue
                stem = f.stem.replace("_", " ").strip()
                # map to existing position intelligence key if case-insensitive match
                key = None
                for existing in self.position_intelligence.keys():
                    if existing.lower() == stem.lower():
                        key = existing
                        break
                if key is None:
                    key = stem
                    # Create minimal structure if new
                    if key not in self.position_intelligence:
                        self.position_intelligence[key] = {
                            "must_have_skills": [],
                            "nice_to_have_skills": [],
                            "technical_skills": [],
                            "soft_skills": [],
                            "experience_indicators": [],
                            "cultural_fit_keywords": [],
                            "disqualifying_factors": [],
                            "scoring_weights": {"experience":0.30,"skills":0.30,"cultural_fit":0.20,"hospitality":0.20},
                            "min_experience_years": 0,
                            "preferred_experience_years": 1
                        }
                kw = extract_keywords(raw)
                existing_sets = set(
                    (self.position_intelligence[key].get("job_description_keywords") or []) +
                    self.position_intelligence[key].get("must_have_skills", []) +
                    self.position_intelligence[key].get("nice_to_have_skills", []) +
                    self.position_intelligence[key].get("technical_skills", []) +
                    self.position_intelligence[key].get("experience_indicators", [])
                )
                jd_keywords = [w for w in kw if w not in existing_sets][:50]
                if jd_keywords:
                    self.position_intelligence[key]["job_description_keywords"] = jd_keywords
                    enriched += 1
                self._jd_raw_texts[key] = raw
            except Exception as fe:
                logger.debug(f"Job description parse failed for {f.name}: {fe}")
                continue

        logger.info(f"🧠 Job description enrichment applied to {enriched} position(s)")
    
    def _load_enhanced_position_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive hotel position intelligence with advanced requirements for ALL hotel positions."""
        return {
            # ==========================================
            # EXECUTIVE MANAGEMENT
            # ==========================================
            "General Manager": {
                "must_have_skills": [
                    "executive leadership", "strategic planning", "operations management", "financial management",
                    "team leadership", "hospitality management", "budget planning", "revenue management",
                    "staff development", "guest relations", "crisis management", "business development"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "resort management", "franchise operations", "asset management",
                    "brand standards", "market analysis", "competitor analysis", "ROI optimization",
                    "stakeholder management", "board reporting", "contract negotiation"
                ],
                "technical_skills": [
                    "hotel management systems", "financial reporting", "performance analytics",
                    "revenue management systems", "budgeting software", "market intelligence tools"
                ],
                "soft_skills": [
                    "visionary leadership", "emotional intelligence", "decision making", "communication",
                    "adaptability", "crisis management", "strategic thinking", "negotiation"
                ],
                "cultural_fit_keywords": [
                    "visionary", "leader", "strategic", "results-driven", "innovative",
                    "guest-focused", "team builder", "ethical", "accountable", "inspiring"
                ],
                "disqualifying_factors": [
                    "poor leadership record", "financial mismanagement", "lack of hospitality experience",
                    "poor communication", "inflexible", "unethical behavior"
                ],
                "experience_indicators": [
                    "general manager", "hotel manager", "resort manager", "operations manager",
                    "executive leadership", "hospitality management", "property management"
                ],
                "education_preferences": [
                    "hospitality management", "business administration", "hotel administration",
                    "MBA", "management", "finance", "operations management"
                ],
                "certifications": [
                    "CHA (Certified Hotel Administrator)", "hospitality management", "leadership",
                    "financial management", "revenue management"
                ],
                "scoring_weights": {
                    "experience": 0.45, "skills": 0.25, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 10,
                "preferred_experience_years": 15,
                "salary_range": {"min": 120000, "max": 250000},
                "growth_potential": "Executive",
                "training_requirements": "Strategic"
            },

            "Hotel Manager": {
                "must_have_skills": [
                    "hotel operations", "team management", "guest relations", "budget management",
                    "staff supervision", "quality control", "hospitality operations", "leadership",
                    "problem solving", "communication", "time management", "performance management"
                ],
                "nice_to_have_skills": [
                    "luxury service", "resort operations", "event management", "sales coordination",
                    "revenue optimization", "brand standards", "guest recovery", "staff training"
                ],
                "technical_skills": [
                    "hotel management systems", "PMS systems", "reporting tools", "scheduling software",
                    "performance dashboards", "guest feedback systems"
                ],
                "soft_skills": [
                    "leadership", "emotional intelligence", "decision making", "conflict resolution",
                    "adaptability", "stress management", "mentoring", "communication"
                ],
                "cultural_fit_keywords": [
                    "leader", "hospitality-focused", "team player", "results-oriented",
                    "guest advocate", "professional", "reliable", "innovative"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of hospitality experience", "poor communication",
                    "inability to handle stress", "inflexible"
                ],
                "experience_indicators": [
                    "hotel manager", "assistant manager", "operations manager", "hospitality management",
                    "hotel operations", "property management", "guest services management"
                ],
                "education_preferences": [
                    "hospitality management", "hotel administration", "business management",
                    "tourism management", "operations management"
                ],
                "certifications": [
                    "hospitality management", "hotel operations", "leadership certification",
                    "customer service excellence", "revenue management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.25, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 65000, "max": 120000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Operations Manager": {
                "must_have_skills": [
                    "operations management", "process optimization", "team coordination", "quality control",
                    "budget oversight", "performance monitoring", "staff management", "logistics coordination",
                    "vendor management", "compliance", "efficiency improvement", "project management"
                ],
                "nice_to_have_skills": [
                    "hotel operations", "multi-department coordination", "systems integration",
                    "cost reduction", "workflow optimization", "technology implementation", "training coordination"
                ],
                "technical_skills": [
                    "operations software", "performance analytics", "project management tools",
                    "reporting systems", "workflow management", "compliance tracking"
                ],
                "soft_skills": [
                    "analytical thinking", "leadership", "problem solving", "communication",
                    "organization", "attention to detail", "adaptability", "decision making"
                ],
                "cultural_fit_keywords": [
                    "efficient", "organized", "analytical", "improvement-focused",
                    "team player", "results-driven", "detail-oriented", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of analytical skills", "poor communication",
                    "inability to manage complexity", "inflexible"
                ],
                "experience_indicators": [
                    "operations manager", "process manager", "operations coordination",
                    "operations analysis", "workflow management", "efficiency improvement"
                ],
                "education_preferences": [
                    "operations management", "business administration", "industrial engineering",
                    "process management", "project management", "hospitality management"
                ],
                "certifications": [
                    "operations management", "project management", "process improvement",
                    "quality management", "efficiency certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 55000, "max": 85000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Assistant Manager": {
                "must_have_skills": [
                    "assistant management", "team support", "guest relations", "operations support",
                    "staff coordination", "problem solving", "communication", "time management",
                    "hospitality operations", "customer service", "quality assurance", "administrative skills"
                ],
                "nice_to_have_skills": [
                    "management training", "leadership development", "guest recovery",
                    "staff training", "performance monitoring", "event coordination", "multi-tasking"
                ],
                "technical_skills": [
                    "hotel systems", "administrative software", "reporting tools",
                    "scheduling systems", "communication platforms", "guest management systems"
                ],
                "soft_skills": [
                    "support skills", "communication", "adaptability", "teamwork",
                    "problem solving", "attention to detail", "reliability", "learning agility"
                ],
                "cultural_fit_keywords": [
                    "supportive", "reliable", "team player", "guest-focused",
                    "professional", "adaptable", "eager to learn", "helpful"
                ],
                "disqualifying_factors": [
                    "poor communication", "lack of reliability", "poor customer service",
                    "inability to work in team", "inflexible"
                ],
                "experience_indicators": [
                    "assistant manager", "supervisor", "team lead", "hospitality support",
                    "guest services", "operations support", "management trainee"
                ],
                "education_preferences": [
                    "hospitality management", "business administration", "hotel management",
                    "customer service", "management studies"
                ],
                "certifications": [
                    "hospitality management", "customer service", "leadership development",
                    "hotel operations", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.25, "cultural_fit": 0.25, "hospitality": 0.20
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Executive Assistant": {
                "must_have_skills": [
                    "executive support", "administrative skills", "communication", "organization",
                    "calendar management", "correspondence", "meeting coordination", "document preparation",
                    "travel arrangements", "confidentiality", "time management", "multitasking"
                ],
                "nice_to_have_skills": [
                    "project coordination", "event planning", "stakeholder communication",
                    "presentation preparation", "database management", "expense management", "protocol knowledge"
                ],
                "technical_skills": [
                    "Microsoft Office Suite", "calendar software", "communication platforms",
                    "travel booking systems", "expense management software", "document management"
                ],
                "soft_skills": [
                    "discretion", "professionalism", "attention to detail", "communication",
                    "adaptability", "problem solving", "reliability", "interpersonal skills"
                ],
                "cultural_fit_keywords": [
                    "professional", "discreet", "organized", "reliable",
                    "detail-oriented", "proactive", "supportive", "confidential"
                ],
                "disqualifying_factors": [
                    "poor organization", "lack of discretion", "poor communication",
                    "unreliable", "inability to handle confidential information"
                ],
                "experience_indicators": [
                    "executive assistant", "administrative assistant", "personal assistant",
                    "executive support", "administrative support", "office management"
                ],
                "education_preferences": [
                    "business administration", "office administration", "communications",
                    "secretarial studies", "administrative management"
                ],
                "certifications": [
                    "administrative professional", "executive assistant certification",
                    "office management", "business communication"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Duty Manager": {
                "must_have_skills": [
                    "emergency management", "guest relations", "staff coordination", "problem solving",
                    "communication", "decision making", "hospitality operations", "crisis management",
                    "customer service", "time management", "multi-department coordination", "leadership"
                ],
                "nice_to_have_skills": [
                    "night operations", "security coordination", "guest recovery",
                    "incident reporting", "staff support", "emergency procedures", "conflict resolution"
                ],
                "technical_skills": [
                    "hotel management systems", "emergency systems", "communication systems",
                    "reporting tools", "security systems", "guest management systems"
                ],
                "soft_skills": [
                    "calm under pressure", "leadership", "decision making", "communication",
                    "adaptability", "problem solving", "reliability", "confidence"
                ],
                "cultural_fit_keywords": [
                    "calm under pressure", "reliable", "leader", "problem-solver",
                    "guest-focused", "professional", "decisive", "responsible"
                ],
                "disqualifying_factors": [
                    "poor under pressure", "poor decision making", "lack of leadership",
                    "poor communication", "unreliable"
                ],
                "experience_indicators": [
                    "duty manager", "night manager", "operations manager", "guest services",
                    "hospitality management", "emergency management", "shift supervisor"
                ],
                "education_preferences": [
                    "hospitality management", "hotel management", "business management",
                    "emergency management", "customer service"
                ],
                "certifications": [
                    "hospitality management", "emergency management", "guest relations",
                    "leadership certification", "crisis management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # FRONT OFFICE & GUEST SERVICES
            # ==========================================
            "Front Office Manager": {
                "must_have_skills": [
                    "front office operations", "team management", "guest relations", "PMS systems",
                    "reservations management", "staff supervision", "revenue optimization", "training",
                    "quality control", "performance management", "customer service", "communication"
                ],
                "nice_to_have_skills": [
                    "luxury service", "group bookings", "VIP services", "guest recovery",
                    "upselling strategies", "night audit", "forecasting", "inventory management"
                ],
                "technical_skills": [
                    "Opera PMS", "Maestro PMS", "reservation systems", "revenue management systems",
                    "reporting tools", "guest feedback systems", "channel management"
                ],
                "soft_skills": [
                    "leadership", "communication", "problem solving", "attention to detail",
                    "multitasking", "stress management", "team building", "customer focus"
                ],
                "cultural_fit_keywords": [
                    "guest-focused", "leader", "professional", "detail-oriented",
                    "team builder", "service excellence", "efficient", "welcoming"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of hospitality experience", "poor customer service",
                    "inability to handle stress", "poor communication"
                ],
                "experience_indicators": [
                    "front office manager", "guest services manager", "front desk supervisor",
                    "hotel front office", "reservations manager", "guest relations"
                ],
                "education_preferences": [
                    "hospitality management", "hotel administration", "tourism management",
                    "business management", "customer service"
                ],
                "certifications": [
                    "hospitality management", "PMS certification", "guest relations",
                    "revenue management", "customer service excellence"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Front Desk Supervisor": {
                "must_have_skills": [
                    "front desk operations", "team supervision", "guest services", "PMS systems",
                    "customer service", "problem solving", "communication", "staff training",
                    "quality control", "cash handling", "reservations", "multitasking"
                ],
                "nice_to_have_skills": [
                    "guest recovery", "upselling", "night audit", "VIP services",
                    "group check-ins", "conflict resolution", "performance coaching", "scheduling"
                ],
                "technical_skills": [
                    "Opera PMS", "reservation systems", "payment processing", "guest management systems",
                    "reporting tools", "telephone systems", "keycard systems"
                ],
                "soft_skills": [
                    "leadership", "communication", "patience", "problem solving",
                    "attention to detail", "multitasking", "team building", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "guest-focused", "team leader", "professional", "helpful",
                    "patient", "reliable", "welcoming", "supportive"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "poor customer service", "lack of patience",
                    "poor communication", "inability to multitask"
                ],
                "experience_indicators": [
                    "front desk supervisor", "guest services supervisor", "front desk agent",
                    "hotel reception", "guest services", "hospitality supervision"
                ],
                "education_preferences": [
                    "hospitality management", "hotel management", "customer service",
                    "business administration", "tourism"
                ],
                "certifications": [
                    "hospitality management", "PMS certification", "customer service",
                    "leadership development", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 55000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Front Desk Agent": {
                "must_have_skills": [
                    "customer service", "computer skills", "communication", "multitasking",
                    "phone etiquette", "PMS systems", "problem solving", "cash handling",
                    "attention to detail", "time management"
                ],
                "nice_to_have_skills": [
                    "Opera PMS", "Maestro PMS", "guest relations", "check-in procedures",
                    "check-out procedures", "reservations management", "multilingual",
                    "upselling", "conflict resolution", "night audit", "folio management"
                ],
                "technical_skills": [
                    "Opera PMS", "Maestro PMS", "RMS Cloud", "keycard systems",
                    "telephone systems", "credit card processing", "room blocking",
                    "group reservations", "walk-in management"
                ],
                "soft_skills": [
                    "interpersonal communication", "patience", "empathy", "adaptability",
                    "stress management", "cultural sensitivity", "active listening"
                ],
                "cultural_fit_keywords": [
                    "team player", "friendly", "professional", "positive attitude",
                    "guest-focused", "helpful", "detail-oriented", "reliable", "patient",
                    "welcoming", "enthusiastic"
                ],
                "disqualifying_factors": [
                    "poor communication", "unreliable", "inflexible", "antisocial",
                    "impatient with guests", "dishonest", "unprofessional appearance"
                ],
                "experience_indicators": [
                    "front desk", "reception", "guest services", "hotel reception",
                    "hospitality", "check-in", "check-out", "concierge", "customer service",
                    "reservations", "hotel operations"
                ],
                "education_preferences": [
                    "hospitality management", "tourism", "business administration",
                    "communications", "hotel management", "customer service"
                ],
                "certifications": [
                    "hospitality certification", "customer service certification",
                    "PMS certification", "hotel operations", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.25,
                    "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Receptionist": {
                "must_have_skills": [
                    "customer service", "communication", "phone skills", "computer skills",
                    "multitasking", "organization", "attention to detail", "professional demeanor",
                    "administrative skills", "time management", "problem solving"
                ],
                "nice_to_have_skills": [
                    "hospitality experience", "multilingual", "guest relations", "appointment scheduling",
                    "visitor management", "office administration", "data entry", "filing"
                ],
                "technical_skills": [
                    "phone systems", "computer software", "visitor management systems",
                    "appointment scheduling", "email management", "basic office equipment"
                ],
                "soft_skills": [
                    "communication", "friendliness", "professionalism", "patience",
                    "adaptability", "interpersonal skills", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "professional", "welcoming", "helpful",
                    "organized", "reliable", "positive", "courteous"
                ],
                "disqualifying_factors": [
                    "poor communication", "unprofessional demeanor", "unreliable",
                    "poor phone skills", "lack of computer skills"
                ],
                "experience_indicators": [
                    "receptionist", "front desk", "customer service", "administrative assistant",
                    "office support", "guest services", "visitor services"
                ],
                "education_preferences": [
                    "high school diploma", "customer service", "office administration",
                    "communications", "business studies"
                ],
                "certifications": [
                    "customer service", "office administration", "communication skills",
                    "hospitality basics", "professional development"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Night Auditor": {
                "must_have_skills": [
                    "night audit procedures", "accounting basics", "PMS systems", "customer service",
                    "attention to detail", "security awareness", "problem solving", "independence",
                    "cash handling", "report generation", "computer skills", "communication"
                ],
                "nice_to_have_skills": [
                    "accounting experience", "night operations", "emergency procedures",
                    "guest services", "security protocols", "inventory management", "data entry"
                ],
                "technical_skills": [
                    "Opera PMS", "audit software", "accounting systems", "reporting tools",
                    "payment processing", "security systems", "telephone systems"
                ],
                "soft_skills": [
                    "independence", "attention to detail", "reliability", "problem solving",
                    "calm under pressure", "self-motivation", "accuracy", "responsibility"
                ],
                "cultural_fit_keywords": [
                    "reliable", "independent", "detail-oriented", "responsible",
                    "calm", "professional", "accurate", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "unreliable", "lack of independence",
                    "poor computer skills", "inability to work alone"
                ],
                "experience_indicators": [
                    "night auditor", "night audit", "accounting", "night operations",
                    "front desk", "audit procedures", "night shift"
                ],
                "education_preferences": [
                    "accounting", "hospitality management", "business administration",
                    "hotel management", "bookkeeping"
                ],
                "certifications": [
                    "accounting basics", "hospitality operations", "night audit certification",
                    "bookkeeping", "PMS certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 32000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Guest Services Manager": {
                "must_have_skills": [
                    "leadership", "customer service", "team management", "problem solving",
                    "communication", "hospitality operations", "training", "performance management"
                ],
                "nice_to_have_skills": [
                    "guest recovery", "luxury service", "VIP handling", "complaint resolution",
                    "staff development", "quality assurance", "mystery shopper programs",
                    "service standards", "guest satisfaction metrics"
                ],
                "technical_skills": [
                    "hospitality software", "performance metrics", "guest feedback systems",
                    "training platforms", "scheduling software", "budget management"
                ],
                "soft_skills": [
                    "leadership", "emotional intelligence", "conflict resolution",
                    "mentoring", "decision making", "strategic thinking"
                ],
                "cultural_fit_keywords": [
                    "leader", "mentor", "service excellence", "guest advocate",
                    "team builder", "innovative", "results-oriented", "diplomatic"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "inability to handle stress",
                    "lack of hospitality experience", "poor communication"
                ],
                "experience_indicators": [
                    "guest services", "hospitality management", "team leadership",
                    "customer service management", "hotel operations", "front office management"
                ],
                "education_preferences": [
                    "hospitality management", "business management", "hotel administration",
                    "tourism management", "organizational leadership"
                ],
                "certifications": [
                    "hospitality management", "customer service excellence",
                    "leadership certification", "hotel operations management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.25, "cultural_fit": 0.20,
                    "hospitality": 0.20
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Concierge": {
                "must_have_skills": [
                    "customer service", "local knowledge", "communication", "organization", "problem solving",
                    "multilingual", "cultural awareness", "networking", "information management", "hospitality"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "event planning", "restaurant knowledge", "tour coordination",
                    "VIP services", "transportation arrangements", "entertainment booking", "personal shopping"
                ],
                "technical_skills": [
                    "reservation systems", "concierge software", "communication platforms",
                    "mapping tools", "booking platforms", "guest management systems"
                ],
                "soft_skills": [
                    "resourcefulness", "communication", "cultural sensitivity", "patience",
                    "attention to detail", "networking", "adaptability", "sophistication"
                ],
                "cultural_fit_keywords": [
                    "sophisticated", "knowledgeable", "helpful", "professional", "well-connected",
                    "resourceful", "cultured", "service-oriented", "diplomatic", "refined"
                ],
                "disqualifying_factors": [
                    "poor local knowledge", "lack of sophistication", "poor communication",
                    "inflexible", "lack of cultural awareness"
                ],
                "experience_indicators": [
                    "concierge", "guest services", "luxury hospitality", "customer service",
                    "tour guide", "travel services", "hospitality", "personal assistant"
                ],
                "education_preferences": [
                    "hospitality management", "tourism", "languages", "cultural studies",
                    "communications", "travel and tourism"
                ],
                "certifications": [
                    "concierge certification", "hospitality excellence", "cultural awareness",
                    "language certifications", "luxury service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Bellman": {
                "must_have_skills": [
                    "customer service", "physical fitness", "communication", "hospitality",
                    "luggage handling", "guest assistance", "local knowledge", "professional appearance",
                    "reliability", "time management", "courtesy", "safety awareness"
                ],
                "nice_to_have_skills": [
                    "multilingual", "concierge services", "transportation knowledge",
                    "guest relations", "tip etiquette", "luxury service", "door services"
                ],
                "technical_skills": [
                    "luggage equipment", "transportation systems", "communication devices",
                    "guest tracking systems", "safety equipment"
                ],
                "soft_skills": [
                    "friendliness", "professionalism", "helpfulness", "patience",
                    "physical stamina", "reliability", "courtesy", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "helpful", "courteous", "professional", "reliable",
                    "friendly", "service-oriented", "welcoming", "attentive"
                ],
                "disqualifying_factors": [
                    "poor physical condition", "unprofessional appearance", "poor customer service",
                    "unreliable", "lack of courtesy"
                ],
                "experience_indicators": [
                    "bellman", "bell captain", "porter", "guest services", "hospitality",
                    "customer service", "luggage services", "door services"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality basics", "customer service",
                    "communications", "physical education"
                ],
                "certifications": [
                    "hospitality service", "customer service", "safety training",
                    "guest relations", "physical fitness"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Porter": {
                "must_have_skills": [
                    "physical fitness", "customer service", "luggage handling", "communication",
                    "reliability", "safety awareness", "guest assistance", "professional appearance",
                    "time management", "courtesy", "hospitality", "teamwork"
                ],
                "nice_to_have_skills": [
                    "equipment operation", "maintenance awareness", "guest relations",
                    "multilingual", "transportation knowledge", "inventory management"
                ],
                "technical_skills": [
                    "luggage carts", "transportation equipment", "lifting equipment",
                    "safety equipment", "communication devices"
                ],
                "soft_skills": [
                    "physical stamina", "reliability", "helpfulness", "courtesy",
                    "teamwork", "positive attitude", "patience", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "hardworking", "reliable", "helpful", "courteous",
                    "team player", "professional", "strong", "dependable"
                ],
                "disqualifying_factors": [
                    "poor physical condition", "unreliable", "poor safety awareness",
                    "unprofessional", "lack of teamwork"
                ],
                "experience_indicators": [
                    "porter", "luggage porter", "baggage handler", "guest services",
                    "hospitality support", "customer service", "physical labor"
                ],
                "education_preferences": [
                    "high school diploma", "physical education", "hospitality basics",
                    "customer service", "safety training"
                ],
                "certifications": [
                    "safety training", "hospitality service", "customer service",
                    "physical fitness", "equipment operation"
                ],
                "scoring_weights": {
                    "experience": 0.15, "skills": 0.35, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 22000, "max": 30000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Bell Captain": {
                "must_have_skills": [
                    "team leadership", "customer service", "guest relations", "communication",
                    "staff supervision", "hospitality operations", "training", "quality control",
                    "luggage operations", "guest assistance", "problem solving", "scheduling"
                ],
                "nice_to_have_skills": [
                    "luxury service", "VIP handling", "concierge coordination",
                    "staff development", "performance management", "guest recovery", "multilingual"
                ],
                "technical_skills": [
                    "staff scheduling", "guest tracking systems", "communication systems",
                    "performance monitoring", "training platforms", "luggage systems"
                ],
                "soft_skills": [
                    "leadership", "communication", "teamwork", "problem solving",
                    "customer focus", "attention to detail", "reliability", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "leader", "professional", "guest-focused", "team builder",
                    "service excellence", "reliable", "courteous", "experienced"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of hospitality experience", "poor customer service",
                    "inability to supervise", "unprofessional"
                ],
                "experience_indicators": [
                    "bell captain", "bellman supervisor", "guest services supervisor",
                    "hospitality supervision", "team leadership", "bell services"
                ],
                "education_preferences": [
                    "hospitality management", "hotel management", "business administration",
                    "customer service", "leadership development"
                ],
                "certifications": [
                    "hospitality leadership", "guest services management", "team leadership",
                    "customer service excellence", "hospitality operations"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.25, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Rooms Division Manager": {
                "must_have_skills": [
                    "rooms division management", "team leadership", "operations management", "revenue optimization",
                    "quality control", "staff supervision", "guest relations", "performance management",
                    "budget management", "inventory control", "training coordination", "strategic planning"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "multi-property experience", "technology implementation",
                    "process improvement", "vendor management", "compliance management", "analytics"
                ],
                "technical_skills": [
                    "hotel management systems", "revenue management systems", "performance analytics",
                    "budgeting software", "scheduling systems", "quality management systems"
                ],
                "soft_skills": [
                    "strategic leadership", "analytical thinking", "communication", "decision making",
                    "change management", "team building", "problem solving", "innovation"
                ],
                "cultural_fit_keywords": [
                    "strategic", "leader", "analytical", "guest-focused",
                    "results-driven", "innovative", "collaborative", "excellence-oriented"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of strategic thinking", "poor analytical skills",
                    "inability to manage complexity", "poor communication"
                ],
                "experience_indicators": [
                    "rooms division manager", "hotel operations manager", "front office manager",
                    "housekeeping manager", "rooms operations", "hospitality management"
                ],
                "education_preferences": [
                    "hospitality management", "hotel administration", "business management",
                    "operations management", "MBA"
                ],
                "certifications": [
                    "hospitality management", "revenue management", "operations management",
                    "leadership certification", "hotel administration"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.25, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 70000, "max": 110000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Reservations Manager": {
                "must_have_skills": [
                    "reservations management", "revenue optimization", "team leadership", "forecasting",
                    "inventory management", "customer service", "sales coordination", "performance analysis",
                    "staff training", "quality control", "communication", "strategic planning"
                ],
                "nice_to_have_skills": [
                    "channel management", "group bookings", "corporate sales", "yield management",
                    "competitive analysis", "market segmentation", "pricing strategies", "analytics"
                ],
                "technical_skills": [
                    "reservations systems", "revenue management systems", "channel management platforms",
                    "analytics tools", "forecasting software", "performance dashboards"
                ],
                "soft_skills": [
                    "analytical thinking", "leadership", "strategic thinking", "communication",
                    "attention to detail", "problem solving", "customer focus", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "analytical", "strategic", "detail-oriented", "results-driven",
                    "guest-focused", "leader", "innovative", "performance-oriented"
                ],
                "disqualifying_factors": [
                    "poor analytical skills", "lack of attention to detail", "poor leadership",
                    "inability to work with data", "poor communication"
                ],
                "experience_indicators": [
                    "reservations manager", "revenue manager", "reservations supervisor",
                    "hotel reservations", "booking management", "yield management"
                ],
                "education_preferences": [
                    "hospitality management", "revenue management", "business administration",
                    "hotel management", "analytics", "marketing"
                ],
                "certifications": [
                    "revenue management", "reservations management", "hospitality analytics",
                    "yield management", "hotel operations"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Reservations Agent": {
                "must_have_skills": [
                    "reservations systems", "customer service", "communication", "attention to detail",
                    "sales skills", "computer skills", "phone etiquette", "problem solving",
                    "multitasking", "time management", "accuracy", "hospitality"
                ],
                "nice_to_have_skills": [
                    "upselling", "group bookings", "multilingual", "travel knowledge",
                    "guest relations", "conflict resolution", "inventory management", "reporting"
                ],
                "technical_skills": [
                    "reservations software", "booking platforms", "payment processing",
                    "phone systems", "communication tools", "reporting systems"
                ],
                "soft_skills": [
                    "communication", "patience", "persuasion", "attention to detail",
                    "adaptability", "customer focus", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "helpful", "detail-oriented", "professional",
                    "patient", "sales-oriented", "guest-focused", "reliable"
                ],
                "disqualifying_factors": [
                    "poor communication", "lack of attention to detail", "poor customer service",
                    "inability to use technology", "impatient"
                ],
                "experience_indicators": [
                    "reservations agent", "booking agent", "customer service", "call center",
                    "travel agent", "sales agent", "hospitality"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality management", "customer service",
                    "tourism", "communications", "business studies"
                ],
                "certifications": [
                    "reservations certification", "customer service", "hospitality basics",
                    "sales training", "communication skills"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 28000, "max": 40000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Call Centre Agent": {
                "must_have_skills": [
                    "phone skills", "customer service", "communication", "computer skills",
                    "problem solving", "multitasking", "attention to detail", "patience",
                    "conflict resolution", "time management", "accuracy", "typing skills"
                ],
                "nice_to_have_skills": [
                    "hospitality knowledge", "sales skills", "multilingual", "data entry",
                    "CRM systems", "call center experience", "technical support", "documentation"
                ],
                "technical_skills": [
                    "call center software", "CRM systems", "phone systems", "computer applications",
                    "data entry systems", "ticketing systems", "communication platforms"
                ],
                "soft_skills": [
                    "communication", "patience", "empathy", "active listening",
                    "stress management", "adaptability", "problem solving", "resilience"
                ],
                "cultural_fit_keywords": [
                    "patient", "helpful", "professional", "calm",
                    "good listener", "problem-solver", "reliable", "courteous"
                ],
                "disqualifying_factors": [
                    "poor phone skills", "impatient", "poor listening skills",
                    "inability to handle stress", "poor computer skills"
                ],
                "experience_indicators": [
                    "call center", "customer service", "phone support", "technical support",
                    "help desk", "telemarketing", "customer care"
                ],
                "education_preferences": [
                    "high school diploma", "customer service", "communications",
                    "business studies", "hospitality basics"
                ],
                "certifications": [
                    "customer service", "call center operations", "communication skills",
                    "computer skills", "conflict resolution"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Lobby Ambassador": {
                "must_have_skills": [
                    "customer service", "communication", "hospitality", "guest relations",
                    "professional appearance", "local knowledge", "problem solving", "multilingual",
                    "cultural awareness", "patience", "friendliness", "adaptability"
                ],
                "nice_to_have_skills": [
                    "concierge services", "luxury hospitality", "event coordination",
                    "guest assistance", "information management", "networking", "VIP services"
                ],
                "technical_skills": [
                    "guest management systems", "communication devices", "information systems",
                    "booking platforms", "mobile devices", "hospitality software"
                ],
                "soft_skills": [
                    "interpersonal skills", "cultural sensitivity", "warmth", "professionalism",
                    "approachability", "helpfulness", "patience", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "welcoming", "friendly", "professional", "helpful",
                    "cultured", "sophisticated", "warm", "approachable"
                ],
                "disqualifying_factors": [
                    "poor communication", "unprofessional appearance", "lack of cultural awareness",
                    "unfriendly demeanor", "inflexible"
                ],
                "experience_indicators": [
                    "lobby ambassador", "guest relations", "concierge", "customer service",
                    "hospitality", "guest services", "tourism"
                ],
                "education_preferences": [
                    "hospitality management", "tourism", "communications", "languages",
                    "cultural studies", "customer service"
                ],
                "certifications": [
                    "hospitality service", "guest relations", "cultural awareness",
                    "customer service", "language certifications"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.25, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # DIAMOND CLUB / VIP SERVICES
            # ==========================================
            "Diamond Club Manager": {
                "must_have_skills": [
                    "luxury service", "VIP management", "customer relations", "team leadership",
                    "exclusive services", "hospitality excellence", "communication", "problem solving",
                    "attention to detail", "cultural sensitivity", "discretion", "service standards"
                ],
                "nice_to_have_skills": [
                    "personal concierge", "luxury hospitality", "exclusive events", "high-end dining",
                    "premium amenities", "guest personalization", "luxury brands", "etiquette"
                ],
                "technical_skills": [
                    "guest management systems", "VIP tracking", "luxury service platforms",
                    "communication systems", "event management software", "guest preferences"
                ],
                "soft_skills": [
                    "sophistication", "discretion", "excellence", "leadership",
                    "attention to detail", "cultural awareness", "refinement", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "luxury-focused", "sophisticated", "discreet", "excellence-driven",
                    "refined", "exclusive", "high-standards", "professional"
                ],
                "disqualifying_factors": [
                    "lack of luxury experience", "poor attention to detail", "unprofessional",
                    "lack of discretion", "poor communication"
                ],
                "experience_indicators": [
                    "diamond club", "VIP services", "luxury hospitality", "concierge management",
                    "exclusive services", "high-end service", "luxury management"
                ],
                "education_preferences": [
                    "hospitality management", "luxury service", "hotel administration",
                    "business management", "customer experience"
                ],
                "certifications": [
                    "luxury service", "hospitality excellence", "VIP management",
                    "customer experience", "concierge certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.25, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 60000, "max": 90000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Head Butler": {
                "must_have_skills": [
                    "luxury service", "personal service", "attention to detail", "discretion",
                    "VIP handling", "team leadership", "communication", "anticipation",
                    "guest preferences", "high standards", "professionalism", "etiquette"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "wine service", "personal shopping", "event coordination",
                    "travel arrangements", "personal concierge", "luxury brands", "cultural knowledge"
                ],
                "technical_skills": [
                    "guest preference systems", "communication devices", "luxury service tools",
                    "scheduling systems", "personal service equipment"
                ],
                "soft_skills": [
                    "anticipation", "discretion", "sophistication", "attentiveness",
                    "cultural sensitivity", "professionalism", "reliability", "excellence"
                ],
                "cultural_fit_keywords": [
                    "sophisticated", "discreet", "attentive", "anticipatory",
                    "refined", "professional", "service-excellence", "detail-oriented"
                ],
                "disqualifying_factors": [
                    "lack of sophistication", "poor attention to detail", "indiscreet",
                    "unprofessional", "lack of service excellence"
                ],
                "experience_indicators": [
                    "head butler", "personal butler", "luxury service", "VIP service",
                    "personal concierge", "high-end service", "exclusive service"
                ],
                "education_preferences": [
                    "hospitality excellence", "luxury service", "hotel management",
                    "butler training", "customer experience"
                ],
                "certifications": [
                    "butler certification", "luxury service", "VIP service",
                    "hospitality excellence", "personal service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 70000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Butler Supervisor": {
                "must_have_skills": [
                    "luxury service", "team supervision", "service standards", "training",
                    "guest relations", "communication", "attention to detail", "leadership",
                    "quality control", "VIP service", "professional standards", "discretion"
                ],
                "nice_to_have_skills": [
                    "butler training", "luxury hospitality", "personal service", "event coordination",
                    "staff development", "service excellence", "guest preferences", "etiquette"
                ],
                "technical_skills": [
                    "staff scheduling", "service management", "guest tracking", "quality systems",
                    "communication platforms", "training systems"
                ],
                "soft_skills": [
                    "leadership", "attention to detail", "professionalism", "discretion",
                    "sophistication", "communication", "reliability", "excellence"
                ],
                "cultural_fit_keywords": [
                    "professional", "sophisticated", "detail-oriented", "leader",
                    "service-focused", "discreet", "refined", "quality-driven"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of attention to detail", "unprofessional",
                    "lack of service experience", "poor communication"
                ],
                "experience_indicators": [
                    "butler supervisor", "luxury service supervisor", "VIP services",
                    "personal service", "hospitality supervision", "service management"
                ],
                "education_preferences": [
                    "hospitality management", "luxury service", "butler training",
                    "service excellence", "team leadership"
                ],
                "certifications": [
                    "butler certification", "luxury service", "hospitality leadership",
                    "service excellence", "team management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.25, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Butler": {
                "must_have_skills": [
                    "personal service", "attention to detail", "discretion", "guest relations",
                    "communication", "professionalism", "VIP service", "anticipation",
                    "luxury standards", "etiquette", "cultural sensitivity", "reliability"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "wine knowledge", "personal shopping", "travel assistance",
                    "event coordination", "luxury brands", "cultural knowledge", "languages"
                ],
                "technical_skills": [
                    "guest preference systems", "communication devices", "service equipment",
                    "scheduling tools", "personal service technology"
                ],
                "soft_skills": [
                    "discretion", "attentiveness", "sophistication", "anticipation",
                    "professionalism", "cultural sensitivity", "reliability", "patience"
                ],
                "cultural_fit_keywords": [
                    "discreet", "attentive", "professional", "sophisticated",
                    "service-oriented", "refined", "anticipatory", "reliable"
                ],
                "disqualifying_factors": [
                    "lack of discretion", "poor attention to detail", "unprofessional",
                    "lack of sophistication", "poor service attitude"
                ],
                "experience_indicators": [
                    "butler", "personal service", "VIP service", "luxury service",
                    "personal concierge", "exclusive service", "high-end service"
                ],
                "education_preferences": [
                    "butler training", "hospitality service", "luxury service",
                    "customer service", "cultural studies"
                ],
                "certifications": [
                    "butler certification", "luxury service", "personal service",
                    "VIP service", "hospitality excellence"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # HOUSEKEEPING & LAUNDRY
            # ==========================================
            "Executive Housekeeper": {
                "must_have_skills": [
                    "housekeeping operations", "team management", "budget management", "quality control",
                    "staff supervision", "inventory management", "training", "safety protocols",
                    "performance management", "vendor relations", "strategic planning", "cost control"
                ],
                "nice_to_have_skills": [
                    "luxury housekeeping", "laundry operations", "facility management", "green practices",
                    "technology implementation", "process improvement", "compliance", "multi-property"
                ],
                "technical_skills": [
                    "housekeeping management systems", "inventory software", "scheduling platforms",
                    "budgeting tools", "performance analytics", "compliance tracking"
                ],
                "soft_skills": [
                    "strategic leadership", "organization", "attention to detail", "communication",
                    "analytical thinking", "problem solving", "team building", "innovation"
                ],
                "cultural_fit_keywords": [
                    "organized", "strategic", "detail-oriented", "efficient",
                    "quality-focused", "leader", "innovative", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of leadership", "poor attention to detail",
                    "inability to manage budgets", "poor communication"
                ],
                "experience_indicators": [
                    "executive housekeeper", "housekeeping manager", "facility management",
                    "housekeeping operations", "hotel housekeeping", "cleaning operations"
                ],
                "education_preferences": [
                    "hospitality management", "facility management", "business management",
                    "housekeeping management", "operations management"
                ],
                "certifications": [
                    "executive housekeeping", "facility management", "hospitality management",
                    "safety certification", "quality management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 55000, "max": 80000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Housekeeping Supervisor": {
                "must_have_skills": [
                    "housekeeping operations", "team supervision", "quality control",
                    "inventory management", "scheduling", "training", "safety protocols",
                    "staff coordination", "guest room standards", "time management", "communication"
                ],
                "nice_to_have_skills": [
                    "laundry operations", "deep cleaning", "maintenance coordination",
                    "budget management", "eco-friendly practices", "lost and found", "housekeeping software"
                ],
                "technical_skills": [
                    "housekeeping software", "scheduling systems", "inventory management",
                    "cleaning equipment", "laundry systems", "room management systems"
                ],
                "soft_skills": [
                    "leadership", "organization", "attention to detail", "time management",
                    "problem solving", "communication", "training abilities", "reliability"
                ],
                "cultural_fit_keywords": [
                    "organized", "detail-oriented", "efficient", "reliable", "leader",
                    "quality-focused", "thorough", "systematic", "accountable"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail",
                    "inability to manage teams", "poor time management", "unreliable"
                ],
                "experience_indicators": [
                    "housekeeping supervisor", "housekeeping management", "hotel housekeeping",
                    "room attendant supervisor", "laundry operations", "cleaning supervision"
                ],
                "education_preferences": [
                    "hospitality management", "hotel operations", "business management",
                    "facility management", "housekeeping management"
                ],
                "certifications": [
                    "housekeeping certification", "hospitality operations",
                    "safety certification", "management training", "quality control"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Room Attendant": {
                "must_have_skills": [
                    "cleaning", "attention to detail", "time management", "physical stamina", "organization",
                    "guest room standards", "housekeeping procedures", "safety awareness", "efficiency",
                    "inventory awareness", "quality standards", "reliability"
                ],
                "nice_to_have_skills": [
                    "hotel cleaning", "laundry", "inventory", "guest interaction",
                    "deep cleaning", "eco-friendly practices", "maintenance reporting", "luxury standards"
                ],
                "technical_skills": [
                    "cleaning equipment", "housekeeping supplies", "laundry equipment",
                    "room maintenance tools", "safety equipment", "cleaning chemicals"
                ],
                "soft_skills": [
                    "thoroughness", "reliability", "efficiency", "professionalism",
                    "physical stamina", "attention to detail", "time management", "independence"
                ],
                "cultural_fit_keywords": [
                    "thorough", "reliable", "efficient", "professional", "hardworking",
                    "detail-oriented", "independent", "quality-focused", "conscientious"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of physical stamina", "unreliable",
                    "poor time management", "unprofessional"
                ],
                "experience_indicators": [
                    "room attendant", "housekeeper", "hotel cleaning", "housekeeping",
                    "cleaning services", "room cleaning", "hospitality cleaning"
                ],
                "education_preferences": [
                    "high school diploma", "housekeeping training", "hospitality basics",
                    "cleaning services", "customer service"
                ],
                "certifications": [
                    "housekeeping certification", "safety training", "cleaning certification",
                    "hospitality service", "chemical safety"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 22000, "max": 32000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Public Area Attendant": {
                "must_have_skills": [
                    "cleaning", "public space maintenance", "attention to detail", "time management",
                    "safety awareness", "customer interaction", "efficiency", "reliability",
                    "organization", "quality standards", "physical stamina", "professionalism"
                ],
                "nice_to_have_skills": [
                    "floor care", "carpet cleaning", "window cleaning", "maintenance awareness",
                    "guest interaction", "equipment operation", "eco-friendly practices", "inventory"
                ],
                "technical_skills": [
                    "cleaning equipment", "floor care machines", "carpet cleaners",
                    "window cleaning tools", "safety equipment", "maintenance tools"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "efficiency", "customer awareness",
                    "physical stamina", "independence", "professionalism", "thoroughness"
                ],
                "cultural_fit_keywords": [
                    "thorough", "reliable", "efficient", "professional",
                    "detail-oriented", "hardworking", "guest-aware", "quality-focused"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of physical stamina", "unreliable",
                    "unprofessional appearance", "poor customer interaction"
                ],
                "experience_indicators": [
                    "public area attendant", "cleaning", "facility cleaning", "maintenance cleaning",
                    "commercial cleaning", "hospitality cleaning", "janitorial"
                ],
                "education_preferences": [
                    "high school diploma", "cleaning services", "facility maintenance",
                    "customer service", "safety training"
                ],
                "certifications": [
                    "cleaning certification", "safety training", "equipment operation",
                    "chemical safety", "hospitality service"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Turndown Attendant": {
                "must_have_skills": [
                    "guest service", "attention to detail", "discretion", "time management",
                    "luxury service", "guest room preparation", "quality standards", "professionalism",
                    "cultural sensitivity", "reliability", "efficiency", "hospitality"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "guest preferences", "evening service", "amenity placement",
                    "room ambiance", "personalized service", "VIP service", "cultural awareness"
                ],
                "technical_skills": [
                    "room preparation", "amenity placement", "lighting control",
                    "guest preference systems", "luxury service tools"
                ],
                "soft_skills": [
                    "discretion", "attention to detail", "cultural sensitivity", "professionalism",
                    "reliability", "efficiency", "guest awareness", "sophistication"
                ],
                "cultural_fit_keywords": [
                    "discreet", "detail-oriented", "professional", "sophisticated",
                    "guest-focused", "reliable", "quality-oriented", "refined"
                ],
                "disqualifying_factors": [
                    "lack of discretion", "poor attention to detail", "unprofessional",
                    "lack of cultural sensitivity", "unreliable"
                ],
                "experience_indicators": [
                    "turndown service", "evening service", "luxury hospitality", "guest services",
                    "room service", "hospitality service", "hotel service"
                ],
                "education_preferences": [
                    "hospitality service", "luxury hospitality", "customer service",
                    "hotel service", "guest relations"
                ],
                "certifications": [
                    "luxury service", "hospitality service", "guest relations",
                    "customer service", "cultural awareness"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Houseman": {
                "must_have_skills": [
                    "physical labor", "cleaning support", "maintenance awareness", "reliability",
                    "teamwork", "safety awareness", "equipment operation", "time management",
                    "attention to detail", "following instructions", "housekeeping support", "efficiency"
                ],
                "nice_to_have_skills": [
                    "basic maintenance", "equipment maintenance", "inventory support",
                    "guest interaction", "heavy lifting", "grounds maintenance", "janitorial"
                ],
                "technical_skills": [
                    "cleaning equipment", "maintenance tools", "safety equipment",
                    "heavy equipment", "support machinery", "housekeeping supplies"
                ],
                "soft_skills": [
                    "physical stamina", "reliability", "teamwork", "following directions",
                    "attention to detail", "safety consciousness", "hardworking", "dependability"
                ],
                "cultural_fit_keywords": [
                    "hardworking", "reliable", "team player", "dependable",
                    "strong", "efficient", "supportive", "conscientious"
                ],
                "disqualifying_factors": [
                    "poor physical condition", "unreliable", "lack of teamwork",
                    "poor safety awareness", "inability to follow instructions"
                ],
                "experience_indicators": [
                    "houseman", "cleaning support", "facility support", "maintenance support",
                    "janitorial", "physical labor", "hospitality support"
                ],
                "education_preferences": [
                    "high school diploma", "physical labor", "facility maintenance",
                    "safety training", "equipment operation"
                ],
                "certifications": [
                    "safety training", "equipment operation", "physical fitness",
                    "cleaning certification", "workplace safety"
                ],
                "scoring_weights": {
                    "experience": 0.15, "skills": 0.35, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Linen Room Supervisor": {
                "must_have_skills": [
                    "linen management", "inventory control", "team supervision", "quality control",
                    "organization", "scheduling", "staff training", "efficiency optimization",
                    "cost control", "vendor relations", "performance management", "communication"
                ],
                "nice_to_have_skills": [
                    "laundry operations", "textile knowledge", "budget management", "process improvement",
                    "technology implementation", "compliance", "waste reduction", "sustainability"
                ],
                "technical_skills": [
                    "inventory management systems", "linen tracking", "scheduling software",
                    "quality control systems", "cost analysis tools", "performance metrics"
                ],
                "soft_skills": [
                    "organization", "leadership", "attention to detail", "analytical thinking",
                    "problem solving", "communication", "time management", "efficiency"
                ],
                "cultural_fit_keywords": [
                    "organized", "efficient", "detail-oriented", "systematic",
                    "quality-focused", "leader", "analytical", "cost-conscious"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail", "poor leadership",
                    "inability to manage inventory", "poor communication"
                ],
                "experience_indicators": [
                    "linen room supervisor", "inventory supervisor", "laundry supervisor",
                    "textile management", "housekeeping supervision", "linen management"
                ],
                "education_preferences": [
                    "inventory management", "hospitality operations", "business management",
                    "textile management", "operations management"
                ],
                "certifications": [
                    "inventory management", "laundry operations", "hospitality operations",
                    "quality control", "supervisory training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Laundry Manager": {
                "must_have_skills": [
                    "laundry operations", "team management", "equipment management", "quality control",
                    "cost control", "scheduling", "staff supervision", "performance management",
                    "vendor relations", "compliance", "efficiency optimization", "budget management"
                ],
                "nice_to_have_skills": [
                    "textile knowledge", "chemical management", "equipment maintenance", "process improvement",
                    "environmental compliance", "energy efficiency", "technology implementation", "training"
                ],
                "technical_skills": [
                    "laundry equipment", "chemical management systems", "quality control systems",
                    "scheduling software", "cost analysis tools", "maintenance management"
                ],
                "soft_skills": [
                    "leadership", "organization", "analytical thinking", "problem solving",
                    "communication", "attention to detail", "efficiency", "innovation"
                ],
                "cultural_fit_keywords": [
                    "efficient", "organized", "analytical", "quality-focused",
                    "leader", "cost-conscious", "innovative", "systematic"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of technical knowledge", "poor organizational skills",
                    "inability to manage costs", "poor safety awareness"
                ],
                "experience_indicators": [
                    "laundry manager", "laundry operations", "textile management", "commercial laundry",
                    "laundry supervisor", "facility operations", "housekeeping operations"
                ],
                "education_preferences": [
                    "operations management", "facility management", "textile management",
                    "business management", "hospitality operations"
                ],
                "certifications": [
                    "laundry operations", "facility management", "chemical safety",
                    "equipment operation", "environmental compliance"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Laundry Supervisor": {
                "must_have_skills": [
                    "laundry operations", "team supervision", "equipment operation", "quality control",
                    "scheduling", "staff training", "safety protocols", "efficiency",
                    "cost awareness", "problem solving", "communication", "time management"
                ],
                "nice_to_have_skills": [
                    "textile knowledge", "chemical handling", "equipment maintenance", "inventory management",
                    "process improvement", "training development", "performance monitoring", "compliance"
                ],
                "technical_skills": [
                    "laundry equipment", "washing machines", "dryers", "pressing equipment",
                    "chemical dispensing", "quality control tools", "scheduling systems"
                ],
                "soft_skills": [
                    "leadership", "organization", "attention to detail", "problem solving",
                    "communication", "reliability", "efficiency", "safety consciousness"
                ],
                "cultural_fit_keywords": [
                    "organized", "efficient", "detail-oriented", "reliable",
                    "quality-focused", "leader", "safety-conscious", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail", "poor leadership",
                    "unsafe practices", "poor communication"
                ],
                "experience_indicators": [
                    "laundry supervisor", "laundry operations", "commercial laundry", "textile processing",
                    "laundry attendant", "housekeeping laundry", "industrial laundry"
                ],
                "education_preferences": [
                    "laundry operations", "textile processing", "hospitality operations",
                    "supervisory training", "safety management"
                ],
                "certifications": [
                    "laundry operations", "chemical safety", "equipment operation",
                    "safety certification", "supervisory training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 32000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Laundry Attendant": {
                "must_have_skills": [
                    "laundry operations", "equipment operation", "attention to detail", "efficiency",
                    "safety awareness", "quality control", "time management", "following procedures",
                    "chemical handling", "reliability", "physical stamina", "organization"
                ],
                "nice_to_have_skills": [
                    "textile knowledge", "stain removal", "pressing", "folding", "sorting",
                    "equipment maintenance", "inventory awareness", "customer service"
                ],
                "technical_skills": [
                    "washing machines", "dryers", "pressing equipment", "folding equipment",
                    "chemical dispensing", "quality control", "laundry supplies"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "efficiency", "physical stamina",
                    "following directions", "safety consciousness", "thoroughness", "consistency"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "efficient", "hardworking",
                    "thorough", "consistent", "quality-focused", "safety-conscious"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "unreliable", "unsafe practices",
                    "lack of physical stamina", "inability to follow procedures"
                ],
                "experience_indicators": [
                    "laundry attendant", "laundry operator", "commercial laundry", "dry cleaning",
                    "textile processing", "laundry services", "industrial laundry"
                ],
                "education_preferences": [
                    "high school diploma", "laundry training", "equipment operation",
                    "safety training", "chemical handling"
                ],
                "certifications": [
                    "laundry operations", "equipment operation", "chemical safety",
                    "safety training", "quality control"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 20000, "max": 30000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Laundry Technician": {
                "must_have_skills": [
                    "equipment maintenance", "technical troubleshooting", "laundry operations", "mechanical skills",
                    "electrical basics", "preventive maintenance", "repair skills", "safety protocols",
                    "problem solving", "attention to detail", "reliability", "communication"
                ],
                "nice_to_have_skills": [
                    "equipment installation", "HVAC basics", "plumbing basics", "chemical systems",
                    "computerized controls", "energy efficiency", "predictive maintenance", "vendor coordination"
                ],
                "technical_skills": [
                    "laundry equipment", "mechanical systems", "electrical systems", "control systems",
                    "diagnostic tools", "maintenance tools", "repair equipment"
                ],
                "soft_skills": [
                    "problem solving", "analytical thinking", "attention to detail", "reliability",
                    "communication", "learning agility", "safety consciousness", "independence"
                ],
                "cultural_fit_keywords": [
                    "technical", "analytical", "problem-solver", "reliable",
                    "detail-oriented", "safety-conscious", "efficient", "skilled"
                ],
                "disqualifying_factors": [
                    "poor technical skills", "unsafe practices", "unreliable",
                    "poor problem-solving", "lack of attention to detail"
                ],
                "experience_indicators": [
                    "laundry technician", "equipment technician", "maintenance technician", "laundry maintenance",
                    "equipment repair", "mechanical repair", "commercial laundry"
                ],
                "education_preferences": [
                    "technical training", "mechanical engineering", "electrical training",
                    "equipment maintenance", "facility maintenance"
                ],
                "certifications": [
                    "equipment maintenance", "electrical certification", "mechanical certification",
                    "safety certification", "HVAC basics"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Uniform Room Attendant": {
                "must_have_skills": [
                    "inventory management", "organization", "attention to detail", "customer service",
                    "time management", "efficiency", "record keeping", "communication",
                    "quality control", "distribution management", "reliability", "professionalism"
                ],
                "nice_to_have_skills": [
                    "sizing knowledge", "alteration basics", "textile knowledge", "computer skills",
                    "staff interaction", "problem solving", "inventory software", "scheduling"
                ],
                "technical_skills": [
                    "inventory systems", "distribution tracking", "computer applications",
                    "sizing tools", "record keeping systems", "communication tools"
                ],
                "soft_skills": [
                    "organization", "attention to detail", "customer service", "communication",
                    "reliability", "efficiency", "professionalism", "helpfulness"
                ],
                "cultural_fit_keywords": [
                    "organized", "detail-oriented", "helpful", "efficient",
                    "reliable", "professional", "service-oriented", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail", "poor customer service",
                    "unreliable", "poor communication"
                ],
                "experience_indicators": [
                    "uniform room", "inventory management", "distribution", "retail experience",
                    "customer service", "hospitality support", "clothing distribution"
                ],
                "education_preferences": [
                    "high school diploma", "inventory management", "customer service",
                    "retail experience", "hospitality service"
                ],
                "certifications": [
                    "inventory management", "customer service", "hospitality service",
                    "retail operations", "communication skills"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 22000, "max": 32000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            # ==========================================
            # FOOD & BEVERAGE DEPARTMENT
            # ==========================================
            "Director of Food & Beverage": {
                "must_have_skills": [
                    "F&B operations", "strategic leadership", "revenue management", "team management",
                    "cost control", "menu development", "vendor management", "quality standards",
                    "performance management", "budget planning", "staff development", "guest satisfaction"
                ],
                "nice_to_have_skills": [
                    "multi-unit operations", "franchise operations", "luxury dining", "beverage programs",
                    "event catering", "wine programs", "sustainability", "technology implementation"
                ],
                "technical_skills": [
                    "POS systems", "inventory management", "cost analysis", "revenue analytics",
                    "performance dashboards", "scheduling software", "compliance tracking"
                ],
                "soft_skills": [
                    "strategic leadership", "analytical thinking", "communication", "innovation",
                    "team building", "decision making", "customer focus", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "strategic", "innovative", "results-driven", "guest-focused",
                    "leader", "analytical", "quality-oriented", "collaborative"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of F&B experience", "poor financial management",
                    "inability to handle complexity", "poor communication"
                ],
                "experience_indicators": [
                    "F&B director", "restaurant operations", "food service management", "hospitality management",
                    "multi-unit management", "food and beverage", "culinary management"
                ],
                "education_preferences": [
                    "hospitality management", "culinary management", "business administration",
                    "food service management", "restaurant management", "MBA"
                ],
                "certifications": [
                    "ServSafe", "hospitality management", "food service management",
                    "revenue management", "wine certification"
                ],
                "scoring_weights": {
                    "experience": 0.45, "skills": 0.25, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 8,
                "preferred_experience_years": 12,
                "salary_range": {"min": 90000, "max": 140000},
                "growth_potential": "Executive",
                "training_requirements": "Strategic"
            },

            "Restaurant Manager": {
                "must_have_skills": [
                    "restaurant operations", "team management", "customer service", "cost control",
                    "staff supervision", "quality control", "performance management", "scheduling",
                    "inventory management", "guest relations", "problem solving", "communication"
                ],
                "nice_to_have_skills": [
                    "fine dining", "wine knowledge", "event management", "catering", "marketing",
                    "revenue optimization", "staff training", "menu development", "POS systems"
                ],
                "technical_skills": [
                    "POS systems", "restaurant management software", "inventory systems",
                    "scheduling platforms", "performance analytics", "cost analysis tools"
                ],
                "soft_skills": [
                    "leadership", "customer focus", "problem solving", "communication",
                    "multitasking", "stress management", "team building", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "guest-focused", "leader", "service-oriented", "quality-driven",
                    "team builder", "efficient", "professional", "hospitality-minded"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of restaurant experience", "poor customer service",
                    "inability to handle stress", "poor communication"
                ],
                "experience_indicators": [
                    "restaurant manager", "food service manager", "dining manager", "hospitality management",
                    "restaurant operations", "food and beverage", "restaurant supervision"
                ],
                "education_preferences": [
                    "hospitality management", "restaurant management", "culinary management",
                    "business administration", "food service management"
                ],
                "certifications": [
                    "ServSafe", "restaurant management", "food service", "wine certification",
                    "hospitality management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 70000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Restaurant Supervisor": {
                "must_have_skills": [
                    "restaurant operations", "team supervision", "customer service", "quality control",
                    "staff training", "communication", "problem solving", "multitasking",
                    "performance monitoring", "guest relations", "conflict resolution", "efficiency"
                ],
                "nice_to_have_skills": [
                    "POS systems", "inventory awareness", "cash handling", "scheduling support",
                    "menu knowledge", "wine basics", "event support", "staff development"
                ],
                "technical_skills": [
                    "POS systems", "restaurant software", "communication tools",
                    "scheduling systems", "performance tracking", "guest management"
                ],
                "soft_skills": [
                    "leadership", "communication", "customer focus", "problem solving",
                    "adaptability", "team building", "patience", "multitasking"
                ],
                "cultural_fit_keywords": [
                    "team leader", "guest-focused", "supportive", "professional",
                    "service-oriented", "reliable", "efficient", "helpful"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "poor customer service", "inability to multitask",
                    "poor communication", "lack of restaurant experience"
                ],
                "experience_indicators": [
                    "restaurant supervisor", "dining supervisor", "food service supervisor",
                    "restaurant lead", "hospitality supervision", "food and beverage"
                ],
                "education_preferences": [
                    "hospitality management", "restaurant management", "food service",
                    "customer service", "business studies"
                ],
                "certifications": [
                    "ServSafe", "restaurant operations", "food service", "customer service",
                    "leadership development"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Captain": {
                "must_have_skills": [
                    "fine dining service", "guest relations", "communication", "team coordination",
                    "menu knowledge", "wine service", "attention to detail", "hospitality",
                    "professional presentation", "problem solving", "multitasking", "leadership"
                ],
                "nice_to_have_skills": [
                    "sommelier knowledge", "luxury service", "event service", "language skills",
                    "cultural awareness", "special dietary knowledge", "POS systems", "training"
                ],
                "technical_skills": [
                    "POS systems", "wine service tools", "dining service equipment",
                    "communication systems", "reservation systems", "payment processing"
                ],
                "soft_skills": [
                    "sophistication", "communication", "leadership", "attention to detail",
                    "customer focus", "cultural sensitivity", "professionalism", "patience"
                ],
                "cultural_fit_keywords": [
                    "sophisticated", "professional", "service-excellence", "refined",
                    "knowledgeable", "leader", "guest-focused", "attentive"
                ],
                "disqualifying_factors": [
                    "lack of fine dining experience", "poor communication", "unprofessional",
                    "lack of sophistication", "poor attention to detail"
                ],
                "experience_indicators": [
                    "captain", "fine dining", "restaurant captain", "dining service",
                    "luxury service", "hospitality service", "wine service"
                ],
                "education_preferences": [
                    "hospitality management", "culinary arts", "wine studies",
                    "fine dining service", "restaurant management"
                ],
                "certifications": [
                    "wine certification", "fine dining service", "sommelier",
                    "hospitality excellence", "luxury service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Waiter": {
                "must_have_skills": [
                    "customer service", "communication", "multitasking", "attention to detail",
                    "team work", "menu knowledge", "cash handling", "problem solving",
                    "time management", "professional appearance", "hospitality", "efficiency"
                ],
                "nice_to_have_skills": [
                    "fine dining experience", "wine knowledge", "multilingual", "POS systems",
                    "upselling", "special dietary knowledge", "event service", "cultural awareness"
                ],
                "technical_skills": [
                    "POS systems", "payment processing", "order management",
                    "communication devices", "service equipment", "cash handling"
                ],
                "soft_skills": [
                    "communication", "friendliness", "patience", "adaptability",
                    "team work", "customer focus", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "professional", "attentive", "team player",
                    "guest-focused", "reliable", "energetic", "service-oriented"
                ],
                "disqualifying_factors": [
                    "poor customer service", "inability to multitask", "poor communication",
                    "unreliable", "unprofessional appearance"
                ],
                "experience_indicators": [
                    "waiter", "server", "restaurant service", "food service", "dining service",
                    "hospitality service", "customer service", "restaurant experience"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "food service",
                    "customer service", "restaurant training"
                ],
                "certifications": [
                    "food service", "customer service", "responsible beverage service",
                    "hospitality basics", "POS certification"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 20000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Server": {
                "must_have_skills": [
                    "customer service", "food knowledge", "multitasking", "communication", "cash handling",
                    "attention to detail", "team work", "time management", "hospitality", "efficiency",
                    "problem solving", "professional appearance"
                ],
                "nice_to_have_skills": [
                    "fine dining", "wine service", "allergen knowledge", "upselling", "POS systems",
                    "multilingual", "event service", "special dietary needs", "beverage knowledge"
                ],
                "technical_skills": [
                    "POS systems", "payment processing", "order management", "communication tools",
                    "service equipment", "cash register", "mobile ordering"
                ],
                "soft_skills": [
                    "friendliness", "attentiveness", "professionalism", "team player", "energetic",
                    "patience", "adaptability", "customer focus", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "friendly", "attentive", "professional", "team player", "energetic",
                    "guest-focused", "reliable", "service-oriented", "welcoming"
                ],
                "disqualifying_factors": [
                    "poor customer service", "inability to multitask", "poor communication",
                    "unreliable", "unprofessional", "lack of energy"
                ],
                "experience_indicators": [
                    "server", "food server", "restaurant server", "dining service", "food service",
                    "hospitality service", "customer service", "waitstaff"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "food service training",
                    "customer service", "restaurant experience"
                ],
                "certifications": [
                    "food service", "responsible beverage service", "customer service",
                    "allergen awareness", "hospitality basics"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 18000, "max": 32000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Host/Hostess": {
                "must_have_skills": [
                    "customer service", "communication", "organization", "multitasking",
                    "professional appearance", "hospitality", "phone skills", "problem solving",
                    "attention to detail", "time management", "guest relations", "teamwork"
                ],
                "nice_to_have_skills": [
                    "reservation systems", "multilingual", "event coordination", "guest recognition",
                    "wait management", "conflict resolution", "cultural awareness", "upselling"
                ],
                "technical_skills": [
                    "reservation systems", "POS systems", "phone systems", "seating management",
                    "communication tools", "scheduling software", "guest management"
                ],
                "soft_skills": [
                    "friendliness", "professionalism", "patience", "organization",
                    "communication", "adaptability", "positive attitude", "welcoming nature"
                ],
                "cultural_fit_keywords": [
                    "welcoming", "friendly", "professional", "organized",
                    "guest-focused", "positive", "reliable", "courteous"
                ],
                "disqualifying_factors": [
                    "poor communication", "unprofessional appearance", "lack of organization",
                    "unfriendly demeanor", "poor customer service"
                ],
                "experience_indicators": [
                    "host", "hostess", "restaurant host", "guest seating", "front of house",
                    "customer service", "hospitality", "restaurant experience"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "customer service",
                    "communication skills", "restaurant training"
                ],
                "certifications": [
                    "customer service", "hospitality basics", "communication skills",
                    "guest relations", "restaurant operations"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 18000, "max": 28000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Banquet Manager": {
                "must_have_skills": [
                    "event management", "team leadership", "banquet operations", "customer service",
                    "coordination", "communication", "problem solving", "time management",
                    "staff supervision", "quality control", "logistics", "budget awareness"
                ],
                "nice_to_have_skills": [
                    "large scale events", "wedding coordination", "corporate events", "menu planning",
                    "vendor coordination", "audio visual", "decoration", "protocol knowledge"
                ],
                "technical_skills": [
                    "event management software", "banquet management systems", "audio visual equipment",
                    "communication tools", "scheduling systems", "logistics software"
                ],
                "soft_skills": [
                    "leadership", "organization", "problem solving", "communication",
                    "multitasking", "attention to detail", "stress management", "flexibility"
                ],
                "cultural_fit_keywords": [
                    "organized", "leader", "detail-oriented", "flexible",
                    "guest-focused", "professional", "efficient", "collaborative"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "inability to handle stress", "poor leadership",
                    "lack of event experience", "poor communication"
                ],
                "experience_indicators": [
                    "banquet manager", "event manager", "catering manager", "banquet operations",
                    "event coordination", "hospitality events", "function management"
                ],
                "education_preferences": [
                    "hospitality management", "event management", "hotel management",
                    "business administration", "catering management"
                ],
                "certifications": [
                    "event management", "banquet operations", "hospitality management",
                    "catering certification", "food service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Banquet Captain": {
                "must_have_skills": [
                    "banquet service", "team coordination", "event execution", "guest service",
                    "communication", "leadership", "attention to detail", "time management",
                    "staff coordination", "quality control", "problem solving", "professionalism"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "large events", "wedding service", "corporate events",
                    "wine service", "protocol knowledge", "multilingual", "training abilities"
                ],
                "technical_skills": [
                    "banquet equipment", "audio visual basics", "communication devices",
                    "service tools", "event setup equipment", "catering equipment"
                ],
                "soft_skills": [
                    "leadership", "organization", "communication", "adaptability",
                    "team coordination", "attention to detail", "stress management", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "leader", "organized", "professional", "detail-oriented",
                    "team player", "guest-focused", "reliable", "efficient"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of event experience", "poor communication",
                    "inability to handle stress", "unprofessional"
                ],
                "experience_indicators": [
                    "banquet captain", "event captain", "banquet service", "event service",
                    "catering service", "function service", "hospitality events"
                ],
                "education_preferences": [
                    "hospitality management", "event management", "food service",
                    "banquet operations", "customer service"
                ],
                "certifications": [
                    "banquet service", "event service", "food service", "hospitality operations",
                    "team leadership"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 32000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Banquet Server": {
                "must_have_skills": [
                    "banquet service", "customer service", "team work", "communication",
                    "attention to detail", "time management", "physical stamina", "professionalism",
                    "event service", "multitasking", "efficiency", "following procedures"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "wine service", "large event experience", "setup/breakdown",
                    "special dietary awareness", "multilingual", "formal service", "protocol"
                ],
                "technical_skills": [
                    "banquet equipment", "service tools", "audio visual basics",
                    "communication devices", "catering equipment", "setup equipment"
                ],
                "soft_skills": [
                    "teamwork", "adaptability", "communication", "attention to detail",
                    "physical stamina", "professionalism", "reliability", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "team player", "professional", "reliable", "efficient",
                    "guest-focused", "adaptable", "hardworking", "detail-oriented"
                ],
                "disqualifying_factors": [
                    "poor teamwork", "lack of physical stamina", "unprofessional",
                    "poor attention to detail", "unreliable"
                ],
                "experience_indicators": [
                    "banquet server", "event server", "catering server", "function server",
                    "banquet service", "event service", "hospitality service"
                ],
                "education_preferences": [
                    "high school diploma", "food service training", "banquet service",
                    "hospitality basics", "customer service"
                ],
                "certifications": [
                    "food service", "banquet service", "customer service",
                    "responsible beverage service", "hospitality basics"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 18000, "max": 28000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Bar Manager": {
                "must_have_skills": [
                    "bar operations", "team management", "beverage knowledge", "inventory control",
                    "cost control", "staff supervision", "customer service", "communication",
                    "performance management", "scheduling", "quality control", "problem solving"
                ],
                "nice_to_have_skills": [
                    "craft cocktails", "wine program", "beer knowledge", "mixology", "bar design",
                    "event bars", "training development", "vendor relations", "marketing"
                ],
                "technical_skills": [
                    "POS systems", "inventory management", "bar equipment", "cost analysis",
                    "scheduling software", "performance analytics", "beverage systems"
                ],
                "soft_skills": [
                    "leadership", "creativity", "communication", "analytical thinking",
                    "customer focus", "team building", "problem solving", "innovation"
                ],
                "cultural_fit_keywords": [
                    "creative", "leader", "knowledgeable", "innovative",
                    "guest-focused", "quality-driven", "team builder", "professional"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of beverage knowledge", "poor cost control",
                    "inability to manage staff", "poor communication"
                ],
                "experience_indicators": [
                    "bar manager", "beverage manager", "bar operations", "bartending",
                    "cocktail program", "beverage operations", "bar supervision"
                ],
                "education_preferences": [
                    "hospitality management", "beverage management", "culinary arts",
                    "business administration", "bar management"
                ],
                "certifications": [
                    "responsible beverage service", "sommelier", "bartending certification",
                    "bar management", "mixology"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Bar Supervisor": {
                "must_have_skills": [
                    "bar operations", "team supervision", "beverage knowledge", "customer service",
                    "inventory awareness", "staff training", "communication", "quality control",
                    "multitasking", "problem solving", "cash handling", "efficiency"
                ],
                "nice_to_have_skills": [
                    "mixology", "wine knowledge", "craft cocktails", "POS systems",
                    "cost awareness", "event support", "training abilities", "scheduling"
                ],
                "technical_skills": [
                    "POS systems", "bar equipment", "inventory systems", "communication tools",
                    "scheduling systems", "beverage equipment", "cash handling"
                ],
                "soft_skills": [
                    "leadership", "communication", "customer focus", "problem solving",
                    "multitasking", "team building", "adaptability", "reliability"
                ],
                "cultural_fit_keywords": [
                    "team leader", "knowledgeable", "professional", "guest-focused",
                    "supportive", "efficient", "reliable", "quality-oriented"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "lack of beverage knowledge", "poor customer service",
                    "inability to multitask", "poor communication"
                ],
                "experience_indicators": [
                    "bar supervisor", "lead bartender", "bar operations", "beverage service",
                    "bar management", "bartending", "cocktail service"
                ],
                "education_preferences": [
                    "hospitality management", "beverage studies", "bartending training",
                    "customer service", "bar operations"
                ],
                "certifications": [
                    "responsible beverage service", "bartending certification", "mixology",
                    "customer service", "bar operations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Bartender": {
                "must_have_skills": [
                    "mixology", "customer service", "cash handling", "multitasking", "product knowledge",
                    "beverage preparation", "communication", "attention to detail", "efficiency",
                    "team work", "hospitality", "problem solving"
                ],
                "nice_to_have_skills": [
                    "craft cocktails", "wine knowledge", "beer knowledge", "inventory", "POS systems",
                    "flair bartending", "event service", "multilingual", "upselling"
                ],
                "technical_skills": [
                    "bar equipment", "POS systems", "beverage systems", "cash register",
                    "cocktail tools", "beer systems", "wine service tools"
                ],
                "soft_skills": [
                    "personable", "energetic", "professional", "friendly", "entertaining",
                    "adaptability", "creativity", "patience", "multitasking"
                ],
                "cultural_fit_keywords": [
                    "personable", "energetic", "professional", "friendly", "entertaining",
                    "guest-focused", "creative", "reliable", "knowledgeable"
                ],
                "disqualifying_factors": [
                    "poor customer service", "lack of beverage knowledge", "inability to multitask",
                    "poor cash handling", "unprofessional", "slow service"
                ],
                "experience_indicators": [
                    "bartender", "mixologist", "bar service", "beverage service", "cocktail service",
                    "bar operations", "customer service", "hospitality"
                ],
                "education_preferences": [
                    "bartending school", "hospitality training", "beverage studies",
                    "customer service", "mixology training"
                ],
                "certifications": [
                    "responsible beverage service", "bartending certification", "mixology",
                    "customer service", "food safety"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 20000, "max": 40000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Cocktail Waitress": {
                "must_have_skills": [
                    "customer service", "beverage service", "communication", "multitasking",
                    "cash handling", "attention to detail", "hospitality", "professional appearance",
                    "team work", "efficiency", "problem solving", "time management"
                ],
                "nice_to_have_skills": [
                    "beverage knowledge", "upselling", "multilingual", "event service",
                    "POS systems", "cocktail knowledge", "wine basics", "guest relations"
                ],
                "technical_skills": [
                    "POS systems", "beverage service equipment", "cash handling",
                    "communication devices", "service trays", "payment processing"
                ],
                "soft_skills": [
                    "friendliness", "professionalism", "adaptability", "energy",
                    "customer focus", "communication", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "energetic", "professional", "attentive",
                    "guest-focused", "reliable", "personable", "service-oriented"
                ],
                "disqualifying_factors": [
                    "poor customer service", "unprofessional appearance", "inability to multitask",
                    "poor communication", "unreliable", "lack of energy"
                ],
                "experience_indicators": [
                    "cocktail waitress", "beverage server", "bar server", "lounge server",
                    "drink service", "hospitality service", "customer service"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "customer service",
                    "beverage service", "restaurant experience"
                ],
                "certifications": [
                    "responsible beverage service", "customer service", "hospitality basics",
                    "food safety", "beverage service"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 18000, "max": 30000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Mini-Bar Attendant": {
                "must_have_skills": [
                    "inventory management", "attention to detail", "organization", "reliability",
                    "customer service", "time management", "efficiency", "record keeping",
                    "product knowledge", "communication", "professional appearance", "discretion"
                ],
                "nice_to_have_skills": [
                    "guest interaction", "beverage knowledge", "upselling", "multilingual",
                    "computer skills", "problem solving", "cultural awareness", "hotel operations"
                ],
                "technical_skills": [
                    "inventory systems", "POS systems", "computer applications",
                    "communication devices", "record keeping systems", "mobile devices"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "organization", "efficiency",
                    "discretion", "professionalism", "customer awareness", "independence"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "organized", "efficient",
                    "discreet", "professional", "thorough", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "unreliable", "lack of organization",
                    "poor customer service", "dishonest"
                ],
                "experience_indicators": [
                    "mini-bar attendant", "inventory management", "guest services", "hotel operations",
                    "customer service", "hospitality", "retail experience"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "customer service",
                    "inventory management", "retail experience"
                ],
                "certifications": [
                    "hospitality service", "customer service", "inventory management",
                    "beverage service", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Sommelier": {
                "must_have_skills": [
                    "wine knowledge", "wine service", "customer education", "communication", "attention to detail",
                    "tasting skills", "pairing knowledge", "professional presentation", "hospitality",
                    "cultural knowledge", "sales skills", "problem solving"
                ],
                "nice_to_have_skills": [
                    "wine certification", "multiple languages", "spirits knowledge", "sake knowledge",
                    "cheese pairing", "food knowledge", "wine storage", "cellar management"
                ],
                "technical_skills": [
                    "wine service tools", "cellar management", "inventory systems",
                    "tasting equipment", "storage systems", "POS systems"
                ],
                "soft_skills": [
                    "sophistication", "communication", "knowledge sharing", "cultural sensitivity",
                    "patience", "professionalism", "passion", "continuous learning"
                ],
                "cultural_fit_keywords": [
                    "knowledgeable", "sophisticated", "passionate", "educational",
                    "refined", "professional", "cultured", "expert"
                ],
                "disqualifying_factors": [
                    "lack of wine knowledge", "poor communication", "lack of sophistication",
                    "poor customer service", "inability to educate"
                ],
                "experience_indicators": [
                    "sommelier", "wine service", "wine education", "fine dining", "wine sales",
                    "beverage management", "wine consulting", "cellar management"
                ],
                "education_preferences": [
                    "wine studies", "sommelier certification", "hospitality management",
                    "culinary arts", "beverage management"
                ],
                "certifications": [
                    "sommelier certification", "wine certification", "beverage education",
                    "court of master sommeliers", "wine and spirit education"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 45000, "max": 80000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # CULINARY DEPARTMENT
            # ==========================================
            "Executive Chef": {
                "must_have_skills": [
                    "culinary leadership", "menu development", "kitchen management", "food safety",
                    "cost control", "staff management", "quality control", "food procurement",
                    "culinary innovation", "performance management", "budget management", "training"
                ],
                "nice_to_have_skills": [
                    "international cuisine", "dietary restrictions", "sustainable practices", "wine pairing",
                    "banquet cooking", "specialty diets", "culinary trends", "vendor relations"
                ],
                "technical_skills": [
                    "kitchen equipment", "food safety systems", "inventory management", "cost analysis",
                    "recipe development", "nutrition analysis", "kitchen technology", "scheduling"
                ],
                "soft_skills": [
                    "leadership", "creativity", "innovation", "communication", "stress management",
                    "team building", "problem solving", "attention to detail", "time management"
                ],
                "cultural_fit_keywords": [
                    "innovative", "leader", "creative", "quality-driven", "perfectionist",
                    "passionate", "professional", "mentor", "culinary excellence"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of culinary training", "poor food safety knowledge",
                    "inability to handle stress", "poor cost management"
                ],
                "experience_indicators": [
                    "executive chef", "head chef", "culinary director", "kitchen management",
                    "chef de cuisine", "culinary leadership", "restaurant chef"
                ],
                "education_preferences": [
                    "culinary arts", "culinary management", "hospitality management",
                    "food service management", "nutrition", "business administration"
                ],
                "certifications": [
                    "culinary certification", "food safety", "ServSafe", "nutritional analysis",
                    "culinary management", "kitchen management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 8,
                "preferred_experience_years": 12,
                "salary_range": {"min": 80000, "max": 120000},
                "growth_potential": "Executive",
                "training_requirements": "Advanced"
            },

            "Sous Chef": {
                "must_have_skills": [
                    "culinary skills", "kitchen operations", "food safety", "staff supervision",
                    "menu execution", "quality control", "cost awareness", "training abilities",
                    "communication", "time management", "problem solving", "leadership"
                ],
                "nice_to_have_skills": [
                    "menu development", "international cuisine", "dietary restrictions", "banquet cooking",
                    "inventory management", "cost control", "scheduling", "vendor knowledge"
                ],
                "technical_skills": [
                    "kitchen equipment", "food safety systems", "recipe execution", "inventory systems",
                    "kitchen technology", "cost analysis", "scheduling software"
                ],
                "soft_skills": [
                    "leadership", "creativity", "communication", "stress management", "adaptability",
                    "team work", "attention to detail", "problem solving", "multitasking"
                ],
                "cultural_fit_keywords": [
                    "creative", "leader", "quality-focused", "passionate", "professional",
                    "supportive", "dedicated", "culinary excellence", "team player"
                ],
                "disqualifying_factors": [
                    "poor culinary skills", "lack of leadership", "poor food safety",
                    "inability to handle stress", "poor communication"
                ],
                "experience_indicators": [
                    "sous chef", "kitchen supervisor", "line cook supervisor", "culinary supervisor",
                    "chef de partie", "kitchen management", "culinary operations"
                ],
                "education_preferences": [
                    "culinary arts", "culinary management", "food service", "hospitality management",
                    "nutrition", "culinary training"
                ],
                "certifications": [
                    "culinary certification", "food safety", "ServSafe", "kitchen management",
                    "culinary arts", "nutritional awareness"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 4,
                "preferred_experience_years": 7,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Chef de Partie": {
                "must_have_skills": [
                    "culinary skills", "station management", "food safety", "quality control",
                    "recipe execution", "kitchen operations", "time management", "teamwork",
                    "communication", "attention to detail", "consistency", "efficiency"
                ],
                "nice_to_have_skills": [
                    "specialty cooking", "international cuisine", "garde manger", "pastry basics",
                    "sauce making", "grilling", "knife skills", "presentation skills"
                ],
                "technical_skills": [
                    "kitchen equipment", "cooking techniques", "food safety protocols",
                    "knife skills", "cooking methods", "recipe following", "presentation"
                ],
                "soft_skills": [
                    "attention to detail", "consistency", "teamwork", "communication",
                    "stress management", "adaptability", "learning ability", "precision"
                ],
                "cultural_fit_keywords": [
                    "precise", "consistent", "dedicated", "team player", "quality-focused",
                    "passionate", "professional", "detail-oriented", "reliable"
                ],
                "disqualifying_factors": [
                    "poor culinary skills", "lack of consistency", "poor food safety",
                    "inability to work in team", "poor attention to detail"
                ],
                "experience_indicators": [
                    "chef de partie", "line cook", "station chef", "kitchen cook", "culinary specialist",
                    "cook", "kitchen operations", "food preparation"
                ],
                "education_preferences": [
                    "culinary arts", "culinary training", "food service", "cooking school",
                    "hospitality training", "culinary certificate"
                ],
                "certifications": [
                    "culinary certification", "food safety", "ServSafe", "cooking certification",
                    "culinary arts", "kitchen operations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Line Cook": {
                "must_have_skills": [
                    "cooking skills", "food safety", "recipe following", "kitchen operations",
                    "time management", "teamwork", "communication", "attention to detail",
                    "consistency", "efficiency", "basic knife skills", "quality awareness"
                ],
                "nice_to_have_skills": [
                    "specialty cooking", "grill skills", "sauté skills", "food presentation",
                    "multitasking", "speed", "adaptability", "kitchen equipment knowledge"
                ],
                "technical_skills": [
                    "cooking equipment", "basic kitchen tools", "food safety protocols",
                    "cooking techniques", "recipe execution", "food handling", "cleaning"
                ],
                "soft_skills": [
                    "teamwork", "communication", "reliability", "attention to detail",
                    "stress management", "adaptability", "learning willingness", "consistency"
                ],
                "cultural_fit_keywords": [
                    "reliable", "team player", "dedicated", "consistent", "hardworking",
                    "passionate", "learning-oriented", "quality-focused", "efficient"
                ],
                "disqualifying_factors": [
                    "poor cooking skills", "food safety violations", "inability to follow recipes",
                    "poor teamwork", "inconsistent performance"
                ],
                "experience_indicators": [
                    "line cook", "prep cook", "kitchen cook", "cook", "food preparation",
                    "kitchen operations", "restaurant cook", "culinary assistant"
                ],
                "education_preferences": [
                    "culinary training", "food service training", "cooking school",
                    "high school diploma", "culinary certificate", "kitchen experience"
                ],
                "certifications": [
                    "food safety", "ServSafe", "cooking certification", "culinary basics",
                    "kitchen operations", "food handling"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 40000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Prep Cook": {
                "must_have_skills": [
                    "food preparation", "knife skills", "food safety", "organization",
                    "efficiency", "recipe following", "attention to detail", "teamwork",
                    "time management", "consistency", "basic cooking", "cleanliness"
                ],
                "nice_to_have_skills": [
                    "vegetable preparation", "protein preparation", "sauce preparation", "inventory awareness",
                    "kitchen operations", "food storage", "portion control", "speed"
                ],
                "technical_skills": [
                    "knife skills", "food preparation equipment", "food safety protocols",
                    "storage systems", "preparation techniques", "measuring tools", "cleaning"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "teamwork", "consistency",
                    "time management", "organization", "learning willingness", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "efficient", "team player", "organized",
                    "consistent", "hardworking", "dedicated", "quality-focused"
                ],
                "disqualifying_factors": [
                    "poor knife skills", "food safety violations", "lack of organization",
                    "inconsistent work", "poor attention to detail"
                ],
                "experience_indicators": [
                    "prep cook", "food preparation", "kitchen prep", "culinary prep",
                    "food prep", "kitchen assistant", "prep assistant", "kitchen helper"
                ],
                "education_preferences": [
                    "culinary training", "food service training", "high school diploma",
                    "cooking basics", "food safety training", "kitchen experience"
                ],
                "certifications": [
                    "food safety", "ServSafe", "food handling", "knife skills",
                    "culinary basics", "kitchen safety"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.40, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 22000, "max": 32000},
                "growth_potential": "High",
                "training_requirements": "Basic"
            },

            "Pastry Chef": {
                "must_have_skills": [
                    "pastry skills", "baking techniques", "dessert creation", "food safety",
                    "recipe development", "attention to detail", "creativity", "presentation skills",
                    "time management", "quality control", "consistency", "precision"
                ],
                "nice_to_have_skills": [
                    "chocolate work", "sugar art", "cake decorating", "bread making", "gluten-free baking",
                    "international pastries", "special dietary desserts", "cost control", "menu development"
                ],
                "technical_skills": [
                    "pastry equipment", "baking ovens", "mixing equipment", "decorating tools",
                    "temperature control", "recipe scaling", "nutritional analysis", "food safety"
                ],
                "soft_skills": [
                    "creativity", "precision", "attention to detail", "artistic ability",
                    "patience", "innovation", "communication", "time management", "consistency"
                ],
                "cultural_fit_keywords": [
                    "creative", "artistic", "precise", "innovative", "passionate",
                    "detail-oriented", "quality-focused", "perfectionist", "dedicated"
                ],
                "disqualifying_factors": [
                    "poor pastry skills", "lack of creativity", "inconsistent results",
                    "poor attention to detail", "food safety violations"
                ],
                "experience_indicators": [
                    "pastry chef", "baker", "dessert chef", "pastry cook", "baking specialist",
                    "confectioner", "cake decorator", "pastry artist"
                ],
                "education_preferences": [
                    "pastry arts", "baking and pastry", "culinary arts", "confectionery arts",
                    "pastry certification", "culinary management"
                ],
                "certifications": [
                    "pastry certification", "baking certification", "food safety", "ServSafe",
                    "pastry arts", "confectionery certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 40000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Baker": {
                "must_have_skills": [
                    "baking skills", "bread making", "recipe following", "food safety",
                    "consistency", "time management", "attention to detail", "quality control",
                    "oven operation", "dough handling", "proofing", "temperature control"
                ],
                "nice_to_have_skills": [
                    "artisan breads", "pastry basics", "cake baking", "specialty breads",
                    "gluten-free baking", "cost awareness", "inventory", "equipment maintenance"
                ],
                "technical_skills": [
                    "baking ovens", "mixing equipment", "proofing equipment", "measuring tools",
                    "temperature monitoring", "timer management", "baking techniques", "food safety"
                ],
                "soft_skills": [
                    "consistency", "attention to detail", "patience", "reliability",
                    "time management", "precision", "quality focus", "learning ability"
                ],
                "cultural_fit_keywords": [
                    "consistent", "reliable", "detail-oriented", "quality-focused", "patient",
                    "dedicated", "precise", "hardworking", "traditional"
                ],
                "disqualifying_factors": [
                    "poor baking skills", "inconsistent results", "food safety violations",
                    "poor time management", "lack of attention to detail"
                ],
                "experience_indicators": [
                    "baker", "bread baker", "production baker", "baking specialist",
                    "bakery worker", "baking assistant", "bread production"
                ],
                "education_preferences": [
                    "baking certification", "culinary training", "pastry arts", "food service",
                    "baking school", "culinary arts"
                ],
                "certifications": [
                    "baking certification", "food safety", "ServSafe", "bread baking",
                    "pastry basics", "culinary fundamentals"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 25000, "max": 40000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Kitchen Steward": {
                "must_have_skills": [
                    "cleaning", "dishwashing", "kitchen sanitation", "equipment cleaning",
                    "food safety", "organization", "efficiency", "teamwork",
                    "reliability", "physical stamina", "attention to detail", "time management"
                ],
                "nice_to_have_skills": [
                    "equipment maintenance", "inventory support", "kitchen operations awareness",
                    "chemical safety", "waste management", "recycling", "deep cleaning"
                ],
                "technical_skills": [
                    "dishwashing equipment", "cleaning chemicals", "sanitizing systems",
                    "kitchen equipment", "cleaning tools", "safety protocols", "waste systems"
                ],
                "soft_skills": [
                    "reliability", "hardworking", "teamwork", "attention to detail",
                    "physical endurance", "organization", "efficiency", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "reliable", "hardworking", "team player", "efficient", "organized",
                    "dedicated", "thorough", "supportive", "dependable"
                ],
                "disqualifying_factors": [
                    "poor work ethic", "unreliable", "food safety violations",
                    "inability to handle physical demands", "poor teamwork"
                ],
                "experience_indicators": [
                    "kitchen steward", "dishwasher", "kitchen cleaner", "sanitation worker",
                    "kitchen assistant", "utility worker", "kitchen support"
                ],
                "education_preferences": [
                    "high school diploma", "food safety training", "kitchen experience",
                    "sanitation training", "workplace safety"
                ],
                "certifications": [
                    "food safety", "sanitation certification", "workplace safety",
                    "chemical safety", "kitchen operations"
                ],
                "scoring_weights": {
                    "experience": 0.15, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            # ==========================================
            # ENGINEERING & MAINTENANCE DEPARTMENT
            # ==========================================
            "Chief Engineer": {
                "must_have_skills": [
                    "facility management", "engineering leadership", "maintenance management", "HVAC systems",
                    "electrical systems", "plumbing systems", "team management", "safety protocols",
                    "budget management", "preventive maintenance", "emergency response", "project management"
                ],
                "nice_to_have_skills": [
                    "energy management", "sustainability", "building automation", "fire safety systems",
                    "pool maintenance", "elevator systems", "vendor management", "cost control"
                ],
                "technical_skills": [
                    "HVAC systems", "electrical systems", "plumbing", "building automation", "maintenance software",
                    "energy management", "safety systems", "mechanical systems", "preventive maintenance"
                ],
                "soft_skills": [
                    "leadership", "problem solving", "analytical thinking", "communication",
                    "project management", "team building", "decision making", "stress management"
                ],
                "cultural_fit_keywords": [
                    "leader", "technical", "reliable", "problem-solver", "safety-focused",
                    "efficient", "analytical", "experienced", "responsible"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of technical knowledge", "safety violations",
                    "poor communication", "inability to manage emergencies"
                ],
                "experience_indicators": [
                    "chief engineer", "facility manager", "maintenance manager", "engineering manager",
                    "building engineer", "property engineer", "technical manager"
                ],
                "education_preferences": [
                    "engineering", "facility management", "mechanical engineering", "electrical engineering",
                    "building systems", "HVAC certification", "property management"
                ],
                "certifications": [
                    "engineering license", "HVAC certification", "electrical certification",
                    "facility management", "safety certification", "energy management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 8,
                "preferred_experience_years": 12,
                "salary_range": {"min": 75000, "max": 110000},
                "growth_potential": "Moderate",
                "training_requirements": "Advanced"
            },

            "Maintenance Manager": {
                "must_have_skills": [
                    "maintenance management", "team supervision", "preventive maintenance", "HVAC basics",
                    "electrical basics", "plumbing basics", "safety protocols", "scheduling",
                    "work order management", "vendor coordination", "budget awareness", "emergency response"
                ],
                "nice_to_have_skills": [
                    "building systems", "energy efficiency", "project management", "cost control",
                    "maintenance software", "contractor management", "inventory management", "training"
                ],
                "technical_skills": [
                    "maintenance systems", "HVAC basics", "electrical basics", "plumbing", "hand tools",
                    "power tools", "maintenance software", "safety equipment", "diagnostic tools"
                ],
                "soft_skills": [
                    "leadership", "organization", "communication", "problem solving",
                    "time management", "team building", "reliability", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "organized", "leader", "reliable", "efficient", "problem-solver",
                    "safety-conscious", "team player", "responsible", "experienced"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of maintenance experience", "safety violations",
                    "poor organization", "inability to handle emergencies"
                ],
                "experience_indicators": [
                    "maintenance manager", "maintenance supervisor", "facility maintenance",
                    "building maintenance", "property maintenance", "technical supervisor"
                ],
                "education_preferences": [
                    "facility management", "maintenance management", "technical training",
                    "building systems", "mechanical training", "electrical training"
                ],
                "certifications": [
                    "facility management", "maintenance certification", "HVAC basics",
                    "electrical basics", "safety certification", "leadership training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Maintenance Technician": {
                "must_have_skills": [
                    "general maintenance", "basic electrical", "basic plumbing", "HVAC basics",
                    "hand tools", "power tools", "safety protocols", "troubleshooting",
                    "repair skills", "preventive maintenance", "communication", "reliability"
                ],
                "nice_to_have_skills": [
                    "appliance repair", "carpentry", "painting", "tile work", "equipment maintenance",
                    "pool maintenance", "landscaping", "welding", "locksmith skills"
                ],
                "technical_skills": [
                    "hand tools", "power tools", "electrical tools", "plumbing tools", "HVAC tools",
                    "diagnostic equipment", "maintenance equipment", "safety equipment", "measuring tools"
                ],
                "soft_skills": [
                    "problem solving", "reliability", "attention to detail", "communication",
                    "learning ability", "adaptability", "teamwork", "initiative", "patience"
                ],
                "cultural_fit_keywords": [
                    "reliable", "handy", "problem-solver", "detail-oriented", "hardworking",
                    "versatile", "safety-conscious", "team player", "dependable"
                ],
                "disqualifying_factors": [
                    "lack of technical skills", "safety violations", "unreliable",
                    "poor communication", "inability to learn"
                ],
                "experience_indicators": [
                    "maintenance technician", "maintenance worker", "handyman", "building maintenance",
                    "facility maintenance", "property maintenance", "general maintenance"
                ],
                "education_preferences": [
                    "technical training", "trade school", "maintenance certification",
                    "electrical training", "plumbing training", "HVAC training"
                ],
                "certifications": [
                    "maintenance certification", "electrical basics", "plumbing basics",
                    "HVAC basics", "safety certification", "tool certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "HVAC Technician": {
                "must_have_skills": [
                    "HVAC systems", "heating systems", "cooling systems", "ventilation systems",
                    "HVAC repair", "HVAC maintenance", "electrical basics", "troubleshooting",
                    "safety protocols", "system diagnostics", "preventive maintenance", "refrigeration"
                ],
                "nice_to_have_skills": [
                    "building automation", "energy efficiency", "commercial HVAC", "industrial HVAC",
                    "controls systems", "ductwork", "heat pumps", "boilers", "chillers"
                ],
                "technical_skills": [
                    "HVAC tools", "diagnostic equipment", "electrical meters", "refrigeration tools",
                    "pressure gauges", "leak detectors", "calibration tools", "safety equipment"
                ],
                "soft_skills": [
                    "problem solving", "attention to detail", "analytical thinking", "reliability",
                    "learning ability", "communication", "patience", "precision", "safety awareness"
                ],
                "cultural_fit_keywords": [
                    "technical", "precise", "reliable", "problem-solver", "detail-oriented",
                    "safety-conscious", "experienced", "knowledgeable", "professional"
                ],
                "disqualifying_factors": [
                    "lack of HVAC knowledge", "safety violations", "poor troubleshooting",
                    "unreliable", "inability to learn new systems"
                ],
                "experience_indicators": [
                    "HVAC technician", "heating technician", "cooling technician", "air conditioning technician",
                    "refrigeration technician", "HVAC service", "climate control"
                ],
                "education_preferences": [
                    "HVAC certification", "technical training", "trade school", "mechanical training",
                    "refrigeration training", "electrical training", "building systems"
                ],
                "certifications": [
                    "HVAC certification", "EPA certification", "refrigeration license",
                    "electrical certification", "safety certification", "manufacturer certifications"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Electrician": {
                "must_have_skills": [
                    "electrical systems", "electrical repair", "electrical installation", "wiring",
                    "electrical troubleshooting", "electrical safety", "electrical codes", "power systems",
                    "lighting systems", "electrical maintenance", "circuit analysis", "motor control"
                ],
                "nice_to_have_skills": [
                    "industrial electrical", "commercial electrical", "building automation", "fire alarm systems",
                    "security systems", "emergency power", "generators", "transformers", "control panels"
                ],
                "technical_skills": [
                    "electrical tools", "multimeters", "wire strippers", "conduit benders",
                    "voltage testers", "oscilloscopes", "power tools", "safety equipment", "diagnostic tools"
                ],
                "soft_skills": [
                    "problem solving", "attention to detail", "safety awareness", "precision",
                    "analytical thinking", "reliability", "communication", "learning ability", "patience"
                ],
                "cultural_fit_keywords": [
                    "technical", "precise", "safety-conscious", "reliable", "problem-solver",
                    "detail-oriented", "experienced", "professional", "knowledgeable"
                ],
                "disqualifying_factors": [
                    "lack of electrical knowledge", "safety violations", "poor troubleshooting",
                    "code violations", "unreliable work"
                ],
                "experience_indicators": [
                    "electrician", "electrical technician", "electrical maintenance", "electrical service",
                    "electrical repair", "electrical installation", "power systems"
                ],
                "education_preferences": [
                    "electrical training", "trade school", "electrical certification", "technical training",
                    "electrical apprenticeship", "electrical engineering technology"
                ],
                "certifications": [
                    "electrical license", "electrical certification", "safety certification",
                    "electrical codes", "OSHA certification", "manufacturer certifications"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Plumber": {
                "must_have_skills": [
                    "plumbing systems", "pipe installation", "pipe repair", "water systems",
                    "drainage systems", "plumbing troubleshooting", "plumbing tools", "safety protocols",
                    "leak detection", "pipe fitting", "water pressure", "plumbing codes"
                ],
                "nice_to_have_skills": [
                    "commercial plumbing", "industrial plumbing", "backflow prevention", "water heaters",
                    "sewage systems", "pump systems", "gas lines", "hydro-jetting", "pipe cleaning"
                ],
                "technical_skills": [
                    "plumbing tools", "pipe cutters", "pipe threaders", "drain snakes",
                    "pressure testers", "leak detectors", "welding equipment", "power tools", "hand tools"
                ],
                "soft_skills": [
                    "problem solving", "attention to detail", "physical stamina", "reliability",
                    "communication", "patience", "precision", "learning ability", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "reliable", "skilled", "problem-solver", "detail-oriented", "hardworking",
                    "experienced", "professional", "dependable", "thorough"
                ],
                "disqualifying_factors": [
                    "lack of plumbing knowledge", "poor workmanship", "safety violations",
                    "unreliable", "inability to diagnose problems"
                ],
                "experience_indicators": [
                    "plumber", "plumbing technician", "pipefitter", "plumbing maintenance",
                    "plumbing service", "plumbing repair", "water systems"
                ],
                "education_preferences": [
                    "plumbing training", "trade school", "plumbing certification", "technical training",
                    "plumbing apprenticeship", "pipefitting training"
                ],
                "certifications": [
                    "plumbing license", "plumbing certification", "backflow certification",
                    "safety certification", "welding certification", "gas line certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Pool Technician": {
                "must_have_skills": [
                    "pool maintenance", "water chemistry", "pool equipment", "pool cleaning",
                    "chemical handling", "pool safety", "equipment maintenance", "troubleshooting",
                    "pump systems", "filtration systems", "water testing", "preventive maintenance"
                ],
                "nice_to_have_skills": [
                    "spa maintenance", "automated systems", "pool repairs", "tile cleaning",
                    "equipment installation", "water features", "deck maintenance", "customer service"
                ],
                "technical_skills": [
                    "pool equipment", "water testing kits", "chemical dispensers", "cleaning equipment",
                    "pump systems", "filter systems", "heater systems", "automation systems", "hand tools"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "safety awareness", "communication",
                    "problem solving", "physical stamina", "organization", "customer focus", "patience"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "safety-conscious", "thorough", "responsible",
                    "guest-focused", "professional", "knowledgeable", "dependable"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "safety violations", "unreliable",
                    "lack of chemical knowledge", "poor guest interaction"
                ],
                "experience_indicators": [
                    "pool technician", "pool maintenance", "pool service", "aquatic maintenance",
                    "water treatment", "pool operator", "pool cleaner"
                ],
                "education_preferences": [
                    "pool operator certification", "water treatment training", "chemical safety training",
                    "aquatic facility training", "equipment training"
                ],
                "certifications": [
                    "pool operator license", "chemical safety", "water treatment certification",
                    "aquatic facility certification", "safety certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Groundskeeper": {
                "must_have_skills": [
                    "landscaping", "lawn maintenance", "plant care", "irrigation systems",
                    "equipment operation", "grounds maintenance", "pest control", "seasonal care",
                    "outdoor equipment", "safety protocols", "physical stamina", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "horticulture", "tree care", "fertilization", "disease control", "design basics",
                    "equipment maintenance", "irrigation repair", "pesticide application", "customer service"
                ],
                "technical_skills": [
                    "lawn equipment", "irrigation systems", "hand tools", "power tools",
                    "pesticide equipment", "fertilizer spreaders", "mowers", "trimmers", "blowers"
                ],
                "soft_skills": [
                    "attention to detail", "physical stamina", "reliability", "independence",
                    "time management", "pride in work", "adaptability", "safety awareness", "patience"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "hardworking", "independent", "thorough",
                    "nature-loving", "dedicated", "physical", "responsible"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of physical stamina", "unreliable",
                    "safety violations", "inability to work outdoors"
                ],
                "experience_indicators": [
                    "groundskeeper", "landscaper", "lawn care", "grounds maintenance",
                    "landscape maintenance", "turf care", "outdoor maintenance"
                ],
                "education_preferences": [
                    "landscaping training", "horticulture training", "turf management",
                    "pesticide certification", "equipment operation training"
                ],
                "certifications": [
                    "pesticide license", "landscaping certification", "equipment certification",
                    "safety certification", "horticulture certification"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 38000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            # ==========================================
            # SECURITY DEPARTMENT
            # ==========================================
            "Security Manager": {
                "must_have_skills": [
                    "security management", "team leadership", "emergency response", "safety protocols",
                    "security systems", "incident management", "risk assessment", "communication",
                    "training development", "report writing", "conflict resolution", "law enforcement"
                ],
                "nice_to_have_skills": [
                    "surveillance systems", "access control", "investigation skills", "crowd control",
                    "emergency medical", "fire safety", "loss prevention", "vendor management"
                ],
                "technical_skills": [
                    "security systems", "surveillance equipment", "access control systems", "alarm systems",
                    "communication equipment", "computer systems", "incident reporting", "emergency equipment"
                ],
                "soft_skills": [
                    "leadership", "communication", "problem solving", "decision making",
                    "stress management", "attention to detail", "reliability", "integrity"
                ],
                "cultural_fit_keywords": [
                    "leader", "reliable", "responsible", "professional", "trustworthy",
                    "vigilant", "calm", "experienced", "safety-focused"
                ],
                "disqualifying_factors": [
                    "criminal background", "poor leadership", "lack of security experience",
                    "poor communication", "inability to handle emergencies"
                ],
                "experience_indicators": [
                    "security manager", "security supervisor", "loss prevention manager",
                    "safety manager", "security operations", "law enforcement", "military"
                ],
                "education_preferences": [
                    "criminal justice", "security management", "law enforcement", "military",
                    "emergency management", "business administration", "public safety"
                ],
                "certifications": [
                    "security certification", "CPR/First Aid", "emergency response",
                    "security management", "loss prevention", "safety certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Security Supervisor": {
                "must_have_skills": [
                    "security operations", "team supervision", "emergency response", "safety protocols",
                    "incident response", "communication", "report writing", "patrol duties",
                    "conflict resolution", "training abilities", "attention to detail", "reliability"
                ],
                "nice_to_have_skills": [
                    "surveillance monitoring", "access control", "crowd control", "investigation basics",
                    "emergency medical", "loss prevention", "customer service", "technology use"
                ],
                "technical_skills": [
                    "security equipment", "surveillance systems", "communication devices", "alarm systems",
                    "access control", "computer basics", "reporting software", "emergency equipment"
                ],
                "soft_skills": [
                    "leadership", "communication", "problem solving", "reliability",
                    "attention to detail", "stress management", "integrity", "alertness"
                ],
                "cultural_fit_keywords": [
                    "reliable", "professional", "vigilant", "responsible", "trustworthy",
                    "calm", "leader", "safety-focused", "experienced"
                ],
                "disqualifying_factors": [
                    "criminal background", "poor reliability", "lack of security experience",
                    "poor communication", "inability to handle stress"
                ],
                "experience_indicators": [
                    "security supervisor", "security officer", "loss prevention", "safety officer",
                    "security guard", "law enforcement", "military", "corrections"
                ],
                "education_preferences": [
                    "criminal justice", "security training", "law enforcement", "military",
                    "security certification", "public safety", "emergency response"
                ],
                "certifications": [
                    "security license", "CPR/First Aid", "security training", "emergency response",
                    "loss prevention", "safety certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Security Officer": {
                "must_have_skills": [
                    "security patrol", "observation skills", "communication", "emergency response",
                    "report writing", "safety awareness", "conflict de-escalation", "customer service",
                    "attention to detail", "reliability", "integrity", "physical fitness"
                ],
                "nice_to_have_skills": [
                    "surveillance monitoring", "access control", "crowd control", "basic investigation",
                    "emergency medical", "technology use", "multilingual", "guest relations"
                ],
                "technical_skills": [
                    "security equipment", "communication devices", "surveillance systems", "alarm systems",
                    "access control", "report writing", "computer basics", "mobile devices"
                ],
                "soft_skills": [
                    "alertness", "communication", "reliability", "integrity", "patience",
                    "stress management", "customer focus", "problem solving", "physical stamina"
                ],
                "cultural_fit_keywords": [
                    "reliable", "professional", "vigilant", "trustworthy", "alert",
                    "responsible", "calm", "guest-focused", "honest"
                ],
                "disqualifying_factors": [
                    "criminal background", "poor reliability", "lack of integrity",
                    "poor communication", "inability to stay alert"
                ],
                "experience_indicators": [
                    "security officer", "security guard", "loss prevention", "safety officer",
                    "patrol officer", "law enforcement", "military", "customer service"
                ],
                "education_preferences": [
                    "high school diploma", "security training", "criminal justice", "law enforcement",
                    "military", "security certification", "customer service"
                ],
                "certifications": [
                    "security license", "CPR/First Aid", "security training", "customer service",
                    "emergency response", "conflict resolution"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 38000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # SALES & MARKETING DEPARTMENT
            # ==========================================
            "Director of Sales": {
                "must_have_skills": [
                    "sales leadership", "revenue management", "team management", "strategic planning",
                    "market analysis", "client relations", "negotiation", "performance management",
                    "budget management", "sales forecasting", "contract negotiation", "business development"
                ],
                "nice_to_have_skills": [
                    "hospitality sales", "group sales", "corporate sales", "wedding sales", "event sales",
                    "digital marketing", "CRM systems", "lead generation", "pricing strategies"
                ],
                "technical_skills": [
                    "CRM systems", "sales analytics", "revenue management systems", "presentation software",
                    "database management", "social media", "marketing automation", "reporting tools"
                ],
                "soft_skills": [
                    "leadership", "communication", "negotiation", "strategic thinking", "relationship building",
                    "presentation skills", "analytical thinking", "results orientation", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "results-driven", "leader", "strategic", "relationship-builder", "innovative",
                    "competitive", "growth-oriented", "professional", "persuasive"
                ],
                "disqualifying_factors": [
                    "poor sales performance", "lack of leadership", "poor communication",
                    "inability to meet targets", "poor relationship skills"
                ],
                "experience_indicators": [
                    "sales director", "sales manager", "revenue manager", "business development",
                    "hospitality sales", "group sales", "corporate sales", "account management"
                ],
                "education_preferences": [
                    "business administration", "marketing", "hospitality management", "sales management",
                    "communications", "MBA", "revenue management"
                ],
                "certifications": [
                    "sales certification", "revenue management", "hospitality sales", "CRM certification",
                    "marketing certification", "business development"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 7,
                "preferred_experience_years": 10,
                "salary_range": {"min": 75000, "max": 120000},
                "growth_potential": "Executive",
                "training_requirements": "Advanced"
            },

            "Sales Manager": {
                "must_have_skills": [
                    "sales management", "team leadership", "client relations", "revenue generation",
                    "performance management", "market analysis", "negotiation", "communication",
                    "sales forecasting", "lead generation", "contract management", "customer service"
                ],
                "nice_to_have_skills": [
                    "hospitality sales", "group sales", "event sales", "corporate accounts",
                    "digital marketing", "CRM systems", "pricing strategies", "market research"
                ],
                "technical_skills": [
                    "CRM systems", "sales software", "presentation tools", "database management",
                    "analytics tools", "social media", "email marketing", "reporting systems"
                ],
                "soft_skills": [
                    "leadership", "communication", "negotiation", "relationship building",
                    "persuasion", "analytical thinking", "results orientation", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "results-driven", "relationship-builder", "persuasive", "professional",
                    "competitive", "goal-oriented", "customer-focused", "innovative"
                ],
                "disqualifying_factors": [
                    "poor sales performance", "lack of leadership", "poor communication",
                    "inability to build relationships", "poor customer service"
                ],
                "experience_indicators": [
                    "sales manager", "account manager", "business development", "sales representative",
                    "hospitality sales", "group sales", "revenue management", "client relations"
                ],
                "education_preferences": [
                    "business administration", "marketing", "hospitality management", "sales",
                    "communications", "customer relations", "business development"
                ],
                "certifications": [
                    "sales certification", "hospitality sales", "CRM certification", "customer service",
                    "marketing certification", "negotiation training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 4,
                "preferred_experience_years": 7,
                "salary_range": {"min": 50000, "max": 80000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Sales Representative": {
                "must_have_skills": [
                    "sales skills", "customer relations", "communication", "lead generation",
                    "negotiation", "product knowledge", "client presentation", "follow-up",
                    "relationship building", "goal achievement", "time management", "customer service"
                ],
                "nice_to_have_skills": [
                    "hospitality knowledge", "event sales", "group sales", "corporate sales",
                    "digital marketing", "social media", "CRM use", "market research"
                ],
                "technical_skills": [
                    "CRM systems", "presentation software", "social media platforms", "email systems",
                    "database management", "mobile apps", "communication tools", "analytics"
                ],
                "soft_skills": [
                    "communication", "persuasion", "relationship building", "persistence",
                    "enthusiasm", "adaptability", "goal orientation", "customer focus"
                ],
                "cultural_fit_keywords": [
                    "persuasive", "relationship-builder", "enthusiastic", "goal-oriented",
                    "customer-focused", "professional", "persistent", "results-driven"
                ],
                "disqualifying_factors": [
                    "poor communication", "lack of sales skills", "poor customer service",
                    "inability to build relationships", "lack of persistence"
                ],
                "experience_indicators": [
                    "sales representative", "sales associate", "account executive", "business development",
                    "customer relations", "hospitality sales", "retail sales", "customer service"
                ],
                "education_preferences": [
                    "business administration", "marketing", "communications", "hospitality",
                    "sales training", "customer service", "business studies"
                ],
                "certifications": [
                    "sales certification", "customer service", "communication skills",
                    "hospitality training", "CRM certification", "product knowledge"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # FINANCE & ACCOUNTING DEPARTMENT
            # ==========================================
            "Finance Manager": {
                "must_have_skills": [
                    "financial management", "accounting", "budgeting", "financial analysis",
                    "cost control", "financial reporting", "cash flow management", "audit preparation",
                    "team management", "compliance", "forecasting", "performance analysis"
                ],
                "nice_to_have_skills": [
                    "hospitality accounting", "revenue management", "tax preparation", "payroll management",
                    "accounts payable", "accounts receivable", "financial systems", "investment analysis"
                ],
                "technical_skills": [
                    "accounting software", "Excel advanced", "financial systems", "ERP systems",
                    "budgeting software", "reporting tools", "database management", "analytics"
                ],
                "soft_skills": [
                    "analytical thinking", "attention to detail", "leadership", "communication",
                    "problem solving", "organization", "integrity", "time management"
                ],
                "cultural_fit_keywords": [
                    "analytical", "detail-oriented", "reliable", "honest", "organized",
                    "leader", "professional", "accurate", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor financial knowledge", "lack of accuracy", "poor leadership",
                    "integrity issues", "poor communication"
                ],
                "experience_indicators": [
                    "finance manager", "accounting manager", "financial analyst", "controller",
                    "hospitality finance", "budget manager", "cost accounting", "financial reporting"
                ],
                "education_preferences": [
                    "accounting", "finance", "business administration", "economics",
                    "hospitality management", "MBA", "financial management"
                ],
                "certifications": [
                    "CPA", "CMA", "accounting certification", "financial management",
                    "hospitality finance", "QuickBooks", "Excel certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 60000, "max": 90000},
                "growth_potential": "High",
                "training_requirements": "Advanced"
            },

            "Accountant": {
                "must_have_skills": [
                    "accounting", "bookkeeping", "financial records", "data entry", "reconciliation",
                    "accounts payable", "accounts receivable", "payroll", "tax preparation",
                    "attention to detail", "accuracy", "financial software"
                ],
                "nice_to_have_skills": [
                    "hospitality accounting", "cost accounting", "budget assistance", "audit support",
                    "financial analysis", "reporting", "compliance", "inventory accounting"
                ],
                "technical_skills": [
                    "accounting software", "QuickBooks", "Excel", "payroll systems", "tax software",
                    "database entry", "financial systems", "reporting tools"
                ],
                "soft_skills": [
                    "attention to detail", "accuracy", "organization", "analytical thinking",
                    "reliability", "integrity", "time management", "communication"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "accurate", "reliable", "organized", "honest",
                    "analytical", "professional", "thorough", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of accuracy", "integrity issues",
                    "poor organization", "inability to meet deadlines"
                ],
                "experience_indicators": [
                    "accountant", "bookkeeper", "accounting clerk", "financial clerk",
                    "accounts payable", "accounts receivable", "payroll clerk", "tax preparer"
                ],
                "education_preferences": [
                    "accounting", "finance", "business administration", "bookkeeping",
                    "accounting certification", "business studies"
                ],
                "certifications": [
                    "accounting certification", "bookkeeping certification", "QuickBooks",
                    "payroll certification", "tax preparation", "Excel certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Accounts Payable Clerk": {
                "must_have_skills": [
                    "accounts payable", "data entry", "invoice processing", "vendor relations",
                    "payment processing", "record keeping", "attention to detail", "organization",
                    "accuracy", "communication", "time management", "accounting software"
                ],
                "nice_to_have_skills": [
                    "vendor management", "expense reporting", "reconciliation", "audit support",
                    "purchase order processing", "cost analysis", "filing systems", "customer service"
                ],
                "technical_skills": [
                    "accounting software", "QuickBooks", "Excel", "database entry", "email systems",
                    "payment systems", "scanning equipment", "filing systems"
                ],
                "soft_skills": [
                    "attention to detail", "accuracy", "organization", "reliability",
                    "communication", "time management", "patience", "problem solving"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "accurate", "organized", "reliable", "efficient",
                    "professional", "thorough", "dependable", "systematic"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of accuracy", "poor organization",
                    "unreliable", "poor communication"
                ],
                "experience_indicators": [
                    "accounts payable", "accounting clerk", "data entry", "bookkeeping",
                    "invoice processing", "vendor relations", "payment processing"
                ],
                "education_preferences": [
                    "high school diploma", "accounting", "business administration", "bookkeeping",
                    "data entry training", "office administration"
                ],
                "certifications": [
                    "accounting basics", "QuickBooks", "data entry", "office software",
                    "customer service", "bookkeeping basics"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.40, "cultural_fit": 0.25, "hospitality": 0.10
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 38000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            # ==========================================
            # HUMAN RESOURCES DEPARTMENT
            # ==========================================
            "HR Manager": {
                "must_have_skills": [
                    "HR management", "employee relations", "recruitment", "performance management",
                    "policy development", "training coordination", "compliance", "conflict resolution",
                    "team leadership", "compensation", "benefits administration", "employment law"
                ],
                "nice_to_have_skills": [
                    "hospitality HR", "labor relations", "HRIS systems", "onboarding", "succession planning",
                    "employee engagement", "diversity initiatives", "organizational development", "safety programs"
                ],
                "technical_skills": [
                    "HRIS systems", "payroll systems", "recruitment software", "performance management systems",
                    "compliance tracking", "reporting tools", "database management", "Microsoft Office"
                ],
                "soft_skills": [
                    "leadership", "communication", "empathy", "problem solving", "confidentiality",
                    "decision making", "interpersonal skills", "analytical thinking", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "leader", "supportive", "fair", "confidential", "professional",
                    "empathetic", "organized", "strategic", "people-focused"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of HR knowledge", "confidentiality breaches",
                    "poor communication", "bias or discrimination"
                ],
                "experience_indicators": [
                    "HR manager", "human resources", "personnel manager", "employee relations",
                    "recruitment manager", "training manager", "compensation manager", "hospitality HR"
                ],
                "education_preferences": [
                    "human resources", "business administration", "psychology", "organizational behavior",
                    "hospitality management", "employment law", "MBA"
                ],
                "certifications": [
                    "SHRM-CP", "SHRM-SCP", "PHR", "SPHR", "HR certification", "employment law", "HRIS"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 55000, "max": 85000},
                "growth_potential": "High",
                "training_requirements": "Advanced"
            },

            "HR Coordinator": {
                "must_have_skills": [
                    "HR support", "recruitment assistance", "employee records", "onboarding",
                    "data entry", "communication", "organization", "confidentiality",
                    "filing systems", "scheduling", "customer service", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "HRIS systems", "benefits administration", "training coordination", "employee relations",
                    "compliance tracking", "payroll support", "interview scheduling", "background checks"
                ],
                "technical_skills": [
                    "HRIS systems", "Microsoft Office", "database management", "scanning systems",
                    "email systems", "scheduling software", "recruitment tools", "filing systems"
                ],
                "soft_skills": [
                    "organization", "communication", "confidentiality", "attention to detail",
                    "customer service", "multitasking", "reliability", "interpersonal skills"
                ],
                "cultural_fit_keywords": [
                    "organized", "confidential", "supportive", "professional", "reliable",
                    "detail-oriented", "helpful", "efficient", "people-focused"
                ],
                "disqualifying_factors": [
                    "poor organization", "confidentiality breaches", "poor communication",
                    "lack of attention to detail", "poor customer service"
                ],
                "experience_indicators": [
                    "HR coordinator", "HR assistant", "personnel assistant", "recruitment coordinator",
                    "administrative assistant", "office coordinator", "employee services"
                ],
                "education_preferences": [
                    "human resources", "business administration", "office administration",
                    "communications", "psychology", "hospitality management"
                ],
                "certifications": [
                    "HR certification", "office administration", "customer service",
                    "Microsoft Office", "HRIS training", "confidentiality training"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # IT DEPARTMENT
            # ==========================================
            "IT Manager": {
                "must_have_skills": [
                    "IT management", "system administration", "network management", "team leadership",
                    "project management", "security protocols", "hardware management", "software management",
                    "troubleshooting", "vendor management", "budget management", "strategic planning"
                ],
                "nice_to_have_skills": [
                    "hospitality technology", "PMS systems", "cloud computing", "cybersecurity",
                    "database administration", "Wi-Fi management", "mobile technology", "backup systems"
                ],
                "technical_skills": [
                    "Windows Server", "networking", "Active Directory", "virtualization", "cloud platforms",
                    "cybersecurity tools", "database management", "backup systems", "monitoring tools"
                ],
                "soft_skills": [
                    "leadership", "problem solving", "communication", "analytical thinking",
                    "project management", "team building", "adaptability", "stress management"
                ],
                "cultural_fit_keywords": [
                    "technical", "leader", "innovative", "problem-solver", "reliable",
                    "analytical", "strategic", "professional", "efficient"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of technical knowledge", "poor communication",
                    "inability to handle stress", "poor project management"
                ],
                "experience_indicators": [
                    "IT manager", "system administrator", "network administrator", "IT director",
                    "technology manager", "hospitality IT", "infrastructure manager"
                ],
                "education_preferences": [
                    "computer science", "information technology", "network administration",
                    "cybersecurity", "business administration", "hospitality technology"
                ],
                "certifications": [
                    "CompTIA", "Cisco", "Microsoft", "VMware", "security certifications",
                    "project management", "ITIL", "cloud certifications"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 65000, "max": 95000},
                "growth_potential": "High",
                "training_requirements": "Advanced"
            },

            "IT Technician": {
                "must_have_skills": [
                    "computer repair", "troubleshooting", "hardware installation", "software installation",
                    "network support", "user support", "system maintenance", "documentation",
                    "customer service", "problem solving", "technical communication", "equipment setup"
                ],
                "nice_to_have_skills": [
                    "hospitality systems", "PMS support", "mobile device support", "printer support",
                    "Wi-Fi troubleshooting", "phone systems", "audio visual", "cable management"
                ],
                "technical_skills": [
                    "Windows", "Mac OS", "networking basics", "hardware components", "software applications",
                    "mobile devices", "printers", "audio visual equipment", "diagnostic tools"
                ],
                "soft_skills": [
                    "problem solving", "communication", "patience", "customer service",
                    "learning ability", "attention to detail", "reliability", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "technical", "helpful", "patient", "problem-solver", "reliable",
                    "professional", "learning-oriented", "customer-focused", "detail-oriented"
                ],
                "disqualifying_factors": [
                    "poor technical skills", "poor customer service", "inability to learn",
                    "poor communication", "lack of patience"
                ],
                "experience_indicators": [
                    "IT technician", "computer technician", "help desk", "technical support",
                    "hardware technician", "system support", "user support"
                ],
                "education_preferences": [
                    "computer science", "information technology", "technical training",
                    "computer repair", "networking", "hospitality technology"
                ],
                "certifications": [
                    "CompTIA A+", "Microsoft", "hardware certification", "networking basics",
                    "customer service", "technical support", "hospitality systems"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # SPA & WELLNESS DEPARTMENT
            # ==========================================
            "Spa Manager": {
                "must_have_skills": [
                    "spa management", "wellness programs", "team leadership", "customer service",
                    "treatment scheduling", "inventory management", "staff training", "performance management",
                    "budget awareness", "quality control", "guest relations", "communication"
                ],
                "nice_to_have_skills": [
                    "massage therapy", "esthetics", "wellness coaching", "retail management",
                    "marketing", "event coordination", "product knowledge", "health protocols"
                ],
                "technical_skills": [
                    "spa software", "scheduling systems", "POS systems", "inventory systems",
                    "treatment equipment", "sound systems", "lighting controls", "sanitation equipment"
                ],
                "soft_skills": [
                    "leadership", "wellness focus", "communication", "empathy", "organization",
                    "customer focus", "attention to detail", "calming presence", "team building"
                ],
                "cultural_fit_keywords": [
                    "wellness-focused", "leader", "calming", "professional", "empathetic",
                    "organized", "guest-focused", "quality-oriented", "serene"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of wellness knowledge", "poor customer service",
                    "inability to create calm environment", "poor communication"
                ],
                "experience_indicators": [
                    "spa manager", "wellness manager", "spa director", "massage therapy manager",
                    "esthetics manager", "fitness manager", "wellness coordinator"
                ],
                "education_preferences": [
                    "spa management", "wellness", "massage therapy", "esthetics", "hospitality management",
                    "business administration", "health and wellness"
                ],
                "certifications": [
                    "spa management", "massage therapy", "esthetics license", "wellness coaching",
                    "hospitality management", "customer service", "health and safety"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 45000, "max": 70000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Massage Therapist": {
                "must_have_skills": [
                    "massage therapy", "anatomy knowledge", "therapeutic techniques", "customer service",
                    "communication", "professional boundaries", "sanitation protocols", "physical stamina",
                    "empathy", "attention to detail", "time management", "wellness focus"
                ],
                "nice_to_have_skills": [
                    "specialized techniques", "aromatherapy", "hot stone", "deep tissue", "prenatal massage",
                    "reflexology", "sports massage", "energy work", "product knowledge"
                ],
                "technical_skills": [
                    "massage equipment", "treatment table setup", "sanitation equipment", "essential oils",
                    "hot stone equipment", "music systems", "lighting controls", "treatment tools"
                ],
                "soft_skills": [
                    "empathy", "communication", "professionalism", "healing touch", "calming presence",
                    "physical stamina", "attention to detail", "wellness mindset", "patience"
                ],
                "cultural_fit_keywords": [
                    "healing", "empathetic", "professional", "calming", "skilled",
                    "wellness-focused", "therapeutic", "caring", "serene"
                ],
                "disqualifying_factors": [
                    "lack of license", "poor boundaries", "inappropriate behavior",
                    "poor technique", "lack of empathy"
                ],
                "experience_indicators": [
                    "massage therapist", "therapeutic massage", "spa therapist", "wellness therapist",
                    "bodywork", "massage practice", "healing arts"
                ],
                "education_preferences": [
                    "massage therapy school", "therapeutic massage", "bodywork training",
                    "wellness studies", "anatomy and physiology", "healing arts"
                ],
                "certifications": [
                    "massage therapy license", "therapeutic massage", "specialized techniques",
                    "anatomy certification", "CPR/First Aid", "wellness certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 50000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # ENTERTAINMENT & ACTIVITIES DEPARTMENT
            # ==========================================
            "Activities Manager": {
                "must_have_skills": [
                    "activity planning", "event coordination", "team leadership", "guest entertainment",
                    "program development", "scheduling", "customer service", "communication",
                    "creativity", "problem solving", "performance management", "safety awareness"
                ],
                "nice_to_have_skills": [
                    "sports knowledge", "water activities", "cultural programming", "children's activities",
                    "entertainment skills", "music knowledge", "dance knowledge", "arts and crafts"
                ],
                "technical_skills": [
                    "sound systems", "microphones", "activity equipment", "scheduling software",
                    "entertainment technology", "safety equipment", "sports equipment", "craft supplies"
                ],
                "soft_skills": [
                    "creativity", "leadership", "enthusiasm", "communication", "energy",
                    "customer focus", "problem solving", "adaptability", "team building"
                ],
                "cultural_fit_keywords": [
                    "energetic", "creative", "leader", "enthusiastic", "fun",
                    "guest-focused", "organized", "innovative", "entertaining"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of creativity", "low energy", "poor customer service",
                    "safety violations"
                ],
                "experience_indicators": [
                    "activities manager", "recreation manager", "entertainment manager", "program coordinator",
                    "activities coordinator", "resort activities", "guest services"
                ],
                "education_preferences": [
                    "recreation management", "hospitality management", "sports management",
                    "entertainment", "event management", "physical education"
                ],
                "certifications": [
                    "recreation certification", "activity management", "water safety", "first aid/CPR",
                    "entertainment certification", "sports certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Activities Coordinator": {
                "must_have_skills": [
                    "activity coordination", "guest interaction", "event support", "scheduling",
                    "customer service", "communication", "enthusiasm", "organization",
                    "team work", "problem solving", "safety awareness", "multitasking"
                ],
                "nice_to_have_skills": [
                    "sports knowledge", "entertainment skills", "children's activities", "water safety",
                    "arts and crafts", "music", "dance", "multilingual", "first aid"
                ],
                "technical_skills": [
                    "activity equipment", "sound systems", "sports equipment", "craft supplies",
                    "safety equipment", "entertainment technology", "communication devices"
                ],
                "soft_skills": [
                    "enthusiasm", "energy", "communication", "customer focus", "creativity",
                    "teamwork", "adaptability", "patience", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "enthusiastic", "energetic", "fun", "guest-focused", "creative",
                    "positive", "team player", "entertaining", "engaging"
                ],
                "disqualifying_factors": [
                    "low energy", "poor customer service", "lack of enthusiasm",
                    "safety violations", "poor communication"
                ],
                "experience_indicators": [
                    "activities coordinator", "recreation assistant", "entertainment staff",
                    "camp counselor", "activity leader", "guest services", "youth programs"
                ],
                "education_preferences": [
                    "recreation", "hospitality", "sports management", "education",
                    "entertainment", "customer service", "physical education"
                ],
                "certifications": [
                    "recreation certification", "water safety", "first aid/CPR", "activity leadership",
                    "customer service", "entertainment skills"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 22000, "max": 35000},
                "growth_potential": "High",
                "training_requirements": "Basic"
            },

            # ==========================================
            # GUEST RELATIONS & VIP SERVICES
            # ==========================================
            "Guest Relations Manager": {
                "must_have_skills": [
                    "guest relations", "complaint resolution", "VIP services", "luxury hospitality",
                    "customer service excellence", "communication", "problem solving", "team leadership"
                ],
                "nice_to_have_skills": [
                    "multilingual", "cultural sensitivity", "luxury brands", "personalized service"
                ],
                "technical_skills": ["CRM systems", "guest feedback platforms", "luxury service standards"],
                "soft_skills": ["empathy", "diplomacy", "patience", "emotional intelligence"],
                "cultural_fit_keywords": ["guest-focused", "diplomatic", "luxury-minded", "solution-oriented"],
                "disqualifying_factors": ["poor guest interaction", "inflexibility", "lack of empathy"],
                "experience_indicators": ["guest relations", "VIP services", "luxury hospitality", "customer service"],
                "education_preferences": ["hospitality management", "communications", "business", "psychology"],
                "certifications": ["guest relations", "luxury service", "hospitality management"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 70000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Guest Services Representative": {
                "must_have_skills": [
                    "customer service", "guest assistance", "problem resolution", "communication",
                    "hospitality service", "attention to detail", "multitasking", "professional demeanor"
                ],
                "nice_to_have_skills": [
                    "multilingual", "local knowledge", "concierge services", "reservation systems"
                ],
                "technical_skills": ["guest service software", "reservation systems", "communication tools"],
                "soft_skills": ["patience", "empathy", "adaptability", "positive attitude"],
                "cultural_fit_keywords": ["service-oriented", "helpful", "friendly", "professional"],
                "disqualifying_factors": ["poor communication", "impatience", "negative attitude"],
                "experience_indicators": ["guest services", "customer service", "hospitality", "reception"],
                "education_preferences": ["hospitality", "communications", "business", "tourism"],
                "certifications": ["customer service", "guest relations", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 32000, "max": 48000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "VIP Host": {
                "must_have_skills": [
                    "VIP service", "luxury hospitality", "personalized service", "attention to detail",
                    "discretion", "communication", "problem solving", "cultural sensitivity"
                ],
                "nice_to_have_skills": [
                    "multilingual", "fine dining", "wine knowledge", "etiquette", "luxury brands"
                ],
                "technical_skills": ["luxury service systems", "VIP databases", "communication tools"],
                "soft_skills": ["sophistication", "discretion", "anticipation", "refinement"],
                "cultural_fit_keywords": ["sophisticated", "discreet", "attentive", "refined"],
                "disqualifying_factors": ["lack of sophistication", "poor discretion", "inflexibility"],
                "experience_indicators": ["VIP services", "luxury hospitality", "personal service", "high-end"],
                "education_preferences": ["hospitality", "luxury service", "business", "communications"],
                "certifications": ["luxury service", "VIP training", "hospitality management"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Club Level Coordinator": {
                "must_have_skills": [
                    "club level service", "exclusive amenities", "member relations", "luxury service",
                    "attention to detail", "communication", "organization", "problem solving"
                ],
                "nice_to_have_skills": [
                    "wine service", "culinary knowledge", "event coordination", "cultural awareness"
                ],
                "technical_skills": ["club management systems", "member databases", "service platforms"],
                "soft_skills": ["exclusivity mindset", "attention to detail", "sophistication", "discretion"],
                "cultural_fit_keywords": ["exclusive", "sophisticated", "attentive", "premium"],
                "disqualifying_factors": ["lack of attention to detail", "poor service orientation", "inflexibility"],
                "experience_indicators": ["club level", "luxury service", "member services", "exclusive hospitality"],
                "education_preferences": ["hospitality", "luxury service", "business", "culinary"],
                "certifications": ["luxury service", "club management", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 58000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Personal Concierge": {
                "must_have_skills": [
                    "personal assistance", "concierge services", "local knowledge", "reservation management",
                    "itinerary planning", "communication", "problem solving", "discretion"
                ],
                "nice_to_have_skills": [
                    "multilingual", "cultural knowledge", "luxury services", "travel planning"
                ],
                "technical_skills": ["concierge software", "reservation systems", "communication tools"],
                "soft_skills": ["anticipation", "discretion", "resourcefulness", "sophistication"],
                "cultural_fit_keywords": ["resourceful", "discreet", "knowledgeable", "helpful"],
                "disqualifying_factors": ["poor local knowledge", "lack of resourcefulness", "poor communication"],
                "experience_indicators": ["concierge", "personal assistance", "travel services", "luxury hospitality"],
                "education_preferences": ["hospitality", "tourism", "communications", "local studies"],
                "certifications": ["concierge certification", "travel planning", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 42000, "max": 62000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            # ==========================================
            # TRANSPORTATION & LOGISTICS
            # ==========================================
            "Valet Parking Attendant": {
                "must_have_skills": [
                    "driving skills", "customer service", "vehicle handling", "parking management",
                    "professional appearance", "communication", "attention to detail", "physical fitness"
                ],
                "nice_to_have_skills": [
                    "luxury vehicle experience", "manual transmission", "defensive driving", "guest relations"
                ],
                "technical_skills": ["parking systems", "vehicle operation", "safety protocols"],
                "soft_skills": ["trustworthiness", "responsibility", "courtesy", "reliability"],
                "cultural_fit_keywords": ["trustworthy", "responsible", "courteous", "professional"],
                "disqualifying_factors": ["poor driving record", "unreliable", "poor customer service"],
                "experience_indicators": ["valet", "parking", "driving", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "automotive"],
                "certifications": ["valid driver's license", "defensive driving", "valet training"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 2,
                "salary_range": {"min": 28000, "max": 40000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Transportation Coordinator": {
                "must_have_skills": [
                    "transportation coordination", "scheduling", "logistics", "customer service",
                    "vendor management", "route planning", "communication", "organization"
                ],
                "nice_to_have_skills": [
                    "fleet management", "GPS systems", "airport transfers", "group transportation"
                ],
                "technical_skills": ["transportation software", "GPS systems", "scheduling tools"],
                "soft_skills": ["organization", "reliability", "problem solving", "communication"],
                "cultural_fit_keywords": ["organized", "reliable", "efficient", "service-oriented"],
                "disqualifying_factors": ["poor organization", "unreliable", "poor communication"],
                "experience_indicators": ["transportation", "logistics", "coordination", "hospitality"],
                "education_preferences": ["logistics", "transportation", "hospitality", "business"],
                "certifications": ["transportation management", "logistics", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Mid-Level", "training_requirements": "Standard"
            },

            "Shuttle Driver": {
                "must_have_skills": [
                    "safe driving", "customer service", "vehicle maintenance", "punctuality",
                    "professional appearance", "local knowledge", "communication", "reliability"
                ],
                "nice_to_have_skills": [
                    "commercial license", "multilingual", "tour guiding", "defensive driving"
                ],
                "technical_skills": ["vehicle operation", "GPS systems", "safety protocols"],
                "soft_skills": ["responsibility", "courtesy", "punctuality", "professionalism"],
                "cultural_fit_keywords": ["reliable", "safe", "courteous", "professional"],
                "disqualifying_factors": ["poor driving record", "unreliable", "unprofessional"],
                "experience_indicators": ["driving", "transportation", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "transportation"],
                "certifications": ["valid driver's license", "commercial license", "defensive driving"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Luggage Porter": {
                "must_have_skills": [
                    "physical fitness", "customer service", "luggage handling", "attention to detail",
                    "reliability", "communication", "professional appearance", "teamwork"
                ],
                "nice_to_have_skills": [
                    "multilingual", "guest relations", "hospitality experience", "equipment operation"
                ],
                "technical_skills": ["luggage equipment", "safety protocols", "handling techniques"],
                "soft_skills": ["helpfulness", "courtesy", "reliability", "strength"],
                "cultural_fit_keywords": ["helpful", "courteous", "reliable", "strong"],
                "disqualifying_factors": ["physical limitations", "poor customer service", "unreliable"],
                "experience_indicators": ["porter", "luggage handling", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "customer service"],
                "certifications": ["safety training", "customer service", "hospitality"],
                "scoring_weights": {"experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15},
                "min_experience_years": 0, "preferred_experience_years": 1,
                "salary_range": {"min": 26000, "max": 35000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Airport Transfer Coordinator": {
                "must_have_skills": [
                    "transfer coordination", "scheduling", "customer service", "logistics",
                    "communication", "organization", "problem solving", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "airline knowledge", "flight tracking", "multilingual", "travel coordination"
                ],
                "technical_skills": ["booking systems", "flight tracking", "communication tools"],
                "soft_skills": ["organization", "reliability", "flexibility", "communication"],
                "cultural_fit_keywords": ["organized", "reliable", "flexible", "service-oriented"],
                "disqualifying_factors": ["poor organization", "inflexibility", "poor communication"],
                "experience_indicators": ["transfer coordination", "travel services", "hospitality", "logistics"],
                "education_preferences": ["hospitality", "travel", "logistics", "business"],
                "certifications": ["travel coordination", "hospitality", "logistics"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            # ==========================================
            # CONFERENCE & EVENTS
            # ==========================================
            "Conference Manager": {
                "must_have_skills": [
                    "conference management", "event planning", "vendor coordination", "logistics",
                    "customer service", "project management", "communication", "budget management"
                ],
                "nice_to_have_skills": [
                    "AV equipment", "catering coordination", "group sales", "contract negotiation"
                ],
                "technical_skills": ["event management software", "AV systems", "booking systems"],
                "soft_skills": ["organization", "multitasking", "problem solving", "leadership"],
                "cultural_fit_keywords": ["organized", "detail-oriented", "service-focused", "professional"],
                "disqualifying_factors": ["poor organization", "inflexibility", "poor communication"],
                "experience_indicators": ["conference management", "event planning", "meetings", "hospitality"],
                "education_preferences": ["event management", "hospitality", "business", "communications"],
                "certifications": ["event planning", "conference management", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            "Event Coordinator": {
                "must_have_skills": [
                    "event planning", "coordination", "vendor management", "timeline management",
                    "customer service", "budget management", "communication", "problem solving"
                ],
                "nice_to_have_skills": [
                    "wedding planning", "corporate events", "social events", "catering coordination"
                ],
                "technical_skills": ["event software", "booking systems", "project management tools"],
                "soft_skills": ["creativity", "organization", "flexibility", "attention to detail"],
                "cultural_fit_keywords": ["creative", "organized", "flexible", "service-oriented"],
                "disqualifying_factors": ["poor organization", "inflexibility", "poor client relations"],
                "experience_indicators": ["event planning", "coordination", "hospitality", "weddings"],
                "education_preferences": ["event management", "hospitality", "business", "marketing"],
                "certifications": ["event planning", "wedding planning", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "Wedding Coordinator": {
                "must_have_skills": [
                    "wedding planning", "event coordination", "vendor management", "client relations",
                    "timeline management", "attention to detail", "communication", "problem solving"
                ],
                "nice_to_have_skills": [
                    "floral design", "catering coordination", "photography coordination", "ceremony planning"
                ],
                "technical_skills": ["wedding planning software", "event management tools", "booking systems"],
                "soft_skills": ["patience", "creativity", "organization", "emotional intelligence"],
                "cultural_fit_keywords": ["creative", "patient", "detail-oriented", "romantic"],
                "disqualifying_factors": ["poor organization", "impatience", "inflexibility"],
                "experience_indicators": ["wedding planning", "event coordination", "hospitality", "celebrations"],
                "education_preferences": ["event management", "hospitality", "business", "design"],
                "certifications": ["wedding planning", "event coordination", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 58000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Meeting Planner": {
                "must_have_skills": [
                    "meeting planning", "logistics coordination", "venue management", "vendor relations",
                    "budget management", "timeline management", "communication", "problem solving"
                ],
                "nice_to_have_skills": [
                    "corporate events", "conference planning", "AV coordination", "travel coordination"
                ],
                "technical_skills": ["meeting planning software", "booking systems", "budget tools"],
                "soft_skills": ["organization", "attention to detail", "multitasking", "communication"],
                "cultural_fit_keywords": ["organized", "detail-oriented", "professional", "efficient"],
                "disqualifying_factors": ["poor organization", "missed deadlines", "poor communication"],
                "experience_indicators": ["meeting planning", "event coordination", "corporate events", "hospitality"],
                "education_preferences": ["event management", "business", "hospitality", "communications"],
                "certifications": ["meeting planning", "event coordination", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 42000, "max": 62000}, "growth_potential": "Mid-Level", "training_requirements": "Standard"
            },

            "AV Technician": {
                "must_have_skills": [
                    "AV equipment", "technical setup", "troubleshooting", "equipment maintenance",
                    "customer service", "problem solving", "attention to detail", "safety protocols"
                ],
                "nice_to_have_skills": [
                    "lighting systems", "sound systems", "video equipment", "streaming technology"
                ],
                "technical_skills": ["AV systems", "technical equipment", "troubleshooting", "setup procedures"],
                "soft_skills": ["technical aptitude", "problem solving", "reliability", "communication"],
                "cultural_fit_keywords": ["technical", "reliable", "problem-solver", "detail-oriented"],
                "disqualifying_factors": ["poor technical skills", "unreliable", "safety violations"],
                "experience_indicators": ["AV technician", "technical support", "equipment operation", "events"],
                "education_preferences": ["technical education", "electronics", "communications", "hospitality"],
                "certifications": ["AV certification", "technical training", "safety certification"],
                "scoring_weights": {"experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Mid-Level", "training_requirements": "Technical"
            },

            # ==========================================
            # HEALTH & WELLNESS
            # ==========================================
            "Wellness Coordinator": {
                "must_have_skills": [
                    "wellness programs", "health promotion", "fitness coordination", "customer service",
                    "program development", "safety protocols", "communication", "organization"
                ],
                "nice_to_have_skills": [
                    "nutrition knowledge", "fitness training", "spa services", "meditation"
                ],
                "technical_skills": ["wellness software", "fitness equipment", "health tracking tools"],
                "soft_skills": ["motivation", "empathy", "enthusiasm", "patience"],
                "cultural_fit_keywords": ["health-conscious", "motivating", "caring", "positive"],
                "disqualifying_factors": ["poor health habits", "lack of enthusiasm", "safety violations"],
                "experience_indicators": ["wellness", "fitness", "health promotion", "spa", "hospitality"],
                "education_preferences": ["health sciences", "fitness", "wellness", "hospitality"],
                "certifications": ["wellness coaching", "fitness", "CPR", "first aid"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 60000}, "growth_potential": "Mid-Level", "training_requirements": "Standard"
            },

            "Fitness Instructor": {
                "must_have_skills": [
                    "fitness instruction", "exercise programming", "safety protocols", "customer service",
                    "motivational skills", "anatomy knowledge", "equipment operation", "communication"
                ],
                "nice_to_have_skills": [
                    "specialized training", "nutrition knowledge", "injury prevention", "group fitness"
                ],
                "technical_skills": ["fitness equipment", "exercise software", "heart rate monitors"],
                "soft_skills": ["motivation", "enthusiasm", "patience", "energy"],
                "cultural_fit_keywords": ["energetic", "motivating", "health-focused", "positive"],
                "disqualifying_factors": ["poor fitness", "safety violations", "lack of certification"],
                "experience_indicators": ["fitness instruction", "personal training", "group fitness", "wellness"],
                "education_preferences": ["exercise science", "kinesiology", "fitness", "health"],
                "certifications": ["fitness certification", "CPR", "first aid", "specialized training"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Specialized"
            },

            "Yoga Instructor": {
                "must_have_skills": [
                    "yoga instruction", "meditation", "breathing techniques", "anatomy knowledge",
                    "safety protocols", "class management", "customer service", "flexibility"
                ],
                "nice_to_have_skills": [
                    "multiple yoga styles", "spiritual guidance", "wellness coaching", "injury modification"
                ],
                "technical_skills": ["yoga equipment", "sound systems", "class scheduling"],
                "soft_skills": ["calmness", "patience", "mindfulness", "spirituality"],
                "cultural_fit_keywords": ["mindful", "calm", "spiritual", "wellness-focused"],
                "disqualifying_factors": ["poor physical condition", "lack of certification", "impatience"],
                "experience_indicators": ["yoga instruction", "meditation", "wellness", "fitness"],
                "education_preferences": ["yoga studies", "fitness", "wellness", "spiritual studies"],
                "certifications": ["yoga certification", "meditation training", "CPR", "first aid"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 32000, "max": 48000}, "growth_potential": "Entry-Mid", "training_requirements": "Specialized"
            },

            "Pool Attendant": {
                "must_have_skills": [
                    "water safety", "customer service", "pool maintenance", "safety protocols",
                    "equipment operation", "cleaning", "communication", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "lifeguard certification", "CPR", "first aid", "swimming instruction"
                ],
                "technical_skills": ["pool equipment", "chemical testing", "cleaning equipment"],
                "soft_skills": ["vigilance", "responsibility", "helpfulness", "reliability"],
                "cultural_fit_keywords": ["safety-conscious", "responsible", "helpful", "vigilant"],
                "disqualifying_factors": ["poor swimming ability", "safety violations", "unreliable"],
                "experience_indicators": ["pool maintenance", "water safety", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "recreation", "water safety"],
                "certifications": ["water safety", "pool maintenance", "CPR", "first aid"],
                "scoring_weights": {"experience": 0.25, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 0, "preferred_experience_years": 2,
                "salary_range": {"min": 28000, "max": 40000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Lifeguard": {
                "must_have_skills": [
                    "water safety", "rescue techniques", "first aid", "CPR", "emergency response",
                    "surveillance", "communication", "physical fitness"
                ],
                "nice_to_have_skills": [
                    "swimming instruction", "water sports", "emergency medical training", "customer service"
                ],
                "technical_skills": ["rescue equipment", "first aid equipment", "communication devices"],
                "soft_skills": ["vigilance", "quick response", "calmness under pressure", "responsibility"],
                "cultural_fit_keywords": ["vigilant", "responsible", "quick-thinking", "safety-focused"],
                "disqualifying_factors": ["poor swimming ability", "slow response time", "safety violations"],
                "experience_indicators": ["lifeguarding", "water safety", "rescue", "emergency response"],
                "education_preferences": ["water safety", "emergency response", "recreation", "health"],
                "certifications": ["lifeguard certification", "CPR", "first aid", "water safety instructor"],
                "scoring_weights": {"experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 0, "preferred_experience_years": 2,
                "salary_range": {"min": 30000, "max": 45000}, "growth_potential": "Entry", "training_requirements": "Specialized"
            },

            # ==========================================
            # BUSINESS & REVENUE MANAGEMENT
            # ==========================================
            "Revenue Manager": {
                "must_have_skills": [
                    "revenue management", "pricing strategy", "demand forecasting", "data analysis",
                    "financial analysis", "market research", "optimization", "reporting"
                ],
                "nice_to_have_skills": [
                    "revenue management systems", "competitive analysis", "yield management", "distribution"
                ],
                "technical_skills": ["revenue management software", "analytics tools", "pricing systems"],
                "soft_skills": ["analytical thinking", "strategic planning", "communication", "attention to detail"],
                "cultural_fit_keywords": ["analytical", "strategic", "results-driven", "detail-oriented"],
                "disqualifying_factors": ["poor analytical skills", "lack of financial acumen", "inflexibility"],
                "experience_indicators": ["revenue management", "pricing", "analytics", "finance", "hospitality"],
                "education_preferences": ["finance", "economics", "business", "hospitality", "mathematics"],
                "certifications": ["revenue management", "financial analysis", "hospitality finance"],
                "scoring_weights": {"experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 65000, "max": 95000}, "growth_potential": "Senior", "training_requirements": "Advanced"
            },

            "Purchasing Manager": {
                "must_have_skills": [
                    "procurement", "vendor management", "contract negotiation", "inventory management",
                    "cost analysis", "supplier relations", "budget management", "quality control"
                ],
                "nice_to_have_skills": [
                    "hospitality purchasing", "food & beverage procurement", "sustainability", "logistics"
                ],
                "technical_skills": ["procurement software", "inventory systems", "ERP systems"],
                "soft_skills": ["negotiation", "analytical thinking", "organization", "communication"],
                "cultural_fit_keywords": ["cost-conscious", "analytical", "negotiator", "organized"],
                "disqualifying_factors": ["poor negotiation", "disorganization", "lack of cost awareness"],
                "experience_indicators": ["purchasing", "procurement", "vendor management", "hospitality"],
                "education_preferences": ["business", "supply chain", "hospitality", "finance"],
                "certifications": ["procurement", "supply chain", "hospitality purchasing"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 55000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            "Business Analyst": {
                "must_have_skills": [
                    "business analysis", "data analysis", "process improvement", "reporting",
                    "requirements gathering", "problem solving", "communication", "project management"
                ],
                "nice_to_have_skills": [
                    "hospitality analytics", "performance metrics", "dashboard creation", "process mapping"
                ],
                "technical_skills": ["analytics software", "business intelligence tools", "database systems"],
                "soft_skills": ["analytical thinking", "attention to detail", "communication", "critical thinking"],
                "cultural_fit_keywords": ["analytical", "detail-oriented", "improvement-focused", "systematic"],
                "disqualifying_factors": ["poor analytical skills", "inflexibility", "poor communication"],
                "experience_indicators": ["business analysis", "data analysis", "process improvement", "hospitality"],
                "education_preferences": ["business", "analytics", "hospitality", "information systems"],
                "certifications": ["business analysis", "analytics", "project management"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 50000, "max": 70000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Controller": {
                "must_have_skills": [
                    "financial control", "accounting oversight", "financial reporting", "budget management",
                    "compliance", "audit management", "team leadership", "financial analysis"
                ],
                "nice_to_have_skills": [
                    "hospitality accounting", "cost control", "forecasting", "tax preparation"
                ],
                "technical_skills": ["accounting software", "financial systems", "reporting tools"],
                "soft_skills": ["attention to detail", "leadership", "communication", "analytical thinking"],
                "cultural_fit_keywords": ["detail-oriented", "accurate", "responsible", "analytical"],
                "disqualifying_factors": ["poor attention to detail", "lack of accounting knowledge", "poor leadership"],
                "experience_indicators": ["controller", "accounting management", "financial oversight", "hospitality"],
                "education_preferences": ["accounting", "finance", "business", "hospitality finance"],
                "certifications": ["CPA", "CMA", "financial management", "hospitality accounting"],
                "scoring_weights": {"experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 5, "preferred_experience_years": 8,
                "salary_range": {"min": 70000, "max": 100000}, "growth_potential": "Senior", "training_requirements": "Advanced"
            },

            "Cost Control Manager": {
                "must_have_skills": [
                    "cost control", "budget analysis", "variance analysis", "financial reporting",
                    "inventory control", "process improvement", "team leadership", "analytical skills"
                ],
                "nice_to_have_skills": [
                    "hospitality cost control", "food cost management", "labor cost analysis", "forecasting"
                ],
                "technical_skills": ["cost control software", "analytics tools", "inventory systems"],
                "soft_skills": ["analytical thinking", "attention to detail", "communication", "organization"],
                "cultural_fit_keywords": ["cost-conscious", "analytical", "efficient", "detail-oriented"],
                "disqualifying_factors": ["poor analytical skills", "lack of cost awareness", "disorganization"],
                "experience_indicators": ["cost control", "financial analysis", "budget management", "hospitality"],
                "education_preferences": ["finance", "accounting", "business", "hospitality management"],
                "certifications": ["cost control", "financial analysis", "hospitality finance"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 55000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            # ==========================================
            # DIGITAL MARKETING & TECHNOLOGY
            # ==========================================
            "Digital Marketing Manager": {
                "must_have_skills": [
                    "digital marketing", "social media", "online advertising", "content creation",
                    "SEO/SEM", "analytics", "campaign management", "brand management"
                ],
                "nice_to_have_skills": [
                    "hospitality marketing", "OTA management", "email marketing", "influencer marketing"
                ],
                "technical_skills": ["marketing platforms", "analytics tools", "social media tools"],
                "soft_skills": ["creativity", "analytical thinking", "communication", "adaptability"],
                "cultural_fit_keywords": ["creative", "digital-savvy", "innovative", "results-driven"],
                "disqualifying_factors": ["poor digital skills", "lack of creativity", "poor analytics"],
                "experience_indicators": ["digital marketing", "social media", "online marketing", "hospitality"],
                "education_preferences": ["marketing", "communications", "business", "digital media"],
                "certifications": ["digital marketing", "Google Analytics", "social media", "hospitality marketing"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            "Social Media Coordinator": {
                "must_have_skills": [
                    "social media management", "content creation", "community management", "photography",
                    "copywriting", "brand consistency", "engagement", "analytics"
                ],
                "nice_to_have_skills": [
                    "video creation", "graphic design", "influencer relations", "paid advertising"
                ],
                "technical_skills": ["social media platforms", "content creation tools", "analytics tools"],
                "soft_skills": ["creativity", "communication", "attention to detail", "adaptability"],
                "cultural_fit_keywords": ["creative", "social", "trendy", "engaging"],
                "disqualifying_factors": ["poor social skills", "lack of creativity", "poor attention to detail"],
                "experience_indicators": ["social media", "content creation", "marketing", "communications"],
                "education_preferences": ["marketing", "communications", "graphic design", "media"],
                "certifications": ["social media marketing", "content creation", "digital marketing"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "Content Creator": {
                "must_have_skills": [
                    "content creation", "writing", "photography", "video production", "editing",
                    "storytelling", "brand voice", "social media content"
                ],
                "nice_to_have_skills": [
                    "graphic design", "animation", "SEO writing", "hospitality content"
                ],
                "technical_skills": ["content creation software", "editing tools", "design software"],
                "soft_skills": ["creativity", "attention to detail", "storytelling", "visual sense"],
                "cultural_fit_keywords": ["creative", "visual", "storyteller", "artistic"],
                "disqualifying_factors": ["lack of creativity", "poor visual sense", "poor writing"],
                "experience_indicators": ["content creation", "photography", "video production", "marketing"],
                "education_preferences": ["media arts", "communications", "graphic design", "marketing"],
                "certifications": ["content creation", "photography", "video production", "social media"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "SEO Specialist": {
                "must_have_skills": [
                    "search engine optimization", "keyword research", "content optimization", "analytics",
                    "link building", "technical SEO", "reporting", "strategy development"
                ],
                "nice_to_have_skills": [
                    "local SEO", "hospitality SEO", "PPC", "conversion optimization"
                ],
                "technical_skills": ["SEO tools", "analytics platforms", "content management systems"],
                "soft_skills": ["analytical thinking", "attention to detail", "persistence", "communication"],
                "cultural_fit_keywords": ["analytical", "detail-oriented", "strategic", "results-driven"],
                "disqualifying_factors": ["poor analytical skills", "lack of technical knowledge", "impatience"],
                "experience_indicators": ["SEO", "digital marketing", "web optimization", "analytics"],
                "education_preferences": ["marketing", "computer science", "communications", "business"],
                "certifications": ["SEO certification", "Google Analytics", "digital marketing"],
                "scoring_weights": {"experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 45000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Web Developer": {
                "must_have_skills": [
                    "web development", "HTML", "CSS", "JavaScript", "responsive design",
                    "website maintenance", "troubleshooting", "user experience"
                ],
                "nice_to_have_skills": [
                    "CMS platforms", "e-commerce", "booking systems", "mobile development"
                ],
                "technical_skills": ["programming languages", "development tools", "web technologies"],
                "soft_skills": ["problem solving", "attention to detail", "logical thinking", "patience"],
                "cultural_fit_keywords": ["technical", "detail-oriented", "problem-solver", "innovative"],
                "disqualifying_factors": ["poor technical skills", "lack of attention to detail", "inflexibility"],
                "experience_indicators": ["web development", "programming", "website design", "IT"],
                "education_preferences": ["computer science", "web development", "information technology"],
                "certifications": ["web development", "programming", "IT certifications"],
                "scoring_weights": {"experience": 0.40, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.05},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 50000, "max": 75000}, "growth_potential": "Mid-Level", "training_requirements": "Technical"
            },

            # ==========================================
            # SPECIALTY SERVICES & INTERNATIONAL
            # ==========================================
            "Cultural Liaison": {
                "must_have_skills": [
                    "cultural sensitivity", "multilingual communication", "guest relations", "cultural programs",
                    "interpretation", "customer service", "conflict resolution", "communication"
                ],
                "nice_to_have_skills": [
                    "multiple languages", "cultural training", "international experience", "tourism"
                ],
                "technical_skills": ["translation tools", "cultural databases", "communication systems"],
                "soft_skills": ["cultural awareness", "empathy", "patience", "adaptability"],
                "cultural_fit_keywords": ["culturally sensitive", "multilingual", "worldly", "inclusive"],
                "disqualifying_factors": ["cultural insensitivity", "language barriers", "poor communication"],
                "experience_indicators": ["cultural work", "international", "tourism", "guest relations"],
                "education_preferences": ["cultural studies", "international relations", "languages", "hospitality"],
                "certifications": ["cultural training", "language certification", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 60000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Pet Concierge": {
                "must_have_skills": [
                    "animal care", "pet services", "customer service", "pet safety", "communication",
                    "organization", "attention to detail", "problem solving"
                ],
                "nice_to_have_skills": [
                    "veterinary knowledge", "pet training", "pet grooming", "animal behavior"
                ],
                "technical_skills": ["pet care equipment", "safety protocols", "booking systems"],
                "soft_skills": ["animal love", "patience", "caring", "reliability"],
                "cultural_fit_keywords": ["animal lover", "caring", "responsible", "service-oriented"],
                "disqualifying_factors": ["animal allergies", "fear of animals", "poor animal handling"],
                "experience_indicators": ["pet care", "animal services", "veterinary", "hospitality"],
                "education_preferences": ["animal science", "veterinary", "hospitality", "biology"],
                "certifications": ["pet care", "animal handling", "first aid", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Specialized"
            },

            "Wine Steward": {
                "must_have_skills": [
                    "wine knowledge", "wine service", "wine pairing", "customer service", "wine storage",
                    "inventory management", "presentation", "communication"
                ],
                "nice_to_have_skills": [
                    "sommelier training", "wine certification", "food pairing", "cellar management"
                ],
                "technical_skills": ["wine preservation", "cellar management", "POS systems"],
                "soft_skills": ["sophistication", "attention to detail", "passion", "communication"],
                "cultural_fit_keywords": ["sophisticated", "knowledgeable", "passionate", "refined"],
                "disqualifying_factors": ["poor wine knowledge", "alcohol problems", "poor presentation"],
                "experience_indicators": ["wine service", "sommelier", "fine dining", "hospitality"],
                "education_preferences": ["hospitality", "culinary", "wine studies", "business"],
                "certifications": ["wine certification", "sommelier", "alcohol service", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Personal Shopper": {
                "must_have_skills": [
                    "personal shopping", "fashion sense", "customer service", "local knowledge",
                    "budget management", "communication", "organization", "trend awareness"
                ],
                "nice_to_have_skills": [
                    "luxury brands", "styling", "cultural knowledge", "multilingual"
                ],
                "technical_skills": ["shopping apps", "budget tools", "communication devices"],
                "soft_skills": ["style sense", "empathy", "patience", "discretion"],
                "cultural_fit_keywords": ["stylish", "trendy", "helpful", "sophisticated"],
                "disqualifying_factors": ["poor fashion sense", "overspending", "poor customer service"],
                "experience_indicators": ["personal shopping", "retail", "fashion", "customer service"],
                "education_preferences": ["fashion", "retail", "business", "hospitality"],
                "certifications": ["personal shopping", "styling", "customer service"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "Art Curator": {
                "must_have_skills": [
                    "art knowledge", "curation", "exhibition planning", "art history", "preservation",
                    "organization", "communication", "aesthetic sense"
                ],
                "nice_to_have_skills": [
                    "museum experience", "gallery management", "art valuation", "cultural programming"
                ],
                "technical_skills": ["preservation techniques", "cataloging systems", "exhibition tools"],
                "soft_skills": ["aesthetic sense", "attention to detail", "cultural awareness", "creativity"],
                "cultural_fit_keywords": ["artistic", "cultured", "sophisticated", "knowledgeable"],
                "disqualifying_factors": ["poor art knowledge", "lack of aesthetic sense", "carelessness"],
                "experience_indicators": ["art curation", "museum", "gallery", "cultural institutions"],
                "education_preferences": ["art history", "museum studies", "fine arts", "cultural studies"],
                "certifications": ["art curation", "museum studies", "preservation"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 42000, "max": 62000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Sustainability Coordinator": {
                "must_have_skills": [
                    "sustainability programs", "environmental management", "waste reduction", "energy efficiency",
                    "green initiatives", "compliance", "reporting", "project management"
                ],
                "nice_to_have_skills": [
                    "LEED certification", "carbon footprint", "renewable energy", "green building"
                ],
                "technical_skills": ["environmental monitoring", "sustainability software", "reporting tools"],
                "soft_skills": ["environmental consciousness", "innovation", "communication", "organization"],
                "cultural_fit_keywords": ["environmentally conscious", "innovative", "responsible", "forward-thinking"],
                "disqualifying_factors": ["lack of environmental awareness", "resistance to change", "poor organization"],
                "experience_indicators": ["sustainability", "environmental", "green programs", "hospitality"],
                "education_preferences": ["environmental science", "sustainability", "business", "hospitality"],
                "certifications": ["sustainability", "LEED", "environmental management", "green hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 45000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Accessibility Coordinator": {
                "must_have_skills": [
                    "accessibility compliance", "ADA knowledge", "guest assistance", "accommodation planning",
                    "disability awareness", "communication", "problem solving", "empathy"
                ],
                "nice_to_have_skills": [
                    "sign language", "assistive technology", "universal design", "accessibility auditing"
                ],
                "technical_skills": ["assistive technology", "accessibility tools", "compliance software"],
                "soft_skills": ["empathy", "patience", "problem solving", "advocacy"],
                "cultural_fit_keywords": ["inclusive", "empathetic", "helpful", "accommodating"],
                "disqualifying_factors": ["lack of empathy", "poor understanding of disabilities", "impatience"],
                "experience_indicators": ["accessibility", "disability services", "ADA compliance", "guest services"],
                "education_preferences": ["disability studies", "social work", "hospitality", "public administration"],
                "certifications": ["ADA compliance", "accessibility", "disability services"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            }
        }
    
    def _build_skill_taxonomy(self) -> Dict[str, List[str]]:
        """Build comprehensive skill taxonomy for semantic matching."""
        return {
            "customer_service": [
                "customer service", "guest relations", "client relations", "customer care",
                "guest satisfaction", "service excellence", "hospitality", "guest experience"
            ],
            "communication": [
                "communication", "verbal communication", "written communication",
                "interpersonal skills", "public speaking", "presentation skills",
                "listening skills", "multilingual", "bilingual"
            ],
            "leadership": [
                "leadership", "team leadership", "management", "supervision",
                "team building", "mentoring", "coaching", "delegation", "motivation"
            ],
            "technical": [
                "computer skills", "software", "systems", "technology", "technical",
                "digital literacy", "database", "applications", "platforms"
            ],
            "hospitality_specific": [
                "hospitality", "hotel", "resort", "restaurant", "food service",
                "accommodation", "tourism", "travel", "leisure", "guest services"
            ],
            "food_service": [
                "culinary", "cooking", "chef", "kitchen", "food preparation",
                "menu planning", "food safety", "restaurant", "catering", "baking"
            ],
            "maintenance": [
                "maintenance", "repair", "HVAC", "plumbing", "electrical",
                "facility management", "preventive maintenance", "troubleshooting"
            ],
            "housekeeping": [
                "housekeeping", "cleaning", "laundry", "room service",
                "sanitation", "hygiene", "facility cleaning", "room maintenance"
            ]
        }
    
    def _setup_patterns(self):
        """Setup spaCy patterns for advanced entity recognition."""
        if not spacy_available or not nlp:
            return
        
        # Email patterns
        email_pattern = [{"TEXT": {"REGEX": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"}}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone patterns
        phone_pattern = [{"TEXT": {"REGEX": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}}]
        self.matcher.add("PHONE", [phone_pattern])
        
        # Experience patterns
        exp_pattern = [{"LOWER": {"IN": ["years", "year"]}}, {"TEXT": "of"}, {"TEXT": "experience"}]
        self.matcher.add("EXPERIENCE", [exp_pattern])
    
    def enhanced_skill_extraction(self, text: str, position: str) -> Dict[str, Any]:
        """Enhanced skill extraction using semantic matching and NLP."""
        skills_found = set()
        confidence_scores: Dict[str, float] = {}
        alias_expansions: List[Dict[str, Any]] = []
        alias_details: Dict[str, List[str]] = {}

        text_lower = text.lower()

        # Get position requirements
        position_data = self.position_intelligence.get(position, {})
        all_required_skills = (
            position_data.get("must_have_skills", []) +
            position_data.get("nice_to_have_skills", []) +
            position_data.get("technical_skills", []) +
            position_data.get("soft_skills", [])
        )

        # Build reverse alias index for fast lookup
        reverse_alias: Dict[str, str] = {}
        for canonical, variants in SKILL_ALIASES.items():  # type: ignore
            for variant in variants:
                reverse_alias[variant.lower()] = canonical

        # Direct skill matching with context + alias normalization
        for skill in all_required_skills:
            skill_lower = skill.lower()
            if skill_lower in text_lower:
                skills_found.add(skill)
                confidence_scores[skill] = 1.0
                context_indicators = [
                    "experience with", "skilled in", "proficient in", "expert in",
                    "knowledge of", "familiar with", "certified in", "trained in"
                ]
                for indicator in context_indicators:
                    if f"{indicator} {skill_lower}" in text_lower:
                        confidence_scores[skill] = min(confidence_scores[skill] + 0.2, 1.0)

        # Alias-based detection: find alias variants even if canonical not explicitly listed
        for variant, canonical in reverse_alias.items():
            if variant in text_lower:
                matching_required = next((s for s in all_required_skills if s.lower() == canonical.lower()), None)
                canonical_key = matching_required or canonical
                if canonical_key not in skills_found:
                    skills_found.add(canonical_key)
                    confidence_scores[canonical_key] = 0.75
                    alias_details.setdefault(canonical_key, []).append(variant)
                    alias_expansions.append({
                        "canonical": canonical_key,
                        "matched_variant": variant,
                        "confidence": 0.75
                    })
                else:
                    if confidence_scores.get(canonical_key, 0) < 0.9:
                        confidence_scores[canonical_key] = min(0.9, confidence_scores.get(canonical_key, 0) + 0.05)
                    alias_details.setdefault(canonical_key, []).append(variant)

        # Semantic skill matching using taxonomy
        for category, related_skills in self.skill_taxonomy.items():
            for skill in related_skills:
                if skill.lower() in text_lower and skill not in skills_found:
                    for req_skill in all_required_skills:
                        if any(term in req_skill.lower() for term in skill.lower().split()):
                            skills_found.add(req_skill)
                            confidence_scores[req_skill] = 0.8
                            break

        # Advanced NLP-based extraction if spaCy available
        if spacy_available and nlp:
            doc = nlp(text)
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                for skill in all_required_skills:
                    if skill.lower() in chunk_text and skill not in skills_found:
                        skills_found.add(skill)
                        confidence_scores[skill] = 0.7

        return {
            "skills": list(skills_found),
            "confidence_scores": confidence_scores,
            "total_skills_found": len(skills_found),
            "alias_expansions": alias_expansions,
            "alias_details": alias_details
        }
    
    def advanced_experience_analysis(self, text: str, position: str) -> Dict[str, Any]:
        """Advanced experience analysis with semantic understanding."""
        analysis = {
            "total_years": 0,
            "relevant_years": 0,
            "has_direct_experience": False,
            "has_related_experience": False,
            "experience_quality": "Unknown",
            "leadership_experience": False,
            "training_experience": False,
            "certifications": [],
            "education_level": "Unknown",
            "normalized_titles": [],
            "raw_title_hits": []
        }
        
        text_lower = text.lower()
        position_data = self.position_intelligence.get(position, {})
        
        # Extract years of experience
        year_patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
            r"(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)",
            r"experience[:\s]*(\d+)\+?\s*years?",
            r"(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:hospitality|hotel|restaurant|food)"
        ]
        
        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            years_found.extend([int(match) for match in matches if match.isdigit()])
        
        if years_found:
            analysis["total_years"] = max(years_found)
        
        # Title normalization pass
        raw_hits = []
        normalized = set()
        for canonical, variants in ROLE_ONTOLOGY.items():
            all_forms = {canonical} | set(variants)
            for form in all_forms:
                if form in text_lower:
                    raw_hits.append(form)
                    normalized.add(canonical)
        analysis["raw_title_hits"] = sorted(set(raw_hits))
        analysis["normalized_titles"] = sorted(normalized)

        # Check for direct experience via indicators or normalized match
        experience_indicators = position_data.get("experience_indicators", [])
        direct = False
        for indicator in experience_indicators:
            if indicator.lower() in text_lower:
                direct = True
                break
        # If searched position maps into ontology
        pos_lower = position.lower()
        for canonical, variants in ROLE_ONTOLOGY.items():
            if pos_lower == canonical or pos_lower in variants:
                if canonical in normalized:
                    direct = True
                    break
        if direct:
            analysis["has_direct_experience"] = True
            analysis["relevant_years"] = max(analysis["relevant_years"], analysis["total_years"])  # heuristic
        
        # Check for leadership experience
        leadership_terms = [
            "manager", "supervisor", "lead", "director", "head", "chief",
            "team lead", "assistant manager", "department head"
        ]
        
        for term in leadership_terms:
            if term in text_lower:
                analysis["leadership_experience"] = True
                break
        
        # Check for training experience
        training_terms = ["training", "trainer", "mentor", "coach", "instructor", "teach"]
        for term in training_terms:
            if term in text_lower:
                analysis["training_experience"] = True
                break
        
        # Extract certifications
        cert_patterns = [
            r"certified\s+(?:in\s+)?([^,.]+)",
            r"certification\s+(?:in\s+)?([^,.]+)",
            r"license\s+(?:in\s+)?([^,.]+)",
            r"diploma\s+(?:in\s+)?([^,.]+)"
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            analysis["certifications"].extend(matches)
        
        # Determine experience quality
        if analysis["has_direct_experience"] and analysis["total_years"] >= 3:
            analysis["experience_quality"] = "Excellent"
        elif analysis["has_direct_experience"] or analysis["total_years"] >= 2:
            analysis["experience_quality"] = "Good"
        elif analysis["total_years"] >= 1:
            analysis["experience_quality"] = "Fair"
        else:
            analysis["experience_quality"] = "Limited"
        
        return analysis
    
    def calculate_enhanced_score(self, candidate: Dict[str, Any], position: str) -> Dict[str, Any]:
        """Calculate intelligent, position-specific candidate score with deep content analysis."""
        position_data = self.position_intelligence.get(position, {})
        if not position_data:
            logger.warning(f"Position '{position}' not found in intelligence database")
            return {"total_score": 0, "breakdown": {}, "recommendation": "Unable to evaluate"}
        
        resume_text = candidate.get("resume_text", "").lower()

        # -------------------------------------------------------------
        # Job Description Keyword & Similarity Bonuses (optional)
        # -------------------------------------------------------------
        jd_keywords = position_data.get("job_description_keywords", []) or []
        jd_bonus_experience = 0.0
        jd_bonus_skills = 0.0
        jd_similarity = None
        matched_jd_terms: List[str] = []
        emb_similarity = None
        if jd_keywords:
            try:
                resume_tokens = set(re.findall(r"\b[a-z]{3,}\b", resume_text))
                kw_set = set(jd_keywords)
                matched_jd_terms = sorted(list(kw_set.intersection(resume_tokens)))
                if matched_jd_terms:
                    coverage = len(matched_jd_terms) / (len(kw_set) + 1e-6)
                    total_bonus = min(0.10, coverage * 0.10)
                    jd_bonus_experience = total_bonus * 0.6
                    jd_bonus_skills = total_bonus * 0.4
            except Exception:
                pass

            # Embedding similarity (optional)
            try:
                jd_raw = ""
                if hasattr(self, "_jd_raw_texts"):
                    jd_raw = self._jd_raw_texts.get(position) or ""  # type: ignore
                if jd_raw and len(jd_raw) > 60:
                    jd_vec = self._embed_text(jd_raw[:5000])
                    res_vec = self._embed_text(candidate.get("resume_text", "")[:5000])
                    if jd_vec and res_vec:
                        emb_similarity = self._embedding_cosine(jd_vec, res_vec)
                        emb_bonus = min(0.04, (emb_similarity or 0) * 0.04)
                        jd_bonus_experience += emb_bonus * 0.5
                        jd_bonus_skills += emb_bonus * 0.5
            except Exception:
                pass

            # TF-IDF similarity bonus
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
                from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
                if hasattr(self, "_jd_raw_texts"):
                    jd_raw = self._jd_raw_texts.get(position) or ""  # type: ignore
                    if jd_raw and len(jd_raw) > 40:
                        vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
                        docs = [jd_raw.lower(), resume_text]
                        tfidf = vectorizer.fit_transform(docs)
                        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                        jd_similarity = float(sim)
                        sim_bonus = min(0.05, sim * 0.05)
                        jd_bonus_experience += sim_bonus * 0.5
                        jd_bonus_skills += sim_bonus * 0.5
            except Exception:
                pass
        
        # Get position-specific requirements
        must_have_skills = position_data.get("must_have_skills", [])
        nice_to_have_skills = position_data.get("nice_to_have_skills", [])
        technical_skills = position_data.get("technical_skills", [])
        experience_indicators = position_data.get("experience_indicators", [])
        education_preferences = position_data.get("education_preferences", [])
        certifications = position_data.get("certifications", [])
        disqualifying_factors = position_data.get("disqualifying_factors", [])
        cultural_fit_keywords = position_data.get("cultural_fit_keywords", [])
        
        # Initialize detailed scoring
        scores = {
            "experience_relevance": 0.0,
            "skills_match": 0.0,
            "education_fit": 0.0,
            "technical_competency": 0.0,
            "cultural_alignment": 0.0,
            "position_specific": 0.0,
            "communication_quality": 0.0,
            "career_progression": 0.0
        }
        
        breakdown = {}
        
        # 1. EXPERIENCE RELEVANCE ANALYSIS (35% weight)
        experience_score = 0.0
        relevant_exp_count = 0
        exp_analysis = candidate.get("experience_analysis", {})
        total_experience_years = exp_analysis.get("total_years", 0)
        
        # Check for direct position experience with context analysis
        position_lower = position.lower()
        for indicator in experience_indicators:
            indicator_lower = indicator.lower()
            if indicator_lower in resume_text:
                # Enhanced context checking
                context_words = ["experience", "worked", "position", "role", "job", "years", "manager", "supervisor"]
                context_found = any(word in resume_text[max(0, resume_text.find(indicator_lower)-50):resume_text.find(indicator_lower)+50] 
                                  for word in context_words)
                if context_found:
                    experience_score += 0.25
                    relevant_exp_count += 1
                else:
                    experience_score += 0.1  # Lower score if no context
        
        # Position-specific keyword analysis with frequency weighting
        position_keywords = {
            # General hospitality
            "hotel": 0.15, "hospitality": 0.15, "resort": 0.15, "guest": 0.1,
            "customer service": 0.12, "team": 0.08, "management": 0.1,
            
            # Position-specific enhancements
            position_lower: 0.3,  # Direct position match gets highest score
        }
        
        # Add position-specific keywords
        if "front desk" in position_lower or "reception" in position_lower:
            position_keywords.update({"check-in": 0.2, "check-out": 0.2, "reservation": 0.15, "pms": 0.15})
        elif "housekeeping" in position_lower:
            position_keywords.update({"cleaning": 0.2, "room": 0.15, "maintenance": 0.1, "laundry": 0.1})
        elif "server" in position_lower or "waiter" in position_lower:
            position_keywords.update({"restaurant": 0.2, "food service": 0.2, "menu": 0.1, "dining": 0.15})
        elif "chef" in position_lower or "cook" in position_lower:
            position_keywords.update({"kitchen": 0.2, "cooking": 0.2, "culinary": 0.2, "food prep": 0.15})
        elif "manager" in position_lower:
            position_keywords.update({"leadership": 0.2, "supervision": 0.15, "budget": 0.1, "operations": 0.15})
        
        for keyword, base_weight in position_keywords.items():
            if keyword in resume_text:
                # Count frequency and context for better scoring
                frequency = resume_text.count(keyword)
                # Bonus for multiple mentions (up to 3x)
                frequency_multiplier = min(1 + (frequency - 1) * 0.3, 2.0)
                experience_score += base_weight * frequency_multiplier
        
        # Experience years analysis with position requirements
        min_years = position_data.get("min_experience_years", 0)
        preferred_years = position_data.get("preferred_experience_years", 3)
        
        if total_experience_years >= preferred_years:
            experience_score += 0.2
        elif total_experience_years >= min_years:
            experience_score += 0.15
        elif total_experience_years > 0:
            experience_score += 0.1
        
        # Leadership and progression indicators
        leadership_terms = ["manager", "supervisor", "lead", "coordinator", "head", "chief", "director"]
        progression_terms = ["promoted", "advanced", "grew", "developed", "improved", "increased"]
        
        leadership_found = sum(1 for term in leadership_terms if term in resume_text)
        progression_found = sum(1 for term in progression_terms if term in resume_text)
        
        if leadership_found > 0:
            experience_score += 0.1 * min(leadership_found, 3)
        if progression_found > 0:
            experience_score += 0.05 * min(progression_found, 2)

        # Apply JD experience bonus
        experience_score += jd_bonus_experience
        # Temporal weighting multiplier
        temporal_multiplier = 1.0
        if candidate.get("timeline"):
            temporal_multiplier = self._temporal_experience_weight(candidate.get("timeline"))
            experience_score *= temporal_multiplier
        scores["experience_relevance"] = min(experience_score, 1.0)
        breakdown["experience"] = {
            "score": scores["experience_relevance"],
            "relevant_positions": relevant_exp_count,
            "total_years": total_experience_years,
            "temporal_multiplier": round(temporal_multiplier,4),
            "meets_minimum": total_experience_years >= min_years,
            "leadership_indicators": leadership_found,
            "progression_indicators": progression_found
        }
        
        # 2. SKILLS MATCH ANALYSIS (30% weight)
        skills_score = 0.0
        must_have_found = 0
        nice_to_have_found = 0
        technical_found = 0
        
        # Must-have skills with context verification
        for skill in must_have_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_text:
                # Check for skill context
                skill_pos = resume_text.find(skill_lower)
                context = resume_text[max(0, skill_pos-30):skill_pos+len(skill_lower)+30]
                
                # Higher score if skill is mentioned with experience context
                if any(word in context for word in ["experience", "skilled", "proficient", "expert", "years"]):
                    skills_score += 0.12
                else:
                    skills_score += 0.08
                must_have_found += 1
        
        # Nice-to-have skills
        for skill in nice_to_have_skills:
            if skill.lower() in resume_text:
                skills_score += 0.06
                nice_to_have_found += 1
        
        # Technical skills with proficiency checking
        for skill in technical_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_text:
                skill_pos = resume_text.find(skill_lower)
                context = resume_text[max(0, skill_pos-40):skill_pos+len(skill_lower)+40]
                
                # Higher score for proficiency indicators
                if any(word in context for word in ["advanced", "expert", "proficient", "certified", "experienced"]):
                    skills_score += 0.1
                else:
                    skills_score += 0.06
                technical_found += 1
        
        # Bonus for comprehensive skill coverage
        total_skills_available = len(must_have_skills) + len(nice_to_have_skills) + len(technical_skills)
        total_skills_found = must_have_found + nice_to_have_found + technical_found
        
        if total_skills_available > 0:
            coverage_ratio = total_skills_found / total_skills_available
            if coverage_ratio > 0.7:
                skills_score += 0.15  # Bonus for high coverage
            elif coverage_ratio > 0.5:
                skills_score += 0.1
        
        # Apply JD skills bonus before capping
        skills_score += jd_bonus_skills
        scores["skills_match"] = min(skills_score, 1.0)
        breakdown["skills"] = {
            "score": scores["skills_match"],
            "must_have_found": must_have_found,
            "must_have_total": len(must_have_skills),
            "nice_to_have_found": nice_to_have_found,
            "technical_found": technical_found,
            "coverage_ratio": total_skills_found / max(total_skills_available, 1)
        }

        # Minor canonical alias coverage bonus (max +0.02 to skills score, applied before final weighting cap)
        try:
            extracted = candidate.get("skills_extraction") or {}
            alias_details = extracted.get("alias_details") or {}
            # If multiple distinct canonical groups hit via aliases, add tiny boost
            distinct_alias_groups = len([k for k,v in alias_details.items() if v])
            if distinct_alias_groups >= 3:
                scores["skills_match"] = min(1.0, scores["skills_match"] + 0.02)
                breakdown.setdefault("skills", {})["alias_coverage_bonus"] = 0.02
            elif distinct_alias_groups == 2:
                scores["skills_match"] = min(1.0, scores["skills_match"] + 0.01)
                breakdown.setdefault("skills", {})["alias_coverage_bonus"] = 0.01
            if alias_details:
                breakdown.setdefault("skills", {})["alias_details"] = alias_details
        except Exception:
            pass

        # Job description enrichment breakdown
        breakdown["job_description"] = {
            "keywords_defined": bool(jd_keywords),
            "keyword_count": len(jd_keywords),
            "matched_keywords": matched_jd_terms,
            "experience_bonus": round(jd_bonus_experience, 4),
            "skills_bonus": round(jd_bonus_skills, 4),
            "similarity": round(jd_similarity, 4) if jd_similarity is not None else None,
            "embedding_similarity": round(emb_similarity,4) if emb_similarity is not None else None
        }
        
        # 3. EDUCATION & CERTIFICATION ANALYSIS (15% weight)
        education_score = 0.0
        
        # General education indicators
        education_terms = ["degree", "bachelor", "master", "diploma", "certificate", "graduate", "university", "college", "education"]
        education_found = sum(1 for term in education_terms if term in resume_text)
        
        if education_found > 0:
            education_score += 0.3
            
            # Check for relevant education
            for pref in education_preferences:
                if pref.lower() in resume_text:
                    education_score += 0.4
                    break
        
        # Check for certifications
        certification_found = 0
        for cert in certifications:
            if cert.lower() in resume_text:
                education_score += 0.2
                certification_found += 1
        
        # Professional development indicators
        development_terms = ["training", "course", "certification", "workshop", "seminar", "license"]
        development_found = sum(1 for term in development_terms if term in resume_text)
        if development_found > 0:
            education_score += 0.1 * min(development_found, 2)
        
        scores["education_fit"] = min(education_score, 1.0)
        breakdown["education"] = {
            "score": scores["education_fit"],
            "has_education": education_found > 0,
            "relevant_field": any(pref.lower() in resume_text for pref in education_preferences),
            "certifications_found": certification_found,
            "professional_development": development_found
        }
        
        # 4. CULTURAL ALIGNMENT & SOFT SKILLS (10% weight)
        cultural_score = 0.0
        
        # Cultural fit keywords with context
        cultural_matches = 0
        for keyword in cultural_fit_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in resume_text:
                # Check if used in positive context
                keyword_pos = resume_text.find(keyword_lower)
                context = resume_text[max(0, keyword_pos-20):keyword_pos+len(keyword_lower)+20]
                
                # Boost if in positive context
                positive_context = any(word in context for word in ["excellent", "strong", "proven", "demonstrated"])
                if positive_context:
                    cultural_score += 0.2
                else:
                    cultural_score += 0.15
                cultural_matches += 1
        
        # Hospitality mindset indicators
        hospitality_mindset = ["guest satisfaction", "customer satisfaction", "service excellence", "team player", 
                              "positive attitude", "professional", "reliable", "dedicated", "passionate"]
        mindset_found = sum(1 for term in hospitality_mindset if term in resume_text)
        if mindset_found > 0:
            cultural_score += 0.1 * min(mindset_found, 3)
        
        scores["cultural_alignment"] = min(cultural_score, 1.0)
        breakdown["cultural"] = {
            "score": scores["cultural_alignment"],
            "keywords_found": cultural_matches,
            "hospitality_mindset": mindset_found
        }
        
        # 5. COMMUNICATION & PROFESSIONALISM (5% weight)
        comm_score = 0.0
        
        # Resume quality indicators
        if len(resume_text) > 300:  # Reasonable detail
            comm_score += 0.3
        
        # Professional language and achievements
        professional_terms = ["responsible for", "achieved", "managed", "developed", "implemented", "improved", 
                             "coordinated", "supervised", "led", "organized", "maintained", "increased", "reduced", "optimized", "accelerated"]
        professional_found = sum(1 for term in professional_terms if term in resume_text)
        comm_score += min(professional_found * 0.05, 0.4)

        # Action verb richness (distinct strong verbs)
        action_verbs = {"led","managed","coordinated","developed","implemented","improved","optimized","achieved","increased","reduced","designed","launched","trained","supervised","streamlined","analyzed"}
        used_verbs = set()
        for v in action_verbs:
            if v in resume_text:
                used_verbs.add(v)
        richness = len(used_verbs)
        verb_bonus = min(0.15, richness * 0.02)  # up to 0.15
        comm_score += verb_bonus

        # Quantified accomplishments (numbers with %, $, +, or k)
        quant_pattern = re.compile(r"(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|usd|\$|k|\/yr)?|\b\d+\+\b)")
        quant_hits = quant_pattern.findall(resume_text)
        quant_count = len(quant_hits)
        quant_bonus = min(0.15, quant_count * 0.015)
        comm_score += quant_bonus
        
        # Language skills (bonus for hospitality)
        language_terms = ["bilingual", "multilingual", "spanish", "french", "languages", "fluent"]
        if any(term in resume_text for term in language_terms):
            comm_score += 0.3
        
        scores["communication_quality"] = min(comm_score, 1.0)
        breakdown["communication"] = {
            "score": scores["communication_quality"],
            "resume_length": len(resume_text),
            "professional_terms": professional_found,
            "multilingual": any(term in resume_text for term in language_terms),
            "action_verb_richness": richness,
            "action_verb_bonus": round(verb_bonus,4),
            "quantified_achievements": quant_count,
            "quantified_bonus": round(quant_bonus,4)
        }
        
    # 6. POSITION-SPECIFIC INTELLIGENCE (5% weight)
        position_score = 0.0
        
        # Direct position title matching with variations
        position_variations = [position.lower()]
        if " " in position.lower():
            position_variations.extend(position.lower().split())
        
        for variation in position_variations:
            if variation in resume_text:
                position_score += 0.4
        
        # Industry-specific terminology
        if "front desk" in position.lower():
            industry_terms = ["check-in", "check-out", "pms", "folio", "reservation", "concierge"]
        elif "housekeeping" in position.lower():
            industry_terms = ["room status", "amenities", "turnover", "inventory", "cleaning protocols"]
        elif "food" in position.lower() or "restaurant" in position.lower():
            industry_terms = ["pos system", "menu knowledge", "food safety", "allergies", "wine pairing"]
        else:
            industry_terms = ["hospitality", "guest services", "customer satisfaction"]
        
        industry_found = sum(1 for term in industry_terms if term in resume_text)
        position_score += min(industry_found * 0.1, 0.4)
        
        scores["position_specific"] = min(position_score, 1.0)
        breakdown["position_specific"] = {
            "score": scores["position_specific"],
            "direct_match": any(var in resume_text for var in position_variations),
            "industry_terms": industry_found
        }
        
        # Establish weights early (can be referenced by plugins)
        weights = {
            "experience_relevance": 0.35,
            "skills_match": 0.30,
            "education_fit": 0.15,
            "cultural_alignment": 0.10,
            "communication_quality": 0.05,
            "position_specific": 0.05
        }
        # Apply overrides from config if present
        weights = self._apply_scoring_overrides(weights)

        # Plugin hook: allow external modules to tweak category scores pre-penalty
        plugin_reports: List[Dict[str, Any]] = []
        try:
            hooks = self._load_plugins()
            if hooks:
                for h in hooks:
                    try:
                        ctx = {
                            "position": position,
                            "candidate": candidate,
                            "scores": dict(scores),
                            "breakdown": breakdown,
                            "weights": dict(weights),
                            "version": SCORING_VERSION,
                        }
                        res = h["fn"](ctx)
                        if isinstance(res, dict):
                            deltas = res.get("category_deltas") or {}
                            # Apply deltas safely
                            for k, dv in deltas.items():
                                if k in scores and isinstance(dv, (int, float)):
                                    scores[k] = float(max(0.0, min(1.0, scores[k] + float(dv))))
                            report = {
                                "id": res.get("id") or h.get("id"),
                                "applied_deltas": {k: float(deltas[k]) for k in deltas.keys() if k in scores},
                                "notes": res.get("notes"),
                            }
                            # Optional post bonus (to be applied after penalties later)
                            post_bonus = res.get("post_bonus")
                            if isinstance(post_bonus, (int, float)):
                                report["post_bonus"] = float(post_bonus)
                            if res.get("details") is not None:
                                report["details"] = res.get("details")
                            plugin_reports.append(report)
                    except Exception as pe:
                        plugin_reports.append({"id": h.get("id"), "error": str(pe)})
        except Exception:
            pass

        # CHECK FOR DISQUALIFYING FACTORS
        disqualification_penalty = 0.0
        disqualified_reasons = []
        
        for factor in disqualifying_factors:
            if factor.lower() in resume_text:
                disqualification_penalty += 0.3
                disqualified_reasons.append(factor)
        
        # CALCULATE FINAL WEIGHTED SCORE

        # Negative domain penalty (soft) – penalize unrelated industry/domain references
        neg_hits = sum(1 for term in NEGATIVE_DOMAIN_TERMS if term in resume_text)
        neg_penalty = min(0.15, neg_hits * 0.04) if neg_hits else 0.0

        total_score = sum(scores[key] * weights[key] for key in weights.keys())
        # Content quality soft penalty
        content_quality = candidate.get("content_quality") or {}
        content_penalty = 0.0
        if content_quality.get("low_information"):
            content_penalty += 0.04
        elif content_quality.get("length_flag") and content_quality.get("diversity_flag"):
            content_penalty += 0.02
        total_score = max(0.0, total_score - disqualification_penalty - neg_penalty - content_penalty)

        # Apply any plugin post bonuses after penalties
        if plugin_reports:
            post_total_bonus = 0.0
            for pr in plugin_reports:
                b = pr.get("post_bonus")
                if isinstance(b, (int, float)):
                    post_total_bonus += float(b)
            if post_total_bonus:
                total_score = float(max(0.0, min(1.0, total_score + post_total_bonus)))

        # Record penalties & weights for transparency
        breakdown["penalties"] = {
            "disqualification_penalty": round(disqualification_penalty, 4),
            "negative_domain_penalty": round(neg_penalty, 4),
            "negative_domain_hits": neg_hits,
            "content_penalty": round(content_penalty, 4)
        }
        breakdown["weights_used"] = weights
        if content_quality:
            breakdown["content_quality"] = content_quality
        if plugin_reports:
            breakdown["plugins"] = plugin_reports
        
        # INTELLIGENT RECOMMENDATIONS
        if total_score >= 0.85:
            recommendation = "EXCEPTIONAL CANDIDATE"
            recommendation_reason = "Outstanding match across all criteria"
        elif total_score >= 0.70:
            recommendation = "HIGHLY RECOMMENDED"
            recommendation_reason = "Excellent fit with strong qualifications"
        elif total_score >= 0.55:
            recommendation = "RECOMMENDED"
            recommendation_reason = "Good candidate with solid experience"
        elif total_score >= 0.40:
            recommendation = "CONSIDER WITH INTERVIEW"
            recommendation_reason = "Potential candidate, interview to assess fit"
        elif total_score >= 0.25:
            recommendation = "MARGINAL CANDIDATE"
            recommendation_reason = "Significant gaps, consider only if limited options"
        else:
            recommendation = "NOT RECOMMENDED"
            recommendation_reason = "Poor fit for this position"
        
        # Enhanced category scores for backward compatibility
        category_scores = {
            "experience": scores["experience_relevance"],
            "skills": scores["skills_match"],
            "cultural_fit": scores["cultural_alignment"],
            "hospitality": (scores["experience_relevance"] + scores["cultural_alignment"]) / 2
        }
        # Optional explanation (lightweight) embedded in breakdown
        if self._explain_mode or candidate.get("explain"):
            weights_used = breakdown.get("weights_used", {})
            contributions = {}
            for k, v in scores.items():
                w = weights_used.get(k, 0.0)
                contributions[k] = {
                    "raw_score": round(v, 4),
                    "weight": round(w, 4),
                    "weighted": round(v * w, 4)
                }
            breakdown["explanation"] = {
                "contributions": contributions,
                "jd": {
                    "matched_terms": matched_jd_terms,
                    "similarity_tfidf": jd_similarity,
                    "similarity_embedding": emb_similarity,
                    "bonus_experience": round(jd_bonus_experience, 4),
                    "bonus_skills": round(jd_bonus_skills, 4)
                },
                "experience_temporal_multiplier": breakdown.get("experience", {}).get("temporal_multiplier"),
                "communication": {
                    "action_verb_richness": breakdown.get("communication", {}).get("action_verb_richness"),
                    "action_verb_bonus": breakdown.get("communication", {}).get("action_verb_bonus"),
                    "quantified_achievements": breakdown.get("communication", {}).get("quantified_achievements"),
                    "quantified_bonus": breakdown.get("communication", {}).get("quantified_bonus"),
                },
                "alias_coverage_bonus": breakdown.get("skills_match", {}).get("alias_coverage_bonus"),
                "penalties": breakdown.get("penalties"),
                "scoring_version": SCORING_VERSION
            }
            if self._debug_emb:
                breakdown["explanation"]["_embedding_debug"] = {
                    "has_model": bool(self._embedding_model),
                    "cache_size": len(self._embed_cache)
                }
            if plugin_reports:
                breakdown["explanation"]["plugins"] = plugin_reports

        # Benchmarks: compute percentiles vs. history and attach to breakdown
        try:
            samples = self._read_score_samples()
            global_vals = [float(s.get("score", 0.0)) for s in samples]
            role_vals = [float(s.get("score", 0.0)) for s in samples if s.get("position") == position]
            gp = self._percentile(global_vals, total_score)
            rp = self._percentile(role_vals, total_score)
            breakdown["benchmark"] = {
                "samples_global": len(global_vals),
                "samples_role": len(role_vals),
                "percentile_global": gp,
                "percentile_role": rp,
            }
        except Exception:
            pass
        
        return {
            "total_score": total_score,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
            "category_scores": category_scores,
            "detailed_scores": scores,
            "breakdown": breakdown,
            "disqualified_reasons": disqualified_reasons,
            "negative_domain_hits": neg_hits,
            "position": position,
            "scoring_methodology": f"Enhanced AI Analysis v{SCORING_VERSION} - Deep Content Analysis",
            "evidence": candidate.get("evidence")
        }
    
    def process_single_resume(self, file_path: Path, position: str) -> Dict[str, Any]:
        """Process a single resume with comprehensive analysis."""
        try:
            logger.info(f"📄 Processing: {file_path.name}")
            
            # Extract text from file
            text = self._extract_text_from_file(file_path)
            # Allow more leniency to avoid dropping all candidates when text extraction is weak
            if not text or len(text.strip()) < 20:
                logger.warning(f"⚠️ Insufficient text extracted from {file_path.name} (len={len(text.strip()) if text else 0})")
                return None

            # Duplicate detection (hash of normalized text)
            norm_for_hash = re.sub(r"\s+", " ", text.lower()).strip()
            file_hash = hashlib.md5(norm_for_hash.encode('utf-8')).hexdigest()
            if file_hash in self._seen_hashes:
                logger.info(f"♻️ Duplicate resume detected (skipping): {file_path.name}")
                return None
            self._seen_hashes.add(file_hash)
            
            # Extract basic information
            candidate_info = self._extract_candidate_info(text)
            sections = self._segment_resume_sections(text)
            skill_analysis = self.enhanced_skill_extraction(text, position)
            experience_analysis = self.advanced_experience_analysis(text, position)
            timeline = self._analyze_career_timeline(text)
            position_data = self.position_intelligence.get(position, {})

            # Content sufficiency assessment
            content_quality = self._assess_content_sufficiency(text, sections)

            # Vocabulary gap update (non-fatal)
            try:
                self._update_vocabulary_gap(text)
            except Exception:
                pass

            # Evidence snippets (non-fatal if fails)
            try:
                evidence = self._collect_evidence_snippets(text, position_data)
            except Exception:
                evidence = None

            candidate_data = {
                "resume_text": text,
                "skill_analysis": skill_analysis,
                "experience_analysis": experience_analysis,
                "sections": sections,
                "timeline": timeline,
                "evidence": evidence,
                "content_quality": content_quality,
                **candidate_info
            }

            # Detect gender
            gender_info = self._detect_gender(text, candidate_info.get("name", "Unknown"))
            
            scoring_result = self.calculate_enhanced_score(candidate_data, position)

            # Append score to history log and refresh benchmark in result (non-fatal on error)
            try:
                self._append_score_sample(position, scoring_result.get("total_score", 0.0))
            except Exception:
                pass

            # Detect explicit role evidence (title or training/certification for the searched position)
            role_evidence_details = self._detect_explicit_role_evidence(text, position)
            
            # Merge scoring_result with extra metadata
            scoring_result.update({
                "sections_detected": list(sections.keys()),
                "explicit_role_evidence": role_evidence_details.get("has_evidence", False),
                "role_evidence_details": role_evidence_details,
                "processed_at": datetime.now().isoformat(),
            })
            # Enrich result with identity/contact and summary fields expected by exporters/UI
            enriched_result = dict(scoring_result)
            enriched_result["candidate_name"] = candidate_info.get("name") or "Unknown"
            # Keep original name field too for compatibility
            enriched_result["name"] = enriched_result["candidate_name"]
            enriched_result["email"] = candidate_info.get("email", "Not found")
            enriched_result["phone"] = candidate_info.get("phone", "Not found")
            enriched_result["location"] = candidate_info.get("location", "Not specified")
            enriched_result["file_name"] = file_path.name
            enriched_result["file_path"] = str(file_path)
            # Preserve full resume text for optional LLM full-review
            enriched_result["resume_text"] = text
            # Flatten key analysis summaries
            enriched_result["skills_found"] = list(skill_analysis.get("skills", []))
            enriched_result["experience_years"] = experience_analysis.get("total_years", 0)
            enriched_result["experience_quality"] = experience_analysis.get("experience_quality", "Unknown")

            logger.info(f"✅ {file_path.name}: {enriched_result['total_score']:.1%} - {enriched_result['recommendation']}")
            return enriched_result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return None

    def _detect_explicit_role_evidence(self, text: str, position: str) -> Dict[str, Any]:
        """Detect explicit evidence that the resume matches the searched role.

        Evidence includes:
        - Exact/alias job titles (e.g., "Head Butler", "Personal Butler", "Butler Supervisor")
        - Training/certifications clearly tied to the role (e.g., "Butler certification", "Butler training")

        Returns dict with keys:
        { has_evidence: bool, exact_title_match: bool, matched_titles: List[str], training_hits: List[str], certification_hits: List[str] }
        """
        try:
            txt = (text or "").lower()
            # Avoid common false positives
            false_positive_blocks = [
                "butler university",  # educational institution unrelated to role
                "butler county",
            ]
            for fp in false_positive_blocks:
                txt = txt.replace(fp, "")

            pos = position.strip()
            pos_lower = pos.lower()

            # Collect role aliases and indicators from position intelligence
            position_data = self.position_intelligence.get(pos, {})
            indicators = set()
            if position_data:
                for key in ("experience_indicators",):
                    for v in position_data.get(key, []) or []:
                        indicators.add(v.lower())

            # Always include the position itself
            indicators.add(pos_lower)

            # Derive core tokens (e.g., for "Head Butler" -> ["head butler", "butler"]) but avoid single-token matches that are too generic unless paired
            core_tokens = [pos_lower]
            parts = [p for p in pos_lower.split() if p]
            if len(parts) > 1:
                # add the last word (often the core role, e.g., "butler") but only use it with context
                core_tokens.append(parts[-1])

            # Title patterns to check
            matched_titles: list[str] = []
            exact_title_match = False
            for phrase in sorted(indicators, key=len, reverse=True):
                if phrase and phrase in txt:
                    matched_titles.append(phrase)
                    if phrase == pos_lower:
                        exact_title_match = True

            # Additional contextual title cues
            contextual_cues = [
                "worked as ", "experience as ", "position: ", "role: ", "title: ", "promoted to ", "served as ", "hired as "
            ]
            context_hit = any(any(f"{cue}{phrase}" in txt for cue in contextual_cues) for phrase in indicators)

            # Training / certification evidence near role tokens
            train_words = ["training", "trained", "diploma", "certificate", "certification", "certified", "course"]
            certification_hits: list[str] = []
            training_hits: list[str] = []

            def _near_role(word: str) -> bool:
                # Simple proximity check around role tokens
                for token in core_tokens:
                    if not token:
                        continue
                    # require both appear within a short window
                    idx = txt.find(token)
                    while idx != -1:
                        start = max(0, idx - 60)
                        end = min(len(txt), idx + len(token) + 60)
                        window = txt[start:end]
                        if word in window and (token in (pos_lower, parts[-1] if parts else token)):
                            return True
                        idx = txt.find(token, idx + 1)
                return False

            # Populate training/cert hits
            for w in train_words:
                if _near_role(w):
                    if "cert" in w:
                        certification_hits.append(w)
                    else:
                        training_hits.append(w)

            has_evidence = bool(exact_title_match or context_hit or matched_titles or certification_hits or training_hits)

            return {
                "has_evidence": has_evidence,
                "exact_title_match": bool(exact_title_match),
                "matched_titles": sorted(set(matched_titles)),
                "training_hits": sorted(set(training_hits)),
                "certification_hits": sorted(set(certification_hits)),
            }
        except Exception:
            return {
                "has_evidence": False,
                "exact_title_match": False,
                "matched_titles": [],
                "training_hits": [],
                "certification_hits": [],
            }

    def _collect_evidence_snippets(self, resume_text: str, position_data: Dict[str, Any], max_per_cat: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Collect concise evidence snippets to justify scoring decisions.

        Returns dict categories: experience, skills, job_description
        Each item: {term, snippet, count}
        """
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", resume_text) if ln.strip()]
        lowered = [l.lower() for l in lines]
        evidence: Dict[str, List[Dict[str, Any]]] = {"experience": [], "skills": [], "job_description": []}

        def add(cat: str, term: str, idx: int):
            if len(evidence[cat]) >= max_per_cat:
                return
            l = lines[idx]
            evidence[cat].append({
                "term": term,
                "snippet": l[:240],
                "count": lowered[idx].count(term.lower())
            })

        for term in position_data.get("experience_indicators", [])[:40]:
            t = term.lower()
            for i, l in enumerate(lowered):
                if t in l:
                    add("experience", term, i)
                    break

        skill_terms = position_data.get("must_have_skills", []) + position_data.get("technical_skills", [])
        seen = set()
        for term in skill_terms[:60]:
            t = term.lower()
            for i, l in enumerate(lowered):
                if t in l and term not in seen:
                    add("skills", term, i)
                    seen.add(term)
                    break

        for term in (position_data.get("job_description_keywords", []) or [])[:60]:
            t = term.lower()
            for i, l in enumerate(lowered):
                if t in l:
                    add("job_description", term, i)
                    break

        return evidence

    def _update_vocabulary_gap(self, text: str, min_freq: int = 2) -> None:
        """Track uncommon tokens to surface emerging terminology.

        Maintains a JSON file 'vocabulary_gap.json' accumulating counts
        of tokens not in a basic hospitality seed list. Lightweight and
        tolerant to failures.
        """
        seed = getattr(self, "_seed_vocab", None)
        if seed is None:
            self._seed_vocab = seed = set([
                "hotel","resort","guest","service","customer","experience","manager","team","food","beverage",
                "kitchen","housekeeping","front","desk","hospitality","cleaning","maintenance","chef","cook","server",
                "bar","restaurant","safety","training","skills","communication","supervisor","leader","operation"
            ])
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", text)]
        counts: Dict[str,int] = {}
        for t in tokens:
            if t not in seed and len(t) < 24:
                counts[t] = counts.get(t,0)+1
        gap = {k:v for k,v in counts.items() if v >= min_freq}
        if not gap:
            return
        vocab_path = Path("vocabulary_gap.json")
        existing: Dict[str,int] = {}
        try:
            if vocab_path.exists():
                existing = json.loads(vocab_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        for k,v in gap.items():
            existing[k] = existing.get(k,0)+v
        try:
            vocab_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Robust multi-engine text extraction with fallbacks and minimal reconciliation."""
        file_ext = file_path.suffix.lower()
        primary_text = ""
        alt_text = ""
        ocr_text = ""
        try:
            if file_ext == '.txt':
                try:
                    primary_text = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    return ""
            elif file_ext == '.pdf':
                # Engine 1: PyPDF2
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            extracted = page.extract_text() or ""
                            primary_text += extracted + "\n"
                except Exception:
                    pass
                # Engine 2: pdfminer.six if weak and not in fast mode
                if len(primary_text.strip()) < 40 and not getattr(self, 'fast_mode', False):
                    try:
                        from pdfminer.high_level import extract_text as pdfminer_extract
                        alt_text = pdfminer_extract(str(file_path)) or ""
                    except Exception:
                        alt_text = ""
                # Engine 3: OCR fallback if enabled and still weak
                if (
                    ocr_available
                    and len((primary_text or '').strip() + (alt_text or '').strip()) < 40
                    and not getattr(self, 'disable_ocr', False)
                    and not getattr(self, 'fast_mode', False)
                ):
                    try:
                        pages = pdf2image.convert_from_path(file_path, dpi=300)
                        max_pages = getattr(self, 'ocr_max_pages', 2) or 2
                        for pg in pages[:max_pages]:
                            try:
                                try:
                                    enhanced = self._enhance_image_for_ocr(pg)
                                except Exception:
                                    enhanced = pg
                                ocr_text += pytesseract.image_to_string(enhanced, lang='eng', config=self._tess_config) + "\n"
                            except Exception:
                                continue
                    except Exception:
                        pass
            elif file_ext in ['.docx', '.doc']:
                # Primary attempt: python-docx for .docx
                try:
                    import docx  # type: ignore
                    if file_ext == '.docx':
                        doc = docx.Document(file_path)
                        for p in doc.paragraphs:
                            primary_text += p.text + "\n"
                    else:
                        primary_text = ""
                except Exception:
                    primary_text = ""
                # Fallback 1: docx2txt (works for .docx)
                if len(primary_text.strip()) < 40 and file_ext == '.docx':
                    try:
                        import docx2txt  # type: ignore
                        alt_text = docx2txt.process(str(file_path)) or ""
                    except Exception:
                        pass
                # Fallback 1b (Windows/.doc only): Use Word COM automation to export to text if available
                if file_ext == '.doc' and len((primary_text + alt_text).strip()) < 40:
                    try:
                        import platform
                        if platform.system().lower().startswith('win'):
                            try:
                                import win32com.client  # type: ignore
                                tmp_txt = self._cache_dir() / (file_path.stem + "_export.txt")
                                word = win32com.client.Dispatch("Word.Application")  # type: ignore
                                word.Visible = False  # type: ignore
                                doc_obj = word.Documents.Open(str(file_path))  # type: ignore
                                # 7 = wdFormatText
                                doc_obj.SaveAs(str(tmp_txt), FileFormat=7)  # type: ignore
                                doc_obj.Close(False)  # type: ignore
                                word.Quit()  # type: ignore
                                try:
                                    alt_text = tmp_txt.read_text(encoding='utf-8', errors='ignore')
                                except Exception:
                                    try:
                                        alt_text = tmp_txt.read_text(errors='ignore')
                                    except Exception:
                                        alt_text = ""
                                try:
                                    tmp_txt.unlink(missing_ok=True)  # type: ignore
                                except Exception:
                                    pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                # Fallback 2: textract (can handle .doc/.docx when dependencies available)
                if len((primary_text + alt_text).strip()) < 40:
                    try:
                        import textract  # type: ignore
                        _bytes = textract.process(str(file_path))
                        if isinstance(_bytes, (bytes, bytearray)):
                            alt_text = _bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        pass
                # Fallback 3: OCR images embedded in DOCX if still weak
                if (
                    file_ext == '.docx'
                    and ocr_available
                    and not getattr(self, 'disable_ocr', False)
                    and not getattr(self, 'fast_mode', False)
                    and len((primary_text + alt_text).strip()) < 40
                ):
                    try:
                        import zipfile
                        with zipfile.ZipFile(file_path, 'r') as z:
                            image_names = [n for n in z.namelist() if n.startswith('word/media/')]
                            max_imgs = getattr(self, 'ocr_max_pages', 2) or 2
                            for name in image_names[:max_imgs]:
                                try:
                                    data = z.read(name)
                                    from io import BytesIO
                                    img = Image.open(BytesIO(data))
                                    try:
                                        enhanced = self._enhance_image_for_ocr(img)
                                    except Exception:
                                        enhanced = img
                                    ocr_text += pytesseract.image_to_string(enhanced, lang='eng', config=self._tess_config) + "\n"
                                except Exception:
                                    continue
                    except Exception:
                        pass
                # Fallback 3: naive RTF stripper if file actually contains RTF content
                if len((primary_text + alt_text).strip()) < 40:
                    try:
                        raw = file_path.read_bytes()
                        # Detect RTF header
                        if raw[:5].decode(errors='ignore').startswith('{\\rtf'):
                            raw_txt = raw.decode('utf-8', errors='ignore')
                            # Remove RTF control words and groups
                            cleaned = re.sub(r'\\pard|\\par', '\n', raw_txt)
                            cleaned = re.sub(r'\\[a-zA-Z]+-?\d*\s?', ' ', cleaned)
                            cleaned = re.sub(r'{[^{}]*}', ' ', cleaned)
                            cleaned = re.sub(r'\s+', ' ', cleaned)
                            alt_text = cleaned
                    except Exception:
                        pass
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'] and ocr_available and not getattr(self, 'disable_ocr', False) and not getattr(self, 'fast_mode', False):
                try:
                    image = Image.open(file_path)
                    try:
                        enhanced = self._enhance_image_for_ocr(image)
                    except Exception:
                        enhanced = image
                    ocr_text = pytesseract.image_to_string(enhanced, lang='eng', config=self._tess_config)
                except Exception:
                    return ""
            else:
                # Unsupported type short-circuit
                return ""

            # Reconcile: pick longest non-empty; merge unique lines if OCR adds data
            candidates = [t for t in [primary_text, alt_text, ocr_text] if t and len(t.strip()) > 0]
            if not candidates:
                return ""
            base = max(candidates, key=lambda x: len(x))
            if ocr_text and ocr_text not in base and len(ocr_text) > 40:
                # append lines not already present
                base_lines = set(base.splitlines())
                new_lines = [ln for ln in ocr_text.splitlines() if ln not in base_lines]
                if new_lines:
                    base += "\n" + "\n".join(new_lines)
            return base.strip()
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path.name}: {e}")
            return ""

    def _assess_content_sufficiency(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Heuristic content sufficiency check used for soft penalties and diagnostics.

        Returns a dict with:
        - token_count: int
        - unique_token_ratio: float (0..1)
        - length_flag: bool (True if too short)
        - diversity_flag: bool (True if vocabulary diversity is low)
        - low_information: bool (True if clearly low-information content)
        - sections_detected: int
        """
        try:
            txt = (text or "").strip()
            if not txt:
                return {"token_count": 0, "unique_token_ratio": 0.0, "length_flag": True, "diversity_flag": True, "low_information": True, "sections_detected": 0}

            tokens = re.findall(r"[A-Za-z]{2,}", txt)
            token_count = len(tokens)
            unique = len(set(t.lower() for t in tokens))
            uniq_ratio = (unique / token_count) if token_count else 0.0

            # Thresholds tuned to be lenient to avoid dropping all candidates
            length_flag = token_count < 120  # short resumes often <120 tokens
            diversity_flag = uniq_ratio < 0.18  # very repetitive
            low_information = token_count < 60 or (token_count < 120 and uniq_ratio < 0.14)

            return {
                "token_count": int(token_count),
                "unique_token_ratio": float(round(uniq_ratio, 4)),
                "length_flag": bool(length_flag),
                "diversity_flag": bool(diversity_flag),
                "low_information": bool(low_information),
                "sections_detected": int(len(sections or {})),
            }
        except Exception:
            return {"token_count": 0, "unique_token_ratio": 0.0, "length_flag": False, "diversity_flag": False, "low_information": False, "sections_detected": 0}
    
    # ---------------- LLM Full-Text Review (Optional) -----------------
    def _get_openai_api_key(self) -> Optional[str]:
        """Resolve OpenAI API key from env, config, or .env without hard-coding secrets.

        Search order:
        1) Environment variable OPENAI_API_KEY
        2) File config/openai_api_key.txt (single-line key)
        3) YAML file config/secrets.yaml (keys: llm.openai_api_key or openai_api_key)
        4) .env file in project root (line OPENAI_API_KEY=...)
        """
        # 1) Env
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key.strip()
        # 2) Text file
        try:
            fp = Path("config") / "openai_api_key.txt"
            if fp.exists():
                txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
                if txt and "PUT_YOUR_KEY_HERE" not in txt:
                    return txt
        except Exception:
            pass
        # 3) YAML secrets
        try:
            yfp = Path("config") / "secrets.yaml"
            if yfp.exists():
                data = yaml.safe_load(yfp.read_text(encoding="utf-8", errors="ignore")) or {}
                if isinstance(data, dict):
                    # Try nested llm key first
                    llm = data.get("llm") or {}
                    if isinstance(llm, dict) and llm.get("openai_api_key"):
                        return str(llm["openai_api_key"]).strip()
                    if data.get("openai_api_key"):
                        return str(data["openai_api_key"]).strip()
        except Exception:
            pass
        # 4) .env simple parse (avoid requiring python-dotenv)
        try:
            envp = Path(".env")
            if envp.exists():
                for line in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.upper().startswith("OPENAI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass
        return None

    def _init_llm_client(self):
        """Initialize OpenAI client if available and key present."""
        api_key = self._get_openai_api_key()
        if not (_OPENAI_OK and api_key):
            return None
        try:
            # Reduce SDK-level retries; we implement our own polite backoff
            kwargs = {"api_key": api_key}
            try:
                kwargs["max_retries"] = int(getattr(self, "_llm_max_retries", 0) or 0)
            except Exception:
                kwargs["max_retries"] = 0
            try:
                # Set a default per-request timeout on the client where supported
                kwargs["timeout"] = int(getattr(self, "_llm_timeout", 45) or 45)
            except Exception:
                pass
            return OpenAI(**kwargs)
        except Exception:
            return None

    def _llm_cache_key(self, payload: Dict[str, Any]) -> str:
        try:
            s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        except Exception:
            s = str(payload)
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _llm_gate_and_call(self, model: str, messages: List[Dict[str, str]], temperature: float, timeout: int):
        """Rate-limit aware wrapper for chat.completions.create with backoff on 429.

        - Enforces requests_per_minute via a moving next-allowed timestamp.
        - Applies a short jitter to avoid thundering herd.
        - On HTTP 429, sleeps cooldown_on_429_secs (+ jitter) and retries once.
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")
        with self._llm_sema:
            now = time.time()
            wait_for = max(0.0, (self._llm_next_allowed_ts or 0.0) - now)
            if wait_for > 0:
                time.sleep(wait_for + random.uniform(0, 0.2))
            # First attempt
            try:
                self._llm_metrics["calls_total"] += 1
                resp = self._llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                )
                self._llm_last_status = 200
                self._llm_next_allowed_ts = time.time() + (self._llm_min_interval or 0.0)
                self._llm_metrics["success_total"] += 1
                return resp
            except Exception as e:
                msg = repr(e)
                is_429 = ("429" in msg) or ("Too Many Requests" in msg)
                if is_429:
                    self._llm_last_status = 429
                    cool = max(1, int(self._llm_cooldown_on_429 or 30))
                    sleep_secs = cool + random.randint(0, min(10, cool))
                    self._llm_metrics["rate_limit_events"] += 1
                    self._llm_metrics["cooldown_seconds"] += sleep_secs
                    logger.info(f"⏳ Rate limited (429). Cooling down for {sleep_secs}s before retry...")
                    if self._llm_show_cooldown_progress:
                        try:
                            for remaining in range(sleep_secs, 0, -1):
                                mins, secs = divmod(remaining, 60)
                                logger.info(f"  ⏳ Cooldown: {mins:02d}:{secs:02d} remaining…")
                                time.sleep(1)
                        except Exception:
                            time.sleep(sleep_secs)
                    else:
                        time.sleep(sleep_secs)
                    try:
                        self._llm_metrics["calls_total"] += 1
                        resp = self._llm_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            timeout=timeout,
                        )
                        self._llm_last_status = 200
                        self._llm_next_allowed_ts = time.time() + (self._llm_min_interval or 0.0)
                        self._llm_metrics["success_total"] += 1
                        return resp
                    except Exception as e2:
                        # Do not disable LLM for rate limits; let caller try fallbacks or move on
                        self._llm_last_status = 429 if ("429" in repr(e2)) else None
                        self._llm_metrics["fail_total"] += 1
                        raise e2
                # For non-429, re-raise to let caller decide
                self._llm_metrics["fail_total"] += 1
                raise

    def _llm_chunk_text(self, text: str) -> List[str]:
        txt = (text or "").strip()
        if not txt:
            return []
        max_chars = max(1000, int(self._llm_chunk_chars))
        if len(txt) <= max_chars:
            return [txt]
        chunks: List[str] = []
        start = 0
        while start < len(txt):
            end = min(len(txt), start + max_chars)
            chunks.append(txt[start:end])
            start = end
        return chunks

    def _build_jd_text(self, position: str) -> str:
        # Prefer raw job description text if available
        try:
            if hasattr(self, "_jd_raw_texts") and isinstance(self._jd_raw_texts, dict):
                jd = self._jd_raw_texts.get(position)
                if jd:
                    return str(jd)[:5000]
        except Exception:
            pass
        # Fallback: synthesize JD text from position intelligence
        pdict = self.position_intelligence.get(position, {})
        parts = [f"Position: {position}"]
        def add(label: str, key: str, cap: int = 30):
            vals = [v for v in pdict.get(key, []) if isinstance(v, str)]
            if vals:
                parts.append(f"{label}: " + ", ".join(vals[:cap]))
        add("Must Have", "must_have_skills")
        add("Nice To Have", "nice_to_have_skills")
        add("Technical", "technical_skills")
        add("Experience Indicators", "experience_indicators")
        return "\n".join(parts)[:5000]

    def _llm_eval_single_chunk(self, chunk: str, position: str) -> Dict[str, Any]:
        """Extract evidence from a single chunk. Best-effort; returns compact JSON dict."""
        if not (self._llm_client and chunk):
            return {}
        sys = (
            "You extract structured evidence from resume text for a hotel role. "
            "Return concise JSON with keys: titles_found (list), skills_found (list), hospitality_terms (list), "
            "years_mentions (list of strings), evidence (list of <=3 short quotes)."
        )
        user = {"position": position, "resume_chunk": chunk[: self._llm_chunk_chars]}
        models_to_try = [self._llm_model] + [m for m in self._llm_fallback_models if m]
        for mdl in models_to_try:
            try:
                resp = self._llm_gate_and_call(
                    model=mdl,
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                    temperature=self._llm_temperature,
                    timeout=self._llm_timeout,
                )
                content = (resp.choices[0].message.content or "") if resp and resp.choices else ""
                try:
                    return json.loads(content)
                except Exception:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        return json.loads(content[start:end+1])
            except Exception as e:
                logger.debug(f"LLM single-chunk eval failed on model {mdl}: {e}")
                continue
        if self._llm_disable_on_error:
            logger.warning("🛑 Disabling LLM full review due to repeated single-chunk errors.")
            self._llm_client = None
            self._llm_full_review_enabled = False
        return {}
        return {}

    def _llm_eval_candidate_full(self, position: str, resume_text: str, jd_text: str) -> Optional[Dict[str, Any]]:
        """Evaluate a candidate by reading the entire resume text. Uses chunking if needed."""
        if not (self._llm_client and resume_text):
            return None

        chunks = self._llm_chunk_text(resume_text)
        # If many chunks, sample evenly across the resume to respect rate limits
        limit = int(getattr(self, "_llm_max_chunks_per_resume", 0) or 0)
        if limit > 0 and len(chunks) > limit:
            selected = []
            step = max(1, int(len(chunks) / limit))
            idx = 0
            while len(selected) < limit and idx < len(chunks):
                selected.append(chunks[idx])
                idx += step
            # Ensure last chunk is represented
            if selected[-1] != chunks[-1]:
                selected[-1] = chunks[-1]
            chunks = selected
        if len(chunks) == 1:
            sys = (
                "You are an expert hospitality recruiter. Read the resume text thoroughly and score "
                "fit for the given position based ONLY on the resume content (word-for-word). "
                "Return strict JSON with keys: score (0-100), reason (<=240 chars)."
            )
            user = {"position": position, "job_description": jd_text, "resume_text": chunks[0]}
            models_to_try = [self._llm_model] + [m for m in self._llm_fallback_models if m]
            for mdl in models_to_try:
                try:
                    resp = self._llm_gate_and_call(
                        model=mdl,
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                        temperature=self._llm_temperature,
                        timeout=self._llm_timeout,
                    )
                    content = (resp.choices[0].message.content or "") if resp and resp.choices else ""
                    data = None
                    try:
                        data = json.loads(content)
                    except Exception:
                        start = content.find("{")
                        end = content.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            data = json.loads(content[start:end+1])
                    if isinstance(data, dict) and "score" in data:
                        data["score"] = float(data.get("score", 0))
                        data["reason"] = str(data.get("reason", ""))[:400]
                        return data
                except Exception as e:
                    logger.debug(f"LLM single-chunk scoring failed on model {mdl}: {e}")
                    continue
            if self._llm_disable_on_error:
                logger.warning("🛑 Disabling LLM full review due to repeated single-chunk scoring errors.")
                self._llm_client = None
                self._llm_full_review_enabled = False
            return None
            return None

        # Multi-chunk: map then reduce (respecting chunk limit above)
        summaries: List[Dict[str, Any]] = []
        for ch in chunks:
            ev = self._llm_eval_single_chunk(ch, position)
            if ev:
                summaries.append({
                    "titles_found": ev.get("titles_found", [])[:5],
                    "skills_found": ev.get("skills_found", [])[:10],
                    "hospitality_terms": ev.get("hospitality_terms", [])[:10],
                    "years_mentions": ev.get("years_mentions", [])[:5],
                    "evidence": ev.get("evidence", [])[:3],
                })

        sys = (
            "You are an expert hospitality recruiter. Aggregate the evidence summaries from ALL resume chunks "
            "to produce a single final evaluation for the position. Assume the summaries cover the entire resume. "
            "Return strict JSON: { score: 0-100, reason: '<=240 chars' }."
        )
        user = {"position": position, "job_description": jd_text, "evidence_summaries": summaries}
        models_to_try = [self._llm_model] + [m for m in self._llm_fallback_models if m]
        for mdl in models_to_try:
            try:
                resp = self._llm_gate_and_call(
                    model=mdl,
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                    temperature=self._llm_temperature,
                    timeout=self._llm_timeout,
                )
                content = (resp.choices[0].message.content or "") if resp and resp.choices else ""
                data = None
                try:
                    data = json.loads(content)
                except Exception:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        data = json.loads(content[start:end+1])
                if isinstance(data, dict) and "score" in data:
                    data["score"] = float(data.get("score", 0))
                    data["reason"] = str(data.get("reason", ""))[:400]
                    return data
            except Exception as e:
                logger.debug(f"LLM reduce step failed on model {mdl}: {e}")
                continue
        if self._llm_disable_on_error:
            logger.warning("🛑 Disabling LLM full review due to repeated reduce-step errors.")
            self._llm_client = None
            self._llm_full_review_enabled = False
        return None
        return None

    def _extract_candidate_info(self, text: str) -> Dict[str, Any]:
        """Extract basic candidate information from resume text."""
        info = {
            "name": "Unknown",
            "email": "Not found",
            "phone": "Not found",
            "location": "Not specified"
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            info["email"] = email_matches[0]
        
        # Extract phone
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                info["phone"] = phone_matches[0]
                break
        
        # Extract name (first non-empty line typically)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Look for a line that looks like a name
            for line in lines[:5]:  # Check first 5 lines
                if (len(line.split()) in [2, 3] and 
                    not '@' in line and 
                    not any(char.isdigit() for char in line) and
                    len(line) < 50):
                    info["name"] = line
                    break
        
        # Extract location
        location_patterns = [
            r'(?:Address|Location|Lives?|Resides?|Based)\s*:?\s*([A-Za-z\s,]+)',
            r'([A-Za-z\s]+,\s*[A-Z]{2}(?:\s+\d{5})?)',  # City, State ZIP
            r'([A-Za-z\s]+,\s*[A-Za-z\s]+)'  # City, Country
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                info["location"] = matches[0].strip()
                break
        
        return info

    def _enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Lightweight enhancement for OCR: grayscale, autocontrast, upscale.

        Avoids heavy thresholding to preserve small fonts. Scales up to ~1600px on the longest side.
        """
        try:
            if img.mode != 'L':
                img = img.convert('L')
            try:
                img = ImageOps.autocontrast(img)
            except Exception:
                pass
            w, h = img.size
            longest = max(w, h)
            target = 1600
            if longest < target:
                scale = target / float(longest)
                new_w, new_h = int(w * scale), int(h * scale)
                resample = getattr(Image, 'LANCZOS', getattr(Image, 'BICUBIC', Image.BILINEAR))
                img = img.resize((new_w, new_h), resample)
        except Exception:
            return img
        return img
    
    def _detect_gender(self, text: str, name: str) -> Dict[str, Any]:
        """Intelligently detect candidate gender from resume content."""
        gender_info = {
            "gender": "Unknown",
            "confidence": 0.0,
            "indicators": []
        }
        
        text_lower = text.lower()
        name_lower = name.lower()
        
        # Common male names (more comprehensive list)
        male_names = {
            'john', 'james', 'robert', 'michael', 'william', 'david', 'richard', 'joseph', 'thomas', 'christopher',
            'charles', 'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua',
            'kenneth', 'kevin', 'brian', 'george', 'timothy', 'ronald', 'jason', 'edward', 'jeffrey', 'ryan',
            'jacob', 'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott', 'brandon',
            'benjamin', 'samuel', 'gregory', 'alexander', 'patrick', 'frank', 'raymond', 'jack', 'dennis', 'jerry',
            'alex', 'jose', 'henry', 'douglas', 'peter', 'zachary', 'noah', 'carl', 'arthur', 'gerald',
            'wayne', 'harold', 'ralph', 'louis', 'philip', 'bobby', 'russell', 'craig', 'alan', 'sean',
            'juan', 'luis', 'carlos', 'miguel', 'antonio', 'angel', 'francisco', 'victor', 'jesus', 'salvador',
            'adam', 'nathan', 'aaron', 'kyle', 'jose', 'manuel', 'edgar', 'fernando', 'mario', 'ricardo'
        }
        
        # Common female names (more comprehensive list)
        female_names = {
            'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
            'nancy', 'lisa', 'betty', 'helen', 'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle',
            'laura', 'sarah', 'kimberly', 'deborah', 'dorothy', 'lisa', 'nancy', 'karen', 'betty', 'helen',
            'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle', 'laura', 'emily', 'kimberly', 'deborah',
            'dorothy', 'amy', 'angela', 'ashley', 'brenda', 'emma', 'olivia', 'cynthia', 'marie', 'janet',
            'catherine', 'frances', 'christine', 'virginia', 'samantha', 'debra', 'rachel', 'carolyn', 'janet',
            'virginia', 'maria', 'heather', 'diane', 'julie', 'joyce', 'victoria', 'kelly', 'christina', 'joan',
            'evelyn', 'lauren', 'judith', 'megan', 'cheryl', 'andrea', 'hannah', 'jacqueline', 'martha', 'gloria',
            'teresa', 'sara', 'janice', 'marie', 'julia', 'kathryn', 'anna', 'rose', 'grace', 'sophia',
            'isabella', 'ava', 'mia', 'charlotte', 'abigail', 'ella', 'madison', 'scarlett', 'victoria', 'aria'
        }
        
        # Gender-specific pronouns and references
        male_pronouns = ['he', 'him', 'his', 'himself', 'mr', 'mr.', 'mister']
        female_pronouns = ['she', 'her', 'hers', 'herself', 'ms', 'ms.', 'mrs', 'mrs.', 'miss', 'missus']
        
        # Professional titles that can indicate gender
        male_titles = ['mr', 'mister', 'sir', 'king', 'lord', 'duke', 'prince', 'baron', 'gentleman']
        female_titles = ['ms', 'mrs', 'miss', 'madam', 'lady', 'queen', 'duchess', 'princess', 'baroness']
        
        # Military/professional gender indicators
        male_military = ['seaman', 'airman', 'fireman', 'policeman', 'businessman', 'salesman', 'chairman']
        female_military = ['seawoman', 'airwoman', 'firewoman', 'policewoman', 'businesswoman', 'saleswoman', 'chairwoman']
        
        # Sports and activities with gender tendencies (careful with stereotypes)
        male_sports = ['football', 'rugby', 'wrestling', 'boxing', 'ice hockey', 'baseball']
        female_sports = ['softball', 'field hockey', 'synchronized swimming', 'rhythmic gymnastics']
        
        confidence_score = 0.0
        indicators = []
        
        # Check first name
        first_name = name_lower.split()[0] if name_lower.split() else ""
        if first_name in male_names:
            confidence_score += 0.7
            indicators.append(f"Male name: {first_name}")
            gender_info["gender"] = "Male"
        elif first_name in female_names:
            confidence_score += 0.7
            indicators.append(f"Female name: {first_name}")
            gender_info["gender"] = "Female"
        
        # Check pronouns in text
        male_pronoun_count = sum(1 for pronoun in male_pronouns if pronoun in text_lower)
        female_pronoun_count = sum(1 for pronoun in female_pronouns if pronoun in text_lower)
        
        if male_pronoun_count > female_pronoun_count and male_pronoun_count > 0:
            confidence_score += 0.3
            indicators.append(f"Male pronouns found: {male_pronoun_count}")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Male"
        elif female_pronoun_count > male_pronoun_count and female_pronoun_count > 0:
            confidence_score += 0.3
            indicators.append(f"Female pronouns found: {female_pronoun_count}")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Female"
        
        # Check titles
        for title in male_titles:
            if title in text_lower:
                confidence_score += 0.5
                indicators.append(f"Male title: {title}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Male"
                break
        
        for title in female_titles:
            if title in text_lower:
                confidence_score += 0.5
                indicators.append(f"Female title: {title}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Female"
                break
        
        # Check professional gender-specific terms
        for term in male_military:
            if term in text_lower:
                confidence_score += 0.2
                indicators.append(f"Male professional term: {term}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Male"
        
        for term in female_military:
            if term in text_lower:
                confidence_score += 0.2
                indicators.append(f"Female professional term: {term}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Female"
        
        # Check for gendered organizations (fraternities vs sororities)
        if any(word in text_lower for word in ['fraternity', 'brotherhood', 'alpha phi alpha', 'kappa alpha psi']):
            confidence_score += 0.3
            indicators.append("Male organization membership")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Male"
        
        if any(word in text_lower for word in ['sorority', 'sisterhood', 'alpha kappa alpha', 'delta sigma theta']):
            confidence_score += 0.3
            indicators.append("Female organization membership")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Female"
        
        # Check for gender-specific life events or contexts
        if any(phrase in text_lower for phrase in ['maternity leave', 'pregnancy', 'maiden name']):
            confidence_score += 0.4
            indicators.append("Female life event reference")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Female"
        
        # Final confidence adjustment
        gender_info["confidence"] = min(confidence_score, 1.0)
        gender_info["indicators"] = indicators
        
        # If confidence is too low, mark as unknown
        if gender_info["confidence"] < 0.3:
            gender_info["gender"] = "Unknown"
        
        return gender_info
    
    def screen_candidates(self, position: str, max_candidates: Optional[int] = None, require_explicit_role: bool = True) -> List[Dict[str, Any]]:
        """Screen all candidates for a specific position."""
        logger.info(f"🎯 Starting candidate screening for: {position}")
        
        # Find resume files
        extensions = ['*.txt', '*.pdf', '*.docx', '*.doc', '*.jpg', '*.jpeg', '*.png']
        resume_files = []
        for ext in extensions:
            resume_files.extend(self.input_dir.glob(ext))
        # Optional cap for performance
        if getattr(self, 'max_files', 0) and len(resume_files) > self.max_files:
            resume_files = resume_files[: self.max_files]
        
        if not resume_files:
            logger.warning(f"📭 No resume files found in {self.input_dir}")
            return []
        
        logger.info(f"📚 Found {len(resume_files)} resume files")
        
        # Process each resume
        candidates = []
        for file_path in resume_files:
            result = self.process_single_resume(file_path, position)
            if result:
                candidates.append(result)

        # Optional strict filter: keep only candidates with explicit role/title/training evidence
        if require_explicit_role:
            with_evidence = [c for c in candidates if c.get("explicit_role_evidence", False)]
            # Only enforce if at least a small fraction have evidence; otherwise keep all to avoid empty results
            if with_evidence and (len(with_evidence) >= max(1, int(0.2 * len(candidates)))):
                logger.info(f"🔎 Strict role filter retained {len(with_evidence)} of {len(candidates)} candidates")
                candidates = with_evidence
            else:
                logger.info("🔎 Strict role filter not enforced (insufficient explicit matches); returning unfiltered results")
        
        # Optional LLM full-text review re-ranking (outside strict filter block)
        if getattr(self, "_llm_full_review_enabled", False) and getattr(self, "_llm_client", None) and candidates:
            logger.info("🧠 Running LLM full-text review for all candidates (ChatGPT)")
            jd_text = self._build_jd_text(position)
            failures = 0
            for c in candidates:
                try:
                    resume_text = c.get("resume_text") or ""
                    if not resume_text:
                        continue
                    payload = {
                        "pos": position,
                        "model": getattr(self, "_llm_model", "gpt-4o-mini"),
                        "text_hash": hashlib.md5(resume_text.encode('utf-8')).hexdigest(),
                        "jd_hash": hashlib.md5(jd_text.encode('utf-8')).hexdigest(),
                    }
                    ck = self._llm_cache_key(payload)
                    result = self._llm_cache.get(ck)
                    if not result:
                        result = self._llm_eval_candidate_full(position, resume_text, jd_text)
                        if result:
                            self._llm_cache[ck] = result
                            try:
                                self._llm_cache_path.write_text(json.dumps(self._llm_cache, ensure_ascii=False, indent=2), encoding='utf-8')
                            except Exception:
                                pass
                    if result:
                        c["llm_full_score"] = float(result.get("score", 0))
                        c["llm_full_reason"] = str(result.get("reason", ""))
                        c["llm_full_read"] = True
                        self._llm_metrics["candidates_scored"] = self._llm_metrics.get("candidates_scored",0) + 1
                    else:
                        # If the last status was 429 (rate limit), don't count as a hard failure
                        if getattr(self, "_llm_last_status", None) == 429:
                            logger.info("⏳ Skipping candidate due to temporary rate limit; will continue with others.")
                            self._llm_metrics["candidates_skipped_429"] = self._llm_metrics.get("candidates_skipped_429",0) + 1
                        else:
                            failures += 1
                except Exception as e:
                    logger.debug(f"LLM full-review error: {e}")
                    if getattr(self, "_llm_last_status", None) == 429:
                        # Don't penalize the run for provider throttling
                        self._llm_metrics["candidates_skipped_429"] = self._llm_metrics.get("candidates_skipped_429",0) + 1
                        pass
                    else:
                        failures += 1
                    continue

            if failures >= max(1, int(0.5 * len(candidates))) and getattr(self, "_llm_disable_on_error", False):
                logger.warning("🛑 Disabling LLM for this run due to repeated request failures.")
                self._llm_client = None
                self._llm_full_review_enabled = False

            # Sort by LLM score if present, otherwise fallback to traditional score
            if any("llm_full_score" in c for c in candidates):
                candidates.sort(key=lambda x: x.get("llm_full_score", 0.0), reverse=True)
            else:
                candidates.sort(key=lambda x: x["total_score"], reverse=True)
        else:
            # Sort by score (highest first)
            candidates.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Limit results if specified
        if max_candidates:
            candidates = candidates[:max_candidates]
        
        logger.info(f"✅ Screening complete: {len(candidates)} candidates processed")
        
        return candidates
    
    def generate_report(self, candidates: List[Dict[str, Any]], position: str) -> str:
        """Generate comprehensive screening report."""
        if not candidates:
            return "No candidates found to analyze."
        
        report = f"""
🏨 HOTEL AI RESUME SCREENER - ENHANCED REPORT
{'='*60}

Position: {position}
Candidates Analyzed: {len(candidates)}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}
TOP CANDIDATES
{'='*60}
"""
        
        for i, candidate in enumerate(candidates[:10], 1):
            score = candidate["total_score"]
            recommendation = candidate["recommendation"]
            
            report += f"""
{i}. {candidate['candidate_name']} - {score:.1%} ({recommendation})
   📧 {candidate['email']} | 📞 {candidate['phone']}
   📍 {candidate['location']}
   💼 Experience: {candidate['experience_years']} years ({candidate['experience_quality']})
   🎯 Skills Found: {len(candidate['skills_found'])} relevant skills
   
   Score Breakdown:
   - Experience: {candidate['category_scores']['experience']:.1%}
   - Skills: {candidate['category_scores']['skills']:.1%}
   - Cultural Fit: {candidate['category_scores']['cultural_fit']:.1%}
   - Hospitality: {candidate['category_scores']['hospitality']:.1%}
   
   Key Skills: {', '.join(candidate['skills_found'][:5])}{'...' if len(candidate['skills_found']) > 5 else ''}
"""
        
        # Add statistics
        scores = [c["total_score"] for c in candidates]
        report += f"""
{'='*60}
SCREENING STATISTICS
{'='*60}

Average Score: {statistics.mean(scores):.1%}
Median Score: {statistics.median(scores):.1%}
Top Score: {max(scores):.1%}
Candidates Above 80%: {sum(1 for s in scores if s >= 0.8)}
Candidates Above 65%: {sum(1 for s in scores if s >= 0.65)}

Recommendation Distribution:
- Highly Recommended: {sum(1 for c in candidates if c['recommendation'] == 'Highly Recommended')}
- Recommended: {sum(1 for c in candidates if c['recommendation'] == 'Recommended')}
- Consider with Interview: {sum(1 for c in candidates if c['recommendation'] == 'Consider with Interview')}
- Not Recommended: {sum(1 for c in candidates if c['recommendation'] == 'Not Recommended')}

{'='*60}
End of Report
{'='*60}
"""

        # Append LLM metrics if present
        try:
            m = getattr(self, "_llm_metrics", None)
            if isinstance(m, dict) and m.get("calls_total") is not None:
                report += f"\nLLM SUMMARY\n{'-'*40}\n"
                report += f"Calls: {m.get('calls_total',0)}\n"
                report += f"Success: {m.get('success_total',0)}\n"
                report += f"Failures: {m.get('fail_total',0)}\n"
                report += f"Rate limit events: {m.get('rate_limit_events',0)}\n"
                report += f"Cooldown seconds: {m.get('cooldown_seconds',0)}\n"
                report += f"Candidates scored: {m.get('candidates_scored',0)}\n"
                report += f"Skipped due to 429: {m.get('candidates_skipped_429',0)}\n"
        except Exception:
            pass
        
        return report
    
    def export_to_excel(self, candidates: List[Dict[str, Any]], position: str) -> str:
        """Export results to Excel with multiple sheets."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screening_results_{position}_{timestamp}.xlsx"
        filepath = self.output_dir / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main results sheet
                main_data = []
                for candidate in candidates:
                    main_data.append({
                        'Rank': candidates.index(candidate) + 1,
                        'Name': candidate['candidate_name'],
                        'Email': candidate['email'],
                        'Phone': candidate['phone'],
                        'Location': candidate['location'],
                        'Total Score': f"{candidate['total_score']:.1%}",
                        'Recommendation': candidate['recommendation'],
                        'Experience Score': f"{candidate['category_scores']['experience']:.1%}",
                        'Skills Score': f"{candidate['category_scores']['skills']:.1%}",
                        'Cultural Fit': f"{candidate['category_scores']['cultural_fit']:.1%}",
                        'Hospitality Score': f"{candidate['category_scores']['hospitality']:.1%}",
                        'Experience Years': candidate['experience_years'],
                        'Experience Quality': candidate['experience_quality'],
                        'Skills Found': len(candidate['skills_found']),
                        'Key Skills': ', '.join(candidate['skills_found'][:3]),
                        'File Name': candidate['file_name'],
                        'LLM Full Score': candidate.get('llm_full_score'),
                        'LLM Reason': candidate.get('llm_full_reason')
                    })
                
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name='Candidate Rankings', index=False)
                
                # Detailed skills sheet
                skills_data = []
                for candidate in candidates:
                    for skill in candidate['skills_found']:
                        skills_data.append({
                            'Candidate': candidate['candidate_name'],
                            'Skill': skill,
                            'Category': 'Technical' if skill in self.position_intelligence.get(position, {}).get('technical_skills', []) else 'General'
                        })
                
                if skills_data:
                    df_skills = pd.DataFrame(skills_data)
                    df_skills.to_excel(writer, sheet_name='Skills Analysis', index=False)
                
                # Position requirements sheet
                position_data = self.position_intelligence.get(position, {})
                req_data = []
                
                for skill_type in ['must_have_skills', 'nice_to_have_skills', 'technical_skills']:
                    for skill in position_data.get(skill_type, []):
                        req_data.append({
                            'Skill': skill,
                            'Type': skill_type.replace('_', ' ').title(),
                            'Candidates with Skill': sum(1 for c in candidates if skill in c['skills_found'])
                        })
                
                if req_data:
                    df_req = pd.DataFrame(req_data)
                    df_req.to_excel(writer, sheet_name='Position Requirements', index=False)

                # LLM summary sheet
                try:
                    m = getattr(self, "_llm_metrics", None)
                    if isinstance(m, dict) and m:
                        llm_rows = [{
                            'Metric': 'Calls', 'Value': m.get('calls_total',0)
                        },{
                            'Metric': 'Success', 'Value': m.get('success_total',0)
                        },{
                            'Metric': 'Failures', 'Value': m.get('fail_total',0)
                        },{
                            'Metric': 'Rate limit events', 'Value': m.get('rate_limit_events',0)
                        },{
                            'Metric': 'Cooldown seconds', 'Value': m.get('cooldown_seconds',0)
                        },{
                            'Metric': 'Candidates scored', 'Value': m.get('candidates_scored',0)
                        },{
                            'Metric': 'Skipped due to 429', 'Value': m.get('candidates_skipped_429',0)
                        }]
                        df_llm = pd.DataFrame(llm_rows)
                        df_llm.to_excel(writer, sheet_name='LLM Summary', index=False)
                except Exception:
                    pass
            
            logger.info(f"📊 Excel report exported: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Excel export failed: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Hotel AI Resume Screener")
    parser.add_argument("--input", "-i", default="input_resumes", help="Input directory with resumes")
    parser.add_argument("--output", "-o", default="screening_results", help="Output directory for results")
    parser.add_argument("--position", "-p", required=True, help="Position to screen for")
    parser.add_argument("--max-candidates", "-m", type=int, help="Maximum number of candidates to return")
    parser.add_argument("--thoroughness", choices=["fast","balanced","full"], help="Control LLM chunking and pacing for this run")
    parser.add_argument("--export-excel", "-e", action="store_true", help="Export results to Excel")
    
    args = parser.parse_args()
    
    # Initialize screener
    screener = EnhancedHotelAIScreener(args.input, args.output)
    # Apply thoroughness mode from CLI (override env)
    if getattr(args, 'thoroughness', None):
        os.environ['HAT_THOROUGHNESS'] = args.thoroughness
    
    # Screen candidates
    candidates = screener.screen_candidates(args.position, args.max_candidates)
    
    if not candidates:
        print("❌ No candidates found or processed successfully.")
        return
    
    # Generate report
    report = screener.generate_report(candidates, args.position)
    print(report)
    
    # Save text report
    report_file = screener.output_dir / f"screening_report_{args.position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved: {report_file}")
    
    # Export to Excel if requested
    if args.export_excel:
        excel_file = screener.export_to_excel(candidates, args.position)
        if excel_file:
            print(f"📊 Excel report saved: {excel_file}")


if __name__ == "__main__":
    main()
