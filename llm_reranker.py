"""
Optional LLM-based re-ranking for top candidates.

Safely integrates with the Streamlit app without breaking offline/local operation.
If the OpenAI client or API key isn't available, this becomes a no-op.
"""
from __future__ import annotations

import os
import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import OpenAI client lazily; keep feature optional
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


def _redact_pii(text: str) -> str:
    """Redact common PII like emails and phone numbers from text."""
    if not text:
        return text
    try:
        # Email
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
        # Phone numbers (simple patterns)
        text = re.sub(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b", "[REDACTED_PHONE]", text)
        return text
    except Exception:
        return text


class LLMReranker:
    """Encapsulates LLM ranking invocation with caching and safe fallbacks."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        timeout_seconds: int = 30,
        redact_pii: bool = True,
        cache_path: Path | str = Path("var") / "llm_cache.json",
    ) -> None:
        self.model = model
        self.temperature = float(max(0.0, min(temperature, 1.0)))
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.redact_pii = redact_pii
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._enabled = False
        self._init_client()

        # Load cache file if present
        self._cache: Dict[str, Any] = {}
        try:
            if self.cache_path.exists():
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}

    def _init_client(self) -> None:
        """Initialize OpenAI client if available and API key is set."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if _OPENAI_AVAILABLE and api_key:
            try:
                self._client = OpenAI(api_key=api_key)
                self._enabled = True
            except Exception:
                # Fail closed; keep disabled
                self._client = None
                self._enabled = False
        else:
            self._client = None
            self._enabled = False

    @staticmethod
    def _hash_key(data: Any) -> str:
        """Stable hash for caching based on JSON serialization."""
        try:
            payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        except Exception:
            payload = str(data)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _summarize_candidate(c: Dict[str, Any], max_skills: int = 8) -> Dict[str, Any]:
        """Produce a compact summary for LLM consumption."""
        name = c.get("candidate_name") or c.get("name") or "Candidate"
        score = float(c.get("total_score", 0.0))
        rec = c.get("recommendation", "")
        exp_years = c.get("experience_years", None)
        exp_quality = c.get("experience_quality", None)
        skills = c.get("skills_found") or []
        if not isinstance(skills, list):
            skills = []
        skills = [str(s) for s in skills][:max_skills]
        # Pull a couple of evidence lines if available
        ev = c.get("evidence") or {}
        ev_lines: List[str] = []
        try:
            for cat in ("experience", "skills", "job_description"):
                for item in (ev.get(cat) or [])[:2]:
                    snip = str(item.get("snippet") or "")[:180]
                    if snip:
                        ev_lines.append(snip)
                    if len(ev_lines) >= 4:
                        break
                if len(ev_lines) >= 4:
                    break
        except Exception:
            ev_lines = []
        text = " | ".join(ev_lines)
        return {
            "name": name,
            "score": round(score * 100, 1),
            "recommendation": rec,
            "experience_years": exp_years,
            "experience_quality": exp_quality,
            "skills": skills,
            "evidence": text,
        }

    def _build_prompt(self, position: str, jd_text: str, cand_summaries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Construct chat messages and cache key payload."""
        # Lightly redact PII in evidence text only (names/skills/years are fine)
        summaries = []
        for s in cand_summaries:
            s2 = dict(s)
            if self.redact_pii and s2.get("evidence"):
                s2["evidence"] = _redact_pii(str(s2["evidence"]))
            summaries.append(s2)

        jd_snippet = (jd_text or "").strip()
        if self.redact_pii:
            jd_snippet = _redact_pii(jd_snippet)
        jd_snippet = jd_snippet[:2000]

        sys = (
            "You are an expert hospitality recruiter for a luxury resort. "
            "Rank candidates for the given position using the job description and concise candidate summaries. "
            "Be decisive and prefer evidence tied to the role’s must-have skills and hospitality-specific experience."
        )
        user = {
            "position": position,
            "job_description": jd_snippet,
            "candidates": summaries,
            "instructions": (
                "Return strict JSON with a 'ranked' array. Each item must include: "
                "index (0-based input order), score (0-100), and reason (<= 240 chars). "
                "Do not include any fields besides 'ranked'."
            ),
            "output_schema": {"ranked": [{"index": 0, "score": 87, "reason": "..."}]},
        }
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        cache_payload = {
            "position": position,
            "jd": jd_snippet,
            "candidates": summaries,
            "model": self.model,
        }
        return messages, cache_payload

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        # Try direct load
        try:
            return json.loads(content)
        except Exception:
            pass
        # Try to extract the largest JSON object within
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start : end + 1])
        except Exception:
            pass
        # Try fenced code blocks
        try:
            m = re.search(r"```json\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
            if m:
                return json.loads(m.group(1))
        except Exception:
            pass
        return None

    def rerank(
        self,
        position: str,
        jd_text: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return a new ordered list of candidates with LLM annotations.

        If disabled/unavailable, returns the original list.
        """
        if not candidates or not self._enabled or not self._client:
            return candidates

        # Build summaries and prompt
        cand_summaries = [self._summarize_candidate(c) for c in candidates]
        messages, cache_payload = self._build_prompt(position, jd_text, cand_summaries)
        cache_key = self._hash_key(cache_payload)

        # Cache hit
        if cache_key in self._cache:
            try:
                ranked = self._cache[cache_key]["ranked"]
                return self._apply_ranking(candidates, ranked)
            except Exception:
                pass

        try:
            # Use chat.completions with JSON-like guidance in prompt
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
            )
            content = (resp.choices[0].message.content or "") if resp and resp.choices else ""
        except Exception:
            content = ""

        data = self._parse_json(content)
        if not data or not isinstance(data.get("ranked"), list):
            # No valid JSON; return original
            return candidates

        # Persist cache (best effort)
        try:
            self._cache[cache_key] = {"ranked": data["ranked"]}
            self._save_cache()
        except Exception:
            pass

        return self._apply_ranking(candidates, data["ranked"])

    @staticmethod
    def _apply_ranking(candidates: List[Dict[str, Any]], ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply LLM-provided ranking to the list, annotating reasons and scores."""
        n = len(candidates)
        # Build map from index to (rank_position, score, reason)
        annotated: Dict[int, Tuple[int, float, str]] = {}
        for i, item in enumerate(ranked):
            try:
                idx = int(item.get("index"))
                if 0 <= idx < n and idx not in annotated:
                    score = float(item.get("score", 0.0))
                    reason = str(item.get("reason", ""))[:400]
                    annotated[idx] = (i + 1, score, reason)
            except Exception:
                continue

        # Create new list: first those with annotations in order, then the rest
        ordered: List[Dict[str, Any]] = []
        used = set()
        for idx, (rank_pos, score, reason) in sorted(annotated.items(), key=lambda x: x[1][0]):
            c = dict(candidates[idx])
            c["llm_rank"] = rank_pos
            c["llm_score"] = score
            c["llm_reason"] = reason
            ordered.append(c)
            used.add(idx)
        for idx, c in enumerate(candidates):
            if idx not in used:
                ordered.append(c)
        return ordered
