"""
Example plugin: RecencyBoost

Contract:
- augment_score(ctx: dict) -> dict
  ctx keys: position, candidate, scores, breakdown, weights, version
  return keys:
    - id: optional plugin id display
    - category_deltas: dict of {category_name: delta_float in [-1,1]}
    - post_bonus: optional float 0..0.05 to be added post-penalties
    - notes/details: any plugin-specific info

Behavior:
- If the candidate has experience in the last 2 years (from timeline), add a tiny boost to experience_relevance and communication_quality.
- Add a small post_bonus when both experience and skills are strong.
"""
from __future__ import annotations
from typing import Dict, Any
from datetime import datetime

def augment_score(ctx: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cand = ctx.get("candidate", {}) or {}
        timeline = cand.get("timeline") or {}
        intervals = timeline.get("intervals") or []
        now = datetime.now().year
        recent = any(e >= now - 2 for _, e in intervals) if intervals else False
        deltas = {}
        notes = {}
        if recent:
            deltas["experience_relevance"] = 0.02
            deltas["communication_quality"] = 0.01
            notes["recent_experience"] = True
        scores = ctx.get("scores") or {}
        post_bonus = 0.0
        if scores.get("experience_relevance", 0) > 0.6 and scores.get("skills_match", 0) > 0.6:
            post_bonus = 0.01
        return {
            "id": "recency_boost",
            "category_deltas": deltas,
            "post_bonus": post_bonus,
            "notes": notes,
        }
    except Exception as e:
        return {"id": "recency_boost", "error": str(e)}
