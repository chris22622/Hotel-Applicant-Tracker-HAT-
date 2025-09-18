#!/usr/bin/env python3
"""
LLM Quick Pick CLI

Usage:
  python llm_quick_pick.py -p "Front Desk Agent" -t 10

It scans `input_resumes/`, scores locally, then (optionally) uses ChatGPT to re-rank
the top candidates and prints/saves the final top-N.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from enhanced_ai_screener import EnhancedHotelAIScreener
except Exception as e:  # pragma: no cover - CLI guard
    print(f"ERROR: Cannot import EnhancedHotelAIScreener: {e}")
    raise SystemExit(1)

# Optional LLM reranker
try:
    from llm_reranker import LLMReranker
except Exception:
    LLMReranker = None  # type: ignore


def _export(results: List[Dict[str, Any]], position: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Save JSON
    (out_dir / f"quick_pick_{position}_{ts}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    # Save Excel via screener helper
    try:
        scr = EnhancedHotelAIScreener()
        scr.output_dir = out_dir
        scr.export_to_excel(results, position)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM Quick Pick: find top candidates by position")
    ap.add_argument("-p", "--position", required=True, help="Position title to hire for")
    ap.add_argument("-t", "--top", type=int, required=True, help="How many final candidates to return")
    ap.add_argument("-i", "--input", default="input_resumes", help="Folder with resumes")
    ap.add_argument("-o", "--output", default="screening_results", help="Folder to save outputs")
    ap.add_argument("--strict", action="store_true", help="Require explicit role/title/training evidence when enough present")
    ap.add_argument("--threshold", type=float, default=0.3, help="Minimum local score to consider (0-1)")
    ap.add_argument("--pool", type=int, default=None, help="How many candidates to feed into LLM (default: same as --top)")
    # LLM options
    ap.add_argument("--llm", action="store_true", help="Enable LLM re-ranking (requires OPENAI_API_KEY)")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    ap.add_argument("--temp", type=float, default=0.1, help="Sampling temperature")
    ap.add_argument("--timeout", type=int, default=30, help="Model timeout seconds")
    ap.add_argument("--no-redact", action="store_true", help="Do not redact PII in context sent to LLM")
    args = ap.parse_args()

    position = args.position.strip()
    top_n = max(1, int(args.top))
    pool_n = max(1, int(args.pool)) if args.pool else top_n

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: input folder not found: {input_dir}")
        raise SystemExit(2)

    # Initialize screener and run local scoring
    screener = EnhancedHotelAIScreener(str(input_dir), str(output_dir))
    print(f"\n🎯 Screening for: {position}")
    print(f"📁 Folder: {input_dir}")
    results = screener.screen_candidates(position, max_candidates=None, require_explicit_role=args.strict)

    if not results:
        print("No candidates found.")
        raise SystemExit(0)

    # Apply local threshold
    results = [r for r in results if float(r.get('total_score', 0)) >= float(args.threshold)]
    if not results:
        print("No candidates meet the threshold.")
        raise SystemExit(0)

    # If LLM enabled and available, rerank the top pool
    if args.llm and LLMReranker is not None and os.environ.get("OPENAI_API_KEY"):
        pool = results[: pool_n]
        # JD text from screener if available
        try:
            jd_text = (getattr(screener, "_jd_raw_texts", {}) or {}).get(position, "")
        except Exception:
            jd_text = ""
        reranker = LLMReranker(
            model=str(args.model),
            temperature=float(args.temp),
            timeout_seconds=int(args.timeout),
            redact_pii=not bool(args.no_redact),
        )
        try:
            pool = reranker.rerank(position, jd_text, pool)
        except Exception as e:
            print(f"LLM re-ranking skipped due to error: {e}")
        results = pool + results[pool_n:]
    else:
        if args.llm and not os.environ.get("OPENAI_API_KEY"):
            print("⚠️ LLM requested but OPENAI_API_KEY not set. Running with local ranking only.")
        elif args.llm and LLMReranker is None:
            print("⚠️ LLM requested but llm_reranker not available. Running with local ranking only.")

    # Final top-N slice
    results = results[: top_n]

    # Print concise summary
    print("\n🏆 Top Candidates:")
    for i, c in enumerate(results, 1):
        name = c.get('candidate_name', 'Unknown')
        score = float(c.get('total_score', 0)) * 100
        rec = c.get('recommendation', '')
        line = f"{i}. {name} — {score:.1f}% ({rec})"
        if c.get('llm_rank') or c.get('llm_score'):
            line += f" | LLM rank {c.get('llm_rank','—')}, LLM score {c.get('llm_score','—')}"
        print(line)
        if c.get('llm_reason'):
            print(f"   🤖 {c.get('llm_reason')}")

    # Save outputs
    _export(results, position, output_dir)
    print(f"\n💾 Saved JSON/Excel to: {output_dir}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
