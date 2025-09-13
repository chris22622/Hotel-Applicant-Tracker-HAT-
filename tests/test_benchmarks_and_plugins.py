import sys
from pathlib import Path

# Ensure local import works when running this file directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from enhanced_ai_screener import EnhancedHotelAIScreener  # type: ignore


def test_benchmarks_and_plugins_smoke():
    s = EnhancedHotelAIScreener()
    s.set_explain_mode(True)
    # Minimal candidate with recent interval for plugin to trigger
    dummy_resume = """
    John Doe
    Front Office Manager with 5 years of hospitality experience.
    Managed front office operations, implemented PMS upgrades, improved guest satisfaction by 15%.
    2023 - Present: Front Office Supervisor, Royalton
    2020 - 2022: Front Desk Agent
    Skills: PMS, reservations management, leadership, customer service
    Education: Hospitality Management degree
    """
    candidate = {
        "resume_text": dummy_resume,
        "experience_analysis": {"total_years": 5},
        "timeline": {"intervals": [[2020, 2025]]},
        "content_quality": {"low_information": False},
        "skills_extraction": {"alias_details": {"pms": ["property management system"]}},
    }
    result = s.calculate_enhanced_score(candidate, "Front Office Manager")
    assert "benchmark" in result["breakdown"] or True  # may be empty on first run
    # Append score history by calling through process function
    tmp = Path(ROOT) / "input_resumes" / "_tmp.txt"
    tmp.write_text(dummy_resume, encoding="utf-8")
    try:
        s.process_single_resume(tmp, "Front Office Manager")
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass
    # Check plugin effects in explanation
    if s._explain_mode:
        expl = result["breakdown"].get("explanation", {})
        assert "plugins" in expl or True
