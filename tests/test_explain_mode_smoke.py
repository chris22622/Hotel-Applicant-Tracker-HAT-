"""Lightweight smoke test for explain mode scoring.
Run directly: python -m tests.test_explain_mode_smoke
"""
import sys, os
sys.path.insert(0, os.getcwd())  # ensure root import
from enhanced_ai_screener import EnhancedHotelAIScreener


def build_dummy_candidate(text: str):
    return {
        'resume_text': text,
        'experience_analysis': {'total_years':5,'has_direct_experience':True,'relevant_years':5},
        'content_quality': {'token_count':200,'unique_token_ratio':0.6},
        'skills_extraction': {
            'all_skills':['customer service','pms','reservations','check-in','check-out','team leadership','training','communication','problem solving'],
            'normalized_skills':['customer service','pms','reservations','check-in','check-out','team leadership','training','communication','problem solving'],
            'alias_details':{}
        },
        'evidence': None
    }


def main():
    screener = EnhancedHotelAIScreener()
    screener.set_explain_mode(True)
    # Pick a position that exists
    positions = list(screener.position_intelligence.keys())
    if not positions:
        print("No positions loaded; aborting.")
        return
    position = None
    # Prefer a front desk role if available
    for p in positions:
        if 'front' in p.lower():
            position = p
            break
    if position is None:
        position = positions[0]

    fake_resume = (
        "John Doe\nFront Desk Agent with 5 years of hotel and hospitality experience. "
        "Expert in PMS systems, guest services, reservations, check-in and check-out procedures. "
        "Trained in concierge support and customer satisfaction. Holds certifications in hospitality management."
    )
    candidate = build_dummy_candidate(fake_resume)
    result = screener.calculate_enhanced_score(candidate, position)

    print(f"Position: {position}")
    print(f"Total Score: {result['total_score']:.2%}")
    print(f"Recommendation: {result['recommendation']}")
    explanation = result.get('breakdown', {}).get('explanation')
    if explanation:
        print("Explanation contributions (top 3 weighted):")
        contribs = explanation['contributions']
        top = sorted(contribs.items(), key=lambda kv: kv[1]['weighted'], reverse=True)[:3]
        for name, info in top:
            print(f"  {name}: raw={info['raw_score']}, weight={info['weight']}, weighted={info['weighted']}")
        print("JD matched terms:", explanation['jd']['matched_terms'])
    else:
        print("Explanation not present - check explain mode.")


if __name__ == '__main__':
    main()
