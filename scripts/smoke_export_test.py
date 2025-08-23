import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Ensure workspace root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports from app
from enhanced_ai_screener import EnhancedHotelAIScreener
from enhanced_streamlit_app import export_results_to_excel, export_results_to_csv


def main():
    input_dir = ROOT / "input_resumes"
    out_dir = ROOT / "screening_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    screener = EnhancedHotelAIScreener(str(input_dir))
    available = list(screener.position_intelligence.keys())

    # Prefer a common role if available
    preferred = [
        "Front Desk Agent",
        "Guest Services Agent",
        "Housekeeping Supervisor",
        "Bartender",
        "Server"
    ]
    selected = next((p for p in preferred if p in available), available[0] if available else None)
    if not selected:
        print("No positions found.")
        sys.exit(1)

    print(f"Position: {selected}")
    results = screener.screen_candidates(selected, max_candidates=25)
    print(f"Candidates: {len(results)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out_dir / f"smoke_{selected}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"JSON -> {json_path}")

    # Text report
    try:
        report = screener.generate_report(results, selected)
    except Exception as e:
        report = f"Report generation failed: {e}"
    txt_path = out_dir / f"smoke_{selected}_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Text -> {txt_path}")

    # CSV
    csv_bytes = export_results_to_csv(results, selected)
    if csv_bytes:
        csv_path = out_dir / f"smoke_{selected}_{ts}.csv"
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)
        print(f"CSV -> {csv_path}")
    else:
        print("CSV export returned no data.")

    # Excel
    try:
        excel_data = export_results_to_excel(results, selected)
        if excel_data:
            xlsx_path = out_dir / f"smoke_{selected}_{ts}.xlsx"
            with open(xlsx_path, "wb") as f:
                f.write(excel_data.getvalue())
            print(f"Excel -> {xlsx_path}")
        else:
            print("Excel export returned no data.")
    except Exception as e:
        print(f"Excel export failed: {e}")


if __name__ == "__main__":
    main()
