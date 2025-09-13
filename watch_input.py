"""
Watch mode: poll input_resumes for new files and process automatically.
Usage:
  python watch_input.py --position "Front Desk Agent" --interval 5
"""
import time
import argparse
from pathlib import Path
from enhanced_ai_screener import EnhancedHotelAIScreener


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--position', required=True, help='Target position to screen for')
    ap.add_argument('--input', default='input_resumes', help='Input directory to watch')
    ap.add_argument('--interval', type=int, default=5, help='Polling interval seconds')
    ap.add_argument('--explain', action='store_true', help='Enable explain mode')
    args = ap.parse_args()

    input_dir = Path(args.input)
    input_dir.mkdir(exist_ok=True)

    screener = EnhancedHotelAIScreener(input_dir=str(input_dir))
    if args.explain:
        screener.set_explain_mode(True)

    seen = set(p.name for p in input_dir.iterdir() if p.is_file())
    print(f"Watching {input_dir.resolve()} for new files (position={args.position})...")
    while True:
        try:
            for p in input_dir.iterdir():
                if not p.is_file():
                    continue
                if p.name in seen:
                    continue
                seen.add(p.name)
                result = screener.process_single_resume(p, args.position)
                if result:
                    print(f"Processed: {p.name} -> {result['total_score']:.1%} {result['recommendation']}")
        except KeyboardInterrupt:
            print("Stopping watch mode...")
            break
        except Exception as e:
            print(f"Error during watch: {e}")
        time.sleep(max(1, args.interval))


if __name__ == '__main__':
    main()
