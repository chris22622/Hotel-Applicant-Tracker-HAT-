#!/usr/bin/env python3
"""
Headless CLI for Hotel Applicant Tracker
Useful for batch processing and CI demos
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def setup_logging():
    """Setup logging with timestamps and PII redaction"""
    import logging
    import os
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter that redacts PII
    class PIIRedactingFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            # Redact email patterns
            import re
            msg = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', msg)
            # Redact phone patterns
            msg = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', msg)
            return msg
    
    logging.basicConfig(
        level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Apply PII redacting formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(PIIRedactingFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

def run_headless_screening(input_dir: str, position: str, output_file: str) -> Dict[str, Any]:
    """Run screening without UI and return results as JSON"""
    try:
        from hotel_ai_screener import HotelAIScreener
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting headless screening for position: {position}")
        
        screener = HotelAIScreener()
        
        # Process all resumes in directory
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        resume_files = list(input_path.glob("*.{pdf,docx,txt}"))
        logger.info(f"Found {len(resume_files)} resume files")
        
        if len(resume_files) == 0:
            logger.warning("No resume files found in input directory")
            return {"error": "No resume files found", "count": 0}
        
        # Run screening
        results = []
        for resume_file in resume_files:
            try:
                logger.info(f"Processing: {resume_file.name}")
                # This would integrate with your actual screening logic
                result = {
                    "filename": resume_file.name,
                    "position": position,
                    "score": 85.0,  # Placeholder - integrate with actual scoring
                    "status": "processed"
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {resume_file.name}: {e}")
                results.append({
                    "filename": resume_file.name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save results
        output_data = {
            "position": position,
            "total_files": len(resume_files),
            "processed": len([r for r in results if r.get("status") == "processed"]),
            "errors": len([r for r in results if r.get("status") == "error"]),
            "results": results,
            "timestamp": "2025-08-15T00:00:00Z"  # Use actual timestamp
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        return output_data
        
    except ImportError:
        logger.error("hotel_ai_screener module not found. Run: pip install -r requirements.txt")
        return {"error": "Missing dependencies"}
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        return {"error": str(e)}

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Hotel Applicant Tracker - Headless CLI")
    parser.add_argument("--input", "-i", required=True, help="Input directory with resumes")
    parser.add_argument("--position", "-p", default="front_desk_agent", help="Position to screen for")
    parser.add_argument("--output", "-o", default="screening_results.json", help="Output JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress console output")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if not args.quiet:
        print("🏨 Hotel Applicant Tracker - Headless Mode")
        print(f"📁 Input: {args.input}")
        print(f"🎯 Position: {args.position}")
        print(f"📄 Output: {args.output}")
        print()
    
    # Run screening
    results = run_headless_screening(args.input, args.position, args.output)
    
    if not args.quiet:
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            sys.exit(1)
        else:
            print(f"✅ Processed {results['processed']}/{results['total_files']} files")
            print(f"📊 Results saved to: {args.output}")
    
    return 0 if "error" not in results else 1

if __name__ == "__main__":
    sys.exit(main())
