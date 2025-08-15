#!/usr/bin/env python3
"""
Royalton Resort AI Resume Screener
One-click intelligent candidate selection with AI recommendations
"""

import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Tuple
import re  # Used for email, phone, and experience pattern matching

# Verify re module is accessible at module level
assert re.findall  # Suppress false "unused import" warning

# Enhanced imports for OCR and semantic matching
try:
    import pytesseract  # type: ignore[misc]
    from PIL import Image  # type: ignore[misc]
    import pdf2image  # type: ignore[misc]
    ocr_available = True
    
    # Check if Tesseract is actually available
    try:
        pytesseract.get_tesseract_version()  # type: ignore[misc]
        tesseract_available = True
    except:
        tesseract_available = False
except ImportError:
    ocr_available = False
    tesseract_available = False

try:
    import spacy  # type: ignore[misc]
    spacy_available = True
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")  # type: ignore[misc]
    except OSError:
        nlp = None
        spacy_available = False
except ImportError:
    spacy_available = False
    nlp = None

try:
    import yaml  # type: ignore[misc]
    yaml_available = True
except ImportError:
    yaml_available = False

class RoyaltonAIScreener:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / "input_resumes"
        self.output_dir = self.base_dir / "screening_results"
        self.resort_name = "Royalton Resort"
        self.config = self.load_config()
        self.setup_directories()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = self.base_dir / "royalton_config.yaml"
        
        if yaml_available and config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)  # type: ignore[misc]
                print("✅ Loaded configuration from royalton_config.yaml")
                return config
            except Exception as e:
                print(f"⚠ Error loading config file: {e}, using defaults")
        
        # Fallback to built-in configuration
        return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available."""
        return {
            'positions': {
                'front_desk_agent': {
                    'must_have_skills': ['customer service', 'computer skills', 'communication', 'multitasking', 'phone skills'],
                    'nice_to_have_skills': ['hotel experience', 'PMS systems', 'Opera', 'front desk', 'guest services'],
                    'cultural_fit_keywords': ['team player', 'friendly', 'professional', 'positive attitude'],
                    'experience_weight': 0.3, 'skills_weight': 0.25, 'cultural_fit_weight': 0.25, 'hospitality_weight': 0.2
                },
                'chef': {
                    'must_have_skills': ['cooking', 'kitchen management', 'food safety', 'menu planning', 'culinary arts'],
                    'nice_to_have_skills': ['restaurant experience', 'fine dining', 'international cuisine', 'cost control'],
                    'cultural_fit_keywords': ['creative', 'attention to detail', 'team leader', 'high pressure'],
                    'experience_weight': 0.4, 'skills_weight': 0.3, 'cultural_fit_weight': 0.15, 'hospitality_weight': 0.15
                }
            },
            'skill_synonyms': {
                'customer_service': ['guest service', 'client service', 'customer care', 'guest relations'],
                'communication': ['verbal communication', 'written communication', 'interpersonal', 'people skills']
            },
            'ocr': {'enabled': True, 'dpi': 300},
            'output': {'excel_detailed': True, 'csv_export': True, 'create_folders': True}
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create instructions file
        instructions_file = self.input_dir / "README.txt"
        if not instructions_file.exists():
            instructions_file.write_text(f"""
=== {self.resort_name.upper()} AI RESUME SCREENER ===

1. Drop all your resume files (PDF, DOCX, TXT) into this folder
2. Run: Royalton_AI_Screener.bat
3. Tell the AI what position you need and how many candidates
4. Get intelligent recommendations with detailed explanations

AI Features:
✅ Smart position matching
✅ Candidate ranking with reasons
✅ Automatic top-N selection
✅ Detailed hiring recommendations
✅ Cultural fit assessment
✅ Experience level matching

Ready for intelligent hiring at Royalton Resort! 🤖🏨
            """)
    
    def get_resume_files(self) -> List[Path]:
        """Get all resume files from input directory."""
        supported_extensions = {'.pdf', '.docx', '.txt', '.doc'}
        files: List[Path] = []
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        return files
    
    def get_royalton_job_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """AI-enhanced job requirements with cultural fit and soft skills - now configurable."""
        # Use configuration if available
        if 'positions' in self.config:
            return self.config['positions']
        
        # Fallback to built-in intelligence
        return {
            # Executive Management
            'general manager': {
                'must_have': ['general manager', 'hotel management', 'operations', 'leadership', 'hospitality'],
                'nice_to_have': ['resort', 'p&l', 'budget', 'revenue', 'team management', 'gm', 'luxury'],
                'cultural_fit': ['leadership', 'vision', 'communication', 'problem solving', 'cultural sensitivity'],
                'experience_keywords': ['years', 'managed', 'led', 'directed', 'oversaw'],
                'min_years': 10,
                'ai_weight': {'experience': 0.4, 'skills': 0.3, 'cultural_fit': 0.2, 'hospitality': 0.1},
                'description': 'Strategic leader for overall resort operations'
            },
            'hotel manager': {
                'must_have': ['hotel manager', 'operations', 'management', 'hospitality', 'leadership'],
                'nice_to_have': ['resort', 'guest service', 'team management', 'budget', 'luxury'],
                'cultural_fit': ['leadership', 'communication', 'problem solving', 'adaptability'],
                'experience_keywords': ['managed', 'supervised', 'coordinated', 'improved'],
                'min_years': 7,
                'ai_weight': {'experience': 0.35, 'skills': 0.35, 'cultural_fit': 0.2, 'hospitality': 0.1},
                'description': 'Operational excellence and guest satisfaction leader'
            },
            'front desk': {
                'must_have': ['customer service', 'front desk', 'hotel', 'check-in', 'hospitality'],
                'nice_to_have': ['opera pms', 'concierge', 'multilingual', 'resort', 'guest relations'],
                'cultural_fit': ['friendly', 'helpful', 'patient', 'professional', 'multilingual'],
                'experience_keywords': ['served', 'assisted', 'handled', 'welcomed'],
                'min_years': 1,
                'ai_weight': {'experience': 0.25, 'skills': 0.35, 'cultural_fit': 0.3, 'hospitality': 0.1},
                'description': 'Guest experience ambassador and service excellence'
            },
            'front desk agent': {
                'must_have': ['customer service', 'front desk', 'hotel', 'check-in', 'hospitality'],
                'nice_to_have': ['opera pms', 'concierge', 'multilingual', 'resort', 'guest relations'],
                'cultural_fit': ['friendly', 'helpful', 'patient', 'professional', 'multilingual'],
                'experience_keywords': ['served', 'assisted', 'handled', 'welcomed'],
                'min_years': 1,
                'ai_weight': {'experience': 0.25, 'skills': 0.35, 'cultural_fit': 0.3, 'hospitality': 0.1},
                'description': 'Guest experience ambassador and service excellence'
            },
            'chef': {
                'must_have': ['chef', 'cooking', 'kitchen management', 'culinary', 'food preparation'],
                'nice_to_have': ['menu development', 'food cost', 'team leadership', 'hotel', 'resort'],
                'cultural_fit': ['creative', 'passionate', 'detail-oriented', 'leadership', 'innovative'],
                'experience_keywords': ['cooked', 'prepared', 'managed', 'created', 'developed'],
                'min_years': 3,
                'ai_weight': {'experience': 0.4, 'skills': 0.4, 'cultural_fit': 0.15, 'hospitality': 0.05},
                'description': 'Culinary artist and kitchen operations leader'
            },
            'executive chef': {
                'must_have': ['executive chef', 'chef', 'cooking', 'kitchen management', 'culinary'],
                'nice_to_have': ['menu development', 'food cost', 'team leadership', 'hotel', 'resort'],
                'cultural_fit': ['creative', 'passionate', 'leadership', 'innovative', 'mentor'],
                'experience_keywords': ['led', 'managed', 'created', 'developed', 'supervised'],
                'min_years': 5,
                'ai_weight': {'experience': 0.4, 'skills': 0.35, 'cultural_fit': 0.2, 'hospitality': 0.05},
                'description': 'Culinary excellence leader and kitchen innovator'
            },
            'housekeeping': {
                'must_have': ['housekeeping', 'cleaning', 'hotel', 'room cleaning'],
                'nice_to_have': ['attention to detail', 'hospitality', 'resort', 'guest service'],
                'cultural_fit': ['detail-oriented', 'reliable', 'hardworking', 'pride in work'],
                'experience_keywords': ['cleaned', 'maintained', 'organized', 'sanitized'],
                'min_years': 0,
                'ai_weight': {'experience': 0.2, 'skills': 0.4, 'cultural_fit': 0.3, 'hospitality': 0.1},
                'description': 'Excellence in cleanliness and guest comfort'
            },
            'bartender': {
                'must_have': ['bartender', 'bar', 'cocktails', 'customer service'],
                'nice_to_have': ['mixology', 'wine knowledge', 'hospitality', 'resort', 'entertainment'],
                'cultural_fit': ['outgoing', 'entertaining', 'knowledgeable', 'friendly', 'creative'],
                'experience_keywords': ['served', 'mixed', 'created', 'entertained'],
                'min_years': 1,
                'ai_weight': {'experience': 0.3, 'skills': 0.35, 'cultural_fit': 0.25, 'hospitality': 0.1},
                'description': 'Beverage expert and guest entertainment specialist'
            },
            'server': {
                'must_have': ['server', 'food service', 'customer service', 'restaurant'],
                'nice_to_have': ['fine dining', 'wine knowledge', 'upselling', 'hospitality', 'resort'],
                'cultural_fit': ['friendly', 'attentive', 'professional', 'team player'],
                'experience_keywords': ['served', 'attended', 'recommended', 'upsold'],
                'min_years': 1,
                'ai_weight': {'experience': 0.25, 'skills': 0.35, 'cultural_fit': 0.3, 'hospitality': 0.1},
                'description': 'Dining experience specialist and guest satisfaction expert'
            },
            'maintenance': {
                'must_have': ['maintenance', 'repair', 'electrical', 'plumbing', 'technical'],
                'nice_to_have': ['hotel maintenance', 'preventive maintenance', 'tools', 'resort'],
                'cultural_fit': ['problem solver', 'reliable', 'detail-oriented', 'safety-conscious'],
                'experience_keywords': ['repaired', 'maintained', 'fixed', 'installed'],
                'min_years': 2,
                'ai_weight': {'experience': 0.4, 'skills': 0.45, 'cultural_fit': 0.1, 'hospitality': 0.05},
                'description': 'Technical excellence and facility reliability specialist'
            },
            'security': {
                'must_have': ['security', 'safety', 'surveillance', 'protection'],
                'nice_to_have': ['hotel security', 'emergency response', 'patrol', 'resort'],
                'cultural_fit': ['vigilant', 'responsible', 'calm under pressure', 'trustworthy'],
                'experience_keywords': ['protected', 'monitored', 'patrolled', 'secured'],
                'min_years': 1,
                'ai_weight': {'experience': 0.35, 'skills': 0.35, 'cultural_fit': 0.25, 'hospitality': 0.05},
                'description': 'Safety and security excellence specialist'
            },
            'spa': {
                'must_have': ['spa', 'massage', 'therapy', 'wellness', 'relaxation'],
                'nice_to_have': ['certification', 'beauty treatments', 'hospitality', 'resort'],
                'cultural_fit': ['calming', 'caring', 'professional', 'healing touch'],
                'experience_keywords': ['treated', 'relaxed', 'healed', 'cared for'],
                'min_years': 1,
                'ai_weight': {'experience': 0.3, 'skills': 0.4, 'cultural_fit': 0.25, 'hospitality': 0.05},
                'description': 'Wellness and relaxation experience specialist'
            },
            'activities': {
                'must_have': ['activities', 'recreation', 'entertainment', 'guest engagement'],
                'nice_to_have': ['sports', 'water sports', 'animation', 'hospitality', 'resort'],
                'cultural_fit': ['energetic', 'entertaining', 'outgoing', 'fun', 'engaging'],
                'experience_keywords': ['organized', 'led', 'entertained', 'engaged'],
                'min_years': 1,
                'ai_weight': {'experience': 0.25, 'skills': 0.3, 'cultural_fit': 0.35, 'hospitality': 0.1},
                'description': 'Guest entertainment and engagement specialist'
            },
            'manager': {
                'must_have': ['management', 'leadership', 'supervisor', 'hospitality'],
                'nice_to_have': ['hotel', 'resort', 'team building', 'budget', 'operations'],
                'cultural_fit': ['leadership', 'communication', 'problem solving', 'mentoring'],
                'experience_keywords': ['managed', 'led', 'supervised', 'improved'],
                'min_years': 3,
                'ai_weight': {'experience': 0.35, 'skills': 0.3, 'cultural_fit': 0.25, 'hospitality': 0.1},
                'description': 'Leadership excellence and team development specialist'
            }
        }
    
    def extract_text_simple(self, file_path: str) -> str:
        """Enhanced text extraction with OCR fallback for scanned documents."""
        try:
            text: str = ""
            file_size = Path(file_path).stat().st_size
            
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    
            elif file_path.endswith('.pdf'):
                # Try standard PDF text extraction first
                try:
                    from pdfminer.high_level import extract_text
                    text = extract_text(file_path)
                    print(f"📄 PDF text extracted: {len(text)} characters")
                except ImportError:
                    print("⚠ pdfminer not available, trying PyPDF2...")
                    try:
                        import PyPDF2  # type: ignore[misc]
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)  # type: ignore[misc]
                            text = ""
                            for page in reader.pages:  # type: ignore[misc]
                                page_text = page.extract_text()  # type: ignore[misc]
                                if page_text and isinstance(page_text, str):
                                    text += page_text
                        print(f"📄 PyPDF2 text extracted: {len(text)} characters")
                    except Exception:
                        text = ""
                
                # Enhanced OCR fallback detection
                needs_ocr = (
                    not str(text).strip() or  # No text extracted  # type: ignore[misc]
                    len(str(text).strip()) < 50 or  # Very little text  # type: ignore[misc]
                    len(str(text).strip()) / file_size < 0.001  # Text ratio too low (likely scanned)  # type: ignore[misc]
                )
                
                if needs_ocr and ocr_available:
                    print(f"🔍 Text extraction yielded {len(str(text).strip())} chars - trying OCR...")  # type: ignore[misc]
                    ocr_text = self._extract_text_with_ocr(file_path)
                    if len(ocr_text.strip()) > len(str(text).strip()):  # type: ignore[misc]
                        text = ocr_text
                        print(f"✅ OCR successful: {len(text)} characters extracted")
                    
            elif file_path.endswith('.docx'):
                try:
                    from docx import Document  # type: ignore[misc]
                    doc = Document(file_path)  # type: ignore[misc]
                    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])  # type: ignore[misc]
                    print(f"📄 DOCX text extracted: {len(text)} characters")
                except Exception as e:
                    print(f"⚠ DOCX extraction failed: {e}")
                    # Try OCR as fallback for corrupted DOCX
                    if ocr_available:
                        print("🔍 Trying OCR fallback for DOCX...")
                        text = self._extract_text_with_ocr(file_path)
                        if text:
                            print(f"✅ OCR fallback successful: {len(text)} characters")
                    else:
                        text = ""
                    
            elif file_path.endswith('.doc'):
                # Legacy DOC files - try OCR directly
                print("📄 Legacy .doc file detected - using OCR...")
                if ocr_available:
                    text = self._extract_text_with_ocr(file_path)
                    if text:
                        print(f"✅ OCR successful: {len(text)} characters")
                else:
                    print("❌ OCR not available for .doc files")
                    text = ""
            
            # Final validation
            if not text.strip():
                print("❌ No text could be extracted from this file")
                return ""
            
            return text
            
        except Exception as e:
            print(f"⚠ Error extracting text from {file_path}: {e}")
            # Last resort OCR attempt
            if ocr_available:
                print("🔍 Last resort OCR attempt...")
                try:
                    text = self._extract_text_with_ocr(file_path)
                    if text:
                        print(f"✅ Last resort OCR successful: {len(text)} characters")
                        return text
                except:
                    pass
            return ""
    
    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text using OCR for scanned documents with enhanced processing."""
        if not ocr_available or not tesseract_available:
            if not ocr_available:
                print("⚠ OCR libraries not available - install pytesseract, pdf2image, Pillow")
            else:
                print("⚠ Tesseract OCR engine not installed - skipping OCR")
                print("💡 Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            return ""
            
        try:
            if file_path.endswith('.pdf'):
                print("🔍 Converting PDF to images for OCR...")
                # Convert PDF to images and OCR with enhanced settings
                pages = pdf2image.convert_from_path(  # type: ignore[misc]
                    file_path, 
                    dpi=300,  # High DPI for better OCR accuracy
                    first_page=1,
                    last_page=3,  # Limit to first 3 pages for performance
                )
                text_parts: List[str] = []
                for i, page in enumerate(pages, 1):  # type: ignore[misc]
                    print(f"🔍 OCR processing page {i}...")
                    try:
                        # Enhanced OCR with configuration
                        page_text = pytesseract.image_to_string(page, lang='eng')  # type: ignore[misc]
                        if page_text and page_text.strip():  # type: ignore[misc]
                            text_parts.append(page_text)  # type: ignore[misc]
                    except Exception as e:
                        print(f"⚠ OCR failed on page {i}: {e}")
                        continue
                
                final_text = '\n'.join(text_parts) if text_parts else ""
                if final_text:
                    print(f"✅ OCR completed: {len(final_text)} characters from {len(pages)} pages")  # type: ignore[misc]
                else:
                    print("❌ OCR could not extract text from PDF")
                return final_text
                
            elif file_path.endswith(('.docx', '.doc')):
                print("⚠ Document OCR not supported - use PDF or image format")
                return ""
                    
            else:
                # For other formats, try to open as image directly
                print("🔍 Processing image file with OCR...")
                try:
                    image = Image.open(file_path)  # type: ignore[misc]
                    
                    # Enhance image for better OCR
                    if hasattr(image, 'mode') and image.mode != 'RGB':  # type: ignore[misc]
                        image = image.convert('RGB')  # type: ignore[misc]
                    
                    # Simple OCR without complex configuration
                    text = pytesseract.image_to_string(image, lang='eng')  # type: ignore[misc]
                    print(f"✅ Image OCR completed: {len(text)} characters")  # type: ignore[misc]
                    return text if text else ""  # type: ignore[misc]
                except Exception as e:
                    print(f"⚠ Image OCR failed: {e}")
                    return ""
                
        except Exception as e:
            print(f"⚠ OCR failed for {file_path}: {e}")
            return ""
    
    def extract_location(self, text: str) -> str:
        """Extract location information from resume text."""
        text_lower = text.lower()
        
        # Common location patterns
        location_patterns = [
            r'(?:address|location|residence|live|located).*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s*[A-Z]{2})',
            r'([A-Z][a-z]+,\s*[A-Z]{2}(?:\s+\d{5})?)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2})',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',  # City, State
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        # Look for state abbreviations or common cities
        common_locations = [
            'new york', 'los angeles', 'chicago', 'houston', 'philadelphia',
            'phoenix', 'san antonio', 'san diego', 'dallas', 'austin',
            'miami', 'atlanta', 'boston', 'seattle', 'denver', 'las vegas',
            'nashville', 'charlotte', 'tampa', 'orlando', 'richmond',
            'brooklyn', 'queens', 'manhattan', 'bronx', 'detroit'
        ]
        
        for location in common_locations:
            if location in text_lower:
                return location.title()
        
        # Look for state codes
        state_codes = [
            'NY', 'CA', 'FL', 'TX', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
            'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'
        ]
        
        for state in state_codes:
            if f' {state} ' in text or f',{state}' in text or f' {state}\n' in text:
                return state
        
        return "Location not specified"
    
    def extract_ai_candidate_profile(self, text: str) -> Dict[str, Any]:
        """AI-enhanced candidate profile extraction."""
        import re
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        email = emails[0] if emails else ''
        
        # Extract phone
        phone_patterns = [
            r'\(\d{3}\)\s*\d{3}[-\.\s]?\d{4}',
            r'\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}',
        ]
        phone = ''
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                phone = phones[0]
                break
        
        # Enhanced name extraction
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name = ''
        for line in lines[:5]:
            if not any(word in line.lower() for word in ['resume', 'cv', 'email', 'phone', 'address', 'objective']):
                words = line.split()
                if 2 <= len(words) <= 4 and all(word.replace('.', '').isalpha() for word in words):
                    name = line
                    break
        
        # Enhanced experience extraction
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\-\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:experience|exp)',
        ]
        years = 0
        for pattern in exp_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                years = max(int(match) for match in matches)
                break
        
        # Enhanced skills extraction with AI context
        skills: List[str] = []
        skill_categories = {
            'hospitality': ['customer service', 'hospitality', 'hotel', 'resort', 'guest service', 'guest relations'],
            'technical': ['computer', 'software', 'system', 'technical', 'pms', 'opera'],
            'leadership': ['management', 'leadership', 'supervisor', 'team lead', 'coordinator'],
            'languages': ['bilingual', 'multilingual', 'spanish', 'french', 'english'],
            'certifications': ['certified', 'license', 'certification', 'trained'],
            'soft_skills': ['communication', 'problem solving', 'teamwork', 'adaptable', 'reliable']
        }
        
        text_lower = text.lower()
        for _, keywords in skill_categories.items():
            for skill in keywords:
                if skill in text_lower:
                    skills.append(skill)
        
        # Cultural fit indicators
        cultural_indicators: List[str] = []
        culture_keywords = {
            'service_oriented': ['service', 'helpful', 'caring', 'attentive'],
            'team_player': ['team', 'collaborate', 'work together', 'support'],
            'professional': ['professional', 'punctual', 'reliable', 'dedicated'],
            'positive_attitude': ['positive', 'enthusiastic', 'passionate', 'motivated'],
            'adaptable': ['flexible', 'adaptable', 'learn', 'quick learner']
        }
        
        for trait, keywords in culture_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                cultural_indicators.append(trait)
        
        return {
            'name': name,
            'email': email,
            'phone': phone,
            'experience_years': years,
            'skills': skills,
            'cultural_fit': cultural_indicators,
            'text': text
        }
    
    def analyze_role_relevant_experience(self, text: str, position: str, requirements: Dict[str, Any]) -> Dict[str, bool]:
        """AI-powered analysis of role-relevant experience - SMART matching."""
        
        # Role-specific keywords for direct experience detection
        role_keywords = {
            'front desk': ['front desk', 'reception', 'check-in', 'check-out', 'guest services', 'hotel clerk'],
            'chef': ['chef', 'cook', 'kitchen', 'culinary', 'food preparation', 'menu', 'restaurant'],
            'bartender': ['bartender', 'bar', 'cocktails', 'drinks', 'mixology', 'beverage'],
            'housekeeping': ['housekeeping', 'cleaning', 'room attendant', 'maid', 'cleaner'],
            'manager': ['manager', 'supervisor', 'director', 'lead', 'management', 'leadership'],
            'security': ['security', 'guard', 'patrol', 'surveillance', 'safety'],
            'spa': ['spa', 'massage', 'therapy', 'wellness', 'beauty', 'facial'],
            'server': ['server', 'waiter', 'waitress', 'food service', 'restaurant'],
            'maintenance': ['maintenance', 'repair', 'technician', 'electrician', 'plumber']
        }
        
        # Hospitality industry keywords
        hospitality_keywords = [
            'hotel', 'resort', 'hospitality', 'guest', 'customer service',
            'tourism', 'vacation', 'lodge', 'inn', 'motel', 'casino',
            'cruise', 'restaurant', 'catering', 'events'
        ]
        
        # Transferable skills by role
        transferable_skills = {
            'front desk': ['customer service', 'communication', 'computer', 'phone', 'reception', 'administration'],
            'chef': ['cooking', 'food', 'restaurant', 'catering', 'nutrition', 'food safety'],
            'bartender': ['customer service', 'sales', 'cash handling', 'entertainment', 'social'],
            'housekeeping': ['cleaning', 'organizing', 'attention to detail', 'time management'],
            'manager': ['management', 'leadership', 'supervision', 'training', 'coordination'],
            'security': ['law enforcement', 'military', 'safety', 'emergency', 'surveillance'],
            'spa': ['beauty', 'wellness', 'health', 'therapy', 'relaxation', 'massage'],
            'server': ['customer service', 'food service', 'sales', 'hospitality', 'restaurant'],
            'maintenance': ['repair', 'technical', 'engineering', 'construction', 'tools']
        }
        
        # Check for direct role experience
        direct_match = False
        position_keywords = role_keywords.get(position, [position])
        for keyword in position_keywords:
            if keyword in text:
                direct_match = True
                break
        
        # Check for hospitality industry experience
        hospitality_match = any(keyword in text for keyword in hospitality_keywords)
        
        # Check for transferable skills
        transferable_match = False
        role_transferable = transferable_skills.get(position, [])
        for skill in role_transferable:
            if skill in text:
                transferable_match = True
                break
        
        return {
            'direct_match': direct_match,
            'transferable': transferable_match,
            'hospitality': hospitality_match
        }
    
    def smart_skill_match(self, skill: str, text: str, skills: List[str], position: str) -> bool:
        """Enhanced smart skill matching with configuration and semantic similarity."""
        skill_lower = skill.lower()
        text_lower = text.lower()
        
        # Direct match
        if skill_lower in text_lower or any(skill_lower in s.lower() for s in skills):
            return True
        
        # Use configured synonyms if available
        config_synonyms = self.config.get('skill_synonyms', {})
        if skill_lower in config_synonyms:
            synonyms = config_synonyms[skill_lower]
            for synonym in synonyms:
                if synonym.lower() in text_lower:
                    return True
        
        # Fallback to built-in synonyms
        skill_synonyms = {
            'customer service': ['guest service', 'client service', 'customer care', 'guest relations', 'hospitality service'],
            'management': ['leadership', 'supervision', 'managing', 'coordinating', 'team lead'],
            'communication': ['interpersonal', 'verbal', 'written', 'speaking', 'people skills'],
            'hospitality': ['hotel', 'resort', 'guest', 'service', 'tourism'],
            'food service': ['restaurant', 'dining', 'catering', 'food prep', 'culinary'],
            'cleaning': ['housekeeping', 'sanitation', 'maintenance', 'organizing']
        }
        
        synonyms = skill_synonyms.get(skill_lower, [])
        for synonym in synonyms:
            if synonym in text_lower:
                return True
        
        # If spaCy is available, try semantic similarity
        if spacy_available and nlp:
            try:
                skill_doc = nlp(skill)  # type: ignore[misc]
                # Check similarity with key phrases in text
                sentences = [sent.text for sent in nlp(text).sents if len(sent.text.strip()) > 10][:5]  # type: ignore[misc]
                for sentence in sentences:  # type: ignore[misc]
                    sentence_doc = nlp(sentence)  # type: ignore[misc]
                    if skill_doc.similarity(sentence_doc) > 0.7:  # type: ignore[misc]
                        return True
            except:
                pass  # Fall back to keyword matching if spaCy fails
        
        return False
    
    def detect_role_specific_skills(self, text: str, position: str) -> List[str]:
        """Detect role-specific skills that give candidates an edge."""
        
        role_expert_skills = {
            'front desk': [
                'opera pms', 'micros', 'hotel software', 'bilingual', 'multilingual',
                'concierge', 'reservations', 'check-in', 'guest relations'
            ],
            'chef': [
                'menu development', 'food costing', 'haccp', 'servsafe', 'culinary arts',
                'kitchen management', 'food safety', 'recipe creation', 'inventory control'
            ],
            'bartender': [
                'mixology', 'cocktail creation', 'wine knowledge', 'flair bartending',
                'pos systems', 'inventory management', 'craft cocktails'
            ],
            'housekeeping': [
                'deep cleaning', 'laundry operations', 'inventory control', 'quality inspection',
                'room setup', 'guest amenities', 'cleaning chemicals'
            ],
            'manager': [
                'budget management', 'staff training', 'performance reviews', 'scheduling',
                'revenue management', 'guest satisfaction', 'team building'
            ],
            'security': [
                'surveillance systems', 'emergency response', 'crowd control', 'cpr certified',
                'first aid', 'incident reporting', 'patrol procedures'
            ],
            'spa': [
                'massage therapy', 'facial treatments', 'aromatherapy', 'wellness',
                'beauty treatments', 'relaxation techniques', 'customer consultation'
            ]
        }
        
        expert_skills = role_expert_skills.get(position, [])
        found_skills: List[str] = []
        
        for skill in expert_skills:
            if skill in text:
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_ai_score(self, candidate: Dict[str, Any], requirements: Dict[str, Any], position: str) -> Tuple[float, List[str], List[str], str]:
        """AI-powered scoring with detailed reasoning - SMART role-specific matching."""
        score = 15.0  # Conservative base score
        reasons: List[str] = []
        ai_insights: List[str] = []
        
        text = candidate.get('text', '').lower()
        skills = [s.lower() for s in candidate.get('skills', [])]
        cultural_fit = candidate.get('cultural_fit', [])
        experience_years = candidate.get('experience_years', 0)
        
        # AI SMART ROLE MATCHING - Context-aware scoring
        position_lower = position.lower()
        
        # INTELLIGENT EXPERIENCE ANALYSIS
        role_relevant_experience = self.analyze_role_relevant_experience(text, position_lower, requirements)
        direct_role_experience = role_relevant_experience['direct_match']
        transferable_experience = role_relevant_experience['transferable']
        industry_experience = role_relevant_experience['hospitality']
        
        # Smart experience scoring based on role relevance
        exp_score = 0.0
        if direct_role_experience:
            exp_score = 95  # Perfect match!
            reasons.append(f"🎯 PERFECT MATCH: Direct {position} experience found")
            ai_insights.append(f"IDEAL CANDIDATE - Has specific {position} background")
            score += 25  # Bonus for perfect match
        elif transferable_experience and industry_experience:
            exp_score = 80  # Great transferable skills in hospitality
            reasons.append(f"🌟 EXCELLENT FIT: Hospitality experience + transferable skills")
            ai_insights.append("STRONG CANDIDATE - Proven hospitality background")
            score += 15
        elif industry_experience:
            exp_score = 70  # Hospitality experience
            reasons.append(f"✓ GOOD FIT: Hospitality industry experience")
            ai_insights.append("QUALIFIED - Hotel/resort background")
            score += 10
        elif transferable_experience:
            exp_score = 60  # Transferable skills from other industries
            reasons.append(f"✓ POTENTIAL: Relevant transferable experience")
            ai_insights.append("TRAINABLE - Good foundation skills")
            score += 5
        else:
            exp_score = 35  # Entry level
            reasons.append(f"⚡ ENTRY LEVEL: Fresh candidate with potential")
            ai_insights.append("DEVELOPMENT CANDIDATE - Needs training")
        
        # Factor in years of experience
        if experience_years >= 5:
            exp_score += 15
            reasons.append(f"+ EXPERIENCED: {experience_years} years in field")
        elif experience_years >= 2:
            exp_score += 10
            reasons.append(f"+ MODERATE EXPERIENCE: {experience_years} years")
        elif experience_years >= 1:
            exp_score += 5
            reasons.append(f"+ SOME EXPERIENCE: {experience_years} year(s)")
        
        # AI Weight system - role-specific weighting
        weights = requirements.get('ai_weight', {
            'experience': 0.30, 'skills': 0.35, 'cultural_fit': 0.25, 'role_match': 0.10
        })
        
        score += exp_score * weights.get('experience', 0.30)
        
        # SMART SKILLS MATCHING - Role-specific intelligence
        must_have = requirements.get('must_have', [])
        nice_to_have = requirements.get('nice_to_have', [])
        
        must_have_found: List[str] = []
        nice_to_have_found: List[str] = []
        role_specific_skills: List[str] = []
        
        # Enhanced skill detection with role context
        for skill in must_have:
            if self.smart_skill_match(skill, text, skills, position_lower):
                must_have_found.append(skill)
        
        for skill in nice_to_have:
            if self.smart_skill_match(skill, text, skills, position_lower):
                nice_to_have_found.append(skill)
        
        # AI Role-specific skill bonuses
        role_specific_skills = self.detect_role_specific_skills(text, position_lower)
        
        # Intelligent skills scoring
        skills_score = 25  # Base skills score
        
        if must_have:
            match_percentage = (len(must_have_found) / len(must_have)) * 100
            if match_percentage >= 80:
                skills_score = 90
                reasons.append(f"🎯 PERFECT SKILLS: {len(must_have_found)}/{len(must_have)} required skills")
                ai_insights.append("SKILL PERFECT - All essential abilities present")
            elif match_percentage >= 60:
                skills_score = 75
                reasons.append(f"✅ STRONG SKILLS: {len(must_have_found)}/{len(must_have)} required skills")
                ai_insights.append("WELL-SKILLED - Minor gaps easily filled")
            elif match_percentage >= 40:
                skills_score = 60
                reasons.append(f"✓ DECENT SKILLS: {len(must_have_found)}/{len(must_have)} required skills")
                ai_insights.append("ADEQUATE SKILLS - Some training needed")
            elif match_percentage >= 20:
                skills_score = 45
                reasons.append(f"⚡ BASIC SKILLS: {len(must_have_found)}/{len(must_have)} required skills")
                ai_insights.append("FOUNDATION SKILLS - Needs development")
            else:
                skills_score = 35
                reasons.append(f"⚠ LIMITED MATCH: Few required skills but has potential")
                ai_insights.append("TRAINABLE - Focus on transferable strengths")
        
        # Role-specific skill bonuses (SMART detection)
        if role_specific_skills:
            bonus = min(20, len(role_specific_skills) * 4)
            skills_score += bonus
            reasons.append(f"🌟 ROLE EXPERTISE: {', '.join(role_specific_skills[:3])}")
        
        # Nice-to-have bonuses
        if nice_to_have_found:
            bonus = min(15, len(nice_to_have_found) * 3)
            skills_score += bonus
            reasons.append(f"+ BONUS SKILLS: {', '.join(nice_to_have_found[:3])}")

        score += min(100, skills_score) * weights.get('skills', 0.35)
        
        # Cultural fit scoring (weighted)
        required_culture = requirements.get('cultural_fit', [])
        culture_score = 0.0
        culture_matches: List[str] = []
        
        for trait in required_culture:
            if trait.lower() in text or trait in cultural_fit:
                culture_matches.append(trait)
        
        if required_culture:
            culture_percentage = (len(culture_matches) / len(required_culture)) * 100
            culture_score = culture_percentage
            
            if culture_percentage >= 80:
                reasons.append(f"🤝 Excellent cultural fit: {len(culture_matches)}/{len(required_culture)} traits")
                ai_insights.append("CULTURAL CHAMPION - Perfect alignment with Royalton values")
            elif culture_percentage >= 50:
                reasons.append(f"✓ Good cultural fit: {len(culture_matches)}/{len(required_culture)} traits")
                ai_insights.append("GOOD FIT - Aligns well with resort culture")
            else:
                reasons.append(f"⚠ Cultural fit needs assessment")
                ai_insights.append("NEEDS EVALUATION - Cultural alignment unclear")
        
        score += culture_score * weights.get('cultural_fit', 0.2)
        
        # Hospitality experience bonus
        hospitality_keywords = ['hotel', 'resort', 'hospitality', 'guest service', 'customer service']
        hospitality_found = [kw for kw in hospitality_keywords if kw in text]
        if hospitality_found:
            hospitality_bonus = min(20, len(hospitality_found) * 5)
            score += hospitality_bonus * weights.get('hospitality', 0.1)
            reasons.append(f"🏨 Hospitality experience bonus")
            ai_insights.append("INDUSTRY VETERAN - Understands hospitality excellence")
        
        # Luxury/Premium experience
        luxury_keywords = ['royalton', 'luxury', 'five star', '5 star', 'premium', 'upscale', 'high-end']
        if any(kw in text for kw in luxury_keywords):
            score += 10
            reasons.append(f"💎 Luxury experience bonus")
            ai_insights.append("LUXURY SPECIALIST - Exceeds standard service expectations")
        
        # Position-specific bonus
        if position.lower() in text:
            score += 5
            reasons.append(f"🎯 Direct position experience")
        
        # AI recommendation level (always finds potential)
        if score >= 85:
            ai_recommendation = "HIRE IMMEDIATELY - Exceptional candidate"
        elif score >= 75:
            ai_recommendation = "STRONG HIRE - Excellent fit for position"
        elif score >= 65:
            ai_recommendation = "GOOD CANDIDATE - Interview recommended"
        elif score >= 55:
            ai_recommendation = "POTENTIAL HIRE - Good training candidate"
        elif score >= 45:
            ai_recommendation = "TRAINABLE - Worth considering for development"
        else:
            ai_recommendation = "ENTRY LEVEL - Could work with proper support"
        
        return round(score, 1), reasons, ai_insights, ai_recommendation
    
    def process_resume_ai(self, file_path: Path) -> Dict[str, Any]:
        """AI-enhanced resume processing."""
        try:
            print(f"🤖 AI analyzing: {file_path.name}")
            
            # Extract text from file
            text = self.extract_text_simple(str(file_path))
            
            if not text or len(text.strip()) < 50:
                return {
                    'file_path': file_path,
                    'error': 'Could not extract meaningful text',
                    'name': file_path.stem,
                    'ai_recommendation': 'CANNOT PROCESS - File issue',
                    'score': 0,
                    'reasons': ['File could not be processed']
                }
            
            # AI candidate profile extraction
            candidate_profile = self.extract_ai_candidate_profile(text)
            
            return {
                'file_path': file_path,
                'name': candidate_profile.get('name', file_path.stem),
                'email': candidate_profile.get('email', ''),
                'phone': candidate_profile.get('phone', ''),
                'skills': candidate_profile.get('skills', []),
                'cultural_fit': candidate_profile.get('cultural_fit', []),
                'experience_years': candidate_profile.get('experience_years', 0),
                'text': text,
                'error': None
            }
            
        except Exception as e:
            print(f"❌ AI error processing {file_path.name}: {e}")
            return {
                'file_path': file_path,
                'error': str(e),
                'name': file_path.stem,
                'ai_recommendation': 'PROCESSING ERROR',
                'score': 0,
                'reasons': [f'Processing error: {str(e)}']
            }
    
    def ai_screen_candidates(self, position: str, candidates: List[Dict[str, Any]], num_needed: int = 5) -> List[Dict[str, Any]]:
        """AI-powered candidate screening and ranking."""
        print(f"\n🤖 AI screening candidates for {self.resort_name} position: {position}")
        print(f"🎯 Looking for top {num_needed} candidates")
        
        job_intelligence = self.get_royalton_job_intelligence()
        
        # AI position matching
        position_lower = position.lower()
        requirements: Dict[str, Any] = {}
        
        # Exact match first
        if position_lower in job_intelligence:
            requirements = job_intelligence[position_lower]
        else:
            # AI fuzzy matching
            best_match = None
            best_score = 0
            
            for job_type, _ in job_intelligence.items():
                # Calculate similarity score
                job_words = set(job_type.split())
                position_words = set(position_lower.split())
                overlap = len(job_words.intersection(position_words))
                
                if overlap > best_score:
                    best_score = overlap
                    best_match = job_type
                
                # Check if position words are in job type
                if any(word in job_type for word in position_words):
                    if len(position_words) > best_score:
                        best_score = len(position_words)
                        best_match = job_type
            
            if best_match:
                requirements = job_intelligence[best_match]
                print(f"🔍 AI matched '{position}' to '{best_match}'")
        
        # Default AI requirements if no match
        if not requirements:
            requirements = {
                'must_have': ['customer service', 'hospitality'] + position_lower.split(),
                'nice_to_have': ['hotel', 'resort', 'guest service', 'professional'],
                'cultural_fit': ['friendly', 'professional', 'reliable'],
                'min_years': 1,
                'ai_weight': {'experience': 0.3, 'skills': 0.4, 'cultural_fit': 0.2, 'hospitality': 0.1},
                'description': f'AI-generated profile for: {position}'
            }
        
        # AI score each candidate
        scored_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            if candidate.get('error'):
                candidate['score'] = 0
                candidate['reasons'] = [candidate['error']]
                candidate['ai_insights'] = []
                candidate['ai_recommendation'] = 'PROCESSING ERROR'
            else:
                score, reasons, ai_insights, ai_recommendation = self.calculate_ai_score(candidate, requirements, position)
                candidate['score'] = score
                candidate['reasons'] = reasons
                candidate['ai_insights'] = ai_insights
                candidate['ai_recommendation'] = ai_recommendation
            
            scored_candidates.append(candidate)
        
        # Remove duplicates based on name, email, and phone
        print(f"🧹 Removing duplicates...")
        unique_candidates: List[Dict[str, Any]] = []
        seen_candidates: set[str] = set()
        
        for candidate in scored_candidates:
            # Create unique identifier from name, email, phone
            name = candidate.get('name', '').strip().lower()
            email = candidate.get('email', '').strip().lower()
            phone = candidate.get('phone', '').strip()
            
            # Clean phone number for comparison
            clean_phone = ''.join(filter(str.isdigit, phone))
            
            # Create identifier (prefer email, fallback to name+phone, finally just filename)
            if email and '@' in email:
                identifier = email
            elif name and len(name) > 2:
                identifier = f"{name}_{clean_phone}"
            else:
                identifier = str(candidate.get('file_path', 'unknown'))
            
            if identifier not in seen_candidates:
                seen_candidates.add(identifier)
                unique_candidates.append(candidate)
            else:
                print(f"   ⚠ Removing duplicate: {candidate.get('name', 'Unknown')} - {candidate.get('file_path', {}).name if hasattr(candidate.get('file_path', {}), 'name') else 'Unknown file'}")
        
        print(f"✅ Kept {len(unique_candidates)} unique candidates (removed {len(scored_candidates) - len(unique_candidates)} duplicates)")
        
        # AI ranking
        unique_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # AI insights
        total_candidates = len(unique_candidates)
        qualified_candidates = len([c for c in unique_candidates if c['score'] >= 60])
        excellent_candidates = len([c for c in unique_candidates if c['score'] >= 80])
        
        print(f"📊 AI Analysis Complete:")
        print(f"   • {excellent_candidates} excellent candidates (80+ score)")
        print(f"   • {qualified_candidates} qualified candidates (60+ score)")
        print(f"   • {total_candidates} total candidates analyzed")
        
        return unique_candidates
    
    def create_ai_output_folder(self, position: str, num_needed: int) -> Path:
        """Create AI-enhanced output folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_position = "".join(c for c in position if c.isalnum() or c in " -_").strip()
        safe_position = safe_position.replace(" ", "_")
        
        folder_name = f"{timestamp}_AI_{safe_position}_Top{num_needed}"
        output_folder = self.output_dir / folder_name
        output_folder.mkdir(exist_ok=True)
        
        (output_folder / "top_recommendations").mkdir(exist_ok=True)
        (output_folder / "qualified_candidates").mkdir(exist_ok=True)
        (output_folder / "all_candidates").mkdir(exist_ok=True)
        
        return output_folder
    
    def export_ai_results(self, candidates: List[Dict[str, Any]], output_folder: Path, position: str, num_needed: int) -> Path:
        """Export AI-enhanced results to Excel with HR-friendly contact sheet."""
        excel_data: List[Dict[str, Any]] = []
        
        for i, candidate in enumerate(candidates, 1):
            # Extract location information from text
            location = self.extract_location(candidate.get('text', ''))
            
            # Determine confidence level
            score = candidate.get('score', 0)
            if score >= 90:
                confidence = "EXCELLENT"
                confidence_desc = "Hire immediately - Perfect match"
            elif score >= 80:
                confidence = "HIGH" 
                confidence_desc = "Strong candidate - Schedule interview ASAP"
            elif score >= 70:
                confidence = "GOOD"
                confidence_desc = "Good fit - Worth interviewing"
            elif score >= 60:
                confidence = "MODERATE"
                confidence_desc = "Consider for interview if positions available"
            else:
                confidence = "LOW"
                confidence_desc = "Not recommended for this role"
            
            excel_data.append({
                'AI_Rank': i,
                'Name': candidate.get('name', 'Unknown'),
                'Email': candidate.get('email', ''),
                'Phone': candidate.get('phone', ''),
                'Location': location,
                'AI_Score': round(candidate.get('score', 0), 1),
                'Confidence': confidence,
                'Confidence_Description': confidence_desc,
                'AI_Recommendation': candidate.get('ai_recommendation', ''),
                'Experience_Years': candidate.get('experience_years', 0),
                'Key_Skills': ', '.join(candidate.get('skills', [])[:5]),
                'Cultural_Fit': ', '.join(candidate.get('cultural_fit', [])[:3]),
                'AI_Insights': ' | '.join(candidate.get('ai_insights', [])[:2]),
                'Screening_Notes': ' | '.join(candidate.get('reasons', [])[:3]),
                'File_Name': candidate['file_path'].name,
                'Hire_Priority': 'TOP CHOICE' if i <= num_needed and candidate.get('score', 0) >= 70 else
                               'INTERVIEW' if candidate.get('score', 0) >= 60 else
                               'CONSIDER' if candidate.get('score', 0) >= 40 else 'PASS'
            })
        
        df = pd.DataFrame(excel_data)
        
        # MAIN EXCEL FILE - This should be the primary output
        excel_file = output_folder / f"Royalton_Screening_Results_{position.replace(' ', '_')}.xlsx"
        
        print(f"📊 Creating Excel file: {excel_file.name}")
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                
                # 0. HR INSTRUCTIONS - How to use this file
                instructions_data = [
                    ['📞 CANDIDATES TO CALL', 'Start here! Pre-filtered qualified candidates with contact info'],
                    ['🏆 Top X Picks', 'Your highest-rated candidates for immediate interviews'],
                    ['🤖 Full AI Analysis', 'Complete analysis of all candidates with detailed scores'],
                    ['Qualified Pool', 'All candidates scoring 60+ (interview-worthy)'],
                    ['📊 Summary', 'Overall statistics and AI confidence level'],
                    ['', ''],
                    ['HOW TO USE:', ''],
                    ['1. Call candidates from "CANDIDATES TO CALL" sheet first', ''],
                    ['2. Start with EXCELLENT confidence candidates', ''],
                    ['3. Work down by RANK order', ''],
                    ['4. Use WHY_HIRE column for talking points', ''],
                    ['5. Check NOTES for specific AI insights', ''],
                    ['', ''],
                    ['CONFIDENCE LEVELS:', ''],
                    ['EXCELLENT (90+)', 'Hire immediately - Perfect match for role'],
                    ['HIGH (80-89)', 'Strong candidate - Schedule interview ASAP'],
                    ['GOOD (70-79)', 'Good fit - Worth interviewing'],
                    ['MODERATE (60-69)', 'Consider if positions available'],
                    ['', ''],
                    ['SCORING BREAKDOWN:', ''],
                    ['Experience Match', '30-40% of score'],
                    ['Skills Match', '25-30% of score'],
                    ['Cultural Fit', '15-25% of score'],
                    ['Hospitality Background', '15-20% of score']
                ]
                
                instructions_df = pd.DataFrame(instructions_data, columns=['FEATURE', 'DESCRIPTION'])
                instructions_df.to_excel(writer, sheet_name='📋 HOW TO USE', index=False)  # type: ignore[misc]
                
                # 1. CANDIDATES TO CALL - HR-ready contact sheet
                contact_data: List[Dict[str, Any]] = []
                for candidate in excel_data:
                    if candidate['AI_Score'] >= 60:  # Only include qualified candidates
                        contact_data.append({
                            'RANK': candidate['AI_Rank'],
                            'NAME': candidate['Name'],
                            'PHONE': candidate['Phone'],
                            'EMAIL': candidate['Email'], 
                            'LOCATION': candidate['Location'],
                            'CONFIDENCE': candidate['Confidence'],
                            'SCORE': candidate['AI_Score'],
                            'WHY_HIRE': candidate['Confidence_Description'],
                            'KEY_SKILLS': candidate['Key_Skills'],
                            'NOTES': candidate['Screening_Notes']
                        })
                
                if contact_data:
                    contact_df = pd.DataFrame(contact_data)
                    contact_df.to_excel(writer, sheet_name='📞 CANDIDATES TO CALL', index=False)  # type: ignore[misc]
                    print(f"✅ Created 'CANDIDATES TO CALL' sheet with {len(contact_data)} qualified candidates")
                
                # 2. Complete AI Analysis
                df.to_excel(writer, sheet_name='🤖 Full AI Analysis', index=False)  # type: ignore[misc]
                print(f"✅ Created 'Full AI Analysis' sheet with {len(df)} candidates")
                
                # 3. Top recommendations
                top_candidates = df.head(num_needed)
                if not top_candidates.empty:
                    top_candidates.to_excel(writer, sheet_name=f'🏆 Top {num_needed} Picks', index=False)  # type: ignore[misc]
                    print(f"✅ Created 'Top {num_needed} Picks' sheet")
                
                # 4. Qualified candidates
                qualified = df[df['AI_Score'] >= 60]
                if not qualified.empty:
                    qualified.to_excel(writer, sheet_name='📞 Qualified Pool', index=False)  # type: ignore[misc]
                    print(f"✅ Created 'Qualified Pool' sheet with {len(qualified)} candidates")
                
                # 5. Enhanced AI Summary with actionable insights
                summary_data: Dict[str, Any] = {
                    'Metric': [
                        'Resort', 'Position', 'Candidates Needed', 'Date', 'Total Analyzed',
                        'EXCELLENT Candidates (90+)', 'HIGH Confidence (80-89)', 'GOOD Fit (70-79)', 
                        'MODERATE (60-69)', 'Low Score (<60)', 'Average Score', 'AI Confidence',
                        '', 'HIRING RECOMMENDATION:', 'Next Steps:', 'Call Priority Order:',
                        'Estimated Interview Time:', 'Quality Assessment:'
                    ],
                    'Value': [
                        self.resort_name, position, num_needed, datetime.now().strftime("%Y-%m-%d"),
                        len(df), 
                        len(df[df['AI_Score'] >= 90]),
                        len(df[(df['AI_Score'] >= 80) & (df['AI_Score'] < 90)]),
                        len(df[(df['AI_Score'] >= 70) & (df['AI_Score'] < 80)]),
                        len(df[(df['AI_Score'] >= 60) & (df['AI_Score'] < 70)]),
                        len(df[df['AI_Score'] < 60]),
                        round(df['AI_Score'].mean(), 1),
                        'HIGH' if df['AI_Score'].max() >= 80 else 'MEDIUM' if df['AI_Score'].max() >= 60 else 'LOW',
                        '',
                        f"Interview top {min(num_needed, len(df[df['AI_Score'] >= 60]))} candidates immediately",
                        "1. Call EXCELLENT candidates first\n2. Schedule HIGH confidence next\n3. Consider GOOD fits for backup",
                        "EXCELLENT → HIGH → GOOD → MODERATE",
                        f"{len(df[df['AI_Score'] >= 80]) * 30} minutes (30 min/top candidate)",
                        'EXCELLENT' if len(df[df['AI_Score'] >= 90]) >= num_needed else 
                        'VERY GOOD' if len(df[df['AI_Score'] >= 80]) >= num_needed else
                        'GOOD' if len(df[df['AI_Score'] >= 70]) >= num_needed else 'MODERATE'
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='📊 Summary', index=False)  # type: ignore[misc]
                print(f"✅ Created 'Summary' sheet")
            
            print(f"🎉 Excel file successfully created: {excel_file.name}")
            
        except Exception as e:
            print(f"❌ Error creating Excel file: {e}")
            print("📊 Falling back to CSV export...")
            
            # Fallback to CSV if Excel fails
            csv_file = output_folder / f"Royalton_Screening_Results_{position.replace(' ', '_')}.csv"
            df.to_csv(csv_file, index=False)
            print(f"📊 CSV file created: {csv_file.name}")
            excel_file = csv_file  # Return CSV path if Excel failed
            contact_data: List[Dict[str, Any]] = []
            for candidate in excel_data:
                if candidate['AI_Score'] >= 60:  # Only include qualified candidates
                    contact_data.append({
                        'RANK': candidate['AI_Rank'],
                        'NAME': candidate['Name'],
                        'PHONE': candidate['Phone'],
                        'EMAIL': candidate['Email'], 
                        'LOCATION': candidate['Location'],
                        'CONFIDENCE': candidate['Confidence'],
                        'SCORE': candidate['AI_Score'],
                        'WHY_HIRE': candidate['Confidence_Description'],
                        'KEY_SKILLS': candidate['Key_Skills'],
                        'NOTES': candidate['Screening_Notes']
                    })
            
            if contact_data:
                contact_df = pd.DataFrame(contact_data)
                contact_df.to_excel(writer, sheet_name='📞 CANDIDATES TO CALL', index=False)  # type: ignore[misc]
            
            # 2. Complete AI Analysis
            df.to_excel(writer, sheet_name='🤖 Full AI Analysis', index=False)  # type: ignore[misc]
            
            # 3. Top recommendations
            top_candidates = df.head(num_needed)
            if not top_candidates.empty:
                top_candidates.to_excel(writer, sheet_name=f'🏆 Top {num_needed} Picks', index=False)  # type: ignore[misc]
            
            # 4. Qualified candidates
            qualified = df[df['AI_Score'] >= 60]
            if not qualified.empty:
                qualified.to_excel(writer, sheet_name='Qualified Pool', index=False)  # type: ignore[misc]
            
            # 5. Enhanced AI Summary with actionable insights
            summary_data: Dict[str, Any] = {
                'Metric': [
                    'Resort', 'Position', 'Candidates Needed', 'Date', 'Total Analyzed',
                    'EXCELLENT Candidates (90+)', 'HIGH Confidence (80-89)', 'GOOD Fit (70-79)', 
                    'MODERATE (60-69)', 'Low Score (<60)', 'Average Score', 'AI Confidence',
                    '', 'HIRING RECOMMENDATION:', 'Next Steps:', 'Call Priority Order:',
                    'Estimated Interview Time:', 'Quality Assessment:'
                ],
                'Value': [
                    self.resort_name, position, num_needed, datetime.now().strftime("%Y-%m-%d"),
                    len(df), 
                    len(df[df['AI_Score'] >= 90]),
                    len(df[(df['AI_Score'] >= 80) & (df['AI_Score'] < 90)]),
                    len(df[(df['AI_Score'] >= 70) & (df['AI_Score'] < 80)]),
                    len(df[(df['AI_Score'] >= 60) & (df['AI_Score'] < 70)]),
                    len(df[df['AI_Score'] < 60]),
                    round(df['AI_Score'].mean(), 1),
                    'HIGH' if df['AI_Score'].max() >= 80 else 'MEDIUM' if df['AI_Score'].max() >= 60 else 'LOW',
                    '',
                    f"Interview top {min(num_needed, len(df[df['AI_Score'] >= 60]))} candidates immediately",
                    "1. Call EXCELLENT candidates first\n2. Schedule HIGH confidence next\n3. Consider GOOD fits for backup",
                    "EXCELLENT → HIGH → GOOD → MODERATE",
                    f"{len(df[df['AI_Score'] >= 80]) * 30} minutes (30 min/top candidate)",
                    'EXCELLENT' if len(df[df['AI_Score'] >= 90]) >= num_needed else 
                    'VERY GOOD' if len(df[df['AI_Score'] >= 80]) >= num_needed else
                    'GOOD' if len(df[df['AI_Score'] >= 70]) >= num_needed else 'MODERATE'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='📊 Summary', index=False)  # type: ignore[misc]
        
        # Export CSV files if configured
        if self.config.get('output', {}).get('csv_export', True):
            csv_folder = output_folder / "csv_exports"
            csv_folder.mkdir(exist_ok=True)
            
            # Main results CSV
            csv_file = csv_folder / f"AI_Results_{position.replace(' ', '_')}.csv"
            df.to_csv(csv_file, index=False)
            
            # Top candidates CSV
            if not top_candidates.empty:
                top_csv = csv_folder / f"Top_{num_needed}_{position.replace(' ', '_')}.csv"
                top_candidates.to_csv(top_csv, index=False)
            
            print(f"📊 CSV exports saved to: {csv_folder}")
        
        print(f"📊 AI report saved: {excel_file.name}")
        return excel_file
    
    def copy_ai_resumes(self, candidates: List[Dict[str, Any]], output_folder: Path, num_needed: int) -> None:
        """Copy resumes with AI-enhanced organization."""
        top_folder = output_folder / "top_recommendations"
        qualified_folder = output_folder / "qualified_candidates"
        all_folder = output_folder / "all_candidates"
        
        copied_top = 0
        copied_qualified = 0
        copied_all = 0
        
        for i, candidate in enumerate(candidates, 1):
            source_file = candidate['file_path']
            
            try:
                score = candidate.get('score', 0)
                name = candidate.get('name', 'Unknown').replace(' ', '_')
                safe_name = f"Rank{i:02d}_[{score:05.1f}]_{name}_{source_file.name}"
                
                # Copy to all folder
                dest_all = all_folder / safe_name
                shutil.copy2(source_file, dest_all)
                copied_all += 1
                
                # Copy to qualified if score >= 60
                if score >= 60:
                    dest_qualified = qualified_folder / safe_name
                    shutil.copy2(source_file, dest_qualified)
                    copied_qualified += 1
                
                # Copy to top folder if in top N and score >= 60
                if i <= num_needed and score >= 60:
                    dest_top = top_folder / safe_name
                    shutil.copy2(source_file, dest_top)
                    copied_top += 1
                    
            except Exception as e:
                print(f"❌ Error copying {source_file.name}: {e}")
        
        print(f"📁 File organization complete:")
        print(f"   • {copied_top} top recommendations")
        print(f"   • {copied_qualified} qualified candidates")
        print(f"   • {copied_all} total candidates")
    
    def run_ai_screening(self) -> None:
        """Main AI screening workflow."""
        print("=" * 70)
        print(f"    🤖 {self.resort_name.upper()} AI HIRING ASSISTANT")
        print("    Smart • Fast • Intelligent Candidate Selection")
        print("=" * 70)
        
        # Check for resume files
        resume_files = self.get_resume_files()
        
        if not resume_files:
            print(f"\n❌ No resume files found in: {self.input_dir}")
            print("\nPlease add PDF, DOCX, or TXT files to the 'input_resumes' folder and try again.")
            input("\nPress Enter to exit...")
            return
        
        print(f"\n✅ Found {len(resume_files)} resume files")
        
        # Get position and number needed
        print(f"\n🏨 What position are you hiring for at {self.resort_name}?")
        print("Examples: Front Desk Agent, Chef, Bartender, Manager, Security, Spa Therapist")
        position = input("Position: ").strip()
        
        if not position:
            print("❌ Please enter a job position.")
            return
        
        print(f"\n🎯 How many candidates do you need to interview?")
        print("AI will rank ALL candidates but highlight your top choices")
        try:
            num_needed = int(input("Number needed (default 5): ") or "5")
        except ValueError:
            num_needed = 5
        
        print(f"\n🤖 AI analyzing {len(resume_files)} candidates for: {position}")
        print(f"🎯 Targeting top {num_needed} recommendations")
        print("-" * 70)
        
        # AI process resumes
        candidates: List[Dict[str, Any]] = []
        for file_path in resume_files:
            candidate = self.process_resume_ai(file_path)
            candidates.append(candidate)
        
        # AI screen candidates
        scored_candidates = self.ai_screen_candidates(position, candidates, num_needed)
        
        # Create AI output
        output_folder = self.create_ai_output_folder(position, num_needed)
        excel_file = self.export_ai_results(scored_candidates, output_folder, position, num_needed)
        self.copy_ai_resumes(scored_candidates, output_folder, num_needed)
        
        # Show AI results
        print("\n" + "=" * 70)
        print(f"    🤖 {self.resort_name.upper()} AI ANALYSIS COMPLETE!")
        print("=" * 70)
        
        top_candidates = scored_candidates[:num_needed]
        excellent = [c for c in scored_candidates if c.get('score', 0) >= 80]
        qualified = [c for c in scored_candidates if c.get('score', 0) >= 60]
        
        print(f"\n📊 AI RESULTS FOR {position.upper()}:")
        print(f"   • Total candidates analyzed: {len(scored_candidates)}")
        print(f"   • Excellent candidates (80+): {len(excellent)}")
        print(f"   • Qualified for interview: {len(qualified)}")
        print(f"   • Your top {num_needed} recommendations: {min(num_needed, len(qualified))}")
        
        if qualified:
            print(f"\n🏆 AI TOP RECOMMENDATIONS FOR {self.resort_name.upper()}:")
            for i, candidate in enumerate(top_candidates[:5], 1):
                score = candidate.get('score', 0)
                name = candidate.get('name', 'Unknown')
                recommendation = candidate.get('ai_recommendation', 'No recommendation')
                phone = candidate.get('phone', 'No phone')
                print(f"   {i}. {name} - Score: {score:.1f} - {phone}")
                print(f"      🤖 AI: {recommendation}")
        
        print(f"\n📁 All results saved to: {output_folder}")
        print(f"📊 MAIN EXCEL FILE: {excel_file.name}")
        print(f"📄 Top {num_needed} picks: top_recommendations/ folder")
        print(f"📄 All qualified: qualified_candidates/ folder")
        
        print(f"\n💡 TO OPEN EXCEL FILE:")
        print(f"   • Double-click: {excel_file.name}")
        print(f"   • Right-click → Open with → Excel")
        print(f"   • Go to '📞 CANDIDATES TO CALL' sheet first")
        
        print(f"\n🏨 AI-powered hiring for {self.resort_name}!")
        print("🤖 Intelligent • Accurate • Ready to hire!")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    screener = RoyaltonAIScreener()
    screener.run_ai_screening()
