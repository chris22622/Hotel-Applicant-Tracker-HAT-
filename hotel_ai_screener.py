#!/usr/bin/env python3
"""
Hotel AI Resume Screener
One-click intelligent candidate selection with AI recommendations for hotels and resorts
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


class HotelAIScreener:
    """AI-powered resume screening system for hotels and resorts."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / "input_resumes"
        self.output_dir = self.base_dir / "screening_results"
        self.property_name = "Hotel"
        self.config = self.load_config()
        self.setup_directories()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = self.base_dir / "hotel_config.yaml"
        
        if yaml_available and config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)  # type: ignore[misc]
                print("✅ Loaded configuration from hotel_config.yaml")
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
                },
                'housekeeping_supervisor': {
                    'must_have_skills': ['housekeeping', 'supervision', 'time management', 'quality control', 'team leadership'],
                    'nice_to_have_skills': ['hotel housekeeping', 'inventory management', 'training', 'scheduling'],
                    'cultural_fit_keywords': ['detail oriented', 'reliable', 'organized', 'team leader'],
                    'experience_weight': 0.35, 'skills_weight': 0.3, 'cultural_fit_weight': 0.2, 'hospitality_weight': 0.15
                },
                'security_officer': {
                    'must_have_skills': ['security', 'surveillance', 'incident reporting', 'safety protocols', 'communication'],
                    'nice_to_have_skills': ['hotel security', 'CCTV', 'emergency response', 'guest relations'],
                    'cultural_fit_keywords': ['alert', 'responsible', 'professional', 'calm under pressure'],
                    'experience_weight': 0.4, 'skills_weight': 0.3, 'cultural_fit_weight': 0.15, 'hospitality_weight': 0.15
                },
                'guest_services_agent': {
                    'must_have_skills': ['customer service', 'communication', 'problem solving', 'computer skills', 'hospitality'],
                    'nice_to_have_skills': ['concierge', 'guest relations', 'tour booking', 'multilingual'],
                    'cultural_fit_keywords': ['helpful', 'friendly', 'professional', 'service oriented'],
                    'experience_weight': 0.3, 'skills_weight': 0.25, 'cultural_fit_weight': 0.25, 'hospitality_weight': 0.2
                }
            },
            'skill_synonyms': {
                'customer_service': ['guest service', 'client service', 'customer care', 'guest relations'],
                'communication': ['verbal communication', 'written communication', 'interpersonal', 'people skills'],
                'hotel_experience': ['hospitality', 'resort', 'accommodation', 'lodging'],
                'food_service': ['restaurant', 'dining', 'culinary', 'kitchen']
            },
            'ocr': {'enabled': True, 'dpi': 300},
            'output': {'excel_detailed': True, 'csv_export': True, 'create_folders': True}
        }
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create input_resumes README if it doesn't exist
        readme_file = self.input_dir / "README.txt"
        if not readme_file.exists():
            with open(readme_file, 'w') as f:
                f.write("Place resume files (.pdf, .docx, .txt) in this folder for processing.\n")
                f.write("The AI screener will automatically scan all files in this directory.\n")
    
    def get_resume_files(self) -> List[Path]:
        """Get all resume files from input directory."""
        extensions = ['*.pdf', '*.docx', '*.doc', '*.txt']
        files = []
        for ext in extensions:
            files.extend(self.input_dir.glob(ext))
        return sorted(files)
    
    def get_hotel_job_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive job requirements for hotel positions."""
        return {
            # Front of House Positions
            "Front Desk Agent": {
                "must_have_skills": ["customer service", "computer skills", "communication", "multitasking", "phone skills", "PMS systems"],
                "nice_to_have_skills": ["hotel experience", "Opera PMS", "guest services", "check-in/check-out", "reservations", "multilingual"],
                "cultural_fit_keywords": ["team player", "friendly", "professional", "positive attitude", "guest-focused", "helpful"],
                "experience_weight": 0.3, "skills_weight": 0.25, "cultural_fit_weight": 0.25, "hospitality_weight": 0.2
            },
            "Guest Services Agent": {
                "must_have_skills": ["customer service", "communication", "problem solving", "computer skills", "hospitality"],
                "nice_to_have_skills": ["concierge experience", "guest relations", "tour booking", "multilingual", "local knowledge"],
                "cultural_fit_keywords": ["helpful", "friendly", "professional", "service oriented", "proactive", "resourceful"],
                "experience_weight": 0.3, "skills_weight": 0.25, "cultural_fit_weight": 0.25, "hospitality_weight": 0.2
            },
            "Concierge": {
                "must_have_skills": ["customer service", "local knowledge", "communication", "organization", "problem solving"],
                "nice_to_have_skills": ["luxury hospitality", "event planning", "restaurant knowledge", "tour coordination"],
                "cultural_fit_keywords": ["sophisticated", "knowledgeable", "helpful", "professional", "well-connected"],
                "experience_weight": 0.35, "skills_weight": 0.3, "cultural_fit_weight": 0.2, "hospitality_weight": 0.15
            },
            
            # Food & Beverage
            "Executive Chef": {
                "must_have_skills": ["culinary arts", "kitchen management", "food safety", "menu planning", "cost control", "staff supervision"],
                "nice_to_have_skills": ["fine dining", "international cuisine", "dietary restrictions", "large volume cooking", "wine pairing"],
                "cultural_fit_keywords": ["creative", "leader", "organized", "passionate", "detail-oriented", "high standards"],
                "experience_weight": 0.4, "skills_weight": 0.3, "cultural_fit_weight": 0.15, "hospitality_weight": 0.15
            },
            "Sous Chef": {
                "must_have_skills": ["cooking", "food preparation", "kitchen operations", "food safety", "team work"],
                "nice_to_have_skills": ["menu development", "inventory management", "staff training", "multiple cuisines"],
                "cultural_fit_keywords": ["reliable", "creative", "team player", "adaptable", "efficient"],
                "experience_weight": 0.35, "skills_weight": 0.3, "cultural_fit_weight": 0.2, "hospitality_weight": 0.15
            },
            "Line Cook": {
                "must_have_skills": ["food preparation", "cooking techniques", "food safety", "speed", "accuracy"],
                "nice_to_have_skills": ["grill experience", "sauté", "prep work", "high volume", "special diets"],
                "cultural_fit_keywords": ["fast-paced", "reliable", "team player", "focused", "consistent"],
                "experience_weight": 0.3, "skills_weight": 0.35, "cultural_fit_weight": 0.2, "hospitality_weight": 0.15
            },
            "Bartender": {
                "must_have_skills": ["mixology", "customer service", "cash handling", "multitasking", "product knowledge"],
                "nice_to_have_skills": ["craft cocktails", "wine knowledge", "beer knowledge", "inventory", "POS systems"],
                "cultural_fit_keywords": ["personable", "energetic", "professional", "friendly", "entertaining"],
                "experience_weight": 0.3, "skills_weight": 0.3, "cultural_fit_weight": 0.25, "hospitality_weight": 0.15
            },
            "Server": {
                "must_have_skills": ["customer service", "food knowledge", "multitasking", "communication", "cash handling"],
                "nice_to_have_skills": ["fine dining", "wine service", "allergen knowledge", "upselling", "POS systems"],
                "cultural_fit_keywords": ["friendly", "attentive", "professional", "team player", "energetic"],
                "experience_weight": 0.25, "skills_weight": 0.3, "cultural_fit_weight": 0.25, "hospitality_weight": 0.2
            },
            
            # Housekeeping
            "Housekeeping Supervisor": {
                "must_have_skills": ["housekeeping", "supervision", "time management", "quality control", "team leadership", "scheduling"],
                "nice_to_have_skills": ["hotel housekeeping", "inventory management", "training", "budget management"],
                "cultural_fit_keywords": ["detail oriented", "reliable", "organized", "team leader", "efficient"],
                "experience_weight": 0.35, "skills_weight": 0.3, "cultural_fit_weight": 0.2, "hospitality_weight": 0.15
            },
            "Room Attendant": {
                "must_have_skills": ["cleaning", "attention to detail", "time management", "physical stamina", "organization"],
                "nice_to_have_skills": ["hotel cleaning", "laundry", "inventory", "guest interaction"],
                "cultural_fit_keywords": ["thorough", "reliable", "efficient", "professional", "hardworking"],
                "experience_weight": 0.25, "skills_weight": 0.35, "cultural_fit_weight": 0.25, "hospitality_weight": 0.15
            },
            
            # Security & Maintenance
            "Security Officer": {
                "must_have_skills": ["security", "surveillance", "incident reporting", "safety protocols", "communication"],
                "nice_to_have_skills": ["hotel security", "CCTV", "emergency response", "guest relations", "crowd control"],
                "cultural_fit_keywords": ["alert", "responsible", "professional", "calm under pressure", "observant"],
                "experience_weight": 0.4, "skills_weight": 0.3, "cultural_fit_weight": 0.15, "hospitality_weight": 0.15
            },
            "Maintenance Technician": {
                "must_have_skills": ["mechanical skills", "electrical", "plumbing", "HVAC", "troubleshooting", "repair"],
                "nice_to_have_skills": ["hotel maintenance", "preventive maintenance", "pool maintenance", "carpentry"],
                "cultural_fit_keywords": ["handy", "problem solver", "reliable", "safety conscious", "proactive"],
                "experience_weight": 0.4, "skills_weight": 0.35, "cultural_fit_weight": 0.15, "hospitality_weight": 0.1
            },
            
            # Sales & Events
            "Sales Manager": {
                "must_have_skills": ["sales", "customer relationship", "communication", "negotiation", "organization"],
                "nice_to_have_skills": ["hotel sales", "event planning", "group bookings", "revenue management", "CRM"],
                "cultural_fit_keywords": ["persuasive", "relationship builder", "goal oriented", "professional", "confident"],
                "experience_weight": 0.35, "skills_weight": 0.3, "cultural_fit_weight": 0.2, "hospitality_weight": 0.15
            },
            "Event Coordinator": {
                "must_have_skills": ["event planning", "organization", "communication", "time management", "attention to detail"],
                "nice_to_have_skills": ["hotel events", "wedding planning", "vendor management", "budget management"],
                "cultural_fit_keywords": ["organized", "creative", "calm under pressure", "detail oriented", "flexible"],
                "experience_weight": 0.3, "skills_weight": 0.3, "cultural_fit_weight": 0.25, "hospitality_weight": 0.15
            }
        }
    
    def extract_text_simple(self, file_path: str) -> str:
        """Extract text from various file formats with OCR fallback."""
        try:
            text = ""
            file_path_obj = Path(file_path)
            
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                print(f"📄 Text file read: {len(text)} characters")
                
            elif file_path.endswith(('.docx', '.doc')):
                try:
                    from docx import Document  # type: ignore[misc]
                    doc = Document(file_path)
                    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    print(f"📄 DOCX text extracted: {len(text)} characters")
                except ImportError:
                    print("⚠ python-docx not available for .docx files")
                    text = ""
                
            elif file_path.endswith('.pdf'):
                # Try pdfminer first, then PyPDF2 as fallback
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
                                if page_text:
                                    text += str(page_text)
                        print(f"📄 PyPDF2 text extracted: {len(text)} characters")
                    except Exception:
                        text = ""
                
                # Enhanced OCR fallback detection
                needs_ocr = (
                    len(text.strip()) < 50 or  # Very little text extracted
                    (file_path_obj.stat().st_size > 1024 and len(text) < file_path_obj.stat().st_size * 0.01) or  # Low text-to-size ratio
                    len([c for c in text if c.isalnum()]) < len(text) * 0.5  # High non-alphanumeric ratio
                )
                
                if needs_ocr and (ocr_available and tesseract_available):
                    print("🔍 Low text extraction detected - trying OCR fallback...")
                    ocr_text = self._extract_text_with_ocr(file_path)
                    if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                        print(f"✅ OCR provided better results: {len(ocr_text)} vs {len(text)} characters")
                        text = ocr_text
                elif needs_ocr:
                    print("⚠ Document may need OCR but Tesseract not available")
                
            # Clean and return text
            if not text.strip():
                print(f"⚠ No text extracted from {file_path}")
                return ""
            
            return text
            
        except Exception as e:
            print(f"❌ Error extracting text from {file_path}: {e}")
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
        # Common location patterns
        location_patterns = [
            r'(?:Address|Location|Lives?|Resides?|Based)\s*:?\s*([A-Za-z\s,]+(?:State|Province|Country)?)',
            r'([A-Za-z\s]+,\s*[A-Z]{2,3}(?:\s+\d{5})?)',  # City, State ZIP
            r'([A-Za-z\s]+,\s*[A-Za-z\s]+)',  # City, Country
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return "Not specified"
    
    def extract_ai_candidate_profile(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive candidate information using AI-enhanced parsing."""
        profile = {
            'name': 'Not found',
            'email': 'Not found', 
            'phone': 'Not found',
            'location': self.extract_location(text),
            'experience_years': 0,
            'skills': [],
            'education': [],
            'work_history': [],
            'certifications': []
        }
        
        # Extract name (first few lines, common patterns)
        lines = [line.strip() for line in text.split('\n') if line.strip()][:5]
        for line in lines:
            if len(line.split()) <= 4 and any(char.isupper() for char in line):
                if not any(keyword in line.lower() for keyword in ['email', 'phone', 'address', 'resume']):
                    profile['name'] = line
                    break
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            profile['email'] = emails[0]
        
        # Extract phone
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]\d{3}[-.\s]\d{4}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                profile['phone'] = phones[0]
                break
        
        # Extract experience years
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in'
        ]
        years = []
        for pattern in exp_patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches if match.isdigit()])
        if years:
            profile['experience_years'] = max(years)
        
        # Extract skills using smart detection
        skill_keywords = [
            'customer service', 'communication', 'leadership', 'management', 'sales',
            'marketing', 'accounting', 'finance', 'computer', 'microsoft office',
            'excel', 'word', 'powerpoint', 'hotel', 'hospitality', 'restaurant',
            'food service', 'cleaning', 'housekeeping', 'maintenance', 'security',
            'front desk', 'reception', 'phone', 'multitasking', 'organization'
        ]
        
        found_skills = []
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        profile['skills'] = list(set(found_skills))
        
        return profile
    
    def analyze_role_relevant_experience(self, text: str, position: str, requirements: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze if candidate has role-specific experience."""
        text_lower = text.lower()
        experience_analysis = {}
        
        # Check for direct role experience
        role_keywords = {
            'front desk': ['front desk', 'reception', 'check-in', 'check-out', 'guest services'],
            'chef': ['chef', 'cook', 'kitchen', 'culinary', 'food preparation'],
            'housekeeping': ['housekeeping', 'cleaning', 'room attendant', 'laundry'],
            'security': ['security', 'guard', 'surveillance', 'safety'],
            'maintenance': ['maintenance', 'repair', 'technical', 'facilities'],
            'server': ['server', 'waitress', 'waiter', 'food service', 'restaurant'],
            'bartender': ['bartender', 'mixology', 'bar', 'cocktails', 'drinks']
        }
        
        position_lower = position.lower()
        relevant_keywords = []
        for role, keywords in role_keywords.items():
            if role in position_lower:
                relevant_keywords.extend(keywords)
        
        # Check for hotel/hospitality experience
        hospitality_keywords = ['hotel', 'resort', 'hospitality', 'guest', 'service', 'accommodation']
        experience_analysis['has_hospitality_experience'] = any(keyword in text_lower for keyword in hospitality_keywords)
        
        # Check for direct role experience
        experience_analysis['has_direct_role_experience'] = any(keyword in text_lower for keyword in relevant_keywords)
        
        # Check for customer service experience
        service_keywords = ['customer service', 'client service', 'guest service', 'customer care']
        experience_analysis['has_customer_service'] = any(keyword in text_lower for keyword in service_keywords)
        
        return experience_analysis
    
    def smart_skill_match(self, skill: str, text: str, skills: List[str], position: str) -> bool:
        """Enhanced skill matching with context awareness."""
        text_lower = text.lower()
        skill_lower = skill.lower()
        
        # Direct match
        if skill_lower in text_lower:
            return True
        
        # Check synonyms from config
        synonyms = self.config.get('skill_synonyms', {})
        for skill_key, skill_synonyms in synonyms.items():
            if skill_lower == skill_key or skill_lower in skill_synonyms:
                if any(synonym in text_lower for synonym in skill_synonyms):
                    return True
        
        # Context-aware matching
        if 'customer' in skill_lower and any(term in text_lower for term in ['guest', 'client', 'service']):
            return True
        
        if 'computer' in skill_lower and any(term in text_lower for term in ['microsoft', 'excel', 'word', 'software']):
            return True
        
        return False
    
    def detect_role_specific_skills(self, text: str, position: str) -> List[str]:
        """Detect skills that are specifically relevant to the position."""
        detected_skills = []
        text_lower = text.lower()
        
        # Get position-specific skill sets
        skill_sets = {
            'front desk': ['PMS systems', 'Opera', 'reservations', 'check-in', 'check-out', 'guest relations'],
            'chef': ['menu planning', 'food safety', 'kitchen management', 'culinary arts', 'cost control'],
            'housekeeping': ['room cleaning', 'laundry', 'inventory', 'quality control', 'time management'],
            'security': ['surveillance', 'incident reporting', 'CCTV', 'emergency response', 'patrol'],
            'maintenance': ['HVAC', 'plumbing', 'electrical', 'preventive maintenance', 'troubleshooting'],
            'server': ['food service', 'wine knowledge', 'upselling', 'POS systems', 'allergen awareness'],
            'bartender': ['mixology', 'cocktail preparation', 'wine knowledge', 'inventory management']
        }
        
        position_lower = position.lower()
        for role, skills in skill_sets.items():
            if role in position_lower:
                for skill in skills:
                    if skill.lower() in text_lower:
                        detected_skills.append(skill)
        
        return detected_skills
    
    def calculate_ai_score(self, candidate: Dict[str, Any], requirements: Dict[str, Any], position: str) -> Tuple[float, List[str], List[str], str]:
        """Calculate comprehensive AI-based candidate score with detailed feedback."""
        score = 0.0
        strengths = []
        weaknesses = []
        
        # Get weights from requirements or use defaults
        exp_weight = requirements.get('experience_weight', 0.3)
        skills_weight = requirements.get('skills_weight', 0.25)
        cultural_weight = requirements.get('cultural_fit_weight', 0.25)
        hospitality_weight = requirements.get('hospitality_weight', 0.2)
        
        # 1. Experience Score (based on years and relevance)
        exp_years = candidate.get('experience_years', 0)
        exp_score = min(exp_years / 5.0, 1.0)  # Max score at 5+ years
        
        # Boost for relevant experience
        if candidate.get('role_experience', {}).get('has_direct_role_experience', False):
            exp_score = min(exp_score * 1.3, 1.0)
            strengths.append(f"Direct {position.lower()} experience")
        elif candidate.get('role_experience', {}).get('has_customer_service', False):
            exp_score = min(exp_score * 1.1, 1.0)
            strengths.append("Customer service background")
        
        if exp_years < 1:
            weaknesses.append("Limited work experience")
        
        score += exp_score * exp_weight
        
        # 2. Skills Score
        must_have = requirements.get('must_have_skills', [])
        nice_to_have = requirements.get('nice_to_have_skills', [])
        candidate_skills = [skill.lower() for skill in candidate.get('skills', [])]
        
        must_have_count = sum(1 for skill in must_have if any(skill.lower() in cs for cs in candidate_skills))
        nice_to_have_count = sum(1 for skill in nice_to_have if any(skill.lower() in cs for cs in candidate_skills))
        
        skills_score = 0
        if must_have:
            must_have_ratio = must_have_count / len(must_have)
            skills_score += must_have_ratio * 0.7  # 70% weight for must-have skills
            
            if must_have_ratio >= 0.8:
                strengths.append(f"Strong match for required skills ({must_have_count}/{len(must_have)})")
            elif must_have_ratio < 0.5:
                weaknesses.append(f"Missing key skills ({must_have_count}/{len(must_have)} required)")
        
        if nice_to_have:
            nice_to_have_ratio = nice_to_have_count / len(nice_to_have)
            skills_score += nice_to_have_ratio * 0.3  # 30% weight for nice-to-have skills
            
            if nice_to_have_count > 0:
                strengths.append(f"Additional relevant skills ({nice_to_have_count} bonus skills)")
        
        score += skills_score * skills_weight
        
        # 3. Cultural Fit Score
        cultural_keywords = requirements.get('cultural_fit_keywords', [])
        resume_text = candidate.get('raw_text', '').lower()
        cultural_matches = sum(1 for keyword in cultural_keywords if keyword.lower() in resume_text)
        
        cultural_score = 0
        if cultural_keywords:
            cultural_score = cultural_matches / len(cultural_keywords)
            if cultural_matches >= len(cultural_keywords) * 0.6:
                strengths.append("Good cultural fit indicators")
            elif cultural_matches == 0:
                weaknesses.append("Limited cultural fit evidence")
        
        score += cultural_score * cultural_weight
        
        # 4. Hospitality Experience Score
        hospitality_score = 0
        if candidate.get('role_experience', {}).get('has_hospitality_experience', False):
            hospitality_score = 1.0
            strengths.append("Hotel/hospitality experience")
        elif candidate.get('role_experience', {}).get('has_customer_service', False):
            hospitality_score = 0.6
            strengths.append("Service industry background")
        else:
            weaknesses.append("No hospitality experience")
        
        score += hospitality_score * hospitality_weight
        
        # Generate recommendation
        if score >= 0.8:
            recommendation = "Highly Recommended - Excellent match for the position"
        elif score >= 0.6:
            recommendation = "Recommended - Good candidate with minor gaps"
        elif score >= 0.4:
            recommendation = "Consider - Meets basic requirements"
        else:
            recommendation = "Not Recommended - Significant gaps in requirements"
        
        return min(score, 1.0), strengths, weaknesses, recommendation
    
    def process_resume_ai(self, file_path: Path) -> Dict[str, Any]:
        """Process a single resume with AI-enhanced analysis."""
        print(f"\n📄 Processing: {file_path.name}")
        
        # Extract text
        text = self.extract_text_simple(str(file_path))
        if not text:
            return {
                'file_name': file_path.name,
                'error': 'Could not extract text from file'
            }
        
        # Extract candidate profile
        profile = self.extract_ai_candidate_profile(text)
        profile['file_name'] = file_path.name
        profile['file_path'] = str(file_path)
        profile['raw_text'] = text
        
        print(f"✅ Extracted profile for: {profile['name']}")
        return profile
    
    def ai_screen_candidates(self, position: str, candidates: List[Dict[str, Any]], num_needed: int = 5) -> List[Dict[str, Any]]:
        """Screen candidates using AI with position-specific intelligence."""
        print(f"\n🤖 AI Screening for {position} (need {num_needed} candidates)")
        print("=" * 60)
        
        # Get job requirements
        job_intelligence = self.get_hotel_job_intelligence()
        requirements = job_intelligence.get(position, {})
        
        if not requirements:
            print(f"⚠ No specific requirements found for '{position}', using general hospitality criteria")
            requirements = {
                'must_have_skills': ['customer service', 'communication', 'hospitality'],
                'nice_to_have_skills': ['hotel experience', 'team work'],
                'cultural_fit_keywords': ['professional', 'friendly', 'reliable'],
                'experience_weight': 0.3, 'skills_weight': 0.3, 'cultural_fit_weight': 0.2, 'hospitality_weight': 0.2
            }
        
        # Analyze each candidate
        scored_candidates = []
        for candidate in candidates:
            if 'error' in candidate:
                continue
                
            # Analyze role-specific experience
            candidate['role_experience'] = self.analyze_role_relevant_experience(
                candidate.get('raw_text', ''), position, requirements
            )
            
            # Detect position-specific skills
            role_skills = self.detect_role_specific_skills(candidate.get('raw_text', ''), position)
            candidate['role_specific_skills'] = role_skills
            
            # Calculate AI score
            score, strengths, weaknesses, recommendation = self.calculate_ai_score(candidate, requirements, position)
            
            candidate.update({
                'ai_score': round(score * 100, 1),
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendation': recommendation,
                'position_applied': position
            })
            
            scored_candidates.append(candidate)
            print(f"📊 {candidate['name']}: {candidate['ai_score']:.1f}% - {recommendation}")
        
        # Sort by score and return top candidates
        scored_candidates.sort(key=lambda x: x['ai_score'], reverse=True)
        
        print(f"\n🎯 Top {min(num_needed, len(scored_candidates))} candidates selected")
        return scored_candidates[:num_needed]
    
    def create_ai_output_folder(self, position: str, num_needed: int) -> Path:
        """Create organized output folder for screening results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{position.replace(' ', '_')}_{num_needed}_candidates_{timestamp}"
        output_folder = self.output_dir / folder_name
        output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (output_folder / "resumes").mkdir(exist_ok=True)
        (output_folder / "csv_exports").mkdir(exist_ok=True)
        
        return output_folder
    
    def export_ai_results(self, candidates: List[Dict[str, Any]], output_folder: Path, position: str, num_needed: int) -> Path:
        """Export screening results to Excel with HR-friendly format and CSV fallback."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare data for export
        export_data = []
        for i, candidate in enumerate(candidates, 1):
            export_data.append({
                'Rank': i,
                'Name': candidate.get('name', 'Not found'),
                'Phone': candidate.get('phone', 'Not found'),
                'Email': candidate.get('email', 'Not found'),
                'Location': candidate.get('location', 'Not specified'),
                'AI Score (%)': candidate.get('ai_score', 0),
                'Recommendation': candidate.get('recommendation', 'No recommendation'),
                'Experience (Years)': candidate.get('experience_years', 0),
                'Key Strengths': '; '.join(candidate.get('strengths', [])),
                'Areas for Development': '; '.join(candidate.get('weaknesses', [])),
                'Skills': '; '.join(candidate.get('skills', [])),
                'Role-Specific Skills': '; '.join(candidate.get('role_specific_skills', [])),
                'File Name': candidate.get('file_name', 'Unknown')
            })
        
        df = pd.DataFrame(export_data)
        
        # Try Excel export first
        excel_file = output_folder / f"Hotel_Screening_Results_{position.replace(' ', '_')}.xlsx"
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name='Screening Results', index=False)
                
                # Contact sheet for HR
                contact_df = df[['Rank', 'Name', 'Phone', 'Email', 'AI Score (%)', 'Recommendation']].copy()
                contact_df.to_excel(writer, sheet_name='Contact List', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Position', 'Candidates Needed', 'Candidates Screened', 'Average Score', 'Top Score', 'Generated'],
                    'Value': [position, num_needed, len(candidates), 
                             f"{df['AI Score (%)'].mean():.1f}%", f"{df['AI Score (%)'].max():.1f}%", timestamp]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"✅ Excel report created: {excel_file.name}")
            return excel_file
            
        except Exception as e:
            print(f"⚠ Excel export failed ({e}), creating CSV fallback...")
            
            # CSV fallback - main results
            csv_file = output_folder / f"Hotel_Screening_Results_{position.replace(' ', '_')}.csv"
            df.to_csv(csv_file, index=False)
            
            # CSV contact list
            contact_csv = output_folder / "csv_exports" / f"Top_{num_needed}_Contacts_{position.replace(' ', '_')}.csv"
            contact_df = df[['Rank', 'Name', 'Phone', 'Email', 'AI Score (%)', 'Recommendation']].copy()
            contact_df.to_csv(contact_csv, index=False)
            
            # CSV qualified candidates list
            qualified_csv = output_folder / "csv_exports" / f"Qualified_Candidates_{position.replace(' ', '_')}.csv"
            qualified_df = df[df['AI Score (%)'] >= 60].copy()  # Candidates with 60%+ score
            qualified_df.to_csv(qualified_csv, index=False)
            
            print(f"✅ CSV reports created: {csv_file.name}")
            print(f"📋 Additional CSV exports in csv_exports/ folder")
            return csv_file
    
    def copy_ai_resumes(self, candidates: List[Dict[str, Any]], output_folder: Path, num_needed: int) -> None:
        """Copy selected candidate resumes to output folder."""
        resumes_folder = output_folder / "resumes"
        
        for i, candidate in enumerate(candidates[:num_needed], 1):
            if 'file_path' in candidate:
                source_file = Path(candidate['file_path'])
                if source_file.exists():
                    # Create descriptive filename
                    score = candidate.get('ai_score', 0)
                    name = candidate.get('name', 'Unknown').replace(' ', '_')
                    dest_file = resumes_folder / f"{i:02d}_{name}_{score:.0f}pct_{source_file.name}"
                    
                    try:
                        shutil.copy2(source_file, dest_file)
                        print(f"📁 Copied: {dest_file.name}")
                    except Exception as e:
                        print(f"⚠ Failed to copy {source_file.name}: {e}")
    
    def run_ai_screening(self) -> None:
        """Main screening workflow with AI intelligence."""
        print("=" * 80)
        print("🏨 HOTEL AI RESUME SCREENER")
        print("   Intelligent candidate selection for hotels and resorts")
        print("=" * 80)
        
        # Check for resumes
        resume_files = self.get_resume_files()
        if not resume_files:
            print("❌ No resume files found in input_resumes folder")
            print("📁 Please add PDF, DOCX, or TXT files to the input_resumes folder")
            input("Press Enter to exit...")
            return
        
        print(f"📄 Found {len(resume_files)} resume files")
        
        # Get position and requirements
        print("\n🎯 POSITION SELECTION")
        print("Available hotel positions:")
        positions = list(self.get_hotel_job_intelligence().keys())
        for i, pos in enumerate(positions, 1):
            print(f"  {i:2d}. {pos}")
        
        position = input("\nEnter position name (or type custom): ").strip()
        if not position:
            print("❌ Position is required")
            input("Press Enter to exit...")
            return
        
        # Check if it's a numbered selection
        if position.isdigit() and 1 <= int(position) <= len(positions):
            position = positions[int(position) - 1]
        
        # Get number of candidates needed
        try:
            num_needed = int(input(f"\nHow many candidates needed for {position}? [5]: ") or "5")
        except ValueError:
            num_needed = 5
        
        print(f"\n🔍 Screening {len(resume_files)} candidates for {position}")
        print(f"🎯 Selecting top {num_needed} candidates")
        
        # Process all resumes
        all_candidates = []
        for file_path in resume_files:
            candidate = self.process_resume_ai(file_path)
            all_candidates.append(candidate)
        
        # AI screening
        top_candidates = self.ai_screen_candidates(position, all_candidates, num_needed)
        
        if not top_candidates:
            print("❌ No suitable candidates found")
            input("Press Enter to exit...")
            return
        
        # Create output folder and export results
        output_folder = self.create_ai_output_folder(position, num_needed)
        results_file = self.export_ai_results(top_candidates, output_folder, position, num_needed)
        self.copy_ai_resumes(top_candidates, output_folder, num_needed)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 SCREENING COMPLETE!")
        print("=" * 80)
        print(f"📊 Results saved to: {results_file}")
        print(f"📁 Resume copies in: {output_folder / 'resumes'}")
        print(f"🎯 Top {len(top_candidates)} candidates ready for interview")
        
        print("\n🏆 TOP CANDIDATES:")
        for i, candidate in enumerate(top_candidates, 1):
            print(f"  {i}. {candidate['name']} - {candidate['ai_score']:.1f}% ({candidate['recommendation']})")
        
        print(f"\n📂 Open results folder: {output_folder}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    screener = HotelAIScreener()
    screener.run_ai_screening()
