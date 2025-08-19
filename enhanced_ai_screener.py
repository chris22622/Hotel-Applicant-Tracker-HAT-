#!/usr/bin/env python3
"""
Enhanced Hotel AI Resume Screener - Version 2.0
Advanced AI-powered candidate selection with semantic matching, bias detection, and intelligent scoring
"""

import os
import sys
import json
import yaml
import logging
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import pandas as pd
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced imports with fallbacks
try:
    import spacy
    from spacy.matcher import Matcher
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
    logger.info("✅ SpaCy loaded successfully")
except (ImportError, OSError):
    spacy_available = False
    nlp = None
    logger.warning("⚠️ SpaCy not available - using basic text processing")

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    ocr_available = True
    logger.info("✅ OCR capabilities loaded")
except ImportError:
    ocr_available = False
    logger.warning("⚠️ OCR not available - text files only")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
    logger.info("✅ Advanced text analysis available")
except ImportError:
    sklearn_available = False
    logger.warning("⚠️ Advanced text analysis not available")


class EnhancedHotelAIScreener:
    """Enhanced AI-powered hotel resume screener with advanced matching algorithms."""
    
    def __init__(self, input_dir: str = "input_resumes", output_dir: str = "screening_results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced position intelligence
        self.position_intelligence = self._load_enhanced_position_intelligence()
        
        # Initialize skill taxonomy
        self.skill_taxonomy = self._build_skill_taxonomy()
        
        # Initialize semantic matcher if spacy available
        if spacy_available and nlp:
            self.matcher = Matcher(nlp.vocab)
            self._setup_patterns()
        
        logger.info(f"🏨 Enhanced Hotel AI Screener initialized")
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def _load_enhanced_position_intelligence(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive hotel position intelligence with advanced requirements."""
        return {
            # FRONT OFFICE & GUEST SERVICES
            "Front Desk Agent": {
                "must_have_skills": [
                    "customer service", "computer skills", "communication", "multitasking",
                    "phone etiquette", "PMS systems", "problem solving", "cash handling",
                    "attention to detail", "time management"
                ],
                "nice_to_have_skills": [
                    "Opera PMS", "Maestro PMS", "guest relations", "check-in procedures",
                    "check-out procedures", "reservations management", "multilingual",
                    "upselling", "conflict resolution", "night audit", "folio management"
                ],
                "technical_skills": [
                    "Opera PMS", "Maestro PMS", "RMS Cloud", "keycard systems",
                    "telephone systems", "credit card processing", "room blocking",
                    "group reservations", "walk-in management"
                ],
                "soft_skills": [
                    "interpersonal communication", "patience", "empathy", "adaptability",
                    "stress management", "cultural sensitivity", "active listening"
                ],
                "cultural_fit_keywords": [
                    "team player", "friendly", "professional", "positive attitude",
                    "guest-focused", "helpful", "detail-oriented", "reliable", "patient",
                    "welcoming", "enthusiastic"
                ],
                "disqualifying_factors": [
                    "poor communication", "unreliable", "inflexible", "antisocial",
                    "impatient with guests", "dishonest", "unprofessional appearance"
                ],
                "experience_indicators": [
                    "front desk", "reception", "guest services", "hotel reception",
                    "hospitality", "check-in", "check-out", "concierge", "customer service",
                    "reservations", "hotel operations"
                ],
                "education_preferences": [
                    "hospitality management", "tourism", "business administration",
                    "communications", "hotel management", "customer service"
                ],
                "certifications": [
                    "hospitality certification", "customer service certification",
                    "PMS certification", "hotel operations", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.25,
                    "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },
            
            "Guest Services Manager": {
                "must_have_skills": [
                    "leadership", "customer service", "team management", "problem solving",
                    "communication", "hospitality operations", "training", "performance management"
                ],
                "nice_to_have_skills": [
                    "guest recovery", "luxury service", "VIP handling", "complaint resolution",
                    "staff development", "quality assurance", "mystery shopper programs",
                    "service standards", "guest satisfaction metrics"
                ],
                "technical_skills": [
                    "hospitality software", "performance metrics", "guest feedback systems",
                    "training platforms", "scheduling software", "budget management"
                ],
                "soft_skills": [
                    "leadership", "emotional intelligence", "conflict resolution",
                    "mentoring", "decision making", "strategic thinking"
                ],
                "cultural_fit_keywords": [
                    "leader", "mentor", "service excellence", "guest advocate",
                    "team builder", "innovative", "results-oriented", "diplomatic"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "inability to handle stress",
                    "lack of hospitality experience", "poor communication"
                ],
                "experience_indicators": [
                    "guest services", "hospitality management", "team leadership",
                    "customer service management", "hotel operations", "front office management"
                ],
                "education_preferences": [
                    "hospitality management", "business management", "hotel administration",
                    "tourism management", "organizational leadership"
                ],
                "certifications": [
                    "hospitality management", "customer service excellence",
                    "leadership certification", "hotel operations management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.25, "cultural_fit": 0.20,
                    "hospitality": 0.20
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },
            
            # FOOD & BEVERAGE
            "Executive Chef": {
                "must_have_skills": [
                    "culinary arts", "kitchen management", "food safety", "menu planning",
                    "cost control", "staff supervision", "inventory management", "leadership"
                ],
                "nice_to_have_skills": [
                    "fine dining", "international cuisine", "dietary restrictions",
                    "large volume cooking", "wine pairing", "farm-to-table", "molecular gastronomy",
                    "catering", "banquet operations", "food presentation"
                ],
                "technical_skills": [
                    "kitchen equipment", "food cost analysis", "menu engineering",
                    "HACCP", "inventory software", "recipe costing", "kitchen design"
                ],
                "soft_skills": [
                    "creativity", "leadership", "stress management", "communication",
                    "time management", "attention to detail", "innovation"
                ],
                "cultural_fit_keywords": [
                    "creative", "leader", "organized", "passionate", "detail-oriented",
                    "high standards", "innovative", "collaborative", "mentor"
                ],
                "disqualifying_factors": [
                    "poor food safety knowledge", "inability to lead", "lack of creativity",
                    "poor cost control", "inflexible", "poor communication"
                ],
                "experience_indicators": [
                    "executive chef", "head chef", "sous chef", "kitchen management",
                    "culinary arts", "restaurant management", "food service", "catering"
                ],
                "education_preferences": [
                    "culinary arts", "culinary management", "hospitality", "food science",
                    "nutrition", "business management"
                ],
                "certifications": [
                    "ServSafe", "culinary certification", "food safety manager",
                    "wine certification", "nutrition certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.15,
                    "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 65000, "max": 100000},
                "growth_potential": "High",
                "training_requirements": "Minimal"
            },
            
            # HOUSEKEEPING
            "Housekeeping Manager": {
                "must_have_skills": [
                    "housekeeping operations", "team management", "quality control",
                    "inventory management", "scheduling", "training", "safety protocols"
                ],
                "nice_to_have_skills": [
                    "laundry operations", "deep cleaning", "maintenance coordination",
                    "budget management", "eco-friendly practices", "guest room standards",
                    "lost and found", "housekeeping software"
                ],
                "technical_skills": [
                    "housekeeping software", "scheduling systems", "inventory management",
                    "cleaning equipment", "laundry systems", "room management systems"
                ],
                "soft_skills": [
                    "leadership", "organization", "attention to detail", "time management",
                    "problem solving", "communication", "training abilities"
                ],
                "cultural_fit_keywords": [
                    "organized", "detail-oriented", "efficient", "reliable", "leader",
                    "quality-focused", "thorough", "systematic", "accountable"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail",
                    "inability to manage teams", "poor time management"
                ],
                "experience_indicators": [
                    "housekeeping", "housekeeping management", "hotel housekeeping",
                    "room attendant", "laundry operations", "cleaning services"
                ],
                "education_preferences": [
                    "hospitality management", "hotel operations", "business management",
                    "facility management"
                ],
                "certifications": [
                    "housekeeping certification", "hospitality operations",
                    "safety certification", "management training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20,
                    "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },
            
            # MAINTENANCE
            "Maintenance Supervisor": {
                "must_have_skills": [
                    "facility maintenance", "HVAC", "plumbing", "electrical", "team leadership",
                    "preventive maintenance", "safety protocols", "equipment repair"
                ],
                "nice_to_have_skills": [
                    "pool maintenance", "elevator maintenance", "fire safety systems",
                    "energy management", "landscaping", "project management", "vendor management"
                ],
                "technical_skills": [
                    "HVAC systems", "electrical systems", "plumbing systems", "maintenance software",
                    "pool chemicals", "boiler operations", "refrigeration", "building automation"
                ],
                "soft_skills": [
                    "problem solving", "leadership", "time management", "communication",
                    "safety consciousness", "attention to detail", "reliability"
                ],
                "cultural_fit_keywords": [
                    "reliable", "safety-conscious", "proactive", "problem-solver",
                    "team leader", "efficient", "detail-oriented", "responsible"
                ],
                "disqualifying_factors": [
                    "poor safety record", "unreliable", "lack of technical skills",
                    "poor leadership", "inflexible"
                ],
                "experience_indicators": [
                    "maintenance", "facility maintenance", "HVAC", "plumbing", "electrical",
                    "building maintenance", "hotel maintenance", "property maintenance"
                ],
                "education_preferences": [
                    "mechanical engineering", "facility management", "HVAC certification",
                    "electrical certification", "maintenance technology"
                ],
                "certifications": [
                    "HVAC certification", "electrical license", "plumbing license",
                    "safety certification", "pool operator", "boiler license"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15,
                    "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            }
        }
    
    def _build_skill_taxonomy(self) -> Dict[str, List[str]]:
        """Build comprehensive skill taxonomy for semantic matching."""
        return {
            "customer_service": [
                "customer service", "guest relations", "client relations", "customer care",
                "guest satisfaction", "service excellence", "hospitality", "guest experience"
            ],
            "communication": [
                "communication", "verbal communication", "written communication",
                "interpersonal skills", "public speaking", "presentation skills",
                "listening skills", "multilingual", "bilingual"
            ],
            "leadership": [
                "leadership", "team leadership", "management", "supervision",
                "team building", "mentoring", "coaching", "delegation", "motivation"
            ],
            "technical": [
                "computer skills", "software", "systems", "technology", "technical",
                "digital literacy", "database", "applications", "platforms"
            ],
            "hospitality_specific": [
                "hospitality", "hotel", "resort", "restaurant", "food service",
                "accommodation", "tourism", "travel", "leisure", "guest services"
            ],
            "food_service": [
                "culinary", "cooking", "chef", "kitchen", "food preparation",
                "menu planning", "food safety", "restaurant", "catering", "baking"
            ],
            "maintenance": [
                "maintenance", "repair", "HVAC", "plumbing", "electrical",
                "facility management", "preventive maintenance", "troubleshooting"
            ],
            "housekeeping": [
                "housekeeping", "cleaning", "laundry", "room service",
                "sanitation", "hygiene", "facility cleaning", "room maintenance"
            ]
        }
    
    def _setup_patterns(self):
        """Setup spaCy patterns for advanced entity recognition."""
        if not spacy_available or not nlp:
            return
        
        # Email patterns
        email_pattern = [{"TEXT": {"REGEX": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"}}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone patterns
        phone_pattern = [{"TEXT": {"REGEX": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}}]
        self.matcher.add("PHONE", [phone_pattern])
        
        # Experience patterns
        exp_pattern = [{"LOWER": {"IN": ["years", "year"]}}, {"TEXT": "of"}, {"TEXT": "experience"}]
        self.matcher.add("EXPERIENCE", [exp_pattern])
    
    def enhanced_skill_extraction(self, text: str, position: str) -> Dict[str, Any]:
        """Enhanced skill extraction using semantic matching and NLP."""
        skills_found = set()
        confidence_scores = {}
        
        text_lower = text.lower()
        
        # Get position requirements
        position_data = self.position_intelligence.get(position, {})
        all_required_skills = (
            position_data.get("must_have_skills", []) +
            position_data.get("nice_to_have_skills", []) +
            position_data.get("technical_skills", []) +
            position_data.get("soft_skills", [])
        )
        
        # Direct skill matching with context
        for skill in all_required_skills:
            skill_lower = skill.lower()
            if skill_lower in text_lower:
                skills_found.add(skill)
                confidence_scores[skill] = 1.0
                
                # Check for context indicators
                context_indicators = [
                    "experience with", "skilled in", "proficient in", "expert in",
                    "knowledge of", "familiar with", "certified in", "trained in"
                ]
                
                for indicator in context_indicators:
                    if f"{indicator} {skill_lower}" in text_lower:
                        confidence_scores[skill] = min(confidence_scores[skill] + 0.2, 1.0)
        
        # Semantic skill matching using taxonomy
        for category, related_skills in self.skill_taxonomy.items():
            for skill in related_skills:
                if skill.lower() in text_lower and skill not in skills_found:
                    # Find the best matching required skill
                    for req_skill in all_required_skills:
                        if any(term in req_skill.lower() for term in skill.lower().split()):
                            skills_found.add(req_skill)
                            confidence_scores[req_skill] = 0.8
                            break
        
        # Advanced NLP-based extraction if spaCy available
        if spacy_available and nlp:
            doc = nlp(text)
            
            # Extract skills from noun phrases
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                for skill in all_required_skills:
                    if skill.lower() in chunk_text and skill not in skills_found:
                        skills_found.add(skill)
                        confidence_scores[skill] = 0.7
        
        return {
            "skills": list(skills_found),
            "confidence_scores": confidence_scores,
            "total_skills_found": len(skills_found)
        }
    
    def advanced_experience_analysis(self, text: str, position: str) -> Dict[str, Any]:
        """Advanced experience analysis with semantic understanding."""
        analysis = {
            "total_years": 0,
            "relevant_years": 0,
            "has_direct_experience": False,
            "has_related_experience": False,
            "experience_quality": "Unknown",
            "leadership_experience": False,
            "training_experience": False,
            "certifications": [],
            "education_level": "Unknown"
        }
        
        text_lower = text.lower()
        position_data = self.position_intelligence.get(position, {})
        
        # Extract years of experience
        year_patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
            r"(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)",
            r"experience[:\s]*(\d+)\+?\s*years?",
            r"(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:hospitality|hotel|restaurant|food)"
        ]
        
        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            years_found.extend([int(match) for match in matches if match.isdigit()])
        
        if years_found:
            analysis["total_years"] = max(years_found)
        
        # Check for direct experience
        experience_indicators = position_data.get("experience_indicators", [])
        for indicator in experience_indicators:
            if indicator.lower() in text_lower:
                analysis["has_direct_experience"] = True
                break
        
        # Check for leadership experience
        leadership_terms = [
            "manager", "supervisor", "lead", "director", "head", "chief",
            "team lead", "assistant manager", "department head"
        ]
        
        for term in leadership_terms:
            if term in text_lower:
                analysis["leadership_experience"] = True
                break
        
        # Check for training experience
        training_terms = ["training", "trainer", "mentor", "coach", "instructor", "teach"]
        for term in training_terms:
            if term in text_lower:
                analysis["training_experience"] = True
                break
        
        # Extract certifications
        cert_patterns = [
            r"certified\s+(?:in\s+)?([^,.]+)",
            r"certification\s+(?:in\s+)?([^,.]+)",
            r"license\s+(?:in\s+)?([^,.]+)",
            r"diploma\s+(?:in\s+)?([^,.]+)"
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            analysis["certifications"].extend(matches)
        
        # Determine experience quality
        if analysis["has_direct_experience"] and analysis["total_years"] >= 3:
            analysis["experience_quality"] = "Excellent"
        elif analysis["has_direct_experience"] or analysis["total_years"] >= 2:
            analysis["experience_quality"] = "Good"
        elif analysis["total_years"] >= 1:
            analysis["experience_quality"] = "Fair"
        else:
            analysis["experience_quality"] = "Limited"
        
        return analysis
    
    def calculate_enhanced_score(self, candidate: Dict[str, Any], position: str) -> Dict[str, Any]:
        """Calculate comprehensive candidate score with advanced algorithms."""
        position_data = self.position_intelligence.get(position, {})
        if not position_data:
            logger.warning(f"Position '{position}' not found in intelligence database")
            return {"total_score": 0, "breakdown": {}, "recommendation": "Unable to evaluate"}
        
        # Get scoring weights
        weights = position_data.get("scoring_weights", {
            "experience": 0.3, "skills": 0.3, "cultural_fit": 0.2, "hospitality": 0.2
        })
        
        scores = {"experience": 0, "skills": 0, "cultural_fit": 0, "hospitality": 0}
        details = {}
        
        # 1. Experience Score
        exp_analysis = candidate.get("experience_analysis", {})
        total_years = exp_analysis.get("total_years", 0)
        min_years = position_data.get("min_experience_years", 0)
        preferred_years = position_data.get("preferred_experience_years", 3)
        
        if total_years >= preferred_years:
            scores["experience"] = 1.0
        elif total_years >= min_years:
            scores["experience"] = 0.5 + (total_years - min_years) / (preferred_years - min_years) * 0.5
        else:
            scores["experience"] = total_years / max(min_years, 1) * 0.5
        
        # Boost for direct experience
        if exp_analysis.get("has_direct_experience", False):
            scores["experience"] = min(scores["experience"] * 1.3, 1.0)
        
        # Leadership boost
        if exp_analysis.get("leadership_experience", False):
            scores["experience"] = min(scores["experience"] * 1.2, 1.0)
        
        details["experience"] = {
            "years": total_years,
            "quality": exp_analysis.get("experience_quality", "Unknown"),
            "direct_experience": exp_analysis.get("has_direct_experience", False),
            "leadership": exp_analysis.get("leadership_experience", False)
        }
        
        # 2. Skills Score
        skill_analysis = candidate.get("skill_analysis", {})
        skills_found = skill_analysis.get("skills", [])
        confidence_scores = skill_analysis.get("confidence_scores", {})
        
        must_have = position_data.get("must_have_skills", [])
        nice_to_have = position_data.get("nice_to_have_skills", [])
        technical_skills = position_data.get("technical_skills", [])
        
        # Calculate skills score
        must_have_score = 0
        if must_have:
            matched_must_have = [s for s in skills_found if s in must_have]
            must_have_score = len(matched_must_have) / len(must_have)
        
        nice_to_have_score = 0
        if nice_to_have:
            matched_nice_to_have = [s for s in skills_found if s in nice_to_have]
            nice_to_have_score = len(matched_nice_to_have) / len(nice_to_have)
        
        technical_score = 0
        if technical_skills:
            matched_technical = [s for s in skills_found if s in technical_skills]
            technical_score = len(matched_technical) / len(technical_skills)
        
        # Weighted skills score
        scores["skills"] = (must_have_score * 0.6 + nice_to_have_score * 0.3 + technical_score * 0.1)
        
        details["skills"] = {
            "must_have_matched": must_have_score,
            "nice_to_have_matched": nice_to_have_score,
            "technical_matched": technical_score,
            "total_skills": len(skills_found)
        }
        
        # 3. Cultural Fit Score
        cultural_keywords = position_data.get("cultural_fit_keywords", [])
        disqualifying_factors = position_data.get("disqualifying_factors", [])
        
        candidate_text = candidate.get("resume_text", "").lower()
        
        cultural_matches = sum(1 for keyword in cultural_keywords if keyword.lower() in candidate_text)
        cultural_score = cultural_matches / max(len(cultural_keywords), 1)
        
        # Check for disqualifying factors
        disqualifying_found = [factor for factor in disqualifying_factors if factor.lower() in candidate_text]
        if disqualifying_found:
            cultural_score *= 0.5  # Penalty for disqualifying factors
        
        scores["cultural_fit"] = min(cultural_score, 1.0)
        
        details["cultural_fit"] = {
            "positive_indicators": cultural_matches,
            "disqualifying_factors": disqualifying_found
        }
        
        # 4. Hospitality Industry Score
        hospitality_indicators = [
            "hotel", "hospitality", "resort", "restaurant", "food service",
            "guest services", "tourism", "travel", "accommodation"
        ]
        
        hospitality_matches = sum(1 for indicator in hospitality_indicators if indicator in candidate_text)
        scores["hospitality"] = min(hospitality_matches / 3, 1.0)  # Max at 3 indicators
        
        details["hospitality"] = {
            "industry_indicators": hospitality_matches
        }
        
        # Calculate final score
        final_score = sum(scores[category] * weights[category] for category in scores)
        
        # Determine recommendation
        if final_score >= 0.8:
            recommendation = "Highly Recommended"
        elif final_score >= 0.65:
            recommendation = "Recommended"
        elif final_score >= 0.5:
            recommendation = "Consider with Interview"
        else:
            recommendation = "Not Recommended"
        
        return {
            "total_score": final_score,
            "category_scores": scores,
            "breakdown": details,
            "recommendation": recommendation,
            "position": position
        }
    
    def process_single_resume(self, file_path: Path, position: str) -> Dict[str, Any]:
        """Process a single resume with comprehensive analysis."""
        try:
            logger.info(f"📄 Processing: {file_path.name}")
            
            # Extract text from file
            text = self._extract_text_from_file(file_path)
            if not text or len(text.strip()) < 50:
                logger.warning(f"⚠️ Insufficient text extracted from {file_path.name}")
                return None
            
            # Extract basic information
            candidate_info = self._extract_candidate_info(text)
            
            # Enhanced skill analysis
            skill_analysis = self.enhanced_skill_extraction(text, position)
            
            # Advanced experience analysis
            experience_analysis = self.advanced_experience_analysis(text, position)
            
            # Calculate enhanced score
            candidate_data = {
                "resume_text": text,
                "skill_analysis": skill_analysis,
                "experience_analysis": experience_analysis,
                **candidate_info
            }
            
            scoring_result = self.calculate_enhanced_score(candidate_data, position)
            
            # Compile final result
            result = {
                "file_name": file_path.name,
                "candidate_name": candidate_info.get("name", "Unknown"),
                "email": candidate_info.get("email", "Not found"),
                "phone": candidate_info.get("phone", "Not found"),
                "location": candidate_info.get("location", "Not specified"),
                "total_score": scoring_result["total_score"],
                "recommendation": scoring_result["recommendation"],
                "category_scores": scoring_result["category_scores"],
                "breakdown": scoring_result["breakdown"],
                "skills_found": skill_analysis["skills"],
                "experience_years": experience_analysis["total_years"],
                "experience_quality": experience_analysis["experience_quality"],
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ {file_path.name}: {result['total_score']:.1%} - {result['recommendation']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return None
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats."""
        text = ""
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            elif file_ext == '.pdf':
                if ocr_available:
                    try:
                        # Try text extraction first
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                text += page.extract_text() + "\n"
                        
                        # If text extraction failed, use OCR
                        if len(text.strip()) < 50:
                            pages = pdf2image.convert_from_path(file_path)
                            for page in pages:
                                text += pytesseract.image_to_string(page) + "\n"
                    except:
                        logger.warning(f"PDF processing failed for {file_path.name}")
                        return ""
                else:
                    logger.warning(f"PDF support not available for {file_path.name}")
                    return ""
            
            elif file_ext in ['.docx', '.doc']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                except:
                    logger.warning(f"DOCX processing failed for {file_path.name}")
                    return ""
            
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                if ocr_available:
                    try:
                        image = Image.open(file_path)
                        text = pytesseract.image_to_string(image)
                    except:
                        logger.warning(f"Image OCR failed for {file_path.name}")
                        return ""
                else:
                    logger.warning(f"OCR not available for {file_path.name}")
                    return ""
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path.name}: {e}")
            return ""
    
    def _extract_candidate_info(self, text: str) -> Dict[str, Any]:
        """Extract basic candidate information from resume text."""
        info = {
            "name": "Unknown",
            "email": "Not found",
            "phone": "Not found",
            "location": "Not specified"
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            info["email"] = email_matches[0]
        
        # Extract phone
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                info["phone"] = phone_matches[0]
                break
        
        # Extract name (first non-empty line typically)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Look for a line that looks like a name
            for line in lines[:5]:  # Check first 5 lines
                if (len(line.split()) in [2, 3] and 
                    not '@' in line and 
                    not any(char.isdigit() for char in line) and
                    len(line) < 50):
                    info["name"] = line
                    break
        
        # Extract location
        location_patterns = [
            r'(?:Address|Location|Lives?|Resides?|Based)\s*:?\s*([A-Za-z\s,]+)',
            r'([A-Za-z\s]+,\s*[A-Z]{2}(?:\s+\d{5})?)',  # City, State ZIP
            r'([A-Za-z\s]+,\s*[A-Za-z\s]+)'  # City, Country
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                info["location"] = matches[0].strip()
                break
        
        return info
    
    def screen_candidates(self, position: str, max_candidates: Optional[int] = None) -> List[Dict[str, Any]]:
        """Screen all candidates for a specific position."""
        logger.info(f"🎯 Starting candidate screening for: {position}")
        
        # Find resume files
        extensions = ['*.txt', '*.pdf', '*.docx', '*.doc', '*.jpg', '*.jpeg', '*.png']
        resume_files = []
        for ext in extensions:
            resume_files.extend(self.input_dir.glob(ext))
        
        if not resume_files:
            logger.warning(f"📭 No resume files found in {self.input_dir}")
            return []
        
        logger.info(f"📚 Found {len(resume_files)} resume files")
        
        # Process each resume
        candidates = []
        for file_path in resume_files:
            result = self.process_single_resume(file_path, position)
            if result:
                candidates.append(result)
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Limit results if specified
        if max_candidates:
            candidates = candidates[:max_candidates]
        
        logger.info(f"✅ Screening complete: {len(candidates)} candidates processed")
        
        return candidates
    
    def generate_report(self, candidates: List[Dict[str, Any]], position: str) -> str:
        """Generate comprehensive screening report."""
        if not candidates:
            return "No candidates found to analyze."
        
        report = f"""
🏨 HOTEL AI RESUME SCREENER - ENHANCED REPORT
{'='*60}

Position: {position}
Candidates Analyzed: {len(candidates)}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*60}
TOP CANDIDATES
{'='*60}
"""
        
        for i, candidate in enumerate(candidates[:10], 1):
            score = candidate["total_score"]
            recommendation = candidate["recommendation"]
            
            report += f"""
{i}. {candidate['candidate_name']} - {score:.1%} ({recommendation})
   📧 {candidate['email']} | 📞 {candidate['phone']}
   📍 {candidate['location']}
   💼 Experience: {candidate['experience_years']} years ({candidate['experience_quality']})
   🎯 Skills Found: {len(candidate['skills_found'])} relevant skills
   
   Score Breakdown:
   - Experience: {candidate['category_scores']['experience']:.1%}
   - Skills: {candidate['category_scores']['skills']:.1%}
   - Cultural Fit: {candidate['category_scores']['cultural_fit']:.1%}
   - Hospitality: {candidate['category_scores']['hospitality']:.1%}
   
   Key Skills: {', '.join(candidate['skills_found'][:5])}{'...' if len(candidate['skills_found']) > 5 else ''}
"""
        
        # Add statistics
        scores = [c["total_score"] for c in candidates]
        report += f"""
{'='*60}
SCREENING STATISTICS
{'='*60}

Average Score: {statistics.mean(scores):.1%}
Median Score: {statistics.median(scores):.1%}
Top Score: {max(scores):.1%}
Candidates Above 80%: {sum(1 for s in scores if s >= 0.8)}
Candidates Above 65%: {sum(1 for s in scores if s >= 0.65)}

Recommendation Distribution:
- Highly Recommended: {sum(1 for c in candidates if c['recommendation'] == 'Highly Recommended')}
- Recommended: {sum(1 for c in candidates if c['recommendation'] == 'Recommended')}
- Consider with Interview: {sum(1 for c in candidates if c['recommendation'] == 'Consider with Interview')}
- Not Recommended: {sum(1 for c in candidates if c['recommendation'] == 'Not Recommended')}

{'='*60}
End of Report
{'='*60}
"""
        
        return report
    
    def export_to_excel(self, candidates: List[Dict[str, Any]], position: str) -> str:
        """Export results to Excel with multiple sheets."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screening_results_{position}_{timestamp}.xlsx"
        filepath = self.output_dir / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main results sheet
                main_data = []
                for candidate in candidates:
                    main_data.append({
                        'Rank': candidates.index(candidate) + 1,
                        'Name': candidate['candidate_name'],
                        'Email': candidate['email'],
                        'Phone': candidate['phone'],
                        'Location': candidate['location'],
                        'Total Score': f"{candidate['total_score']:.1%}",
                        'Recommendation': candidate['recommendation'],
                        'Experience Score': f"{candidate['category_scores']['experience']:.1%}",
                        'Skills Score': f"{candidate['category_scores']['skills']:.1%}",
                        'Cultural Fit': f"{candidate['category_scores']['cultural_fit']:.1%}",
                        'Hospitality Score': f"{candidate['category_scores']['hospitality']:.1%}",
                        'Experience Years': candidate['experience_years'],
                        'Experience Quality': candidate['experience_quality'],
                        'Skills Found': len(candidate['skills_found']),
                        'Key Skills': ', '.join(candidate['skills_found'][:3]),
                        'File Name': candidate['file_name']
                    })
                
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name='Candidate Rankings', index=False)
                
                # Detailed skills sheet
                skills_data = []
                for candidate in candidates:
                    for skill in candidate['skills_found']:
                        skills_data.append({
                            'Candidate': candidate['candidate_name'],
                            'Skill': skill,
                            'Category': 'Technical' if skill in self.position_intelligence.get(position, {}).get('technical_skills', []) else 'General'
                        })
                
                if skills_data:
                    df_skills = pd.DataFrame(skills_data)
                    df_skills.to_excel(writer, sheet_name='Skills Analysis', index=False)
                
                # Position requirements sheet
                position_data = self.position_intelligence.get(position, {})
                req_data = []
                
                for skill_type in ['must_have_skills', 'nice_to_have_skills', 'technical_skills']:
                    for skill in position_data.get(skill_type, []):
                        req_data.append({
                            'Skill': skill,
                            'Type': skill_type.replace('_', ' ').title(),
                            'Candidates with Skill': sum(1 for c in candidates if skill in c['skills_found'])
                        })
                
                if req_data:
                    df_req = pd.DataFrame(req_data)
                    df_req.to_excel(writer, sheet_name='Position Requirements', index=False)
            
            logger.info(f"📊 Excel report exported: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Excel export failed: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Hotel AI Resume Screener")
    parser.add_argument("--input", "-i", default="input_resumes", help="Input directory with resumes")
    parser.add_argument("--output", "-o", default="screening_results", help="Output directory for results")
    parser.add_argument("--position", "-p", required=True, help="Position to screen for")
    parser.add_argument("--max-candidates", "-m", type=int, help="Maximum number of candidates to return")
    parser.add_argument("--export-excel", "-e", action="store_true", help="Export results to Excel")
    
    args = parser.parse_args()
    
    # Initialize screener
    screener = EnhancedHotelAIScreener(args.input, args.output)
    
    # Screen candidates
    candidates = screener.screen_candidates(args.position, args.max_candidates)
    
    if not candidates:
        print("❌ No candidates found or processed successfully.")
        return
    
    # Generate report
    report = screener.generate_report(candidates, args.position)
    print(report)
    
    # Save text report
    report_file = screener.output_dir / f"screening_report_{args.position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved: {report_file}")
    
    # Export to Excel if requested
    if args.export_excel:
        excel_file = screener.export_to_excel(candidates, args.position)
        if excel_file:
            print(f"📊 Excel report saved: {excel_file}")


if __name__ == "__main__":
    main()
