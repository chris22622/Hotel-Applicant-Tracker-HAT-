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
        """Load comprehensive hotel position intelligence with advanced requirements for ALL hotel positions."""
        return {
            # ==========================================
            # EXECUTIVE MANAGEMENT
            # ==========================================
            "General Manager": {
                "must_have_skills": [
                    "executive leadership", "strategic planning", "operations management", "financial management",
                    "team leadership", "hospitality management", "budget planning", "revenue management",
                    "staff development", "guest relations", "crisis management", "business development"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "resort management", "franchise operations", "asset management",
                    "brand standards", "market analysis", "competitor analysis", "ROI optimization",
                    "stakeholder management", "board reporting", "contract negotiation"
                ],
                "technical_skills": [
                    "hotel management systems", "financial reporting", "performance analytics",
                    "revenue management systems", "budgeting software", "market intelligence tools"
                ],
                "soft_skills": [
                    "visionary leadership", "emotional intelligence", "decision making", "communication",
                    "adaptability", "crisis management", "strategic thinking", "negotiation"
                ],
                "cultural_fit_keywords": [
                    "visionary", "leader", "strategic", "results-driven", "innovative",
                    "guest-focused", "team builder", "ethical", "accountable", "inspiring"
                ],
                "disqualifying_factors": [
                    "poor leadership record", "financial mismanagement", "lack of hospitality experience",
                    "poor communication", "inflexible", "unethical behavior"
                ],
                "experience_indicators": [
                    "general manager", "hotel manager", "resort manager", "operations manager",
                    "executive leadership", "hospitality management", "property management"
                ],
                "education_preferences": [
                    "hospitality management", "business administration", "hotel administration",
                    "MBA", "management", "finance", "operations management"
                ],
                "certifications": [
                    "CHA (Certified Hotel Administrator)", "hospitality management", "leadership",
                    "financial management", "revenue management"
                ],
                "scoring_weights": {
                    "experience": 0.45, "skills": 0.25, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 10,
                "preferred_experience_years": 15,
                "salary_range": {"min": 120000, "max": 250000},
                "growth_potential": "Executive",
                "training_requirements": "Strategic"
            },

            "Hotel Manager": {
                "must_have_skills": [
                    "hotel operations", "team management", "guest relations", "budget management",
                    "staff supervision", "quality control", "hospitality operations", "leadership",
                    "problem solving", "communication", "time management", "performance management"
                ],
                "nice_to_have_skills": [
                    "luxury service", "resort operations", "event management", "sales coordination",
                    "revenue optimization", "brand standards", "guest recovery", "staff training"
                ],
                "technical_skills": [
                    "hotel management systems", "PMS systems", "reporting tools", "scheduling software",
                    "performance dashboards", "guest feedback systems"
                ],
                "soft_skills": [
                    "leadership", "emotional intelligence", "decision making", "conflict resolution",
                    "adaptability", "stress management", "mentoring", "communication"
                ],
                "cultural_fit_keywords": [
                    "leader", "hospitality-focused", "team player", "results-oriented",
                    "guest advocate", "professional", "reliable", "innovative"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of hospitality experience", "poor communication",
                    "inability to handle stress", "inflexible"
                ],
                "experience_indicators": [
                    "hotel manager", "assistant manager", "operations manager", "hospitality management",
                    "hotel operations", "property management", "guest services management"
                ],
                "education_preferences": [
                    "hospitality management", "hotel administration", "business management",
                    "tourism management", "operations management"
                ],
                "certifications": [
                    "hospitality management", "hotel operations", "leadership certification",
                    "customer service excellence", "revenue management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.25, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 65000, "max": 120000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Operations Manager": {
                "must_have_skills": [
                    "operations management", "process optimization", "team coordination", "quality control",
                    "budget oversight", "performance monitoring", "staff management", "logistics coordination",
                    "vendor management", "compliance", "efficiency improvement", "project management"
                ],
                "nice_to_have_skills": [
                    "hotel operations", "multi-department coordination", "systems integration",
                    "cost reduction", "workflow optimization", "technology implementation", "training coordination"
                ],
                "technical_skills": [
                    "operations software", "performance analytics", "project management tools",
                    "reporting systems", "workflow management", "compliance tracking"
                ],
                "soft_skills": [
                    "analytical thinking", "leadership", "problem solving", "communication",
                    "organization", "attention to detail", "adaptability", "decision making"
                ],
                "cultural_fit_keywords": [
                    "efficient", "organized", "analytical", "improvement-focused",
                    "team player", "results-driven", "detail-oriented", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of analytical skills", "poor communication",
                    "inability to manage complexity", "inflexible"
                ],
                "experience_indicators": [
                    "operations manager", "process manager", "operations coordination",
                    "operations analysis", "workflow management", "efficiency improvement"
                ],
                "education_preferences": [
                    "operations management", "business administration", "industrial engineering",
                    "process management", "project management", "hospitality management"
                ],
                "certifications": [
                    "operations management", "project management", "process improvement",
                    "quality management", "efficiency certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 55000, "max": 85000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Assistant Manager": {
                "must_have_skills": [
                    "assistant management", "team support", "guest relations", "operations support",
                    "staff coordination", "problem solving", "communication", "time management",
                    "hospitality operations", "customer service", "quality assurance", "administrative skills"
                ],
                "nice_to_have_skills": [
                    "management training", "leadership development", "guest recovery",
                    "staff training", "performance monitoring", "event coordination", "multi-tasking"
                ],
                "technical_skills": [
                    "hotel systems", "administrative software", "reporting tools",
                    "scheduling systems", "communication platforms", "guest management systems"
                ],
                "soft_skills": [
                    "support skills", "communication", "adaptability", "teamwork",
                    "problem solving", "attention to detail", "reliability", "learning agility"
                ],
                "cultural_fit_keywords": [
                    "supportive", "reliable", "team player", "guest-focused",
                    "professional", "adaptable", "eager to learn", "helpful"
                ],
                "disqualifying_factors": [
                    "poor communication", "lack of reliability", "poor customer service",
                    "inability to work in team", "inflexible"
                ],
                "experience_indicators": [
                    "assistant manager", "supervisor", "team lead", "hospitality support",
                    "guest services", "operations support", "management trainee"
                ],
                "education_preferences": [
                    "hospitality management", "business administration", "hotel management",
                    "customer service", "management studies"
                ],
                "certifications": [
                    "hospitality management", "customer service", "leadership development",
                    "hotel operations", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.25, "cultural_fit": 0.25, "hospitality": 0.20
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Executive Assistant": {
                "must_have_skills": [
                    "executive support", "administrative skills", "communication", "organization",
                    "calendar management", "correspondence", "meeting coordination", "document preparation",
                    "travel arrangements", "confidentiality", "time management", "multitasking"
                ],
                "nice_to_have_skills": [
                    "project coordination", "event planning", "stakeholder communication",
                    "presentation preparation", "database management", "expense management", "protocol knowledge"
                ],
                "technical_skills": [
                    "Microsoft Office Suite", "calendar software", "communication platforms",
                    "travel booking systems", "expense management software", "document management"
                ],
                "soft_skills": [
                    "discretion", "professionalism", "attention to detail", "communication",
                    "adaptability", "problem solving", "reliability", "interpersonal skills"
                ],
                "cultural_fit_keywords": [
                    "professional", "discreet", "organized", "reliable",
                    "detail-oriented", "proactive", "supportive", "confidential"
                ],
                "disqualifying_factors": [
                    "poor organization", "lack of discretion", "poor communication",
                    "unreliable", "inability to handle confidential information"
                ],
                "experience_indicators": [
                    "executive assistant", "administrative assistant", "personal assistant",
                    "executive support", "administrative support", "office management"
                ],
                "education_preferences": [
                    "business administration", "office administration", "communications",
                    "secretarial studies", "administrative management"
                ],
                "certifications": [
                    "administrative professional", "executive assistant certification",
                    "office management", "business communication"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Duty Manager": {
                "must_have_skills": [
                    "emergency management", "guest relations", "staff coordination", "problem solving",
                    "communication", "decision making", "hospitality operations", "crisis management",
                    "customer service", "time management", "multi-department coordination", "leadership"
                ],
                "nice_to_have_skills": [
                    "night operations", "security coordination", "guest recovery",
                    "incident reporting", "staff support", "emergency procedures", "conflict resolution"
                ],
                "technical_skills": [
                    "hotel management systems", "emergency systems", "communication systems",
                    "reporting tools", "security systems", "guest management systems"
                ],
                "soft_skills": [
                    "calm under pressure", "leadership", "decision making", "communication",
                    "adaptability", "problem solving", "reliability", "confidence"
                ],
                "cultural_fit_keywords": [
                    "calm under pressure", "reliable", "leader", "problem-solver",
                    "guest-focused", "professional", "decisive", "responsible"
                ],
                "disqualifying_factors": [
                    "poor under pressure", "poor decision making", "lack of leadership",
                    "poor communication", "unreliable"
                ],
                "experience_indicators": [
                    "duty manager", "night manager", "operations manager", "guest services",
                    "hospitality management", "emergency management", "shift supervisor"
                ],
                "education_preferences": [
                    "hospitality management", "hotel management", "business management",
                    "emergency management", "customer service"
                ],
                "certifications": [
                    "hospitality management", "emergency management", "guest relations",
                    "leadership certification", "crisis management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # FRONT OFFICE & GUEST SERVICES
            # ==========================================
            "Front Office Manager": {
                "must_have_skills": [
                    "front office operations", "team management", "guest relations", "PMS systems",
                    "reservations management", "staff supervision", "revenue optimization", "training",
                    "quality control", "performance management", "customer service", "communication"
                ],
                "nice_to_have_skills": [
                    "luxury service", "group bookings", "VIP services", "guest recovery",
                    "upselling strategies", "night audit", "forecasting", "inventory management"
                ],
                "technical_skills": [
                    "Opera PMS", "Maestro PMS", "reservation systems", "revenue management systems",
                    "reporting tools", "guest feedback systems", "channel management"
                ],
                "soft_skills": [
                    "leadership", "communication", "problem solving", "attention to detail",
                    "multitasking", "stress management", "team building", "customer focus"
                ],
                "cultural_fit_keywords": [
                    "guest-focused", "leader", "professional", "detail-oriented",
                    "team builder", "service excellence", "efficient", "welcoming"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of hospitality experience", "poor customer service",
                    "inability to handle stress", "poor communication"
                ],
                "experience_indicators": [
                    "front office manager", "guest services manager", "front desk supervisor",
                    "hotel front office", "reservations manager", "guest relations"
                ],
                "education_preferences": [
                    "hospitality management", "hotel administration", "tourism management",
                    "business management", "customer service"
                ],
                "certifications": [
                    "hospitality management", "PMS certification", "guest relations",
                    "revenue management", "customer service excellence"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Front Desk Supervisor": {
                "must_have_skills": [
                    "front desk operations", "team supervision", "guest services", "PMS systems",
                    "customer service", "problem solving", "communication", "staff training",
                    "quality control", "cash handling", "reservations", "multitasking"
                ],
                "nice_to_have_skills": [
                    "guest recovery", "upselling", "night audit", "VIP services",
                    "group check-ins", "conflict resolution", "performance coaching", "scheduling"
                ],
                "technical_skills": [
                    "Opera PMS", "reservation systems", "payment processing", "guest management systems",
                    "reporting tools", "telephone systems", "keycard systems"
                ],
                "soft_skills": [
                    "leadership", "communication", "patience", "problem solving",
                    "attention to detail", "multitasking", "team building", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "guest-focused", "team leader", "professional", "helpful",
                    "patient", "reliable", "welcoming", "supportive"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "poor customer service", "lack of patience",
                    "poor communication", "inability to multitask"
                ],
                "experience_indicators": [
                    "front desk supervisor", "guest services supervisor", "front desk agent",
                    "hotel reception", "guest services", "hospitality supervision"
                ],
                "education_preferences": [
                    "hospitality management", "hotel management", "customer service",
                    "business administration", "tourism"
                ],
                "certifications": [
                    "hospitality management", "PMS certification", "customer service",
                    "leadership development", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 55000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

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

            "Receptionist": {
                "must_have_skills": [
                    "customer service", "communication", "phone skills", "computer skills",
                    "multitasking", "organization", "attention to detail", "professional demeanor",
                    "administrative skills", "time management", "problem solving"
                ],
                "nice_to_have_skills": [
                    "hospitality experience", "multilingual", "guest relations", "appointment scheduling",
                    "visitor management", "office administration", "data entry", "filing"
                ],
                "technical_skills": [
                    "phone systems", "computer software", "visitor management systems",
                    "appointment scheduling", "email management", "basic office equipment"
                ],
                "soft_skills": [
                    "communication", "friendliness", "professionalism", "patience",
                    "adaptability", "interpersonal skills", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "professional", "welcoming", "helpful",
                    "organized", "reliable", "positive", "courteous"
                ],
                "disqualifying_factors": [
                    "poor communication", "unprofessional demeanor", "unreliable",
                    "poor phone skills", "lack of computer skills"
                ],
                "experience_indicators": [
                    "receptionist", "front desk", "customer service", "administrative assistant",
                    "office support", "guest services", "visitor services"
                ],
                "education_preferences": [
                    "high school diploma", "customer service", "office administration",
                    "communications", "business studies"
                ],
                "certifications": [
                    "customer service", "office administration", "communication skills",
                    "hospitality basics", "professional development"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Night Auditor": {
                "must_have_skills": [
                    "night audit procedures", "accounting basics", "PMS systems", "customer service",
                    "attention to detail", "security awareness", "problem solving", "independence",
                    "cash handling", "report generation", "computer skills", "communication"
                ],
                "nice_to_have_skills": [
                    "accounting experience", "night operations", "emergency procedures",
                    "guest services", "security protocols", "inventory management", "data entry"
                ],
                "technical_skills": [
                    "Opera PMS", "audit software", "accounting systems", "reporting tools",
                    "payment processing", "security systems", "telephone systems"
                ],
                "soft_skills": [
                    "independence", "attention to detail", "reliability", "problem solving",
                    "calm under pressure", "self-motivation", "accuracy", "responsibility"
                ],
                "cultural_fit_keywords": [
                    "reliable", "independent", "detail-oriented", "responsible",
                    "calm", "professional", "accurate", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "unreliable", "lack of independence",
                    "poor computer skills", "inability to work alone"
                ],
                "experience_indicators": [
                    "night auditor", "night audit", "accounting", "night operations",
                    "front desk", "audit procedures", "night shift"
                ],
                "education_preferences": [
                    "accounting", "hospitality management", "business administration",
                    "hotel management", "bookkeeping"
                ],
                "certifications": [
                    "accounting basics", "hospitality operations", "night audit certification",
                    "bookkeeping", "PMS certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 32000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
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

            "Concierge": {
                "must_have_skills": [
                    "customer service", "local knowledge", "communication", "organization", "problem solving",
                    "multilingual", "cultural awareness", "networking", "information management", "hospitality"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "event planning", "restaurant knowledge", "tour coordination",
                    "VIP services", "transportation arrangements", "entertainment booking", "personal shopping"
                ],
                "technical_skills": [
                    "reservation systems", "concierge software", "communication platforms",
                    "mapping tools", "booking platforms", "guest management systems"
                ],
                "soft_skills": [
                    "resourcefulness", "communication", "cultural sensitivity", "patience",
                    "attention to detail", "networking", "adaptability", "sophistication"
                ],
                "cultural_fit_keywords": [
                    "sophisticated", "knowledgeable", "helpful", "professional", "well-connected",
                    "resourceful", "cultured", "service-oriented", "diplomatic", "refined"
                ],
                "disqualifying_factors": [
                    "poor local knowledge", "lack of sophistication", "poor communication",
                    "inflexible", "lack of cultural awareness"
                ],
                "experience_indicators": [
                    "concierge", "guest services", "luxury hospitality", "customer service",
                    "tour guide", "travel services", "hospitality", "personal assistant"
                ],
                "education_preferences": [
                    "hospitality management", "tourism", "languages", "cultural studies",
                    "communications", "travel and tourism"
                ],
                "certifications": [
                    "concierge certification", "hospitality excellence", "cultural awareness",
                    "language certifications", "luxury service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Bellman": {
                "must_have_skills": [
                    "customer service", "physical fitness", "communication", "hospitality",
                    "luggage handling", "guest assistance", "local knowledge", "professional appearance",
                    "reliability", "time management", "courtesy", "safety awareness"
                ],
                "nice_to_have_skills": [
                    "multilingual", "concierge services", "transportation knowledge",
                    "guest relations", "tip etiquette", "luxury service", "door services"
                ],
                "technical_skills": [
                    "luggage equipment", "transportation systems", "communication devices",
                    "guest tracking systems", "safety equipment"
                ],
                "soft_skills": [
                    "friendliness", "professionalism", "helpfulness", "patience",
                    "physical stamina", "reliability", "courtesy", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "helpful", "courteous", "professional", "reliable",
                    "friendly", "service-oriented", "welcoming", "attentive"
                ],
                "disqualifying_factors": [
                    "poor physical condition", "unprofessional appearance", "poor customer service",
                    "unreliable", "lack of courtesy"
                ],
                "experience_indicators": [
                    "bellman", "bell captain", "porter", "guest services", "hospitality",
                    "customer service", "luggage services", "door services"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality basics", "customer service",
                    "communications", "physical education"
                ],
                "certifications": [
                    "hospitality service", "customer service", "safety training",
                    "guest relations", "physical fitness"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Porter": {
                "must_have_skills": [
                    "physical fitness", "customer service", "luggage handling", "communication",
                    "reliability", "safety awareness", "guest assistance", "professional appearance",
                    "time management", "courtesy", "hospitality", "teamwork"
                ],
                "nice_to_have_skills": [
                    "equipment operation", "maintenance awareness", "guest relations",
                    "multilingual", "transportation knowledge", "inventory management"
                ],
                "technical_skills": [
                    "luggage carts", "transportation equipment", "lifting equipment",
                    "safety equipment", "communication devices"
                ],
                "soft_skills": [
                    "physical stamina", "reliability", "helpfulness", "courtesy",
                    "teamwork", "positive attitude", "patience", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "hardworking", "reliable", "helpful", "courteous",
                    "team player", "professional", "strong", "dependable"
                ],
                "disqualifying_factors": [
                    "poor physical condition", "unreliable", "poor safety awareness",
                    "unprofessional", "lack of teamwork"
                ],
                "experience_indicators": [
                    "porter", "luggage porter", "baggage handler", "guest services",
                    "hospitality support", "customer service", "physical labor"
                ],
                "education_preferences": [
                    "high school diploma", "physical education", "hospitality basics",
                    "customer service", "safety training"
                ],
                "certifications": [
                    "safety training", "hospitality service", "customer service",
                    "physical fitness", "equipment operation"
                ],
                "scoring_weights": {
                    "experience": 0.15, "skills": 0.35, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 22000, "max": 30000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Bell Captain": {
                "must_have_skills": [
                    "team leadership", "customer service", "guest relations", "communication",
                    "staff supervision", "hospitality operations", "training", "quality control",
                    "luggage operations", "guest assistance", "problem solving", "scheduling"
                ],
                "nice_to_have_skills": [
                    "luxury service", "VIP handling", "concierge coordination",
                    "staff development", "performance management", "guest recovery", "multilingual"
                ],
                "technical_skills": [
                    "staff scheduling", "guest tracking systems", "communication systems",
                    "performance monitoring", "training platforms", "luggage systems"
                ],
                "soft_skills": [
                    "leadership", "communication", "teamwork", "problem solving",
                    "customer focus", "attention to detail", "reliability", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "leader", "professional", "guest-focused", "team builder",
                    "service excellence", "reliable", "courteous", "experienced"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of hospitality experience", "poor customer service",
                    "inability to supervise", "unprofessional"
                ],
                "experience_indicators": [
                    "bell captain", "bellman supervisor", "guest services supervisor",
                    "hospitality supervision", "team leadership", "bell services"
                ],
                "education_preferences": [
                    "hospitality management", "hotel management", "business administration",
                    "customer service", "leadership development"
                ],
                "certifications": [
                    "hospitality leadership", "guest services management", "team leadership",
                    "customer service excellence", "hospitality operations"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.25, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Rooms Division Manager": {
                "must_have_skills": [
                    "rooms division management", "team leadership", "operations management", "revenue optimization",
                    "quality control", "staff supervision", "guest relations", "performance management",
                    "budget management", "inventory control", "training coordination", "strategic planning"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "multi-property experience", "technology implementation",
                    "process improvement", "vendor management", "compliance management", "analytics"
                ],
                "technical_skills": [
                    "hotel management systems", "revenue management systems", "performance analytics",
                    "budgeting software", "scheduling systems", "quality management systems"
                ],
                "soft_skills": [
                    "strategic leadership", "analytical thinking", "communication", "decision making",
                    "change management", "team building", "problem solving", "innovation"
                ],
                "cultural_fit_keywords": [
                    "strategic", "leader", "analytical", "guest-focused",
                    "results-driven", "innovative", "collaborative", "excellence-oriented"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of strategic thinking", "poor analytical skills",
                    "inability to manage complexity", "poor communication"
                ],
                "experience_indicators": [
                    "rooms division manager", "hotel operations manager", "front office manager",
                    "housekeeping manager", "rooms operations", "hospitality management"
                ],
                "education_preferences": [
                    "hospitality management", "hotel administration", "business management",
                    "operations management", "MBA"
                ],
                "certifications": [
                    "hospitality management", "revenue management", "operations management",
                    "leadership certification", "hotel administration"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.25, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 70000, "max": 110000},
                "growth_potential": "Very High",
                "training_requirements": "Extensive"
            },

            "Reservations Manager": {
                "must_have_skills": [
                    "reservations management", "revenue optimization", "team leadership", "forecasting",
                    "inventory management", "customer service", "sales coordination", "performance analysis",
                    "staff training", "quality control", "communication", "strategic planning"
                ],
                "nice_to_have_skills": [
                    "channel management", "group bookings", "corporate sales", "yield management",
                    "competitive analysis", "market segmentation", "pricing strategies", "analytics"
                ],
                "technical_skills": [
                    "reservations systems", "revenue management systems", "channel management platforms",
                    "analytics tools", "forecasting software", "performance dashboards"
                ],
                "soft_skills": [
                    "analytical thinking", "leadership", "strategic thinking", "communication",
                    "attention to detail", "problem solving", "customer focus", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "analytical", "strategic", "detail-oriented", "results-driven",
                    "guest-focused", "leader", "innovative", "performance-oriented"
                ],
                "disqualifying_factors": [
                    "poor analytical skills", "lack of attention to detail", "poor leadership",
                    "inability to work with data", "poor communication"
                ],
                "experience_indicators": [
                    "reservations manager", "revenue manager", "reservations supervisor",
                    "hotel reservations", "booking management", "yield management"
                ],
                "education_preferences": [
                    "hospitality management", "revenue management", "business administration",
                    "hotel management", "analytics", "marketing"
                ],
                "certifications": [
                    "revenue management", "reservations management", "hospitality analytics",
                    "yield management", "hotel operations"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Reservations Agent": {
                "must_have_skills": [
                    "reservations systems", "customer service", "communication", "attention to detail",
                    "sales skills", "computer skills", "phone etiquette", "problem solving",
                    "multitasking", "time management", "accuracy", "hospitality"
                ],
                "nice_to_have_skills": [
                    "upselling", "group bookings", "multilingual", "travel knowledge",
                    "guest relations", "conflict resolution", "inventory management", "reporting"
                ],
                "technical_skills": [
                    "reservations software", "booking platforms", "payment processing",
                    "phone systems", "communication tools", "reporting systems"
                ],
                "soft_skills": [
                    "communication", "patience", "persuasion", "attention to detail",
                    "adaptability", "customer focus", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "helpful", "detail-oriented", "professional",
                    "patient", "sales-oriented", "guest-focused", "reliable"
                ],
                "disqualifying_factors": [
                    "poor communication", "lack of attention to detail", "poor customer service",
                    "inability to use technology", "impatient"
                ],
                "experience_indicators": [
                    "reservations agent", "booking agent", "customer service", "call center",
                    "travel agent", "sales agent", "hospitality"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality management", "customer service",
                    "tourism", "communications", "business studies"
                ],
                "certifications": [
                    "reservations certification", "customer service", "hospitality basics",
                    "sales training", "communication skills"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 28000, "max": 40000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Call Centre Agent": {
                "must_have_skills": [
                    "phone skills", "customer service", "communication", "computer skills",
                    "problem solving", "multitasking", "attention to detail", "patience",
                    "conflict resolution", "time management", "accuracy", "typing skills"
                ],
                "nice_to_have_skills": [
                    "hospitality knowledge", "sales skills", "multilingual", "data entry",
                    "CRM systems", "call center experience", "technical support", "documentation"
                ],
                "technical_skills": [
                    "call center software", "CRM systems", "phone systems", "computer applications",
                    "data entry systems", "ticketing systems", "communication platforms"
                ],
                "soft_skills": [
                    "communication", "patience", "empathy", "active listening",
                    "stress management", "adaptability", "problem solving", "resilience"
                ],
                "cultural_fit_keywords": [
                    "patient", "helpful", "professional", "calm",
                    "good listener", "problem-solver", "reliable", "courteous"
                ],
                "disqualifying_factors": [
                    "poor phone skills", "impatient", "poor listening skills",
                    "inability to handle stress", "poor computer skills"
                ],
                "experience_indicators": [
                    "call center", "customer service", "phone support", "technical support",
                    "help desk", "telemarketing", "customer care"
                ],
                "education_preferences": [
                    "high school diploma", "customer service", "communications",
                    "business studies", "hospitality basics"
                ],
                "certifications": [
                    "customer service", "call center operations", "communication skills",
                    "computer skills", "conflict resolution"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Lobby Ambassador": {
                "must_have_skills": [
                    "customer service", "communication", "hospitality", "guest relations",
                    "professional appearance", "local knowledge", "problem solving", "multilingual",
                    "cultural awareness", "patience", "friendliness", "adaptability"
                ],
                "nice_to_have_skills": [
                    "concierge services", "luxury hospitality", "event coordination",
                    "guest assistance", "information management", "networking", "VIP services"
                ],
                "technical_skills": [
                    "guest management systems", "communication devices", "information systems",
                    "booking platforms", "mobile devices", "hospitality software"
                ],
                "soft_skills": [
                    "interpersonal skills", "cultural sensitivity", "warmth", "professionalism",
                    "approachability", "helpfulness", "patience", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "welcoming", "friendly", "professional", "helpful",
                    "cultured", "sophisticated", "warm", "approachable"
                ],
                "disqualifying_factors": [
                    "poor communication", "unprofessional appearance", "lack of cultural awareness",
                    "unfriendly demeanor", "inflexible"
                ],
                "experience_indicators": [
                    "lobby ambassador", "guest relations", "concierge", "customer service",
                    "hospitality", "guest services", "tourism"
                ],
                "education_preferences": [
                    "hospitality management", "tourism", "communications", "languages",
                    "cultural studies", "customer service"
                ],
                "certifications": [
                    "hospitality service", "guest relations", "cultural awareness",
                    "customer service", "language certifications"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.25, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # DIAMOND CLUB / VIP SERVICES
            # ==========================================
            "Diamond Club Manager": {
                "must_have_skills": [
                    "luxury service", "VIP management", "customer relations", "team leadership",
                    "exclusive services", "hospitality excellence", "communication", "problem solving",
                    "attention to detail", "cultural sensitivity", "discretion", "service standards"
                ],
                "nice_to_have_skills": [
                    "personal concierge", "luxury hospitality", "exclusive events", "high-end dining",
                    "premium amenities", "guest personalization", "luxury brands", "etiquette"
                ],
                "technical_skills": [
                    "guest management systems", "VIP tracking", "luxury service platforms",
                    "communication systems", "event management software", "guest preferences"
                ],
                "soft_skills": [
                    "sophistication", "discretion", "excellence", "leadership",
                    "attention to detail", "cultural awareness", "refinement", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "luxury-focused", "sophisticated", "discreet", "excellence-driven",
                    "refined", "exclusive", "high-standards", "professional"
                ],
                "disqualifying_factors": [
                    "lack of luxury experience", "poor attention to detail", "unprofessional",
                    "lack of discretion", "poor communication"
                ],
                "experience_indicators": [
                    "diamond club", "VIP services", "luxury hospitality", "concierge management",
                    "exclusive services", "high-end service", "luxury management"
                ],
                "education_preferences": [
                    "hospitality management", "luxury service", "hotel administration",
                    "business management", "customer experience"
                ],
                "certifications": [
                    "luxury service", "hospitality excellence", "VIP management",
                    "customer experience", "concierge certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.25, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 60000, "max": 90000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Head Butler": {
                "must_have_skills": [
                    "luxury service", "personal service", "attention to detail", "discretion",
                    "VIP handling", "team leadership", "communication", "anticipation",
                    "guest preferences", "high standards", "professionalism", "etiquette"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "wine service", "personal shopping", "event coordination",
                    "travel arrangements", "personal concierge", "luxury brands", "cultural knowledge"
                ],
                "technical_skills": [
                    "guest preference systems", "communication devices", "luxury service tools",
                    "scheduling systems", "personal service equipment"
                ],
                "soft_skills": [
                    "anticipation", "discretion", "sophistication", "attentiveness",
                    "cultural sensitivity", "professionalism", "reliability", "excellence"
                ],
                "cultural_fit_keywords": [
                    "sophisticated", "discreet", "attentive", "anticipatory",
                    "refined", "professional", "service-excellence", "detail-oriented"
                ],
                "disqualifying_factors": [
                    "lack of sophistication", "poor attention to detail", "indiscreet",
                    "unprofessional", "lack of service excellence"
                ],
                "experience_indicators": [
                    "head butler", "personal butler", "luxury service", "VIP service",
                    "personal concierge", "high-end service", "exclusive service"
                ],
                "education_preferences": [
                    "hospitality excellence", "luxury service", "hotel management",
                    "butler training", "customer experience"
                ],
                "certifications": [
                    "butler certification", "luxury service", "VIP service",
                    "hospitality excellence", "personal service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 70000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Butler Supervisor": {
                "must_have_skills": [
                    "luxury service", "team supervision", "service standards", "training",
                    "guest relations", "communication", "attention to detail", "leadership",
                    "quality control", "VIP service", "professional standards", "discretion"
                ],
                "nice_to_have_skills": [
                    "butler training", "luxury hospitality", "personal service", "event coordination",
                    "staff development", "service excellence", "guest preferences", "etiquette"
                ],
                "technical_skills": [
                    "staff scheduling", "service management", "guest tracking", "quality systems",
                    "communication platforms", "training systems"
                ],
                "soft_skills": [
                    "leadership", "attention to detail", "professionalism", "discretion",
                    "sophistication", "communication", "reliability", "excellence"
                ],
                "cultural_fit_keywords": [
                    "professional", "sophisticated", "detail-oriented", "leader",
                    "service-focused", "discreet", "refined", "quality-driven"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of attention to detail", "unprofessional",
                    "lack of service experience", "poor communication"
                ],
                "experience_indicators": [
                    "butler supervisor", "luxury service supervisor", "VIP services",
                    "personal service", "hospitality supervision", "service management"
                ],
                "education_preferences": [
                    "hospitality management", "luxury service", "butler training",
                    "service excellence", "team leadership"
                ],
                "certifications": [
                    "butler certification", "luxury service", "hospitality leadership",
                    "service excellence", "team management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.25, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Butler": {
                "must_have_skills": [
                    "personal service", "attention to detail", "discretion", "guest relations",
                    "communication", "professionalism", "VIP service", "anticipation",
                    "luxury standards", "etiquette", "cultural sensitivity", "reliability"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "wine knowledge", "personal shopping", "travel assistance",
                    "event coordination", "luxury brands", "cultural knowledge", "languages"
                ],
                "technical_skills": [
                    "guest preference systems", "communication devices", "service equipment",
                    "scheduling tools", "personal service technology"
                ],
                "soft_skills": [
                    "discretion", "attentiveness", "sophistication", "anticipation",
                    "professionalism", "cultural sensitivity", "reliability", "patience"
                ],
                "cultural_fit_keywords": [
                    "discreet", "attentive", "professional", "sophisticated",
                    "service-oriented", "refined", "anticipatory", "reliable"
                ],
                "disqualifying_factors": [
                    "lack of discretion", "poor attention to detail", "unprofessional",
                    "lack of sophistication", "poor service attitude"
                ],
                "experience_indicators": [
                    "butler", "personal service", "VIP service", "luxury service",
                    "personal concierge", "exclusive service", "high-end service"
                ],
                "education_preferences": [
                    "butler training", "hospitality service", "luxury service",
                    "customer service", "cultural studies"
                ],
                "certifications": [
                    "butler certification", "luxury service", "personal service",
                    "VIP service", "hospitality excellence"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # HOUSEKEEPING & LAUNDRY
            # ==========================================
            "Executive Housekeeper": {
                "must_have_skills": [
                    "housekeeping operations", "team management", "budget management", "quality control",
                    "staff supervision", "inventory management", "training", "safety protocols",
                    "performance management", "vendor relations", "strategic planning", "cost control"
                ],
                "nice_to_have_skills": [
                    "luxury housekeeping", "laundry operations", "facility management", "green practices",
                    "technology implementation", "process improvement", "compliance", "multi-property"
                ],
                "technical_skills": [
                    "housekeeping management systems", "inventory software", "scheduling platforms",
                    "budgeting tools", "performance analytics", "compliance tracking"
                ],
                "soft_skills": [
                    "strategic leadership", "organization", "attention to detail", "communication",
                    "analytical thinking", "problem solving", "team building", "innovation"
                ],
                "cultural_fit_keywords": [
                    "organized", "strategic", "detail-oriented", "efficient",
                    "quality-focused", "leader", "innovative", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of leadership", "poor attention to detail",
                    "inability to manage budgets", "poor communication"
                ],
                "experience_indicators": [
                    "executive housekeeper", "housekeeping manager", "facility management",
                    "housekeeping operations", "hotel housekeeping", "cleaning operations"
                ],
                "education_preferences": [
                    "hospitality management", "facility management", "business management",
                    "housekeeping management", "operations management"
                ],
                "certifications": [
                    "executive housekeeping", "facility management", "hospitality management",
                    "safety certification", "quality management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 55000, "max": 80000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Housekeeping Supervisor": {
                "must_have_skills": [
                    "housekeeping operations", "team supervision", "quality control",
                    "inventory management", "scheduling", "training", "safety protocols",
                    "staff coordination", "guest room standards", "time management", "communication"
                ],
                "nice_to_have_skills": [
                    "laundry operations", "deep cleaning", "maintenance coordination",
                    "budget management", "eco-friendly practices", "lost and found", "housekeeping software"
                ],
                "technical_skills": [
                    "housekeeping software", "scheduling systems", "inventory management",
                    "cleaning equipment", "laundry systems", "room management systems"
                ],
                "soft_skills": [
                    "leadership", "organization", "attention to detail", "time management",
                    "problem solving", "communication", "training abilities", "reliability"
                ],
                "cultural_fit_keywords": [
                    "organized", "detail-oriented", "efficient", "reliable", "leader",
                    "quality-focused", "thorough", "systematic", "accountable"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail",
                    "inability to manage teams", "poor time management", "unreliable"
                ],
                "experience_indicators": [
                    "housekeeping supervisor", "housekeeping management", "hotel housekeeping",
                    "room attendant supervisor", "laundry operations", "cleaning supervision"
                ],
                "education_preferences": [
                    "hospitality management", "hotel operations", "business management",
                    "facility management", "housekeeping management"
                ],
                "certifications": [
                    "housekeeping certification", "hospitality operations",
                    "safety certification", "management training", "quality control"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Room Attendant": {
                "must_have_skills": [
                    "cleaning", "attention to detail", "time management", "physical stamina", "organization",
                    "guest room standards", "housekeeping procedures", "safety awareness", "efficiency",
                    "inventory awareness", "quality standards", "reliability"
                ],
                "nice_to_have_skills": [
                    "hotel cleaning", "laundry", "inventory", "guest interaction",
                    "deep cleaning", "eco-friendly practices", "maintenance reporting", "luxury standards"
                ],
                "technical_skills": [
                    "cleaning equipment", "housekeeping supplies", "laundry equipment",
                    "room maintenance tools", "safety equipment", "cleaning chemicals"
                ],
                "soft_skills": [
                    "thoroughness", "reliability", "efficiency", "professionalism",
                    "physical stamina", "attention to detail", "time management", "independence"
                ],
                "cultural_fit_keywords": [
                    "thorough", "reliable", "efficient", "professional", "hardworking",
                    "detail-oriented", "independent", "quality-focused", "conscientious"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of physical stamina", "unreliable",
                    "poor time management", "unprofessional"
                ],
                "experience_indicators": [
                    "room attendant", "housekeeper", "hotel cleaning", "housekeeping",
                    "cleaning services", "room cleaning", "hospitality cleaning"
                ],
                "education_preferences": [
                    "high school diploma", "housekeeping training", "hospitality basics",
                    "cleaning services", "customer service"
                ],
                "certifications": [
                    "housekeeping certification", "safety training", "cleaning certification",
                    "hospitality service", "chemical safety"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 22000, "max": 32000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Public Area Attendant": {
                "must_have_skills": [
                    "cleaning", "public space maintenance", "attention to detail", "time management",
                    "safety awareness", "customer interaction", "efficiency", "reliability",
                    "organization", "quality standards", "physical stamina", "professionalism"
                ],
                "nice_to_have_skills": [
                    "floor care", "carpet cleaning", "window cleaning", "maintenance awareness",
                    "guest interaction", "equipment operation", "eco-friendly practices", "inventory"
                ],
                "technical_skills": [
                    "cleaning equipment", "floor care machines", "carpet cleaners",
                    "window cleaning tools", "safety equipment", "maintenance tools"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "efficiency", "customer awareness",
                    "physical stamina", "independence", "professionalism", "thoroughness"
                ],
                "cultural_fit_keywords": [
                    "thorough", "reliable", "efficient", "professional",
                    "detail-oriented", "hardworking", "guest-aware", "quality-focused"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of physical stamina", "unreliable",
                    "unprofessional appearance", "poor customer interaction"
                ],
                "experience_indicators": [
                    "public area attendant", "cleaning", "facility cleaning", "maintenance cleaning",
                    "commercial cleaning", "hospitality cleaning", "janitorial"
                ],
                "education_preferences": [
                    "high school diploma", "cleaning services", "facility maintenance",
                    "customer service", "safety training"
                ],
                "certifications": [
                    "cleaning certification", "safety training", "equipment operation",
                    "chemical safety", "hospitality service"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Turndown Attendant": {
                "must_have_skills": [
                    "guest service", "attention to detail", "discretion", "time management",
                    "luxury service", "guest room preparation", "quality standards", "professionalism",
                    "cultural sensitivity", "reliability", "efficiency", "hospitality"
                ],
                "nice_to_have_skills": [
                    "luxury hospitality", "guest preferences", "evening service", "amenity placement",
                    "room ambiance", "personalized service", "VIP service", "cultural awareness"
                ],
                "technical_skills": [
                    "room preparation", "amenity placement", "lighting control",
                    "guest preference systems", "luxury service tools"
                ],
                "soft_skills": [
                    "discretion", "attention to detail", "cultural sensitivity", "professionalism",
                    "reliability", "efficiency", "guest awareness", "sophistication"
                ],
                "cultural_fit_keywords": [
                    "discreet", "detail-oriented", "professional", "sophisticated",
                    "guest-focused", "reliable", "quality-oriented", "refined"
                ],
                "disqualifying_factors": [
                    "lack of discretion", "poor attention to detail", "unprofessional",
                    "lack of cultural sensitivity", "unreliable"
                ],
                "experience_indicators": [
                    "turndown service", "evening service", "luxury hospitality", "guest services",
                    "room service", "hospitality service", "hotel service"
                ],
                "education_preferences": [
                    "hospitality service", "luxury hospitality", "customer service",
                    "hotel service", "guest relations"
                ],
                "certifications": [
                    "luxury service", "hospitality service", "guest relations",
                    "customer service", "cultural awareness"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 25000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Houseman": {
                "must_have_skills": [
                    "physical labor", "cleaning support", "maintenance awareness", "reliability",
                    "teamwork", "safety awareness", "equipment operation", "time management",
                    "attention to detail", "following instructions", "housekeeping support", "efficiency"
                ],
                "nice_to_have_skills": [
                    "basic maintenance", "equipment maintenance", "inventory support",
                    "guest interaction", "heavy lifting", "grounds maintenance", "janitorial"
                ],
                "technical_skills": [
                    "cleaning equipment", "maintenance tools", "safety equipment",
                    "heavy equipment", "support machinery", "housekeeping supplies"
                ],
                "soft_skills": [
                    "physical stamina", "reliability", "teamwork", "following directions",
                    "attention to detail", "safety consciousness", "hardworking", "dependability"
                ],
                "cultural_fit_keywords": [
                    "hardworking", "reliable", "team player", "dependable",
                    "strong", "efficient", "supportive", "conscientious"
                ],
                "disqualifying_factors": [
                    "poor physical condition", "unreliable", "lack of teamwork",
                    "poor safety awareness", "inability to follow instructions"
                ],
                "experience_indicators": [
                    "houseman", "cleaning support", "facility support", "maintenance support",
                    "janitorial", "physical labor", "hospitality support"
                ],
                "education_preferences": [
                    "high school diploma", "physical labor", "facility maintenance",
                    "safety training", "equipment operation"
                ],
                "certifications": [
                    "safety training", "equipment operation", "physical fitness",
                    "cleaning certification", "workplace safety"
                ],
                "scoring_weights": {
                    "experience": 0.15, "skills": 0.35, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Linen Room Supervisor": {
                "must_have_skills": [
                    "linen management", "inventory control", "team supervision", "quality control",
                    "organization", "scheduling", "staff training", "efficiency optimization",
                    "cost control", "vendor relations", "performance management", "communication"
                ],
                "nice_to_have_skills": [
                    "laundry operations", "textile knowledge", "budget management", "process improvement",
                    "technology implementation", "compliance", "waste reduction", "sustainability"
                ],
                "technical_skills": [
                    "inventory management systems", "linen tracking", "scheduling software",
                    "quality control systems", "cost analysis tools", "performance metrics"
                ],
                "soft_skills": [
                    "organization", "leadership", "attention to detail", "analytical thinking",
                    "problem solving", "communication", "time management", "efficiency"
                ],
                "cultural_fit_keywords": [
                    "organized", "efficient", "detail-oriented", "systematic",
                    "quality-focused", "leader", "analytical", "cost-conscious"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail", "poor leadership",
                    "inability to manage inventory", "poor communication"
                ],
                "experience_indicators": [
                    "linen room supervisor", "inventory supervisor", "laundry supervisor",
                    "textile management", "housekeeping supervision", "linen management"
                ],
                "education_preferences": [
                    "inventory management", "hospitality operations", "business management",
                    "textile management", "operations management"
                ],
                "certifications": [
                    "inventory management", "laundry operations", "hospitality operations",
                    "quality control", "supervisory training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Laundry Manager": {
                "must_have_skills": [
                    "laundry operations", "team management", "equipment management", "quality control",
                    "cost control", "scheduling", "staff supervision", "performance management",
                    "vendor relations", "compliance", "efficiency optimization", "budget management"
                ],
                "nice_to_have_skills": [
                    "textile knowledge", "chemical management", "equipment maintenance", "process improvement",
                    "environmental compliance", "energy efficiency", "technology implementation", "training"
                ],
                "technical_skills": [
                    "laundry equipment", "chemical management systems", "quality control systems",
                    "scheduling software", "cost analysis tools", "maintenance management"
                ],
                "soft_skills": [
                    "leadership", "organization", "analytical thinking", "problem solving",
                    "communication", "attention to detail", "efficiency", "innovation"
                ],
                "cultural_fit_keywords": [
                    "efficient", "organized", "analytical", "quality-focused",
                    "leader", "cost-conscious", "innovative", "systematic"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of technical knowledge", "poor organizational skills",
                    "inability to manage costs", "poor safety awareness"
                ],
                "experience_indicators": [
                    "laundry manager", "laundry operations", "textile management", "commercial laundry",
                    "laundry supervisor", "facility operations", "housekeeping operations"
                ],
                "education_preferences": [
                    "operations management", "facility management", "textile management",
                    "business management", "hospitality operations"
                ],
                "certifications": [
                    "laundry operations", "facility management", "chemical safety",
                    "equipment operation", "environmental compliance"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Laundry Supervisor": {
                "must_have_skills": [
                    "laundry operations", "team supervision", "equipment operation", "quality control",
                    "scheduling", "staff training", "safety protocols", "efficiency",
                    "cost awareness", "problem solving", "communication", "time management"
                ],
                "nice_to_have_skills": [
                    "textile knowledge", "chemical handling", "equipment maintenance", "inventory management",
                    "process improvement", "training development", "performance monitoring", "compliance"
                ],
                "technical_skills": [
                    "laundry equipment", "washing machines", "dryers", "pressing equipment",
                    "chemical dispensing", "quality control tools", "scheduling systems"
                ],
                "soft_skills": [
                    "leadership", "organization", "attention to detail", "problem solving",
                    "communication", "reliability", "efficiency", "safety consciousness"
                ],
                "cultural_fit_keywords": [
                    "organized", "efficient", "detail-oriented", "reliable",
                    "quality-focused", "leader", "safety-conscious", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail", "poor leadership",
                    "unsafe practices", "poor communication"
                ],
                "experience_indicators": [
                    "laundry supervisor", "laundry operations", "commercial laundry", "textile processing",
                    "laundry attendant", "housekeeping laundry", "industrial laundry"
                ],
                "education_preferences": [
                    "laundry operations", "textile processing", "hospitality operations",
                    "supervisory training", "safety management"
                ],
                "certifications": [
                    "laundry operations", "chemical safety", "equipment operation",
                    "safety certification", "supervisory training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 32000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Laundry Attendant": {
                "must_have_skills": [
                    "laundry operations", "equipment operation", "attention to detail", "efficiency",
                    "safety awareness", "quality control", "time management", "following procedures",
                    "chemical handling", "reliability", "physical stamina", "organization"
                ],
                "nice_to_have_skills": [
                    "textile knowledge", "stain removal", "pressing", "folding", "sorting",
                    "equipment maintenance", "inventory awareness", "customer service"
                ],
                "technical_skills": [
                    "washing machines", "dryers", "pressing equipment", "folding equipment",
                    "chemical dispensing", "quality control", "laundry supplies"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "efficiency", "physical stamina",
                    "following directions", "safety consciousness", "thoroughness", "consistency"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "efficient", "hardworking",
                    "thorough", "consistent", "quality-focused", "safety-conscious"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "unreliable", "unsafe practices",
                    "lack of physical stamina", "inability to follow procedures"
                ],
                "experience_indicators": [
                    "laundry attendant", "laundry operator", "commercial laundry", "dry cleaning",
                    "textile processing", "laundry services", "industrial laundry"
                ],
                "education_preferences": [
                    "high school diploma", "laundry training", "equipment operation",
                    "safety training", "chemical handling"
                ],
                "certifications": [
                    "laundry operations", "equipment operation", "chemical safety",
                    "safety training", "quality control"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 20000, "max": 30000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Laundry Technician": {
                "must_have_skills": [
                    "equipment maintenance", "technical troubleshooting", "laundry operations", "mechanical skills",
                    "electrical basics", "preventive maintenance", "repair skills", "safety protocols",
                    "problem solving", "attention to detail", "reliability", "communication"
                ],
                "nice_to_have_skills": [
                    "equipment installation", "HVAC basics", "plumbing basics", "chemical systems",
                    "computerized controls", "energy efficiency", "predictive maintenance", "vendor coordination"
                ],
                "technical_skills": [
                    "laundry equipment", "mechanical systems", "electrical systems", "control systems",
                    "diagnostic tools", "maintenance tools", "repair equipment"
                ],
                "soft_skills": [
                    "problem solving", "analytical thinking", "attention to detail", "reliability",
                    "communication", "learning agility", "safety consciousness", "independence"
                ],
                "cultural_fit_keywords": [
                    "technical", "analytical", "problem-solver", "reliable",
                    "detail-oriented", "safety-conscious", "efficient", "skilled"
                ],
                "disqualifying_factors": [
                    "poor technical skills", "unsafe practices", "unreliable",
                    "poor problem-solving", "lack of attention to detail"
                ],
                "experience_indicators": [
                    "laundry technician", "equipment technician", "maintenance technician", "laundry maintenance",
                    "equipment repair", "mechanical repair", "commercial laundry"
                ],
                "education_preferences": [
                    "technical training", "mechanical engineering", "electrical training",
                    "equipment maintenance", "facility maintenance"
                ],
                "certifications": [
                    "equipment maintenance", "electrical certification", "mechanical certification",
                    "safety certification", "HVAC basics"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Uniform Room Attendant": {
                "must_have_skills": [
                    "inventory management", "organization", "attention to detail", "customer service",
                    "time management", "efficiency", "record keeping", "communication",
                    "quality control", "distribution management", "reliability", "professionalism"
                ],
                "nice_to_have_skills": [
                    "sizing knowledge", "alteration basics", "textile knowledge", "computer skills",
                    "staff interaction", "problem solving", "inventory software", "scheduling"
                ],
                "technical_skills": [
                    "inventory systems", "distribution tracking", "computer applications",
                    "sizing tools", "record keeping systems", "communication tools"
                ],
                "soft_skills": [
                    "organization", "attention to detail", "customer service", "communication",
                    "reliability", "efficiency", "professionalism", "helpfulness"
                ],
                "cultural_fit_keywords": [
                    "organized", "detail-oriented", "helpful", "efficient",
                    "reliable", "professional", "service-oriented", "systematic"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "lack of attention to detail", "poor customer service",
                    "unreliable", "poor communication"
                ],
                "experience_indicators": [
                    "uniform room", "inventory management", "distribution", "retail experience",
                    "customer service", "hospitality support", "clothing distribution"
                ],
                "education_preferences": [
                    "high school diploma", "inventory management", "customer service",
                    "retail experience", "hospitality service"
                ],
                "certifications": [
                    "inventory management", "customer service", "hospitality service",
                    "retail operations", "communication skills"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 22000, "max": 32000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            # ==========================================
            # FOOD & BEVERAGE DEPARTMENT
            # ==========================================
            "Director of Food & Beverage": {
                "must_have_skills": [
                    "F&B operations", "strategic leadership", "revenue management", "team management",
                    "cost control", "menu development", "vendor management", "quality standards",
                    "performance management", "budget planning", "staff development", "guest satisfaction"
                ],
                "nice_to_have_skills": [
                    "multi-unit operations", "franchise operations", "luxury dining", "beverage programs",
                    "event catering", "wine programs", "sustainability", "technology implementation"
                ],
                "technical_skills": [
                    "POS systems", "inventory management", "cost analysis", "revenue analytics",
                    "performance dashboards", "scheduling software", "compliance tracking"
                ],
                "soft_skills": [
                    "strategic leadership", "analytical thinking", "communication", "innovation",
                    "team building", "decision making", "customer focus", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "strategic", "innovative", "results-driven", "guest-focused",
                    "leader", "analytical", "quality-oriented", "collaborative"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of F&B experience", "poor financial management",
                    "inability to handle complexity", "poor communication"
                ],
                "experience_indicators": [
                    "F&B director", "restaurant operations", "food service management", "hospitality management",
                    "multi-unit management", "food and beverage", "culinary management"
                ],
                "education_preferences": [
                    "hospitality management", "culinary management", "business administration",
                    "food service management", "restaurant management", "MBA"
                ],
                "certifications": [
                    "ServSafe", "hospitality management", "food service management",
                    "revenue management", "wine certification"
                ],
                "scoring_weights": {
                    "experience": 0.45, "skills": 0.25, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 8,
                "preferred_experience_years": 12,
                "salary_range": {"min": 90000, "max": 140000},
                "growth_potential": "Executive",
                "training_requirements": "Strategic"
            },

            "Restaurant Manager": {
                "must_have_skills": [
                    "restaurant operations", "team management", "customer service", "cost control",
                    "staff supervision", "quality control", "performance management", "scheduling",
                    "inventory management", "guest relations", "problem solving", "communication"
                ],
                "nice_to_have_skills": [
                    "fine dining", "wine knowledge", "event management", "catering", "marketing",
                    "revenue optimization", "staff training", "menu development", "POS systems"
                ],
                "technical_skills": [
                    "POS systems", "restaurant management software", "inventory systems",
                    "scheduling platforms", "performance analytics", "cost analysis tools"
                ],
                "soft_skills": [
                    "leadership", "customer focus", "problem solving", "communication",
                    "multitasking", "stress management", "team building", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "guest-focused", "leader", "service-oriented", "quality-driven",
                    "team builder", "efficient", "professional", "hospitality-minded"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of restaurant experience", "poor customer service",
                    "inability to handle stress", "poor communication"
                ],
                "experience_indicators": [
                    "restaurant manager", "food service manager", "dining manager", "hospitality management",
                    "restaurant operations", "food and beverage", "restaurant supervision"
                ],
                "education_preferences": [
                    "hospitality management", "restaurant management", "culinary management",
                    "business administration", "food service management"
                ],
                "certifications": [
                    "ServSafe", "restaurant management", "food service", "wine certification",
                    "hospitality management"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 70000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Restaurant Supervisor": {
                "must_have_skills": [
                    "restaurant operations", "team supervision", "customer service", "quality control",
                    "staff training", "communication", "problem solving", "multitasking",
                    "performance monitoring", "guest relations", "conflict resolution", "efficiency"
                ],
                "nice_to_have_skills": [
                    "POS systems", "inventory awareness", "cash handling", "scheduling support",
                    "menu knowledge", "wine basics", "event support", "staff development"
                ],
                "technical_skills": [
                    "POS systems", "restaurant software", "communication tools",
                    "scheduling systems", "performance tracking", "guest management"
                ],
                "soft_skills": [
                    "leadership", "communication", "customer focus", "problem solving",
                    "adaptability", "team building", "patience", "multitasking"
                ],
                "cultural_fit_keywords": [
                    "team leader", "guest-focused", "supportive", "professional",
                    "service-oriented", "reliable", "efficient", "helpful"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "poor customer service", "inability to multitask",
                    "poor communication", "lack of restaurant experience"
                ],
                "experience_indicators": [
                    "restaurant supervisor", "dining supervisor", "food service supervisor",
                    "restaurant lead", "hospitality supervision", "food and beverage"
                ],
                "education_preferences": [
                    "hospitality management", "restaurant management", "food service",
                    "customer service", "business studies"
                ],
                "certifications": [
                    "ServSafe", "restaurant operations", "food service", "customer service",
                    "leadership development"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Captain": {
                "must_have_skills": [
                    "fine dining service", "guest relations", "communication", "team coordination",
                    "menu knowledge", "wine service", "attention to detail", "hospitality",
                    "professional presentation", "problem solving", "multitasking", "leadership"
                ],
                "nice_to_have_skills": [
                    "sommelier knowledge", "luxury service", "event service", "language skills",
                    "cultural awareness", "special dietary knowledge", "POS systems", "training"
                ],
                "technical_skills": [
                    "POS systems", "wine service tools", "dining service equipment",
                    "communication systems", "reservation systems", "payment processing"
                ],
                "soft_skills": [
                    "sophistication", "communication", "leadership", "attention to detail",
                    "customer focus", "cultural sensitivity", "professionalism", "patience"
                ],
                "cultural_fit_keywords": [
                    "sophisticated", "professional", "service-excellence", "refined",
                    "knowledgeable", "leader", "guest-focused", "attentive"
                ],
                "disqualifying_factors": [
                    "lack of fine dining experience", "poor communication", "unprofessional",
                    "lack of sophistication", "poor attention to detail"
                ],
                "experience_indicators": [
                    "captain", "fine dining", "restaurant captain", "dining service",
                    "luxury service", "hospitality service", "wine service"
                ],
                "education_preferences": [
                    "hospitality management", "culinary arts", "wine studies",
                    "fine dining service", "restaurant management"
                ],
                "certifications": [
                    "wine certification", "fine dining service", "sommelier",
                    "hospitality excellence", "luxury service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Waiter": {
                "must_have_skills": [
                    "customer service", "communication", "multitasking", "attention to detail",
                    "team work", "menu knowledge", "cash handling", "problem solving",
                    "time management", "professional appearance", "hospitality", "efficiency"
                ],
                "nice_to_have_skills": [
                    "fine dining experience", "wine knowledge", "multilingual", "POS systems",
                    "upselling", "special dietary knowledge", "event service", "cultural awareness"
                ],
                "technical_skills": [
                    "POS systems", "payment processing", "order management",
                    "communication devices", "service equipment", "cash handling"
                ],
                "soft_skills": [
                    "communication", "friendliness", "patience", "adaptability",
                    "team work", "customer focus", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "professional", "attentive", "team player",
                    "guest-focused", "reliable", "energetic", "service-oriented"
                ],
                "disqualifying_factors": [
                    "poor customer service", "inability to multitask", "poor communication",
                    "unreliable", "unprofessional appearance"
                ],
                "experience_indicators": [
                    "waiter", "server", "restaurant service", "food service", "dining service",
                    "hospitality service", "customer service", "restaurant experience"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "food service",
                    "customer service", "restaurant training"
                ],
                "certifications": [
                    "food service", "customer service", "responsible beverage service",
                    "hospitality basics", "POS certification"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 20000, "max": 35000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Server": {
                "must_have_skills": [
                    "customer service", "food knowledge", "multitasking", "communication", "cash handling",
                    "attention to detail", "team work", "time management", "hospitality", "efficiency",
                    "problem solving", "professional appearance"
                ],
                "nice_to_have_skills": [
                    "fine dining", "wine service", "allergen knowledge", "upselling", "POS systems",
                    "multilingual", "event service", "special dietary needs", "beverage knowledge"
                ],
                "technical_skills": [
                    "POS systems", "payment processing", "order management", "communication tools",
                    "service equipment", "cash register", "mobile ordering"
                ],
                "soft_skills": [
                    "friendliness", "attentiveness", "professionalism", "team player", "energetic",
                    "patience", "adaptability", "customer focus", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "friendly", "attentive", "professional", "team player", "energetic",
                    "guest-focused", "reliable", "service-oriented", "welcoming"
                ],
                "disqualifying_factors": [
                    "poor customer service", "inability to multitask", "poor communication",
                    "unreliable", "unprofessional", "lack of energy"
                ],
                "experience_indicators": [
                    "server", "food server", "restaurant server", "dining service", "food service",
                    "hospitality service", "customer service", "waitstaff"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "food service training",
                    "customer service", "restaurant experience"
                ],
                "certifications": [
                    "food service", "responsible beverage service", "customer service",
                    "allergen awareness", "hospitality basics"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 18000, "max": 32000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Host/Hostess": {
                "must_have_skills": [
                    "customer service", "communication", "organization", "multitasking",
                    "professional appearance", "hospitality", "phone skills", "problem solving",
                    "attention to detail", "time management", "guest relations", "teamwork"
                ],
                "nice_to_have_skills": [
                    "reservation systems", "multilingual", "event coordination", "guest recognition",
                    "wait management", "conflict resolution", "cultural awareness", "upselling"
                ],
                "technical_skills": [
                    "reservation systems", "POS systems", "phone systems", "seating management",
                    "communication tools", "scheduling software", "guest management"
                ],
                "soft_skills": [
                    "friendliness", "professionalism", "patience", "organization",
                    "communication", "adaptability", "positive attitude", "welcoming nature"
                ],
                "cultural_fit_keywords": [
                    "welcoming", "friendly", "professional", "organized",
                    "guest-focused", "positive", "reliable", "courteous"
                ],
                "disqualifying_factors": [
                    "poor communication", "unprofessional appearance", "lack of organization",
                    "unfriendly demeanor", "poor customer service"
                ],
                "experience_indicators": [
                    "host", "hostess", "restaurant host", "guest seating", "front of house",
                    "customer service", "hospitality", "restaurant experience"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "customer service",
                    "communication skills", "restaurant training"
                ],
                "certifications": [
                    "customer service", "hospitality basics", "communication skills",
                    "guest relations", "restaurant operations"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 18000, "max": 28000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Banquet Manager": {
                "must_have_skills": [
                    "event management", "team leadership", "banquet operations", "customer service",
                    "coordination", "communication", "problem solving", "time management",
                    "staff supervision", "quality control", "logistics", "budget awareness"
                ],
                "nice_to_have_skills": [
                    "large scale events", "wedding coordination", "corporate events", "menu planning",
                    "vendor coordination", "audio visual", "decoration", "protocol knowledge"
                ],
                "technical_skills": [
                    "event management software", "banquet management systems", "audio visual equipment",
                    "communication tools", "scheduling systems", "logistics software"
                ],
                "soft_skills": [
                    "leadership", "organization", "problem solving", "communication",
                    "multitasking", "attention to detail", "stress management", "flexibility"
                ],
                "cultural_fit_keywords": [
                    "organized", "leader", "detail-oriented", "flexible",
                    "guest-focused", "professional", "efficient", "collaborative"
                ],
                "disqualifying_factors": [
                    "poor organizational skills", "inability to handle stress", "poor leadership",
                    "lack of event experience", "poor communication"
                ],
                "experience_indicators": [
                    "banquet manager", "event manager", "catering manager", "banquet operations",
                    "event coordination", "hospitality events", "function management"
                ],
                "education_preferences": [
                    "hospitality management", "event management", "hotel management",
                    "business administration", "catering management"
                ],
                "certifications": [
                    "event management", "banquet operations", "hospitality management",
                    "catering certification", "food service"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Banquet Captain": {
                "must_have_skills": [
                    "banquet service", "team coordination", "event execution", "guest service",
                    "communication", "leadership", "attention to detail", "time management",
                    "staff coordination", "quality control", "problem solving", "professionalism"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "large events", "wedding service", "corporate events",
                    "wine service", "protocol knowledge", "multilingual", "training abilities"
                ],
                "technical_skills": [
                    "banquet equipment", "audio visual basics", "communication devices",
                    "service tools", "event setup equipment", "catering equipment"
                ],
                "soft_skills": [
                    "leadership", "organization", "communication", "adaptability",
                    "team coordination", "attention to detail", "stress management", "professionalism"
                ],
                "cultural_fit_keywords": [
                    "leader", "organized", "professional", "detail-oriented",
                    "team player", "guest-focused", "reliable", "efficient"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of event experience", "poor communication",
                    "inability to handle stress", "unprofessional"
                ],
                "experience_indicators": [
                    "banquet captain", "event captain", "banquet service", "event service",
                    "catering service", "function service", "hospitality events"
                ],
                "education_preferences": [
                    "hospitality management", "event management", "food service",
                    "banquet operations", "customer service"
                ],
                "certifications": [
                    "banquet service", "event service", "food service", "hospitality operations",
                    "team leadership"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 32000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Banquet Server": {
                "must_have_skills": [
                    "banquet service", "customer service", "team work", "communication",
                    "attention to detail", "time management", "physical stamina", "professionalism",
                    "event service", "multitasking", "efficiency", "following procedures"
                ],
                "nice_to_have_skills": [
                    "fine dining service", "wine service", "large event experience", "setup/breakdown",
                    "special dietary awareness", "multilingual", "formal service", "protocol"
                ],
                "technical_skills": [
                    "banquet equipment", "service tools", "audio visual basics",
                    "communication devices", "catering equipment", "setup equipment"
                ],
                "soft_skills": [
                    "teamwork", "adaptability", "communication", "attention to detail",
                    "physical stamina", "professionalism", "reliability", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "team player", "professional", "reliable", "efficient",
                    "guest-focused", "adaptable", "hardworking", "detail-oriented"
                ],
                "disqualifying_factors": [
                    "poor teamwork", "lack of physical stamina", "unprofessional",
                    "poor attention to detail", "unreliable"
                ],
                "experience_indicators": [
                    "banquet server", "event server", "catering server", "function server",
                    "banquet service", "event service", "hospitality service"
                ],
                "education_preferences": [
                    "high school diploma", "food service training", "banquet service",
                    "hospitality basics", "customer service"
                ],
                "certifications": [
                    "food service", "banquet service", "customer service",
                    "responsible beverage service", "hospitality basics"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 18000, "max": 28000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Bar Manager": {
                "must_have_skills": [
                    "bar operations", "team management", "beverage knowledge", "inventory control",
                    "cost control", "staff supervision", "customer service", "communication",
                    "performance management", "scheduling", "quality control", "problem solving"
                ],
                "nice_to_have_skills": [
                    "craft cocktails", "wine program", "beer knowledge", "mixology", "bar design",
                    "event bars", "training development", "vendor relations", "marketing"
                ],
                "technical_skills": [
                    "POS systems", "inventory management", "bar equipment", "cost analysis",
                    "scheduling software", "performance analytics", "beverage systems"
                ],
                "soft_skills": [
                    "leadership", "creativity", "communication", "analytical thinking",
                    "customer focus", "team building", "problem solving", "innovation"
                ],
                "cultural_fit_keywords": [
                    "creative", "leader", "knowledgeable", "innovative",
                    "guest-focused", "quality-driven", "team builder", "professional"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of beverage knowledge", "poor cost control",
                    "inability to manage staff", "poor communication"
                ],
                "experience_indicators": [
                    "bar manager", "beverage manager", "bar operations", "bartending",
                    "cocktail program", "beverage operations", "bar supervision"
                ],
                "education_preferences": [
                    "hospitality management", "beverage management", "culinary arts",
                    "business administration", "bar management"
                ],
                "certifications": [
                    "responsible beverage service", "sommelier", "bartending certification",
                    "bar management", "mixology"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Bar Supervisor": {
                "must_have_skills": [
                    "bar operations", "team supervision", "beverage knowledge", "customer service",
                    "inventory awareness", "staff training", "communication", "quality control",
                    "multitasking", "problem solving", "cash handling", "efficiency"
                ],
                "nice_to_have_skills": [
                    "mixology", "wine knowledge", "craft cocktails", "POS systems",
                    "cost awareness", "event support", "training abilities", "scheduling"
                ],
                "technical_skills": [
                    "POS systems", "bar equipment", "inventory systems", "communication tools",
                    "scheduling systems", "beverage equipment", "cash handling"
                ],
                "soft_skills": [
                    "leadership", "communication", "customer focus", "problem solving",
                    "multitasking", "team building", "adaptability", "reliability"
                ],
                "cultural_fit_keywords": [
                    "team leader", "knowledgeable", "professional", "guest-focused",
                    "supportive", "efficient", "reliable", "quality-oriented"
                ],
                "disqualifying_factors": [
                    "poor leadership skills", "lack of beverage knowledge", "poor customer service",
                    "inability to multitask", "poor communication"
                ],
                "experience_indicators": [
                    "bar supervisor", "lead bartender", "bar operations", "beverage service",
                    "bar management", "bartending", "cocktail service"
                ],
                "education_preferences": [
                    "hospitality management", "beverage studies", "bartending training",
                    "customer service", "bar operations"
                ],
                "certifications": [
                    "responsible beverage service", "bartending certification", "mixology",
                    "customer service", "bar operations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Bartender": {
                "must_have_skills": [
                    "mixology", "customer service", "cash handling", "multitasking", "product knowledge",
                    "beverage preparation", "communication", "attention to detail", "efficiency",
                    "team work", "hospitality", "problem solving"
                ],
                "nice_to_have_skills": [
                    "craft cocktails", "wine knowledge", "beer knowledge", "inventory", "POS systems",
                    "flair bartending", "event service", "multilingual", "upselling"
                ],
                "technical_skills": [
                    "bar equipment", "POS systems", "beverage systems", "cash register",
                    "cocktail tools", "beer systems", "wine service tools"
                ],
                "soft_skills": [
                    "personable", "energetic", "professional", "friendly", "entertaining",
                    "adaptability", "creativity", "patience", "multitasking"
                ],
                "cultural_fit_keywords": [
                    "personable", "energetic", "professional", "friendly", "entertaining",
                    "guest-focused", "creative", "reliable", "knowledgeable"
                ],
                "disqualifying_factors": [
                    "poor customer service", "lack of beverage knowledge", "inability to multitask",
                    "poor cash handling", "unprofessional", "slow service"
                ],
                "experience_indicators": [
                    "bartender", "mixologist", "bar service", "beverage service", "cocktail service",
                    "bar operations", "customer service", "hospitality"
                ],
                "education_preferences": [
                    "bartending school", "hospitality training", "beverage studies",
                    "customer service", "mixology training"
                ],
                "certifications": [
                    "responsible beverage service", "bartending certification", "mixology",
                    "customer service", "food safety"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 20000, "max": 40000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Cocktail Waitress": {
                "must_have_skills": [
                    "customer service", "beverage service", "communication", "multitasking",
                    "cash handling", "attention to detail", "hospitality", "professional appearance",
                    "team work", "efficiency", "problem solving", "time management"
                ],
                "nice_to_have_skills": [
                    "beverage knowledge", "upselling", "multilingual", "event service",
                    "POS systems", "cocktail knowledge", "wine basics", "guest relations"
                ],
                "technical_skills": [
                    "POS systems", "beverage service equipment", "cash handling",
                    "communication devices", "service trays", "payment processing"
                ],
                "soft_skills": [
                    "friendliness", "professionalism", "adaptability", "energy",
                    "customer focus", "communication", "positive attitude", "reliability"
                ],
                "cultural_fit_keywords": [
                    "friendly", "energetic", "professional", "attentive",
                    "guest-focused", "reliable", "personable", "service-oriented"
                ],
                "disqualifying_factors": [
                    "poor customer service", "unprofessional appearance", "inability to multitask",
                    "poor communication", "unreliable", "lack of energy"
                ],
                "experience_indicators": [
                    "cocktail waitress", "beverage server", "bar server", "lounge server",
                    "drink service", "hospitality service", "customer service"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "customer service",
                    "beverage service", "restaurant experience"
                ],
                "certifications": [
                    "responsible beverage service", "customer service", "hospitality basics",
                    "food safety", "beverage service"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 18000, "max": 30000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            "Mini-Bar Attendant": {
                "must_have_skills": [
                    "inventory management", "attention to detail", "organization", "reliability",
                    "customer service", "time management", "efficiency", "record keeping",
                    "product knowledge", "communication", "professional appearance", "discretion"
                ],
                "nice_to_have_skills": [
                    "guest interaction", "beverage knowledge", "upselling", "multilingual",
                    "computer skills", "problem solving", "cultural awareness", "hotel operations"
                ],
                "technical_skills": [
                    "inventory systems", "POS systems", "computer applications",
                    "communication devices", "record keeping systems", "mobile devices"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "organization", "efficiency",
                    "discretion", "professionalism", "customer awareness", "independence"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "organized", "efficient",
                    "discreet", "professional", "thorough", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "unreliable", "lack of organization",
                    "poor customer service", "dishonest"
                ],
                "experience_indicators": [
                    "mini-bar attendant", "inventory management", "guest services", "hotel operations",
                    "customer service", "hospitality", "retail experience"
                ],
                "education_preferences": [
                    "high school diploma", "hospitality training", "customer service",
                    "inventory management", "retail experience"
                ],
                "certifications": [
                    "hospitality service", "customer service", "inventory management",
                    "beverage service", "guest relations"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.35, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Low",
                "training_requirements": "Basic"
            },

            "Sommelier": {
                "must_have_skills": [
                    "wine knowledge", "wine service", "customer education", "communication", "attention to detail",
                    "tasting skills", "pairing knowledge", "professional presentation", "hospitality",
                    "cultural knowledge", "sales skills", "problem solving"
                ],
                "nice_to_have_skills": [
                    "wine certification", "multiple languages", "spirits knowledge", "sake knowledge",
                    "cheese pairing", "food knowledge", "wine storage", "cellar management"
                ],
                "technical_skills": [
                    "wine service tools", "cellar management", "inventory systems",
                    "tasting equipment", "storage systems", "POS systems"
                ],
                "soft_skills": [
                    "sophistication", "communication", "knowledge sharing", "cultural sensitivity",
                    "patience", "professionalism", "passion", "continuous learning"
                ],
                "cultural_fit_keywords": [
                    "knowledgeable", "sophisticated", "passionate", "educational",
                    "refined", "professional", "cultured", "expert"
                ],
                "disqualifying_factors": [
                    "lack of wine knowledge", "poor communication", "lack of sophistication",
                    "poor customer service", "inability to educate"
                ],
                "experience_indicators": [
                    "sommelier", "wine service", "wine education", "fine dining", "wine sales",
                    "beverage management", "wine consulting", "cellar management"
                ],
                "education_preferences": [
                    "wine studies", "sommelier certification", "hospitality management",
                    "culinary arts", "beverage management"
                ],
                "certifications": [
                    "sommelier certification", "wine certification", "beverage education",
                    "court of master sommeliers", "wine and spirit education"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 45000, "max": 80000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # CULINARY DEPARTMENT
            # ==========================================
            "Executive Chef": {
                "must_have_skills": [
                    "culinary leadership", "menu development", "kitchen management", "food safety",
                    "cost control", "staff management", "quality control", "food procurement",
                    "culinary innovation", "performance management", "budget management", "training"
                ],
                "nice_to_have_skills": [
                    "international cuisine", "dietary restrictions", "sustainable practices", "wine pairing",
                    "banquet cooking", "specialty diets", "culinary trends", "vendor relations"
                ],
                "technical_skills": [
                    "kitchen equipment", "food safety systems", "inventory management", "cost analysis",
                    "recipe development", "nutrition analysis", "kitchen technology", "scheduling"
                ],
                "soft_skills": [
                    "leadership", "creativity", "innovation", "communication", "stress management",
                    "team building", "problem solving", "attention to detail", "time management"
                ],
                "cultural_fit_keywords": [
                    "innovative", "leader", "creative", "quality-driven", "perfectionist",
                    "passionate", "professional", "mentor", "culinary excellence"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of culinary training", "poor food safety knowledge",
                    "inability to handle stress", "poor cost management"
                ],
                "experience_indicators": [
                    "executive chef", "head chef", "culinary director", "kitchen management",
                    "chef de cuisine", "culinary leadership", "restaurant chef"
                ],
                "education_preferences": [
                    "culinary arts", "culinary management", "hospitality management",
                    "food service management", "nutrition", "business administration"
                ],
                "certifications": [
                    "culinary certification", "food safety", "ServSafe", "nutritional analysis",
                    "culinary management", "kitchen management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 8,
                "preferred_experience_years": 12,
                "salary_range": {"min": 80000, "max": 120000},
                "growth_potential": "Executive",
                "training_requirements": "Advanced"
            },

            "Sous Chef": {
                "must_have_skills": [
                    "culinary skills", "kitchen operations", "food safety", "staff supervision",
                    "menu execution", "quality control", "cost awareness", "training abilities",
                    "communication", "time management", "problem solving", "leadership"
                ],
                "nice_to_have_skills": [
                    "menu development", "international cuisine", "dietary restrictions", "banquet cooking",
                    "inventory management", "cost control", "scheduling", "vendor knowledge"
                ],
                "technical_skills": [
                    "kitchen equipment", "food safety systems", "recipe execution", "inventory systems",
                    "kitchen technology", "cost analysis", "scheduling software"
                ],
                "soft_skills": [
                    "leadership", "creativity", "communication", "stress management", "adaptability",
                    "team work", "attention to detail", "problem solving", "multitasking"
                ],
                "cultural_fit_keywords": [
                    "creative", "leader", "quality-focused", "passionate", "professional",
                    "supportive", "dedicated", "culinary excellence", "team player"
                ],
                "disqualifying_factors": [
                    "poor culinary skills", "lack of leadership", "poor food safety",
                    "inability to handle stress", "poor communication"
                ],
                "experience_indicators": [
                    "sous chef", "kitchen supervisor", "line cook supervisor", "culinary supervisor",
                    "chef de partie", "kitchen management", "culinary operations"
                ],
                "education_preferences": [
                    "culinary arts", "culinary management", "food service", "hospitality management",
                    "nutrition", "culinary training"
                ],
                "certifications": [
                    "culinary certification", "food safety", "ServSafe", "kitchen management",
                    "culinary arts", "nutritional awareness"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 4,
                "preferred_experience_years": 7,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Chef de Partie": {
                "must_have_skills": [
                    "culinary skills", "station management", "food safety", "quality control",
                    "recipe execution", "kitchen operations", "time management", "teamwork",
                    "communication", "attention to detail", "consistency", "efficiency"
                ],
                "nice_to_have_skills": [
                    "specialty cooking", "international cuisine", "garde manger", "pastry basics",
                    "sauce making", "grilling", "knife skills", "presentation skills"
                ],
                "technical_skills": [
                    "kitchen equipment", "cooking techniques", "food safety protocols",
                    "knife skills", "cooking methods", "recipe following", "presentation"
                ],
                "soft_skills": [
                    "attention to detail", "consistency", "teamwork", "communication",
                    "stress management", "adaptability", "learning ability", "precision"
                ],
                "cultural_fit_keywords": [
                    "precise", "consistent", "dedicated", "team player", "quality-focused",
                    "passionate", "professional", "detail-oriented", "reliable"
                ],
                "disqualifying_factors": [
                    "poor culinary skills", "lack of consistency", "poor food safety",
                    "inability to work in team", "poor attention to detail"
                ],
                "experience_indicators": [
                    "chef de partie", "line cook", "station chef", "kitchen cook", "culinary specialist",
                    "cook", "kitchen operations", "food preparation"
                ],
                "education_preferences": [
                    "culinary arts", "culinary training", "food service", "cooking school",
                    "hospitality training", "culinary certificate"
                ],
                "certifications": [
                    "culinary certification", "food safety", "ServSafe", "cooking certification",
                    "culinary arts", "kitchen operations"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Line Cook": {
                "must_have_skills": [
                    "cooking skills", "food safety", "recipe following", "kitchen operations",
                    "time management", "teamwork", "communication", "attention to detail",
                    "consistency", "efficiency", "basic knife skills", "quality awareness"
                ],
                "nice_to_have_skills": [
                    "specialty cooking", "grill skills", "sauté skills", "food presentation",
                    "multitasking", "speed", "adaptability", "kitchen equipment knowledge"
                ],
                "technical_skills": [
                    "cooking equipment", "basic kitchen tools", "food safety protocols",
                    "cooking techniques", "recipe execution", "food handling", "cleaning"
                ],
                "soft_skills": [
                    "teamwork", "communication", "reliability", "attention to detail",
                    "stress management", "adaptability", "learning willingness", "consistency"
                ],
                "cultural_fit_keywords": [
                    "reliable", "team player", "dedicated", "consistent", "hardworking",
                    "passionate", "learning-oriented", "quality-focused", "efficient"
                ],
                "disqualifying_factors": [
                    "poor cooking skills", "food safety violations", "inability to follow recipes",
                    "poor teamwork", "inconsistent performance"
                ],
                "experience_indicators": [
                    "line cook", "prep cook", "kitchen cook", "cook", "food preparation",
                    "kitchen operations", "restaurant cook", "culinary assistant"
                ],
                "education_preferences": [
                    "culinary training", "food service training", "cooking school",
                    "high school diploma", "culinary certificate", "kitchen experience"
                ],
                "certifications": [
                    "food safety", "ServSafe", "cooking certification", "culinary basics",
                    "kitchen operations", "food handling"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 40000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Prep Cook": {
                "must_have_skills": [
                    "food preparation", "knife skills", "food safety", "organization",
                    "efficiency", "recipe following", "attention to detail", "teamwork",
                    "time management", "consistency", "basic cooking", "cleanliness"
                ],
                "nice_to_have_skills": [
                    "vegetable preparation", "protein preparation", "sauce preparation", "inventory awareness",
                    "kitchen operations", "food storage", "portion control", "speed"
                ],
                "technical_skills": [
                    "knife skills", "food preparation equipment", "food safety protocols",
                    "storage systems", "preparation techniques", "measuring tools", "cleaning"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "teamwork", "consistency",
                    "time management", "organization", "learning willingness", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "efficient", "team player", "organized",
                    "consistent", "hardworking", "dedicated", "quality-focused"
                ],
                "disqualifying_factors": [
                    "poor knife skills", "food safety violations", "lack of organization",
                    "inconsistent work", "poor attention to detail"
                ],
                "experience_indicators": [
                    "prep cook", "food preparation", "kitchen prep", "culinary prep",
                    "food prep", "kitchen assistant", "prep assistant", "kitchen helper"
                ],
                "education_preferences": [
                    "culinary training", "food service training", "high school diploma",
                    "cooking basics", "food safety training", "kitchen experience"
                ],
                "certifications": [
                    "food safety", "ServSafe", "food handling", "knife skills",
                    "culinary basics", "kitchen safety"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.40, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 22000, "max": 32000},
                "growth_potential": "High",
                "training_requirements": "Basic"
            },

            "Pastry Chef": {
                "must_have_skills": [
                    "pastry skills", "baking techniques", "dessert creation", "food safety",
                    "recipe development", "attention to detail", "creativity", "presentation skills",
                    "time management", "quality control", "consistency", "precision"
                ],
                "nice_to_have_skills": [
                    "chocolate work", "sugar art", "cake decorating", "bread making", "gluten-free baking",
                    "international pastries", "special dietary desserts", "cost control", "menu development"
                ],
                "technical_skills": [
                    "pastry equipment", "baking ovens", "mixing equipment", "decorating tools",
                    "temperature control", "recipe scaling", "nutritional analysis", "food safety"
                ],
                "soft_skills": [
                    "creativity", "precision", "attention to detail", "artistic ability",
                    "patience", "innovation", "communication", "time management", "consistency"
                ],
                "cultural_fit_keywords": [
                    "creative", "artistic", "precise", "innovative", "passionate",
                    "detail-oriented", "quality-focused", "perfectionist", "dedicated"
                ],
                "disqualifying_factors": [
                    "poor pastry skills", "lack of creativity", "inconsistent results",
                    "poor attention to detail", "food safety violations"
                ],
                "experience_indicators": [
                    "pastry chef", "baker", "dessert chef", "pastry cook", "baking specialist",
                    "confectioner", "cake decorator", "pastry artist"
                ],
                "education_preferences": [
                    "pastry arts", "baking and pastry", "culinary arts", "confectionery arts",
                    "pastry certification", "culinary management"
                ],
                "certifications": [
                    "pastry certification", "baking certification", "food safety", "ServSafe",
                    "pastry arts", "confectionery certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 40000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Baker": {
                "must_have_skills": [
                    "baking skills", "bread making", "recipe following", "food safety",
                    "consistency", "time management", "attention to detail", "quality control",
                    "oven operation", "dough handling", "proofing", "temperature control"
                ],
                "nice_to_have_skills": [
                    "artisan breads", "pastry basics", "cake baking", "specialty breads",
                    "gluten-free baking", "cost awareness", "inventory", "equipment maintenance"
                ],
                "technical_skills": [
                    "baking ovens", "mixing equipment", "proofing equipment", "measuring tools",
                    "temperature monitoring", "timer management", "baking techniques", "food safety"
                ],
                "soft_skills": [
                    "consistency", "attention to detail", "patience", "reliability",
                    "time management", "precision", "quality focus", "learning ability"
                ],
                "cultural_fit_keywords": [
                    "consistent", "reliable", "detail-oriented", "quality-focused", "patient",
                    "dedicated", "precise", "hardworking", "traditional"
                ],
                "disqualifying_factors": [
                    "poor baking skills", "inconsistent results", "food safety violations",
                    "poor time management", "lack of attention to detail"
                ],
                "experience_indicators": [
                    "baker", "bread baker", "production baker", "baking specialist",
                    "bakery worker", "baking assistant", "bread production"
                ],
                "education_preferences": [
                    "baking certification", "culinary training", "pastry arts", "food service",
                    "baking school", "culinary arts"
                ],
                "certifications": [
                    "baking certification", "food safety", "ServSafe", "bread baking",
                    "pastry basics", "culinary fundamentals"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 25000, "max": 40000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Kitchen Steward": {
                "must_have_skills": [
                    "cleaning", "dishwashing", "kitchen sanitation", "equipment cleaning",
                    "food safety", "organization", "efficiency", "teamwork",
                    "reliability", "physical stamina", "attention to detail", "time management"
                ],
                "nice_to_have_skills": [
                    "equipment maintenance", "inventory support", "kitchen operations awareness",
                    "chemical safety", "waste management", "recycling", "deep cleaning"
                ],
                "technical_skills": [
                    "dishwashing equipment", "cleaning chemicals", "sanitizing systems",
                    "kitchen equipment", "cleaning tools", "safety protocols", "waste systems"
                ],
                "soft_skills": [
                    "reliability", "hardworking", "teamwork", "attention to detail",
                    "physical endurance", "organization", "efficiency", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "reliable", "hardworking", "team player", "efficient", "organized",
                    "dedicated", "thorough", "supportive", "dependable"
                ],
                "disqualifying_factors": [
                    "poor work ethic", "unreliable", "food safety violations",
                    "inability to handle physical demands", "poor teamwork"
                ],
                "experience_indicators": [
                    "kitchen steward", "dishwasher", "kitchen cleaner", "sanitation worker",
                    "kitchen assistant", "utility worker", "kitchen support"
                ],
                "education_preferences": [
                    "high school diploma", "food safety training", "kitchen experience",
                    "sanitation training", "workplace safety"
                ],
                "certifications": [
                    "food safety", "sanitation certification", "workplace safety",
                    "chemical safety", "kitchen operations"
                ],
                "scoring_weights": {
                    "experience": 0.15, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.20
                },
                "min_experience_years": 0,
                "preferred_experience_years": 1,
                "salary_range": {"min": 20000, "max": 28000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            # ==========================================
            # ENGINEERING & MAINTENANCE DEPARTMENT
            # ==========================================
            "Chief Engineer": {
                "must_have_skills": [
                    "facility management", "engineering leadership", "maintenance management", "HVAC systems",
                    "electrical systems", "plumbing systems", "team management", "safety protocols",
                    "budget management", "preventive maintenance", "emergency response", "project management"
                ],
                "nice_to_have_skills": [
                    "energy management", "sustainability", "building automation", "fire safety systems",
                    "pool maintenance", "elevator systems", "vendor management", "cost control"
                ],
                "technical_skills": [
                    "HVAC systems", "electrical systems", "plumbing", "building automation", "maintenance software",
                    "energy management", "safety systems", "mechanical systems", "preventive maintenance"
                ],
                "soft_skills": [
                    "leadership", "problem solving", "analytical thinking", "communication",
                    "project management", "team building", "decision making", "stress management"
                ],
                "cultural_fit_keywords": [
                    "leader", "technical", "reliable", "problem-solver", "safety-focused",
                    "efficient", "analytical", "experienced", "responsible"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of technical knowledge", "safety violations",
                    "poor communication", "inability to manage emergencies"
                ],
                "experience_indicators": [
                    "chief engineer", "facility manager", "maintenance manager", "engineering manager",
                    "building engineer", "property engineer", "technical manager"
                ],
                "education_preferences": [
                    "engineering", "facility management", "mechanical engineering", "electrical engineering",
                    "building systems", "HVAC certification", "property management"
                ],
                "certifications": [
                    "engineering license", "HVAC certification", "electrical certification",
                    "facility management", "safety certification", "energy management"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 8,
                "preferred_experience_years": 12,
                "salary_range": {"min": 75000, "max": 110000},
                "growth_potential": "Moderate",
                "training_requirements": "Advanced"
            },

            "Maintenance Manager": {
                "must_have_skills": [
                    "maintenance management", "team supervision", "preventive maintenance", "HVAC basics",
                    "electrical basics", "plumbing basics", "safety protocols", "scheduling",
                    "work order management", "vendor coordination", "budget awareness", "emergency response"
                ],
                "nice_to_have_skills": [
                    "building systems", "energy efficiency", "project management", "cost control",
                    "maintenance software", "contractor management", "inventory management", "training"
                ],
                "technical_skills": [
                    "maintenance systems", "HVAC basics", "electrical basics", "plumbing", "hand tools",
                    "power tools", "maintenance software", "safety equipment", "diagnostic tools"
                ],
                "soft_skills": [
                    "leadership", "organization", "communication", "problem solving",
                    "time management", "team building", "reliability", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "organized", "leader", "reliable", "efficient", "problem-solver",
                    "safety-conscious", "team player", "responsible", "experienced"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of maintenance experience", "safety violations",
                    "poor organization", "inability to handle emergencies"
                ],
                "experience_indicators": [
                    "maintenance manager", "maintenance supervisor", "facility maintenance",
                    "building maintenance", "property maintenance", "technical supervisor"
                ],
                "education_preferences": [
                    "facility management", "maintenance management", "technical training",
                    "building systems", "mechanical training", "electrical training"
                ],
                "certifications": [
                    "facility management", "maintenance certification", "HVAC basics",
                    "electrical basics", "safety certification", "leadership training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Maintenance Technician": {
                "must_have_skills": [
                    "general maintenance", "basic electrical", "basic plumbing", "HVAC basics",
                    "hand tools", "power tools", "safety protocols", "troubleshooting",
                    "repair skills", "preventive maintenance", "communication", "reliability"
                ],
                "nice_to_have_skills": [
                    "appliance repair", "carpentry", "painting", "tile work", "equipment maintenance",
                    "pool maintenance", "landscaping", "welding", "locksmith skills"
                ],
                "technical_skills": [
                    "hand tools", "power tools", "electrical tools", "plumbing tools", "HVAC tools",
                    "diagnostic equipment", "maintenance equipment", "safety equipment", "measuring tools"
                ],
                "soft_skills": [
                    "problem solving", "reliability", "attention to detail", "communication",
                    "learning ability", "adaptability", "teamwork", "initiative", "patience"
                ],
                "cultural_fit_keywords": [
                    "reliable", "handy", "problem-solver", "detail-oriented", "hardworking",
                    "versatile", "safety-conscious", "team player", "dependable"
                ],
                "disqualifying_factors": [
                    "lack of technical skills", "safety violations", "unreliable",
                    "poor communication", "inability to learn"
                ],
                "experience_indicators": [
                    "maintenance technician", "maintenance worker", "handyman", "building maintenance",
                    "facility maintenance", "property maintenance", "general maintenance"
                ],
                "education_preferences": [
                    "technical training", "trade school", "maintenance certification",
                    "electrical training", "plumbing training", "HVAC training"
                ],
                "certifications": [
                    "maintenance certification", "electrical basics", "plumbing basics",
                    "HVAC basics", "safety certification", "tool certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "HVAC Technician": {
                "must_have_skills": [
                    "HVAC systems", "heating systems", "cooling systems", "ventilation systems",
                    "HVAC repair", "HVAC maintenance", "electrical basics", "troubleshooting",
                    "safety protocols", "system diagnostics", "preventive maintenance", "refrigeration"
                ],
                "nice_to_have_skills": [
                    "building automation", "energy efficiency", "commercial HVAC", "industrial HVAC",
                    "controls systems", "ductwork", "heat pumps", "boilers", "chillers"
                ],
                "technical_skills": [
                    "HVAC tools", "diagnostic equipment", "electrical meters", "refrigeration tools",
                    "pressure gauges", "leak detectors", "calibration tools", "safety equipment"
                ],
                "soft_skills": [
                    "problem solving", "attention to detail", "analytical thinking", "reliability",
                    "learning ability", "communication", "patience", "precision", "safety awareness"
                ],
                "cultural_fit_keywords": [
                    "technical", "precise", "reliable", "problem-solver", "detail-oriented",
                    "safety-conscious", "experienced", "knowledgeable", "professional"
                ],
                "disqualifying_factors": [
                    "lack of HVAC knowledge", "safety violations", "poor troubleshooting",
                    "unreliable", "inability to learn new systems"
                ],
                "experience_indicators": [
                    "HVAC technician", "heating technician", "cooling technician", "air conditioning technician",
                    "refrigeration technician", "HVAC service", "climate control"
                ],
                "education_preferences": [
                    "HVAC certification", "technical training", "trade school", "mechanical training",
                    "refrigeration training", "electrical training", "building systems"
                ],
                "certifications": [
                    "HVAC certification", "EPA certification", "refrigeration license",
                    "electrical certification", "safety certification", "manufacturer certifications"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Electrician": {
                "must_have_skills": [
                    "electrical systems", "electrical repair", "electrical installation", "wiring",
                    "electrical troubleshooting", "electrical safety", "electrical codes", "power systems",
                    "lighting systems", "electrical maintenance", "circuit analysis", "motor control"
                ],
                "nice_to_have_skills": [
                    "industrial electrical", "commercial electrical", "building automation", "fire alarm systems",
                    "security systems", "emergency power", "generators", "transformers", "control panels"
                ],
                "technical_skills": [
                    "electrical tools", "multimeters", "wire strippers", "conduit benders",
                    "voltage testers", "oscilloscopes", "power tools", "safety equipment", "diagnostic tools"
                ],
                "soft_skills": [
                    "problem solving", "attention to detail", "safety awareness", "precision",
                    "analytical thinking", "reliability", "communication", "learning ability", "patience"
                ],
                "cultural_fit_keywords": [
                    "technical", "precise", "safety-conscious", "reliable", "problem-solver",
                    "detail-oriented", "experienced", "professional", "knowledgeable"
                ],
                "disqualifying_factors": [
                    "lack of electrical knowledge", "safety violations", "poor troubleshooting",
                    "code violations", "unreliable work"
                ],
                "experience_indicators": [
                    "electrician", "electrical technician", "electrical maintenance", "electrical service",
                    "electrical repair", "electrical installation", "power systems"
                ],
                "education_preferences": [
                    "electrical training", "trade school", "electrical certification", "technical training",
                    "electrical apprenticeship", "electrical engineering technology"
                ],
                "certifications": [
                    "electrical license", "electrical certification", "safety certification",
                    "electrical codes", "OSHA certification", "manufacturer certifications"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 45000, "max": 65000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Plumber": {
                "must_have_skills": [
                    "plumbing systems", "pipe installation", "pipe repair", "water systems",
                    "drainage systems", "plumbing troubleshooting", "plumbing tools", "safety protocols",
                    "leak detection", "pipe fitting", "water pressure", "plumbing codes"
                ],
                "nice_to_have_skills": [
                    "commercial plumbing", "industrial plumbing", "backflow prevention", "water heaters",
                    "sewage systems", "pump systems", "gas lines", "hydro-jetting", "pipe cleaning"
                ],
                "technical_skills": [
                    "plumbing tools", "pipe cutters", "pipe threaders", "drain snakes",
                    "pressure testers", "leak detectors", "welding equipment", "power tools", "hand tools"
                ],
                "soft_skills": [
                    "problem solving", "attention to detail", "physical stamina", "reliability",
                    "communication", "patience", "precision", "learning ability", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "reliable", "skilled", "problem-solver", "detail-oriented", "hardworking",
                    "experienced", "professional", "dependable", "thorough"
                ],
                "disqualifying_factors": [
                    "lack of plumbing knowledge", "poor workmanship", "safety violations",
                    "unreliable", "inability to diagnose problems"
                ],
                "experience_indicators": [
                    "plumber", "plumbing technician", "pipefitter", "plumbing maintenance",
                    "plumbing service", "plumbing repair", "water systems"
                ],
                "education_preferences": [
                    "plumbing training", "trade school", "plumbing certification", "technical training",
                    "plumbing apprenticeship", "pipefitting training"
                ],
                "certifications": [
                    "plumbing license", "plumbing certification", "backflow certification",
                    "safety certification", "welding certification", "gas line certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 40000, "max": 60000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Pool Technician": {
                "must_have_skills": [
                    "pool maintenance", "water chemistry", "pool equipment", "pool cleaning",
                    "chemical handling", "pool safety", "equipment maintenance", "troubleshooting",
                    "pump systems", "filtration systems", "water testing", "preventive maintenance"
                ],
                "nice_to_have_skills": [
                    "spa maintenance", "automated systems", "pool repairs", "tile cleaning",
                    "equipment installation", "water features", "deck maintenance", "customer service"
                ],
                "technical_skills": [
                    "pool equipment", "water testing kits", "chemical dispensers", "cleaning equipment",
                    "pump systems", "filter systems", "heater systems", "automation systems", "hand tools"
                ],
                "soft_skills": [
                    "attention to detail", "reliability", "safety awareness", "communication",
                    "problem solving", "physical stamina", "organization", "customer focus", "patience"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "safety-conscious", "thorough", "responsible",
                    "guest-focused", "professional", "knowledgeable", "dependable"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "safety violations", "unreliable",
                    "lack of chemical knowledge", "poor guest interaction"
                ],
                "experience_indicators": [
                    "pool technician", "pool maintenance", "pool service", "aquatic maintenance",
                    "water treatment", "pool operator", "pool cleaner"
                ],
                "education_preferences": [
                    "pool operator certification", "water treatment training", "chemical safety training",
                    "aquatic facility training", "equipment training"
                ],
                "certifications": [
                    "pool operator license", "chemical safety", "water treatment certification",
                    "aquatic facility certification", "safety certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Groundskeeper": {
                "must_have_skills": [
                    "landscaping", "lawn maintenance", "plant care", "irrigation systems",
                    "equipment operation", "grounds maintenance", "pest control", "seasonal care",
                    "outdoor equipment", "safety protocols", "physical stamina", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "horticulture", "tree care", "fertilization", "disease control", "design basics",
                    "equipment maintenance", "irrigation repair", "pesticide application", "customer service"
                ],
                "technical_skills": [
                    "lawn equipment", "irrigation systems", "hand tools", "power tools",
                    "pesticide equipment", "fertilizer spreaders", "mowers", "trimmers", "blowers"
                ],
                "soft_skills": [
                    "attention to detail", "physical stamina", "reliability", "independence",
                    "time management", "pride in work", "adaptability", "safety awareness", "patience"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "reliable", "hardworking", "independent", "thorough",
                    "nature-loving", "dedicated", "physical", "responsible"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of physical stamina", "unreliable",
                    "safety violations", "inability to work outdoors"
                ],
                "experience_indicators": [
                    "groundskeeper", "landscaper", "lawn care", "grounds maintenance",
                    "landscape maintenance", "turf care", "outdoor maintenance"
                ],
                "education_preferences": [
                    "landscaping training", "horticulture training", "turf management",
                    "pesticide certification", "equipment operation training"
                ],
                "certifications": [
                    "pesticide license", "landscaping certification", "equipment certification",
                    "safety certification", "horticulture certification"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 38000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            # ==========================================
            # SECURITY DEPARTMENT
            # ==========================================
            "Security Manager": {
                "must_have_skills": [
                    "security management", "team leadership", "emergency response", "safety protocols",
                    "security systems", "incident management", "risk assessment", "communication",
                    "training development", "report writing", "conflict resolution", "law enforcement"
                ],
                "nice_to_have_skills": [
                    "surveillance systems", "access control", "investigation skills", "crowd control",
                    "emergency medical", "fire safety", "loss prevention", "vendor management"
                ],
                "technical_skills": [
                    "security systems", "surveillance equipment", "access control systems", "alarm systems",
                    "communication equipment", "computer systems", "incident reporting", "emergency equipment"
                ],
                "soft_skills": [
                    "leadership", "communication", "problem solving", "decision making",
                    "stress management", "attention to detail", "reliability", "integrity"
                ],
                "cultural_fit_keywords": [
                    "leader", "reliable", "responsible", "professional", "trustworthy",
                    "vigilant", "calm", "experienced", "safety-focused"
                ],
                "disqualifying_factors": [
                    "criminal background", "poor leadership", "lack of security experience",
                    "poor communication", "inability to handle emergencies"
                ],
                "experience_indicators": [
                    "security manager", "security supervisor", "loss prevention manager",
                    "safety manager", "security operations", "law enforcement", "military"
                ],
                "education_preferences": [
                    "criminal justice", "security management", "law enforcement", "military",
                    "emergency management", "business administration", "public safety"
                ],
                "certifications": [
                    "security certification", "CPR/First Aid", "emergency response",
                    "security management", "loss prevention", "safety certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 50000, "max": 75000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Security Supervisor": {
                "must_have_skills": [
                    "security operations", "team supervision", "emergency response", "safety protocols",
                    "incident response", "communication", "report writing", "patrol duties",
                    "conflict resolution", "training abilities", "attention to detail", "reliability"
                ],
                "nice_to_have_skills": [
                    "surveillance monitoring", "access control", "crowd control", "investigation basics",
                    "emergency medical", "loss prevention", "customer service", "technology use"
                ],
                "technical_skills": [
                    "security equipment", "surveillance systems", "communication devices", "alarm systems",
                    "access control", "computer basics", "reporting software", "emergency equipment"
                ],
                "soft_skills": [
                    "leadership", "communication", "problem solving", "reliability",
                    "attention to detail", "stress management", "integrity", "alertness"
                ],
                "cultural_fit_keywords": [
                    "reliable", "professional", "vigilant", "responsible", "trustworthy",
                    "calm", "leader", "safety-focused", "experienced"
                ],
                "disqualifying_factors": [
                    "criminal background", "poor reliability", "lack of security experience",
                    "poor communication", "inability to handle stress"
                ],
                "experience_indicators": [
                    "security supervisor", "security officer", "loss prevention", "safety officer",
                    "security guard", "law enforcement", "military", "corrections"
                ],
                "education_preferences": [
                    "criminal justice", "security training", "law enforcement", "military",
                    "security certification", "public safety", "emergency response"
                ],
                "certifications": [
                    "security license", "CPR/First Aid", "security training", "emergency response",
                    "loss prevention", "safety certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Security Officer": {
                "must_have_skills": [
                    "security patrol", "observation skills", "communication", "emergency response",
                    "report writing", "safety awareness", "conflict de-escalation", "customer service",
                    "attention to detail", "reliability", "integrity", "physical fitness"
                ],
                "nice_to_have_skills": [
                    "surveillance monitoring", "access control", "crowd control", "basic investigation",
                    "emergency medical", "technology use", "multilingual", "guest relations"
                ],
                "technical_skills": [
                    "security equipment", "communication devices", "surveillance systems", "alarm systems",
                    "access control", "report writing", "computer basics", "mobile devices"
                ],
                "soft_skills": [
                    "alertness", "communication", "reliability", "integrity", "patience",
                    "stress management", "customer focus", "problem solving", "physical stamina"
                ],
                "cultural_fit_keywords": [
                    "reliable", "professional", "vigilant", "trustworthy", "alert",
                    "responsible", "calm", "guest-focused", "honest"
                ],
                "disqualifying_factors": [
                    "criminal background", "poor reliability", "lack of integrity",
                    "poor communication", "inability to stay alert"
                ],
                "experience_indicators": [
                    "security officer", "security guard", "loss prevention", "safety officer",
                    "patrol officer", "law enforcement", "military", "customer service"
                ],
                "education_preferences": [
                    "high school diploma", "security training", "criminal justice", "law enforcement",
                    "military", "security certification", "customer service"
                ],
                "certifications": [
                    "security license", "CPR/First Aid", "security training", "customer service",
                    "emergency response", "conflict resolution"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.30, "cultural_fit": 0.30, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 38000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # SALES & MARKETING DEPARTMENT
            # ==========================================
            "Director of Sales": {
                "must_have_skills": [
                    "sales leadership", "revenue management", "team management", "strategic planning",
                    "market analysis", "client relations", "negotiation", "performance management",
                    "budget management", "sales forecasting", "contract negotiation", "business development"
                ],
                "nice_to_have_skills": [
                    "hospitality sales", "group sales", "corporate sales", "wedding sales", "event sales",
                    "digital marketing", "CRM systems", "lead generation", "pricing strategies"
                ],
                "technical_skills": [
                    "CRM systems", "sales analytics", "revenue management systems", "presentation software",
                    "database management", "social media", "marketing automation", "reporting tools"
                ],
                "soft_skills": [
                    "leadership", "communication", "negotiation", "strategic thinking", "relationship building",
                    "presentation skills", "analytical thinking", "results orientation", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "results-driven", "leader", "strategic", "relationship-builder", "innovative",
                    "competitive", "growth-oriented", "professional", "persuasive"
                ],
                "disqualifying_factors": [
                    "poor sales performance", "lack of leadership", "poor communication",
                    "inability to meet targets", "poor relationship skills"
                ],
                "experience_indicators": [
                    "sales director", "sales manager", "revenue manager", "business development",
                    "hospitality sales", "group sales", "corporate sales", "account management"
                ],
                "education_preferences": [
                    "business administration", "marketing", "hospitality management", "sales management",
                    "communications", "MBA", "revenue management"
                ],
                "certifications": [
                    "sales certification", "revenue management", "hospitality sales", "CRM certification",
                    "marketing certification", "business development"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.15, "hospitality": 0.15
                },
                "min_experience_years": 7,
                "preferred_experience_years": 10,
                "salary_range": {"min": 75000, "max": 120000},
                "growth_potential": "Executive",
                "training_requirements": "Advanced"
            },

            "Sales Manager": {
                "must_have_skills": [
                    "sales management", "team leadership", "client relations", "revenue generation",
                    "performance management", "market analysis", "negotiation", "communication",
                    "sales forecasting", "lead generation", "contract management", "customer service"
                ],
                "nice_to_have_skills": [
                    "hospitality sales", "group sales", "event sales", "corporate accounts",
                    "digital marketing", "CRM systems", "pricing strategies", "market research"
                ],
                "technical_skills": [
                    "CRM systems", "sales software", "presentation tools", "database management",
                    "analytics tools", "social media", "email marketing", "reporting systems"
                ],
                "soft_skills": [
                    "leadership", "communication", "negotiation", "relationship building",
                    "persuasion", "analytical thinking", "results orientation", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "results-driven", "relationship-builder", "persuasive", "professional",
                    "competitive", "goal-oriented", "customer-focused", "innovative"
                ],
                "disqualifying_factors": [
                    "poor sales performance", "lack of leadership", "poor communication",
                    "inability to build relationships", "poor customer service"
                ],
                "experience_indicators": [
                    "sales manager", "account manager", "business development", "sales representative",
                    "hospitality sales", "group sales", "revenue management", "client relations"
                ],
                "education_preferences": [
                    "business administration", "marketing", "hospitality management", "sales",
                    "communications", "customer relations", "business development"
                ],
                "certifications": [
                    "sales certification", "hospitality sales", "CRM certification", "customer service",
                    "marketing certification", "negotiation training"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 4,
                "preferred_experience_years": 7,
                "salary_range": {"min": 50000, "max": 80000},
                "growth_potential": "High",
                "training_requirements": "Extensive"
            },

            "Sales Representative": {
                "must_have_skills": [
                    "sales skills", "customer relations", "communication", "lead generation",
                    "negotiation", "product knowledge", "client presentation", "follow-up",
                    "relationship building", "goal achievement", "time management", "customer service"
                ],
                "nice_to_have_skills": [
                    "hospitality knowledge", "event sales", "group sales", "corporate sales",
                    "digital marketing", "social media", "CRM use", "market research"
                ],
                "technical_skills": [
                    "CRM systems", "presentation software", "social media platforms", "email systems",
                    "database management", "mobile apps", "communication tools", "analytics"
                ],
                "soft_skills": [
                    "communication", "persuasion", "relationship building", "persistence",
                    "enthusiasm", "adaptability", "goal orientation", "customer focus"
                ],
                "cultural_fit_keywords": [
                    "persuasive", "relationship-builder", "enthusiastic", "goal-oriented",
                    "customer-focused", "professional", "persistent", "results-driven"
                ],
                "disqualifying_factors": [
                    "poor communication", "lack of sales skills", "poor customer service",
                    "inability to build relationships", "lack of persistence"
                ],
                "experience_indicators": [
                    "sales representative", "sales associate", "account executive", "business development",
                    "customer relations", "hospitality sales", "retail sales", "customer service"
                ],
                "education_preferences": [
                    "business administration", "marketing", "communications", "hospitality",
                    "sales training", "customer service", "business studies"
                ],
                "certifications": [
                    "sales certification", "customer service", "communication skills",
                    "hospitality training", "CRM certification", "product knowledge"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 50000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # FINANCE & ACCOUNTING DEPARTMENT
            # ==========================================
            "Finance Manager": {
                "must_have_skills": [
                    "financial management", "accounting", "budgeting", "financial analysis",
                    "cost control", "financial reporting", "cash flow management", "audit preparation",
                    "team management", "compliance", "forecasting", "performance analysis"
                ],
                "nice_to_have_skills": [
                    "hospitality accounting", "revenue management", "tax preparation", "payroll management",
                    "accounts payable", "accounts receivable", "financial systems", "investment analysis"
                ],
                "technical_skills": [
                    "accounting software", "Excel advanced", "financial systems", "ERP systems",
                    "budgeting software", "reporting tools", "database management", "analytics"
                ],
                "soft_skills": [
                    "analytical thinking", "attention to detail", "leadership", "communication",
                    "problem solving", "organization", "integrity", "time management"
                ],
                "cultural_fit_keywords": [
                    "analytical", "detail-oriented", "reliable", "honest", "organized",
                    "leader", "professional", "accurate", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor financial knowledge", "lack of accuracy", "poor leadership",
                    "integrity issues", "poor communication"
                ],
                "experience_indicators": [
                    "finance manager", "accounting manager", "financial analyst", "controller",
                    "hospitality finance", "budget manager", "cost accounting", "financial reporting"
                ],
                "education_preferences": [
                    "accounting", "finance", "business administration", "economics",
                    "hospitality management", "MBA", "financial management"
                ],
                "certifications": [
                    "CPA", "CMA", "accounting certification", "financial management",
                    "hospitality finance", "QuickBooks", "Excel certification"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 60000, "max": 90000},
                "growth_potential": "High",
                "training_requirements": "Advanced"
            },

            "Accountant": {
                "must_have_skills": [
                    "accounting", "bookkeeping", "financial records", "data entry", "reconciliation",
                    "accounts payable", "accounts receivable", "payroll", "tax preparation",
                    "attention to detail", "accuracy", "financial software"
                ],
                "nice_to_have_skills": [
                    "hospitality accounting", "cost accounting", "budget assistance", "audit support",
                    "financial analysis", "reporting", "compliance", "inventory accounting"
                ],
                "technical_skills": [
                    "accounting software", "QuickBooks", "Excel", "payroll systems", "tax software",
                    "database entry", "financial systems", "reporting tools"
                ],
                "soft_skills": [
                    "attention to detail", "accuracy", "organization", "analytical thinking",
                    "reliability", "integrity", "time management", "communication"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "accurate", "reliable", "organized", "honest",
                    "analytical", "professional", "thorough", "trustworthy"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of accuracy", "integrity issues",
                    "poor organization", "inability to meet deadlines"
                ],
                "experience_indicators": [
                    "accountant", "bookkeeper", "accounting clerk", "financial clerk",
                    "accounts payable", "accounts receivable", "payroll clerk", "tax preparer"
                ],
                "education_preferences": [
                    "accounting", "finance", "business administration", "bookkeeping",
                    "accounting certification", "business studies"
                ],
                "certifications": [
                    "accounting certification", "bookkeeping certification", "QuickBooks",
                    "payroll certification", "tax preparation", "Excel certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 2,
                "preferred_experience_years": 4,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            "Accounts Payable Clerk": {
                "must_have_skills": [
                    "accounts payable", "data entry", "invoice processing", "vendor relations",
                    "payment processing", "record keeping", "attention to detail", "organization",
                    "accuracy", "communication", "time management", "accounting software"
                ],
                "nice_to_have_skills": [
                    "vendor management", "expense reporting", "reconciliation", "audit support",
                    "purchase order processing", "cost analysis", "filing systems", "customer service"
                ],
                "technical_skills": [
                    "accounting software", "QuickBooks", "Excel", "database entry", "email systems",
                    "payment systems", "scanning equipment", "filing systems"
                ],
                "soft_skills": [
                    "attention to detail", "accuracy", "organization", "reliability",
                    "communication", "time management", "patience", "problem solving"
                ],
                "cultural_fit_keywords": [
                    "detail-oriented", "accurate", "organized", "reliable", "efficient",
                    "professional", "thorough", "dependable", "systematic"
                ],
                "disqualifying_factors": [
                    "poor attention to detail", "lack of accuracy", "poor organization",
                    "unreliable", "poor communication"
                ],
                "experience_indicators": [
                    "accounts payable", "accounting clerk", "data entry", "bookkeeping",
                    "invoice processing", "vendor relations", "payment processing"
                ],
                "education_preferences": [
                    "high school diploma", "accounting", "business administration", "bookkeeping",
                    "data entry training", "office administration"
                ],
                "certifications": [
                    "accounting basics", "QuickBooks", "data entry", "office software",
                    "customer service", "bookkeeping basics"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.40, "cultural_fit": 0.25, "hospitality": 0.10
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 25000, "max": 38000},
                "growth_potential": "Moderate",
                "training_requirements": "Basic"
            },

            # ==========================================
            # HUMAN RESOURCES DEPARTMENT
            # ==========================================
            "HR Manager": {
                "must_have_skills": [
                    "HR management", "employee relations", "recruitment", "performance management",
                    "policy development", "training coordination", "compliance", "conflict resolution",
                    "team leadership", "compensation", "benefits administration", "employment law"
                ],
                "nice_to_have_skills": [
                    "hospitality HR", "labor relations", "HRIS systems", "onboarding", "succession planning",
                    "employee engagement", "diversity initiatives", "organizational development", "safety programs"
                ],
                "technical_skills": [
                    "HRIS systems", "payroll systems", "recruitment software", "performance management systems",
                    "compliance tracking", "reporting tools", "database management", "Microsoft Office"
                ],
                "soft_skills": [
                    "leadership", "communication", "empathy", "problem solving", "confidentiality",
                    "decision making", "interpersonal skills", "analytical thinking", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "leader", "supportive", "fair", "confidential", "professional",
                    "empathetic", "organized", "strategic", "people-focused"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of HR knowledge", "confidentiality breaches",
                    "poor communication", "bias or discrimination"
                ],
                "experience_indicators": [
                    "HR manager", "human resources", "personnel manager", "employee relations",
                    "recruitment manager", "training manager", "compensation manager", "hospitality HR"
                ],
                "education_preferences": [
                    "human resources", "business administration", "psychology", "organizational behavior",
                    "hospitality management", "employment law", "MBA"
                ],
                "certifications": [
                    "SHRM-CP", "SHRM-SCP", "PHR", "SPHR", "HR certification", "employment law", "HRIS"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 55000, "max": 85000},
                "growth_potential": "High",
                "training_requirements": "Advanced"
            },

            "HR Coordinator": {
                "must_have_skills": [
                    "HR support", "recruitment assistance", "employee records", "onboarding",
                    "data entry", "communication", "organization", "confidentiality",
                    "filing systems", "scheduling", "customer service", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "HRIS systems", "benefits administration", "training coordination", "employee relations",
                    "compliance tracking", "payroll support", "interview scheduling", "background checks"
                ],
                "technical_skills": [
                    "HRIS systems", "Microsoft Office", "database management", "scanning systems",
                    "email systems", "scheduling software", "recruitment tools", "filing systems"
                ],
                "soft_skills": [
                    "organization", "communication", "confidentiality", "attention to detail",
                    "customer service", "multitasking", "reliability", "interpersonal skills"
                ],
                "cultural_fit_keywords": [
                    "organized", "confidential", "supportive", "professional", "reliable",
                    "detail-oriented", "helpful", "efficient", "people-focused"
                ],
                "disqualifying_factors": [
                    "poor organization", "confidentiality breaches", "poor communication",
                    "lack of attention to detail", "poor customer service"
                ],
                "experience_indicators": [
                    "HR coordinator", "HR assistant", "personnel assistant", "recruitment coordinator",
                    "administrative assistant", "office coordinator", "employee services"
                ],
                "education_preferences": [
                    "human resources", "business administration", "office administration",
                    "communications", "psychology", "hospitality management"
                ],
                "certifications": [
                    "HR certification", "office administration", "customer service",
                    "Microsoft Office", "HRIS training", "confidentiality training"
                ],
                "scoring_weights": {
                    "experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # IT DEPARTMENT
            # ==========================================
            "IT Manager": {
                "must_have_skills": [
                    "IT management", "system administration", "network management", "team leadership",
                    "project management", "security protocols", "hardware management", "software management",
                    "troubleshooting", "vendor management", "budget management", "strategic planning"
                ],
                "nice_to_have_skills": [
                    "hospitality technology", "PMS systems", "cloud computing", "cybersecurity",
                    "database administration", "Wi-Fi management", "mobile technology", "backup systems"
                ],
                "technical_skills": [
                    "Windows Server", "networking", "Active Directory", "virtualization", "cloud platforms",
                    "cybersecurity tools", "database management", "backup systems", "monitoring tools"
                ],
                "soft_skills": [
                    "leadership", "problem solving", "communication", "analytical thinking",
                    "project management", "team building", "adaptability", "stress management"
                ],
                "cultural_fit_keywords": [
                    "technical", "leader", "innovative", "problem-solver", "reliable",
                    "analytical", "strategic", "professional", "efficient"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of technical knowledge", "poor communication",
                    "inability to handle stress", "poor project management"
                ],
                "experience_indicators": [
                    "IT manager", "system administrator", "network administrator", "IT director",
                    "technology manager", "hospitality IT", "infrastructure manager"
                ],
                "education_preferences": [
                    "computer science", "information technology", "network administration",
                    "cybersecurity", "business administration", "hospitality technology"
                ],
                "certifications": [
                    "CompTIA", "Cisco", "Microsoft", "VMware", "security certifications",
                    "project management", "ITIL", "cloud certifications"
                ],
                "scoring_weights": {
                    "experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10
                },
                "min_experience_years": 5,
                "preferred_experience_years": 8,
                "salary_range": {"min": 65000, "max": 95000},
                "growth_potential": "High",
                "training_requirements": "Advanced"
            },

            "IT Technician": {
                "must_have_skills": [
                    "computer repair", "troubleshooting", "hardware installation", "software installation",
                    "network support", "user support", "system maintenance", "documentation",
                    "customer service", "problem solving", "technical communication", "equipment setup"
                ],
                "nice_to_have_skills": [
                    "hospitality systems", "PMS support", "mobile device support", "printer support",
                    "Wi-Fi troubleshooting", "phone systems", "audio visual", "cable management"
                ],
                "technical_skills": [
                    "Windows", "Mac OS", "networking basics", "hardware components", "software applications",
                    "mobile devices", "printers", "audio visual equipment", "diagnostic tools"
                ],
                "soft_skills": [
                    "problem solving", "communication", "patience", "customer service",
                    "learning ability", "attention to detail", "reliability", "adaptability"
                ],
                "cultural_fit_keywords": [
                    "technical", "helpful", "patient", "problem-solver", "reliable",
                    "professional", "learning-oriented", "customer-focused", "detail-oriented"
                ],
                "disqualifying_factors": [
                    "poor technical skills", "poor customer service", "inability to learn",
                    "poor communication", "lack of patience"
                ],
                "experience_indicators": [
                    "IT technician", "computer technician", "help desk", "technical support",
                    "hardware technician", "system support", "user support"
                ],
                "education_preferences": [
                    "computer science", "information technology", "technical training",
                    "computer repair", "networking", "hospitality technology"
                ],
                "certifications": [
                    "CompTIA A+", "Microsoft", "hardware certification", "networking basics",
                    "customer service", "technical support", "hospitality systems"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "High",
                "training_requirements": "Moderate"
            },

            # ==========================================
            # SPA & WELLNESS DEPARTMENT
            # ==========================================
            "Spa Manager": {
                "must_have_skills": [
                    "spa management", "wellness programs", "team leadership", "customer service",
                    "treatment scheduling", "inventory management", "staff training", "performance management",
                    "budget awareness", "quality control", "guest relations", "communication"
                ],
                "nice_to_have_skills": [
                    "massage therapy", "esthetics", "wellness coaching", "retail management",
                    "marketing", "event coordination", "product knowledge", "health protocols"
                ],
                "technical_skills": [
                    "spa software", "scheduling systems", "POS systems", "inventory systems",
                    "treatment equipment", "sound systems", "lighting controls", "sanitation equipment"
                ],
                "soft_skills": [
                    "leadership", "wellness focus", "communication", "empathy", "organization",
                    "customer focus", "attention to detail", "calming presence", "team building"
                ],
                "cultural_fit_keywords": [
                    "wellness-focused", "leader", "calming", "professional", "empathetic",
                    "organized", "guest-focused", "quality-oriented", "serene"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of wellness knowledge", "poor customer service",
                    "inability to create calm environment", "poor communication"
                ],
                "experience_indicators": [
                    "spa manager", "wellness manager", "spa director", "massage therapy manager",
                    "esthetics manager", "fitness manager", "wellness coordinator"
                ],
                "education_preferences": [
                    "spa management", "wellness", "massage therapy", "esthetics", "hospitality management",
                    "business administration", "health and wellness"
                ],
                "certifications": [
                    "spa management", "massage therapy", "esthetics license", "wellness coaching",
                    "hospitality management", "customer service", "health and safety"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15
                },
                "min_experience_years": 3,
                "preferred_experience_years": 6,
                "salary_range": {"min": 45000, "max": 70000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            "Massage Therapist": {
                "must_have_skills": [
                    "massage therapy", "anatomy knowledge", "therapeutic techniques", "customer service",
                    "communication", "professional boundaries", "sanitation protocols", "physical stamina",
                    "empathy", "attention to detail", "time management", "wellness focus"
                ],
                "nice_to_have_skills": [
                    "specialized techniques", "aromatherapy", "hot stone", "deep tissue", "prenatal massage",
                    "reflexology", "sports massage", "energy work", "product knowledge"
                ],
                "technical_skills": [
                    "massage equipment", "treatment table setup", "sanitation equipment", "essential oils",
                    "hot stone equipment", "music systems", "lighting controls", "treatment tools"
                ],
                "soft_skills": [
                    "empathy", "communication", "professionalism", "healing touch", "calming presence",
                    "physical stamina", "attention to detail", "wellness mindset", "patience"
                ],
                "cultural_fit_keywords": [
                    "healing", "empathetic", "professional", "calming", "skilled",
                    "wellness-focused", "therapeutic", "caring", "serene"
                ],
                "disqualifying_factors": [
                    "lack of license", "poor boundaries", "inappropriate behavior",
                    "poor technique", "lack of empathy"
                ],
                "experience_indicators": [
                    "massage therapist", "therapeutic massage", "spa therapist", "wellness therapist",
                    "bodywork", "massage practice", "healing arts"
                ],
                "education_preferences": [
                    "massage therapy school", "therapeutic massage", "bodywork training",
                    "wellness studies", "anatomy and physiology", "healing arts"
                ],
                "certifications": [
                    "massage therapy license", "therapeutic massage", "specialized techniques",
                    "anatomy certification", "CPR/First Aid", "wellness certification"
                ],
                "scoring_weights": {
                    "experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10
                },
                "min_experience_years": 1,
                "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 50000},
                "growth_potential": "Moderate",
                "training_requirements": "Extensive"
            },

            # ==========================================
            # ENTERTAINMENT & ACTIVITIES DEPARTMENT
            # ==========================================
            "Activities Manager": {
                "must_have_skills": [
                    "activity planning", "event coordination", "team leadership", "guest entertainment",
                    "program development", "scheduling", "customer service", "communication",
                    "creativity", "problem solving", "performance management", "safety awareness"
                ],
                "nice_to_have_skills": [
                    "sports knowledge", "water activities", "cultural programming", "children's activities",
                    "entertainment skills", "music knowledge", "dance knowledge", "arts and crafts"
                ],
                "technical_skills": [
                    "sound systems", "microphones", "activity equipment", "scheduling software",
                    "entertainment technology", "safety equipment", "sports equipment", "craft supplies"
                ],
                "soft_skills": [
                    "creativity", "leadership", "enthusiasm", "communication", "energy",
                    "customer focus", "problem solving", "adaptability", "team building"
                ],
                "cultural_fit_keywords": [
                    "energetic", "creative", "leader", "enthusiastic", "fun",
                    "guest-focused", "organized", "innovative", "entertaining"
                ],
                "disqualifying_factors": [
                    "poor leadership", "lack of creativity", "low energy", "poor customer service",
                    "safety violations"
                ],
                "experience_indicators": [
                    "activities manager", "recreation manager", "entertainment manager", "program coordinator",
                    "activities coordinator", "resort activities", "guest services"
                ],
                "education_preferences": [
                    "recreation management", "hospitality management", "sports management",
                    "entertainment", "event management", "physical education"
                ],
                "certifications": [
                    "recreation certification", "activity management", "water safety", "first aid/CPR",
                    "entertainment certification", "sports certification"
                ],
                "scoring_weights": {
                    "experience": 0.30, "skills": 0.30, "cultural_fit": 0.25, "hospitality": 0.15
                },
                "min_experience_years": 2,
                "preferred_experience_years": 5,
                "salary_range": {"min": 35000, "max": 55000},
                "growth_potential": "Moderate",
                "training_requirements": "Moderate"
            },

            "Activities Coordinator": {
                "must_have_skills": [
                    "activity coordination", "guest interaction", "event support", "scheduling",
                    "customer service", "communication", "enthusiasm", "organization",
                    "team work", "problem solving", "safety awareness", "multitasking"
                ],
                "nice_to_have_skills": [
                    "sports knowledge", "entertainment skills", "children's activities", "water safety",
                    "arts and crafts", "music", "dance", "multilingual", "first aid"
                ],
                "technical_skills": [
                    "activity equipment", "sound systems", "sports equipment", "craft supplies",
                    "safety equipment", "entertainment technology", "communication devices"
                ],
                "soft_skills": [
                    "enthusiasm", "energy", "communication", "customer focus", "creativity",
                    "teamwork", "adaptability", "patience", "positive attitude"
                ],
                "cultural_fit_keywords": [
                    "enthusiastic", "energetic", "fun", "guest-focused", "creative",
                    "positive", "team player", "entertaining", "engaging"
                ],
                "disqualifying_factors": [
                    "low energy", "poor customer service", "lack of enthusiasm",
                    "safety violations", "poor communication"
                ],
                "experience_indicators": [
                    "activities coordinator", "recreation assistant", "entertainment staff",
                    "camp counselor", "activity leader", "guest services", "youth programs"
                ],
                "education_preferences": [
                    "recreation", "hospitality", "sports management", "education",
                    "entertainment", "customer service", "physical education"
                ],
                "certifications": [
                    "recreation certification", "water safety", "first aid/CPR", "activity leadership",
                    "customer service", "entertainment skills"
                ],
                "scoring_weights": {
                    "experience": 0.20, "skills": 0.30, "cultural_fit": 0.35, "hospitality": 0.15
                },
                "min_experience_years": 0,
                "preferred_experience_years": 2,
                "salary_range": {"min": 22000, "max": 35000},
                "growth_potential": "High",
                "training_requirements": "Basic"
            },

            # ==========================================
            # GUEST RELATIONS & VIP SERVICES
            # ==========================================
            "Guest Relations Manager": {
                "must_have_skills": [
                    "guest relations", "complaint resolution", "VIP services", "luxury hospitality",
                    "customer service excellence", "communication", "problem solving", "team leadership"
                ],
                "nice_to_have_skills": [
                    "multilingual", "cultural sensitivity", "luxury brands", "personalized service"
                ],
                "technical_skills": ["CRM systems", "guest feedback platforms", "luxury service standards"],
                "soft_skills": ["empathy", "diplomacy", "patience", "emotional intelligence"],
                "cultural_fit_keywords": ["guest-focused", "diplomatic", "luxury-minded", "solution-oriented"],
                "disqualifying_factors": ["poor guest interaction", "inflexibility", "lack of empathy"],
                "experience_indicators": ["guest relations", "VIP services", "luxury hospitality", "customer service"],
                "education_preferences": ["hospitality management", "communications", "business", "psychology"],
                "certifications": ["guest relations", "luxury service", "hospitality management"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 45000, "max": 70000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Guest Services Representative": {
                "must_have_skills": [
                    "customer service", "guest assistance", "problem resolution", "communication",
                    "hospitality service", "attention to detail", "multitasking", "professional demeanor"
                ],
                "nice_to_have_skills": [
                    "multilingual", "local knowledge", "concierge services", "reservation systems"
                ],
                "technical_skills": ["guest service software", "reservation systems", "communication tools"],
                "soft_skills": ["patience", "empathy", "adaptability", "positive attitude"],
                "cultural_fit_keywords": ["service-oriented", "helpful", "friendly", "professional"],
                "disqualifying_factors": ["poor communication", "impatience", "negative attitude"],
                "experience_indicators": ["guest services", "customer service", "hospitality", "reception"],
                "education_preferences": ["hospitality", "communications", "business", "tourism"],
                "certifications": ["customer service", "guest relations", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 32000, "max": 48000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "VIP Host": {
                "must_have_skills": [
                    "VIP service", "luxury hospitality", "personalized service", "attention to detail",
                    "discretion", "communication", "problem solving", "cultural sensitivity"
                ],
                "nice_to_have_skills": [
                    "multilingual", "fine dining", "wine knowledge", "etiquette", "luxury brands"
                ],
                "technical_skills": ["luxury service systems", "VIP databases", "communication tools"],
                "soft_skills": ["sophistication", "discretion", "anticipation", "refinement"],
                "cultural_fit_keywords": ["sophisticated", "discreet", "attentive", "refined"],
                "disqualifying_factors": ["lack of sophistication", "poor discretion", "inflexibility"],
                "experience_indicators": ["VIP services", "luxury hospitality", "personal service", "high-end"],
                "education_preferences": ["hospitality", "luxury service", "business", "communications"],
                "certifications": ["luxury service", "VIP training", "hospitality management"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Club Level Coordinator": {
                "must_have_skills": [
                    "club level service", "exclusive amenities", "member relations", "luxury service",
                    "attention to detail", "communication", "organization", "problem solving"
                ],
                "nice_to_have_skills": [
                    "wine service", "culinary knowledge", "event coordination", "cultural awareness"
                ],
                "technical_skills": ["club management systems", "member databases", "service platforms"],
                "soft_skills": ["exclusivity mindset", "attention to detail", "sophistication", "discretion"],
                "cultural_fit_keywords": ["exclusive", "sophisticated", "attentive", "premium"],
                "disqualifying_factors": ["lack of attention to detail", "poor service orientation", "inflexibility"],
                "experience_indicators": ["club level", "luxury service", "member services", "exclusive hospitality"],
                "education_preferences": ["hospitality", "luxury service", "business", "culinary"],
                "certifications": ["luxury service", "club management", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 58000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Personal Concierge": {
                "must_have_skills": [
                    "personal assistance", "concierge services", "local knowledge", "reservation management",
                    "itinerary planning", "communication", "problem solving", "discretion"
                ],
                "nice_to_have_skills": [
                    "multilingual", "cultural knowledge", "luxury services", "travel planning"
                ],
                "technical_skills": ["concierge software", "reservation systems", "communication tools"],
                "soft_skills": ["anticipation", "discretion", "resourcefulness", "sophistication"],
                "cultural_fit_keywords": ["resourceful", "discreet", "knowledgeable", "helpful"],
                "disqualifying_factors": ["poor local knowledge", "lack of resourcefulness", "poor communication"],
                "experience_indicators": ["concierge", "personal assistance", "travel services", "luxury hospitality"],
                "education_preferences": ["hospitality", "tourism", "communications", "local studies"],
                "certifications": ["concierge certification", "travel planning", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 42000, "max": 62000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            # ==========================================
            # TRANSPORTATION & LOGISTICS
            # ==========================================
            "Valet Parking Attendant": {
                "must_have_skills": [
                    "driving skills", "customer service", "vehicle handling", "parking management",
                    "professional appearance", "communication", "attention to detail", "physical fitness"
                ],
                "nice_to_have_skills": [
                    "luxury vehicle experience", "manual transmission", "defensive driving", "guest relations"
                ],
                "technical_skills": ["parking systems", "vehicle operation", "safety protocols"],
                "soft_skills": ["trustworthiness", "responsibility", "courtesy", "reliability"],
                "cultural_fit_keywords": ["trustworthy", "responsible", "courteous", "professional"],
                "disqualifying_factors": ["poor driving record", "unreliable", "poor customer service"],
                "experience_indicators": ["valet", "parking", "driving", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "automotive"],
                "certifications": ["valid driver's license", "defensive driving", "valet training"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 2,
                "salary_range": {"min": 28000, "max": 40000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Transportation Coordinator": {
                "must_have_skills": [
                    "transportation coordination", "scheduling", "logistics", "customer service",
                    "vendor management", "route planning", "communication", "organization"
                ],
                "nice_to_have_skills": [
                    "fleet management", "GPS systems", "airport transfers", "group transportation"
                ],
                "technical_skills": ["transportation software", "GPS systems", "scheduling tools"],
                "soft_skills": ["organization", "reliability", "problem solving", "communication"],
                "cultural_fit_keywords": ["organized", "reliable", "efficient", "service-oriented"],
                "disqualifying_factors": ["poor organization", "unreliable", "poor communication"],
                "experience_indicators": ["transportation", "logistics", "coordination", "hospitality"],
                "education_preferences": ["logistics", "transportation", "hospitality", "business"],
                "certifications": ["transportation management", "logistics", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Mid-Level", "training_requirements": "Standard"
            },

            "Shuttle Driver": {
                "must_have_skills": [
                    "safe driving", "customer service", "vehicle maintenance", "punctuality",
                    "professional appearance", "local knowledge", "communication", "reliability"
                ],
                "nice_to_have_skills": [
                    "commercial license", "multilingual", "tour guiding", "defensive driving"
                ],
                "technical_skills": ["vehicle operation", "GPS systems", "safety protocols"],
                "soft_skills": ["responsibility", "courtesy", "punctuality", "professionalism"],
                "cultural_fit_keywords": ["reliable", "safe", "courteous", "professional"],
                "disqualifying_factors": ["poor driving record", "unreliable", "unprofessional"],
                "experience_indicators": ["driving", "transportation", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "transportation"],
                "certifications": ["valid driver's license", "commercial license", "defensive driving"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 30000, "max": 45000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Luggage Porter": {
                "must_have_skills": [
                    "physical fitness", "customer service", "luggage handling", "attention to detail",
                    "reliability", "communication", "professional appearance", "teamwork"
                ],
                "nice_to_have_skills": [
                    "multilingual", "guest relations", "hospitality experience", "equipment operation"
                ],
                "technical_skills": ["luggage equipment", "safety protocols", "handling techniques"],
                "soft_skills": ["helpfulness", "courtesy", "reliability", "strength"],
                "cultural_fit_keywords": ["helpful", "courteous", "reliable", "strong"],
                "disqualifying_factors": ["physical limitations", "poor customer service", "unreliable"],
                "experience_indicators": ["porter", "luggage handling", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "customer service"],
                "certifications": ["safety training", "customer service", "hospitality"],
                "scoring_weights": {"experience": 0.25, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.15},
                "min_experience_years": 0, "preferred_experience_years": 1,
                "salary_range": {"min": 26000, "max": 35000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Airport Transfer Coordinator": {
                "must_have_skills": [
                    "transfer coordination", "scheduling", "customer service", "logistics",
                    "communication", "organization", "problem solving", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "airline knowledge", "flight tracking", "multilingual", "travel coordination"
                ],
                "technical_skills": ["booking systems", "flight tracking", "communication tools"],
                "soft_skills": ["organization", "reliability", "flexibility", "communication"],
                "cultural_fit_keywords": ["organized", "reliable", "flexible", "service-oriented"],
                "disqualifying_factors": ["poor organization", "inflexibility", "poor communication"],
                "experience_indicators": ["transfer coordination", "travel services", "hospitality", "logistics"],
                "education_preferences": ["hospitality", "travel", "logistics", "business"],
                "certifications": ["travel coordination", "hospitality", "logistics"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            # ==========================================
            # CONFERENCE & EVENTS
            # ==========================================
            "Conference Manager": {
                "must_have_skills": [
                    "conference management", "event planning", "vendor coordination", "logistics",
                    "customer service", "project management", "communication", "budget management"
                ],
                "nice_to_have_skills": [
                    "AV equipment", "catering coordination", "group sales", "contract negotiation"
                ],
                "technical_skills": ["event management software", "AV systems", "booking systems"],
                "soft_skills": ["organization", "multitasking", "problem solving", "leadership"],
                "cultural_fit_keywords": ["organized", "detail-oriented", "service-focused", "professional"],
                "disqualifying_factors": ["poor organization", "inflexibility", "poor communication"],
                "experience_indicators": ["conference management", "event planning", "meetings", "hospitality"],
                "education_preferences": ["event management", "hospitality", "business", "communications"],
                "certifications": ["event planning", "conference management", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            "Event Coordinator": {
                "must_have_skills": [
                    "event planning", "coordination", "vendor management", "timeline management",
                    "customer service", "budget management", "communication", "problem solving"
                ],
                "nice_to_have_skills": [
                    "wedding planning", "corporate events", "social events", "catering coordination"
                ],
                "technical_skills": ["event software", "booking systems", "project management tools"],
                "soft_skills": ["creativity", "organization", "flexibility", "attention to detail"],
                "cultural_fit_keywords": ["creative", "organized", "flexible", "service-oriented"],
                "disqualifying_factors": ["poor organization", "inflexibility", "poor client relations"],
                "experience_indicators": ["event planning", "coordination", "hospitality", "weddings"],
                "education_preferences": ["event management", "hospitality", "business", "marketing"],
                "certifications": ["event planning", "wedding planning", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "Wedding Coordinator": {
                "must_have_skills": [
                    "wedding planning", "event coordination", "vendor management", "client relations",
                    "timeline management", "attention to detail", "communication", "problem solving"
                ],
                "nice_to_have_skills": [
                    "floral design", "catering coordination", "photography coordination", "ceremony planning"
                ],
                "technical_skills": ["wedding planning software", "event management tools", "booking systems"],
                "soft_skills": ["patience", "creativity", "organization", "emotional intelligence"],
                "cultural_fit_keywords": ["creative", "patient", "detail-oriented", "romantic"],
                "disqualifying_factors": ["poor organization", "impatience", "inflexibility"],
                "experience_indicators": ["wedding planning", "event coordination", "hospitality", "celebrations"],
                "education_preferences": ["event management", "hospitality", "business", "design"],
                "certifications": ["wedding planning", "event coordination", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 38000, "max": 58000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Meeting Planner": {
                "must_have_skills": [
                    "meeting planning", "logistics coordination", "venue management", "vendor relations",
                    "budget management", "timeline management", "communication", "problem solving"
                ],
                "nice_to_have_skills": [
                    "corporate events", "conference planning", "AV coordination", "travel coordination"
                ],
                "technical_skills": ["meeting planning software", "booking systems", "budget tools"],
                "soft_skills": ["organization", "attention to detail", "multitasking", "communication"],
                "cultural_fit_keywords": ["organized", "detail-oriented", "professional", "efficient"],
                "disqualifying_factors": ["poor organization", "missed deadlines", "poor communication"],
                "experience_indicators": ["meeting planning", "event coordination", "corporate events", "hospitality"],
                "education_preferences": ["event management", "business", "hospitality", "communications"],
                "certifications": ["meeting planning", "event coordination", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 42000, "max": 62000}, "growth_potential": "Mid-Level", "training_requirements": "Standard"
            },

            "AV Technician": {
                "must_have_skills": [
                    "AV equipment", "technical setup", "troubleshooting", "equipment maintenance",
                    "customer service", "problem solving", "attention to detail", "safety protocols"
                ],
                "nice_to_have_skills": [
                    "lighting systems", "sound systems", "video equipment", "streaming technology"
                ],
                "technical_skills": ["AV systems", "technical equipment", "troubleshooting", "setup procedures"],
                "soft_skills": ["technical aptitude", "problem solving", "reliability", "communication"],
                "cultural_fit_keywords": ["technical", "reliable", "problem-solver", "detail-oriented"],
                "disqualifying_factors": ["poor technical skills", "unreliable", "safety violations"],
                "experience_indicators": ["AV technician", "technical support", "equipment operation", "events"],
                "education_preferences": ["technical education", "electronics", "communications", "hospitality"],
                "certifications": ["AV certification", "technical training", "safety certification"],
                "scoring_weights": {"experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Mid-Level", "training_requirements": "Technical"
            },

            # ==========================================
            # HEALTH & WELLNESS
            # ==========================================
            "Wellness Coordinator": {
                "must_have_skills": [
                    "wellness programs", "health promotion", "fitness coordination", "customer service",
                    "program development", "safety protocols", "communication", "organization"
                ],
                "nice_to_have_skills": [
                    "nutrition knowledge", "fitness training", "spa services", "meditation"
                ],
                "technical_skills": ["wellness software", "fitness equipment", "health tracking tools"],
                "soft_skills": ["motivation", "empathy", "enthusiasm", "patience"],
                "cultural_fit_keywords": ["health-conscious", "motivating", "caring", "positive"],
                "disqualifying_factors": ["poor health habits", "lack of enthusiasm", "safety violations"],
                "experience_indicators": ["wellness", "fitness", "health promotion", "spa", "hospitality"],
                "education_preferences": ["health sciences", "fitness", "wellness", "hospitality"],
                "certifications": ["wellness coaching", "fitness", "CPR", "first aid"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 60000}, "growth_potential": "Mid-Level", "training_requirements": "Standard"
            },

            "Fitness Instructor": {
                "must_have_skills": [
                    "fitness instruction", "exercise programming", "safety protocols", "customer service",
                    "motivational skills", "anatomy knowledge", "equipment operation", "communication"
                ],
                "nice_to_have_skills": [
                    "specialized training", "nutrition knowledge", "injury prevention", "group fitness"
                ],
                "technical_skills": ["fitness equipment", "exercise software", "heart rate monitors"],
                "soft_skills": ["motivation", "enthusiasm", "patience", "energy"],
                "cultural_fit_keywords": ["energetic", "motivating", "health-focused", "positive"],
                "disqualifying_factors": ["poor fitness", "safety violations", "lack of certification"],
                "experience_indicators": ["fitness instruction", "personal training", "group fitness", "wellness"],
                "education_preferences": ["exercise science", "kinesiology", "fitness", "health"],
                "certifications": ["fitness certification", "CPR", "first aid", "specialized training"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Specialized"
            },

            "Yoga Instructor": {
                "must_have_skills": [
                    "yoga instruction", "meditation", "breathing techniques", "anatomy knowledge",
                    "safety protocols", "class management", "customer service", "flexibility"
                ],
                "nice_to_have_skills": [
                    "multiple yoga styles", "spiritual guidance", "wellness coaching", "injury modification"
                ],
                "technical_skills": ["yoga equipment", "sound systems", "class scheduling"],
                "soft_skills": ["calmness", "patience", "mindfulness", "spirituality"],
                "cultural_fit_keywords": ["mindful", "calm", "spiritual", "wellness-focused"],
                "disqualifying_factors": ["poor physical condition", "lack of certification", "impatience"],
                "experience_indicators": ["yoga instruction", "meditation", "wellness", "fitness"],
                "education_preferences": ["yoga studies", "fitness", "wellness", "spiritual studies"],
                "certifications": ["yoga certification", "meditation training", "CPR", "first aid"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 32000, "max": 48000}, "growth_potential": "Entry-Mid", "training_requirements": "Specialized"
            },

            "Pool Attendant": {
                "must_have_skills": [
                    "water safety", "customer service", "pool maintenance", "safety protocols",
                    "equipment operation", "cleaning", "communication", "attention to detail"
                ],
                "nice_to_have_skills": [
                    "lifeguard certification", "CPR", "first aid", "swimming instruction"
                ],
                "technical_skills": ["pool equipment", "chemical testing", "cleaning equipment"],
                "soft_skills": ["vigilance", "responsibility", "helpfulness", "reliability"],
                "cultural_fit_keywords": ["safety-conscious", "responsible", "helpful", "vigilant"],
                "disqualifying_factors": ["poor swimming ability", "safety violations", "unreliable"],
                "experience_indicators": ["pool maintenance", "water safety", "customer service", "hospitality"],
                "education_preferences": ["high school", "hospitality", "recreation", "water safety"],
                "certifications": ["water safety", "pool maintenance", "CPR", "first aid"],
                "scoring_weights": {"experience": 0.25, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 0, "preferred_experience_years": 2,
                "salary_range": {"min": 28000, "max": 40000}, "growth_potential": "Entry", "training_requirements": "Basic"
            },

            "Lifeguard": {
                "must_have_skills": [
                    "water safety", "rescue techniques", "first aid", "CPR", "emergency response",
                    "surveillance", "communication", "physical fitness"
                ],
                "nice_to_have_skills": [
                    "swimming instruction", "water sports", "emergency medical training", "customer service"
                ],
                "technical_skills": ["rescue equipment", "first aid equipment", "communication devices"],
                "soft_skills": ["vigilance", "quick response", "calmness under pressure", "responsibility"],
                "cultural_fit_keywords": ["vigilant", "responsible", "quick-thinking", "safety-focused"],
                "disqualifying_factors": ["poor swimming ability", "slow response time", "safety violations"],
                "experience_indicators": ["lifeguarding", "water safety", "rescue", "emergency response"],
                "education_preferences": ["water safety", "emergency response", "recreation", "health"],
                "certifications": ["lifeguard certification", "CPR", "first aid", "water safety instructor"],
                "scoring_weights": {"experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 0, "preferred_experience_years": 2,
                "salary_range": {"min": 30000, "max": 45000}, "growth_potential": "Entry", "training_requirements": "Specialized"
            },

            # ==========================================
            # BUSINESS & REVENUE MANAGEMENT
            # ==========================================
            "Revenue Manager": {
                "must_have_skills": [
                    "revenue management", "pricing strategy", "demand forecasting", "data analysis",
                    "financial analysis", "market research", "optimization", "reporting"
                ],
                "nice_to_have_skills": [
                    "revenue management systems", "competitive analysis", "yield management", "distribution"
                ],
                "technical_skills": ["revenue management software", "analytics tools", "pricing systems"],
                "soft_skills": ["analytical thinking", "strategic planning", "communication", "attention to detail"],
                "cultural_fit_keywords": ["analytical", "strategic", "results-driven", "detail-oriented"],
                "disqualifying_factors": ["poor analytical skills", "lack of financial acumen", "inflexibility"],
                "experience_indicators": ["revenue management", "pricing", "analytics", "finance", "hospitality"],
                "education_preferences": ["finance", "economics", "business", "hospitality", "mathematics"],
                "certifications": ["revenue management", "financial analysis", "hospitality finance"],
                "scoring_weights": {"experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 65000, "max": 95000}, "growth_potential": "Senior", "training_requirements": "Advanced"
            },

            "Purchasing Manager": {
                "must_have_skills": [
                    "procurement", "vendor management", "contract negotiation", "inventory management",
                    "cost analysis", "supplier relations", "budget management", "quality control"
                ],
                "nice_to_have_skills": [
                    "hospitality purchasing", "food & beverage procurement", "sustainability", "logistics"
                ],
                "technical_skills": ["procurement software", "inventory systems", "ERP systems"],
                "soft_skills": ["negotiation", "analytical thinking", "organization", "communication"],
                "cultural_fit_keywords": ["cost-conscious", "analytical", "negotiator", "organized"],
                "disqualifying_factors": ["poor negotiation", "disorganization", "lack of cost awareness"],
                "experience_indicators": ["purchasing", "procurement", "vendor management", "hospitality"],
                "education_preferences": ["business", "supply chain", "hospitality", "finance"],
                "certifications": ["procurement", "supply chain", "hospitality purchasing"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 55000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            "Business Analyst": {
                "must_have_skills": [
                    "business analysis", "data analysis", "process improvement", "reporting",
                    "requirements gathering", "problem solving", "communication", "project management"
                ],
                "nice_to_have_skills": [
                    "hospitality analytics", "performance metrics", "dashboard creation", "process mapping"
                ],
                "technical_skills": ["analytics software", "business intelligence tools", "database systems"],
                "soft_skills": ["analytical thinking", "attention to detail", "communication", "critical thinking"],
                "cultural_fit_keywords": ["analytical", "detail-oriented", "improvement-focused", "systematic"],
                "disqualifying_factors": ["poor analytical skills", "inflexibility", "poor communication"],
                "experience_indicators": ["business analysis", "data analysis", "process improvement", "hospitality"],
                "education_preferences": ["business", "analytics", "hospitality", "information systems"],
                "certifications": ["business analysis", "analytics", "project management"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 50000, "max": 70000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Controller": {
                "must_have_skills": [
                    "financial control", "accounting oversight", "financial reporting", "budget management",
                    "compliance", "audit management", "team leadership", "financial analysis"
                ],
                "nice_to_have_skills": [
                    "hospitality accounting", "cost control", "forecasting", "tax preparation"
                ],
                "technical_skills": ["accounting software", "financial systems", "reporting tools"],
                "soft_skills": ["attention to detail", "leadership", "communication", "analytical thinking"],
                "cultural_fit_keywords": ["detail-oriented", "accurate", "responsible", "analytical"],
                "disqualifying_factors": ["poor attention to detail", "lack of accounting knowledge", "poor leadership"],
                "experience_indicators": ["controller", "accounting management", "financial oversight", "hospitality"],
                "education_preferences": ["accounting", "finance", "business", "hospitality finance"],
                "certifications": ["CPA", "CMA", "financial management", "hospitality accounting"],
                "scoring_weights": {"experience": 0.40, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 5, "preferred_experience_years": 8,
                "salary_range": {"min": 70000, "max": 100000}, "growth_potential": "Senior", "training_requirements": "Advanced"
            },

            "Cost Control Manager": {
                "must_have_skills": [
                    "cost control", "budget analysis", "variance analysis", "financial reporting",
                    "inventory control", "process improvement", "team leadership", "analytical skills"
                ],
                "nice_to_have_skills": [
                    "hospitality cost control", "food cost management", "labor cost analysis", "forecasting"
                ],
                "technical_skills": ["cost control software", "analytics tools", "inventory systems"],
                "soft_skills": ["analytical thinking", "attention to detail", "communication", "organization"],
                "cultural_fit_keywords": ["cost-conscious", "analytical", "efficient", "detail-oriented"],
                "disqualifying_factors": ["poor analytical skills", "lack of cost awareness", "disorganization"],
                "experience_indicators": ["cost control", "financial analysis", "budget management", "hospitality"],
                "education_preferences": ["finance", "accounting", "business", "hospitality management"],
                "certifications": ["cost control", "financial analysis", "hospitality finance"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 55000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            # ==========================================
            # DIGITAL MARKETING & TECHNOLOGY
            # ==========================================
            "Digital Marketing Manager": {
                "must_have_skills": [
                    "digital marketing", "social media", "online advertising", "content creation",
                    "SEO/SEM", "analytics", "campaign management", "brand management"
                ],
                "nice_to_have_skills": [
                    "hospitality marketing", "OTA management", "email marketing", "influencer marketing"
                ],
                "technical_skills": ["marketing platforms", "analytics tools", "social media tools"],
                "soft_skills": ["creativity", "analytical thinking", "communication", "adaptability"],
                "cultural_fit_keywords": ["creative", "digital-savvy", "innovative", "results-driven"],
                "disqualifying_factors": ["poor digital skills", "lack of creativity", "poor analytics"],
                "experience_indicators": ["digital marketing", "social media", "online marketing", "hospitality"],
                "education_preferences": ["marketing", "communications", "business", "digital media"],
                "certifications": ["digital marketing", "Google Analytics", "social media", "hospitality marketing"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 3, "preferred_experience_years": 5,
                "salary_range": {"min": 50000, "max": 75000}, "growth_potential": "Mid-Senior", "training_requirements": "Advanced"
            },

            "Social Media Coordinator": {
                "must_have_skills": [
                    "social media management", "content creation", "community management", "photography",
                    "copywriting", "brand consistency", "engagement", "analytics"
                ],
                "nice_to_have_skills": [
                    "video creation", "graphic design", "influencer relations", "paid advertising"
                ],
                "technical_skills": ["social media platforms", "content creation tools", "analytics tools"],
                "soft_skills": ["creativity", "communication", "attention to detail", "adaptability"],
                "cultural_fit_keywords": ["creative", "social", "trendy", "engaging"],
                "disqualifying_factors": ["poor social skills", "lack of creativity", "poor attention to detail"],
                "experience_indicators": ["social media", "content creation", "marketing", "communications"],
                "education_preferences": ["marketing", "communications", "graphic design", "media"],
                "certifications": ["social media marketing", "content creation", "digital marketing"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "Content Creator": {
                "must_have_skills": [
                    "content creation", "writing", "photography", "video production", "editing",
                    "storytelling", "brand voice", "social media content"
                ],
                "nice_to_have_skills": [
                    "graphic design", "animation", "SEO writing", "hospitality content"
                ],
                "technical_skills": ["content creation software", "editing tools", "design software"],
                "soft_skills": ["creativity", "attention to detail", "storytelling", "visual sense"],
                "cultural_fit_keywords": ["creative", "visual", "storyteller", "artistic"],
                "disqualifying_factors": ["lack of creativity", "poor visual sense", "poor writing"],
                "experience_indicators": ["content creation", "photography", "video production", "marketing"],
                "education_preferences": ["media arts", "communications", "graphic design", "marketing"],
                "certifications": ["content creation", "photography", "video production", "social media"],
                "scoring_weights": {"experience": 0.30, "skills": 0.40, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "SEO Specialist": {
                "must_have_skills": [
                    "search engine optimization", "keyword research", "content optimization", "analytics",
                    "link building", "technical SEO", "reporting", "strategy development"
                ],
                "nice_to_have_skills": [
                    "local SEO", "hospitality SEO", "PPC", "conversion optimization"
                ],
                "technical_skills": ["SEO tools", "analytics platforms", "content management systems"],
                "soft_skills": ["analytical thinking", "attention to detail", "persistence", "communication"],
                "cultural_fit_keywords": ["analytical", "detail-oriented", "strategic", "results-driven"],
                "disqualifying_factors": ["poor analytical skills", "lack of technical knowledge", "impatience"],
                "experience_indicators": ["SEO", "digital marketing", "web optimization", "analytics"],
                "education_preferences": ["marketing", "computer science", "communications", "business"],
                "certifications": ["SEO certification", "Google Analytics", "digital marketing"],
                "scoring_weights": {"experience": 0.35, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.10},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 45000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Advanced"
            },

            "Web Developer": {
                "must_have_skills": [
                    "web development", "HTML", "CSS", "JavaScript", "responsive design",
                    "website maintenance", "troubleshooting", "user experience"
                ],
                "nice_to_have_skills": [
                    "CMS platforms", "e-commerce", "booking systems", "mobile development"
                ],
                "technical_skills": ["programming languages", "development tools", "web technologies"],
                "soft_skills": ["problem solving", "attention to detail", "logical thinking", "patience"],
                "cultural_fit_keywords": ["technical", "detail-oriented", "problem-solver", "innovative"],
                "disqualifying_factors": ["poor technical skills", "lack of attention to detail", "inflexibility"],
                "experience_indicators": ["web development", "programming", "website design", "IT"],
                "education_preferences": ["computer science", "web development", "information technology"],
                "certifications": ["web development", "programming", "IT certifications"],
                "scoring_weights": {"experience": 0.40, "skills": 0.40, "cultural_fit": 0.15, "hospitality": 0.05},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 50000, "max": 75000}, "growth_potential": "Mid-Level", "training_requirements": "Technical"
            },

            # ==========================================
            # SPECIALTY SERVICES & INTERNATIONAL
            # ==========================================
            "Cultural Liaison": {
                "must_have_skills": [
                    "cultural sensitivity", "multilingual communication", "guest relations", "cultural programs",
                    "interpretation", "customer service", "conflict resolution", "communication"
                ],
                "nice_to_have_skills": [
                    "multiple languages", "cultural training", "international experience", "tourism"
                ],
                "technical_skills": ["translation tools", "cultural databases", "communication systems"],
                "soft_skills": ["cultural awareness", "empathy", "patience", "adaptability"],
                "cultural_fit_keywords": ["culturally sensitive", "multilingual", "worldly", "inclusive"],
                "disqualifying_factors": ["cultural insensitivity", "language barriers", "poor communication"],
                "experience_indicators": ["cultural work", "international", "tourism", "guest relations"],
                "education_preferences": ["cultural studies", "international relations", "languages", "hospitality"],
                "certifications": ["cultural training", "language certification", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 60000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Pet Concierge": {
                "must_have_skills": [
                    "animal care", "pet services", "customer service", "pet safety", "communication",
                    "organization", "attention to detail", "problem solving"
                ],
                "nice_to_have_skills": [
                    "veterinary knowledge", "pet training", "pet grooming", "animal behavior"
                ],
                "technical_skills": ["pet care equipment", "safety protocols", "booking systems"],
                "soft_skills": ["animal love", "patience", "caring", "reliability"],
                "cultural_fit_keywords": ["animal lover", "caring", "responsible", "service-oriented"],
                "disqualifying_factors": ["animal allergies", "fear of animals", "poor animal handling"],
                "experience_indicators": ["pet care", "animal services", "veterinary", "hospitality"],
                "education_preferences": ["animal science", "veterinary", "hospitality", "biology"],
                "certifications": ["pet care", "animal handling", "first aid", "hospitality"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 50000}, "growth_potential": "Entry-Mid", "training_requirements": "Specialized"
            },

            "Wine Steward": {
                "must_have_skills": [
                    "wine knowledge", "wine service", "wine pairing", "customer service", "wine storage",
                    "inventory management", "presentation", "communication"
                ],
                "nice_to_have_skills": [
                    "sommelier training", "wine certification", "food pairing", "cellar management"
                ],
                "technical_skills": ["wine preservation", "cellar management", "POS systems"],
                "soft_skills": ["sophistication", "attention to detail", "passion", "communication"],
                "cultural_fit_keywords": ["sophisticated", "knowledgeable", "passionate", "refined"],
                "disqualifying_factors": ["poor wine knowledge", "alcohol problems", "poor presentation"],
                "experience_indicators": ["wine service", "sommelier", "fine dining", "hospitality"],
                "education_preferences": ["hospitality", "culinary", "wine studies", "business"],
                "certifications": ["wine certification", "sommelier", "alcohol service", "hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.15, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 40000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Personal Shopper": {
                "must_have_skills": [
                    "personal shopping", "fashion sense", "customer service", "local knowledge",
                    "budget management", "communication", "organization", "trend awareness"
                ],
                "nice_to_have_skills": [
                    "luxury brands", "styling", "cultural knowledge", "multilingual"
                ],
                "technical_skills": ["shopping apps", "budget tools", "communication devices"],
                "soft_skills": ["style sense", "empathy", "patience", "discretion"],
                "cultural_fit_keywords": ["stylish", "trendy", "helpful", "sophisticated"],
                "disqualifying_factors": ["poor fashion sense", "overspending", "poor customer service"],
                "experience_indicators": ["personal shopping", "retail", "fashion", "customer service"],
                "education_preferences": ["fashion", "retail", "business", "hospitality"],
                "certifications": ["personal shopping", "styling", "customer service"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 35000, "max": 55000}, "growth_potential": "Entry-Mid", "training_requirements": "Standard"
            },

            "Art Curator": {
                "must_have_skills": [
                    "art knowledge", "curation", "exhibition planning", "art history", "preservation",
                    "organization", "communication", "aesthetic sense"
                ],
                "nice_to_have_skills": [
                    "museum experience", "gallery management", "art valuation", "cultural programming"
                ],
                "technical_skills": ["preservation techniques", "cataloging systems", "exhibition tools"],
                "soft_skills": ["aesthetic sense", "attention to detail", "cultural awareness", "creativity"],
                "cultural_fit_keywords": ["artistic", "cultured", "sophisticated", "knowledgeable"],
                "disqualifying_factors": ["poor art knowledge", "lack of aesthetic sense", "carelessness"],
                "experience_indicators": ["art curation", "museum", "gallery", "cultural institutions"],
                "education_preferences": ["art history", "museum studies", "fine arts", "cultural studies"],
                "certifications": ["art curation", "museum studies", "preservation"],
                "scoring_weights": {"experience": 0.35, "skills": 0.35, "cultural_fit": 0.20, "hospitality": 0.10},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 42000, "max": 62000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Sustainability Coordinator": {
                "must_have_skills": [
                    "sustainability programs", "environmental management", "waste reduction", "energy efficiency",
                    "green initiatives", "compliance", "reporting", "project management"
                ],
                "nice_to_have_skills": [
                    "LEED certification", "carbon footprint", "renewable energy", "green building"
                ],
                "technical_skills": ["environmental monitoring", "sustainability software", "reporting tools"],
                "soft_skills": ["environmental consciousness", "innovation", "communication", "organization"],
                "cultural_fit_keywords": ["environmentally conscious", "innovative", "responsible", "forward-thinking"],
                "disqualifying_factors": ["lack of environmental awareness", "resistance to change", "poor organization"],
                "experience_indicators": ["sustainability", "environmental", "green programs", "hospitality"],
                "education_preferences": ["environmental science", "sustainability", "business", "hospitality"],
                "certifications": ["sustainability", "LEED", "environmental management", "green hospitality"],
                "scoring_weights": {"experience": 0.35, "skills": 0.30, "cultural_fit": 0.20, "hospitality": 0.15},
                "min_experience_years": 2, "preferred_experience_years": 4,
                "salary_range": {"min": 45000, "max": 65000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
            },

            "Accessibility Coordinator": {
                "must_have_skills": [
                    "accessibility compliance", "ADA knowledge", "guest assistance", "accommodation planning",
                    "disability awareness", "communication", "problem solving", "empathy"
                ],
                "nice_to_have_skills": [
                    "sign language", "assistive technology", "universal design", "accessibility auditing"
                ],
                "technical_skills": ["assistive technology", "accessibility tools", "compliance software"],
                "soft_skills": ["empathy", "patience", "problem solving", "advocacy"],
                "cultural_fit_keywords": ["inclusive", "empathetic", "helpful", "accommodating"],
                "disqualifying_factors": ["lack of empathy", "poor understanding of disabilities", "impatience"],
                "experience_indicators": ["accessibility", "disability services", "ADA compliance", "guest services"],
                "education_preferences": ["disability studies", "social work", "hospitality", "public administration"],
                "certifications": ["ADA compliance", "accessibility", "disability services"],
                "scoring_weights": {"experience": 0.30, "skills": 0.35, "cultural_fit": 0.25, "hospitality": 0.10},
                "min_experience_years": 1, "preferred_experience_years": 3,
                "salary_range": {"min": 38000, "max": 55000}, "growth_potential": "Mid-Level", "training_requirements": "Specialized"
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
        """Calculate intelligent, position-specific candidate score with deep content analysis."""
        position_data = self.position_intelligence.get(position, {})
        if not position_data:
            logger.warning(f"Position '{position}' not found in intelligence database")
            return {"total_score": 0, "breakdown": {}, "recommendation": "Unable to evaluate"}
        
        resume_text = candidate.get("resume_text", "").lower()
        
        # Get position-specific requirements
        must_have_skills = position_data.get("must_have_skills", [])
        nice_to_have_skills = position_data.get("nice_to_have_skills", [])
        technical_skills = position_data.get("technical_skills", [])
        experience_indicators = position_data.get("experience_indicators", [])
        education_preferences = position_data.get("education_preferences", [])
        certifications = position_data.get("certifications", [])
        disqualifying_factors = position_data.get("disqualifying_factors", [])
        cultural_fit_keywords = position_data.get("cultural_fit_keywords", [])
        
        # Initialize detailed scoring
        scores = {
            "experience_relevance": 0.0,
            "skills_match": 0.0,
            "education_fit": 0.0,
            "technical_competency": 0.0,
            "cultural_alignment": 0.0,
            "position_specific": 0.0,
            "communication_quality": 0.0,
            "career_progression": 0.0
        }
        
        breakdown = {}
        
        # 1. EXPERIENCE RELEVANCE ANALYSIS (35% weight)
        experience_score = 0.0
        relevant_exp_count = 0
        exp_analysis = candidate.get("experience_analysis", {})
        total_experience_years = exp_analysis.get("total_years", 0)
        
        # Check for direct position experience with context analysis
        position_lower = position.lower()
        for indicator in experience_indicators:
            indicator_lower = indicator.lower()
            if indicator_lower in resume_text:
                # Enhanced context checking
                context_words = ["experience", "worked", "position", "role", "job", "years", "manager", "supervisor"]
                context_found = any(word in resume_text[max(0, resume_text.find(indicator_lower)-50):resume_text.find(indicator_lower)+50] 
                                  for word in context_words)
                if context_found:
                    experience_score += 0.25
                    relevant_exp_count += 1
                else:
                    experience_score += 0.1  # Lower score if no context
        
        # Position-specific keyword analysis with frequency weighting
        position_keywords = {
            # General hospitality
            "hotel": 0.15, "hospitality": 0.15, "resort": 0.15, "guest": 0.1,
            "customer service": 0.12, "team": 0.08, "management": 0.1,
            
            # Position-specific enhancements
            position_lower: 0.3,  # Direct position match gets highest score
        }
        
        # Add position-specific keywords
        if "front desk" in position_lower or "reception" in position_lower:
            position_keywords.update({"check-in": 0.2, "check-out": 0.2, "reservation": 0.15, "pms": 0.15})
        elif "housekeeping" in position_lower:
            position_keywords.update({"cleaning": 0.2, "room": 0.15, "maintenance": 0.1, "laundry": 0.1})
        elif "server" in position_lower or "waiter" in position_lower:
            position_keywords.update({"restaurant": 0.2, "food service": 0.2, "menu": 0.1, "dining": 0.15})
        elif "chef" in position_lower or "cook" in position_lower:
            position_keywords.update({"kitchen": 0.2, "cooking": 0.2, "culinary": 0.2, "food prep": 0.15})
        elif "manager" in position_lower:
            position_keywords.update({"leadership": 0.2, "supervision": 0.15, "budget": 0.1, "operations": 0.15})
        
        for keyword, base_weight in position_keywords.items():
            if keyword in resume_text:
                # Count frequency and context for better scoring
                frequency = resume_text.count(keyword)
                # Bonus for multiple mentions (up to 3x)
                frequency_multiplier = min(1 + (frequency - 1) * 0.3, 2.0)
                experience_score += base_weight * frequency_multiplier
        
        # Experience years analysis with position requirements
        min_years = position_data.get("min_experience_years", 0)
        preferred_years = position_data.get("preferred_experience_years", 3)
        
        if total_experience_years >= preferred_years:
            experience_score += 0.2
        elif total_experience_years >= min_years:
            experience_score += 0.15
        elif total_experience_years > 0:
            experience_score += 0.1
        
        # Leadership and progression indicators
        leadership_terms = ["manager", "supervisor", "lead", "coordinator", "head", "chief", "director"]
        progression_terms = ["promoted", "advanced", "grew", "developed", "improved", "increased"]
        
        leadership_found = sum(1 for term in leadership_terms if term in resume_text)
        progression_found = sum(1 for term in progression_terms if term in resume_text)
        
        if leadership_found > 0:
            experience_score += 0.1 * min(leadership_found, 3)
        if progression_found > 0:
            experience_score += 0.05 * min(progression_found, 2)
        
        scores["experience_relevance"] = min(experience_score, 1.0)
        breakdown["experience"] = {
            "score": scores["experience_relevance"],
            "relevant_positions": relevant_exp_count,
            "total_years": total_experience_years,
            "meets_minimum": total_experience_years >= min_years,
            "leadership_indicators": leadership_found,
            "progression_indicators": progression_found
        }
        
        # 2. SKILLS MATCH ANALYSIS (30% weight)
        skills_score = 0.0
        must_have_found = 0
        nice_to_have_found = 0
        technical_found = 0
        
        # Must-have skills with context verification
        for skill in must_have_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_text:
                # Check for skill context
                skill_pos = resume_text.find(skill_lower)
                context = resume_text[max(0, skill_pos-30):skill_pos+len(skill_lower)+30]
                
                # Higher score if skill is mentioned with experience context
                if any(word in context for word in ["experience", "skilled", "proficient", "expert", "years"]):
                    skills_score += 0.12
                else:
                    skills_score += 0.08
                must_have_found += 1
        
        # Nice-to-have skills
        for skill in nice_to_have_skills:
            if skill.lower() in resume_text:
                skills_score += 0.06
                nice_to_have_found += 1
        
        # Technical skills with proficiency checking
        for skill in technical_skills:
            skill_lower = skill.lower()
            if skill_lower in resume_text:
                skill_pos = resume_text.find(skill_lower)
                context = resume_text[max(0, skill_pos-40):skill_pos+len(skill_lower)+40]
                
                # Higher score for proficiency indicators
                if any(word in context for word in ["advanced", "expert", "proficient", "certified", "experienced"]):
                    skills_score += 0.1
                else:
                    skills_score += 0.06
                technical_found += 1
        
        # Bonus for comprehensive skill coverage
        total_skills_available = len(must_have_skills) + len(nice_to_have_skills) + len(technical_skills)
        total_skills_found = must_have_found + nice_to_have_found + technical_found
        
        if total_skills_available > 0:
            coverage_ratio = total_skills_found / total_skills_available
            if coverage_ratio > 0.7:
                skills_score += 0.15  # Bonus for high coverage
            elif coverage_ratio > 0.5:
                skills_score += 0.1
        
        scores["skills_match"] = min(skills_score, 1.0)
        breakdown["skills"] = {
            "score": scores["skills_match"],
            "must_have_found": must_have_found,
            "must_have_total": len(must_have_skills),
            "nice_to_have_found": nice_to_have_found,
            "technical_found": technical_found,
            "coverage_ratio": total_skills_found / max(total_skills_available, 1)
        }
        
        # 3. EDUCATION & CERTIFICATION ANALYSIS (15% weight)
        education_score = 0.0
        
        # General education indicators
        education_terms = ["degree", "bachelor", "master", "diploma", "certificate", "graduate", "university", "college", "education"]
        education_found = sum(1 for term in education_terms if term in resume_text)
        
        if education_found > 0:
            education_score += 0.3
            
            # Check for relevant education
            for pref in education_preferences:
                if pref.lower() in resume_text:
                    education_score += 0.4
                    break
        
        # Check for certifications
        certification_found = 0
        for cert in certifications:
            if cert.lower() in resume_text:
                education_score += 0.2
                certification_found += 1
        
        # Professional development indicators
        development_terms = ["training", "course", "certification", "workshop", "seminar", "license"]
        development_found = sum(1 for term in development_terms if term in resume_text)
        if development_found > 0:
            education_score += 0.1 * min(development_found, 2)
        
        scores["education_fit"] = min(education_score, 1.0)
        breakdown["education"] = {
            "score": scores["education_fit"],
            "has_education": education_found > 0,
            "relevant_field": any(pref.lower() in resume_text for pref in education_preferences),
            "certifications_found": certification_found,
            "professional_development": development_found
        }
        
        # 4. CULTURAL ALIGNMENT & SOFT SKILLS (10% weight)
        cultural_score = 0.0
        
        # Cultural fit keywords with context
        cultural_matches = 0
        for keyword in cultural_fit_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in resume_text:
                # Check if used in positive context
                keyword_pos = resume_text.find(keyword_lower)
                context = resume_text[max(0, keyword_pos-20):keyword_pos+len(keyword_lower)+20]
                
                # Boost if in positive context
                positive_context = any(word in context for word in ["excellent", "strong", "proven", "demonstrated"])
                if positive_context:
                    cultural_score += 0.2
                else:
                    cultural_score += 0.15
                cultural_matches += 1
        
        # Hospitality mindset indicators
        hospitality_mindset = ["guest satisfaction", "customer satisfaction", "service excellence", "team player", 
                              "positive attitude", "professional", "reliable", "dedicated", "passionate"]
        mindset_found = sum(1 for term in hospitality_mindset if term in resume_text)
        if mindset_found > 0:
            cultural_score += 0.1 * min(mindset_found, 3)
        
        scores["cultural_alignment"] = min(cultural_score, 1.0)
        breakdown["cultural"] = {
            "score": scores["cultural_alignment"],
            "keywords_found": cultural_matches,
            "hospitality_mindset": mindset_found
        }
        
        # 5. COMMUNICATION & PROFESSIONALISM (5% weight)
        comm_score = 0.0
        
        # Resume quality indicators
        if len(resume_text) > 300:  # Reasonable detail
            comm_score += 0.3
        
        # Professional language and achievements
        professional_terms = ["responsible for", "achieved", "managed", "developed", "implemented", "improved", 
                             "coordinated", "supervised", "led", "organized", "maintained"]
        professional_found = sum(1 for term in professional_terms if term in resume_text)
        comm_score += min(professional_found * 0.05, 0.4)
        
        # Language skills (bonus for hospitality)
        language_terms = ["bilingual", "multilingual", "spanish", "french", "languages", "fluent"]
        if any(term in resume_text for term in language_terms):
            comm_score += 0.3
        
        scores["communication_quality"] = min(comm_score, 1.0)
        breakdown["communication"] = {
            "score": scores["communication_quality"],
            "resume_length": len(resume_text),
            "professional_terms": professional_found,
            "multilingual": any(term in resume_text for term in language_terms)
        }
        
        # 6. POSITION-SPECIFIC INTELLIGENCE (5% weight)
        position_score = 0.0
        
        # Direct position title matching with variations
        position_variations = [position.lower()]
        if " " in position.lower():
            position_variations.extend(position.lower().split())
        
        for variation in position_variations:
            if variation in resume_text:
                position_score += 0.4
        
        # Industry-specific terminology
        if "front desk" in position.lower():
            industry_terms = ["check-in", "check-out", "pms", "folio", "reservation", "concierge"]
        elif "housekeeping" in position.lower():
            industry_terms = ["room status", "amenities", "turnover", "inventory", "cleaning protocols"]
        elif "food" in position.lower() or "restaurant" in position.lower():
            industry_terms = ["pos system", "menu knowledge", "food safety", "allergies", "wine pairing"]
        else:
            industry_terms = ["hospitality", "guest services", "customer satisfaction"]
        
        industry_found = sum(1 for term in industry_terms if term in resume_text)
        position_score += min(industry_found * 0.1, 0.4)
        
        scores["position_specific"] = min(position_score, 1.0)
        breakdown["position_specific"] = {
            "score": scores["position_specific"],
            "direct_match": any(var in resume_text for var in position_variations),
            "industry_terms": industry_found
        }
        
        # CHECK FOR DISQUALIFYING FACTORS
        disqualification_penalty = 0.0
        disqualified_reasons = []
        
        for factor in disqualifying_factors:
            if factor.lower() in resume_text:
                disqualification_penalty += 0.3
                disqualified_reasons.append(factor)
        
        # CALCULATE FINAL WEIGHTED SCORE
        weights = {
            "experience_relevance": 0.35,
            "skills_match": 0.30,
            "education_fit": 0.15,
            "cultural_alignment": 0.10,
            "communication_quality": 0.05,
            "position_specific": 0.05
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights.keys())
        total_score = max(0.0, total_score - disqualification_penalty)
        
        # INTELLIGENT RECOMMENDATIONS
        if total_score >= 0.85:
            recommendation = "EXCEPTIONAL CANDIDATE"
            recommendation_reason = "Outstanding match across all criteria"
        elif total_score >= 0.70:
            recommendation = "HIGHLY RECOMMENDED"
            recommendation_reason = "Excellent fit with strong qualifications"
        elif total_score >= 0.55:
            recommendation = "RECOMMENDED"
            recommendation_reason = "Good candidate with solid experience"
        elif total_score >= 0.40:
            recommendation = "CONSIDER WITH INTERVIEW"
            recommendation_reason = "Potential candidate, interview to assess fit"
        elif total_score >= 0.25:
            recommendation = "MARGINAL CANDIDATE"
            recommendation_reason = "Significant gaps, consider only if limited options"
        else:
            recommendation = "NOT RECOMMENDED"
            recommendation_reason = "Poor fit for this position"
        
        # Enhanced category scores for backward compatibility
        category_scores = {
            "experience": scores["experience_relevance"],
            "skills": scores["skills_match"],
            "cultural_fit": scores["cultural_alignment"],
            "hospitality": (scores["experience_relevance"] + scores["cultural_alignment"]) / 2
        }
        
        return {
            "total_score": total_score,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
            "category_scores": category_scores,
            "detailed_scores": scores,
            "breakdown": breakdown,
            "disqualified_reasons": disqualified_reasons,
            "position": position,
            "scoring_methodology": "Enhanced AI Analysis v3.0 - Deep Content Analysis"
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
            
            # Detect gender
            gender_info = self._detect_gender(text, candidate_info.get("name", "Unknown"))
            
            scoring_result = self.calculate_enhanced_score(candidate_data, position)

            # Detect explicit role evidence (title or training/certification for the searched position)
            role_evidence_details = self._detect_explicit_role_evidence(text, position)
            
            # Compile final result
            result = {
                "file_name": file_path.name,
                "candidate_name": candidate_info.get("name", "Unknown"),
                "email": candidate_info.get("email", "Not found"),
                "phone": candidate_info.get("phone", "Not found"),
                "location": candidate_info.get("location", "Not specified"),
                "gender": gender_info["gender"],
                "gender_confidence": gender_info["confidence"],
                "gender_indicators": gender_info["indicators"],
                "total_score": scoring_result["total_score"],
                "recommendation": scoring_result["recommendation"],
                "category_scores": scoring_result["category_scores"],
                "breakdown": scoring_result["breakdown"],
                "skills_found": skill_analysis["skills"],
                "experience_years": experience_analysis["total_years"],
                "experience_quality": experience_analysis["experience_quality"],
                # Strict-role-match support
                "explicit_role_evidence": role_evidence_details.get("has_evidence", False),
                "role_evidence_details": role_evidence_details,
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ {file_path.name}: {result['total_score']:.1%} - {result['recommendation']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return None

    def _detect_explicit_role_evidence(self, text: str, position: str) -> Dict[str, Any]:
        """Detect explicit evidence that the resume matches the searched role.

        Evidence includes:
        - Exact/alias job titles (e.g., "Head Butler", "Personal Butler", "Butler Supervisor")
        - Training/certifications clearly tied to the role (e.g., "Butler certification", "Butler training")

        Returns dict with keys:
        { has_evidence: bool, exact_title_match: bool, matched_titles: List[str], training_hits: List[str], certification_hits: List[str] }
        """
        try:
            txt = (text or "").lower()
            # Avoid common false positives
            false_positive_blocks = [
                "butler university",  # educational institution unrelated to role
                "butler county",
            ]
            for fp in false_positive_blocks:
                txt = txt.replace(fp, "")

            pos = position.strip()
            pos_lower = pos.lower()

            # Collect role aliases and indicators from position intelligence
            position_data = self.position_intelligence.get(pos, {})
            indicators = set()
            if position_data:
                for key in ("experience_indicators",):
                    for v in position_data.get(key, []) or []:
                        indicators.add(v.lower())

            # Always include the position itself
            indicators.add(pos_lower)

            # Derive core tokens (e.g., for "Head Butler" -> ["head butler", "butler"]) but avoid single-token matches that are too generic unless paired
            core_tokens = [pos_lower]
            parts = [p for p in pos_lower.split() if p]
            if len(parts) > 1:
                # add the last word (often the core role, e.g., "butler") but only use it with context
                core_tokens.append(parts[-1])

            # Title patterns to check
            matched_titles: list[str] = []
            exact_title_match = False
            for phrase in sorted(indicators, key=len, reverse=True):
                if phrase and phrase in txt:
                    matched_titles.append(phrase)
                    if phrase == pos_lower:
                        exact_title_match = True

            # Additional contextual title cues
            contextual_cues = [
                "worked as ", "experience as ", "position: ", "role: ", "title: ", "promoted to ", "served as ", "hired as "
            ]
            context_hit = any(any(f"{cue}{phrase}" in txt for cue in contextual_cues) for phrase in indicators)

            # Training / certification evidence near role tokens
            train_words = ["training", "trained", "diploma", "certificate", "certification", "certified", "course"]
            certification_hits: list[str] = []
            training_hits: list[str] = []

            def _near_role(word: str) -> bool:
                # Simple proximity check around role tokens
                for token in core_tokens:
                    if not token:
                        continue
                    # require both appear within a short window
                    idx = txt.find(token)
                    while idx != -1:
                        start = max(0, idx - 60)
                        end = min(len(txt), idx + len(token) + 60)
                        window = txt[start:end]
                        if word in window and (token in (pos_lower, parts[-1] if parts else token)):
                            return True
                        idx = txt.find(token, idx + 1)
                return False

            # Populate training/cert hits
            for w in train_words:
                if _near_role(w):
                    if "cert" in w:
                        certification_hits.append(w)
                    else:
                        training_hits.append(w)

            has_evidence = bool(exact_title_match or context_hit or matched_titles or certification_hits or training_hits)

            return {
                "has_evidence": has_evidence,
                "exact_title_match": bool(exact_title_match),
                "matched_titles": sorted(set(matched_titles)),
                "training_hits": sorted(set(training_hits)),
                "certification_hits": sorted(set(certification_hits)),
            }
        except Exception:
            # Never break the pipeline on detection errors
            return {
                "has_evidence": False,
                "exact_title_match": False,
                "matched_titles": [],
                "training_hits": [],
                "certification_hits": [],
            }
    
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
    
    def _detect_gender(self, text: str, name: str) -> Dict[str, Any]:
        """Intelligently detect candidate gender from resume content."""
        gender_info = {
            "gender": "Unknown",
            "confidence": 0.0,
            "indicators": []
        }
        
        text_lower = text.lower()
        name_lower = name.lower()
        
        # Common male names (more comprehensive list)
        male_names = {
            'john', 'james', 'robert', 'michael', 'william', 'david', 'richard', 'joseph', 'thomas', 'christopher',
            'charles', 'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua',
            'kenneth', 'kevin', 'brian', 'george', 'timothy', 'ronald', 'jason', 'edward', 'jeffrey', 'ryan',
            'jacob', 'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott', 'brandon',
            'benjamin', 'samuel', 'gregory', 'alexander', 'patrick', 'frank', 'raymond', 'jack', 'dennis', 'jerry',
            'alex', 'jose', 'henry', 'douglas', 'peter', 'zachary', 'noah', 'carl', 'arthur', 'gerald',
            'wayne', 'harold', 'ralph', 'louis', 'philip', 'bobby', 'russell', 'craig', 'alan', 'sean',
            'juan', 'luis', 'carlos', 'miguel', 'antonio', 'angel', 'francisco', 'victor', 'jesus', 'salvador',
            'adam', 'nathan', 'aaron', 'kyle', 'jose', 'manuel', 'edgar', 'fernando', 'mario', 'ricardo'
        }
        
        # Common female names (more comprehensive list)
        female_names = {
            'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
            'nancy', 'lisa', 'betty', 'helen', 'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle',
            'laura', 'sarah', 'kimberly', 'deborah', 'dorothy', 'lisa', 'nancy', 'karen', 'betty', 'helen',
            'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle', 'laura', 'emily', 'kimberly', 'deborah',
            'dorothy', 'amy', 'angela', 'ashley', 'brenda', 'emma', 'olivia', 'cynthia', 'marie', 'janet',
            'catherine', 'frances', 'christine', 'virginia', 'samantha', 'debra', 'rachel', 'carolyn', 'janet',
            'virginia', 'maria', 'heather', 'diane', 'julie', 'joyce', 'victoria', 'kelly', 'christina', 'joan',
            'evelyn', 'lauren', 'judith', 'megan', 'cheryl', 'andrea', 'hannah', 'jacqueline', 'martha', 'gloria',
            'teresa', 'sara', 'janice', 'marie', 'julia', 'kathryn', 'anna', 'rose', 'grace', 'sophia',
            'isabella', 'ava', 'mia', 'charlotte', 'abigail', 'ella', 'madison', 'scarlett', 'victoria', 'aria'
        }
        
        # Gender-specific pronouns and references
        male_pronouns = ['he', 'him', 'his', 'himself', 'mr', 'mr.', 'mister']
        female_pronouns = ['she', 'her', 'hers', 'herself', 'ms', 'ms.', 'mrs', 'mrs.', 'miss', 'missus']
        
        # Professional titles that can indicate gender
        male_titles = ['mr', 'mister', 'sir', 'king', 'lord', 'duke', 'prince', 'baron', 'gentleman']
        female_titles = ['ms', 'mrs', 'miss', 'madam', 'lady', 'queen', 'duchess', 'princess', 'baroness']
        
        # Military/professional gender indicators
        male_military = ['seaman', 'airman', 'fireman', 'policeman', 'businessman', 'salesman', 'chairman']
        female_military = ['seawoman', 'airwoman', 'firewoman', 'policewoman', 'businesswoman', 'saleswoman', 'chairwoman']
        
        # Sports and activities with gender tendencies (careful with stereotypes)
        male_sports = ['football', 'rugby', 'wrestling', 'boxing', 'ice hockey', 'baseball']
        female_sports = ['softball', 'field hockey', 'synchronized swimming', 'rhythmic gymnastics']
        
        confidence_score = 0.0
        indicators = []
        
        # Check first name
        first_name = name_lower.split()[0] if name_lower.split() else ""
        if first_name in male_names:
            confidence_score += 0.7
            indicators.append(f"Male name: {first_name}")
            gender_info["gender"] = "Male"
        elif first_name in female_names:
            confidence_score += 0.7
            indicators.append(f"Female name: {first_name}")
            gender_info["gender"] = "Female"
        
        # Check pronouns in text
        male_pronoun_count = sum(1 for pronoun in male_pronouns if pronoun in text_lower)
        female_pronoun_count = sum(1 for pronoun in female_pronouns if pronoun in text_lower)
        
        if male_pronoun_count > female_pronoun_count and male_pronoun_count > 0:
            confidence_score += 0.3
            indicators.append(f"Male pronouns found: {male_pronoun_count}")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Male"
        elif female_pronoun_count > male_pronoun_count and female_pronoun_count > 0:
            confidence_score += 0.3
            indicators.append(f"Female pronouns found: {female_pronoun_count}")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Female"
        
        # Check titles
        for title in male_titles:
            if title in text_lower:
                confidence_score += 0.5
                indicators.append(f"Male title: {title}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Male"
                break
        
        for title in female_titles:
            if title in text_lower:
                confidence_score += 0.5
                indicators.append(f"Female title: {title}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Female"
                break
        
        # Check professional gender-specific terms
        for term in male_military:
            if term in text_lower:
                confidence_score += 0.2
                indicators.append(f"Male professional term: {term}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Male"
        
        for term in female_military:
            if term in text_lower:
                confidence_score += 0.2
                indicators.append(f"Female professional term: {term}")
                if gender_info["gender"] == "Unknown":
                    gender_info["gender"] = "Female"
        
        # Check for gendered organizations (fraternities vs sororities)
        if any(word in text_lower for word in ['fraternity', 'brotherhood', 'alpha phi alpha', 'kappa alpha psi']):
            confidence_score += 0.3
            indicators.append("Male organization membership")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Male"
        
        if any(word in text_lower for word in ['sorority', 'sisterhood', 'alpha kappa alpha', 'delta sigma theta']):
            confidence_score += 0.3
            indicators.append("Female organization membership")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Female"
        
        # Check for gender-specific life events or contexts
        if any(phrase in text_lower for phrase in ['maternity leave', 'pregnancy', 'maiden name']):
            confidence_score += 0.4
            indicators.append("Female life event reference")
            if gender_info["gender"] == "Unknown":
                gender_info["gender"] = "Female"
        
        # Final confidence adjustment
        gender_info["confidence"] = min(confidence_score, 1.0)
        gender_info["indicators"] = indicators
        
        # If confidence is too low, mark as unknown
        if gender_info["confidence"] < 0.3:
            gender_info["gender"] = "Unknown"
        
        return gender_info
    
    def screen_candidates(self, position: str, max_candidates: Optional[int] = None, require_explicit_role: bool = True) -> List[Dict[str, Any]]:
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

        # Optional strict filter: keep only candidates with explicit role/title/training evidence
        if require_explicit_role:
            with_evidence = [c for c in candidates if c.get("explicit_role_evidence", False)]
            if with_evidence:
                logger.info(f"🔎 Strict role filter retained {len(with_evidence)} of {len(candidates)} candidates")
                candidates = with_evidence
            else:
                logger.info("🔎 Strict role filter found no explicit matches; returning unfiltered results")
        
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
