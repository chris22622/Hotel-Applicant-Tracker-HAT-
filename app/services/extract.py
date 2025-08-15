"""Text extraction and structured data parsing."""
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import spacy


# Load spaCy model (will be loaded by worker)
_nlp = None


def get_nlp():
    """Get spaCy NLP model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_email(text: str) -> Optional[str]:
    """Extract email address from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    # Various phone patterns
    patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (123) 456-7890
        r'\b\d{10}\b',  # 1234567890
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    return None


def extract_name(text: str) -> Optional[str]:
    """Extract name from resume text."""
    nlp = get_nlp()
    doc = nlp(text[:500])  # Check first 500 chars
    
    # Look for PERSON entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if persons:
        # Return the first person name that looks like a full name
        for person in persons:
            if len(person.split()) >= 2:  # Has first and last name
                return person
    
    # Fallback: look for name patterns at the beginning
    lines = text.split('\n')[:5]  # First 5 lines
    for line in lines:
        line = line.strip()
        # Skip empty lines and lines with common resume elements
        if not line or any(word in line.lower() for word in ['email', 'phone', 'address', 'objective']):
            continue
        
        # If line has 2-4 words and no numbers, might be a name
        words = line.split()
        if 2 <= len(words) <= 4 and not any(char.isdigit() for char in line):
            return line
    
    return None


def extract_work_experience(text: str) -> List[Dict[str, Any]]:
    """Extract work experience from resume text."""
    experiences = []
    
    # Simple pattern matching for job titles and companies
    # This is a basic implementation - could be enhanced with ML
    
    lines = text.split('\n')
    current_exp = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Look for date patterns (2020-2022, 2020-Present, etc.)
        date_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|Present|Current|Now)'
        date_match = re.search(date_pattern, line, re.IGNORECASE)
        
        if date_match:
            # This line likely contains dates, check surrounding lines for job info
            start_year = int(date_match.group(1))
            end_year = 2024 if date_match.group(2).lower() in ['present', 'current', 'now'] else int(date_match.group(2))
            
            # Look for job title in current or previous line
            title_line = line
            if i > 0:
                prev_line = lines[i-1].strip()
                if prev_line and not re.search(r'\d{4}', prev_line):
                    title_line = prev_line
            
            # Extract title and company
            # Remove dates from title line
            title_clean = re.sub(r'\d{4}[-–]\d{4}|\d{4}[-–]Present|\d{4}[-–]Current', '', title_line).strip()
            
            # Try to split title and company
            parts = title_clean.split(' at ')
            if len(parts) == 2:
                title, company = parts[0].strip(), parts[1].strip()
            elif title_clean:
                title = title_clean
                company = "Unknown"
            else:
                continue
            
            experiences.append({
                'title': title,
                'company': company,
                'start_date': date(start_year, 1, 1),
                'end_date': date(end_year, 12, 31) if end_year != 2024 else None,
                'responsibilities': '',
                'skills': []
            })
    
    return experiences


def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    """Extract skills from text based on skills list."""
    found_skills = []
    text_lower = text.lower()
    
    for skill in skills_list:
        skill_lower = skill.lower()
        if skill_lower in text_lower:
            found_skills.append(skill)
    
    return found_skills


def calculate_years_experience(experiences: List[Dict[str, Any]]) -> float:
    """Calculate total years of experience."""
    total_months = 0
    current_year = datetime.now().year
    
    for exp in experiences:
        start_date = exp.get('start_date')
        end_date = exp.get('end_date') or date(current_year, 12, 31)
        
        if start_date and end_date:
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += max(0, months)
    
    return total_months / 12.0


def extract_structured(text: str) -> Dict[str, Any]:
    """Extract structured data from resume text."""
    # Basic skills list (could be loaded from YAML)
    basic_skills = [
        'Python', 'Java', 'JavaScript', 'SQL', 'HTML', 'CSS', 'React', 'Node.js',
        'AWS', 'Docker', 'Kubernetes', 'Git', 'Linux', 'Windows', 'Excel',
        'Project Management', 'Leadership', 'Communication', 'Problem Solving',
        'Customer Service', 'Sales', 'Marketing', 'Accounting', 'Finance'
    ]
    
    # Extract basic information
    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    experiences = extract_work_experience(text)
    skills = extract_skills(text, basic_skills)
    years_total = calculate_years_experience(experiences)
    
    # Extract current role info
    current_title = None
    current_company = None
    if experiences:
        # Most recent experience (first in list)
        latest = experiences[0]
        if latest.get('end_date') is None:  # Current role
            current_title = latest.get('title')
            current_company = latest.get('company')
    
    return {
        'name': name,
        'email': email,
        'phone': phone,
        'experiences': experiences,
        'skills': skills,
        'years_total': years_total,
        'current_title': current_title,
        'current_company': current_company,
        'education_level': None,  # TODO: implement education extraction
    }
