from src.agent_base import BaseAgent
from typing import Dict, Any, List
import re

class RecruitingAgent(BaseAgent):
    """Agent responsible for CV analysis and matching with job descriptions."""
    
    def __init__(self, api_key: str = None):
        super().__init__(name="Recruiting Agent", api_key=api_key)
        self._register_tools()
        
    def _register_tools(self):
        """Register tools specific to recruiting."""
        self.register_tool("extract_cv_data", self._extract_cv_data)
        self.register_tool("calculate_match_score", self._calculate_match_score)
        
    def process_cv(self, cv_text: str) -> Dict[str, Any]:
        """Extract key information from a CV."""
        return self.execute("Extract key data from CV", {"cv_text": cv_text})
    
    def match_cv_to_jd(self, cv_data: Dict[str, Any], jd_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Match a CV to a job description and calculate a match score."""
        context = {
            "cv_data": cv_data,
            "jd_summary": jd_summary
        }
        
        return self.execute("Calculate match score between CV and job description", context)
    
    def _extract_cv_data(self, cv_text: str) -> Dict[str, Any]:
        """Extract structured data from a CV."""
        # Initialize default values
        cv_data = {
            "personal_info": {"name": "Unknown", "email": "", "phone": ""},
            "education": [],
            "work_experience": [],
            "skills": [],
            "certifications": [],
            "languages": []
        }
        
        if not cv_text or not isinstance(cv_text, str):
            return cv_data
            
        # Split text into lines and clean them
        lines = [line.strip() for line in cv_text.split('\n') if line.strip()]
        
        # Try to find name in first few lines
        for line in lines[:5]:
            if len(line.split()) >= 2 and not any(word in line.lower() for word in ["email", "phone", "address", "linkedin"]):
                cv_data["personal_info"]["name"] = line
                break
        
        # Extract email and phone using regex
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        phone_pattern = r'(\+\d{1,3}[-.]?)?\s*\(?\d{3}\)?[-.]?\s*\d{3}[-.]?\s*\d{4}'
        
        for line in lines:
            # Extract email
            email_match = re.search(email_pattern, line)
            if email_match and not cv_data["personal_info"]["email"]:
                cv_data["personal_info"]["email"] = email_match.group(0)
                
            # Extract phone
            phone_match = re.search(phone_pattern, line)
            if phone_match and not cv_data["personal_info"]["phone"]:
                cv_data["personal_info"]["phone"] = phone_match.group(0)
        
        # Process the text line by line to extract other information
        current_section = None
        section_text = ""
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect section headers
            if any(edu in line_lower for edu in ["education", "academic", "qualification", "degree"]):
                if current_section and section_text:
                    self._process_section(current_section, section_text, cv_data)
                current_section = "education"
                section_text = line + "\n"
            
            elif any(exp in line_lower for exp in ["experience", "employment", "work history", "career"]):
                if current_section and section_text:
                    self._process_section(current_section, section_text, cv_data)
                current_section = "work_experience"
                section_text = line + "\n"
            
            elif any(skill in line_lower for skill in ["skills", "expertise", "proficiency", "competencies"]):
                if current_section and section_text:
                    self._process_section(current_section, section_text, cv_data)
                current_section = "skills"
                section_text = line + "\n"
            
            elif any(cert in line_lower for cert in ["certification", "certificate", "diploma"]):
                if current_section and section_text:
                    self._process_section(current_section, section_text, cv_data)
                current_section = "certifications"
                section_text = line + "\n"
            
            elif any(lang in line_lower for lang in ["language", "linguistic"]):
                if current_section and section_text:
                    self._process_section(current_section, section_text, cv_data)
                current_section = "languages"
                section_text = line + "\n"
            
            else:
                if current_section:
                    section_text += line + "\n"
        
        # Process the last section
        if current_section and section_text:
            self._process_section(current_section, section_text, cv_data)
        
        # If no skills were found in a dedicated section, try to extract them from the entire text
        if not cv_data["skills"]:
            common_skills = [
                "python", "java", "javascript", "typescript", "html", "css", "sql",
                "react", "angular", "vue", "node", "express", "django", "flask",
                "aws", "azure", "gcp", "docker", "kubernetes", "git", "agile",
                "scrum", "jira", "machine learning", "ai", "data science",
                "project management", "leadership", "communication"
            ]
            text_lower = cv_text.lower()
            found_skills = set()
            
            for skill in common_skills:
                if skill in text_lower:
                    found_skills.add(skill)
            
            cv_data["skills"] = list(found_skills)
        
        return cv_data
        
    def _process_section(self, section: str, text: str, cv_data: Dict[str, Any]):
        """Process a section of text and update the CV data."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if section == "education":
            # Skip the header line
            for line in lines[1:]:
                if len(line) > 10:  # Avoid very short lines
                    cv_data["education"].append(line)
        
        elif section == "work_experience":
            # Skip the header line
            current_experience = []
            for line in lines[1:]:
                if len(line) > 10:  # Avoid very short lines
                    if re.search(r'\d{4}', line):  # Line contains a year
                        if current_experience:
                            cv_data["work_experience"].append(" ".join(current_experience))
                            current_experience = []
                    current_experience.append(line)
            if current_experience:
                cv_data["work_experience"].append(" ".join(current_experience))
        
        elif section == "skills":
            # Skip the header line
            for line in lines[1:]:
                skills = re.split(r'[,;]', line)
                cv_data["skills"].extend([skill.strip() for skill in skills if skill.strip()])
        
        elif section == "certifications":
            # Skip the header line
            for line in lines[1:]:
                if len(line) > 5:  # Avoid very short lines
                    cv_data["certifications"].append(line)
        
        elif section == "languages":
            # Skip the header line
            for line in lines[1:]:
                languages = re.split(r'[,;]', line)
                cv_data["languages"].extend([lang.strip() for lang in languages if lang.strip()])
    
    def _calculate_match_score(self, cv_data: Dict[str, Any], jd_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a match score between a CV and a job description."""
        prompt = f"""
        Calculate a match score between the CV and the job description.
        
        Job Description Summary:
        {jd_summary}
        
        CV Data:
        {cv_data}
        
        For each of the following categories, assign a score from 0-100 based on how well the CV matches the job requirements:
        1. Skills Match
        2. Experience Match
        3. Qualifications Match
        4. Overall Match
        
        Provide a brief explanation for each score.
        """
        
        response = self._call_deepseek_api(prompt)
        content = response.get("content", "")
        
        # Parse the scores from the response
        scores = {
            "skills_match": 0,
            "experience_match": 0,
            "qualifications_match": 0,
            "overall_match": 0,
            "explanation": content
        }
        
        # Extract scores using regex
        skills_match = re.search(r'Skills Match:?\s*(\d+)', content, re.IGNORECASE)
        if skills_match:
            scores["skills_match"] = int(skills_match.group(1))
            
        experience_match = re.search(r'Experience Match:?\s*(\d+)', content, re.IGNORECASE)
        if experience_match:
            scores["experience_match"] = int(experience_match.group(1))
            
        qualifications_match = re.search(r'Qualifications Match:?\s*(\d+)', content, re.IGNORECASE)
        if qualifications_match:
            scores["qualifications_match"] = int(qualifications_match.group(1))
            
        overall_match = re.search(r'Overall Match:?\s*(\d+)', content, re.IGNORECASE)
        if overall_match:
            scores["overall_match"] = int(overall_match.group(1))
        
        return scores