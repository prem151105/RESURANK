from src.agent_base import BaseAgent
from typing import Dict, Any, List
import re

class JDSummarizerAgent(BaseAgent):
    """Agent responsible for reading and summarizing job descriptions."""
    
    def __init__(self, api_key: str = None):
        super().__init__(name="JD Summarizer", api_key=api_key)
        self._register_tools()
        
    def _register_tools(self):
        """Register tools specific to JD summarization."""
        self.register_tool("extract_skills", self._extract_skills)
        self.register_tool("extract_experience", self._extract_experience)
        self.register_tool("extract_qualifications", self._extract_qualifications)
        self.register_tool("extract_responsibilities", self._extract_responsibilities)
        
    def summarize_jd(self, jd_text: str) -> Dict[str, Any]:
        """Summarize a job description into structured data."""
        if not jd_text:
            return {
                "skills": [],
                "experience": {"min_years": 0, "max_years": None, "description": ""},
                "qualifications": [],
                "responsibilities": []
            }
        
        # Extract each component directly
        skills = self._extract_skills(jd_text)
        experience = self._extract_experience(jd_text)
        qualifications = self._extract_qualifications(jd_text)
        responsibilities = self._extract_responsibilities(jd_text)
        
        # Combine results
        summary = {
            "skills": skills if isinstance(skills, list) else [],
            "experience": experience if isinstance(experience, dict) else {"min_years": 0, "max_years": None, "description": ""},
            "qualifications": qualifications if isinstance(qualifications, list) else [],
            "responsibilities": responsibilities if isinstance(responsibilities, list) else []
        }
        
        return summary
    
    def _parse_list_from_response(self, response: str) -> List[str]:
        """Parse a list from the API response."""
        items = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line)):
                # Remove the bullet point or numbering
                item = re.sub(r'^[-*]\s*|^\d+\.\s*', '', line).strip()
                if item:
                    items.append(item)
        
        # If no items were found with bullet points, try to split by commas
        if not items and ',' in response:
            items = [item.strip() for item in response.split(',') if item.strip()]
        
        # If still no items, just return the whole response as one item
        if not items:
            items = [response.strip()]
            
        return items
    
    def _parse_experience(self, response: str) -> Dict[str, Any]:
        """Parse experience requirements from the response."""
        years_pattern = r'(\d+)[\+\-]?\s*(?:to\s*)?(\d+)?\s*years?'
        years_match = re.search(years_pattern, response, re.IGNORECASE)
        
        experience = {
            "min_years": 0,
            "max_years": None,
            "description": response.strip()
        }
        
        if years_match:
            min_years = int(years_match.group(1))
            max_years = int(years_match.group(2)) if years_match.group(2) else None
            
            experience["min_years"] = min_years
            experience["max_years"] = max_years
            
        return experience
    
    # Tool methods
    def _extract_skills(self, jd_text: str) -> List[str]:
        """Extract skills from job description."""
        prompt = f"""
        Extract all required skills from the following job description:
        
        {jd_text}
        
        Return only the list of skills, one per line with bullet points.
        """
        response = self._call_deepseek_api(prompt)
        return self._parse_list_from_response(response.get("content", ""))
    
    def _extract_experience(self, jd_text: str) -> Dict[str, Any]:
        """Extract experience requirements from job description."""
        prompt = f"""
        Extract the required work experience from the following job description:
        
        {jd_text}
        
        Specify the minimum years of experience required and any specific domain experience.
        """
        response = self._call_deepseek_api(prompt)
        return self._parse_experience(response.get("content", ""))
    
    def _extract_qualifications(self, jd_text: str) -> List[str]:
        """Extract qualifications from job description."""
        prompt = f"""
        Extract all required qualifications and education requirements from the following job description:
        
        {jd_text}
        
        Return only the list of qualifications, one per line with bullet points.
        """
        response = self._call_deepseek_api(prompt)
        return self._parse_list_from_response(response.get("content", ""))
    
    def _extract_responsibilities(self, jd_text: str) -> List[str]:
        """Extract job responsibilities from job description."""
        prompt = f"""
        Extract all job responsibilities and duties from the following job description:
        
        {jd_text}
        
        Return only the list of responsibilities, one per line with bullet points.
        """
        response = self._call_deepseek_api(prompt)
        return self._parse_list_from_response(response.get("content", ""))