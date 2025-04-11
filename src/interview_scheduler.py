from src.agent_base import BaseAgent
from typing import Dict, Any, List
import datetime

class InterviewSchedulerAgent(BaseAgent):
    """Agent responsible for scheduling interviews with shortlisted candidates."""
    
    def __init__(self, api_key: str = None):
        super().__init__(name="Interview Scheduler", api_key=api_key)
        self._register_tools()
        
    def _register_tools(self):
        """Register tools specific to interview scheduling."""
        self.register_tool("generate_email", self._generate_email)
        
    def schedule_interviews(self, shortlisted: List[Dict[str, Any]], 
                           available_slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Schedule interviews for shortlisted candidates."""
        if not shortlisted or not available_slots:
            raise ValueError("Both shortlisted candidates and available slots are required")
            
        scheduled_interviews = []
        
        # Extract candidates from the shortlisting result
        candidates = shortlisted.get("shortlisted", [])
        if not candidates:
            raise ValueError("No shortlisted candidates found")
        
        for i, candidate in enumerate(candidates):
            if i >= len(available_slots):
                break  # No more slots available
                
            # Assign a slot based on candidate index
            slot = available_slots[i]
            
            # Extract candidate name from CV text
            cv_text = candidate.get("cv_text", "")
            # Simple extraction of name from the first line of CV
            candidate_name = cv_text.split("\n")[0] if cv_text else "Candidate"
            
            # Generate email for the candidate
            email = self._generate_email(
                candidate_name,
                slot.get("date"),
                slot.get("time"),
                slot.get("format", "Video Call")
            )
            
            scheduled_interviews.append({
                "candidate_name": candidate_name,
                "candidate_score": candidate.get("score", 0),
                "slot": slot,
                "email": email
            })
        
        return scheduled_interviews
    
    def _generate_email(self, candidate_name: str, date: str, time: str, format: str) -> str:
        """Generate an interview invitation email."""
        prompt = f"""
        Generate a professional and friendly email to invite {candidate_name} for an interview.
        
        Interview details:
        - Date: {date}
        - Time: {time}
        - Format: {format}
        
        The email should:
        1. Be personalized
        2. Express enthusiasm about the candidate's application
        3. Clearly state the interview details
        4. Ask for confirmation
        5. Provide contact information for questions
        
        Write the complete email with subject line.
        """
        
        response = self._call_deepseek_api(prompt)
        return response.get("content", "")