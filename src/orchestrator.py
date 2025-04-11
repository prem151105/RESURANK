from src.jd_summarizer import JDSummarizerAgent
from src.recruiting_agent import RecruitingAgent
from src.shortlisting_agent import ShortlistingAgent
from src.interview_scheduler import InterviewSchedulerAgent
from typing import Dict, Any, List
import os
import json
import datetime

class RecruitmentOrchestrator:
    """Main orchestrator that coordinates all recruitment agents."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        
        # Initialize all agents
        self.jd_summarizer = JDSummarizerAgent(api_key=self.api_key)
        self.recruiter = RecruitingAgent(api_key=self.api_key)
        self.shortlister = ShortlistingAgent(threshold=80, api_key=self.api_key)
        self.scheduler = InterviewSchedulerAgent(api_key=self.api_key)
        
    def process_job_description(self, jd_text: str) -> Dict[str, Any]:
        """Process a job description and return the summary."""
        return self.jd_summarizer.summarize_jd(jd_text)
    
    def process_candidates(self, jd_summary: Dict[str, Any], cv_texts: List[str]) -> Dict[str, Any]:
        """Process a list of candidate CVs and match them against the job description."""
        candidates = []
        
        for cv_text in cv_texts:
            # Extract CV data
            cv_data = self.recruiter.process_cv(cv_text)
            
            # Match CV to JD
            match_score = self.recruiter.match_cv_to_jd(cv_data, jd_summary)
            
            candidates.append({
                "cv_text": cv_text,
                "cv_data": cv_data,
                "match_score": match_score
            })
        
        return candidates
    
    def shortlist_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Shortlist candidates based on their match scores."""
        shortlisted = self.shortlister.shortlist_candidates(candidates)
        
        # Identify rejected candidates
        rejected = [c for c in candidates if c not in shortlisted]
        
        # Generate report
        report = self.shortlister.generate_shortlist_report(shortlisted, rejected)
        
        return {
            "shortlisted": shortlisted,
            "rejected": rejected,
            "report": report
        }
    
    def schedule_interviews(self, shortlisted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Schedule interviews for shortlisted candidates."""
        # Generate some available slots (in a real system, these would come from a calendar)
        available_slots = self._generate_available_slots(len(shortlisted))
        
        # Schedule interviews
        scheduled_interviews = self.scheduler.schedule_interviews(shortlisted, available_slots)
        
        return scheduled_interviews
    
    def _generate_available_slots(self, num_candidates: int) -> List[Dict[str, Any]]:
        """Generate available interview slots."""
        slots = []
        
        # Start from tomorrow
        start_date = datetime.datetime.now() + datetime.timedelta(days=1)
        
        # Generate slots for the next 5 days
        for i in range(5):
            date = start_date + datetime.timedelta(days=i)
            
            # Generate 3 slots per day
            for hour in [10, 13, 15]:
                slots.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "time": f"{hour}:00",
                    "format": "Video Call"
                })
        
        return slots
    
    def run_full_process(self, jd_text: str, cv_texts: List[str]) -> Dict[str, Any]:
        """Run the full recruitment process from JD to interview scheduling."""
        # Process job description
        jd_summary = self.process_job_description(jd_text)
        
        # Process candidates
        candidates = self.process_candidates(jd_summary, cv_texts)
        
        # Shortlist candidates
        shortlisting_result = self.shortlist_candidates(candidates)
        
        # Schedule interviews
        scheduled_interviews = self.schedule_interviews(shortlisting_result["shortlisted"])
        
        return {
            "jd_summary": jd_summary,
            "candidates": candidates,
            "shortlisting": shortlisting_result,
            "scheduled_interviews": scheduled_interviews
        }