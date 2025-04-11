from src.agent_base import BaseAgent
from typing import Dict, Any, List

class ShortlistingAgent(BaseAgent):
    """Agent responsible for shortlisting candidates based on match scores."""
    
    def __init__(self, threshold: int = 50, api_key: str = None):
        super().__init__(name="Shortlisting Agent", api_key=api_key)
        self.threshold = threshold
        
    def shortlist_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Shortlist candidates based on their match scores."""
        if not candidates:
            return {
                "shortlisted": [],
                "rejected": [],
                "total_candidates": 0,
                "shortlisted_count": 0,
                "rejected_count": 0
            }
        
        shortlisted = []
        rejected = []
        
        for candidate in candidates:
            # Extract the overall match score from the candidate data
            if isinstance(candidate, dict):
                score = None
                if "overall_match" in candidate:
                    score = candidate["overall_match"]
                elif "score" in candidate:
                    if isinstance(candidate["score"], dict):
                        score = candidate["score"].get("overall_match", 0)
                    else:
                        score = candidate["score"]
                elif "match_score" in candidate:
                    if isinstance(candidate["match_score"], dict):
                        score = candidate["match_score"].get("overall_match", 0)
                    else:
                        score = candidate["match_score"]
                
                if score is None:
                    score = 0
                
                # Convert score to float if it's a string
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 0
                
                # Compare score with threshold
                if isinstance(score, (int, float)) and score >= self.threshold:
                    shortlisted.append(candidate)
                else:
                    rejected.append(candidate)
            else:
                rejected.append(candidate)
        
        return {
            "shortlisted": shortlisted,
            "rejected": rejected,
            "total_candidates": len(candidates),
            "shortlisted_count": len(shortlisted),
            "rejected_count": len(rejected)
        }
    
    def generate_shortlist_report(self, shortlisted: List[Dict[str, Any]], 
                                 rejected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a report on the shortlisting process."""
        context = {
            "shortlisted_count": len(shortlisted),
            "rejected_count": len(rejected),
            "threshold": self.threshold,
            "shortlisted": shortlisted,
            "rejected": rejected
        }
        
        return self.execute("Generate a shortlisting report", context)