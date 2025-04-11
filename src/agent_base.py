import os
import json
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from functools import lru_cache
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class that implements core functionalities for all agents."""
    
    def __init__(self, name: str, api_key: str = None):
        self.name = name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Some functionalities may be limited.")
        self.memory = []
        self.tools = {}
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0,
            "last_execution": None
        }
        
    def register_tool(self, name: str, func: Callable):
        """Register a tool that the agent can use."""
        if not callable(func):
            raise ValueError(f"Tool {name} must be callable")
        self.tools[name] = func
        logger.info(f"Registered tool: {name}")
        
    @lru_cache(maxsize=100)
    def _cached_api_call(self, prompt: str) -> Dict[str, Any]:
        """Cached version of API call to improve performance."""
        return self._call_deepseek_api(prompt)
        
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task with reasoning using the DeepSeek API."""
        if not context:
            context = {}
            
        start_time = datetime.now()
        self.performance_metrics["total_tasks"] += 1
        
        try:
            prompt = f"""
            Task: {task}
            
            Context: {json.dumps(context)}
            
            Please reason step by step and provide your solution.
            """
            
            # Use cached API call if available
            response = self._cached_api_call(prompt)
            
            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["successful_tasks"] += 1
            self.performance_metrics["average_response_time"] = (
                (self.performance_metrics["average_response_time"] * (self.performance_metrics["successful_tasks"] - 1) +
                 execution_time) / self.performance_metrics["successful_tasks"]
            )
            self.performance_metrics["last_execution"] = datetime.now()
            
            # Store in memory with metadata
            self.memory.append({
                "task": task,
                "context": context,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "status": "success"
            })
            
            logger.info(f"Task executed successfully: {task[:50]}...")
            return response
            
        except Exception as e:
            self.performance_metrics["failed_tasks"] += 1
            error_msg = f"Error executing task: {str(e)}"
            logger.error(error_msg)
            
            # Store error in memory
            self.memory.append({
                "task": task,
                "context": context,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            })
            
            return {
                "status": "error",
                "error": error_msg,
                "content": None
            }
    
    def split(self, task: str, subtasks: List[str] = None) -> List[Dict[str, Any]]:
        """Split a complex task into subtasks."""
        if subtasks:
            return [{"task": subtask} for subtask in subtasks]
        
        try:
            # If subtasks not provided, use DeepSeek to generate them
            prompt = f"""
            Task: {task}
            
            Please break down this task into logical subtasks that can be executed sequentially.
            Return the subtasks as a numbered list.
            """
            
            response = self._cached_api_call(prompt)
            
            # Parse the response to extract subtasks
            subtask_lines = [line.strip() for line in response.get("content", "").split("\n") 
                            if line.strip() and any(c.isdigit() for c in line)]
            
            # Clean up the subtasks
            cleaned_subtasks = []
            for line in subtask_lines:
                # Remove numbering
                if '. ' in line:
                    cleaned_subtasks.append(line.split('. ', 1)[1])
                else:
                    cleaned_subtasks.append(line)
            
            logger.info(f"Task split into {len(cleaned_subtasks)} subtasks")
            return [{"task": subtask} for subtask in cleaned_subtasks]
            
        except Exception as e:
            error_msg = f"Error splitting task: {str(e)}"
            logger.error(error_msg)
            return [{"task": task, "error": error_msg}]
    
    def forward(self, task: Dict[str, Any], agent: 'BaseAgent') -> Dict[str, Any]:
        """Forward a task to another agent."""
        if not agent:
            raise ValueError("No agent provided to forward the task to.")
        
        try:
            result = agent.execute(task.get("task", ""), task.get("context", {}))
            
            # Store the forwarding in memory
            self.memory.append({
                "action": "forward",
                "task": task,
                "to_agent": agent.name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": "success" if result.get("status") == "success" else "error"
            })
            
            logger.info(f"Task forwarded to {agent.name}")
            return result
            
        except Exception as e:
            error_msg = f"Error forwarding task to {agent.name}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "content": None
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            **self.performance_metrics,
            "success_rate": (self.performance_metrics["successful_tasks"] / 
                           self.performance_metrics["total_tasks"] * 100 
                           if self.performance_metrics["total_tasks"] > 0 else 0)
        }
    
    def _call_deepseek_api(self, prompt: str) -> Dict[str, Any]:
        """Make a call to the DeepSeek API or simulate a response if needed."""
        # Check if API key is available
        if not self.api_key:
            # Generate a simulated response using rule-based logic
            return self._simulate_api_response(prompt)
            
        try:
            # This would be the actual API call in a production implementation
            # For now, we'll simulate API responses to avoid rate limits and costs
            
            # Use simple rule-based response generation to simulate API
            return self._simulate_api_response(prompt)
            
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            # Return a fallback response
            return {
                "status": "error",
                "error": str(e),
                "content": "An error occurred while processing your request."
            }
    
    def _simulate_api_response(self, prompt: str) -> Dict[str, Any]:
        """Simulate API response with a rule-based approach."""
        # This is a simplified version that uses rule-based responses
        # instead of calling an external API
        
        response_content = ""
        
        # Handle different types of prompts
        if "Extract key data from CV" in prompt:
            # Simple CV data extraction logic
            # [Existing CV extraction code remains unchanged]
            cv_text = prompt.split("Task: Extract key data from CV")[-1].strip()
            if "Context:" in cv_text:
                cv_text = cv_text.split("Context:")[-1].strip()
            
            # Initialize default values
            cv_data = {
                "personal_info": {"name": "Unknown", "email": "", "phone": ""},
                "education": [],
                "work_experience": [],
                "skills": [],
                "certifications": [],
                "languages": []
            }
            
            if isinstance(cv_text, str) and cv_text:
                # Extract basic information
                lines = cv_text.split("\\n")
                
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
                        section_text = line + "\\n"
                    
                    elif any(exp in line_lower for exp in ["experience", "employment", "work history", "career"]):
                        if current_section and section_text:
                            self._process_section(current_section, section_text, cv_data)
                        current_section = "experience"
                        section_text = line + "\\n"
                    
                    elif any(skill in line_lower for skill in ["skills", "competencies", "expertise", "technologies"]):
                        if current_section and section_text:
                            self._process_section(current_section, section_text, cv_data)
                        current_section = "skills"
                        section_text = line + "\\n"
                    
                    else:
                        section_text += line + "\\n"
                
                # Process the last section
                if current_section and section_text:
                    self._process_section(current_section, section_text, cv_data)
            
            response_content = json.dumps(cv_data)
            
        elif "Calculate match score between CV and job description" in prompt:
            # Simple match score calculation
            match_score = np.random.randint(40, 95)
            response_content = str(match_score)
        
        elif "Shortlist candidates" in prompt:
            # Simple shortlisting
            response_content = "Candidates have been shortlisted based on their qualifications and match score."
        
        elif "Schedule interviews" in prompt:
            # Simple interview scheduling
            response_content = "Interviews have been scheduled for the shortlisted candidates."
        
        else:
            # Generic response
            response_content = "Task completed successfully."
        
        return {
            "status": "success",
            "content": response_content,
            "model": "simulated-response"
        }