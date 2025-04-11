# AgentNet Integration with Recruiting Agents
# This module connects the AgentNet framework with the recruiting workflow

from typing import Dict, Any, List, Optional
import numpy as np

# Import AgentNet components - Fix the import paths
from agentnet.main import AgentNet
from agentnet.agent import Agent

# Import recruiting components
from src.agent_base import BaseAgent
from src.jd_summarizer import JDSummarizerAgent
from src.recruiting_agent import RecruitingAgent
from src.shortlisting_agent import ShortlistingAgent
from src.interview_scheduler import InterviewSchedulerAgent


class RecruitingTask:
    """
    A task wrapper for recruitment tasks that is compatible with AgentNet.
    """
    
    def __init__(self, task_id: str, description: str, task_type: str, 
                 data: Dict[str, Any], complexity: float = 5.0):
        self.id = task_id
        self.description = description
        self.task_type = task_type  # jd_summary, cv_processing, shortlisting, scheduling
        self.data = data
        self.complexity = complexity
        self.capability_vector = self._generate_capability_vector()
    
    def get_capability_requirements(self):
        return self.capability_vector
    
    def get_parameters(self):
        return {
            "task_type": self.task_type,
            "data": self.data
        }
    
    def get_default_decomposition(self):
        # For complex tasks, provide a default decomposition
        if self.task_type == "cv_processing" and len(self.data.get("cv_texts", [])) > 1:
            # Split CV processing into individual CV tasks
            subtasks = []
            for i, cv_text in enumerate(self.data.get("cv_texts", [])):
                subtask_id = f"{self.id}_cv{i}"
                subtask_description = f"Process CV {i+1}"
                subtask_data = {
                    "cv_text": cv_text,
                    "jd_summary": self.data.get("jd_summary", {})
                }
                subtask = RecruitingTask(
                    subtask_id, 
                    subtask_description, 
                    "single_cv_processing", 
                    subtask_data, 
                    self.complexity / 2
                )
                subtasks.append(subtask)
            return subtasks
        return []
    
    def _generate_capability_vector(self) -> np.ndarray:
        """
        Generate a capability vector based on the task type.
        Different task types require different capabilities.
        """
        # Initialize a zero vector
        vector = np.zeros(10)
        
        # Set different capabilities based on task type
        if self.task_type == "jd_summary":
            # JD summarization requires text analysis and extraction capabilities
            vector[0] = 0.8  # Text analysis
            vector[1] = 0.7  # Information extraction
            vector[2] = 0.5  # Domain knowledge
        
        elif self.task_type == "cv_processing" or self.task_type == "single_cv_processing":
            # CV processing requires matching and evaluation capabilities
            vector[0] = 0.6  # Text analysis
            vector[3] = 0.9  # Pattern matching
            vector[4] = 0.7  # Evaluation
        
        elif self.task_type == "shortlisting":
            # Shortlisting requires ranking and decision making
            vector[4] = 0.8  # Evaluation
            vector[5] = 0.9  # Ranking
            vector[6] = 0.7  # Decision making
        
        elif self.task_type == "scheduling":
            # Scheduling requires planning and coordination
            vector[7] = 0.9  # Planning
            vector[8] = 0.8  # Coordination
            vector[9] = 0.6  # Communication
        
        # Normalize the vector
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector


class AgentNetRecruitingAdapter:
    """
    Adapter class that connects AgentNet with the recruiting agents.
    This class maps AgentNet agents to recruiting agents and handles task execution.
    """
    
    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError("API key is required for AgentNetRecruitingAdapter")
        self.api_key = api_key
        
        try:
            # Initialize recruiting agents
            self.jd_summarizer = JDSummarizerAgent(api_key=self.api_key)
            self.recruiter = RecruitingAgent(api_key=self.api_key)
            self.shortlister = ShortlistingAgent(threshold=60, api_key=self.api_key)
            self.scheduler = InterviewSchedulerAgent(api_key=self.api_key)
            
            # Initialize AgentNet with specialized agents
            self.agent_net = self._initialize_agent_net()
            
            # Map agent IDs to recruiting agents
            self.agent_map = {
                "agent_0": self.jd_summarizer,  # JD Summarizer
                "agent_1": self.recruiter,     # CV Processor
                "agent_2": self.shortlister,   # Shortlister
                "agent_3": self.scheduler,     # Scheduler
                # agent_4 is a general-purpose agent for coordination
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize recruiting agents: {str(e)}")
    
    def _initialize_agent_net(self) -> AgentNet:
        """
        Initialize AgentNet with specialized agents for recruitment tasks.
        """
        # Create AgentNet with 5 agents (4 specialized + 1 coordinator)
        agent_net = AgentNet(num_agents=5, capability_dim=10, api_key=self.api_key)
        
        # Customize agent capabilities to match recruiting tasks
        # Agent 0: JD Summarizer
        jd_agent_capability = np.zeros(10)
        jd_agent_capability[0:3] = [0.8, 0.7, 0.5]  # Text analysis, extraction, domain knowledge
        jd_agent_capability = jd_agent_capability / np.linalg.norm(jd_agent_capability)
        agent_net.agents[0].capability = jd_agent_capability
        
        # Agent 1: CV Processor
        cv_agent_capability = np.zeros(10)
        cv_agent_capability[0] = 0.6  # Text analysis
        cv_agent_capability[3:5] = [0.9, 0.7]  # Pattern matching, evaluation
        cv_agent_capability = cv_agent_capability / np.linalg.norm(cv_agent_capability)
        agent_net.agents[1].capability = cv_agent_capability
        
        # Agent 2: Shortlister
        shortlist_agent_capability = np.zeros(10)
        shortlist_agent_capability[4:7] = [0.8, 0.9, 0.7]  # Evaluation, ranking, decision making
        shortlist_agent_capability = shortlist_agent_capability / np.linalg.norm(shortlist_agent_capability)
        agent_net.agents[2].capability = shortlist_agent_capability
        
        # Agent 3: Scheduler
        scheduler_agent_capability = np.zeros(10)
        scheduler_agent_capability[7:10] = [0.9, 0.8, 0.6]  # Planning, coordination, communication
        scheduler_agent_capability = scheduler_agent_capability / np.linalg.norm(scheduler_agent_capability)
        agent_net.agents[3].capability = scheduler_agent_capability
        
        # Agent 4: Coordinator (balanced capabilities)
        coordinator_capability = np.ones(10) / np.sqrt(10)  # Balanced capabilities
        agent_net.agents[4].capability = coordinator_capability
        
        return agent_net
    
    def execute_task(self, task: RecruitingTask) -> Dict[str, Any]:
        """
        Execute a recruiting task using the appropriate agent based on task type.
        
        Args:
            task: The recruiting task to execute
            
        Returns:
            Task result
        
        Raises:
            ValueError: If task is invalid or missing required data
            RuntimeError: If task execution fails
        """
        if not task:
            raise ValueError("Task cannot be None")
        if not task.task_type:
            raise ValueError("Task type is required")
            
        try:
            # Process the task through AgentNet
            result = self.agent_net.process_task(task)
            if not result:
                raise RuntimeError("AgentNet failed to process task")
            
            # Extract the agent that processed the task
            agent_id = result.get("current_agent", "agent_4")
            
            # Get the corresponding recruiting agent
            recruiting_agent = self.agent_map.get(agent_id)
            
            # If no specific agent was assigned, use the appropriate one based on task type
            if not recruiting_agent:
                if task.task_type == "jd_summary":
                    recruiting_agent = self.jd_summarizer
                elif task.task_type in ["cv_processing", "single_cv_processing"]:
                    recruiting_agent = self.recruiter
                elif task.task_type == "shortlisting":
                    recruiting_agent = self.shortlister
                elif task.task_type == "scheduling":
                    recruiting_agent = self.scheduler
                    
            if not recruiting_agent:
                raise ValueError(f"No suitable agent found for task type: {task.task_type}")
            
            try:
                # Execute the task with the recruiting agent
                task_params = task.get_parameters()
                task_type = task_params.get("task_type")
                task_data = task_params.get("data", {})
                
                if task_type == "jd_summary":
                    jd_text = task_data.get("jd_text")
                    if not jd_text:
                        raise ValueError("Job description text is required")
                    task_result = self.jd_summarizer.summarize_jd(jd_text)
                
                elif task_type == "single_cv_processing":
                    cv_text = task_data.get("cv_text")
                    jd_summary = task_data.get("jd_summary")
                    if not cv_text or not jd_summary:
                        raise ValueError("CV text and JD summary are required")
                    cv_data = self.recruiter.process_cv(cv_text)
                    match_score = self.recruiter.match_cv_to_jd(cv_data, jd_summary)
                    task_result = {"cv_data": cv_data, "match_score": match_score}
                
                elif task_type == "cv_processing":
                    cv_texts = task_data.get("cv_texts", [])
                    jd_summary = task_data.get("jd_summary")
                    if not cv_texts or not jd_summary:
                        raise ValueError("CV texts and JD summary are required")
                    candidates = []
                    
                    for i, cv_text in enumerate(cv_texts):
                        try:
                            cv_data = self.recruiter.process_cv(cv_text)
                            match_score = self.recruiter.match_cv_to_jd(cv_data, jd_summary)
                            candidates.append({
                                "cv_data": cv_data,
                                "match_score": match_score,
                                "overall_match": match_score.get("overall_match", 0)
                            })
                        except Exception as e:
                            print(f"Failed to process CV {i+1}: {str(e)}")
                            continue
                    task_result = candidates
                
                elif task_type == "shortlisting":
                    candidates = task_data.get("candidates", [])
                    if not candidates:
                        raise ValueError("Candidates list is required")
                    task_result = self.shortlister.shortlist_candidates(candidates)
                
                elif task_type == "scheduling":
                    shortlisted = task_data.get("shortlisted", [])
                    available_slots = task_data.get("available_slots", [])
                    if not shortlisted:
                        raise ValueError("No shortlisted candidates found")
                    task_result = self.scheduler.schedule_interviews(shortlisted, available_slots)
                
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                return task_result
                
            except Exception as e:
                raise RuntimeError(f"Task execution failed: {str(e)}")
                
        except Exception as e:
            raise RuntimeError(f"Task processing failed: {str(e)}")


class AgentNetRecruitmentOrchestrator:
    """
    Orchestrator that uses AgentNet for task routing and coordination.
    Replaces the original RecruitmentOrchestrator with AgentNet-powered coordination.
    """
    
    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError("API key is required for AgentNetRecruitmentOrchestrator")
        self.api_key = api_key
        try:
            self.adapter = AgentNetRecruitingAdapter(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AgentNetRecruitingAdapter: {str(e)}")
    
    def _generate_available_slots(self, num_candidates: int) -> List[Dict[str, Any]]:
        """Generate available interview slots."""
        import datetime
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
        """Run the full recruitment process using AgentNet for coordination."""
        if not jd_text or not isinstance(jd_text, str):
            raise ValueError("Job description text is required and must be a string")
        if not cv_texts or not isinstance(cv_texts, list) or not all(isinstance(cv, str) for cv in cv_texts):
            raise ValueError("CV texts must be a non-empty list of strings")

        # Create and execute JD summarization task
        try:
            jd_task = RecruitingTask(
                task_id="jd_summary_task",
                description="Summarize job description",
                task_type="jd_summary",
                data={"jd_text": jd_text},
                complexity=3.0
            )
            jd_summary = self.adapter.execute_task(jd_task)
            if not jd_summary:
                raise RuntimeError("JD summarization task returned no result")
        except Exception as e:
            raise RuntimeError(f"Failed to process job description: {str(e)}")

        # Create and execute CV processing task
        try:
            if not jd_summary:
                raise ValueError("Cannot process CVs without JD summary")
            cv_task = RecruitingTask(
                task_id="cv_processing_task",
                description="Process candidate CVs",
                task_type="cv_processing",
                data={"cv_texts": cv_texts, "jd_summary": jd_summary},
                complexity=7.0
            )
            candidates = self.adapter.execute_task(cv_task)
            if not candidates or not isinstance(candidates, list):
                raise RuntimeError("CV processing task returned invalid result")
            if not candidates:
                raise RuntimeError("No candidates were processed successfully")
        except Exception as e:
            raise RuntimeError(f"CV processing failed: {str(e)}")

        # Create and execute shortlisting task
        try:
            if not candidates:
                raise ValueError("Cannot shortlist without processed candidates")
            shortlist_task = RecruitingTask(
                task_id="shortlisting_task",
                description="Shortlist candidates",
                task_type="shortlisting",
                data={"candidates": candidates},
                complexity=5.0
            )
            shortlisting = self.adapter.execute_task(shortlist_task)
            if not shortlisting or not isinstance(shortlisting, dict):
                raise RuntimeError("Shortlisting task returned invalid result")
            if not shortlisting.get("shortlisted"):
                raise RuntimeError("No candidates were shortlisted")
        except Exception as e:
            raise RuntimeError(f"Shortlisting failed: {str(e)}")
        
        # Create and execute scheduling task
        try:
            available_slots = self._generate_available_slots(len(shortlisting["shortlisted"]))
            if not available_slots:
                raise RuntimeError("No available interview slots generated")

            schedule_task = RecruitingTask(
                task_id="scheduling_task",
                description="Schedule interviews",
                task_type="scheduling",
                data={
                    "shortlisted": shortlisting["shortlisted"],
                    "available_slots": available_slots
                },
                complexity=4.0
            )
            scheduled_interviews = self.adapter.execute_task(schedule_task)
            if not scheduled_interviews:
                raise RuntimeError("Failed to schedule interviews")
        except Exception as e:
            raise RuntimeError(f"Interview scheduling failed: {str(e)}")
        
        try:
            # Compile final result
            result = {
                "jd_summary": jd_summary,
                "candidates": candidates,
                "shortlisting": shortlisting,
                "scheduled_interviews": scheduled_interviews,
                "agentnet_metrics": self.adapter.agent_net.get_performance_metrics()
            }
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compile final results: {str(e)}")