# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Dynamic Task Allocation Mechanism

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .architecture import Agent


class TaskAllocator:
    """
    Implements the dynamic task allocation mechanism of AgentNet.
    
    This mechanism optimizes workload distribution based on agent capabilities and the current system state,
    allowing AgentNet to flexibly allocate tasks to the most suitable agents.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the task allocator.
        
        Args:
            similarity_threshold: Threshold for capability similarity to consider an agent suitable
        """
        self.similarity_threshold = similarity_threshold
    
    def allocate_task(self, task: Any, agents: List[Any], network_topology: Any) -> str:
        """
        Allocate a task to the most suitable agent.
        
        Args:
            task: The task to be allocated
            agents: List of available agents
            network_topology: The network topology
            
        Returns:
            ID of the most suitable agent
        """
        # Extract task capability requirements
        task_capability = self._extract_task_capability(task)
        
        # Calculate capability similarity for each agent
        agent_similarities = []
        for agent in agents:
            similarity = self._calculate_similarity(task_capability, agent.capability)
            agent_similarities.append((agent.id, similarity))
        
        # Sort by similarity in descending order
        agent_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select the most suitable agent
        for agent_id, similarity in agent_similarities:
            if similarity >= self.similarity_threshold:
                return agent_id
        
        # If no agent meets the threshold, return the agent with the highest similarity
        if agent_similarities:
            return agent_similarities[0][0]
        
        return None

    def process_task(self, task: Any, agent_id: str) -> Dict:
        """
        Process a task using the three operations: forward, split, and execute.
        
        Args:
            task: The task to be processed
            agent_id: ID of the agent processing the task
            
        Returns:
            Task result
        """
        # Initialize task state
        task_state = {
            'current_agent': agent_id,
            'status': 'pending',
            'subtasks': [],
            'results': {},
            'context': {}
        }
        
        # Get the agent
        agent = next((a for a in self.agents if a.id == agent_id), None)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Perform task routing
        while task_state['status'] != 'completed':
            # Get task capability requirements
            task_capability = self._extract_task_capability(task)
            
            # Calculate similarity between task and agent capabilities
            similarity = self._calculate_similarity(task_capability, agent.capability)
            
            # Determine operation based on similarity
            if similarity >= self.similarity_threshold:
                # Execute the task
                result = agent.execute_task(task)
                task_state['status'] = 'completed'
                task_state['results'][agent_id] = result
            elif similarity >= 0.5:
                # Split the task into subtasks
                subtasks = agent.decompose_task(task)
                task_state['subtasks'] = subtasks
                task_state['status'] = 'split'
            else:
                # Forward the task to another agent
                next_agent_id = self.allocate_task(task, self.agents, self.network_topology)
                task_state['current_agent'] = next_agent_id
                task_state['status'] = 'forwarded'
                agent = next((a for a in self.agents if a.id == next_agent_id), None)
                if not agent:
                    raise ValueError(f"Agent {next_agent_id} not found")
        
        return task_state
    
    def decompose_task(self, task: Any, agents: List[Any]) -> List[Tuple[Any, str]]:
        """
        Decompose a complex task into subtasks and allocate each to the most suitable agent.
        
        Args:
            task: The task to be decomposed
            agents: List of available agents
            
        Returns:
            List of (subtask, agent_id) pairs
        """
        # In a real implementation, this would use LLM reasoning to decompose the task
        # Simplified implementation
        subtasks = getattr(task, 'get_default_decomposition', lambda: [])()
        
        # Allocate each subtask
        allocations = []
        for subtask in subtasks:
            agent_id = self.allocate_task(subtask, agents, None)
            if agent_id:
                allocations.append((subtask, agent_id))
        
        return allocations
    
    def _extract_task_capability(self, task) -> np.ndarray:
        """
        Extract capability requirements from a task.
        
        Args:
            task: The task to extract capabilities from
            
        Returns:
            Capability vector
        """
        # In a real implementation, this would use LLM reasoning to extract
        # capability requirements from the task description
        # Simplified implementation
        return getattr(task, 'capability_vector', np.ones(10))
    
    def _calculate_similarity(self, capability1: np.ndarray, capability2: np.ndarray) -> float:
        """
        Calculate similarity between two capability vectors.
        
        Args:
            capability1: First capability vector
            capability2: Second capability vector
            
        Returns:
            Similarity score
        """
        # Simple cosine similarity implementation
        return np.dot(capability1, capability2) / (np.linalg.norm(capability1) * np.linalg.norm(capability2))


class LoadBalancer:
    """
    Balances the workload across agents in the AgentNet system.
    
    Works in conjunction with the TaskAllocator to ensure that no agent is overloaded
    and that tasks are distributed efficiently.
    """
    
    def __init__(self, max_load_factor: float = 0.8):
        """
        Initialize the load balancer.
        
        Args:
            max_load_factor: Maximum load factor for an agent (0.0 to 1.0)
        """
        self.max_load_factor = max_load_factor
        self.agent_loads = {}  # {agent_id: current_load}
    
    def update_agent_load(self, agent_id: str, load_delta: float):
        """
        Update the load of an agent.
        
        Args:
            agent_id: ID of the agent
            load_delta: Change in load (positive or negative)
        """
        if agent_id not in self.agent_loads:
            self.agent_loads[agent_id] = 0.0
        
        self.agent_loads[agent_id] += load_delta
        
        # Ensure load is non-negative
        self.agent_loads[agent_id] = max(0.0, self.agent_loads[agent_id])
    
    def get_agent_load(self, agent_id: str) -> float:
        """
        Get the current load of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Current load (0.0 to 1.0)
        """
        return self.agent_loads.get(agent_id, 0.0)
    
    def is_agent_overloaded(self, agent_id: str) -> bool:
        """
        Check if an agent is overloaded.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if the agent is overloaded, False otherwise
        """
        return self.get_agent_load(agent_id) > self.max_load_factor
    
    def get_least_loaded_agents(self, n: int = 3) -> List[str]:
        """
        Get the IDs of the least loaded agents.
        
        Args:
            n: Number of agents to return
            
        Returns:
            List of agent IDs
        """
        # Sort agents by load in ascending order
        sorted_agents = sorted(self.agent_loads.items(), key=lambda x: x[1])
        
        # Return the n least loaded agents
        return [agent_id for agent_id, _ in sorted_agents[:n]]
    
    def estimate_task_load(self, task: Any) -> float:
        """
        Estimate the load that a task will place on an agent.
        
        Args:
            task: The task to estimate load for
            
        Returns:
            Estimated load (0.0 to 1.0)
        """
        # In a real implementation, this would use task properties to estimate load
        # Simplified implementation
        complexity = getattr(task, 'complexity', 1.0)
        return min(1.0, complexity / 10.0)  # Normalize to 0.0-1.0