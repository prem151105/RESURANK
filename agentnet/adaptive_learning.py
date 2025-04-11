# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Adaptive Agent Learning Method

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import datetime
import requests
from .agent import MemoryModule


class AdaptiveLearning:
    """
    Implements the adaptive agent learning method of AgentNet.
    
    This mechanism enables continuous specialization of agents, making the whole MAS more
    scalable and adaptive. Through this learning process, agents can evolve their capabilities
    based on their experiences, leading to a self-organizing system capable of handling
    complex tasks while preserving privacy and adapting to changing environments.
    
    AgentNet's adaptive learning mechanism facilitates continuous improvement and specialization
    of agents based on their task experiences, without the need for explicit role assignment.
    This process enables agents to gradually develop expertise in specific domains, differentiating
    AgentNet from static multi-agent systems and allowing it to adapt to evolving requirements over time.
    
    Agents in AgentNet follow the ReAct (Reasoning + Acting) framework, which empowers agents to
    reason about a given query and its context before deciding appropriate actions for the executor
    modules. In addition to the given query and its context, the agent also retrieves relevant trajectory
    fragments from its memory modules to enhance reasoning and acting. The retrieval process is performed
    using a Retrieval-Augmented Generation (RAG) mechanism, which allows the agent to leverage past
    experiences to generate informed decisions and actions for new tasks.
    
    The system now integrates DeepSeek API for advanced LLM reasoning capabilities.
    """

    def __init__(self, learning_rate: float = 0.1, specialization_factor: float = 0.2, memory_capacity: int = 100, api_key: str = None):
        """
        Initialize the adaptive learning mechanism.
        
        Args:
            learning_rate: Rate at which agents learn from experiences
            specialization_factor: Factor controlling the degree of specialization
            memory_capacity: Maximum capacity for memory modules (Cmax in the paper)
            api_key: DeepSeek API key for LLM reasoning
        """
        self.learning_rate = learning_rate
        self.specialization_factor = specialization_factor
        self.memory_capacity = memory_capacity
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

    def _llm_reasoning(self, prompt: str, context: str = None) -> str:
        """
        Perform LLM reasoning using DeepSeek API.
        
        Args:
            prompt: The prompt for reasoning
            context: Additional context for the reasoning task
        
        Returns:
            The reasoning result from the LLM
        """
        if not self.api_key:
            raise ValueError("DeepSeek API key is required for LLM reasoning")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "user", "content": prompt}]
        if context:
            messages.append({"role": "system", "content": context})

        response = requests.post(
            self.api_url,
            headers=headers,
            json={"messages": messages}
        )

        if response.status_code != 200:
            raise RuntimeError(f"LLM reasoning failed: {response.text}")

        return response.json()['choices'][0]['message']['content']
    
    def update_agent_capability(self, agent: Any, task: Any, success_metric: float) -> np.ndarray:
        """
        Update an agent's capability vector based on task performance.
        
        Args:
            agent: The agent to update
            task: The task that was performed
            success_metric: Measure of success (0.0 to 1.0)
            
        Returns:
            Updated capability vector
        """
        # Extract task capability requirements
        task_capability = self._extract_task_capability(task)
        
        # Current agent capability
        agent_capability = agent.capability
        
        # Calculate the update direction
        # If success is high, move capability toward task requirements
        # If success is low, move capability away from task requirements
        direction = 2 * success_metric - 1  # Maps [0,1] to [-1,1]
        
        # Update the capability vector
        update = direction * self.learning_rate * (task_capability - agent_capability)
        updated_capability = agent_capability + update
        
        # Apply specialization factor
        # This increases the variance in the capability vector, making the agent more specialized
        mean_capability = np.mean(updated_capability)
        specialization_update = self.specialization_factor * (updated_capability - mean_capability)
        updated_capability += specialization_update
        
        # Normalize the capability vector
        updated_capability = self._normalize_capability(updated_capability)
        
        return updated_capability
    
    def evolve_network(self, agents: List[Any], tasks_history: List[Tuple[Any, Dict]], network_topology: Any):
        """
        Evolve the entire agent network based on historical performance.
        
        Args:
            agents: List of agents in the network
            tasks_history: History of tasks and their results
            network_topology: The network topology
        """
        # This would analyze the historical performance of the network and make
        # global adjustments to improve overall performance
        # Simplified implementation
        
        # Update connection weights based on task history
        interactions = self._extract_interactions(tasks_history)
        if network_topology and hasattr(network_topology, 'update_weights'):
            network_topology.update_weights(interactions)
        
        # Identify underperforming agents and adjust their capabilities
        agent_performance = self._calculate_agent_performance(agents, tasks_history)
        for agent_id, performance in agent_performance.items():
            agent = self._get_agent_by_id(agents, agent_id)
            if agent and performance < 0.5:  # Arbitrary threshold
                # Identify successful agents with similar roles
                similar_agents = self._find_similar_agents(agent, agents, agent_performance)
                if similar_agents:
                    # Learn from successful similar agents
                    self._learn_from_peers(agent, similar_agents)
    
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
    
    def _normalize_capability(self, capability: np.ndarray) -> np.ndarray:
        """
        Normalize a capability vector.
        
        Args:
            capability: Capability vector to normalize
            
        Returns:
            Normalized capability vector
        """
        # Ensure all values are positive
        capability = np.maximum(capability, 0.0)
        
        # Normalize to unit length
        norm = np.linalg.norm(capability)
        if norm > 0:
            return capability / norm
        return capability
    
    def _extract_interactions(self, tasks_history: List[Tuple[Any, Dict]]) -> List[Dict]:
        """
        Extract agent interactions from tasks history.
        
        Args:
            tasks_history: History of tasks and their results
            
        Returns:
            List of interactions with success metrics
        """
        interactions = []
        for task, result in tasks_history:
            if 'interactions' in result:
                interactions.extend(result['interactions'])
        return interactions
    
    def _calculate_agent_performance(self, agents: List[Any], tasks_history: List[Tuple[Any, Dict]]) -> Dict[str, float]:
        """
        Calculate performance metrics for each agent based on task history.
        
        Args:
            agents: List of agents
            tasks_history: History of tasks and their results
            
        Returns:
            Dictionary mapping agent IDs to performance scores
        """
        performance = {agent.id: 0.0 for agent in agents}
        counts = {agent.id: 0 for agent in agents}
        
        for task, result in tasks_history:
            if 'interactions' in result:
                for interaction in result['interactions']:
                    agent_id = interaction.get('agent_id')
                    success = interaction.get('success_metric', 0.0)
                    
                    if agent_id in performance:
                        performance[agent_id] += success
                        counts[agent_id] += 1
        
        # Calculate average performance
        for agent_id in performance:
            if counts[agent_id] > 0:
                performance[agent_id] /= counts[agent_id]
        
        return performance
    
    def _get_agent_by_id(self, agents: List[Any], agent_id: str) -> Optional[Any]:
        """
        Get an agent by its ID.
        
        Args:
            agents: List of agents
            agent_id: ID of the agent to find
            
        Returns:
            The agent with the given ID, or None if not found
        """
        for agent in agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def _find_similar_agents(self, agent: Any, agents: List[Any], performance: Dict[str, float]) -> List[Any]:
        """
        Find agents with similar capabilities but better performance.
        
        Args:
            agent: The agent to find similar agents for
            agents: List of all agents
            performance: Dictionary mapping agent IDs to performance scores
            
        Returns:
            List of similar agents with better performance
        """
        similar_agents = []
        for other_agent in agents:
            if other_agent.id == agent.id:
                continue
                
            # Calculate similarity between agent capabilities
            similarity = self._calculate_similarity(agent.capability, other_agent.capability)
            
            # Check if the other agent has better performance
            if similarity > 0.7 and performance.get(other_agent.id, 0.0) > performance.get(agent.id, 0.0):
                similar_agents.append(other_agent)
        
        return similar_agents
    
    def _learn_from_peers(self, agent: Any, similar_agents: List[Any]):
        """
        Update an agent's capability by learning from similar, successful agents.
        
        Args:
            agent: The agent to update
            similar_agents: List of similar, successful agents to learn from
        """
        if not similar_agents:
            return
            
        # Calculate average capability of similar agents
        avg_capability = np.zeros_like(agent.capability)
        for similar_agent in similar_agents:
            avg_capability += similar_agent.capability
        avg_capability /= len(similar_agents)
        
        # Update agent capability to move toward the average of successful similar agents
        update = self.learning_rate * (avg_capability - agent.capability)
        agent.capability = self._normalize_capability(agent.capability + update)
    
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