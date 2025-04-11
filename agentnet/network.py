# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Network Topology Implementation

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from agentnet.architecture import Agent


class NetworkTopology:
    """
    Implements the decentralized network topology of AgentNet.
    
    As illustrated in the paper, AgentNet employs a dual-role design where each agent
    is equipped with a router and an executor. The router facilitates routing decisions
    and the executor executes specific tasks. This design endows AgentNet with a fully
    decentralized network structure because routing decisions are made independently by
    each agent, without relying on a central authority or coordinator.
    """

    def __init__(self, agents: List[Agent], decay_factor: float = 0.5):
        """
        Initialize the network topology.
        
        Args:
            agents: List of agents in the network
            decay_factor: Alpha value in [0,1] that balances historical performance with recent interactions
        """
        self.agents = agents
        self.decay_factor = decay_factor

        # Initialize weight matrix for connections between agents
        n = len(agents)
        self.weight_matrix = np.ones((n, n)) / n  # Initially uniform weights

        # Map of agent IDs to indices in the weight matrix
        self.agent_indices = {agent.id: i for i, agent in enumerate(agents)}
    
    def get_connection_weight(self, from_agent_id: str, to_agent_id: str) -> float:
        """
        Get the weight of the connection from one agent to another.
        
        Args:
            from_agent_id: ID of the source agent
            to_agent_id: ID of the target agent
            
        Returns:
            Connection weight
        """
        from_idx = self.agent_indices.get(from_agent_id)
        to_idx = self.agent_indices.get(to_agent_id)
        
        if from_idx is None or to_idx is None:
            return 0.0
        
        return self.weight_matrix[from_idx, to_idx]
    
    def update_weights(self, interactions: List[Dict]):
        """
        Update the weight matrix based on recent interactions.
        
        Uses the formula:
        w(m+1)(i,j) = α·w(m)(i,j) + (1−α)· sum(S(a(m+1)i, a(m+1)j, t(m+1)))/K
        
        Args:
            interactions: List of recent agent interactions with success metrics
        """
        if not interactions:
            return
        
        # Group interactions by agent pair
        pair_metrics = {}
        for interaction in interactions:
            from_id = interaction.get('from')
            to_id = interaction.get('to')
            success = interaction.get('success_metric', 0.0)
            
            if from_id is None or to_id is None:
                continue
                
            pair_key = (from_id, to_id)
            if pair_key not in pair_metrics:
                pair_metrics[pair_key] = []
            
            pair_metrics[pair_key].append(success)
        
        # Update weights for each agent pair with interactions
        for (from_id, to_id), metrics in pair_metrics.items():
            from_idx = self.agent_indices.get(from_id)
            to_idx = self.agent_indices.get(to_id)
            
            if from_idx is None or to_idx is None:
                continue
            
            # Calculate average success metric
            avg_success = sum(metrics) / len(metrics)
            
            # Update weight using the formula
            old_weight = self.weight_matrix[from_idx, to_idx]
            new_weight = (self.decay_factor * old_weight + 
                         (1 - self.decay_factor) * avg_success)
            
            self.weight_matrix[from_idx, to_idx] = new_weight
    
    def get_best_next_agent(self, current_agent_id: str, task_capability: np.ndarray) -> Optional[str]:
        """
        Get the best next agent to forward a task to, based on connection weights and capability match.
        
        Args:
            current_agent_id: ID of the current agent
            task_capability: Capability requirements of the task
            
        Returns:
            ID of the best next agent, or None if no suitable agent is found
        """
        current_idx = self.agent_indices.get(current_agent_id)
        if current_idx is None:
            return None
        
        # Get weights for connections from current agent
        weights = self.weight_matrix[current_idx]
        
        # Calculate capability match scores
        capability_scores = []
        for agent in self.agents:
            # Skip the current agent
            if agent.id == current_agent_id:
                capability_scores.append(0.0)
                continue
                
            # Calculate similarity between task capability and agent capability
            similarity = self._calculate_similarity(task_capability, agent.capability)
            capability_scores.append(similarity)
        
        # Combine weights and capability scores
        combined_scores = weights * np.array(capability_scores)
        
        # Find the agent with the highest combined score
        best_idx = np.argmax(combined_scores)
        
        # Ensure the score is above a threshold
        if combined_scores[best_idx] > 0.1:  # Arbitrary threshold
            return self.agents[best_idx].id
        
        return None
    
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
    
    def visualize_network(self):
        """
        Generate a visualization of the network topology.
        
        Returns:
            Visualization data structure
        """
        # This would generate a visualization of the network
        # In a real implementation, this might return data for a graph visualization library
        nodes = [{'id': agent.id, 'capability': agent.capability.tolist()} for agent in self.agents]
        
        edges = []
        for i, from_agent in enumerate(self.agents):
            for j, to_agent in enumerate(self.agents):
                if i != j:  # Skip self-connections
                    weight = self.weight_matrix[i, j]
                    if weight > 0.1:  # Only include significant connections
                        edges.append({
                            'from': from_agent.id,
                            'to': to_agent.id,
                            'weight': float(weight)
                        })
        
        return {
            'nodes': nodes,
            'edges': edges
        }