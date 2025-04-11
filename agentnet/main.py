# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Main Module - Integrates all components of the AgentNet architecture

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import sys
import os

# Import local modules
from agentnet.agent import Agent
from agentnet.network import NetworkTopology
from agentnet.task_allocation import TaskAllocator, LoadBalancer
from agentnet.adaptive_learning import AdaptiveLearning
from agentnet.architecture import AgentNet as BaseAgentNet


class AgentNet:
    """
    AgentNet: A privacy-preserving, collective intelligence multi-agent system with high scalability 
    and failure-tolerance by leveraging an innovative framework, consisting of a fully decentralized 
    network architecture, a dynamic task allocation mechanism, and an adaptive agent learning method.
    
    This class serves as the main entry point for using the AgentNet framework, integrating all
    components into a cohesive system.
    """
    
    def __init__(self, num_agents: int = 5, capability_dim: int = 10, decay_factor: float = 0.5,
                 learning_rate: float = 0.1, specialization_factor: float = 0.2, api_key: str = None):
        """
        Initialize the AgentNet system.
        
        Args:
            num_agents: Number of agents in the network
            capability_dim: Dimension of capability vectors
            decay_factor: Alpha value in [0,1] that balances historical performance with recent interactions
            learning_rate: Rate at which agents learn from experiences
            specialization_factor: Factor controlling the degree of specialization
            api_key: DeepSeek API key for LLM reasoning capabilities
        """
        # Create agents with random initial capabilities
        self.agents = []
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            capability = np.random.rand(capability_dim)
            capability = capability / np.linalg.norm(capability)  # Normalize
            agent = Agent(agent_id, capability)
            self.agents.append(agent)
        
        # Initialize components
        self.network_topology = NetworkTopology(self.agents, decay_factor)
        self.task_allocator = TaskAllocator()
        self.load_balancer = LoadBalancer()
        self.adaptive_learning = AdaptiveLearning(learning_rate, specialization_factor, memory_capacity=100, api_key=api_key)
        
        # Initialize base AgentNet for task processing
        self.agent_net_base = BaseAgentNet(self.agents, decay_factor)
        
        # Task history
        self.tasks_history = []
    
    def process_task(self, task: Any) -> Dict:
        """
        Process a task using the AgentNet system.
        
        Args:
            task: The task to be processed
            
        Returns:
            Task result
        """
        # Allocate the task to the most suitable agent
        initial_agent_id = self.task_allocator.allocate_task(task, self.agents, self.network_topology)
        
        # Process the task using the base AgentNet
        result = self.agent_net_base.process_task(task)
        
        # Update the network based on task performance
        self.update_network(task, result)
        
        # Store task in history
        self.tasks_history.append((task, result))
        
        return result
    
    def update_network(self, task: Any, result: Dict):
        """
        Update the network based on task performance.
        
        Args:
            task: The completed task
            result: The task result
        """
        # Extract interactions from the task execution
        interactions = result.get('interactions', [])
        
        # Update network topology
        self.network_topology.update_weights(interactions)
        
        # Update agent capabilities through adaptive learning
        for interaction in interactions:
            agent_id = interaction.get('agent_id')
            success_metric = interaction.get('success_metric', 0.0)
            
            agent = self._get_agent_by_id(agent_id)
            if agent:
                updated_capability = self.adaptive_learning.update_agent_capability(
                    agent, task, success_metric)
                agent.capability = updated_capability
        
        # Periodically evolve the entire network
        if len(self.tasks_history) % 10 == 0:  # Every 10 tasks
            self.adaptive_learning.evolve_network(self.agents, self.tasks_history, self.network_topology)
    
    def optimize(self, tasks: List[Any], eval_function: Callable):
        """
        Optimize the AgentNet for a series of tasks.
        
        The optimization goal is to maximize the evaluated score:
        G* = (A*, E*) = argmax(A,E) Eval(G,T)
        
        Args:
            tasks: List of tasks to be solved
            eval_function: Function to evaluate the performance
            
        Returns:
            Optimized AgentNet
        """
        for task in tasks:
            # Process the task
            result = self.process_task(task)
            
            # Evaluate the result
            score = eval_function(self, task, result)
            
            # Store evaluation score in the result
            result['evaluation_score'] = score
        
        return self
    
    def _get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by its ID.
        
        Args:
            agent_id: ID of the agent to find
            
        Returns:
            The agent with the given ID, or None if not found
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_network_visualization(self):
        """
        Get a visualization of the network topology.
        
        Returns:
            Visualization data structure
        """
        return self.network_topology.visualize_network()
    
    def get_agent_capabilities(self):
        """
        Get the capabilities of all agents in the network.
        
        Returns:
            Dictionary mapping agent IDs to capability vectors
        """
        return {agent.id: agent.capability.tolist() for agent in self.agents}
    
    def get_performance_metrics(self):
        """
        Get performance metrics for the AgentNet system.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate average evaluation score
        avg_score = 0.0
        count = 0
        for _, result in self.tasks_history:
            if 'evaluation_score' in result:
                avg_score += result['evaluation_score']
                count += 1
        
        if count > 0:
            avg_score /= count
        
        # Calculate other metrics
        metrics = {
            'average_evaluation_score': avg_score,
            'tasks_completed': len(self.tasks_history),
            'agent_count': len(self.agents),
        }
        
        return metrics


def visualize_agentnet():
    st.set_page_config(layout="wide")
    st.title("AgentNet Live Dashboard")

    # Real-time updates container
    refresh_rate = st.sidebar.slider("Refresh rate (sec)", 1, 10, 3)
    
    # Main dashboard columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Network Activity")
        network_placeholder = st.empty()
        
        st.subheader("Capability Matrix")
        matrix_placeholder = st.empty()

    with col2:
        st.header("System Metrics")
        metrics_placeholder = st.empty()
        
        st.subheader("Task Queue")
        task_queue_placeholder = st.empty()

    # Live update loop
    while True:
        # Network graph
        network_data = agent_net.get_network_visualization()
        fig = px.parallel_coordinates(
            pd.DataFrame(network_data['nodes']),
            color='capability',
            title="Agent Network Interactions"
        )
        network_placeholder.plotly_chart(fig, use_container_width=True, key=f"network_{time.time()}")

        # Capability heatmap
        capabilities = agent_net.get_agent_capabilities()
        df_cap = pd.DataFrame(capabilities).T
        matrix_fig = px.imshow(
            df_cap,
            labels=dict(x="Capability Dimension", y="Agent ID", color="Strength"),
            title="Agent Specialization Matrix"
        )
        matrix_placeholder.plotly_chart(matrix_fig, use_container_width=True, key=f"matrix_{time.time()}")

        # System metrics
        metrics = agent_net.get_performance_metrics()
        metrics_df = pd.DataFrame([metrics])
        metrics_placeholder.dataframe(metrics_df.T.style.highlight_max(axis=0))

        # Task queue
        tasks = agent_net.tasks_history[-5:]
        task_df = pd.DataFrame([{"Task": t[0], "Status": t[1].get('status', 'pending')} for t in tasks])
        task_queue_placeholder.dataframe(task_df)

        time.sleep(refresh_rate)

# Call the visualization function
if __name__ == "__main__":
    agent_net = AgentNet(num_agents=7, capability_dim=8)
    visualize_agentnet()