# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Basic Usage Example

import numpy as np
from agentnet.main import AgentNet
from agentnet.agent import Agent


class SimpleTask:
    """
    A simple task for demonstration purposes.
    """
    
    def __init__(self, task_id, description, complexity, capability_vector):
        self.id = task_id
        self.description = description
        self.complexity = complexity
        self.capability_vector = capability_vector
    
    def get_capability_requirements(self):
        return self.capability_vector
    
    def get_parameters(self):
        return {}
    
    def get_default_decomposition(self):
        # Simple decomposition into two subtasks
        subtasks = []
        for i in range(2):
            subtask_id = f"{self.id}_sub{i}"
            subtask_description = f"Subtask {i} of {self.description}"
            subtask_complexity = self.complexity / 2
            subtask_capability = self.capability_vector
            subtask = SimpleTask(subtask_id, subtask_description, subtask_complexity, subtask_capability)
            subtasks.append(subtask)
        return subtasks


def evaluation_function(agent_net, task, result):
    """
    Simple evaluation function that returns a score between 0 and 1.
    """
    # In a real implementation, this would evaluate the quality of the result
    # For demonstration, we'll use a simple random score
    return 0.7 + 0.3 * np.random.random()


def main():
    # DeepSeek API key for LLM reasoning capabilities
    api_key = "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"
    
    # Initialize AgentNet with 5 agents and API key
    agent_net = AgentNet(num_agents=5, capability_dim=10, api_key=api_key)
    
    print("Initialized AgentNet with 5 agents")
    
    # Create a list of tasks
    tasks = []
    for i in range(10):
        task_id = f"task_{i}"
        description = f"Task {i} description"
        complexity = 1 + 9 * np.random.random()  # Random complexity between 1 and 10
        capability_vector = np.random.rand(10)  # Random capability requirements
        capability_vector = capability_vector / np.linalg.norm(capability_vector)  # Normalize
        
        task = SimpleTask(task_id, description, complexity, capability_vector)
        tasks.append(task)
    
    print(f"Created {len(tasks)} tasks")
    
    # Process each task individually
    print("\nProcessing tasks individually:")
    for task in tasks[:3]:  # Process first 3 tasks
        print(f"Processing {task.id}: {task.description}")
        result = agent_net.process_task(task)
        # The result is a task_state object with context and other information
        print(f"Result: Task processed by {len(result.get('context', []))} agent(s)")
        print(f"Task state: {result.keys()}")
        
        # Add a simulated output for demonstration purposes
        result['output'] = f"Processed task {task.id} with complexity {task.complexity:.2f}"
    
    # Optimize the network for all tasks
    print("\nOptimizing network for all tasks:")
    agent_net.optimize(tasks, evaluation_function)
    
    # Get performance metrics
    metrics = agent_net.get_performance_metrics()
    print("\nPerformance metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Get agent capabilities after optimization
    capabilities = agent_net.get_agent_capabilities()
    print("\nAgent capabilities after optimization:")
    for agent_id, capability in capabilities.items():
        print(f"{agent_id}: {capability}")
    
    # Get network visualization
    visualization = agent_net.get_network_visualization()
    print("\nNetwork visualization:")
    print(f"Nodes: {len(visualization['nodes'])}")
    print(f"Edges: {len(visualization['edges'])}")


if __name__ == "__main__":
    main()