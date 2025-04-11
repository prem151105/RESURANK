# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Core Architecture Implementation

import numpy as np
from typing import List, Dict, Tuple, Set, Callable, Any, Optional


class AgentNet:
    """
    AgentNet: A privacy-preserving, collective intelligence multi-agent system with high scalability 
    and failure-tolerance by leveraging an innovative framework, consisting of a fully decentralized 
    network architecture, a dynamic task allocation mechanism, and an adaptive agent learning method.
    
    Formally defined as a tuple G = (A,E), where:
    - A = {a1,a2,...,an} represents the set of autonomous agents
    - C = {c1,c2,...,cn} represents each agent's ability
    - E ⊆ A × A represents the communication connections between agents
    """
    
    def __init__(self, agents: List['Agent'], decay_factor: float = 0.5):
        """
        Initialize the AgentNet system.
        
        Args:
            agents: List of Agent objects in the network
            decay_factor: Alpha value in [0,1] that balances historical performance with recent interactions
        """
        self.agents = agents
        self.decay_factor = decay_factor
        self.tasks_history = []
        
        # Initialize weight matrix for connections between agents
        n = len(agents)
        self.weight_matrix = np.ones((n, n)) / n  # Initially uniform weights
        
    def optimize(self, tasks: List['Task'], eval_function: Callable):
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
            # Process the task using the current network configuration
            result = self.process_task(task)
            
            # Update the network based on task performance
            self.update_network(task, result)
            
            # Store task in history
            self.tasks_history.append((task, result))
            
        # Return the optimized network
        return self
    
    def process_task(self, task: 'Task'):
        """
        Process a single task using the AgentNet system.
        
        Args:
            task: The task to be processed
            
        Returns:
            Task result
        """
        # Extract task capability requirements
        task_capability = task.get_capability_requirements()
        
        # Find the most suitable initial agent based on capability similarity
        current_agent = self._find_most_suitable_agent(task_capability)
        
        # Initialize task state
        task_state = {
            'original_task': task,
            'context': [],
            'params': task.get_parameters()
        }
        
        visited_agents = set()
        finished = False
        
        # Process the task through the network until finished or a cycle is detected
        while not finished and current_agent not in visited_agents:
            visited_agents.add(current_agent)
            
            # Get routing decision from current agent
            action = current_agent.route(task, task_state)
            
            if action['type'] == 'forward':
                # Forward to another agent
                next_agent_id = action['target_agent']
                current_agent = self._get_agent_by_id(next_agent_id)
                
            elif action['type'] == 'split':
                # Split task into subtasks
                subtasks = action['subtasks']
                results = []
                
                # Process each subtask
                for subtask in subtasks:
                    subtask_result = self.process_task(subtask)
                    results.append(subtask_result)
                
                # Combine results
                task_state['context'].extend(results)
                finished = all(result.get('completed', False) for result in results)
                
            elif action['type'] == 'execute':
                # Execute the task
                result = current_agent.execute(task, task_state)
                task_state['context'].append(result)
                finished = True
        
        return task_state
    
    def update_network(self, task: 'Task', result: Dict):
        """
        Update the network weights based on task performance.
        
        Uses the formula:
        w(m+1)(i,j) = α·w(m)(i,j) + (1−α)· sum(S(a(m+1)i, a(m+1)j, t(m+1)))/K
        
        Args:
            task: The completed task
            result: The task result
        """
        # Extract interaction data from the task execution
        interactions = self._extract_interactions(result)
        
        # Update weight matrix
        n = len(self.agents)
        new_weights = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Get agent IDs
                agent_i = self.agents[i].id
                agent_j = self.agents[j].id
                
                # Calculate success metric for this agent pair
                success_metrics = []
                for interaction in interactions:
                    if interaction['from'] == agent_i and interaction['to'] == agent_j:
                        success_metrics.append(interaction['success_metric'])
                
                # Update weight using the formula
                if success_metrics:
                    avg_success = sum(success_metrics) / len(success_metrics)
                    new_weights[i, j] = (self.decay_factor * self.weight_matrix[i, j] + 
                                         (1 - self.decay_factor) * avg_success)
                else:
                    new_weights[i, j] = self.weight_matrix[i, j]
        
        self.weight_matrix = new_weights
    
    def _find_most_suitable_agent(self, task_capability):
        """
        Find the most suitable agent for a task based on capability similarity.
        
        Args:
            task_capability: The capability requirements of the task
            
        Returns:
            The most suitable agent
        """
        max_similarity = -1
        most_suitable_agent = None
        
        for agent in self.agents:
            similarity = self._calculate_similarity(task_capability, agent.capability)
            if similarity > max_similarity:
                max_similarity = similarity
                most_suitable_agent = agent
        
        return most_suitable_agent
    
    def _calculate_similarity(self, capability1, capability2):
        """
        Calculate similarity between two capability vectors.
        
        Args:
            capability1: First capability vector
            capability2: Second capability vector
            
        Returns:
            Similarity score
        """
        # Simple cosine similarity implementation
        # In a real implementation, this would be more sophisticated
        return np.dot(capability1, capability2) / (np.linalg.norm(capability1) * np.linalg.norm(capability2))
    
    def _get_agent_by_id(self, agent_id):
        """
        Get an agent by its ID.
        
        Args:
            agent_id: The ID of the agent to find
            
        Returns:
            The agent with the given ID
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def _extract_interactions(self, result):
        """
        Extract agent interactions from a task result.
        
        Args:
            result: The task result
            
        Returns:
            List of interactions with success metrics
        """
        # This would extract the sequence of agent interactions and their success metrics
        # from the task execution result
        # Simplified implementation for now
        return result.get('interactions', [])


class Agent:
    """
    Agent in the AgentNet system.
    
    Each agent contains two key components:
    - Router (rou): Responsible for analyzing received routing queries and making routing decisions
    - Executor (exe): Responsible for responding to executing queries through operations and tools
    """
    
    def __init__(self, agent_id: str, capability: np.ndarray, router_memory_size: int = 100, executor_memory_size: int = 100):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            capability: Vector representing the agent's capabilities
            router_memory_size: Size of the router memory module
            executor_memory_size: Size of the executor memory module
        """
        self.id = agent_id
        self.capability = capability
        
        # Initialize router and executor components
        self.router = Router(memory_size=router_memory_size)
        self.executor = Executor(memory_size=executor_memory_size)
    
    def route(self, task: 'Task', task_state: Dict) -> Dict:
        """
        Make a routing decision for a task.
        
        Args:
            task: The task to be routed
            task_state: Current state of the task
            
        Returns:
            Routing decision
        """
        return self.router.make_decision(task, task_state, self.capability)
    
    def execute(self, task: 'Task', task_state: Dict) -> Dict:
        """
        Execute a task.
        
        Args:
            task: The task to be executed
            task_state: Current state of the task
            
        Returns:
            Execution result
        """
        return self.executor.execute_task(task, task_state, self.capability)


class Router:
    """
    Router component of an agent, responsible for analyzing received routing queries and making routing decisions.
    """
    
    def __init__(self, memory_size: int = 100):
        """
        Initialize a router.
        
        Args:
            memory_size: Size of the memory module
        """
        self.memory = MemoryModule(max_size=memory_size)
    
    def make_decision(self, task: 'Task', task_state: Dict, agent_capability: np.ndarray) -> Dict:
        """
        Make a routing decision for a task.
        
        Args:
            task: The task to be routed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            
        Returns:
            Routing decision
        """
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(task, k=5)
        
        # Analyze task and make routing decision
        # This would involve LLM reasoning in a real implementation
        task_capability = task.get_capability_requirements()
        
        # Simplified decision logic
        if self._can_execute(task_capability, agent_capability):
            return {'type': 'execute'}
        elif self._should_split(task):
            subtasks = self._decompose_task(task)
            return {'type': 'split', 'subtasks': subtasks}
        else:
            # Forward to another agent (in a real implementation, this would use the weight matrix)
            return {'type': 'forward', 'target_agent': 'agent_2'}
    
    def _can_execute(self, task_capability, agent_capability):
        """
        Determine if the agent can execute the task based on capability match.
        """
        # Simplified implementation
        similarity = np.dot(task_capability, agent_capability) / (np.linalg.norm(task_capability) * np.linalg.norm(agent_capability))
        return similarity > 0.8
    
    def _should_split(self, task):
        """
        Determine if the task should be split into subtasks.
        """
        # Simplified implementation
        return task.complexity > 5
    
    def _decompose_task(self, task):
        """
        Decompose a task into subtasks.
        """
        # Simplified implementation
        # In a real implementation, this would use LLM reasoning
        return task.get_default_decomposition()


class Executor:
    """
    Executor component of an agent, responsible for responding to executing queries through operations and tools.
    """
    
    def __init__(self, memory_size: int = 100):
        """
        Initialize an executor.
        
        Args:
            memory_size: Size of the memory module
        """
        self.memory = MemoryModule(max_size=memory_size)
    
    def execute_task(self, task: 'Task', task_state: Dict, agent_capability: np.ndarray) -> Dict:
        """
        Execute a task.
        
        Args:
            task: The task to be executed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            
        Returns:
            Execution result
        """
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(task, k=5)
        
        # Execute the task
        # This would involve LLM reasoning and tool use in a real implementation
        result = {
            'completed': True,
            'output': f"Executed task {task.id}",
            'confidence': 0.9
        }
        
        # Store the execution experience in memory
        self.memory.store(task, result)
        
        return result


class MemoryModule:
    """
    Memory module for storing and retrieving agent experiences.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize a memory module.
        
        Args:
            max_size: Maximum number of memories to store
        """
        self.max_size = max_size
        self.memories = []
    
    def store(self, task: 'Task', result: Dict):
        """
        Store a memory.
        
        Args:
            task: The task that was processed
            result: The result of processing the task
        """
        memory = {
            'task': task,
            'result': result,
            'timestamp': np.datetime64('now')
        }
        
        self.memories.append(memory)
        
        # Trim if exceeding max size
        if len(self.memories) > self.max_size:
            self.memories = self.memories[-self.max_size:]
    
    def retrieve(self, task: 'Task', k: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories for a task.
        
        Args:
            task: The task to retrieve memories for
            k: Number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        # In a real implementation, this would use semantic similarity
        # Simplified implementation: return the k most recent memories
        return self.memories[-k:]


class Task:
    """
    Task to be processed by the AgentNet system.
    """
    
    def __init__(self, task_id: str, description: str, complexity: float, capability_vector: np.ndarray, parameters: Dict = None):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            description: Description of the task
            complexity: Complexity score of the task
            capability_vector: Vector representing the capability requirements of the task
            parameters: Additional parameters for the task
        """
        self.id = task_id
        self.description = description
        self.complexity = complexity
        self.capability_vector = capability_vector
        self.parameters = parameters or {}
    
    def get_capability_requirements(self) -> np.ndarray:
        """
        Get the capability requirements of the task.
        
        Returns:
            Capability vector
        """
        return self.capability_vector
    
    def get_parameters(self) -> Dict:
        """
        Get the parameters of the task.
        
        Returns:
            Task parameters
        """
        return self.parameters
    
    def get_default_decomposition(self) -> List['Task']:
        """
        Get a default decomposition of the task into subtasks.
        
        Returns:
            List of subtasks
        """
        # Simplified implementation
        # In a real implementation, this would be more sophisticated
        subtasks = []
        for i in range(2):  # Split into 2 subtasks by default
            subtask_id = f"{self.id}_sub{i}"
            subtask_description = f"Subtask {i} of {self.description}"
            subtask_complexity = self.complexity / 2
            subtask_capability = self.capability_vector  # Same capability requirements for simplicity
            subtask = Task(subtask_id, subtask_description, subtask_complexity, subtask_capability)
            subtasks.append(subtask)
        return subtasks