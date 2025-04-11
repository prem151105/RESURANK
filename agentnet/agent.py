# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems
# Agent Implementation with Dual-Role Architecture

import numpy as np
from typing import List, Dict, Any, Optional


class MemoryModule:
    """
    Memory module for storing and retrieving agent experiences through the RAG mechanism.
    
    Both router and executor components maintain fixed-size memory modules (Mrou and Mexe),
    providing agents with powerful adaptive evolutionary capabilities.
    
    Each entry in the memory module is a local step fragment represented as: fr = (or,cr,ar), where:
    - or denotes the observation (query of the corresponding task)
    - cr represents the context (partial trajectory before this step)
    - ar is the action or response of the agent
    """
    
    def __init__(self, max_size: int = 100, module_type: str = "rou"):
        """
        Initialize a memory module.
        
        Args:
            max_size: Maximum number of memories to store
            module_type: Type of module ("rou" for router, "exe" for executor)
        """
        self.max_size = max_size
        self.module_type = module_type  # "rou" or "exe"
        self.memories = []
    
    def store(self, observation: Any, context: Any, action: Any, metadata: Dict = None):
        """
        Store a memory entry as a trajectory fragment.
        
        Args:
            observation: The query or task that was processed (or)
            context: The context or partial trajectory before this step (cr)
            action: The action or response of the agent (ar)
            metadata: Additional metadata about the memory
        """
        fragment = {
            'observation': observation,  # or
            'context': context,          # cr
            'action': action,            # ar
            'metadata': metadata or {},
            'timestamp': np.datetime64('now'),
            'usage_count': 0,
            'relevance_score': 1.0  # Initial relevance score
        }
        
        self.memories.append(fragment)
        
        # If exceeding max size, prune the least useful memory
        if len(self.memories) > self.max_size:
            self._prune_memory()
    
    def retrieve(self, observation: Any, context: Any, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories for a query using the RAG mechanism.
        
        Implements the Select(Mr_i, tm+1, k) function from the paper, which selects
        the k most relevant fragments based on semantic similarity.
        
        Args:
            observation: The current observation (query)
            context: The current context (partial trajectory)
            k: Number of memories to retrieve
            
        Returns:
            List of relevant memory fragments
        """
        if not self.memories:
            return []
        
        # Embed the current observation and context
        current_embedding = self._embed(observation, context)
        
        # Calculate similarity scores for all memories
        similarities = []
        for i, fragment in enumerate(self.memories):
            # Embed the fragment's observation and context
            fragment_embedding = self._embed(fragment['observation'], fragment['context'])
            
            # Calculate similarity
            similarity = self._calculate_similarity(current_embedding, fragment_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the k most similar fragments
        retrieved_fragments = []
        for i, similarity in similarities[:k]:
            fragment = self.memories[i].copy()
            fragment['similarity'] = similarity
            
            # Update usage count for the retrieved fragment
            self.memories[i]['usage_count'] += 1
            
            retrieved_fragments.append(fragment)
        
        return retrieved_fragments
    
    def _embed(self, observation: Any, context: Any) -> np.ndarray:
        """
        Semantic embedding function that projects the input context into a high-dimensional vector space.
        
        In a production implementation, this would use a language model or sentence transformer.
        
        Args:
            observation: The observation to embed
            context: The context to embed
            
        Returns:
            Embedding vector
        """
        # Simplified implementation: create a random embedding vector
        # In a real implementation, this would use a language model or sentence transformer
        
        # Convert observation and context to strings if they aren't already
        obs_str = str(observation)
        ctx_str = str(context)
        
        # Create a simple hash-based embedding (for demonstration purposes only)
        combined = obs_str + ctx_str
        hash_val = hash(combined) % 10000
        np.random.seed(hash_val)
        embedding = np.random.rand(128)  # 128-dimensional embedding
        np.random.seed(None)  # Reset the seed
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embedding vectors using cosine similarity.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / \
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Ensure the similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def _prune_memory(self):
        """
        Prune the least useful memory based on a utility score.
        
        The utility score is calculated based on:
        - Recency (more recent memories are more valuable)
        - Usage frequency (more frequently used memories are more valuable)
        - Relevance (memories with higher relevance scores are more valuable)
        """
        if not self.memories:
            return
        
        # Calculate utility scores for all memories
        current_time = np.datetime64('now')
        max_age = np.timedelta64(30, 'D')  # Maximum age to consider (30 days)
        
        utilities = []
        for i, fragment in enumerate(self.memories):
            # Calculate recency factor (newer is better)
            age = current_time - fragment['timestamp']
            recency = 1.0 - min(1.0, age / max_age)
            
            # Usage frequency factor
            usage = min(1.0, fragment['usage_count'] / 10.0)  # Cap at 10 uses
            
            # Relevance factor
            relevance = fragment['relevance_score']
            
            # Combined utility score (weighted sum)
            utility = 0.4 * recency + 0.3 * usage + 0.3 * relevance
            utilities.append((i, utility))
        
        # Find the memory with the lowest utility
        min_utility_idx = min(utilities, key=lambda x: x[1])[0]
        
        # Remove the memory with the lowest utility
        self.memories.pop(min_utility_idx)


class Router:
    """
    Router component of an agent, responsible for analyzing received routing queries 
    and making routing decisions.
    
    The router leverages a substantial LLM that uses its extensive knowledge and 
    understanding to make routing decisions. It follows the ReAct (Reasoning + Acting)
    framework, which empowers it to reason about a given query and its context before
    deciding appropriate actions.
    """
    
    def __init__(self, memory_size: int = 100):
        """
        Initialize a router.
        
        Args:
            memory_size: Size of the memory module
        """
        self.memory = MemoryModule(max_size=memory_size, module_type="rou")
    
    def make_decision(self, task: Any, task_state: Dict, agent_capability: np.ndarray, network_weights: Dict = None) -> Dict:
        """
        Make a routing decision for a task.
        
        Args:
            task: The task to be routed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            network_weights: Weights of connections to other agents
            
        Returns:
            Routing decision with format:
            - type: 'forward', 'split', or 'execute'
            - additional parameters based on type
        """
        # Retrieve relevant fragments from memory using RAG
        relevant_fragments = self.memory.retrieve(task, task_state, k=5)
        
        # First, reason about the task using the ReAct framework
        reasoning = self._reason(task, task_state, agent_capability, network_weights, relevant_fragments)
        
        # Then, act based on the reasoning
        decision = self._act(task, task_state, agent_capability, network_weights, reasoning, relevant_fragments)
        
        # Store this experience in memory
        self.memory.store(
            observation=task,
            context=task_state,
            action=decision,
            metadata={
                'reasoning': reasoning,
                'agent_capability': agent_capability.tolist() if isinstance(agent_capability, np.ndarray) else agent_capability
            }
        )
        
        return decision
        
    def _reason(self, task: Any, task_state: Dict, agent_capability: np.ndarray, 
               network_weights: Dict, relevant_fragments: List[Dict]) -> str:
        """
        Reasoning function for the router module (Rai(tm+1, rou) in the paper).
        
        Args:
            task: The task to be routed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            network_weights: Weights of connections to other agents
            relevant_fragments: Retrieved memory fragments
            
        Returns:
            Reasoning output as a string
        """
        # In a real implementation, this would use an LLM to generate reasoning
        # based on the task, context, and retrieved fragments
        
        # Extract task capability
        task_capability = self._extract_task_capability(task)
        
        # Simplified reasoning logic
        reasoning = f"Analyzing task {getattr(task, 'id', 'unknown')}. "
        
        # Add reasoning based on capability match
        similarity = self._calculate_similarity(task_capability, agent_capability)
        if similarity > 0.8:
            reasoning += f"This agent has high capability match ({similarity:.2f}). "
        else:
            reasoning += f"This agent has low capability match ({similarity:.2f}). "
        
        # Add reasoning based on task complexity
        complexity = getattr(task, 'complexity', 1)
        if complexity > 5:
            reasoning += f"Task complexity is high ({complexity:.2f}). Consider splitting. "
        else:
            reasoning += f"Task complexity is manageable ({complexity:.2f}). "
        
        # Add reasoning based on network weights if available
        if network_weights:
            best_agent = max(network_weights.items(), key=lambda x: x[1])[0]
            best_weight = network_weights[best_agent]
            reasoning += f"Best connected agent is {best_agent} with weight {best_weight:.2f}. "
        
        # Add reasoning based on relevant fragments if available
        if relevant_fragments:
            reasoning += "Found relevant past experiences. "
            for i, fragment in enumerate(relevant_fragments[:2]):  # Only use top 2 for brevity
                reasoning += f"Fragment {i+1} suggests {fragment['action']['type']} action. "
        
        return reasoning
    
    def _act(self, task: Any, task_state: Dict, agent_capability: np.ndarray, 
            network_weights: Dict, reasoning: str, relevant_fragments: List[Dict]) -> Dict:
        """
        Acting function for the router module (Aai(tm+1, rou) in the paper).
        
        Args:
            task: The task to be routed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            network_weights: Weights of connections to other agents
            reasoning: Output from the reasoning function
            relevant_fragments: Retrieved memory fragments
            
        Returns:
            Routing decision
        """
        # In a real implementation, this would use an LLM to generate actions
        # based on the reasoning output and other inputs
        
        # Extract task capability
        task_capability = self._extract_task_capability(task)
        
        # Use the reasoning to inform the decision
        task_capability = self._extract_task_capability(task)
        
        # Check if there are relevant fragments that can guide the decision
        if relevant_fragments:
            # Find the most similar fragment with high similarity
            for fragment in relevant_fragments:
                if fragment.get('similarity', 0) > 0.8:
                    # Follow the action from the most similar past experience
                    return fragment['action']
        
        # If no relevant past experiences, use capability matching
        if self._can_execute(task_capability, agent_capability):
            return {'type': 'execute'}
        elif self._should_split(task):
            subtasks = self._decompose_task(task)
            return {'type': 'split', 'subtasks': subtasks}
        else:
            # Forward to another agent
            target_agent = self._select_best_agent(task_capability, network_weights)
            return {'type': 'forward', 'target_agent': target_agent}
    
    def _extract_task_capability(self, task):
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
        Calculate similarity between two capability vectors using cosine similarity.
        
        Args:
            capability1: First capability vector
            capability2: Second capability vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Cosine similarity
        similarity = np.dot(capability1, capability2) / \
                    (np.linalg.norm(capability1) * np.linalg.norm(capability2))
        
        # Ensure the similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def _can_execute(self, task_capability, agent_capability):
        """
        Determine if the agent can execute the task based on capability match.
        
        Args:
            task_capability: Capability requirements of the task
            agent_capability: Capability vector of the agent
            
        Returns:
            True if the agent can execute the task, False otherwise
        """
        # Calculate similarity between task capability and agent capability
        similarity = np.dot(task_capability, agent_capability) / \
                    (np.linalg.norm(task_capability) * np.linalg.norm(agent_capability))
        return similarity > 0.8  # Arbitrary threshold
    
    def _should_split(self, task):
        """
        Determine if the task should be split into subtasks.
        
        Args:
            task: The task to check
            
        Returns:
            True if the task should be split, False otherwise
        """
        # In a real implementation, this would use LLM reasoning to determine
        # if the task is complex enough to warrant splitting
        # Simplified implementation
        return getattr(task, 'complexity', 1) > 5  # Arbitrary threshold
    
    def _decompose_task(self, task):
        """
        Decompose a task into subtasks.
        
        Args:
            task: The task to decompose
            
        Returns:
            List of subtasks
        """
        # In a real implementation, this would use LLM reasoning to decompose
        # the task into subtasks
        # Simplified implementation
        return getattr(task, 'get_default_decomposition', lambda: [])()
    
    def _select_best_agent(self, task_capability, network_weights):
        """
        Select the best agent to forward the task to.
        
        Args:
            task_capability: Capability requirements of the task
            network_weights: Weights of connections to other agents
            
        Returns:
            ID of the best agent
        """
        # In a real implementation, this would use the network weights and
        # agent capabilities to select the best agent
        # Simplified implementation
        if network_weights:
            # Find the agent with the highest weight
            return max(network_weights.items(), key=lambda x: x[1])[0]
        else:
            # Default agent if no weights are provided
            return 'agent_2'


class Executor:
    """
    Executor component of an agent, responsible for responding to executing queries 
    through operations and tools.
    
    The executor leverages a substantial LLM that uses its extensive knowledge and 
    understanding to solve specific problems. It follows the ReAct (Reasoning + Acting)
    framework, which empowers it to reason about a given query and its context before
    deciding appropriate actions.
    """
    
    def __init__(self, memory_size: int = 100, tool_registry=None):
        """
        Initialize an executor.
        
        Args:
            memory_size: Size of the memory module
            tool_registry: Registry of available tools
        """
        self.memory = MemoryModule(max_size=memory_size, module_type="exe")
        self.tools = {}  # Available tools for the executor
        self.tool_registry = tool_registry
        
        # Register tools from registry if provided
        if self.tool_registry and hasattr(self.tool_registry, 'tools'):
            for tool_name, tool in self.tool_registry.tools.items():
                self.register_tool_instance(tool)
    
    def register_tool(self, tool_name: str, tool_function: callable):
        """
        Register a tool for the executor to use.
        
        Args:
            tool_name: Name of the tool
            tool_function: Function implementing the tool
        """
        self.tools[tool_name] = tool_function
        
    def register_tool_instance(self, tool):
        """
        Register a tool instance from the tool registry.
        
        Args:
            tool: Tool instance to register
        """
        self.tools[tool.name] = tool.execute
        
    def get_available_tools(self):
        """
        Get a list of available tools.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def execute_task(self, task: Any, task_state: Dict, agent_capability: np.ndarray) -> Dict:
        """
        Execute a task.
        
        Args:
            task: The task to be executed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            
        Returns:
            Execution result
        """
        # Retrieve relevant fragments from memory using RAG
        relevant_fragments = self.memory.retrieve(task, task_state, k=5)
        
        # First, reason about the task using the ReAct framework
        reasoning = self._reason(task, task_state, agent_capability, relevant_fragments)
        
        # Then, act based on the reasoning
        result = self._act(task, task_state, agent_capability, reasoning, relevant_fragments)
        
        # Store this experience in memory
        self.memory.store(
            observation=task,
            context=task_state,
            action=result,
            metadata={
                'reasoning': reasoning,
                'agent_capability': agent_capability.tolist() if isinstance(agent_capability, np.ndarray) else agent_capability,
                'tools_used': result.get('tools_used', [])
            }
        )
        
        return result
        
    def _reason(self, task: Any, task_state: Dict, agent_capability: np.ndarray, 
                relevant_fragments: List[Dict]) -> str:
        """
        Reasoning function for the executor module (Rai(tm+1, exe) in the paper).
        
        Args:
            task: The task to be executed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            relevant_fragments: Retrieved memory fragments
            
        Returns:
            Reasoning output as a string
        """
        # In a real implementation, this would use an LLM to generate reasoning
        # based on the task, context, and retrieved fragments
        
        # Simplified reasoning logic
        reasoning = f"Analyzing execution requirements for task {getattr(task, 'id', 'unknown')}. "
        
        # Add reasoning based on available tools
        if self.tools:
            reasoning += f"Available tools: {', '.join(self.tools.keys())}. "
        else:
            reasoning += "No specialized tools available. Will use general problem-solving. "
        
        # Add reasoning based on relevant fragments if available
        if relevant_fragments:
            reasoning += "Found relevant past experiences. "
            for i, fragment in enumerate(relevant_fragments[:2]):  # Only use top 2 for brevity
                if 'tools_used' in fragment.get('metadata', {}):
                    tools = fragment['metadata']['tools_used']
                    if tools:
                        reasoning += f"Fragment {i+1} used tools: {', '.join(tools)}. "
                reasoning += f"Fragment {i+1} had confidence: {fragment['action'].get('confidence', 'unknown')}. "
        
        return reasoning
    
    def _act(self, task: Any, task_state: Dict, agent_capability: np.ndarray, 
             reasoning: str, relevant_fragments: List[Dict]) -> Dict:
        """
        Acting function for the executor module (Aai(tm+1, exe) in the paper).
        
        Args:
            task: The task to be executed
            task_state: Current state of the task
            agent_capability: Capability vector of the agent
            reasoning: Output from the reasoning function
            relevant_fragments: Retrieved memory fragments
            
        Returns:
            Execution result
        """
        # In a real implementation, this would use an LLM to generate actions
        # based on the reasoning output and other inputs
        
        # Check if there are relevant fragments that can guide the execution
        tools_to_use = []
        confidence = 0.7  # Default confidence
        
        if relevant_fragments:
            # Find the most similar fragment with high similarity
            for fragment in relevant_fragments:
                if fragment.get('similarity', 0) > 0.8:
                    # Use the tools from the most similar past experience
                    if 'tools_used' in fragment.get('metadata', {}):
                        tools_to_use = fragment['metadata']['tools_used']
                    # Adjust confidence based on past experience
                    if 'confidence' in fragment['action']:
                        confidence = fragment['action']['confidence']
                    break
        
        # Select tools based on the task if none were selected from fragments
        if not tools_to_use and self.tools:
            tools_to_use = self._select_tools(task)
        
        # Execute the selected tools (in a real implementation)
        # For now, just simulate tool execution
        
        # Prepare the result
        result = {
            'completed': True,
            'output': f"Executed task {getattr(task, 'id', 'unknown')} using {len(tools_to_use)} tools",
            'confidence': confidence,
            'tools_used': tools_to_use
        }
        
        return result
    
    def _select_tools(self, task):
        """
        Select appropriate tools for executing a task.
        
        Args:
            task: The task to select tools for
            
        Returns:
            List of selected tool names
        """
        # In a real implementation, this would use LLM reasoning to select
        # appropriate tools based on the task requirements
        # Simplified implementation
        return list(self.tools.keys())


class Agent:
    """
    Agent in the AgentNet system with dual-role architecture.
    
    Each agent contains two key components:
    - Router (rou): Responsible for analyzing received routing queries and making routing decisions
    - Executor (exe): Responsible for responding to executing queries through operations and tools
    
    Both components are underpinned by a substantial LLM that leverages its extensive knowledge and
    understanding to solve specific problems.
    
    Agents can use various tools to enhance their capabilities, similar to Manus AI and GenSpark.
    These tools extend the agent's ability to interact with external systems and perform specialized tasks.
    """
    
    def __init__(self, agent_id: str, capability: np.ndarray, router_memory_size: int = 100, 
                 executor_memory_size: int = 100, tool_registry=None):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            capability: Vector representing the agent's capabilities
            router_memory_size: Size of the router memory module
            executor_memory_size: Size of the executor memory module
            tool_registry: Registry of available tools for the agent to use
        """
        self.id = agent_id
        self.capability = capability
        
        # Initialize router and executor components with their respective memory modules (Mrou and Mexe)
        self.router = Router(memory_size=router_memory_size)
        self.executor = Executor(memory_size=executor_memory_size, tool_registry=tool_registry)
        
        # Network connections and weights
        self.connections = {}  # {agent_id: weight}
        
        # Store tool registry reference
        self.tool_registry = tool_registry
    
    def update_connection(self, target_agent_id: str, weight: float):
        """
        Update the weight of a connection to another agent.
        
        Args:
            target_agent_id: ID of the target agent
            weight: New connection weight
        """
        self.connections[target_agent_id] = weight
    
    def route(self, task: Any, task_state: Dict) -> Dict:
        """
        Make a routing decision for a task.
        
        Args:
            task: The task to be routed
            task_state: Current state of the task
            
        Returns:
            Routing decision
        """
        return self.router.make_decision(task, task_state, self.capability, self.connections)
    
    def execute(self, task: Any, task_state: Dict) -> Dict:
        """
        Execute a task.
        
        Args:
            task: The task to be executed
            task_state: Current state of the task
            
        Returns:
            Execution result
        """
        return self.executor.execute_task(task, task_state, self.capability)