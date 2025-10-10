'''
module flow data/node/graph
'''

from typing import Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod

'''
@class FlowData
The data flowing through the edges between FlowNodes
'''


class FlowData:
    """
    FlowData represents data flowing through edges between FlowNodes.
    
    This class encapsulates the intermediate analysis results that flow
    through the analysis workflow. It can contain traces, profiles, or
    other performance data structures.
    
    Attributes:
        m_data: Set of data objects (Trace, Profile, etc.)
    """
    
    def __init__(self) -> None:
        """Initialize a FlowData object with an empty data set."""
        self.m_data: Set[Any] = set()
    
    def get_data(self) -> Set[Any]:
        """
        Get the data set.
        
        Returns:
            Set of data objects
        """
        return self.m_data
    
    def add_data(self, data: Any) -> None:
        """
        Add a data object to the flow data.
        
        Args:
            data: Data object to add (e.g., Trace, Profile)
        """
        self.m_data.add(data)
    
    def remove_data(self, data: Any) -> None:
        """
        Remove a data object from the flow data.
        
        Args:
            data: Data object to remove
        """
        self.m_data.discard(data)
    
    def clear(self) -> None:
        """Clear all data from the flow data."""
        self.m_data.clear()
    
    def size(self) -> int:
        """
        Get the number of data objects.
        
        Returns:
            Number of data objects in the set
        """
        return len(self.m_data)


'''
@class FlowNode
The sub-task node
'''


class FlowNode(ABC):
    """
    FlowNode represents a sub-task node in the analysis workflow.
    
    This is an abstract base class for all analysis tasks. Each node
    processes input data and produces output data. Nodes are connected
    in a FlowGraph to form complex analysis workflows.
    
    Attributes:
        m_inputs: Input flow data for this node
        m_outputs: Output flow data from this node
    """
    
    def __init__(self) -> None:
        """Initialize a FlowNode with empty input and output data."""
        self.m_inputs: FlowData = FlowData()
        self.m_outputs: FlowData = FlowData()
    
    def get_inputs(self) -> FlowData:
        """
        Get the input flow data.
        
        Returns:
            FlowData object containing input data
        """
        return self.m_inputs
    
    def get_outputs(self) -> FlowData:
        """
        Get the output flow data.
        
        Returns:
            FlowData object containing output data
        """
        return self.m_outputs
    
    def set_inputs(self, inputs: FlowData) -> None:
        """
        Set the input flow data.
        
        Args:
            inputs: FlowData object to set as inputs
        """
        self.m_inputs = inputs
    
    def set_outputs(self, outputs: FlowData) -> None:
        """
        Set the output flow data.
        
        Args:
            outputs: FlowData object to set as outputs
        """
        self.m_outputs = outputs
    
    @abstractmethod
    def run(self) -> None:
        """
        Execute the analysis task.
        
        This method must be implemented by subclasses to define
        the specific analysis logic.
        """
        pass


'''
@class FlowGraph
The entire workflow of analysis
'''


class FlowGraph:
    """
    FlowGraph represents the entire workflow of analysis.
    
    This class manages a directed graph of FlowNodes connected by edges.
    It orchestrates the execution of the analysis workflow by running
    nodes in the correct order based on their dependencies.
    
    Attributes:
        m_nodes: List of all FlowNodes in the graph
        m_edges: Dictionary mapping each node to its successor nodes
    """
    
    def __init__(self) -> None:
        """Initialize an empty FlowGraph."""
        self.m_nodes: List[FlowNode] = []
        self.m_edges: Dict[FlowNode, List[FlowNode]] = {}
    
    def add_node(self, node: FlowNode) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: FlowNode to add to the graph
        """
        if node not in self.m_nodes:
            self.m_nodes.append(node)
            self.m_edges[node] = []
    
    def add_edge(self, from_node: FlowNode, to_node: FlowNode) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            from_node: Source node
            to_node: Destination node
        """
        # Ensure both nodes are in the graph
        if from_node not in self.m_nodes:
            self.add_node(from_node)
        if to_node not in self.m_nodes:
            self.add_node(to_node)
        
        # Add edge
        if to_node not in self.m_edges[from_node]:
            self.m_edges[from_node].append(to_node)
    
    def get_nodes(self) -> List[FlowNode]:
        """
        Get all nodes in the graph.
        
        Returns:
            List of all FlowNodes
        """
        return self.m_nodes
    
    def get_successors(self, node: FlowNode) -> List[FlowNode]:
        """
        Get successor nodes of a given node.
        
        Args:
            node: FlowNode to get successors for
            
        Returns:
            List of successor nodes
        """
        return self.m_edges.get(node, [])
    
    def run(self) -> None:
        """
        Execute the workflow.
        
        Runs all nodes in the graph in topological order using Kahn's algorithm,
        ensuring that each node runs after all its dependencies have completed.
        
        Raises:
            RuntimeError: If the graph contains a cycle
        """
        # Build in-degree map (count of incoming edges for each node)
        in_degree: Dict[FlowNode, int] = {node: 0 for node in self.m_nodes}
        
        # Calculate in-degrees by examining all edges
        for node in self.m_nodes:
            for successor in self.m_edges[node]:
                in_degree[successor] += 1
        
        # Find all nodes with no incoming edges (starting nodes)
        queue: List[FlowNode] = []
        for node in self.m_nodes:
            if in_degree[node] == 0:
                queue.append(node)
        
        # Process nodes in topological order
        processed_nodes: List[FlowNode] = []
        
        while queue:
            # Take the next node with no dependencies
            current_node = queue.pop(0)
            processed_nodes.append(current_node)
            
            # Execute the node
            current_node.run()
            
            # Pass outputs to successor nodes
            successors = self.get_successors(current_node)
            for successor in successors:
                # Merge outputs from current node into inputs of successor
                for data in current_node.get_outputs().get_data():
                    successor.get_inputs().add_data(data)
                
                # Reduce in-degree of successor
                in_degree[successor] -= 1
                
                # If successor has no more dependencies, add to queue
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Check if all nodes were processed (detect cycles)
        if len(processed_nodes) != len(self.m_nodes):
            raise RuntimeError(
                f"Graph contains a cycle. Processed {len(processed_nodes)} "
                f"nodes out of {len(self.m_nodes)} total nodes."
            )
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.m_nodes.clear()
        self.m_edges.clear()