from graphviz import Digraph
from mingrad.engine import Variable

def trace(root):
    """
    Trace the computational graph of a Variable.

    This function builds the set of nodes and edges of the computational graph
    starting from the given root Variable.

    Args:
        root (Variable): The root Variable from which to start tracing the computational graph.

    Returns:
        tuple: A tuple containing two sets:
            - nodes (set): A set of all Variables (nodes) in the computational graph.
            - edges (set): A set of tuples representing the edges in the computational graph,
                           where each tuple is of the form (child, parent).
    """
    nodes, edges = set(), set()
    
    def build(v):
        """
        Recursively build the set of nodes and edges for the computational graph.

        Args:
            v (Variable): The current Variable being processed.

        This function adds the given Variable to the set of nodes and recursively
        processes its children, adding edges from each child to the given Variable.
        """
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    Draw the computational graph of a Variable using Graphviz.

    Args:
        root (Variable): The root Variable from which to start tracing the computational graph.
        format (str): The format of the output graph. Supported formats include 'png', 'svg', etc.
        rankdir (str): The direction of the graph layout. 'TB' for top-to-bottom, 'LR' for left-to-right.

    Returns:
        Digraph: A Graphviz Digraph object representing the computational graph.
    """
    assert rankdir in ['LR', 'TB'], "rankdir must be either 'LR' (left to right) or 'TB' (top to bottom)"
    
    # Trace the computational graph to get nodes and edges
    nodes, edges = trace(root)
    
    # Create a Graphviz Digraph object
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    # Add nodes to the graph
    for n in nodes:
        dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            # Add operation node
            dot.node(name=str(id(n)) + n._op, label=n._op)
            # Connect operation to result node
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    # Add edges to the graph
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot