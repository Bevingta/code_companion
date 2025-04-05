from matplotlib import pyplot as plt
import networkx as nx
import os
import numpy as np
import torch
from torch_geometric.data import Data
from networkx.drawing.nx_agraph import graphviz_layout

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Location of the current file
GRAPHS_DIR = os.path.join(BASE_DIR, "..", "code_graphs")

# Load GraphML
graph_0 = os.path.join(GRAPHS_DIR, "function_0.graphml")
G = nx.read_graphml(graph_0)

print(G)

###* NODE FEATURE MATRIX *###

"""
### Node Feature Matrix Explanation

The node feature matrix is a numerical representation of all nodes in the graph.
It is a 2D array (tensor) of shape (N, F), where:

- N is the number of nodes.
- F is the number of features per node.

Each row corresponds to a node, and each column represents a specific feature.
Since node attributes in the GraphML file are categorical (e.g., "FunctionDefinition",
"FunctionCall", "ControlStructure", "Variable"), we need to encode them numerically.

#### Example: One-Hot Encoding for Node Types

We define the following node types:
1. FunctionDefinition → [1, 0, 0, 0]
2. FunctionCall → [0, 1, 0, 0]
3. ControlStructure → [0, 0, 1, 0]
4. Variable → [0, 0, 0, 1]

For example, given three nodes:
| Node ID             | Type             | One-Hot Encoding  |
|---------------------|-----------------|-------------------|
| ssl_get_algorithm2 | FunctionDefinition | [1, 0, 0, 0] |
| if_0               | ControlStructure   | [0, 0, 1, 0] |
| alg2               | Variable           | [0, 0, 0, 1] |

This results in the node feature matrix:

X = [
    [1, 0, 0, 0],  # FunctionDefinition
    [0, 0, 1, 0],  # ControlStructure
    [0, 0, 0, 1]   # Variable
]

### Why is the Node Feature Matrix Important?
- It provides structured input features for the Graph Neural Network (GNN).
- The GNN learns patterns from these features combined with the graph's structure.
- More advanced representations (e.g., embeddings for function names or graph-based positional encodings) can improve performance.

"""

def create_a_feature_matrix(G, node_types): #dict node_types, graph G
    node_features = []
    node_mapping = {}  # To map node IDs to indices
    for i, (node, data) in enumerate(G.nodes(data=True)):
        node_mapping[node] = i
        node_type = data.get("type", "Unknown")
        type_vector = [0] * len(node_types)
        if node_type in node_types:
            type_vector[node_types[node_type]] = 1  # One-hot encoding
        node_features.append(type_vector)

    node_features = np.array(node_features)
    return node_features, node_mapping


###* GENERATE EDGE INDEX *###
"""
### Edge Index Explanation

The edge index is a numerical representation of the connections (edges) between nodes in the graph.  
It is stored as a 2D tensor of shape (2, E), where:

- 2 represents the source and target of each edge.
- E is the number of edges in the graph.

Each column in the edge index represents a directed edge (source → target).  
For example, if node 0 connects to node 1, and node 1 connects to node 2,  
the edge index is:

    edge_index = [
        [0, 1],  # Source nodes
        [1, 2]   # Target nodes
    ]

Which means:
- Node 0 → Node 1
- Node 1 → Node 2

### Extracting Edge Index from the GraphML File
In the GraphML file, edges have a `source` and a `target` node.  
We first map node IDs to indices and then construct the edge index.

Example edges from GraphML:

| Source Node         | Target Node | Edge Type  |
|---------------------|------------|------------|
| ssl_get_algorithm2 | alg2       | declares   |
| ssl_get_algorithm2 | if_0       | contains   |

If we assign these node IDs the following indices:
- ssl_get_algorithm2 → 0
- alg2 → 1
- if_0 → 2

Then the edge index becomes:

    edge_index = [
        [0, 0],  # Source nodes
        [1, 2]   # Target nodes
    ]

This means:
- `ssl_get_algorithm2 (0) → alg2 (1)` (declares)
- `ssl_get_algorithm2 (0) → if_0 (2)` (contains)

### Why is the Edge Index Important?
- It defines the **structure** of the graph.
- The GNN uses this to perform **message passing** between connected nodes.
- Combined with the node feature matrix, it allows the model to learn **context-aware representations**.

"""

def generate_edge_index(G, node_mapping):
    edge_index = []
    for source, target, data in G.edges(data=True):
        edge_index.append([node_mapping[source], node_mapping[target]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # Shape: (2, num_edges)
    return edge_index

###* EDGE TYPE *###
'''
### Edge Index Explanation

The edge index is a numerical representation of the connections (edges) between nodes in the graph.  
It is stored as a 2D tensor of shape (2, E), where:

- 2 represents the source and target of each edge.
- E is the number of edges in the graph.

Each column in the edge index represents a directed edge (source → target).  
For example, if node 0 connects to node 1, and node 1 connects to node 2,  
the edge index is:

    edge_index = [
        [0, 1, 1],  # Source nodes
        [1, 2, 3]   # Target nodes
    ]

Which means:
- Node 0 → Node 1
- Node 1 → Node 2
- Node 1 → Node 3

### Extracting Edge Index from the GraphML File
In the GraphML file, edges have a `source` and a `target` node, along with an edge `type`.  
We first map node IDs to indices and then construct the edge index.

### **Edge Types in the GraphML**
The graph contains six types of edges:

| Edge Type           | Meaning |
|---------------------|--------------------------------------------------|
| `declares`         | Function declares a variable or another function |
| `calls`            | Function calls another function                 |
| `contains`         | A function or control structure contains a node  |
| `used_in_condition`| A variable or function is used in a condition    |
| `used_as_parameter`| A variable or function is passed as a parameter  |
| `used_in_body`     | A variable or function appears in a function body |

### **Example Edges from GraphML**
| Source Node        | Target Node      | Edge Type         |
|--------------------|------------------|-------------------|
| ssl_get_algorithm2 | alg2             | declares         |
| ssl_get_algorithm2 | if_0             | contains         |
| if_0               | TLS1_get_version | used_in_condition |
| TLS1_get_version   | s                | used_as_parameter |
| ssl_get_algorithm2 | another_func     | calls            |
| another_func       | var_x            | used_in_body      |

If we assign these node IDs the following indices:
- ssl_get_algorithm2 → 0
- alg2 → 1
- if_0 → 2
- TLS1_get_version → 3
- s → 4
- another_func → 5
- var_x → 6

Then the **edge index** becomes:

    edge_index = [
        [0, 0, 2, 3, 0, 5],  # Source nodes
        [1, 2, 3, 4, 5, 6]   # Target nodes
    ]

This means:
- `ssl_get_algorithm2 (0) → alg2 (1)` (declares)
- `ssl_get_algorithm2 (0) → if_0 (2)` (contains)
- `if_0 (2) → TLS1_get_version (3)` (used_in_condition)
- `TLS1_get_version (3) → s (4)` (used_as_parameter)
- `ssl_get_algorithm2 (0) → another_func (5)` (calls)
- `another_func (5) → var_x (6)` (used_in_body)

### **Edge Type Encoding**
Since edge types are categorical, we encode them using one-hot or index-based encoding:
edge_types = {
    "declares": 0,
    "calls": 1,
    "contains": 2,
    "used_in_condition": 3,
    "used_as_parameter": 4,
    "used_in_body": 5
}
'''

def generate_edge_types(G, node_mapping):
    """Generate an edge index tensor and corresponding edge type encodings."""
    edge_types = {
        "declares": 0,
        "calls": 1,
        "contains": 2,
        "used_in_condition": 3,
        "used_as_parameter": 4,
        "used_in_body": 5
    }

    edge_index = []
    edge_attr = []

    for source, target, data in G.edges(data=True):
        edge_type = data.get("type", None)
        if edge_type in edge_types:
            edge_index.append([node_mapping[source], node_mapping[target]])  # Source → Target
            one_hot_vector = [0] * len(edge_types)
            one_hot_vector[edge_types[edge_type]] = 1  # One-hot encoding
            edge_attr.append(one_hot_vector)

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # Shape: (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # One-hot encoded edge types

    return edge_index, edge_attr

# Code source: ChatGPT
def visualize_with_graphviz(G):
    plt.figure(figsize=(20, 20))

    # Use 'dot' layout from Graphviz (good for hierarchical graphs)
    pos = graphviz_layout(G, prog='dot')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='gray')

    # Draw labels
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Graph Visualization using Graphviz Layout")
    plt.show()


if __name__ == "__main__":
    # move below to some sort of configs file or something
    node_types = {"FunctionDefinition": 0, "FunctionCall": 1, "ControlStructure" : 2, "Variable" : 3}
    graph_0 = os.path.join(GRAPHS_DIR, "function_17.graphml")
    G = nx.read_graphml(graph_0)
    node_feature_matrix, node_mapping = create_a_feature_matrix(G, node_types)
    edge_index, edge_attr = generate_edge_types(G, node_mapping)

    # Create PyG Data object
    data = Data(
        x=torch.tensor(node_feature_matrix, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # Debug outputs
    print("Node Feature Matrix:\n", data.x)
    print("Node Mapping:\n", node_mapping)
    print("Edge Index:\n", data.edge_index)
    print("Edge Attributes:\n", data.edge_attr)
    # NOTE: You need to install pygraphviz for this to work. https://pygraphviz.github.io/documentation/stable/install.html
    visualize_with_graphviz(G)