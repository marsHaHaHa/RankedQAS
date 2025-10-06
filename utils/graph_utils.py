"""
Utility functions for graph operations.
Some of these may be wrappers around functions from the torch_geometric library.
"""
import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from scipy.sparse import coo_matrix
import numpy as np


def adj2edge_index(adj_arr):
    """
    Converts an adjacency matrix to the COO (Coordinate) edge index format.

    :param adj_arr: The input adjacency matrix (NumPy array).
    :return: edge_index: The edge index in COO format (torch.Tensor).

    An alternative method using torch_geometric.utils:
    from torch_geometric.utils import dense_to_sparse
    edge_index, _ = dense_to_sparse(torch.from_numpy(adj_arr))
    """
    coo_A = coo_matrix(adj_arr)
    edge_index = np.array([coo_A.row, coo_A.col])
    # Use torch.from_numpy for better performance
    return torch.from_numpy(edge_index).to(torch.long)


def get_node_depths(data: Data) -> torch.Tensor:
    """
    Calculates the depth of each node in a Directed Acyclic Graph (DAG).

    This function first converts a PyG graph to a NetworkX graph and then utilizes
    topological sorting to efficiently compute the depth of each node.

    - The depth of source nodes (in-degree of 0) is defined as 0.
    - The depth of any other node is 1 + max(depth of all its parent nodes).

    Args:
    - data (torch_geometric.data.Data): A PyG Data object, which must include:
        - edge_index: [2, E], representing the edges of the DAG.
        - num_nodes: integer, the total number of nodes in the graph.

    Returns:
    - torch.Tensor: A 1D tensor of shape [num_nodes], where the value at index i
                      represents the depth of node i. The tensor's dtype is torch.long.
    """
    # 1. Convert the PyG graph to a NetworkX directed graph to use its graph algorithms.
    # Note: It's crucial to set to_undirected=False.
    G = to_networkx(data, to_undirected=False)

    # 2. Check if the graph is indeed a DAG.
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph must be a Directed Acyclic Graph (DAG)!")

    # 3. Get the list of nodes in topological order.
    # Topological sort ensures that when we process a node, all its predecessors (parents)
    # have already been processed.
    topological_nodes = list(nx.topological_sort(G))

    # 4. Initialize a tensor to store the depth of each node.
    depths = torch.full((data.num_nodes,), -1, dtype=torch.long)

    # 5. Iterate through the nodes in topological order and calculate their depths.
    for node in topological_nodes:
        # Get all parent nodes (predecessors) of the current node.
        # G.predecessors(node) returns an iterator.
        parents = list(G.predecessors(node))

        if not parents:
            # If there are no parents, it's a source node, so its depth is 0.
            depths[node] = 0
        else:
            # If there are parents, find the maximum depth among them.
            # Due to the topological sort, the depths of all parents have already been computed.
            parent_depths = depths[parents]
            max_parent_depth = torch.max(parent_depths)
            depths[node] = max_parent_depth + 1

    return depths


def get_reachability_edge_index(data):
    """
    Computes the transitive closure of a DAG and returns a symmetric (undirected)
    reachability edge_index. This implementation is based on the Floyd-Warshall
    algorithm, with a complexity of O(N^3). It is not suitable for large graphs
    but is clear for demonstration on smaller ones.

    Args:
        data (torch_geometric.data.Data): A data object containing the original
                                          edge_index and num_nodes.

    Returns:
        torch.Tensor: An edge_index of shape [2, E_reachable] representing the
                      symmetric reachability relationship.
    """
    num_nodes = data.num_nodes

    # 1. Initialize the adjacency matrix, with the diagonal as True (a node can reach itself).
    adj = torch.eye(num_nodes, dtype=torch.bool)

    # 2. Populate the matrix with the original direct adjacency relationships.
    row, col = data.edge_index
    adj[row, col] = True

    # 3. Compute the Transitive Closure using the Floyd-Warshall algorithm concept.
    for k in range(num_nodes):  # Intermediate node
        for i in range(num_nodes):  # Start node
            for j in range(num_nodes):  # End node
                # If i can reach k and k can reach j, then i can reach j.
                adj[i, j] = adj[i, j] or (adj[i, k] and adj[k, j])

    # 4. Create a symmetric reachability relationship (A | A^T).
    # If i can reach j, OR j can reach i, they are mutually reachable.
    # A | A.t() is an element-wise OR operation.
    reachable_adj = adj | adj.t()

    # 5. Remove self-loops from the adjacency matrix, as self-to-self messages
    # are not typically considered in message passing. A node should not attend to itself
    # unless specifically designed to do so.
    reachable_adj.fill_diagonal_(False)

    # 6. Convert the adjacency matrix back to the edge_index format.
    reachability_edge_index = reachable_adj.nonzero().t().contiguous()

    return reachability_edge_index


def pad_adj_ops(adj_list, ops_list, max_nodes):
    """
    Pads adjacency matrices and node feature matrices to a specified maximum size.
    """
    padded_adj_list, padded_ops_list = [], []
    for adj, ops in zip(adj_list, ops_list):
        num_nodes = adj.shape[0]
        padded_adj = np.zeros((max_nodes, max_nodes))
        padded_adj[:num_nodes, :num_nodes] = adj
        padded_ops = np.zeros((max_nodes, ops.shape[1]))
        padded_ops[:num_nodes, :] = ops
        padded_adj_list.append(padded_adj)
        padded_ops_list.append(padded_ops)
    return np.stack(padded_adj_list), np.stack(padded_ops_list)