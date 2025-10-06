import numpy as np

def pad_adj_ops(adj_list, ops_list, max_nodes):

    padded_adj_list = []
    padded_ops_list = []
    for adj, ops in zip(adj_list, ops_list):
        num_nodes = adj.shape[0]
        padded_adj = np.zeros((max_nodes, max_nodes))
        padded_adj[:num_nodes, :num_nodes] = adj
        padded_ops = np.zeros((max_nodes, ops.shape[1]))
        padded_ops[:num_nodes, :] = ops
        padded_adj_list.append(padded_adj)
        padded_ops_list.append(padded_ops)
    return np.stack(padded_adj_list), np.stack(padded_ops_list)