# --- MODIFIED FILE: DAGTransformer.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax


# ==============================================================================
#                      Module 1: DAG Positional Encoding (DAGPE)
# ==============================================================================
class DAGPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # d_model: Hidden dimension (feature dimension)
        # max_len: Expected maximum depth

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, so it won't be considered a model parameter
        # but will be moved with the model (e.g., to(device))
        self.register_buffer('pe', pe)

    def forward(self, depths):
        """
        Args:
            depths (Tensor): Tensor of depths with shape [num_nodes].
        Returns:
            Tensor: Positional encoding with shape [num_nodes, d_model].
        """
        # Index the positional encodings from the pre-computed table based on depths
        return self.pe[depths]


# ==============================================================================
#                      Module 2: DAG Attention Convolution (DAGRA)
# ==============================================================================
class DAGAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1, **kwargs):
        # aggr='add' means neighbor information is aggregated by summation
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # Ensure the output dimension is divisible by the number of heads
        assert out_channels % heads == 0
        self.d_k = out_channels // heads

        # Linear layers to generate Q, K, V
        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)
        self.lin_out = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E_reachable] (this is the reachability_edge_index)

        # 1. Linear projection & split into multiple heads
        # Q, K, V shape: [N, heads, d_k]
        Q = self.lin_q(x).view(-1, self.heads, self.d_k)
        K = self.lin_k(x).view(-1, self.heads, self.d_k)
        V = self.lin_v(x).view(-1, self.heads, self.d_k)

        # 2. Start message passing
        # self.propagate will call message, aggregate, and update methods
        # We pass K and V to the message method
        out = self.propagate(edge_index, q=Q, k=K, v=V, size=None)

        # 3. Post-processing
        out = out.view(-1, self.out_channels)  # Concatenate heads
        out = self.lin_out(out)  # Final linear layer

        return out

    def message(self, q_i, k_j, v_j, edge_index, size):
        # q_i: Q vector of the target node, shape [E_reachable, heads, d_k]
        # k_j, v_j: K, V vectors of the source node, shape [E_reachable, heads, d_k]

        # 1. Calculate attention score: alpha = (q_i * k_j) / sqrt(d_k)
        alpha = (q_i * k_j).sum(dim=-1) / (self.d_k ** 0.5)

        # 2. Apply softmax to the attention scores
        # Softmax is performed based on the incoming edges of the target node (group by target node)
        alpha = softmax(alpha, edge_index[1], num_nodes=size[1])
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 3. Weight V vectors with the attention scores
        # alpha.unsqueeze(-1) changes its shape to [E_reachable, heads, 1] for multiplication with V
        return v_j * alpha.unsqueeze(-1)


# ==============================================================================
#                      Module 3: DAG Transformer Block
# ==============================================================================
class DAGTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = DAGAttentionConv(d_model, d_model, heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, data):
        x, reachability_edge_index = data.x, data.reachability_edge_index

        # Pre-Layer Normalization (Pre-LN) structure
        x_norm = self.ln1(x)
        # Attention + Residual Connection
        attn_out = self.attn(x_norm, reachability_edge_index)
        x = x + attn_out  # Residual connection

        x_norm = self.ln2(x)
        # FFN + Residual Connection
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out # Residual connection

        data.x = x  # Update the node features in the data object
        return data


# ==============================================================================
#                      Module 4: The Final DAGTransformer Model
# ==============================================================================
# [MODIFIED] Changed to a pure encoder architecture
class DAGTransformer(nn.Module):
    # __init__ remains the same as other modules
    def __init__(self, in_dim, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.node_emb = nn.Linear(in_dim, d_model)
        self.pos_encoder = DAGPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DAGTransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, data):
        # data is a DataBatch object from the DataLoader

        # 1. Initial embedding
        x = self.node_emb(data.x)

        # 2. Add positional encoding
        # Now that data.depths is populated, the dimensions match, and this line works correctly
        pos_encoding = self.pos_encoder(data.depths)
        x = x + pos_encoding

        data.x = x

        # 3. Pass through all Transformer layers
        for layer in self.layers:
            data = layer(data)

        # --- Core modification is here ---
        # 4. **Masked Graph-level Pooling (Masked Readout)**
        # We only perform pooling on the real nodes (where the mask value is True)

        # Filter out the features of the real nodes from data.x
        real_node_features = data.x[data.mask]

        # Similarly, filter out the batch indices corresponding to the real nodes from data.batch
        batch_for_real_nodes = data.batch[data.mask]

        # Perform global average pooling on the filtered real nodes
        graph_repr = global_mean_pool(real_node_features, batch_for_real_nodes)
        # --- End of modification ---

        # 5. Return the graph-level representation
        return graph_repr


if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data
    # Assuming these utils are in a local 'utils' directory
    from utils.graph_utils import adj2edge_index, get_node_depths, get_reachability_edge_index
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    """
    Dataset construction for quantum circuits:
    Available: adj + ops + label
    Missing: edge_index + depths + reachability_edge_index
    How to construct them:
    1. edge_index: Extract edge indices from the adjacency matrix, the `adj2edge_index` function already exists.
    2. depths: Calculate the depth of each node (longest path length), implemented by the `get_node_depths` function.
    3. reachability_edge_index: Construct the reachability_edge_index, implemented by the `get_reachability_edge_index` function.
    """

    # Adjacency matrix
    adj_matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # Build a directed graph
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', arrows=True)
    plt.title("DAG Graph Visualization")
    # plt.show() # Commented out for easy execution in a non-GUI environment

    print(f"Adjacency Matrix:\n{adj_matrix}")

    # edge_index represents the indices of edges, from which node to which node
    edge_index = adj2edge_index(adj_matrix)
    print(f"Edge Index:\n{edge_index}")

    x = torch.randn(4, 5)
    batch = torch.zeros(4, dtype=torch.long)

    # --- Construct the data required for DAGTransformer ---
    # Create a temporary Data object to run the graph utility functions
    temp_data = Data(edge_index=edge_index, num_nodes=adj_matrix.shape[0])
    depths = get_node_depths(temp_data)
    reachability_edge_index = get_reachability_edge_index(temp_data)

    data = Data(x=x, edge_index=edge_index, depths=depths, batch=batch, reachability_edge_index=reachability_edge_index)
    print("\nConstructed Data object:")
    print(data)

    model = DAGTransformer(in_dim=5, d_model=8, num_heads=2, num_layers=2, dropout=0.1)
    # The original code would fail here without a 'mask' attribute.
    # To make this example runnable, we assume all nodes are real nodes.
    data.mask = torch.ones(data.num_nodes, dtype=torch.bool)
    output = model(data)
    print("\nModel Output (Graph Representation) Shape:", output.shape)  # Should be [1, 8]
    print("Model Output:", output)