import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A general-purpose Multi-Layer Perceptron (MLP) module.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dims (list of int or int): The dimensions of the hidden layers.
                                           e.g., [128, 64] means two hidden layers with 128 and 64 units, respectively.
        output_dim (int): The dimensionality of the output layer.
        activation (str, optional): The type of activation function.
                                    Options are 'relu', 'gelu', 'leaky_relu'. Defaults to 'gelu'.
        use_bn (bool, optional): Whether to add Batch Normalization after hidden layers. Defaults to True.
        dropout (float, optional): The dropout probability. 0 means no dropout. Defaults to 0.5.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation='gelu', use_bn=True, dropout=0.5):
        super().__init__()
        layers = []  # A list to store all the network layers

        prev_dim = input_dim  # Input dimension for the first layer
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        for dim in hidden_dims:
            # Add linear layer
            layers.append(nn.Linear(prev_dim, dim))

            # Add Batch Normalization (optional)
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))

            # Add activation function
            layers.append(self.get_activation(activation))

            # Add Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            prev_dim = dim  # Update the dimension of the previous layer

        # Add the output layer (without activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        # Wrap all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """ Forward pass """
        mlp_output = self.network(x)
        # output = torch.sigmoid(mlp_output)
        return mlp_output

    @staticmethod
    def get_activation(name):
        """ Returns an activation function layer based on its name """
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.1)
        }
        # Default to GELU if the name is not found
        return activations.get(name.lower(), nn.GELU())

class SqueezeOutput(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


if __name__ == '__main__':
    mlp = MLP(10, [128, 64], 1)
    print(mlp)

    mlp = MLP(
        input_dim=256,
        hidden_dims=[128, 64],
        output_dim=1,
        activation='gelu'
    )
    dummy_input = torch.randn(32, 256)  # 32 samples, each with 256-dimensional features
    print("MLP output shape:", mlp(dummy_input).shape)  # Should be [32, 1]
    # Output: MLP output shape: torch.Size([32, 1])