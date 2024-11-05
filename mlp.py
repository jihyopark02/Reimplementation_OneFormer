import torch
import torch.nn as nn

class TaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, max_seq_len=128):
        super(TaskMLP, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.max_seq_len = max_seq_len  # Define max_seq_len explicitly

    def forward(self, x):
        # Expecting input shape: [batch_size, max_seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape
        print(f"Input shape before reshaping: {x.shape}")  # Debug print

        # Flatten the sequence dimension into the batch for MLP processing
        x = x.view(batch_size * seq_len, embed_dim)  # Shape: [batch_size * seq_len, embed_dim]
        print(f"Shape after flattening for MLP: {x.shape}")  # Debug print

        # Pass through the MLP
        x = self.mlp(x)  # Shape: [batch_size * seq_len, output_dim]
        print(f"Shape after MLP processing: {x.shape}")  # Debug print

        # Reshape back to the original [batch_size, seq_len, output_dim] shape
        x = x.view(batch_size, seq_len, -1)
        print(f"Shape after reshaping back to [batch_size, seq_len, output_dim]: {x.shape}")  # Debug print

        return x



