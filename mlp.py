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
        self.max_seq_len = max_seq_len 

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        x = x.view(batch_size * seq_len, embed_dim)
        x = self.mlp(x) 
        x = x.view(batch_size, seq_len, -1)

        return x



