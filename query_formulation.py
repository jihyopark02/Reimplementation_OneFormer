from mlp import TaskMLP
import torch.nn as nn

class TaskConditionedQueryFormulator(nn.Module):
    def __init__(self, num_queries=100, embed_dim=256):
        super(TaskConditionedQueryFormulator, self).__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.task_embeddings = nn.Embedding(3, embed_dim)  # For panoptic, instance, semantic tasks
        self.task_mlp = TaskMLP(embed_dim, embed_dim, embed_dim)

    def forward(self, task_embed, batch_size):
        task_embed = self.task_mlp(task_embed)
        task_embed = task_embed.mean(dim=1, keepdim=True)

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        task_conditioned_queries = queries + task_embed.permute(1, 0, 2).repeat(self.num_queries, 1, 1)

        return task_conditioned_queries
