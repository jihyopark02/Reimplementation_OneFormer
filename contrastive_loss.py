import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q_text, q_task):
        if q_text.size(0) == 1:
            q_text = q_text.expand(q_task.size(0), -1, -1)

        q_text = F.normalize(q_text, dim=-1)
        q_task = F.normalize(q_task, dim=-1)

        batch_size, num_tasks, embed_dim = q_text.size()
        _, num_queries, _ = q_task.size()

        q_text = q_text.reshape(batch_size * num_tasks, embed_dim)
        q_task = q_task.reshape(batch_size * num_queries, embed_dim)

        sim_matrix = torch.matmul(q_text, q_task.T) / self.temperature

        labels = torch.arange(batch_size * num_tasks).to(q_text.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

