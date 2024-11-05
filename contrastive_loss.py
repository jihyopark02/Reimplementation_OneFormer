import torch.nn as nn
import torch
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q_text):
        """
        q_text: Tensor of shape [batch_size, 3, embed_dim], where 3 is the number of task types
        """
        batch_size, num_tasks, embed_dim = q_text.size()
        
        # Normalize each embedding vector to unit length
        q_text = F.normalize(q_text, dim=-1)  # Shape: [batch_size, 3, embed_dim]

        # Reshape q_text for contrastive pairwise comparison
        q_text = q_text.view(batch_size * num_tasks, embed_dim)  # Shape: [batch_size * 3, embed_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(q_text, q_text.T) / self.temperature  # Shape: [batch_size * 3, batch_size * 3]

        # Create labels for contrastive learning: similar tasks have the same index
        labels = torch.arange(batch_size).repeat_interleave(num_tasks).to(q_text.device)

        # Mask to avoid self-similarity in contrastive pairs
        mask = torch.eye(batch_size * num_tasks, dtype=torch.bool, device=q_text.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
