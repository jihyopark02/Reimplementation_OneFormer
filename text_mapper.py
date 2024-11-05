import torch
import torch.nn as nn

class TextMapper(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=256):
        super(TextMapper, self).__init__()
        # Task-specific embeddings for Panoptic, Instance, and Semantic texts
        self.panoptic_embedding = nn.Embedding(vocab_size, embed_dim)
        self.instance_embedding = nn.Embedding(vocab_size, embed_dim)
        self.semantic_embedding = nn.Embedding(vocab_size, embed_dim)

        # Projection to unify embeddings into a common feature space
        self.projector = nn.Linear(embed_dim, embed_dim)

    def forward(self, panoptic_text, instance_text, semantic_text):
        # Get embeddings for each task type
        panoptic_embedded = self.panoptic_embedding(panoptic_text)  # Shape: [batch_size, seq_len, embed_dim]
        instance_embedded = self.instance_embedding(instance_text)  # Shape: [batch_size, seq_len, embed_dim]
        semantic_embedded = self.semantic_embedding(semantic_text)  # Shape: [batch_size, seq_len, embed_dim]

        # Project each embedding into a unified space
        panoptic_projected = self.projector(panoptic_embedded.mean(dim=1))  # Mean pooling over sequence dimension
        instance_projected = self.projector(instance_embedded.mean(dim=1))
        semantic_projected = self.projector(semantic_embedded.mean(dim=1))

        # Concatenate the projected embeddings to form a combined task query
        combined_query = torch.stack([panoptic_projected, instance_projected, semantic_projected], dim=1)
        # Shape: [batch_size, 3, embed_dim] where 3 represents the three task types

        return combined_query  # This will be used to condition task-specific queries
