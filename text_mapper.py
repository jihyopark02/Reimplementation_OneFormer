import torch
import torch.nn as nn

class TextMapper(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=256):
        super(TextMapper, self).__init__()

        self.panoptic_embedding = nn.Embedding(vocab_size, embed_dim)
        self.instance_embedding = nn.Embedding(vocab_size, embed_dim)
        self.semantic_embedding = nn.Embedding(vocab_size, embed_dim)

        self.projector = nn.Linear(embed_dim, embed_dim)

    def forward(self, panoptic_text, instance_text, semantic_text):
        panoptic_embedded = self.panoptic_embedding(panoptic_text) 
        instance_embedded = self.instance_embedding(instance_text)  
        semantic_embedded = self.semantic_embedding(semantic_text)  

        panoptic_projected = self.projector(panoptic_embedded.mean(dim=1))  
        instance_projected = self.projector(instance_embedded.mean(dim=1))
        semantic_projected = self.projector(semantic_embedded.mean(dim=1))

        combined_query = torch.stack([panoptic_projected, instance_projected, semantic_projected], dim=1)

        return combined_query
